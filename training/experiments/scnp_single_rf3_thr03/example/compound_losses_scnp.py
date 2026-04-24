from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch.nn import functional as F
from torch import nn


def _sanitize_target(target: torch.Tensor) -> torch.Tensor:
    # nnUNet can provide targets as [B, 1, ...]; SCNP expects class index map [B, ...].
    if target.dim() >= 2 and target.shape[1] == 1:
        target = target[:, 0]
    return target.long()


def _one_hot_from_index(target: torch.Tensor, num_classes: int) -> torch.Tensor:
    # target: [B, ...] -> onehot: [B, C, ...]
    oh = torch.nn.functional.one_hot(target.clamp(min=0), num_classes=num_classes)
    dims = [0, oh.dim() - 1] + list(range(1, oh.dim() - 1))
    return oh.permute(*dims).contiguous().float()


def _valid_mask_and_target(
    target_idx: torch.Tensor, ignore_label: Optional[int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    # When ignore_label exists, we zero out ignored voxels and carry a validity mask.
    # This keeps tensor shapes intact while excluding ignored voxels from CE/Dice reductions.
    if ignore_label is None:
        valid = torch.ones_like(target_idx, dtype=torch.bool)
        return valid, target_idx
    valid = target_idx != int(ignore_label)
    cleaned = target_idx.clone()
    cleaned[~valid] = 0
    return valid, cleaned


def _get_pool_params(ndim: int, receptive_field: int):
    if ndim == 4:
        return torch.nn.functional.max_pool2d, (receptive_field, receptive_field)
    if ndim == 5:
        return torch.nn.functional.max_pool3d, (receptive_field, receptive_field, receptive_field)
    raise RuntimeError(f"Expected net_output with ndim 4/5, got {ndim}")


class SCNPCEDice(nn.Module):
    """
    CE + Dice over SCNP-penalized logits (paper 2603.18671 / SCNP).

    SCNP equations are implemented exactly with min/max pooling:
      V1 = MinPool((Z * Y) + kappa * (1 - Y))
      V2 = MaxPool((Z * (1 - Y)) - kappa * Y)
      Z_bar = (V1 * Y) + (V2 * (1 - Y))
    """

    def __init__(
        self,
        soft_dice_kwargs,
        ce_kwargs,
        weight_ce: float = 1.0,
        weight_dice: float = 1.0,
        ignore_label=None,
        dice_class=None,
        receptive_field: int = 3,
        kappa: float = 9999.0,
        joint_standard_weight: float = 0.0,
        fdm_threshold: float = 0.5,
        fdm_power: float = 1.0,
    ):
        super().__init__()
        _ = (ce_kwargs, dice_class)  # Kept for API compatibility with nnUNet loss constructors.
        self.weight_ce = float(weight_ce)
        self.weight_dice = float(weight_dice)
        self.ignore_label = ignore_label
        self.receptive_field = int(receptive_field)
        self.kappa = float(kappa)
        self.joint_standard_weight = float(joint_standard_weight)
        self.fdm_threshold = float(fdm_threshold)
        self.fdm_power = float(fdm_power)

        self.smooth = float(soft_dice_kwargs.get("smooth", 1e-5))
        self.do_bg = bool(soft_dice_kwargs.get("do_bg", False))

    def _scnp_logits(self, logits: torch.Tensor, target_onehot: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        # SCNP is applied channel-wise with same-size pooling (stride=1, symmetric padding).
        # 2D and 3D are both supported because nnUNet can run in either mode.
        max_pool, kernel_size = _get_pool_params(logits.ndim, self.receptive_field)
        pad = tuple(k // 2 for k in kernel_size)
        stride = tuple(1 for _ in kernel_size)

        valid = valid_mask.unsqueeze(1).float()  # [B, 1, ...]
        fg = target_onehot * valid
        bg = (1.0 - target_onehot) * valid

        # Foreground branch: MinPool((Z * Y) + kappa * (1 - Y))
        # MinPool is implemented as -MaxPool(-x).
        t1 = -max_pool(
            -(logits * fg + self.kappa * (1.0 - fg)),
            kernel_size=kernel_size,
            stride=stride,
            padding=pad,
        )
        # Background branch: MaxPool((Z * (1 - Y)) - kappa * Y)
        t2 = max_pool(
            (logits * bg - self.kappa * (1.0 - bg)),
            kernel_size=kernel_size,
            stride=stride,
            padding=pad,
        )
        # Compose SCNP logits Z_bar = V1*Y + V2*(1-Y).
        return t1 * fg + t2 * bg

    def _prepare_fdm_mask(
        self,
        fdm: Optional[torch.Tensor],
        logits: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if fdm is None:
            return None

        if fdm.dim() == (logits.dim() - 1):
            fdm = fdm.unsqueeze(1)
        if fdm.dim() != logits.dim():
            raise RuntimeError(
                f"Expected FDM tensor with ndim {logits.dim()} or {logits.dim() - 1}, got {fdm.dim()}"
            )
        if fdm.shape[1] != 1:
            fdm = fdm[:, :1]

        if fdm.shape[2:] != logits.shape[2:]:
            mode = "bilinear" if logits.dim() == 4 else "trilinear"
            fdm = F.interpolate(fdm.float(), size=logits.shape[2:], mode=mode, align_corners=False)
        else:
            fdm = fdm.float()

        spatial_axes = tuple(range(2, fdm.dim()))
        fdm_min = torch.amin(fdm, dim=spatial_axes, keepdim=True)
        fdm_max = torch.amax(fdm, dim=spatial_axes, keepdim=True)
        mask = (fdm - fdm_min) / (fdm_max - fdm_min + 1e-8)
        mask = mask.clamp_(0.0, 1.0)

        # Hard gate: FDM only decides where SCNP is enabled, not how strongly it is blended.
        if self.fdm_power != 1.0:
            mask = mask.pow(self.fdm_power)
        mask = (mask >= self.fdm_threshold).float()

        valid = valid_mask.unsqueeze(1).float()
        return mask * valid

    def _ce_loss(self, probs: torch.Tensor, target_onehot: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        # CE = -sum y*log(p) over classes, averaged on valid pixels.
        ce_map = -(target_onehot * torch.log(probs.clamp_min(1e-8))).sum(dim=1)
        valid = valid_mask.float()
        denom = valid.sum().clamp_min(1.0)
        return (ce_map * valid).sum() / denom

    def _dice_loss(self, probs: torch.Tensor, target_onehot: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        # Dice is computed only on valid voxels.
        valid = valid_mask.unsqueeze(1).float()
        probs = probs * valid
        target_onehot = target_onehot * valid

        spatial_axes = tuple(range(2, probs.ndim))
        numerator = 2.0 * (probs * target_onehot).sum(dim=spatial_axes) + self.smooth
        denominator = probs.sum(dim=spatial_axes) + target_onehot.sum(dim=spatial_axes) + self.smooth
        dice_per_class = numerator / denominator  # [B, C]

        if not self.do_bg and dice_per_class.shape[1] > 1:
            dice_per_class = dice_per_class[:, 1:]
        return 1.0 - dice_per_class.mean()

    def _ce_dice_from_logits(
        self, logits: torch.Tensor, target_onehot: torch.Tensor, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        # Keep CE+Dice composition close to standard nnUNet training behavior.
        probs = torch.softmax(logits, dim=1)
        ce = self._ce_loss(probs, target_onehot, valid_mask) if self.weight_ce != 0 else logits.new_tensor(0.0)
        dc = self._dice_loss(probs, target_onehot, valid_mask) if self.weight_dice != 0 else logits.new_tensor(0.0)
        return self.weight_ce * ce + self.weight_dice * dc

    def forward(
        self,
        net_output: torch.Tensor,
        target: torch.Tensor,
        fdm: Optional[torch.Tensor] = None,
    ):
        target_idx = _sanitize_target(target)
        valid_mask, target_idx = _valid_mask_and_target(target_idx, self.ignore_label)
        target_onehot = _one_hot_from_index(target_idx, num_classes=net_output.shape[1])

        # Main branch: CE+Dice on FDM-guided SCNP logits.
        scnp_logits = self._scnp_logits(net_output, target_onehot, valid_mask)
        fdm_mask = self._prepare_fdm_mask(fdm, net_output, valid_mask)
        if fdm_mask is not None:
            scnp_logits = fdm_mask * scnp_logits + (1.0 - fdm_mask) * net_output
        loss = self._ce_dice_from_logits(scnp_logits, target_onehot, valid_mask)

        # Optional joint optimization with standard logits:
        # L = L(SCNP(Z)) + alpha * L(Z)
        if self.joint_standard_weight > 0:
            baseline = self._ce_dice_from_logits(net_output, target_onehot, valid_mask)
            loss = loss + self.joint_standard_weight * baseline
        return loss
