from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import binary_dilation, distance_transform_edt, generate_binary_structure


def _target_tensor_to_label_map(target: torch.Tensor) -> np.ndarray:
    if target.dim() >= 1 and target.shape[0] == 1:
        target = target[0]
    return np.rint(target.detach().cpu().numpy()).astype(np.int16, copy=False)


def _single_case_true_dismap(label_map: np.ndarray, ignore_label: int | None = None) -> np.ndarray:
    """
    Rebuild the FracSegNet-style continuous distance prior from the target label map.

    - label 1 is treated as the main fracture segment
    - labels >= 2 are treated as non-main fracture segments
    - distance is measured from the main/non-main contact boundary
    - the final weights follow the original logistic shaping and stay in [0.2, ~1.0]
    """
    label_map = np.asarray(label_map, dtype=np.int16)
    prior = np.full(label_map.shape, 0.2, dtype=np.float32)

    valid_mask = np.ones(label_map.shape, dtype=bool)
    if ignore_label is not None:
        valid_mask &= label_map != int(ignore_label)

    foreground_mask = valid_mask & (label_map > 0)
    main_mask = valid_mask & (label_map == 1)
    fragment_mask = valid_mask & (label_map > 1)

    if not foreground_mask.any():
        return prior
    if not main_mask.any() or not fragment_mask.any():
        return prior

    structure = generate_binary_structure(label_map.ndim, label_map.ndim)
    edge_mask = main_mask & binary_dilation(fragment_mask, structure=structure)
    if not edge_mask.any():
        return prior

    distance = distance_transform_edt(~edge_mask).astype(np.float32, copy=False)
    foreground_distance = distance[foreground_mask]
    if foreground_distance.size == 0:
        return prior

    max_distance = float(foreground_distance.max())
    if max_distance > 1e-8:
        scaled = distance / max_distance * 10.0
    else:
        scaled = np.zeros_like(distance, dtype=np.float32)

    weights = (1.0 / (1.0 + np.exp(scaled.astype(np.float64) - 5.0))) * 0.8 + 0.2
    prior[foreground_mask] = weights[foreground_mask].astype(np.float32, copy=False)
    return prior


def build_true_dismap_prior(
    target: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
    ignore_label: int | None = None,
) -> torch.Tensor | None:
    if isinstance(target, (list, tuple)):
        if len(target) == 0:
            return None
        target = target[0]

    if not isinstance(target, torch.Tensor):
        return None
    if target.dim() < 2:
        raise RuntimeError(f"Expected target tensor with batch dimension, got shape {tuple(target.shape)}")

    priors = []
    for sample_idx in range(target.shape[0]):
        label_map = _target_tensor_to_label_map(target[sample_idx])
        priors.append(_single_case_true_dismap(label_map, ignore_label=ignore_label))

    stacked = np.stack(priors, axis=0)[:, None]
    return torch.from_numpy(stacked.astype(np.float32, copy=False))
