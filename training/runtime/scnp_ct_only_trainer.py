from __future__ import annotations

import os
from time import time
from typing import List

import matplotlib
import numpy as np
import torch
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from matplotlib import pyplot as plt
from torch import autocast
from torch import distributed as dist
from torch.nn import functional as F

from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.utilities.helpers import dummy_context

from training.runtime.nnunet_dismap_loader import nnUNetDataLoaderWithDisMap

matplotlib.use("Agg")


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() in ("1", "true", "t", "yes", "y")


def _label_id_to_name_map(dataset_json: dict) -> dict[int, str]:
    labels = dataset_json.get("labels", {})
    if not isinstance(labels, dict):
        return {}

    mapped: dict[int, str] = {}
    for name, value in labels.items():
        if isinstance(value, int):
            mapped[int(value)] = str(name)
    return mapped


def _normalize_label_name(name: str) -> str:
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")


class nnUNetTrainerSCNPCTOnlyBase(nnUNetTrainer):
    """
    Shared nnU-Net trainer skeleton for SCNP experiments that keep the network
    forward CT-only while consuming FracSegNet-style disMap sidecars as an
    auxiliary training prior.
    """

    use_fdm_as_loss_prior = True

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int | str,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        if plans.get("label_manager") == "LabelManager":
            plans = dict(plans)
            plans.pop("label_manager", None)

        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = int(os.environ.get("NNUNET_SCNP_NUM_EPOCHS", "1000"))
        self.initial_lr = float(os.environ.get("NNUNET_SCNP_INITIAL_LR", "1e-4"))
        self.weight_decay = float(os.environ.get("NNUNET_SCNP_WEIGHT_DECAY", str(self.weight_decay)))
        self.save_every = int(os.environ.get("NNUNET_SCNP_SAVE_EVERY", str(self.save_every)))
        self.is_smoke_test = _env_flag("NNUNETV2_QUICK_RUN") or _env_flag("NNUNET_SCNP_SMOKE_TEST")
        self.skip_actual_validation_flag = self.is_smoke_test or _env_flag("NNUNETV2_SKIP_ACTUAL_VALIDATION")
        self.log_grouped_secondary_pseudo_dice = self._should_log_grouped_secondary_pseudo_dice(dataset_json)

        if self.is_smoke_test:
            self.num_epochs = 1
            self.num_iterations_per_epoch = 1
            self.num_val_iterations_per_epoch = 1
            self.disable_checkpointing = True

    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ):
        return nnUNetTrainer.build_network_architecture(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            1,
            num_output_channels,
            enable_deep_supervision,
        )

    @staticmethod
    def _prepare_network_input(data: torch.Tensor) -> torch.Tensor:
        return data[:, :1]

    @staticmethod
    def _should_log_grouped_secondary_pseudo_dice(dataset_json: dict) -> bool:
        label_names = {key: _normalize_label_name(value) for key, value in _label_id_to_name_map(dataset_json).items()}
        if label_names.get(1) not in {"main_trunk", "main_fragment", "main_fracture_segment"}:
            return False
        secondary_label_names = [name for label_id, name in label_names.items() if label_id >= 2]
        if not secondary_label_names:
            return False
        return all(
            name in {"other_fragments", "secondary_fragments", "segment_2", "segment_3"}
            or name.startswith("secondary_fragment_")
            or name.startswith("other_fragment_")
            or name.startswith("segment_")
            for name in secondary_label_names
        )

    @staticmethod
    def _group_tp_fp_fn(tp_hard: np.ndarray, fp_hard: np.ndarray, fn_hard: np.ndarray) -> tuple[np.ndarray, ...]:
        tp_hard = np.asarray(tp_hard, dtype=np.float64)
        fp_hard = np.asarray(fp_hard, dtype=np.float64)
        fn_hard = np.asarray(fn_hard, dtype=np.float64)

        if tp_hard.ndim != 1 or fp_hard.ndim != 1 or fn_hard.ndim != 1:
            raise RuntimeError("Expected 1D tp/fp/fn arrays for grouped pseudo dice.")

        if tp_hard.shape != fp_hard.shape or tp_hard.shape != fn_hard.shape:
            raise RuntimeError("tp/fp/fn shapes must match when grouping pseudo dice.")

        if tp_hard.size <= 2:
            grouped_all_tp = tp_hard.copy()
            grouped_all_fp = fp_hard.copy()
            grouped_all_fn = fn_hard.copy()
        else:
            grouped_all_tp = np.asarray([tp_hard[0], tp_hard[1], tp_hard[2:].sum()], dtype=np.float64)
            grouped_all_fp = np.asarray([fp_hard[0], fp_hard[1], fp_hard[2:].sum()], dtype=np.float64)
            grouped_all_fn = np.asarray([fn_hard[0], fn_hard[1], fn_hard[2:].sum()], dtype=np.float64)

        grouped_fg_tp = grouped_all_tp[1:]
        grouped_fg_fp = grouped_all_fp[1:]
        grouped_fg_fn = grouped_all_fn[1:]
        return grouped_fg_tp, grouped_fg_fp, grouped_fg_fn, grouped_all_tp, grouped_all_fp, grouped_all_fn

    @staticmethod
    def _dice_from_stats(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> list[float]:
        tp = np.asarray(tp, dtype=np.float64)
        fp = np.asarray(fp, dtype=np.float64)
        fn = np.asarray(fn, dtype=np.float64)
        denom = 2.0 * tp + fp + fn
        dice = np.divide(
            2.0 * tp,
            denom,
            out=np.full(tp.shape, np.nan, dtype=np.float64),
            where=denom > 0,
        )
        return [float(x) for x in dice.tolist()]

    def _ddp_sum_array(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        world_size = dist.get_world_size()
        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, values)
        return np.vstack([np.atleast_1d(item) for item in gathered]).sum(0)

    def build_scnp_loss(self):
        raise NotImplementedError

    def _build_loss(self):
        loss = self.build_scnp_loss()

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2**i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
        return loss

    def _scnp_gate_log_message(self) -> tuple[str, ...]:
        raise NotImplementedError

    def _get_logger_store(self) -> dict | None:
        store = getattr(self.logger, "my_fantastic_logging", None)
        if isinstance(store, dict):
            return store

        local_logger = getattr(self.logger, "local_logger", None)
        store = getattr(local_logger, "my_fantastic_logging", None)
        if isinstance(store, dict):
            return store
        return None

    def _get_logged_value(self, key: str, step: int | None = None):
        store = self._get_logger_store()
        if store is None or key not in store:
            return None
        values = store.get(key, [])
        if step is None:
            return values
        if not isinstance(values, list) or len(values) == 0:
            return None
        try:
            return values[step]
        except IndexError:
            return None

    def initialize(self):
        super().initialize()
        logger_store = self._get_logger_store()
        if logger_store is not None:
            logger_store.setdefault("dice_grouped_bg_main_secondary", [])

        self.print_to_log_file(
            "Spatially Adaptive SCNP training settings:",
            f"num_epochs={self.num_epochs},",
            f"initial_lr={self.initial_lr},",
            f"weight_decay={self.weight_decay},",
            f"save_every={self.save_every}",
        )
        self.print_to_log_file(*self._scnp_gate_log_message())
        if self.is_smoke_test:
            self.print_to_log_file(
                "Smoke test enabled:",
                f"num_iterations_per_epoch={self.num_iterations_per_epoch},",
                f"num_val_iterations_per_epoch={self.num_val_iterations_per_epoch},",
                f"disable_checkpointing={self.disable_checkpointing}",
            )
        if self.skip_actual_validation_flag:
            self.print_to_log_file("Actual validation will be skipped")
        if self.log_grouped_secondary_pseudo_dice:
            self.print_to_log_file(
                "Primary validation pseudo Dice will stay in the native FracSegNet 4-class label space; "
                "an additional grouped metric will also be logged as background / main fracture segment / merged non-main fragments."
            )

    def get_dataloaders(self):
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

        patch_size = self.configuration_manager.patch_size
        deep_supervision_scales = self._get_deep_supervision_scales()

        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        tr_transforms = self.get_training_transforms(
            patch_size,
            rotation_for_DA,
            deep_supervision_scales,
            mirror_axes,
            do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label,
        )
        val_transforms = self.get_validation_transforms(
            deep_supervision_scales,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label,
        )

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()
        dl_tr = nnUNetDataLoaderWithDisMap(
            dataset_tr,
            self.batch_size,
            initial_patch_size,
            self.configuration_manager.patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None,
            pad_sides=None,
            transforms=tr_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling,
        )
        dl_val = nnUNetDataLoaderWithDisMap(
            dataset_val,
            self.batch_size,
            self.configuration_manager.patch_size,
            self.configuration_manager.patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None,
            pad_sides=None,
            transforms=val_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling,
        )

        allowed_num_processes = int(get_allowed_n_proc_DA())
        if allowed_num_processes <= 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
            self.print_to_log_file("Dataloader mode: SingleThreadedAugmenter (n_proc_DA=0)")
        else:
            if os.name == "nt":
                wait_time = float(os.environ.get("NNUNET_SCNP_DA_WAIT_TIME", "0.02"))
                train_cached = int(os.environ.get("NNUNET_SCNP_DA_CACHED_TRAIN", str(max(2, allowed_num_processes))))
                val_num_proc = max(1, allowed_num_processes // 2)
                val_cached = int(os.environ.get("NNUNET_SCNP_DA_CACHED_VAL", str(max(1, val_num_proc))))
            else:
                wait_time = 0.002
                train_cached = max(6, allowed_num_processes // 2)
                val_num_proc = max(1, allowed_num_processes // 2)
                val_cached = max(3, allowed_num_processes // 4)

            mt_gen_train = NonDetMultiThreadedAugmenter(
                data_loader=dl_tr,
                transform=None,
                num_processes=allowed_num_processes,
                num_cached=train_cached,
                seeds=None,
                pin_memory=self.device.type == "cuda",
                wait_time=wait_time,
            )
            mt_gen_val = NonDetMultiThreadedAugmenter(
                data_loader=dl_val,
                transform=None,
                num_processes=val_num_proc,
                num_cached=val_cached,
                seeds=None,
                pin_memory=self.device.type == "cuda",
                wait_time=wait_time,
            )
            self.print_to_log_file(
                "Dataloader mode: NonDetMultiThreadedAugmenter",
                f"n_proc_DA(train/val)={allowed_num_processes}/{val_num_proc}",
                f"num_cached(train/val)={train_cached}/{val_cached}",
                f"wait_time={wait_time}",
            )

        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        return mt_gen_train, mt_gen_val

    @staticmethod
    def _resize_fdm_like_output(fdm: torch.Tensor | None, output):
        if isinstance(output, tuple):
            if fdm is None:
                return tuple(None for _ in output)
            return tuple(nnUNetTrainerSCNPCTOnlyBase._resize_fdm_like_output(fdm, item) for item in output)
        if isinstance(output, list):
            if fdm is None:
                return [None for _ in output]
            return [nnUNetTrainerSCNPCTOnlyBase._resize_fdm_like_output(fdm, item) for item in output]
        if fdm is None:
            return None
        if fdm.shape[2:] == output.shape[2:]:
            return fdm
        mode = "bilinear" if output.ndim == 4 else "trilinear"
        return F.interpolate(fdm, size=output.shape[2:], mode=mode, align_corners=False)

    def _extract_fdm_prior(self, batch: dict) -> torch.Tensor | None:
        if not self.use_fdm_as_loss_prior:
            return None
        return batch.get("disMap")

    def train_step(self, batch: dict) -> dict:
        data = batch["data"]
        target = batch["target"]
        fdm = self._extract_fdm_prior(batch)

        data = data.to(self.device, non_blocking=True)
        network_input = self._prepare_network_input(data)
        if fdm is not None:
            fdm = fdm.to(self.device, non_blocking=True)
        if isinstance(target, (list, tuple)):
            target = [item.to(self.device, non_blocking=True) for item in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(network_input)
            loss_fdm = self._resize_fdm_like_output(fdm, output)
            loss_value = self.loss(output, target, loss_fdm)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss_value).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {"loss": loss_value.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"]
        target = batch["target"]
        fdm = self._extract_fdm_prior(batch)

        data = data.to(self.device, non_blocking=True)
        network_input = self._prepare_network_input(data)
        if fdm is not None:
            fdm = fdm.to(self.device, non_blocking=True)
        if isinstance(target, (list, tuple)):
            target = [item.to(self.device, non_blocking=True) for item in target]
        else:
            target = target.to(self.device, non_blocking=True)

        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(network_input)
            loss_fdm = self._resize_fdm_like_output(fdm, output)
            loss_value = self.loss(output, target, loss_fdm)

        del data

        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        axes = [0] + list(range(2, output.ndim))
        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float16)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                target[target == self.label_manager.ignore_label] = 0
            else:
                if target.dtype == torch.bool:
                    mask = ~target[:, -1:]
                else:
                    mask = 1 - target[:, -1:]
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)
        tp_hard_full = tp.detach().cpu().numpy()
        fp_hard_full = fp.detach().cpu().numpy()
        fn_hard_full = fn.detach().cpu().numpy()

        if not self.label_manager.has_regions:
            tp_hard_raw = tp_hard_full[1:]
            fp_hard_raw = fp_hard_full[1:]
            fn_hard_raw = fn_hard_full[1:]
        else:
            tp_hard_raw = tp_hard_full
            fp_hard_raw = fp_hard_full
            fn_hard_raw = fn_hard_full

        if self.log_grouped_secondary_pseudo_dice and not self.label_manager.has_regions:
            (
                _,
                _,
                _,
                tp_hard_grouped_all,
                fp_hard_grouped_all,
                fn_hard_grouped_all,
            ) = self._group_tp_fp_fn(tp_hard_full, fp_hard_full, fn_hard_full)

        out = {
            "loss": loss_value.detach().cpu().numpy(),
            "tp_hard": tp_hard_raw,
            "fp_hard": fp_hard_raw,
            "fn_hard": fn_hard_raw,
        }
        if self.log_grouped_secondary_pseudo_dice and not self.label_manager.has_regions:
            out.update(
                {
                    "tp_hard_grouped_all": tp_hard_grouped_all,
                    "fp_hard_grouped_all": fp_hard_grouped_all,
                    "fn_hard_grouped_all": fn_hard_grouped_all,
                }
            )
        return out

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        outputs_collated = collate_outputs(val_outputs)
        tp = np.sum(outputs_collated["tp_hard"], 0)
        fp = np.sum(outputs_collated["fp_hard"], 0)
        fn = np.sum(outputs_collated["fn_hard"], 0)

        if self.log_grouped_secondary_pseudo_dice and "tp_hard_grouped_all" in outputs_collated:
            tp_grouped_all = np.sum(outputs_collated["tp_hard_grouped_all"], 0)
            fp_grouped_all = np.sum(outputs_collated["fp_hard_grouped_all"], 0)
            fn_grouped_all = np.sum(outputs_collated["fn_hard_grouped_all"], 0)
        else:
            tp_grouped_all = fp_grouped_all = fn_grouped_all = None

        if self.is_ddp:
            tp = self._ddp_sum_array(tp)
            fp = self._ddp_sum_array(fp)
            fn = self._ddp_sum_array(fn)

            if tp_grouped_all is not None:
                tp_grouped_all = self._ddp_sum_array(tp_grouped_all)
                fp_grouped_all = self._ddp_sum_array(fp_grouped_all)
                fn_grouped_all = self._ddp_sum_array(fn_grouped_all)

            world_size = dist.get_world_size()
            losses_val = [None for _ in range(world_size)]
            dist.all_gather_object(losses_val, outputs_collated["loss"])
            loss_here = np.vstack(losses_val).mean()
        else:
            loss_here = np.mean(outputs_collated["loss"])

        global_dc_per_class = self._dice_from_stats(tp, fp, fn)
        mean_fg_dice = float(np.nanmean(global_dc_per_class)) if len(global_dc_per_class) else 0.0
        self.logger.log("mean_fg_dice", mean_fg_dice, self.current_epoch)
        self.logger.log("dice_per_class_or_region", global_dc_per_class, self.current_epoch)
        self.logger.log("val_losses", loss_here, self.current_epoch)

        if tp_grouped_all is not None:
            grouped_all_dc = self._dice_from_stats(tp_grouped_all, fp_grouped_all, fn_grouped_all)
            self.logger.log("dice_grouped_bg_main_secondary", grouped_all_dc, self.current_epoch)

    def on_epoch_end(self):
        self.logger.log("epoch_end_timestamps", time(), self.current_epoch)

        train_loss = self._get_logged_value("train_losses", step=-1)
        val_loss = self._get_logged_value("val_losses", step=-1)
        pseudo_dice = self._get_logged_value("dice_per_class_or_region", step=-1)
        epoch_end = self._get_logged_value("epoch_end_timestamps", step=-1)
        epoch_start = self._get_logged_value("epoch_start_timestamps", step=-1)
        ema_fg_dice = self._get_logged_value("ema_fg_dice", step=-1)

        if train_loss is not None:
            self.print_to_log_file("train_loss", np.round(train_loss, decimals=4))
        if val_loss is not None:
            self.print_to_log_file("val_loss", np.round(val_loss, decimals=4))
        if pseudo_dice is not None:
            self.print_to_log_file("Pseudo dice", [np.round(item, decimals=4) for item in pseudo_dice])
        if self.log_grouped_secondary_pseudo_dice:
            grouped_all = self._get_logged_value("dice_grouped_bg_main_secondary", step=-1)
            if grouped_all is not None:
                self.print_to_log_file(
                    "Grouped pseudo dice (bg/main/non-main)",
                    [np.round(item, decimals=4) for item in grouped_all],
                )
        if epoch_end is not None and epoch_start is not None:
            self.print_to_log_file(f"Epoch time: {np.round(epoch_end - epoch_start, decimals=2)} s")

        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(os.path.join(self.output_folder, "checkpoint_latest.pth"))

        if ema_fg_dice is not None and (self._best_ema is None or ema_fg_dice > self._best_ema):
            self._best_ema = ema_fg_dice
            self.print_to_log_file(f"Yayy! New best EMA pseudo Dice: {np.round(self._best_ema, decimals=4)}")
            self.save_checkpoint(os.path.join(self.output_folder, "checkpoint_best.pth"))

        if self.local_rank == 0:
            self.plot_progress()

        self.current_epoch += 1

    def plot_progress(self):
        logs = self._get_logger_store()
        if logs is None:
            return

        train_losses = [float(item) for item in logs.get("train_losses", [])]
        val_losses = [float(item) for item in logs.get("val_losses", [])]
        val_dice = logs.get("mean_fg_dice", None)
        if val_dice is None:
            val_dice = logs.get("ema_fg_dice", None)
        val_dice = [float(item) for item in (val_dice or [])]

        if len(train_losses) == 0 and len(val_losses) == 0 and len(val_dice) == 0:
            return

        fig, axes = plt.subplots(2, 1, figsize=(12, 7.5), dpi=150)
        fig.patch.set_facecolor("white")
        for axis in axes:
            axis.set_facecolor("white")
            axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)

        if len(train_losses):
            axes[0].plot(np.arange(len(train_losses)), train_losses, color="#1f77b4", linewidth=2.0, label="train_loss")
        if len(val_losses):
            axes[0].plot(np.arange(len(val_losses)), val_losses, color="#d62728", linewidth=2.0, label="val_loss")
        axes[0].set_title("Loss")
        axes[0].set_xlabel("Epoch")
        if axes[0].get_legend_handles_labels()[0]:
            axes[0].legend(loc="best")

        if len(val_dice):
            axes[1].plot(np.arange(len(val_dice)), val_dice, color="#2a9d8f", linewidth=2.2, label="val_dice")
        axes[1].set_title("Validation Dice")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylim(-0.02, 1.02)
        if axes[1].get_legend_handles_labels()[0]:
            axes[1].legend(loc="best")

        fig.tight_layout()
        fig.savefig(os.path.join(self.output_folder, "progress.png"), facecolor="white", edgecolor="white")
        plt.close(fig)

    def perform_actual_validation(self, *args, **kwargs):
        if self.skip_actual_validation_flag:
            self.print_to_log_file("Skipping perform_actual_validation")
            return
        return super().perform_actual_validation(*args, **kwargs)


class nnUNetTrainerSCNPSingleRFBase(nnUNetTrainerSCNPCTOnlyBase):
    """
    Shared single-receptive-field SCNP trainer that differs only in the
    specific SCNP loss class or logging preset.
    """

    scnp_loss_class = None
    scnp_gate_mode = "hard_threshold"

    def build_scnp_loss(self):
        if self.scnp_loss_class is None:
            raise RuntimeError("scnp_loss_class must be set on the single-RF trainer subclass.")

        receptive_field = int(os.environ.get("NNUNET_SCNP_RF", "3"))
        scnp_kappa = float(os.environ.get("NNUNET_SCNP_KAPPA", "9999"))
        weight_ce = float(os.environ.get("NNUNET_SCNP_WEIGHT_CE", "1.0"))
        weight_dice = float(os.environ.get("NNUNET_SCNP_WEIGHT_DICE", "1.0"))
        joint_standard_weight = float(os.environ.get("NNUNET_SCNP_JOINT_STD_WEIGHT", "0.0"))
        fdm_threshold = float(os.environ.get("NNUNET_SCNP_FDM_THRESHOLD", "0.5"))
        fdm_power = float(os.environ.get("NNUNET_SCNP_FDM_POWER", "1.0"))

        return self.scnp_loss_class(
            {
                "batch_dice": self.configuration_manager.batch_dice,
                "smooth": 1e-5,
                "do_bg": False,
                "ddp": self.is_ddp,
            },
            {},
            weight_ce=weight_ce,
            weight_dice=weight_dice,
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss,
            receptive_field=receptive_field,
            kappa=scnp_kappa,
            joint_standard_weight=joint_standard_weight,
            fdm_threshold=fdm_threshold,
            fdm_power=fdm_power,
        )

    def _scnp_gate_log_message(self) -> tuple[str, ...]:
        return (
            "SCNP single-RF gating:",
            f"mode={self.scnp_gate_mode},",
            f"fdm_threshold={os.environ.get('NNUNET_SCNP_FDM_THRESHOLD', '0.5')},",
            f"fdm_power={os.environ.get('NNUNET_SCNP_FDM_POWER', '1.0')},",
            f"rf={os.environ.get('NNUNET_SCNP_RF', '3')},",
            "network_input=ct_only,",
            "fdm_loss_prior=on",
        )


class nnUNetTrainerSCNPMultiRFBase(nnUNetTrainerSCNPCTOnlyBase):
    """
    Shared multi-receptive-field SCNP trainer.
    """

    scnp_loss_class = None
    scnp_gate_mode = "multi_rf"

    def build_scnp_loss(self):
        if self.scnp_loss_class is None:
            raise RuntimeError("scnp_loss_class must be set on the multi-RF trainer subclass.")

        receptive_field_low = int(os.environ.get("NNUNET_SCNP_RF_LOW", "3"))
        receptive_field_high = int(os.environ.get("NNUNET_SCNP_RF_HIGH", "5"))
        scnp_kappa = float(os.environ.get("NNUNET_SCNP_KAPPA", "9999"))
        weight_ce = float(os.environ.get("NNUNET_SCNP_WEIGHT_CE", "1.0"))
        weight_dice = float(os.environ.get("NNUNET_SCNP_WEIGHT_DICE", "1.0"))
        joint_standard_weight = float(os.environ.get("NNUNET_SCNP_JOINT_STD_WEIGHT", "0.0"))
        fdm_low = float(os.environ.get("NNUNET_SCNP_FDM_LOW", "0.3"))
        fdm_high = float(os.environ.get("NNUNET_SCNP_FDM_HIGH", "0.5"))
        fdm_power = float(os.environ.get("NNUNET_SCNP_FDM_POWER", "1.0"))

        return self.scnp_loss_class(
            {
                "batch_dice": self.configuration_manager.batch_dice,
                "smooth": 1e-5,
                "do_bg": False,
                "ddp": self.is_ddp,
            },
            {},
            weight_ce=weight_ce,
            weight_dice=weight_dice,
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss,
            receptive_field_low=receptive_field_low,
            receptive_field_high=receptive_field_high,
            kappa=scnp_kappa,
            joint_standard_weight=joint_standard_weight,
            fdm_low=fdm_low,
            fdm_high=fdm_high,
            fdm_power=fdm_power,
        )

    def _scnp_gate_log_message(self) -> tuple[str, ...]:
        return (
            "SCNP multi-RF gating:",
            f"mode={self.scnp_gate_mode},",
            f"fdm_low={os.environ.get('NNUNET_SCNP_FDM_LOW', '0.3')},",
            f"fdm_high={os.environ.get('NNUNET_SCNP_FDM_HIGH', '0.5')},",
            f"rf_low={os.environ.get('NNUNET_SCNP_RF_LOW', '3')},",
            f"rf_high={os.environ.get('NNUNET_SCNP_RF_HIGH', '5')},",
            "network_input=ct_only,",
            "fdm_loss_prior=on",
        )


class nnUNetTrainerSoftSCNPBase(nnUNetTrainerSCNPCTOnlyBase):
    """
    Shared trainer for the soft-SCNP ablation family.
    """

    scnp_loss_class = None
    scnp_gate_mode = "soft_scnp_soft_fdm"

    def build_scnp_loss(self):
        if self.scnp_loss_class is None:
            raise RuntimeError("scnp_loss_class must be set on the soft-SCNP trainer subclass.")

        receptive_field = int(os.environ.get("NNUNET_SCNP_RF", "3"))
        scnp_kappa = float(os.environ.get("NNUNET_SCNP_KAPPA", "9999"))
        weight_ce = float(os.environ.get("NNUNET_SCNP_WEIGHT_CE", "1.0"))
        weight_dice = float(os.environ.get("NNUNET_SCNP_WEIGHT_DICE", "1.0"))
        joint_standard_weight = float(os.environ.get("NNUNET_SCNP_JOINT_STD_WEIGHT", "0.0"))
        fdm_threshold = float(os.environ.get("NNUNET_SCNP_FDM_THRESHOLD", "0.5"))
        fdm_power = float(os.environ.get("NNUNET_SCNP_FDM_POWER", "1.0"))
        fdm_beta = float(os.environ.get("NNUNET_SCNP_FDM_BETA", "10.0"))
        soft_tau = float(os.environ.get("NNUNET_SCNP_SOFT_TAU", "1.0"))

        return self.scnp_loss_class(
            {
                "batch_dice": self.configuration_manager.batch_dice,
                "smooth": 1e-5,
                "do_bg": False,
                "ddp": self.is_ddp,
            },
            {},
            weight_ce=weight_ce,
            weight_dice=weight_dice,
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss,
            receptive_field=receptive_field,
            kappa=scnp_kappa,
            joint_standard_weight=joint_standard_weight,
            fdm_threshold=fdm_threshold,
            fdm_power=fdm_power,
            fdm_beta=fdm_beta,
            soft_scnp_temperature=soft_tau,
        )

    def _scnp_gate_log_message(self) -> tuple[str, ...]:
        fdm_threshold = os.environ.get("NNUNET_SCNP_FDM_THRESHOLD", "0.5")
        fdm_beta = os.environ.get("NNUNET_SCNP_FDM_BETA", "10.0")
        if not self.use_fdm_as_loss_prior:
            fdm_threshold = "disabled"
            fdm_beta = "disabled"
        return (
            "Soft-SCNP training mode:",
            f"mode={self.scnp_gate_mode},",
            f"soft_tau={os.environ.get('NNUNET_SCNP_SOFT_TAU', '1.0')},",
            f"fdm_threshold={fdm_threshold},",
            f"fdm_beta={fdm_beta},",
            f"fdm_power={os.environ.get('NNUNET_SCNP_FDM_POWER', '1.0')},",
            f"rf={os.environ.get('NNUNET_SCNP_RF', '3')},",
            "fdm_input=off,",
            f"fdm_loss_prior={'on' if self.use_fdm_as_loss_prior else 'off'}",
        )
