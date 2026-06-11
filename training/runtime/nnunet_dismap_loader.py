from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple, Union

import blosc2
import numpy as np
import torch
from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from threadpoolctl import threadpool_limits

from training.runtime.project_paths import (
    DISMAP_FILE_SUFFIX,
    DISMAP_PAD_VALUE,
    dismap_sidecar_path as project_dismap_sidecar_path,
    legacy_dismap_sidecar_path,
)


def dismap_sidecar_path(preprocessed_case_folder: str | os.PathLike[str], identifier: str) -> Path:
    return project_dismap_sidecar_path(preprocessed_case_folder, identifier)


def load_dismap_array(preprocessed_case_folder: str | os.PathLike[str], identifier: str) -> np.ndarray:
    path = dismap_sidecar_path(preprocessed_case_folder, identifier)
    if not path.is_file():
        legacy_path = legacy_dismap_sidecar_path(preprocessed_case_folder, identifier)
        if legacy_path.is_file():
            path = legacy_path
        else:
            raise FileNotFoundError(
                f"Missing disMap sidecar for {identifier}: checked {path} and {legacy_path}"
            )
    dismap = np.load(path, mmap_mode="r")
    if dismap.ndim < 4:
        raise RuntimeError(f"Expected disMap with shape [1, ...], got {dismap.shape} at {path}")
    if dismap.shape[0] != 1:
        dismap = dismap[:1]
    return np.asarray(dismap, dtype=np.float32)


def _stack_or_list(values: List[torch.Tensor | list[torch.Tensor]]):
    if isinstance(values[0], list):
        return [torch.stack([item[level] for item in values]) for level in range(len(values[0]))]
    return torch.stack(values)


class nnUNetDataLoaderWithDisMap(nnUNetDataLoader):
    def __init__(
        self,
        *args,
        dismap_pad_value: float = DISMAP_PAD_VALUE,
        **kwargs,
    ):
        self.dismap_pad_value = float(dismap_pad_value)
        super().__init__(*args, **kwargs)

    def _load_case_dismap(self, identifier: str) -> np.ndarray:
        return load_dismap_array(self._data.source_folder, identifier)

    def generate_train_batch(self):
        selected_keys = self.get_indices()
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        dismap_shape = (self.batch_size, 1, *self.patch_size)
        dismap_all = np.full(dismap_shape, self.dismap_pad_value, dtype=np.float32)

        for batch_index, identifier in enumerate(selected_keys):
            force_fg = self.get_do_oversample(batch_index)
            data, seg, seg_prev, properties = self._data.load_case(identifier)
            dismap = self._load_case_dismap(identifier)

            shape = data.shape[1:]
            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties["class_locations"])
            bbox = [[lower, upper] for lower, upper in zip(bbox_lbs, bbox_ubs)]

            data_all[batch_index] = crop_and_pad_nd(data, bbox, 0)

            seg_cropped = crop_and_pad_nd(seg, bbox, -1)
            if seg_prev is not None:
                seg_cropped = np.vstack((seg_cropped, crop_and_pad_nd(seg_prev, bbox, -1)[None]))
            seg_all[batch_index] = seg_cropped

            dismap_all[batch_index] = crop_and_pad_nd(dismap, bbox, self.dismap_pad_value)

        if self.patch_size_was_2d:
            data_all = data_all[:, :, 0]
            seg_all = seg_all[:, :, 0]
            dismap_all = dismap_all[:, :, 0]

        if self.transforms is not None:
            with torch.no_grad():
                with threadpool_limits(limits=1, user_api=None):
                    data_tensor = torch.from_numpy(data_all).float()
                    seg_tensor = torch.from_numpy(seg_all).to(torch.int16)
                    dismap_tensor = torch.from_numpy(dismap_all).float()

                    images = []
                    segs = []
                    dismaps = []
                    for sample_index in range(self.batch_size):
                        transformed = self.transforms(
                            image=data_tensor[sample_index],
                            segmentation=seg_tensor[sample_index],
                            regression_target=dismap_tensor[sample_index],
                        )
                        images.append(transformed["image"])
                        segs.append(transformed["segmentation"])
                        dismaps.append(transformed.get("regression_target", dismap_tensor[sample_index]))

                    data_tensor = torch.stack(images)
                    seg_tensor = _stack_or_list(segs)
                    dismap_tensor = torch.stack(dismaps)

            return {
                "data": data_tensor,
                "target": seg_tensor,
                "disMap": dismap_tensor,
                "keys": selected_keys,
            }

        return {
            "data": data_all,
            "target": seg_all,
            "disMap": dismap_all,
            "keys": selected_keys,
        }
