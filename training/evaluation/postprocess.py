from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk


def keep_only_global_largest_component(input_dir: Path, output_dir: Path) -> None:
    """
    Mirror the FracSegNet export-time largest-connected-component cleanup on a
    directory of NIfTI predictions.
    """
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_files = sorted(input_dir.glob("*.nii.gz"))
    if not pred_files:
        raise RuntimeError(f"No prediction files found in {input_dir}")

    for pred_path in pred_files:
        pred_img = sitk.ReadImage(str(pred_path))
        pred_arr = sitk.GetArrayFromImage(pred_img)
        foreground = pred_arr > 0

        if np.any(foreground):
            cc = sitk.ConnectedComponent(sitk.GetImageFromArray(foreground.astype(np.uint8)))
            relabeled = sitk.RelabelComponent(cc, sortByObjectSize=True)
            largest_component = sitk.GetArrayFromImage(relabeled) == 1
            filtered_arr = np.zeros_like(pred_arr)
            filtered_arr[largest_component] = pred_arr[largest_component]
        else:
            filtered_arr = pred_arr

        out_img = sitk.GetImageFromArray(filtered_arr.astype(pred_arr.dtype, copy=False))
        out_img.CopyInformation(pred_img)
        sitk.WriteImage(out_img, str(output_dir / pred_path.name))
