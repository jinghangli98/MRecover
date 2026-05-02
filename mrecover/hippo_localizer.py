"""Hippocampus localization for restricting inference to relevant slices."""

import numpy as np
import torch
import torchio as tio


def localize_hippocampus_slices(
    input_path,
    target_affine,
    target_shape,
    slice_axis,
    pad_info,
    margin_mm=5.0,
):
    """Identify slice indices containing hippocampus in the resampled+padded volume.

    Runs the hippocampus segmentation on the original input, resamples the
    combined L+R mask onto the same grid as the T1w volume that will be fed to
    the translation model, applies the same padding, and returns a contiguous
    range of slice indices (extended by `margin_mm` mm on each side) along
    `slice_axis` that contain hippocampus.

    Args:
        input_path: Path to original T1w NIfTI input.
        target_affine: (4, 4) affine of the resampled T1w volume.
        target_shape: (X, Y, Z) shape of the resampled T1w volume (pre-padding).
        slice_axis: Axis that the inference loop iterates over after permutation.
        pad_info: dict returned by `pad_volume_to_divisible` for the T1w volume.
        margin_mm: Physical margin in millimetres on each side of the hippocampus.
            Converted to slices using the spacing of `slice_axis` in `target_affine`.

    Returns:
        list[int] of slice indices, or None if no hippocampus was detected.
    """
    from .segmentation import segment_hippocampus

    print("Localizing hippocampus...")
    L_nii, R_nii = segment_hippocampus(input_path)

    mask_data = ((np.asarray(L_nii.get_fdata()) + np.asarray(R_nii.get_fdata())) > 0).astype(np.uint8)
    mask_img = tio.LabelMap(
        tensor=torch.from_numpy(mask_data)[None],
        affine=L_nii.affine,
    )

    target_template = tio.ScalarImage(
        tensor=torch.zeros((1, *target_shape), dtype=torch.float32),
        affine=target_affine,
    )
    mask_img = tio.Resample(target=target_template, image_interpolation="nearest")(mask_img)
    mask_xyz = mask_img.data[0].numpy()

    (x_l, x_r), (y_l, y_r), (z_l, z_r) = pad_info["padding"]
    mask_padded = np.pad(
        mask_xyz,
        ((x_l, x_r), (y_l, y_r), (z_l, z_r)),
        mode="constant",
        constant_values=0,
    )

    other_axes = tuple(a for a in (0, 1, 2) if a != slice_axis)
    slice_sums = mask_padded.sum(axis=other_axes)
    nonzero = np.where(slice_sums > 0)[0]

    total = mask_padded.shape[slice_axis]
    if len(nonzero) == 0:
        print("Warning: no hippocampus detected; falling back to whole-brain inference.")
        return None

    spacing_mm = float(np.linalg.norm(np.asarray(target_affine)[:3, slice_axis]))
    margin_slices = max(0, int(round(float(margin_mm) / spacing_mm))) if spacing_mm > 0 else 0

    lo = max(0, int(nonzero.min()) - margin_slices)
    hi = min(total, int(nonzero.max()) + margin_slices + 1)
    indices = list(range(lo, hi))
    print(
        f"Hippocampus localized along axis {slice_axis}: "
        f"slices [{lo}, {hi}) — {len(indices)} of {total} "
        f"(margin={margin_mm:.1f} mm = {margin_slices} slices @ {spacing_mm:.3f} mm/slice)"
    )
    return indices
