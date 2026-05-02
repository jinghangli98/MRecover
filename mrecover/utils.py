"""Data loading, preprocessing, and saving utilities for MRecover."""

import os
import numpy as np
import torch
import nibabel as nib
import torchio as tio
import SimpleITK as sitk
import pydicom
from pathlib import Path
import torch.nn.functional as F


def pad_volume_to_divisible(data, divisible_by=16):
    """Symmetrically pad a 3-D volume (X, Y, Z) so every dim is divisible by `divisible_by`.

    Returns:
        padded_data: Tensor (X', Y', Z')
        pad_info: dict with 'original_shape' and 'padding'
    """
    original_shape = data.shape

    def _pad_amount(size):
        remainder = size % divisible_by
        if remainder == 0:
            return 0, 0
        total = divisible_by - remainder
        left = total // 2
        return left, total - left

    x_l, x_r = _pad_amount(original_shape[0])
    y_l, y_r = _pad_amount(original_shape[1])
    z_l, z_r = _pad_amount(original_shape[2])

    padded_data = F.pad(data, (z_l, z_r, y_l, y_r, x_l, x_r), mode="constant", value=0)
    pad_info = {
        "original_shape": original_shape,
        "padding": ((x_l, x_r), (y_l, y_r), (z_l, z_r)),
    }
    return padded_data, pad_info


def unpad_volume(padded_data, pad_info):
    """Remove padding added by `pad_volume_to_divisible`.

    Args:
        padded_data: Tensor or ndarray (X', Y', Z')
        pad_info: dict returned by `pad_volume_to_divisible`

    Returns:
        Tensor with shape == pad_info['original_shape']
    """
    if isinstance(padded_data, np.ndarray):
        padded_data = torch.from_numpy(padded_data)

    (x_l, x_r), (y_l, y_r), (z_l, z_r) = pad_info["padding"]
    ox, oy, oz = pad_info["original_shape"]
    return padded_data[x_l : x_l + ox, y_l : y_l + oy, z_l : z_l + oz]


def crop_to_slice_range(volume_xyz, slice_axis, lo, hi, affine):
    """Crop an (X, Y, Z) volume to [lo, hi) along `slice_axis` and shift the affine.

    The returned affine keeps every cropped voxel's world coordinates identical
    to the original — only the origin shifts so that voxel index 0 of the crop
    sits where index `lo` did in the source.

    Args:
        volume_xyz: np.ndarray or torch.Tensor with shape (X, Y, Z).
        slice_axis: int in {0, 1, 2}.
        lo, hi: crop bounds (clamped to [0, dim]).
        affine: (4, 4) np.ndarray of the source volume.

    Returns:
        cropped_volume, new_affine
    """
    dim = volume_xyz.shape[slice_axis]
    lo = max(0, min(int(lo), dim))
    hi = max(lo, min(int(hi), dim))

    crop = [slice(None)] * 3
    crop[slice_axis] = slice(lo, hi)
    cropped = volume_xyz[tuple(crop)]

    new_affine = np.array(affine, copy=True)
    new_affine[:3, 3] = affine[:3, 3] + lo * affine[:3, slice_axis]
    return cropped, new_affine


def quantile_normalization(data, lower_quantile=0.01, upper_quantile=0.99):
    """Normalize voxel values to [0, 1] using percentile clipping."""
    data = np.nan_to_num(data, nan=0.0)
    data_flat = data.flatten()
    lower = np.percentile(data_flat, lower_quantile * 100)
    upper = np.percentile(data_flat, upper_quantile * 100)
    data_clipped = np.clip(data, lower, upper)
    return (data_clipped - lower) / (upper - lower + 1e-3)


def detect_input_format(input_path_str):
    """Auto-detect if input is NIfTI, DICOM, or PNG."""
    path = Path(input_path_str)
    if not path.exists():
        return "nifti"
    if path.is_dir():
        return "dcm"
    suffix = path.name.lower()
    if suffix.endswith(".nii") or suffix.endswith(".nii.gz"):
        return "nifti"
    if suffix.endswith(".dcm"):
        return "dcm"
    if suffix.endswith(".png") or suffix.endswith(".jpg"):
        return "png"
    return "nifti"


def load_dicom_with_sitk(folder_path, reorient=False):
    """Load a DICOM series using SimpleITK."""
    if os.path.isfile(folder_path):
        folder_path = os.path.dirname(folder_path) or "."

    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(folder_path)
    if not series_ids:
        raise ValueError("No DICOM series found.")

    reader.SetFileNames(reader.GetGDCMSeriesFileNames(folder_path, series_ids[0]))
    image = reader.Execute()

    if reorient:
        image = sitk.DICOMOrient(image, "RAS")

    array = sitk.GetArrayFromImage(image).transpose(2, 1, 0)  # (z,y,x) -> (x,y,z)

    affine = np.eye(4)
    affine[:3, :3] = np.array(image.GetDirection()).reshape(3, 3) * np.array(image.GetSpacing())
    affine[:3, 3] = image.GetOrigin()
    affine[0, :] *= -1  # LPS -> RAS
    affine[1, :] *= -1

    return image, array, affine


def load_tse_input_data(input_path, args):
    """Load a T1w NIfTI for T1→TSE translation.

    Two modes controlled by args.tse_registered:

    **Mode A — native T1w** (default):
        Reorient to RAS+, resample in-plane (X, Z) to args.tse_inplane mm,
        optionally resample through-plane (Y) to args.tse_through_plane mm.

    **Mode B — pre-registered** (args.tse_registered=True):
        Load as-is, no resampling.

    Args:
        input_path: str, path to NIfTI file
        args: Namespace with tse_registered, tse_inplane, tse_through_plane

    Returns:
        data_xyz: torch.Tensor (X, Y, Z) — normalised
        slice_axis: int (1 for native, 2 for pre-registered)
        affine: np.ndarray (4, 4)
        header: NIfTI header
    """
    tse_inplane = getattr(args, "tse_inplane", 0.375)
    tse_through = getattr(args, "tse_through_plane", None)
    registered = getattr(args, "tse_registered", False)

    image = tio.ScalarImage(input_path)

    if not registered:
        image = tio.ToCanonical()(image)
        orig_spacing = image.spacing
        if tse_through is not None:
            target_spacing = (tse_inplane, float(tse_through), tse_inplane)
            print(f"Resampling T1w to {tse_inplane}×{tse_through}×{tse_inplane} mm")
        else:
            target_spacing = (tse_inplane, float(orig_spacing[1]), tse_inplane)
            print(f"Resampling T1w in-plane to {tse_inplane}×{tse_inplane} mm "
                  f"(through-plane kept at {orig_spacing[1]:.3f} mm)")
        image = tio.Resample(target_spacing)(image)
    else:
        print(f"Pre-registered mode: loading T1w as-is "
              f"(spacing: {tuple(f'{s:.3f}' for s in image.spacing)} mm)")

    data_np = image.data[0].numpy()  # (X, Y, Z)
    affine = image.affine
    header = nib.Nifti1Image(data_np, affine).header

    data_np = quantile_normalization(data_np)
    slice_axis = 2 if registered else 1
    data_xyz = torch.from_numpy(data_np).float()

    print(f"TSE input tensor shape (X, Y, Z): {tuple(data_xyz.shape)}  slice_axis={slice_axis}")
    return data_xyz, slice_axis, affine, header


def save_enhanced_dicom(enhanced_array, original_dicom_folder, output_folder,
                        new_spacing=(0.375, 1.5, 0.375), series_description_suffix="_T2TSE"):
    """Save TSE-translated image as DICOM series preserving original metadata."""
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(original_dicom_folder)
    if not series_ids:
        raise ValueError("No DICOM series found.")

    series_file_names = reader.GetGDCMSeriesFileNames(original_dicom_folder, series_ids[0])
    reader.SetFileNames(series_file_names)
    native_image = reader.Execute()
    ras_image = sitk.DICOMOrient(native_image, "RAS")

    enhanced_sitk_array = enhanced_array.transpose(2, 1, 0)
    enhanced_image = sitk.GetImageFromArray(enhanced_sitk_array)
    enhanced_image.SetDirection(ras_image.GetDirection())
    enhanced_image.SetOrigin(ras_image.GetOrigin())
    enhanced_image.SetSpacing([float(s) for s in new_spacing])

    orig_orient = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(native_image.GetDirection())
    native_enhanced = sitk.DICOMOrient(enhanced_image, orig_orient)

    native_array = sitk.GetArrayFromImage(native_enhanced)
    native_spacing = native_enhanced.GetSpacing()
    native_origin = native_enhanced.GetOrigin()
    native_direction = np.array(native_enhanced.GetDirection()).reshape(3, 3)

    num_slices = native_array.shape[0]
    new_series_uid = pydicom.uid.generate_uid()

    for i in range(num_slices):
        slice_position = np.array(native_origin) + i * native_spacing[2] * native_direction[:, 2]
        ds = pydicom.dcmread(series_file_names[min(i, len(series_file_names) - 1)])

        slice_data = native_array[i]
        ds.PixelData = slice_data.astype(np.int16).tobytes()
        ds.Rows, ds.Columns = slice_data.shape
        ds.PixelSpacing = [float(native_spacing[1]), float(native_spacing[0])]

        if hasattr(ds, "SliceThickness"):
            ds.SliceThickness = float(native_spacing[2])
        if hasattr(ds, "SpacingBetweenSlices"):
            ds.SpacingBetweenSlices = float(native_spacing[2])

        ds.ImagePositionPatient = [float(slice_position[0]), float(slice_position[1]), float(slice_position[2])]
        ds.SeriesInstanceUID = new_series_uid
        ds.SeriesDescription = getattr(ds, "SeriesDescription", "T2TSE") + series_description_suffix
        ds.InstanceNumber = i + 1
        ds.SOPInstanceUID = pydicom.uid.generate_uid()
        ds.save_as(os.path.join(output_folder, f"mrecover_{i:04d}.dcm"))

    print(f"Saved {num_slices} DICOM files to {output_folder}")


def save_results(generated_volume, output_path, args, affine=None, header=None):
    """Save TSE translation output as NIfTI or DICOM.

    Args:
        generated_volume: np.ndarray (X, Y, Z)
        output_path: str, output file or directory path
        args: Namespace with .format and .input
        affine: (4, 4) affine matrix
        header: NIfTI header (optional)
    """
    finite_mask = np.isfinite(generated_volume)
    if finite_mask.any():
        finite_values = generated_volume[finite_mask]
        min_val = np.min(finite_values)
        max_val = np.max(finite_values)
        if max_val > min_val:
            scaled_volume = np.zeros(generated_volume.shape, dtype=np.float32)
            scaled_volume[finite_mask] = (finite_values - min_val) / (max_val - min_val)
            volume_data = np.clip(scaled_volume * 255, 0, 255).astype(np.uint8)
        else:
            volume_data = np.zeros(generated_volume.shape, dtype=np.uint8)
    else:
        volume_data = np.zeros(generated_volume.shape, dtype=np.uint8)

    if args.format == "nifti":
        os.makedirs(Path(output_path).parent, exist_ok=True)
        nii_img = nib.Nifti1Image(volume_data, affine, header) if header is not None else nib.Nifti1Image(volume_data, affine)
        nii_img.set_data_dtype(np.uint8)
        nib.save(nii_img, output_path)
        print(f"Saved NIfTI: {output_path}")

    elif args.format == "dcm":
        spacing = (abs(float(affine[0][0])), abs(float(affine[1][1])), abs(float(affine[2][2])))
        output_folder = str(Path(output_path) / f"{Path(args.input).name}_T2TSE")
        save_enhanced_dicom(volume_data, args.input, output_folder,
                            new_spacing=spacing, series_description_suffix="_T2TSE")
