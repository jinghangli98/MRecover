"""Data loading, preprocessing, and saving utilities for MRecover."""

import os
import tempfile
from contextlib import contextmanager
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


def _mask_tilt_deg(mask_data, affine, min_voxels=10):
    """Estimate one hippocampus mask's sagittal long-axis angle in degrees."""
    points_ijk = np.argwhere(np.asarray(mask_data) > 0)
    if points_ijk.shape[0] < int(min_voxels):
        return None

    points_h = np.c_[points_ijk.astype(np.float64), np.ones(points_ijk.shape[0])]
    points_world = (np.asarray(affine, dtype=np.float64) @ points_h.T).T[:, :3]
    sagittal_yz = points_world[:, [1, 2]]
    sagittal_yz -= sagittal_yz.mean(axis=0, keepdims=True)

    _, _, vh = np.linalg.svd(sagittal_yz, full_matrices=False)
    long_axis = vh[0]
    if long_axis[0] < 0:
        long_axis *= -1

    return float(np.rad2deg(np.arctan2(long_axis[1], long_axis[0])))


def estimate_tse_tilt_from_masks(left_mask, right_mask, affine, clamp_degrees=45.0, min_voxels=10):
    """Estimate coronal-oblique TSE tilt from hippocampus mask long axes.

    The angle is measured in the sagittal A-S plane from world +A toward +S,
    matching the convention used by ``build_oblique_target`` for rotations
    around the RAS left-right axis.
    """
    angles = []
    for mask in (left_mask, right_mask):
        if mask is None:
            continue
        angle = _mask_tilt_deg(mask, affine, min_voxels=min_voxels)
        if angle is not None and np.isfinite(angle):
            angles.append(angle)

    if not angles:
        raise ValueError("no usable hippocampus voxels for auto tilt")

    tilt_deg = float(np.median(angles))
    clamp = abs(float(clamp_degrees))
    return float(np.clip(tilt_deg, -clamp, clamp))


def estimate_tse_tilt_for_input(input_path, input_format=None):
    """Segment hippocampus and estimate the TSE oblique tilt angle."""
    from contextlib import nullcontext
    from .segmentation import segment_hippocampus

    input_format = input_format or detect_input_format(input_path)
    localizer_ctx = (
        nullcontext(str(input_path))
        if input_format == "nifti"
        else dump_dicom_to_temp_nifti(str(input_path))
    )

    with localizer_ctx as localizer_path:
        left_nii, right_nii = segment_hippocampus(localizer_path)

    left_mask = np.asarray(left_nii.get_fdata()) > 0
    right_mask = np.asarray(right_nii.get_fdata()) > 0
    return estimate_tse_tilt_from_masks(left_mask, right_mask, left_nii.affine)


def build_oblique_target(source, tilt_deg, spacing, inplane_shape):
    """Construct an oblique TSE-like resampling target.

    The acquisition box is rotated by ``tilt_deg`` degrees around the L-R (X)
    axis in RAS+ world coordinates so the slice plane sits perpendicular to a
    tilted hippocampal long axis (mimicking a Siemens coronal-oblique TSE).

    Voxel-to-axis mapping in the returned template:
      * axis 0 — in-plane, along world R direction (``spacing[0]`` mm)
      * axis 1 — in-plane, along the tilted S direction (``spacing[1]`` mm)
      * axis 2 — slice/through-plane, along the tilted A direction (``spacing[2]`` mm)

    The slice count is chosen to cover the source's bounding box along the
    tilted slice axis. The target volume is centred on the source's centroid.

    Args:
        source: ``tio.ScalarImage`` whose FOV must be covered.
        tilt_deg: Tilt angle in degrees around the L-R axis.
        spacing: ``(in0, in1, slice)`` voxel spacing in mm.
        inplane_shape: ``(n0, n1)`` in-plane matrix size.

    Returns:
        ``tio.ScalarImage`` template (zero-valued) with the oblique affine.
    """
    src_affine = np.asarray(source.affine, dtype=np.float64)
    src_shape = source.spatial_shape

    sp_in0 = float(spacing[0])
    sp_in1 = float(spacing[1])
    sp_slice = float(spacing[2])
    n_in0 = int(inplane_shape[0])
    n_in1 = int(inplane_shape[1])

    theta = np.deg2rad(float(tilt_deg))
    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))

    rot_x = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cos_t, -sin_t],
        [0.0, sin_t, cos_t],
    ])

    base_dirs = np.array([
        [1.0, 0.0, 0.0],   # axis 0 → world +X (Right)
        [0.0, 0.0, 1.0],   # axis 1 → world +Z (Superior)
        [0.0, 1.0, 0.0],   # axis 2 → world +Y (Anterior)
    ])
    rotated_dirs = base_dirs @ rot_x.T

    src_corners = np.array([
        src_affine @ np.array([i, j, k, 1.0])
        for i in (0.0, float(src_shape[0]))
        for j in (0.0, float(src_shape[1]))
        for k in (0.0, float(src_shape[2]))
    ])[:, :3]

    slice_proj = src_corners @ rotated_dirs[2]
    span_slice = float(slice_proj.max() - slice_proj.min())
    n_slice = max(int(np.ceil(span_slice / sp_slice)), 1)

    target_affine = np.eye(4)
    target_affine[:3, 0] = rotated_dirs[0] * sp_in0
    target_affine[:3, 1] = rotated_dirs[1] * sp_in1
    target_affine[:3, 2] = rotated_dirs[2] * sp_slice

    src_center = src_corners.mean(axis=0)
    voxel_center = np.array([
        (n_in0 - 1) / 2.0,
        (n_in1 - 1) / 2.0,
        (n_slice - 1) / 2.0,
    ])
    target_affine[:3, 3] = src_center - target_affine[:3, :3] @ voxel_center

    return tio.ScalarImage(
        tensor=torch.zeros((1, n_in0, n_in1, n_slice), dtype=torch.float32),
        affine=target_affine,
    )


def load_tse_input_data(input_path, args):
    """Load a T1w NIfTI/DCM volume for T1→TSE translation.

    Modes selected by ``args``:

    **Mode A — native T1w** (default):
        Reorient to RAS+, resample in-plane (X, Z) to ``args.tse_inplane`` mm,
        optionally resample through-plane (Y) to ``args.tse_through_plane`` mm.

    **Mode B — pre-registered** (``args.tse_registered=True``):
        Load as-is, no resampling.

    **Mode C — oblique** (``args.tse_tilt_deg`` set):
        Reorient to RAS+, then resample onto a coronal-oblique grid tilted by
        ``args.tse_tilt_deg`` degrees around L-R, with in-plane shape from
        ``args.tse_inplane_shape`` (default ``(456, 512)``) and spacing
        ``(tse_inplane, tse_inplane, tse_through_plane or 1.5)`` mm. Slice axis
        is 2.

    Args:
        input_path: str, path to NIfTI file or DICOM directory.
        args: Namespace with ``tse_registered``, ``tse_inplane``,
            ``tse_through_plane``, optional ``tse_tilt_deg`` and
            ``tse_inplane_shape``.

    Returns:
        data_xyz: torch.Tensor (X, Y, Z) — normalised
        slice_axis: int (1 native axial, 2 oblique or pre-registered)
        affine: np.ndarray (4, 4)
        header: NIfTI header
    """
    tse_inplane = getattr(args, "tse_inplane", 0.375)
    tse_through = getattr(args, "tse_through_plane", None)
    registered = getattr(args, "tse_registered", False)
    tilt_deg = getattr(args, "tse_tilt_deg", None)
    inplane_shape = getattr(args, "tse_inplane_shape", (456, 512))

    input_format = getattr(args, "format", None) or detect_input_format(input_path)

    if input_format == "dcm":
        folder = input_path if Path(input_path).is_dir() else str(Path(input_path).parent or ".")
        series_ids = sitk.ImageSeriesReader().GetGDCMSeriesIDs(folder)
        if len(series_ids) > 1:
            print(f"Warning: {len(series_ids)} DICOM series found in {folder}; using first ({series_ids[0]}).")
        _, array_xyz, dcm_affine = load_dicom_with_sitk(input_path, reorient=True)
        tensor = torch.from_numpy(array_xyz).unsqueeze(0).float()
        image = tio.ScalarImage(tensor=tensor, affine=dcm_affine)
    else:
        image = tio.ScalarImage(input_path)

    if registered:
        if input_format != "dcm":
            image = tio.ToCanonical()(image)
        print(f"Pre-registered mode: loading T1w as-is "
              f"(spacing: {tuple(f'{s:.3f}' for s in image.spacing)} mm)")
        slice_axis = 2
    elif tilt_deg is not None:
        image = tio.ToCanonical()(image)
        through = float(tse_through) if tse_through is not None else 1.5
        spacing = (float(tse_inplane), float(tse_inplane), through)
        target = build_oblique_target(image, tilt_deg, spacing, inplane_shape)
        print(f"Resampling T1w to oblique TSE grid "
              f"(tilt={tilt_deg}°, spacing={spacing} mm, "
              f"shape={tuple(target.spatial_shape)})")
        image = tio.Resample(target=target)(image)
        slice_axis = 2
    else:
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
        slice_axis = 1

    data_np = image.data[0].numpy()  # (X, Y, Z)
    affine = image.affine
    header = nib.Nifti1Image(data_np, affine).header

    data_np = quantile_normalization(data_np)
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
        ds.MRAcquisitionType = "2D"

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


def save_oblique_dicom(volume, donor_path, output_folder, *,
                       slice_axis, affine, series_description_suffix="_T2TSE"):
    """Write an oblique-acquired volume as a DICOM series.

    Each slice perpendicular to ``slice_axis`` becomes a single DICOM file.
    ``ImageOrientationPatient`` and ``ImagePositionPatient`` are derived from
    the RAS+ ``affine`` (converted to DICOM's LPS frame) so dcm2niix recovers
    the oblique geometry on read-back.

    Patient/study metadata is inherited from ``donor_path`` (a DICOM file or
    directory). Geometry-related tags are overwritten.

    Args:
        volume: np.ndarray with shape ``(d0, d1, d2)``; the through-plane axis
            is ``slice_axis`` (must be 2 in current usage).
        donor_path: Path to a DICOM file or directory used as a metadata donor.
        output_folder: Destination directory for the new series.
        slice_axis: Index of the through-plane axis (currently always 2).
        affine: (4, 4) RAS+ affine of ``volume``.
        series_description_suffix: Appended to the donor's SeriesDescription.
    """
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    if slice_axis != 2:
        order = [a for a in (0, 1, 2) if a != slice_axis] + [slice_axis]
        volume = np.transpose(volume, order)
        new_affine = np.array(affine, copy=True)
        new_affine[:3, 0] = affine[:3, order[0]]
        new_affine[:3, 1] = affine[:3, order[1]]
        new_affine[:3, 2] = affine[:3, order[2]]
        affine = new_affine

    n_axis0, n_axis1, n_slices = volume.shape

    ras_to_lps = np.diag([-1.0, -1.0, 1.0, 1.0])
    affine_lps = ras_to_lps @ np.asarray(affine, dtype=np.float64)

    sp_axis0 = float(np.linalg.norm(affine_lps[:3, 0]))
    sp_axis1 = float(np.linalg.norm(affine_lps[:3, 1]))
    sp_slice = float(np.linalg.norm(affine_lps[:3, 2]))

    axis0_dir = affine_lps[:3, 0] / sp_axis0
    axis1_dir = affine_lps[:3, 1] / sp_axis1
    slice_dir = affine_lps[:3, 2] / sp_slice
    origin = affine_lps[:3, 3]

    # dcm2niix convention: NIfTI dim[1] ← DICOM Columns, dim[2] ← DICOM Rows.
    # We want NIfTI shape (n_axis0, n_axis1, n_slices), so:
    #   Columns = n_axis0  (varies fastest, "first row direction" = axis0_dir)
    #   Rows    = n_axis1  ("first column direction" = axis1_dir)
    # Pixel layout for each DICOM slice is (Rows, Columns), i.e. volume[:, :, k].T.
    n_dicom_rows = n_axis1
    n_dicom_cols = n_axis0
    pixel_spacing = [sp_axis1, sp_axis0]  # [row spacing, column spacing]

    donor_folder = donor_path if Path(donor_path).is_dir() else str(Path(donor_path).parent or ".")
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(donor_folder)
    if not series_ids:
        raise ValueError(f"No DICOM series found in donor path: {donor_folder}")
    donor_files = reader.GetGDCMSeriesFileNames(donor_folder, series_ids[0])

    new_series_uid = pydicom.uid.generate_uid()

    for k in range(n_slices):
        slice_data = np.ascontiguousarray(volume[:, :, k].T.astype(np.int16))
        slice_position = origin + k * sp_slice * slice_dir

        ds = pydicom.dcmread(donor_files[min(k, len(donor_files) - 1)])

        ds.PixelData = slice_data.tobytes()
        ds.Rows = int(n_dicom_rows)
        ds.Columns = int(n_dicom_cols)
        ds.PixelSpacing = pixel_spacing
        ds.MRAcquisitionType = "2D"
        if hasattr(ds, "SliceThickness"):
            ds.SliceThickness = sp_slice
        if hasattr(ds, "SpacingBetweenSlices"):
            ds.SpacingBetweenSlices = sp_slice
        else:
            ds.SpacingBetweenSlices = sp_slice

        ds.ImageOrientationPatient = [
            float(axis0_dir[0]), float(axis0_dir[1]), float(axis0_dir[2]),
            float(axis1_dir[0]), float(axis1_dir[1]), float(axis1_dir[2]),
        ]
        ds.ImagePositionPatient = [
            float(slice_position[0]),
            float(slice_position[1]),
            float(slice_position[2]),
        ]

        ds.SeriesInstanceUID = new_series_uid
        ds.SeriesDescription = (
            getattr(ds, "SeriesDescription", "T2TSE") + series_description_suffix
        )
        ds.InstanceNumber = k + 1
        ds.SOPInstanceUID = pydicom.uid.generate_uid()
        ds.save_as(os.path.join(output_folder, f"mrecover_{k:04d}.dcm"))

    print(f"Saved {n_slices} oblique DICOM files to {output_folder}")


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

    out_fmt = getattr(args, "output_format", None) or args.format
    tilt_deg = getattr(args, "tse_tilt_deg", None)

    if out_fmt == "nifti":
        os.makedirs(Path(output_path).parent, exist_ok=True)
        nii_img = nib.Nifti1Image(volume_data, affine, header) if header is not None else nib.Nifti1Image(volume_data, affine)
        nii_img.set_data_dtype(np.uint8)
        nib.save(nii_img, output_path)
        print(f"Saved NIfTI: {output_path}")

    elif out_fmt == "dcm":
        output_folder = str(Path(output_path) / f"{Path(args.input).name}_T2TSE")
        if tilt_deg is not None:
            save_oblique_dicom(
                volume_data,
                args.input,
                output_folder,
                slice_axis=2,
                affine=affine,
                series_description_suffix="_T2TSE",
            )
        else:
            spacing = (abs(float(affine[0][0])), abs(float(affine[1][1])), abs(float(affine[2][2])))
            save_enhanced_dicom(volume_data, args.input, output_folder,
                                new_spacing=spacing, series_description_suffix="_T2TSE")


@contextmanager
def dump_dicom_to_temp_nifti(dicom_path):
    """Yield a temp `.nii.gz` path containing the DICOM volume in RAS+ orientation.

    The temp file is deleted on exit. Used so format-agnostic helpers that
    require a NIfTI file path (e.g. the hippocampus localizer) can run on
    DICOM input.
    """
    _, array_xyz, dcm_affine = load_dicom_with_sitk(dicom_path, reorient=True)
    tmp = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False)
    tmp_path = tmp.name
    tmp.close()
    try:
        nib.Nifti1Image(array_xyz.astype(np.float32), dcm_affine).to_filename(tmp_path)
        yield tmp_path
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
