"""MRecover: AI-powered T1w to T2 TSE MRI translation using flow matching."""

__version__ = "0.1.0"

import argparse
import time
from pathlib import Path

import torch
import numpy as np

from .core import AutoregressiveFlowMatcher, tse_flow_matching_inference
from .models import load_model, resolve_device
from .utils import (
    detect_input_format,
    load_tse_input_data,
    pad_volume_to_divisible,
    unpad_volume,
    save_results,
)


def translate(
    input_path,
    output_path,
    *,
    model_path=None,
    steps=1,
    device=None,
    fp16=True,
    compile_model=False,
    autoregressive=True,
    rk4=False,
    tse_inplane=0.375,
    tse_through_plane=None,
    tse_registered=False,
    seed=42,
):
    """Translate a T1w MRI volume to synthetic T2 TSE contrast.

    Args:
        input_path: Path to input T1w NIfTI (.nii / .nii.gz) or DICOM directory.
        output_path: Path for the output file (NIfTI) or directory (DICOM).
        model_path: Path to model checkpoint. If None, auto-downloads from HuggingFace.
        steps: Number of ODE integration steps (default 1; more = higher quality).
        device: 'cuda' or 'cpu'. Defaults to CUDA if available.
        fp16: Use half precision for faster GPU inference (default True).
        compile_model: Apply torch.compile for optimised inference (default False).
        autoregressive: Use generated slices as context for subsequent slices (default True).
        rk4: Use RK4 ODE solver instead of Euler (default False).
        tse_inplane: Target in-plane resolution in mm for native T1w (default 0.375).
        tse_through_plane: Through-plane resolution in mm; None keeps input spacing.
        tse_registered: Set True if input is already in TSE voxel space.
        seed: Random seed for reproducibility (default 42).

    Returns:
        np.ndarray: Generated T2 TSE volume (X, Y, Z).

    Example::

        import mrecover
        mrecover.translate("T1.nii.gz", "T2tse.nii.gz")
        mrecover.translate("T1.nii.gz", "T2tse.nii.gz", steps=10, tse_through_plane=1.5)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = resolve_device(device)

    input_format = detect_input_format(str(input_path))

    args = argparse.Namespace(
        input=str(input_path),
        output=str(output_path),
        format=input_format,
        steps=steps,
        no_fp16=not fp16,
        no_auto=not autoregressive,
        rk4=rk4,
        tse_inplane=tse_inplane,
        tse_through_plane=tse_through_plane,
        tse_registered=tse_registered,
        tse=True,
    )

    model = load_model(
        model_path=model_path,
        device=device,
        fp16=fp16,
        compile_model=compile_model,
    )
    flow_matcher = AutoregressiveFlowMatcher(model)

    volume_xyz, slice_axis, affine, header = load_tse_input_data(str(input_path), args)
    volume_xyz_padded, pad_info = pad_volume_to_divisible(volume_xyz, divisible_by=16)

    other_axes = tuple(a for a in (0, 1, 2) if a != slice_axis)
    perm_to_nhw = (slice_axis,) + other_axes
    inv_perm = tuple(perm_to_nhw.index(a) for a in (0, 1, 2))
    mprage_padded = volume_xyz_padded.permute(*perm_to_nhw)

    start = time.time()
    generated_volume, _ = tse_flow_matching_inference(flow_matcher, mprage_padded, args, device)
    print(f"Translation completed in {time.time() - start:.2f}s")

    generated_xyz = unpad_volume(generated_volume.permute(*inv_perm), pad_info).numpy()

    if input_format == "nifti":
        import torchio as tio
        gen_tensor = torch.from_numpy(generated_xyz).unsqueeze(0).float()
        generated_xyz = tio.ScalarImage(tensor=gen_tensor, affine=affine).data[0].numpy()

    save_results(generated_xyz, str(output_path), args, affine=affine, header=header)
    return generated_xyz


__all__ = [
    "translate",
    "AutoregressiveFlowMatcher",
    "tse_flow_matching_inference",
    "load_model",
    "resolve_device",
]
