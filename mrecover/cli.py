"""Command-line interface for MRecover: T1w to T2 TSE translation."""

import argparse
import sys
import os
import time
from pathlib import Path

import torch
import numpy as np

from .core import AutoregressiveFlowMatcher, tse_flow_matching_inference, DirectInference, direct_inference
from .models import load_model
from .utils import (
    detect_input_format,
    load_tse_input_data,
    pad_volume_to_divisible,
    unpad_volume,
    save_results,
)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="mrecover",
        description="MRecover: AI-powered T1w to T2 TSE MRI translation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Force through-plane resolution (fully anisotropic TSE-like spacing)
  mrecover -i T1.nii.gz -o T2tse.nii.gz --tse-through-plane 1.5

  # Input already registered to TSE space (skip resampling)
  mrecover -i T1_registered.nii.gz -o T2tse.nii.gz --tse-registered

  # More ODE steps (higher quality, slower)
  mrecover -i T1.nii.gz -o T2tse.nii.gz --steps 10

  # Use RK4 integration instead of Euler
  mrecover -i T1.nii.gz -o T2tse.nii.gz --rk4
        """,
    )

    parser.add_argument("-i", "--input", required=True, help="Input T1w MRI file path")
    parser.add_argument("-o", "--output", required=True, help="Output file path")

    parser.add_argument("--model", default=None,
                        help="Path to model checkpoint (default: auto-download from HuggingFace)")

    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument("--baseline", action="store_true", default=False,
                             help="Use baseline UNet model instead of flow matching")
    model_group.add_argument("--pix2pix", action="store_true", default=False,
                             help="Use pix2pix generator model instead of flow matching")
    parser.add_argument("--steps", type=int, default=1,
                        help="Number of ODE sampling steps (default: 1; more = higher quality)")
    parser.add_argument("--device", default="cuda",
                        help="Device: cuda or cpu (default: cuda)")
    parser.add_argument("--format", default=None, choices=["nifti", "dcm"],
                        help="Input format (default: auto-detect)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    parser.add_argument("--no-fp16", action="store_true",
                        help="Disable half precision (fp16)")
    parser.add_argument("--no-auto", action="store_true",
                        help="Disable autoregressive generation (use zeros as context)")
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile model optimization")
    parser.add_argument("--rk4", action="store_true", default=False,
                        help="Use RK4 integration instead of Euler (slower, more accurate)")

    parser.add_argument("--tse-registered", action="store_true", default=False,
                        help="Input T1w is already registered to TSE voxel space; skip resampling")
    parser.add_argument("--tse-inplane", type=float, default=0.375,
                        help="Target in-plane resolution in mm for native T1w (default: 0.375)")
    parser.add_argument("--tse-through-plane", type=float, default=None, dest="tse_through_plane",
                        help="Through-plane resolution in mm (e.g. 1.5); default: keep input spacing")

    return parser


def main():
    """CLI entry point for mrecover."""
    parser = build_parser()
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    if args.format is None:
        args.format = detect_input_format(args.input)
        print(f"Auto-detected input format: {args.format}")

    # Required by save_results for DICOM branching; always True in MRecover
    args.tse = True

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.baseline:
        model_type = "baseline"
    elif args.pix2pix:
        model_type = "pix2pix"
    else:
        model_type = "flowmatching"

    try:
        model = load_model(
            model_path=args.model,
            model_type=model_type,
            fp16=not args.no_fp16,
            compile_model=not args.no_compile,
        )

        if model_type == "flowmatching":
            inferencer = AutoregressiveFlowMatcher(model)
        else:
            inferencer = DirectInference(model)

        print(f"Running T1w to T2 TSE translation with {model_type} model...")
        _run_translation(args, inferencer, device)

        print(f"Done. Output saved to: {args.output}")

    except Exception as e:
        print(f"Error during processing: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


def _run_translation(args, inferencer, device):
    """Execute the full T1w→T2 TSE translation pipeline."""
    volume_xyz, slice_axis, affine, header = load_tse_input_data(args.input, args)

    volume_xyz_padded, pad_info = pad_volume_to_divisible(volume_xyz, divisible_by=16)
    print(f"Input volume (XYZ, padded): {tuple(volume_xyz_padded.shape)}")

    other_axes = tuple(a for a in (0, 1, 2) if a != slice_axis)
    perm_to_nhw = (slice_axis,) + other_axes
    inv_perm = tuple(perm_to_nhw.index(a) for a in (0, 1, 2))
    mprage_padded = volume_xyz_padded.permute(*perm_to_nhw)

    start = time.time()
    if isinstance(inferencer, AutoregressiveFlowMatcher):
        generated_volume, _ = tse_flow_matching_inference(inferencer, mprage_padded, args, device)
    else:
        generated_volume, _ = direct_inference(inferencer, mprage_padded, args, device)
    elapsed = time.time() - start
    print(f"Translation completed in {elapsed:.2f}s ({elapsed / len(mprage_padded):.2f}s/slice)")

    generated_xyz = unpad_volume(generated_volume.permute(*inv_perm), pad_info).numpy()
    print(f"Output shape: {generated_xyz.shape}")

    if args.format == "nifti":
        import torchio as tio
        gen_tensor = torch.from_numpy(generated_xyz).unsqueeze(0).float()
        source_img = tio.ScalarImage(tensor=gen_tensor, affine=affine)
        generated_xyz = source_img.data[0].numpy()

    save_results(generated_xyz, args.output, args, affine=affine, header=header)


if __name__ == "__main__":
    main()
