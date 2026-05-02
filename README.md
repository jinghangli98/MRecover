# MRecover

AI-powered T1w to T2 TSE MRI translation using autoregressive flow matching.

MRecover synthesizes T2-weighted turbo spin echo (TSE) images from T1-weighted (MPRAGE) brain MRI, enabling recovery of T2 TSE contrast without re-scanning.

## Installation from github

```bash
git clone https://github.com/jinghangli98/MRecover.git
cd MRecover
pip install -e .
```
## Installation from PyPI
```bash
pip install mrecover
```

## Model
The models are hosted at https://huggingface.co/jil202/MRecover. You will need to agree to share your contact information to access the models.
The model will be automatically downloaded from HuggingFace on first use. You will need to authenticate:

```bash
huggingface-cli login
```

## Docker

Build the simple CPU image:

```bash
docker build -t mrecover:cpu .
```

Build the CUDA image for NVIDIA GPU hosts:

```bash
docker build -f Dockerfile.cuda -t mrecover:cuda12.1 .
```

Run with your Hugging Face token and a mounted data directory:

```bash
docker run --rm -e HF_TOKEN=$HF_TOKEN -v "$PWD:/data" mrecover:cpu -i /data/T1.nii.gz -o /data/T2tse.nii.gz --device cpu
```

For CUDA, use Docker on a Linux host with NVIDIA Container Toolkit:

```bash
docker run --rm --gpus all -e HF_TOKEN=$HF_TOKEN -v "$PWD:/data" mrecover:cuda12.1 -i /data/T1.nii.gz -o /data/T2tse.nii.gz --device cuda
```

## Quick Start

### CLI

```bash
# Default: only synthesise the coronal slab containing the hippocampus
mrecover -i T1.nii.gz -o T2tse_hippo.nii.gz

# Synthesise the whole brain (skip hippocampus localization)
mrecover -i T1.nii.gz -o T2tse_whole.nii.gz --whole-brain

# Force anisotropic TSE-like spacing (0.375 × 1.5 × 0.375 mm)
mrecover -i T1.nii.gz -o T2tse.nii.gz --tse-through-plane 1.5

# Input already registered to TSE space (skip resampling)
mrecover -i T1_registered.nii.gz -o T2tse.nii.gz --tse-registered

# Higher quality with more ODE steps
mrecover -i T1.nii.gz -o T2tse.nii.gz --steps 10
```

### Hippocampus localization (default)

By default MRecover localizes the hippocampus on the input T1w using a lightweight 3D segmentation model and only synthesises slices that contain it (plus a 10 mm margin on each side for autoregressive warm-up). The margin is specified in millimetres so it is robust across through-plane resolutions. The saved NIfTI is cropped to that slab and its affine is shifted so it overlays the input T1w in any viewer with no spatial offset. Pass `--whole-brain` to skip localization and synthesise every slice; tune the margin with `--hippo-margin <mm>`.

### Python API

```python
import mrecover

# Simple translation
mrecover.translate("T1.nii.gz", "T2tse.nii.gz")

# With options
mrecover.translate(
    "T1.nii.gz",
    "T2tse.nii.gz",
    steps=10,
    tse_through_plane=1.5,   # resample through-plane to 1.5 mm
    tse_inplane=0.375,        # target in-plane resolution
    whole_brain=False,        # default: localize hippocampus and crop output
    hippo_margin=10.0,         # margin in mm on each side of the hippocampus
)

# Returns the generated volume as a numpy array
volume = mrecover.translate("T1.nii.gz", "T2tse.nii.gz")
print(volume.shape)  # (X, Y, Z)
```

## CLI Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `-i, --input` | required | Input T1w NIfTI or DICOM directory |
| `-o, --output` | required | Output file path |
| `--steps` | 1 | ODE integration steps (more = higher quality, slower) |
| `--device` | cuda | Device: `cuda` or `cpu` |
| `--rk4` | off | Use RK4 instead of Euler ODE solver |
| `--no-fp16` | off | Disable half precision |
| `--no-auto` | off | Disable autoregressive slice context |
| `--tse-inplane` | 0.375 | In-plane resampling target (mm) |
| `--tse-through-plane` | None | Through-plane resampling target (mm) |
| `--tse-registered` | off | Skip resampling (input already in TSE space) |
| `--whole-brain` | off | Synthesise every slice. Default: localize hippocampus and synthesise only that slab. |
| `--hippo-margin` | 10.0 | Margin in millimetres on each side of the localized hippocampus region. Ignored with `--whole-brain`. |
| `--model` | None | Path to custom model checkpoint |
| `--seed` | 42 | Random seed |

## Input / Output

**Input:** T1w MPRAGE NIfTI (`.nii` / `.nii.gz`) at any isotropic resolution (e.g. 0.55–1 mm), or a DICOM series directory.

**Output:** Synthetic T2 TSE NIfTI or DICOM. By default the output is resampled to 0.375 mm in-plane with the through-plane spacing preserved from the input. When hippocampus localization is on (default), the NIfTI is cropped to the hippocampus slab with the affine adjusted to preserve world-space alignment with the input.

## Acknowledgments

Hippocampus localization in MRecover uses the segmentation model from [hippodeep_pytorch](https://github.com/bthyreau/hippodeep_pytorch) by Benjamin Thyreau, distributed under that project's terms. If you use the hippocampus-localized output, please cite:

> Thyreau, B., Sato, K., Fukuda, H., & Taki, Y. (2018). Segmentation of the hippocampus by transferring algorithmic knowledge for large cohort processing. *Medical Image Analysis*, 43, 214–228. https://doi.org/10.1016/j.media.2017.11.004
> https://www.sciencedirect.com/science/article/pii/S1361841517301597
