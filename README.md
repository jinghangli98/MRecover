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
The models are hosted at https://huggingface.co/jil202/MRcover. You will need to agree to share your contact information to access the models.
The model will be automatically downloaded from HuggingFace on first use. You will need to authenticate:

```bash
huggingface-cli login
```

## Quick Start

### CLI

```bash
# Force anisotropic TSE-like spacing (0.375 × 1.5 × 0.375 mm)
mrecover -i T1.nii.gz -o T2tse.nii.gz --tse-through-plane 1.5

# Input already registered to TSE space (skip resampling)
mrecover -i T1_registered.nii.gz -o T2tse.nii.gz --tse-registered

# Higher quality with more ODE steps
mrecover -i T1.nii.gz -o T2tse.nii.gz --steps 10

# Use RK4 integration instead of Euler
mrecover -i T1.nii.gz -o T2tse.nii.gz --rk4
```

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
| `--model` | None | Path to custom model checkpoint |
| `--seed` | 42 | Random seed |

## Input / Output

**Input:** T1w MPRAGE NIfTI (`.nii` / `.nii.gz`) at any isotropic resolution (e.g. 0.55–1 mm), or a DICOM series directory.

**Output:** Synthetic T2 TSE NIfTI or DICOM. By default the output is resampled to 0.375 mm in-plane with the through-plane spacing preserved from the input.
