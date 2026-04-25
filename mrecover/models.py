"""Model loading utilities for MRecover."""

import os
import torch
from pathlib import Path
from tqdm import tqdm
from monai.networks.nets import DiffusionModelUNet

REPO_ID = "jil202/MRcover"
MODEL_CACHE_DIR = Path.home() / ".cache" / "mrecover"

MODELS = {
    "flowmatching": "flowmatching_best.pt",
    "baseline":     "baseline_best.pt",
    "pix2pix":      "pix2pix_G_best.pt",
}


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_model(model_type="flowmatching", cache_dir=MODEL_CACHE_DIR):
    """Download a model from HuggingFace to cache directory.

    Args:
        model_type: one of 'flowmatching', 'baseline', 'pix2pix'

    Returns:
        str: Path to downloaded model checkpoint
    """
    from huggingface_hub import hf_hub_download

    if model_type not in MODELS:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose from: {list(MODELS)}")

    filename = MODELS[model_type]
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    model_path = cache_dir / filename
    if model_path.exists():
        print(f"Model already cached at: {model_path}")
        return str(model_path)

    token = os.environ.get("HF_TOKEN") or True

    print(f"Downloading {model_type} model from HuggingFace ({REPO_ID}/{filename})...")
    try:
        downloaded_path = hf_hub_download(repo_id=REPO_ID, filename=filename, token=token)
        import shutil
        shutil.copy(downloaded_path, model_path)
        print(f"Model downloaded to: {model_path}")
        return str(model_path)
    except Exception as e:
        print(f"\nError: Could not download model from HuggingFace")
        print(f"Reason: {e}")
        print("\nTo fix:")
        print(f"1. Accept the model agreement at https://huggingface.co/{REPO_ID}")
        print("2. Set your HuggingFace token: export HF_TOKEN=<your_token>")
        print("   (or login via CLI: huggingface-cli login)")
        raise


def load_model(model_path=None, model_type="flowmatching", device="cuda", fp16=False, compile_model=False):
    """Load a T1w→T2 TSE translation model.

    Args:
        model_path: Path to checkpoint. If None, auto-downloads.
        model_type: one of 'flowmatching', 'baseline', 'pix2pix'
        device: 'cuda' or 'cpu'
        fp16: Use half precision (faster on GPU, requires CUDA)
        compile_model: Apply torch.compile for faster inference

    Returns:
        Loaded and ready model (nn.Module)
    """
    if model_path is None:
        model_path = download_model(model_type=model_type)
    elif not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading T2 TSE model from {model_path}")

    model = DiffusionModelUNet(
        spatial_dims=2,
        in_channels=3,
        out_channels=1,
        channels=(256, 256, 512),
        attention_levels=(False, False, True),
        num_res_blocks=2,
        num_head_channels=512,
        with_conditioning=False,
    )

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    fixed_state_dict = {
        k.replace("module.", "").replace("_orig_mod.", ""): v
        for k, v in state_dict.items()
    }
    model.load_state_dict(fixed_state_dict)
    model.to(device)
    model.eval()

    if fp16:
        model = model.bfloat16()

    if compile_model:
        print("Compiling model for faster inference...")
        model = torch.compile(model, backend="inductor")

    print("Model loaded successfully!")
    return model
