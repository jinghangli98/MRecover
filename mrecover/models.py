"""Model loading utilities for MRecover."""

import os
import torch
from pathlib import Path
from tqdm import tqdm
from monai.networks.nets import DiffusionModelUNet

MODEL_URL = "https://huggingface.co/jil202/nexus_imaging/resolve/main/T2_flow_matching_best_epoch.pt"
MODEL_FILENAME = "T2_flow_matching_best_epoch.pt"
REPO_ID = "jil202/nexus_imaging"
MODEL_CACHE_DIR = Path.home() / ".cache" / "mrecover"


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_model(cache_dir=MODEL_CACHE_DIR):
    """Download T2 TSE model from HuggingFace to cache directory.

    Returns:
        str: Path to downloaded model checkpoint
    """
    from huggingface_hub import hf_hub_download

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    model_path = cache_dir / MODEL_FILENAME
    if model_path.exists():
        print(f"Model already cached at: {model_path}")
        return str(model_path)

    print(f"Downloading T2 TSE model from HuggingFace ({REPO_ID}/{MODEL_FILENAME})...")
    try:
        downloaded_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME, token=True)
        import shutil
        shutil.copy(downloaded_path, model_path)
        print(f"Model downloaded to: {model_path}")
        return str(model_path)
    except Exception as e:
        print(f"\nError: Could not download model from HuggingFace")
        print(f"Reason: {e}")
        print("\nTo fix:")
        print("1. Login to HuggingFace: huggingface-cli login")
        print("2. Make sure you have access to jil202/nexus_imaging")
        raise


def load_model(model_path=None, device="cuda", fp16=False, compile_model=False):
    """Load the T2 TSE flow matching model.

    Args:
        model_path: Path to checkpoint. If None, auto-downloads.
        device: 'cuda' or 'cpu'
        fp16: Use half precision (faster on GPU, requires CUDA)
        compile_model: Apply torch.compile for faster inference

    Returns:
        Loaded and ready model (nn.Module)
    """
    if model_path is None:
        model_path = download_model()
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
        model = model.half()

    if compile_model:
        print("Compiling model for faster inference...")
        model = torch.compile(model, backend="inductor")

    print("Model loaded successfully!")
    return model
