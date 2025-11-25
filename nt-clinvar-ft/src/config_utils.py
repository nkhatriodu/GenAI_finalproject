import os
import yaml
import random
import numpy as np
import torch
from typing import Any, Dict


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML config from disk."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def ensure_dirs(cfg: Dict[str, Any]) -> None:
    """Ensure data/output directories exist."""
    os.makedirs(cfg["paths"]["processed_dir"], exist_ok=True)
    os.makedirs(cfg["paths"]["output_dir"], exist_ok=True)


def set_seed(seed: int) -> None:
    """Set all relevant RNG seeds for reproducibility."""
    import transformers

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    transformers.set_seed(seed)


def get_device_and_precision():
    """Return (device, use_fp16, use_bf16) based on GPU capability."""
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        major = props.major
        use_bf16 = major >= 8  # Ampere+ (A100/H100 etc.)
        use_fp16 = not use_bf16
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        use_bf16 = False
        use_fp16 = False
    return device, use_fp16, use_bf16
