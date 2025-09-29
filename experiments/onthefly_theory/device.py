import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device() -> torch.device:
    if torch.cuda.is_available(): return torch.device("cuda")
    mps_ok = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    return torch.device("mps" if mps_ok else "cpu")
