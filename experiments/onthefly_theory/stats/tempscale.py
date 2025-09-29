import numpy as np
import torch
import torch.nn.functional as F
from . import tempscale as _self  # to allow reuse if split

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"))

def fit_temperature(logits_val: np.ndarray, y_val: np.ndarray) -> float:
    T = torch.tensor([1.0], requires_grad=True, device=device)
    y = torch.tensor(y_val, dtype=torch.long, device=device)
    L = torch.tensor(logits_val, dtype=torch.float32, device=device)
    opt = torch.optim.LBFGS([T], lr=0.1, max_iter=50, line_search_fn="strong_wolfe")
    def _closure():
        opt.zero_grad()
        scaled = L / T.clamp_min(1e-6)
        loss = F.cross_entropy(scaled, y)
        loss.backward()
        return loss
    opt.step(_closure)
    return float(T.detach().cpu().item())

def apply_temperature(logits_np: np.ndarray, T: float) -> np.ndarray:
    return logits_np / max(T, 1e-6)

def fit_temperature_logprobs(logp_val: np.ndarray, y_val: np.ndarray) -> float:
    P = np.exp(np.asarray(logp_val))
    P = np.clip(P, 1e-12, 1.0)
    y = torch.tensor(y_val, dtype=torch.long, device=device)
    P_t = torch.tensor(P, dtype=torch.float32, device=device)
    T = torch.tensor([1.0], requires_grad=True, device=device)
    opt = torch.optim.LBFGS([T], lr=0.1, max_iter=50, line_search_fn="strong_wolfe")
    def _closure():
        opt.zero_grad()
        invT = 1.0 / T.clamp_min(1e-6)
        logpT = (invT * torch.log(P_t) - torch.logsumexp(invT * torch.log(P_t), dim=1, keepdim=True))
        nll = -torch.gather(logpT, 1, y.view(-1,1)).mean()
        nll.backward()
        return nll
    opt.step(_closure)
    return float(T.detach().cpu().item())

def apply_temperature_logprobs(logp_np: np.ndarray, T: float) -> np.ndarray:
    P = np.exp(np.asarray(logp_np))
    P = np.clip(P, 1e-12, 1.0)
    invT = 1.0 / max(T, 1e-6)
    logpT = invT * np.log(P) - np.log(np.sum(P**invT, axis=1, keepdims=True))
    return logpT
