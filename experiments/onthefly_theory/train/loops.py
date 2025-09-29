import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from ..device import get_device

device = get_device()

def train_epoch(model, loader, criterion, optimizer, grad_clip=None):
    model.train(); total = 0.0; n = 0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        pred = model(Xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad(); loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        bs = yb.size(0); total += loss.item() * bs; n += bs
    return total / max(n, 1)

@torch.no_grad()
def eval_epoch(predict_fn, loader, criterion):
    total = 0.0; n = 0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        pred = predict_fn(Xb)
        bs = yb.size(0); total += criterion(pred, yb).item() * bs; n += bs
    return total / max(n, 1)

@torch.no_grad()
def per_sample_losses(model, X_np, y_np, batch_size=2048, desc="per-loss", task="reg"):
    model.eval()
    N = len(y_np)
    losses = np.zeros(N, dtype=np.float32)
    crit = nn.MSELoss(reduction="none") if task == "reg" else nn.CrossEntropyLoss(reduction="none")
    for start in tqdm(range(0, N, batch_size), desc=desc, leave=False):
        end = min(start + batch_size, N)
        Xb = torch.tensor(X_np[start:end], dtype=torch.float32, device=device)
        if task == "reg":
            yb = torch.tensor(y_np[start:end], dtype=torch.float32, device=device).unsqueeze(1)
            pred = model(Xb)
            losses[start:end] = crit(pred, yb).squeeze(1).cpu().numpy()
        else:
            yb = torch.tensor(y_np[start:end], dtype=torch.long, device=device)
            logits = model(Xb)
            losses[start:end] = crit(logits, yb).cpu().numpy()
    return losses
