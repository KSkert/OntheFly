import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm.auto import tqdm
from .loops import device

def train_gate_reg(gate, experts, train_loader, val_loader, hp, criterion, max_epochs: int):
    for ex in experts:
        ex.eval()
        for p in ex.parameters():
            p.requires_grad_(False)
    optg = optim.Adam(gate.parameters(), lr=hp.lr_gate, weight_decay=hp.weight_decay)
    best_val = float("inf"); best_state = None; patience = 0
    pbar = tqdm(range(1, max_epochs + 1), desc="gate(reg)", leave=False)
    for ep in pbar:
        gate.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            with torch.no_grad():
                ex_outs = [ex(Xb) for ex in experts]
            w = gate(Xb)
            out = sum(w[:, [j]] * ex_outs[j] for j in range(len(experts)))
            loss = criterion(out, yb)
            optg.zero_grad(); loss.backward()
            if hp.grad_clip: nn.utils.clip_grad_norm_(gate.parameters(), hp.grad_clip)
            optg.step()
        gate.eval(); val_loss = 0.0; n = 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                ex_outs = [ex(Xb) for ex in experts]
                w = gate(Xb)
                out = sum(w[:, [j]] * ex_outs[j] for j in range(len(experts)))
                bs = yb.size(0); val_loss += criterion(out, yb).item() * bs; n += bs
        val_loss /= max(n, 1)
        pbar.set_postfix(val_mse=f"{val_loss:.6f}", patience=f"{patience}/{hp.gate_patience}")
        if val_loss < best_val - 1e-8:
            best_val = val_loss; best_state = {k: v.detach().cpu().clone() for k, v in gate.state_dict().items()}; patience = 0
        else:
            patience += 1
            if patience > hp.gate_patience:
                pbar.close(); break
    if best_state is not None:
        gate.load_state_dict(best_state)
    return gate

def train_gate_cls(gate, experts, train_loader, val_loader, hp, n_classes: int, max_epochs: int):
    for ex in experts:
        ex.eval()
        for p in ex.parameters():
            p.requires_grad_(False)
    optg = optim.Adam(gate.parameters(), lr=hp.lr_gate, weight_decay=hp.weight_decay)
    crit = nn.NLLLoss()
    best_val = float("inf"); best_state = None; patience = 0

    def mixed_logprobs(xb):
        w = gate(xb)
        logps = [F.log_softmax(ex(xb), dim=1) for ex in experts]
        stack = torch.stack(logps, dim=2)           # (B,C,E)
        m = torch.amax(stack, dim=2, keepdim=True)  # (B,C,1)
        mix = (w.unsqueeze(1) * torch.exp(stack - m)).sum(dim=2)
        return (m.squeeze(2) + torch.log(mix.clamp_min(1e-12)))

    pbar = tqdm(range(1, max_epochs + 1), desc="gate(cls)", leave=False)
    for ep in pbar:
        gate.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            logp = mixed_logprobs(Xb)
            loss = crit(logp, yb)
            optg.zero_grad(); loss.backward()
            if hp.grad_clip: nn.utils.clip_grad_norm_(gate.parameters(), hp.grad_clip)
            optg.step()

        gate.eval(); val_loss = 0.0; n = 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                logp = mixed_logprobs(Xb)
                bs = yb.size(0); val_loss += crit(logp, yb).item() * bs; n += bs
        val_loss /= max(n, 1)
        pbar.set_postfix(val_nll=f"{val_loss:.5f}", patience=f"{patience}/{hp.gate_patience}")
        if val_loss < best_val - 1e-8:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in gate.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience > hp.gate_patience: pbar.close(); break
    if best_state is not None:
        gate.load_state_dict(best_state)
    return gate
