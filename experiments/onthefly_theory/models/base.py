import torch
import torch.nn as nn

class ArrayDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, task="reg"):
        self.X = torch.tensor(X, dtype=torch.float32)
        if task == "reg":
            self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        else:
            self.y = torch.tensor(y, dtype=torch.long)
        self.task = task
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class MLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
    def forward(self, x): return self.net(x)

def make_equal_params_mlp(input_dim: int, target_params: int, dropout: float=0.2):
    lo, hi = 8, 4096; best_H = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        tmp = MLPRegressor(input_dim, hidden_dim=mid, dropout=dropout)
        p = count_params(tmp)
        if p <= target_params:
            best_H = mid; lo = mid + 1
        else:
            hi = mid - 1
    return MLPRegressor(input_dim, hidden_dim=best_H, dropout=dropout)
