import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMGate(nn.Module):
    def __init__(self, seq_len, num_experts, lstm_hidden=32, hidden_dim=128, dropout=0.2, lstm_layers=2):
        super().__init__()
        self.seq_len = seq_len
        self.lstm = nn.LSTM(input_size=1, hidden_size=lstm_hidden, num_layers=lstm_layers,
                            batch_first=True, dropout=dropout if lstm_layers > 1 else 0.0)
        self.fc = nn.Sequential(nn.Linear(lstm_hidden, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
                                nn.Linear(hidden_dim, num_experts))
    def forward(self, x_full):
        x_seq = x_full[:, :self.seq_len].unsqueeze(-1)
        out, _ = self.lstm(x_seq)
        h = out[:, -1, :]
        logits = self.fc(h)
        return F.softmax(logits, dim=1)

class TabularGateMLP(nn.Module):
    def __init__(self, input_dim, num_experts, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_experts)
        )
    def forward(self, x):
        return F.softmax(self.fc(x), dim=1)

@torch.no_grad()
def moe_predict_reg(gate, experts, xb):
    ex_outs = [ex(xb) for ex in experts]
    w = gate(xb)
    out = sum(w[:, [j]] * ex_outs[j] for j in range(len(experts)))
    return out

@torch.no_grad()
def moe_predict_cls(gate, experts, xb):
    w = gate(xb)
    probs = [F.softmax(ex(xb), dim=1) for ex in experts]
    mix = sum(w[:, [j]] * probs[j] for j in range(len(experts)))
    return torch.log(mix.clamp_min(1e-12))
