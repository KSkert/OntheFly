import torch.nn as nn

class LSTMRegressor(nn.Module):
    def __init__(self, seq_len, lstm_hidden=32, lstm_layers=2, dropout=0.2, head_hidden=128):
        super().__init__()
        self.seq_len = seq_len
        self.lstm = nn.LSTM(input_size=1, hidden_size=lstm_hidden, num_layers=lstm_layers,
                            batch_first=True, dropout=dropout if lstm_layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, head_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(head_hidden, 1)
        )
    def forward(self, x_full):
        x_seq = x_full[:, :self.seq_len].unsqueeze(-1)
        out, _ = self.lstm(x_seq)
        h_last = out[:, -1, :]
        return self.head(h_last)
