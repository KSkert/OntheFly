import torch.nn as nn
import torch.nn.functional as F

class CNN1DClassifier(nn.Module):
    def __init__(self, n_features, n_classes, channels=32, kernel=3, hidden=128, dropout=0.2):
        super().__init__()
        pad = (kernel - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=kernel, padding=pad), nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=kernel, padding=pad), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_classes)
        )
        self.n_features = n_features
    def forward(self, x):
        x1 = x.view(x.size(0), 1, self.n_features)
        h = self.conv(x1)
        return self.head(h)
