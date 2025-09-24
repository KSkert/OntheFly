import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from seamless_modeldev import SeamlessSession, AutoForkRules

tfm = transforms.Compose([transforms.ToTensor()])
dataset = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
train_ds, val_ds = random_split(dataset, [55_000, 5_000], generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(28*28, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )
    def penultimate(self, x):
        x = torch.flatten(x, 1)
        x = torch.relu(self.net[1](x))
        x = torch.relu(self.net[3](x))
        return x
    def forward(self, x): return self.net(x)

model = MLP()
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
loss_fn   = nn.CrossEntropyLoss()

session = SeamlessSession(
    project="mnist-demo",
    run_name="baseline",
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    train_loader=train_loader,
    val_loader=val_loader,
    amp=True,
    grad_clip_norm=1.0,
    save_dir="./checkpoints",
)

# Optional: tweak auto-fork defaults
session.set_auto_fork_rules(AutoForkRules(loss_plateau_patience=150, loss_plateau_delta=5e-4))

# Optional: override the training step
@session.training_step
def custom_training_step(batch, state):
    x, y = batch[0].to(state["device"]), batch[1].to(state["device"])
    state["optimizer"].zero_grad(set_to_none=True)
    with state["autocast"]:
        logits = state["model"](x)
        loss = state["loss_fn"](logits, y)
    state["scaler"].scale(loss).backward()
    state["scaler"].unscale_(state["optimizer"])
    torch.nn.utils.clip_grad_norm_(state["model"].parameters(), 1.0)
    state["scaler"].step(state["optimizer"])
    state["scaler"].update()
    # Return loss + extra grad_norm so GUR proxy is nicer
    gn = sum((p.grad.detach().norm(2)**2 for p in state["model"].parameters() if p.grad is not None))
    return {"loss": loss.detach(), "grad_norm": float(gn.sqrt())}

session.serve()
