import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearBaseline(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim) * 0.02)

    def forward(self, x):
        return x @ self.weight.T

class LoRALinear(nn.Module):
    def __init__(self, in_dim, out_dim, r=4, alpha=1.0):
        super().__init__()

        # Base weight (frozen)
        self.weight = nn.Parameter(
            torch.randn(out_dim, in_dim) * 0.02,
            requires_grad=False
        )

        # LoRA matrices
        self.A = nn.Parameter(torch.randn(r, in_dim) * 0.02)
        self.B = nn.Parameter(torch.zeros(out_dim, r))

        self.scaling = alpha / r

    def forward(self, x):
        base = x @ self.weight.T
        lora = (x @ self.A.T) @ self.B.T
        return base + self.scaling * lora

layer = LoRALinear(8, 8, r=2)

for name, p in layer.named_parameters():
    print(name, p.requires_grad, p.shape)

class TinyAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.q_proj = LoRALinear(d_model, d_model, r=2)
        self.v_proj = LoRALinear(d_model, d_model, r=2)

    def forward(self, x):
        Q = self.q_proj(x)
        V = self.v_proj(x)
        return Q + V

model = TinyAttention(8)
optim = torch.optim.Adam(model.parameters(), lr=1e-3)

x = torch.randn(16, 8)
target = torch.randn(16, 8)

for step in range(200):
    optim.zero_grad()
    out = model(x)
    loss = F.mse_loss(out, target)
    loss.backward()
    optim.step()

    if step % 50 == 0:
        print(step, loss.item())

before = model.q_proj.weight.clone()

optim.zero_grad()
model(x).sum().backward()
optim.step()

after = model.q_proj.weight

print(torch.allclose(before, after))
