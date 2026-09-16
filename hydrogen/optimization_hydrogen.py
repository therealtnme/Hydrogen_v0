import math

import torch
from torch import nn


def newton_schulz(grad: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximately orthogonalizes a matrix (quintic Newton-Schulz iteration from Muon)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    x = grad.float()
    tall = x.size(-2) > x.size(-1)
    if tall:
        x = x.mT
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        gram = x @ x.mT
        x = a * x + (b * gram + c * gram @ gram) @ x
    return (x.mT if tall else x).to(grad.dtype)


class Muon(torch.optim.Optimizer):
    """
    MomentUm Orthogonalized by Newton-Schulz, for 2D hidden weight matrices. Each group may set `splits` to
    orthogonalize a fused matrix (e.g. `gate_up_proj`) as that many equal row blocks.
    """

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.0, splits=1):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, weight_decay=weight_decay, splits=splits)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            momentum = group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p.grad)
                buffer = state["momentum_buffer"]
                buffer.lerp_(p.grad, 1 - momentum)
                grad = p.grad.lerp(buffer, momentum) if group["nesterov"] else buffer
                blocks = grad.reshape(grad.size(0), -1).chunk(group["splits"], dim=0)
                update = torch.cat([newton_schulz(block, group["ns_steps"]) for block in blocks]).view_as(p)
                rows, cols = blocks[0].shape
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update, alpha=-group["lr"] * max(1.0, rows / cols) ** 0.5)
        return loss


def build_optimizers(
    model: nn.Module,
    lr: float = 3e-3,
    weight_decay: float = 0.1,
    optimizer: str = "adamw",
    muon_lr: float = 0.02,
    betas: tuple[float, float] = (0.9, 0.95),
) -> list[torch.optim.Optimizer]:
    """
    Returns the optimizers for a Hydrogen model. With `optimizer="muon"`, 2D weights inside the decoder layers use Muon and
    everything else (embeddings, norms, biases, gates, learned scales) uses AdamW. Every group records its `base_lr`
    so schedules can be applied with [`set_lr_factor`].
    """
    decay, no_decay, muon, muon_fused = [], [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        in_layers = ".layers." in f".{name}"
        if optimizer == "muon" and in_layers and p.ndim == 2 and min(p.shape) > 1:
            (muon_fused if name.endswith("gate_up_proj.weight") else muon).append(p)
        elif p.ndim >= 2:
            decay.append(p)
        else:
            no_decay.append(p)

    groups = [dict(params=decay, weight_decay=weight_decay), dict(params=no_decay, weight_decay=0.0)]
    optimizers = [torch.optim.AdamW([g for g in groups if g["params"]], lr=lr, betas=betas, eps=1e-8)]
    if muon or muon_fused:
        muon_groups = [dict(params=muon, splits=1), dict(params=muon_fused, splits=2)]
        optimizers.append(Muon([g for g in muon_groups if g["params"]], lr=muon_lr, weight_decay=weight_decay))
    for opt in optimizers:
        for group in opt.param_groups:
            group["base_lr"] = group["lr"]
    return optimizers


def lr_factor(step: int, total_steps: int, warmup_steps: int, schedule: str = "cosine", min_factor: float = 0.1) -> float:
    """Linear warmup, then `cosine` decay or `wsd` (warmup-stable-decay: flat, then linear decay over the last 20%)."""
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    if schedule == "wsd":
        decay_start = 0.8
        return 1.0 if progress < decay_start else 1.0 - (1.0 - min_factor) * (progress - decay_start) / (1 - decay_start)
    return min_factor + (1.0 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def set_lr_factor(optimizers: list[torch.optim.Optimizer], factor: float) -> None:
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["base_lr"] * factor


__all__ = ["Muon", "build_optimizers", "lr_factor", "set_lr_factor", "newton_schulz"]
