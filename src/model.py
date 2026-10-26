"""ConvNeXt-Tiny with a 6-class head, plus the freezing and optimiser rules for each training step."""
import timm
import torch
from timm.optim import create_optimizer_v2

from src import config as C

BACKBONE = "convnext_tiny.fb_in22k_ft_in1k"
_EARLY = ("stem", "stages.0", "stages.1")


def create_model(pretrained: bool = True) -> torch.nn.Module:
    return timm.create_model(BACKBONE, pretrained=pretrained, num_classes=len(C.CLASSES))


def set_trainable(model: torch.nn.Module, mode: str) -> None:
    """mode: 'head' = head only, 'all' = everything, 'late' = stages 2-3 and head."""
    for name, p in model.named_parameters():
        if mode == "head":
            p.requires_grad = name.startswith("head")
        elif mode == "all":
            p.requires_grad = True
        elif mode == "late":
            p.requires_grad = not name.startswith(_EARLY)
        else:
            raise ValueError(mode)


def make_optimizer(model: torch.nn.Module, mode: str, lr: float, layer_decay: float = 0.85,
                   weight_decay: float = 0.05) -> torch.optim.Optimizer:
    set_trainable(model, mode)
    if mode == "head":
        return torch.optim.AdamW(model.head.parameters(), lr=lr, weight_decay=weight_decay)
    if mode == "all":
        opt = create_optimizer_v2(model, opt="adamw", lr=lr, weight_decay=weight_decay, layer_decay=layer_decay)
        for g in opt.param_groups:
            g["lr"] = lr * g.get("lr_scale", 1.0)  # timm applies lr_scale only inside its own schedulers
        return opt
    body = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("head")]
    return torch.optim.AdamW([{"params": body, "lr": lr}, {"params": model.head.parameters(), "lr": lr * 10}],
                             weight_decay=weight_decay)


def warmup_cosine(opt: torch.optim.Optimizer, total_steps: int, warmup_steps: int):
    import math
    def f(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        return 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / max(1, total_steps - warmup_steps)))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


@torch.no_grad()
def predict(model: torch.nn.Module, loader) -> torch.Tensor:
    """Class probabilities, shape (N, 6), in loader order."""
    model.eval()
    out = []
    for x, _ in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(model(x.cuda(non_blocking=True)).float().softmax(1).cpu())
    return torch.cat(out)
