# Adapted from Dual-Data-Alignment
# (https://github.com/roy-ch/Dual-Data-Alignment, Apache-2.0) and modified.
import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.rank = rank
        self.alpha = alpha
        self.lora_A = nn.Parameter(torch.zeros((rank, base.in_features)))
        self.lora_B = nn.Parameter(torch.zeros((base.out_features, rank)))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        output = self.base(x)
        update = torch.einsum("...d,rd->...r", x, self.lora_A)
        update = torch.einsum("...r,or->...o", update, self.lora_B)
        return output + update * (self.alpha / self.rank)


def apply_lora(model, targets, rank, alpha):
    replaced = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear) or not any(t in name for t in targets):
            continue
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name)
        setattr(parent, child_name, LoRALinear(module, rank, alpha))
        replaced.append(name)
    missing = [t for t in targets if not any(t in name for name in replaced)]
    if missing:
        raise RuntimeError(f"LoRA targets not found in the backbone: {missing}")
    return replaced


def lora_parameters(model):
    params = []
    for module in model.modules():
        if isinstance(module, LoRALinear):
            params.extend([module.lora_A, module.lora_B])
    return params
