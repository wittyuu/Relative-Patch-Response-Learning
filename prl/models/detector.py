import torch
import torch.nn as nn

from .lora import apply_lora, lora_parameters

LORA_TARGETS = (
    "attention.q_proj",
    "attention.k_proj",
    "attention.v_proj",
    "attention.o_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


class DetectionHead(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_v = nn.Linear(dim, hidden_dim)
        self.gate_u = nn.Linear(dim, hidden_dim)
        self.gate_w = nn.Linear(hidden_dim, 1)
        self.fc = nn.Linear(dim, 1)

    def forward(self, tokens):
        gated = torch.tanh(self.gate_v(tokens)) * torch.sigmoid(self.gate_u(tokens))
        weights = torch.softmax(self.gate_w(gated).squeeze(-1), dim=1)
        pooled = (weights.unsqueeze(-1) * tokens).sum(dim=1)
        return self.fc(pooled).squeeze(-1)


class PRLDetector(nn.Module):
    def __init__(self, dinov3, lora_rank=8, lora_alpha=1.0, head_hidden_dim=256):
        super().__init__()
        from transformers import AutoModel

        self.backbone = AutoModel.from_pretrained(dinov3)
        dim = self.backbone.config.hidden_size
        self.num_prefix_tokens = 1 + self.backbone.config.num_register_tokens
        self.patch_head = nn.Linear(dim, 1, bias=False)
        self.detection_head = DetectionHead(dim, head_hidden_dim)
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        targets = list(LORA_TARGETS)
        if getattr(self.backbone.config, "use_gated_mlp", False):
            targets.append("mlp.gate_proj")
        apply_lora(self.backbone, targets, lora_rank, lora_alpha)

    def trainable_parameters(self):
        head = self.detection_head
        params = lora_parameters(self.backbone) + list(self.patch_head.parameters())
        for layer in (head.fc, head.gate_v, head.gate_u, head.gate_w):
            params.extend(layer.parameters())
        return params

    def trainable_state_dict(self):
        trainable = {id(parameter) for parameter in self.trainable_parameters()}
        return {
            name: parameter.detach()
            for name, parameter in self.named_parameters()
            if id(parameter) in trainable
        }

    def load_trainable_weights(self, path):
        from safetensors.torch import load_file

        weights = load_file(path)
        expected = set(self.trainable_state_dict())
        if set(weights) != expected:
            missing = sorted(expected - set(weights))[:5]
            unexpected = sorted(set(weights) - expected)[:5]
            raise ValueError(
                f"Weights do not match the model: missing {missing}, "
                f"unexpected {unexpected}"
            )
        self.load_state_dict(weights, strict=False)

    def _embed_patches(self, images):
        conv = self.backbone.embeddings.patch_embeddings
        patches = conv(images.to(dtype=conv.weight.dtype))
        return patches.flatten(2).transpose(1, 2)

    def _encode(self, patches, rope_images):
        embeddings = self.backbone.embeddings
        batch = patches.shape[0]
        cls_token = embeddings.cls_token.expand(batch, -1, -1).to(
            dtype=patches.dtype, device=patches.device
        )
        registers = embeddings.register_tokens.expand(batch, -1, -1).to(
            dtype=patches.dtype, device=patches.device
        )
        hidden = torch.cat([cls_token, registers, patches], dim=1)
        position = self.backbone.rope_embeddings(
            rope_images.to(device=hidden.device, dtype=hidden.dtype)
        )
        for layer in self.backbone.layer:
            hidden = layer(hidden, attention_mask=None, position_embeddings=position)
            if isinstance(hidden, (tuple, list)):
                hidden = hidden[0]
        hidden = self.backbone.norm(hidden)
        return hidden[:, self.num_prefix_tokens :, :]

    def forward(self, images):
        tokens = self._encode(self._embed_patches(images), rope_images=images)
        patch_scores = self.patch_head(tokens).squeeze(-1)
        logits = self.detection_head(tokens)
        return logits, patch_scores
