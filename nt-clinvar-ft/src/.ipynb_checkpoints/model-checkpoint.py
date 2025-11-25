from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model


def print_trainable_param_stats(model: nn.Module, prefix: str = ""):
    """
    Print how many parameters are trainable vs total (and %).

    If this is a PEFT/LoRA model that implements `print_trainable_parameters`,
    we call that. Otherwise we manually count.
    """
    # PEFT models often expose a helper already
    if hasattr(model, "print_trainable_parameters"):
        print(prefix + "[LoRA] Using PEFT print_trainable_parameters():")
        model.print_trainable_parameters()
        return

    total = 0
    trainable = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n

    pct = 100.0 * trainable / max(total, 1)
    print(
        prefix
        + f"[LoRA] Trainable params: {trainable:,} / {total:,} "
        + f"({pct:.2f}% trainable)"
    )

def build_tokenizer(cfg):
    """Build NT tokenizer with correct k-mer setting."""
    return AutoTokenizer.from_pretrained(
        cfg["model"]["name"],
        trust_remote_code=True,
        kmer=cfg["data"]["kmer"],
    )


class SiameseNTClassifier(nn.Module):
    """Siamese Nucleotide Transformer classifier with flexible embeddings.

    embed_method:
      - "mean": masked mean over all non-padding tokens
      - "cls":  CLS token embedding
      - "central": middle non-CLS token (derived from attention_mask)
    """

    def __init__(
        self,
        base_model_name: str,
        num_labels: int,
        embed_method: str = "mean",
        class_weights: Optional[np.ndarray] = None,
    ):
        super().__init__()
        assert embed_method in ["mean", "cls", "central"]
        self.embed_method = embed_method

        self.encoder = AutoModel.from_pretrained(
            base_model_name,
            trust_remote_code=True,
        )
        hidden = (
            self.encoder.config.hidden_size
            if hasattr(self.encoder, "config")
            else 768
        )
        fuse_dim = hidden * 4  # ref, alt, diff, prod

        self.classifier = nn.Sequential(
            nn.Linear(fuse_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, num_labels),
        )

        if class_weights is not None:
            self.register_buffer(
                "class_weights",
                torch.tensor(class_weights, dtype=torch.float32),
            )
        else:
            self.class_weights = None

    # ---- pooling methods ----

    @staticmethod
    def mean_pool(last_hidden_state, attention_mask):
        """Masked mean pooling over token dimension."""
        mask = attention_mask.unsqueeze(-1)  # (B, T, 1)
        masked = last_hidden_state * mask
        summed = masked.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-6)
        return summed / counts

    @staticmethod
    def cls_pool(last_hidden_state):
        """Take CLS token embedding (position 0)."""
        return last_hidden_state[:, 0, :]

    @staticmethod
    def central_pool(last_hidden_state, attention_mask):
        """Take embedding of the middle non-CLS token.

        Uses attention_mask to be robust to:
          - non-overlapping 6-mers
          - leftover single-nucleotide tokens
          - any variable effective sequence length.
        """
        # lengths including CLS
        lengths = attention_mask.sum(dim=1)  # (B,)
        # number of non-CLS tokens (at least 1)
        non_cls = (lengths - 1).clamp(min=1)
        # central index within the non-CLS chunk
        central_noncls = non_cls // 2  # integer division
        # convert to token index in full sequence (add 1 for CLS)
        central_idx = central_noncls + 1  # (B,)

        batch_size, _, hidden = last_hidden_state.shape
        # gather expects shape (B, 1, H)
        idx = central_idx.view(batch_size, 1, 1).expand(-1, 1, hidden)
        central_tokens = torch.gather(last_hidden_state, 1, idx).squeeze(1)
        return central_tokens

    def forward(
        self,
        ref_input_ids=None,
        ref_attention_mask=None,
        alt_input_ids=None,
        alt_attention_mask=None,
        labels=None,
        **kwargs,
    ):
        # Encode reference and alternate sequences with shared encoder (siamese)
        ref_out = self.encoder(
            input_ids=ref_input_ids,
            attention_mask=ref_attention_mask,
        )
        alt_out = self.encoder(
            input_ids=alt_input_ids,
            attention_mask=alt_attention_mask,
        )

        ref_last = ref_out.last_hidden_state
        alt_last = alt_out.last_hidden_state

        # Choose embedding strategy
        if self.embed_method == "mean":
            ref_repr = self.mean_pool(ref_last, ref_attention_mask)
            alt_repr = self.mean_pool(alt_last, alt_attention_mask)
        elif self.embed_method == "cls":
            ref_repr = self.cls_pool(ref_last)
            alt_repr = self.cls_pool(alt_last)
        else:  # "central"
            ref_repr = self.central_pool(ref_last, ref_attention_mask)
            alt_repr = self.central_pool(alt_last, alt_attention_mask)

        # Fuse embeddings
        diff = alt_repr - ref_repr
        prod = alt_repr * ref_repr
        feat = torch.cat([ref_repr, alt_repr, diff, prod], dim=-1)

        logits = self.classifier(feat)

        loss = None
        if labels is not None:
            if self.class_weights is not None:
                criterion = nn.CrossEntropyLoss(weight=self.class_weights)
            else:
                criterion = nn.CrossEntropyLoss()
            loss = criterion(logits, labels)

        return {"loss": loss, "logits": logits}



def apply_lora_if_enabled(cfg, model: SiameseNTClassifier) -> SiameseNTClassifier:
    """Wrap encoder in LoRA adapters if enabled in config and print stats."""
    lora_cfg = cfg["lora"]

    # If LoRA is disabled, just report that all encoder params are trainable
    if not lora_cfg.get("enabled", False):
        print("[LoRA] disabled in config; all base encoder params are trainable.")
        print_trainable_param_stats(model.encoder, prefix="[LoRA] ")
        return model

    print(
        "[LoRA] enabled with "
        f"r={lora_cfg['r']}, alpha={lora_cfg['alpha']}, "
        f"dropout={lora_cfg['dropout']}"
    )
    print("[LoRA] Target modules:", ", ".join(lora_cfg["target_modules"]))

    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        bias="none",
        target_modules=lora_cfg["target_modules"],
    )
    model.encoder = get_peft_model(model.encoder, peft_config)

    # After wrapping, print how many params are actually trainable
    print_trainable_param_stats(model.encoder, prefix="[LoRA] ")
    return model

