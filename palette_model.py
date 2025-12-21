import colorsys
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# =====================================================
# 0. GoEmotions label config (28 labels)
# =====================================================

GOEMOTIONS_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral",
]

NUM_LABELS = len(GOEMOTIONS_LABELS)


# =====================================================
# 0b. GoEmotions -> hue mapping (warm = positive, cold = negative)
# =====================================================

# Heuristic valence scores in [-1, 1]; tweak as desired.
GOEMOTIONS_VALENCE = {
    "admiration": 0.7,
    "amusement": 0.7,
    "anger": -0.8,
    "annoyance": -0.6,
    "approval": 0.6,
    "caring": 0.5,
    "confusion": -0.2,
    "curiosity": 0.2,
    "desire": 0.3,
    "disappointment": -0.7,
    "disapproval": -0.6,
    "disgust": -0.8,
    "embarrassment": -0.5,
    "excitement": 0.8,
    "fear": -0.9,
    "gratitude": 0.8,
    "grief": -0.9,
    "joy": 1.0,
    "love": 1.0,
    "nervousness": -0.5,
    "optimism": 0.8,
    "pride": 0.6,
    "realization": 0.1,
    "relief": 0.5,
    "remorse": -0.7,
    "sadness": -0.8,
    "surprise": 0.0,
    "neutral": 0.0,
}

# Hue anchor points (degrees), standardized for coordinates: warm/orange-ish for positive, cold/blue-ish for negative.
POS_HUE_DEG = 40.0
NEG_HUE_DEG = 220.0


def _valence_to_hue_deg(valence: float) -> float:
    """
    Map valence [-1,1] to a hue on a cold<->warm line.
    valence=1 -> POS_HUE_DEG, valence=-1 -> NEG_HUE_DEG.
    """
    v = max(-1.0, min(1.0, valence))
    t = (1.0 - v) / 2.0  # 0 for warmest, 1 for coldest
    return POS_HUE_DEG + (NEG_HUE_DEG - POS_HUE_DEG) * t


GOEMOTIONS_LABEL_TO_HUE_DEG = {
    label: _valence_to_hue_deg(GOEMOTIONS_VALENCE[label]) for label in GOEMOTIONS_LABELS
}
GOEMOTIONS_LABEL_HUE_TENSOR = torch.tensor(
    [GOEMOTIONS_LABEL_TO_HUE_DEG[l] for l in GOEMOTIONS_LABELS],
    dtype=torch.float,
)


def color_vectors_to_hues_deg(color_vectors: torch.Tensor) -> torch.Tensor:
    """
    Convert color vectors [*, 4] (sin h, cos h, sat, val) to hue degrees in [0, 360).
    """
    sin_h = color_vectors[..., 0]
    cos_h = color_vectors[..., 1]
    hue_rad = torch.atan2(sin_h, cos_h)
    hue_deg = torch.remainder(torch.rad2deg(hue_rad) + 360.0, 360.0)
    return hue_deg


def hsv_to_rgb_list(h_deg: float, s: float, v: float):
    """
    Convert HSV (degrees, 0-1, 0-1) to RGB list in [0,1] using colorsys.
    """
    h_norm = (h_deg % 360.0) / 360.0
    r, g, b = colorsys.hsv_to_rgb(h_norm, float(s), float(v))
    return [r, g, b]


def nearest_emotion_from_hue(h_deg: torch.Tensor):
    """
    Find nearest emotion label to a hue (deg tensor scalar).
    Returns (label, angular_distance_deg, confidence[0-1]).
    Confidence is 1 at 0° difference, 0 at 180° difference (linear).
    """
    diff = torch.remainder(h_deg - GOEMOTIONS_LABEL_HUE_TENSOR + 180.0, 360.0) - 180.0
    dist = diff.abs()
    idx = dist.argmin().item()
    ang = dist[idx].item()
    conf = max(0.0, 1.0 - ang / 180.0)
    return GOEMOTIONS_LABELS[idx], ang, conf


def decode_seq_color(color_vec: torch.Tensor, seq_logits: torch.Tensor):
    """
    Decode a single sequence color vector and logits into readable values.
    """
    hue_deg = color_vectors_to_hues_deg(color_vec)[...].item()
    sat = torch.clamp(color_vec[2], 0.0, 1.0).item()
    val = torch.clamp(color_vec[3], 0.0, 1.0).item()
    rgb = hsv_to_rgb_list(hue_deg, sat, val)
    nearest_label, hue_delta, hue_conf = nearest_emotion_from_hue(torch.tensor(hue_deg))

    probs = torch.sigmoid(seq_logits)
    topk = torch.topk(probs, k=min(3, probs.numel()))
    top_labels = [
        (GOEMOTIONS_LABELS[i], probs[i].item()) for i in topk.indices.tolist()
    ]

    return {
        "hue_deg": hue_deg,
        "sat": sat,
        "val": val,
        "rgb": rgb,
        "nearest_label": nearest_label,
        "hue_delta": hue_delta,
        "hue_confidence": hue_conf,
        "top_seq_labels": top_labels,
    }


def decode_token_colors(token_colors: torch.Tensor, tokens, attention_mask=None):
    """
    Decode per-token color vectors into hue/RGB and nearest label.
    token_colors: [L, 4] tensor.
    tokens: list of token strings (same length L).
    attention_mask: optional [L] mask to skip paddings.
    """
    hues = color_vectors_to_hues_deg(token_colors)
    results = []
    for i, (tok, hue, col) in enumerate(zip(tokens, hues, token_colors)):
        if attention_mask is not None and attention_mask[i].item() == 0:
            continue
        sat = torch.clamp(col[2], 0.0, 1.0).item()
        val = torch.clamp(col[3], 0.0, 1.0).item()
        rgb = hsv_to_rgb_list(float(hue.item()), sat, val)
        label, delta, conf = nearest_emotion_from_hue(hue)
        results.append(
            {
                "token": tok,
                "hue_deg": float(hue.item()),
                "sat": sat,
                "val": val,
                "rgb": rgb,
                "nearest_label": label,
                "hue_delta": float(delta),
                "hue_confidence": conf,
            }
        )
    return results


def run_palette_inference(text: str, model, tokenizer, device):
    """
    Convenience inference: returns decoded sequence and token-level palette info.
    """
    model.eval()
    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)

    seq_decoded = decode_seq_color(
        out["seq_color"][0].cpu(), out["seq_logits"][0].cpu()
    )
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    token_decoded = decode_token_colors(
        out["token_colors"][0].cpu(), tokens, attention_mask=attention_mask[0]
    )

    return {
        "text": text,
        "seq_color": seq_decoded,
        "tokens": token_decoded,
    }


# =====================================================
# 1. Gated Attention Pooling
# =====================================================

class GatedAttentionPooling(nn.Module):
    """
    H: [B, L, hidden]
    attention_mask: [B, L] (1 = real token, 0 = padding)
    """

    def __init__(self, hidden_size: int, attn_size: int = 256):
        super().__init__()
        self.W_att = nn.Linear(hidden_size, attn_size)
        self.v_att = nn.Linear(attn_size, 1, bias=False)
        self.W_gate = nn.Linear(hidden_size, 1)

    def forward(self, H, attention_mask):
        U = torch.tanh(self.W_att(H))            # [B, L, attn_size]
        raw_scores = self.v_att(U).squeeze(-1)   # [B, L]

        gates = torch.sigmoid(self.W_gate(H)).squeeze(-1)  # [B, L]
        gated_scores = raw_scores * gates                  # [B, L]

        mask = attention_mask == 0
        gated_scores = gated_scores.masked_fill(mask, -1e9)

        attn_weights = torch.softmax(gated_scores, dim=-1)  # [B, L]
        z = torch.bmm(attn_weights.unsqueeze(1), H).squeeze(1)  # [B, hidden]

        return z, attn_weights, gates


# =====================================================
# 2. LoRA-wrapped LLaMA + gated attention head
# =====================================================

class LLaMAGoEmotionsPEFT(nn.Module):
    """
    LLaMA backbone (LoRA enabled) + gated attention pooling + 28-dim GoEmotions head
    """

    def __init__(
        self,
        llama_model,         # LoRA wrapped backbone
        hidden_size: int,
        num_labels: int = NUM_LABELS,
        attn_size: int = 256,
        color_loss_weight: float = 0.0,
    ):
        super().__init__()

        self.llm = llama_model
        self.gated_pool = GatedAttentionPooling(hidden_size, attn_size)

        self.seq_head = nn.Linear(hidden_size, num_labels)
        self.token_head = nn.Linear(hidden_size, num_labels)

        self.loss_seq = nn.BCEWithLogitsLoss()
        self.loss_tok = nn.BCEWithLogitsLoss(reduction="none")

        self.num_labels = num_labels
        self.color_loss_weight = color_loss_weight

        # Precompute label hue unit vectors (sin, cos) for mixing hues.
        hues_deg = torch.tensor(
            [GOEMOTIONS_LABEL_TO_HUE_DEG[l] for l in GOEMOTIONS_LABELS],
            dtype=torch.float,
        )
        hues_rad = torch.deg2rad(hues_deg)
        label_hue_unit = torch.stack([torch.sin(hues_rad), torch.cos(hues_rad)], dim=-1)
        self.register_buffer("label_hue_unit", label_hue_unit)  # [num_labels, 2]

    def forward(
        self,
        input_ids,
        attention_mask,
        seq_targets=None,
        token_targets=None,
    ):
        outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
        H = outputs.last_hidden_state  # [B, L, hidden]

        token_logits = self.token_head(H)
        z, attn, gates = self.gated_pool(H, attention_mask)
        seq_logits = self.seq_head(z)

        # --- Color projection ---
        # Token-level probabilities over emotions.
        token_probs = torch.sigmoid(token_logits)  # [B, L, num_labels]

        # Normalize for hue mixing to keep angles on the unit circle.
        prob_mass = token_probs.sum(dim=-1, keepdim=True) + 1e-8  # [B, L, 1]
        mixed_unit = torch.matmul(
            token_probs / prob_mass, self.label_hue_unit
        )  # [B, L, 2] (sin, cos)

        # Saturation/value as simple intensity summaries.
        sat = token_probs.mean(dim=-1)          # [B, L]
        val = token_probs.max(dim=-1).values    # [B, L]

        token_colors = torch.stack(
            [mixed_unit[..., 0], mixed_unit[..., 1], sat, val], dim=-1
        )  # [B, L, 4] -> (sin h, cos h, sat, val)

        # Fuse colors with gated attention weights.
        fused_color = torch.bmm(attn.unsqueeze(1), token_colors).squeeze(1)  # [B, 4]

        loss = None
        if seq_targets is not None or token_targets is not None:
            losses = []

            if seq_targets is not None:
                seq_loss = self.loss_seq(seq_logits, seq_targets)
                losses.append(seq_loss)

            if token_targets is not None:
                raw_tok_loss = self.loss_tok(token_logits, token_targets)
                mask = attention_mask.unsqueeze(-1).float()
                tok_loss_masked = raw_tok_loss * mask
                denom = mask.sum() * self.num_labels + 1e-8
                tok_loss = tok_loss_masked.sum() / denom
                losses.append(tok_loss)

            loss = sum(losses)

        # Optional color-space alignment loss: align fused hue to label hue.
        if self.color_loss_weight > 0 and seq_targets is not None:
            target_mass = seq_targets.sum(dim=-1, keepdim=True) + 1e-8  # [B, 1]
            target_unit = torch.matmul(
                seq_targets / target_mass, self.label_hue_unit
            )  # [B, 2]

            fused_unit = fused_color[:, :2]  # [B, 2]

            # Cosine similarity on unit vectors; safeguard norms.
            denom = (
                fused_unit.norm(dim=-1) * target_unit.norm(dim=-1) + 1e-8
            )  # [B]
            cos_sim = (fused_unit * target_unit).sum(dim=-1) / denom
            color_loss = (1.0 - cos_sim).mean()

            if loss is None:
                loss = self.color_loss_weight * color_loss
            else:
                loss = loss + self.color_loss_weight * color_loss

        return {
            "loss": loss,
            "seq_logits": seq_logits,
            "token_colors": token_colors,
            "seq_color": fused_color,
            "attn_weights": attn,
            "token_probs": token_probs,
        }


# =====================================================
# 3. Dataset wrapper for SetFit/go_emotions
# =====================================================

class GoEmotionsDataset(Dataset):
    """
    Converts the HF dataset rows into:
        input_ids, attention_mask, label vector
    """

    def __init__(self, split, tokenizer, max_length=128):
        self.data = split
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        text = row["text"]
        labels = [float(row[label]) for label in GOEMOTIONS_LABELS]

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(labels, dtype=torch.float),
        }


def goemotions_collate_fn(batch):
    ids = torch.stack([b["input_ids"] for b in batch])
    mask = torch.stack([b["attention_mask"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}


# =====================================================
# 4. Build LoRA model
# =====================================================

def build_lora_llama(
    base_model_name: str,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
):
    backbone = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        trust_remote_code=True,
    )

    backbone = prepare_model_for_kbit_training(backbone)

    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    peft_model = get_peft_model(backbone, config)
    hidden_size = peft_model.config.hidden_size

    return peft_model, hidden_size


# =====================================================
# 5. Training function (for SLURM call)
# =====================================================

def train_goemotions_lora(
    base_model_name: str,
    batch_size: int,
    lr: float,
    max_length: int,
    epochs: int,
    device: torch.device,
    color_loss_weight: float = 0.1,
    log_jsonl_path: str = None,
    log_every: int = 200,
):

    ds = load_dataset("SetFit/go_emotions")

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        use_fast=False,
        padding_side="left",
    )

    train_set = GoEmotionsDataset(ds["train"], tokenizer, max_length)
    val_set = GoEmotionsDataset(ds["validation"], tokenizer, max_length)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=goemotions_collate_fn,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=goemotions_collate_fn,
    )

    peft_llama, hidden_size = build_lora_llama(base_model_name)
    peft_llama.to(device)

    model = LLaMAGoEmotionsPEFT(
        llama_model=peft_llama,
        hidden_size=hidden_size,
        num_labels=NUM_LABELS,
        attn_size=256,
        color_loss_weight=color_loss_weight,
    ).to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
    )

    for epoch in range(epochs):

        # ---- TRAIN ----
        model.train()
        for step, batch in enumerate(train_loader):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            out = model(
                input_ids=ids,
                attention_mask=mask,
                seq_targets=labels,
                token_targets=None,
            )

            loss = out["loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Optional logging for research/inspection.
            if log_jsonl_path and (step % log_every == 0):
                sample_idx = 0
                text = tokenizer.decode(
                    ids[sample_idx].tolist(), skip_special_tokens=True
                )
                seq_color_cpu = out["seq_color"][sample_idx].detach().cpu()
                seq_logits_cpu = out["seq_logits"][sample_idx].detach().cpu()
                decoded = decode_seq_color(seq_color_cpu, seq_logits_cpu)
                token_colors_cpu = out["token_colors"][sample_idx].detach().cpu()
                token_hues = color_vectors_to_hues_deg(token_colors_cpu).tolist()
                token_nearest = []
                token_nearest_conf = []
                for h in token_hues:
                    lbl, _, conf = nearest_emotion_from_hue(torch.tensor(h))
                    token_nearest.append(lbl)
                    token_nearest_conf.append(conf)
                record = {
                    "split": "train",
                    "epoch": epoch,
                    "step": step,
                    "text": text,
                    "seq_color": out["seq_color"][sample_idx].detach().cpu().tolist(),
                    "seq_color_decoded": decoded,
                    "seq_rgb": decoded["rgb"],
                    "seq_nearest_label": decoded["nearest_label"],
                    "seq_hue_deg": decoded["hue_deg"],
                    "seq_hue_confidence": decoded["hue_confidence"],
                    "token_colors": out["token_colors"][sample_idx].detach().cpu().tolist(),
                    "token_hues": token_hues,
                    "token_nearest_labels": token_nearest,
                    "token_nearest_conf": token_nearest_conf,
                    "seq_logits": out["seq_logits"][sample_idx].detach().cpu().tolist(),
                    "token_probs": out["token_probs"][sample_idx].detach().cpu().tolist(),
                    "top_seq_labels": decoded["top_seq_labels"],
                }
                with open(log_jsonl_path, "a") as f:
                    f.write(json.dumps(record) + "\n")

        # ---- VALIDATE ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for step, batch in enumerate(val_loader):
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                out = model(
                    input_ids=ids,
                    attention_mask=mask,
                    seq_targets=labels,
                )

                val_loss += out["loss"].item() * ids.size(0)

                if log_jsonl_path and (step % log_every == 0):
                    sample_idx = 0
                    text = tokenizer.decode(
                        ids[sample_idx].tolist(), skip_special_tokens=True
                    )
                    seq_color_cpu = out["seq_color"][sample_idx].detach().cpu()
                    seq_logits_cpu = out["seq_logits"][sample_idx].detach().cpu()
                    decoded = decode_seq_color(seq_color_cpu, seq_logits_cpu)
                    token_colors_cpu = out["token_colors"][sample_idx].detach().cpu()
                    token_hues = color_vectors_to_hues_deg(token_colors_cpu).tolist()
                    token_nearest = []
                    token_nearest_conf = []
                    for h in token_hues:
                        lbl, _, conf = nearest_emotion_from_hue(torch.tensor(h))
                        token_nearest.append(lbl)
                        token_nearest_conf.append(conf)
                    record = {
                        "split": "val",
                        "epoch": epoch,
                        "step": step,
                        "text": text,
                        "seq_color": out["seq_color"][sample_idx].detach().cpu().tolist(),
                        "seq_color_decoded": decoded,
                        "seq_rgb": decoded["rgb"],
                        "seq_nearest_label": decoded["nearest_label"],
                        "seq_hue_deg": decoded["hue_deg"],
                        "seq_hue_confidence": decoded["hue_confidence"],
                        "token_colors": out["token_colors"][sample_idx].detach().cpu().tolist(),
                        "token_hues": token_hues,
                        "token_nearest_labels": token_nearest,
                        "token_nearest_conf": token_nearest_conf,
                        "seq_logits": out["seq_logits"][sample_idx].detach().cpu().tolist(),
                        "token_probs": out["token_probs"][sample_idx].detach().cpu().tolist(),
                        "top_seq_labels": decoded["top_seq_labels"],
                    }
                    with open(log_jsonl_path, "a") as f:
                        f.write(json.dumps(record) + "\n")

        val_loss /= len(val_set)
        print(f"[Epoch {epoch+1}] validation loss = {val_loss:.4f}")

    return model
