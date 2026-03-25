"""
Emotion–color augmented GNN–BERT for GoEmotions (production_plan.md).

Phase A: bert-base-uncased produces pooled embeddings and per-label logits.
Phase B: deterministic 3D emotion vectors from COLOR_MAP + sigmoid weights,
         projected 3 -> 128 with GELU, LayerNorm on 768 + 128.
Phase C: batch graph from cosine similarity on CLS embeddings, 2-layer GCN,
         classifier + optional residual from frozen BERT head.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer


# ---------------------------------------------------------------------------
# 0. GoEmotions label order (28 labels)
# ---------------------------------------------------------------------------

GOEMOTIONS_LABELS: List[str] = [
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
]

NUM_LABELS = len(GOEMOTIONS_LABELS)
LABEL_TO_INDEX = {name: i for i, name in enumerate(GOEMOTIONS_LABELS)}


# ---------------------------------------------------------------------------
# 1. Parse COLOR_MAP.txt -> valence, hue (deg), saturation rules
# ---------------------------------------------------------------------------

_ROW_RE = re.compile(
    r"^\|\s*([a-z_]+)\s*\|\s*([-0-9.]+)\s*\|\s*([-0-9.]+)\s*\|\s*$",
    re.IGNORECASE,
)


def load_color_map(path: str) -> Dict[str, Dict[str, float]]:
    """
    Parse the markdown table in COLOR_MAP.txt into label -> {valence, hue_deg}.
    Saturation is not stored in the table: neutral -> 0, all other labels -> 1.0.
    """
    out: Dict[str, Dict[str, float]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            m = _ROW_RE.match(line)
            if not m:
                continue
            label = m.group(1).lower()
            valence = float(m.group(2))
            hue_deg = float(m.group(3))
            out[label] = {"valence": valence, "hue_deg": hue_deg}

    missing = [l for l in GOEMOTIONS_LABELS if l not in out]
    if missing:
        raise ValueError(f"COLOR_MAP missing labels: {missing}")
    return out


def _label_color_table(color_map: Dict[str, Dict[str, float]]) -> torch.Tensor:
    """
    Per-label 3D vectors [sat*cos(h), sat*sin(h), valence] with hue in radians.
    Neutral: saturation 0 (origin in hue plane). Others: saturation 1.0.
    Surprise vs neutral at same hue: valence/saturation distinguish (neutral sat=0).
    """
    rows = []
    for label in GOEMOTIONS_LABELS:
        v = color_map[label]["valence"]
        h_deg = color_map[label]["hue_deg"]
        sat = 0.0 if label == "neutral" else 1.0
        h_rad = torch.deg2rad(torch.tensor(h_deg, dtype=torch.float32))
        cos_h = torch.cos(h_rad)
        sin_h = torch.sin(h_rad)
        x = sat * cos_h
        y = sat * sin_h
        rows.append(torch.stack([x, y, torch.tensor(v, dtype=torch.float32)]))
    return torch.stack(rows, dim=0)  # [28, 3]


def default_color_map_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "COLOR_MAP.txt")


# ---------------------------------------------------------------------------
# 2. Dataset
# ---------------------------------------------------------------------------


class GoEmotionsDataset(Dataset):
    def __init__(self, split, tokenizer, max_length: int = 128):
        self.data = split
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
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


def goemotions_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


# ---------------------------------------------------------------------------
# 3. Models
# ---------------------------------------------------------------------------


class BERTOnlyBaseline(nn.Module):
    """BERT pooled -> linear head. For baseline comparison."""

    def __init__(self, bert_name: str = "bert-base-uncased", num_labels: int = NUM_LABELS):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_name)
        hidden = self.bert.config.hidden_size
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0]
        logits = self.classifier(pooled)
        return {"logits": logits, "pooled": pooled}


class EmotionColorGNNBERT(nn.Module):
    """
    BERT -> (frozen in joint phase) logits for color weights;
    weighted 3D emotion vectors -> Linear 3x128 + GELU + LayerNorm;
    LayerNorm on CLS; concat -> GCN (batch graph) -> classifier;
    optional residual from frozen BERT logits.
    """

    def __init__(
        self,
        bert_name: str = "bert-base-uncased",
        color_map_path: Optional[str] = None,
        gcn_hidden: int = 896,
        num_labels: int = NUM_LABELS,
        adj_temperature: float = 1.0,
        use_residual: bool = True,
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_name)
        hidden = self.bert.config.hidden_size
        self.bert_head = nn.Linear(hidden, num_labels)

        cmap = load_color_map(color_map_path or default_color_map_path())
        table = _label_color_table(cmap)
        self.register_buffer("label_color_vectors", table)

        self.ln_bert = nn.LayerNorm(hidden)
        self.color_proj = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
        )
        self.ln_color = nn.LayerNorm(128)

        concat_dim = hidden + 128
        self.gcn1 = nn.Linear(concat_dim, gcn_hidden)
        self.gcn2 = nn.Linear(gcn_hidden, gcn_hidden)
        self.gnn_classifier = nn.Linear(gcn_hidden, num_labels)

        self.adj_temperature = adj_temperature
        self.use_residual = use_residual
        self.residual_scale = nn.Parameter(torch.tensor(1.0))

        self._loss = nn.BCEWithLogitsLoss()

    def _batch_adjacency(self, pooled: torch.Tensor) -> torch.Tensor:
        """Row-normalized softmax similarity + self-loop (B, B)."""
        z = F.normalize(pooled, p=2, dim=-1)
        sim = torch.matmul(z, z.transpose(0, 1)) / self.adj_temperature
        adj = F.softmax(sim, dim=-1)
        b = adj.size(0)
        eye = torch.eye(b, device=adj.device, dtype=adj.dtype)
        adj = adj + eye
        adj = adj / (adj.sum(dim=-1, keepdim=True) + 1e-8)
        return adj

    def _emotion_vectors(self, logits_bert: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits_bert)
        mass = probs.sum(dim=-1, keepdim=True) + 1e-8
        w = probs / mass
        e = torch.matmul(w, self.label_color_vectors)
        return e

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        bert_only: bool = False,
    ) -> Dict[str, Any]:
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0]
        logits_bert = self.bert_head(pooled)

        if bert_only:
            loss = None
            if labels is not None:
                loss = self._loss(logits_bert, labels)
            return {
                "loss": loss,
                "logits": logits_bert,
                "pooled": pooled,
            }

        e_vec = self._emotion_vectors(logits_bert)
        c = self.color_proj(e_vec)
        x_b = self.ln_bert(pooled)
        x_c = self.ln_color(c)
        x = torch.cat([x_b, x_c], dim=-1)

        adj = self._batch_adjacency(pooled.detach())
        h = torch.matmul(adj, x)
        h = F.gelu(self.gcn1(h))
        h = torch.matmul(adj, h)
        h = F.gelu(self.gcn2(h))
        logits_gnn = self.gnn_classifier(h)

        if self.use_residual:
            logits = logits_gnn + self.residual_scale * logits_bert
        else:
            logits = logits_gnn

        loss = None
        if labels is not None:
            loss = self._loss(logits, labels)

        return {
            "loss": loss,
            "logits": logits,
            "logits_bert": logits_bert,
            "logits_gnn": logits_gnn,
            "pooled": pooled,
            "emotion_vectors": e_vec,
        }

    def set_bert_trainable(self, trainable: bool) -> None:
        for p in self.bert.parameters():
            p.requires_grad = trainable
        for p in self.bert_head.parameters():
            p.requires_grad = trainable


def freeze_gnn_for_warmup(model: EmotionColorGNNBERT) -> None:
    """Train only BERT + bert_head; hold color/GNN parameters fixed."""
    model.set_bert_trainable(True)
    for p in model.color_proj.parameters():
        p.requires_grad = False
    for p in model.ln_bert.parameters():
        p.requires_grad = False
    for p in model.ln_color.parameters():
        p.requires_grad = False
    for p in model.gcn1.parameters():
        p.requires_grad = False
    for p in model.gcn2.parameters():
        p.requires_grad = False
    for p in model.gnn_classifier.parameters():
        p.requires_grad = False
    model.residual_scale.requires_grad = False


def freeze_for_joint(model: EmotionColorGNNBERT) -> None:
    """Freeze BERT + BERT head; train color projection, norms, GCN, classifier."""
    model.set_bert_trainable(False)
    for p in model.color_proj.parameters():
        p.requires_grad = True
    for p in model.ln_bert.parameters():
        p.requires_grad = True
    for p in model.ln_color.parameters():
        p.requires_grad = True
    for p in model.gcn1.parameters():
        p.requires_grad = True
    for p in model.gcn2.parameters():
        p.requires_grad = True
    for p in model.gnn_classifier.parameters():
        p.requires_grad = True
    model.residual_scale.requires_grad = True


def unfreeze_bert(model: EmotionColorGNNBERT) -> None:
    model.set_bert_trainable(True)


# ---------------------------------------------------------------------------
# 4. Metrics (multi-label)
# ---------------------------------------------------------------------------


@torch.no_grad()
def multilabel_f1(
    logits: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    probs = torch.sigmoid(logits)
    pred = (probs >= threshold).float()
    tp = (pred * labels).sum()
    fp = (pred * (1 - labels)).sum()
    fn = ((1 - pred) * labels).sum()
    micro_p = tp / (tp + fp + 1e-8)
    micro_r = tp / (tp + fn + 1e-8)
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r + 1e-8)
    return {
        "micro_precision": float(micro_p.item()),
        "micro_recall": float(micro_r.item()),
        "micro_f1": float(micro_f1.item()),
    }


# ---------------------------------------------------------------------------
# 5. Training loops (Need adjustments)
# ---------------------------------------------------------------------------


def train_goemotions_warmup(
    bert_name: str = "bert-base-uncased",
    batch_size: int = 16,
    lr: float = 2e-5,
    max_length: int = 128,
    epochs: int = 3,
    device: Optional[torch.device] = None,
    color_map_path: Optional[str] = None,
    log_jsonl_path: Optional[str] = None,
    log_every: int = 200,
) -> EmotionColorGNNBERT:
    """Train BERT + bert_head only (GNN path skipped via bert_only=True)."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = load_dataset("SetFit/go_emotions")
    tokenizer = AutoTokenizer.from_pretrained(bert_name)
    train_loader = DataLoader(
        GoEmotionsDataset(ds["train"], tokenizer, max_length),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=goemotions_collate_fn,
    )
    val_loader = DataLoader(
        GoEmotionsDataset(ds["validation"], tokenizer, max_length),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=goemotions_collate_fn,
    )

    model = EmotionColorGNNBERT(
        bert_name=bert_name,
        color_map_path=color_map_path,
    ).to(device)
    freeze_gnn_for_warmup(model)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
    )

    for epoch in range(epochs):
        model.train()
        for step, batch in enumerate(train_loader):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            y = batch["labels"].to(device)
            out = model(ids, mask, labels=y, bert_only=True)
            loss = out["loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if log_jsonl_path and step % log_every == 0:
                _log_step(
                    log_jsonl_path,
                    "warmup_train",
                    epoch,
                    step,
                    tokenizer,
                    ids,
                    out,
                )

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for step, batch in enumerate(val_loader):
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                y = batch["labels"].to(device)
                out = model(ids, mask, labels=y, bert_only=True)
                val_loss += out["loss"].item() * ids.size(0)
                if log_jsonl_path and step % log_every == 0:
                    _log_step(
                        log_jsonl_path,
                        "warmup_val",
                        epoch,
                        step,
                        tokenizer,
                        ids,
                        out,
                    )
        val_loss /= len(val_loader.dataset)
        print(f"[Warmup epoch {epoch + 1}] val loss = {val_loss:.4f}")

    return model


def train_goemotions_joint(
    model: EmotionColorGNNBERT,
    bert_name: str = "bert-base-uncased",
    batch_size: int = 16,
    lr: float = 1e-3,
    max_length: int = 128,
    epochs: int = 3,
    device: Optional[torch.device] = None,
    log_jsonl_path: Optional[str] = None,
    log_every: int = 200,
) -> EmotionColorGNNBERT:
    """Joint: BERT frozen; train color projection + GNN + classifier."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    freeze_for_joint(model)

    ds = load_dataset("SetFit/go_emotions")
    tokenizer = AutoTokenizer.from_pretrained(bert_name)
    train_loader = DataLoader(
        GoEmotionsDataset(ds["train"], tokenizer, max_length),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=goemotions_collate_fn,
    )
    val_loader = DataLoader(
        GoEmotionsDataset(ds["validation"], tokenizer, max_length),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=goemotions_collate_fn,
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
    )

    for epoch in range(epochs):
        model.train()
        for step, batch in enumerate(train_loader):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            y = batch["labels"].to(device)
            out = model(ids, mask, labels=y, bert_only=False)
            loss = out["loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if log_jsonl_path and step % log_every == 0:
                _log_step_joint(
                    log_jsonl_path,
                    "joint_train",
                    epoch,
                    step,
                    tokenizer,
                    ids,
                    out,
                )

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for step, batch in enumerate(val_loader):
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                y = batch["labels"].to(device)
                out = model(ids, mask, labels=y, bert_only=False)
                val_loss += out["loss"].item() * ids.size(0)
                if log_jsonl_path and step % log_every == 0:
                    _log_step_joint(
                        log_jsonl_path,
                        "joint_val",
                        epoch,
                        step,
                        tokenizer,
                        ids,
                        out,
                    )
        val_loss /= len(val_loader.dataset)
        print(f"[Joint epoch {epoch + 1}] val loss = {val_loss:.4f}")

    return model


def train_bert_only_baseline(
    bert_name: str = "bert-base-uncased",
    batch_size: int = 16,
    lr: float = 2e-5,
    max_length: int = 128,
    epochs: int = 3,
    device: Optional[torch.device] = None,
) -> BERTOnlyBaseline:
    """Standalone baseline for evaluation comparison."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = load_dataset("SetFit/go_emotions")
    tokenizer = AutoTokenizer.from_pretrained(bert_name)
    train_loader = DataLoader(
        GoEmotionsDataset(ds["train"], tokenizer, max_length),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=goemotions_collate_fn,
    )
    val_loader = DataLoader(
        GoEmotionsDataset(ds["validation"], tokenizer, max_length),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=goemotions_collate_fn,
    )
    model = BERTOnlyBaseline(bert_name).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            y = batch["labels"].to(device)
            out = model(ids, mask)
            loss = loss_fn(out["logits"], y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                y = batch["labels"].to(device)
                out = model(ids, mask)
                val_loss += loss_fn(out["logits"], y).item() * ids.size(0)
        val_loss /= len(val_loader.dataset)
        print(f"[Baseline epoch {epoch + 1}] val loss = {val_loss:.4f}")

    return model


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    split: str = "validation",
    bert_name: str = "bert-base-uncased",
    batch_size: int = 32,
    max_length: int = 128,
    device: Optional[torch.device] = None,
    gnn_branch: str = "full",
) -> Dict[str, float]:
    """
    For EmotionColorGNNBERTmodel, we set gnn_branch to:
      - 'full': GNN + residual (same as training objective after joint phase)
      - 'bert_only': only the frozen BERT head logits (ablation vs color-GNN path)
    For BERTOnlyBaseline, the gnn_branch is ignored for compairson purposes.
    """
    device = torch.device("cuda") # Force cuda for speed up!
    ds = load_dataset("SetFit/go_emotions")
    tokenizer = AutoTokenizer.from_pretrained(bert_name)
    loader = DataLoader(
        GoEmotionsDataset(ds[split], tokenizer, max_length),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=goemotions_collate_fn,
    )
    model.eval()
    all_logits = []
    all_labels = []
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        y = batch["labels"].to(device)
        if isinstance(model, EmotionColorGNNBERT):
            bo = gnn_branch == "bert_only"
            out = model(ids, mask, labels=None, bert_only=bo)
        else:
            out = model(ids, mask)
        all_logits.append(out["logits"].cpu())
        all_labels.append(y.cpu())
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return multilabel_f1(logits, labels)


def _log_step(
    path: str,
    split: str,
    epoch: int,
    step: int,
    tokenizer,
    input_ids: torch.Tensor,
    out: Dict[str, Any],
) -> None:
    text = tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)
    record = {
        "split": split,
        "epoch": epoch,
        "step": step,
        "text": text,
        "logits": out["logits"][0].detach().cpu().tolist(),
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _log_step_joint(
    path: str,
    split: str,
    epoch: int,
    step: int,
    tokenizer,
    input_ids: torch.Tensor,
    out: Dict[str, Any],
) -> None:
    text = tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)
    record = {
        "split": split,
        "epoch": epoch,
        "step": step,
        "text": text,
        "logits": out["logits"][0].detach().cpu().tolist(),
        "logits_bert": out["logits_bert"][0].detach().cpu().tolist(),
        "logits_gnn": out["logits_gnn"][0].detach().cpu().tolist(),
        "emotion_vectors": out["emotion_vectors"][0].detach().cpu().tolist(),
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def run_full_pipeline(
    bert_name: str = "bert-base-uncased",
    batch_size_warmup: int = 16,
    batch_size_joint: int = 16,
    lr_warmup: float = 2e-5,
    lr_joint: float = 1e-3,
    max_length: int = 128,
    epochs_warmup: int = 3,
    epochs_joint: int = 3,
    device: Optional[torch.device] = None,
    color_map_path: Optional[str] = None,
    log_jsonl_path: Optional[str] = None,
) -> Tuple[EmotionColorGNNBERT, Dict[str, Dict[str, float]]]:
    """
    Warm-up BERT head, then joint GNN + color; evaluate GNN vs BERT-only baseline.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = train_goemotions_warmup(
        bert_name=bert_name,
        batch_size=batch_size_warmup,
        lr=lr_warmup,
        max_length=max_length,
        epochs=epochs_warmup,
        device=device,
        color_map_path=color_map_path,
        log_jsonl_path=log_jsonl_path,
    )
    model = train_goemotions_joint(
        model,
        bert_name=bert_name,
        batch_size=batch_size_joint,
        lr=lr_joint,
        max_length=max_length,
        epochs=epochs_joint,
        device=device,
        log_jsonl_path=log_jsonl_path,
    )

    metrics: Dict[str, Dict[str, float]] = {}
    metrics["gnn_full"] = evaluate_model(
        model,
        split="validation",
        bert_name=bert_name,
        batch_size=32,
        max_length=max_length,
        device=device,
        gnn_branch="full",
    )
    metrics["bert_head_only_same_checkpoint"] = evaluate_model(
        model,
        split="validation",
        bert_name=bert_name,
        batch_size=32,
        max_length=max_length,
        device=device,
        gnn_branch="bert_only",
    )

    print("Validation metrics:", json.dumps(metrics, indent=2))
    return model, metrics
