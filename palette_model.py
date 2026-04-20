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
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


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
# 3. Models + losses
# ---------------------------------------------------------------------------


class AsymmetricLoss(nn.Module):
    """
    Asymmetric multi-label loss (Ben-Baruch et al., ICCV 2021).

    Defaults follow the ASL paper: ``gamma_neg=4``, ``gamma_pos=0``, ``clip=0.05``.
    Down-weights easy negatives and probability-mass on already-confident negatives,
    which helps imbalanced multi-label setups like GoEmotions.
    """

    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 4.0,
        clip: float = 0.05,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        p = torch.sigmoid(logits)
        p_neg = 1.0 - p
        if self.clip > 0:
            p_neg = (p_neg + self.clip).clamp(max=1.0)
        los_pos = targets * torch.log(p.clamp(min=self.eps))
        los_neg = (1.0 - targets) * torch.log(p_neg.clamp(min=self.eps))
        loss = los_pos + los_neg
        if self.gamma_pos > 0 or self.gamma_neg > 0:
            pt = p * targets + p_neg * (1.0 - targets)
            gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
            loss = loss * torch.pow(1.0 - pt, gamma)
        return -loss.mean()


def compute_pos_weight(hf_split) -> torch.Tensor:
    """
    Compute BCE ``pos_weight`` = N_neg / N_pos per label on a GoEmotions split.
    Returns a tensor of shape ``(NUM_LABELS,)``.
    """
    total = len(hf_split)
    pos = torch.zeros(NUM_LABELS)
    for row in hf_split:
        pos += torch.tensor(
            [float(row[label]) for label in GOEMOTIONS_LABELS],
            dtype=torch.float32,
        )
    pos = pos.clamp(min=1.0)
    neg = (total - pos).clamp(min=1.0)
    return neg / pos


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
        adj_topk: Optional[int] = None,
        use_residual: bool = True,
        color_teacher_prob: float = 0.0,
        loss_type: str = "bce",
        pos_weight: Optional[torch.Tensor] = None,
        pos_weight_clip: float = 3.0,
        asl_gamma_pos: float = 0.0,
        asl_gamma_neg: float = 4.0,
        asl_clip: float = 0.05,
        focal_gamma: float = 2.0,
        color_loss_weight: float = 0.1,
        color_anchor_weight: float = 1e-3,
        gcn_dropout: float = 0.1,
        residual_scale_init: float = 0.5,
        color_logit_scale_init: float = 0.5,
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_name)
        hidden = self.bert.config.hidden_size
        self.bert_head = nn.Linear(hidden, num_labels)

        cmap = load_color_map(color_map_path or default_color_map_path())
        table = _label_color_table(cmap)
        # Trainable label color embeddings (initialized from COLOR_MAP); anchored
        # to the hand-designed geometry via ``color_anchor_weight`` so they can
        # refine but not drift.
        self.label_color_vectors = nn.Parameter(table.clone())
        self.register_buffer("label_color_vectors_init", table.clone())

        self.ln_bert = nn.LayerNorm(hidden)
        self.color_proj = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
        )
        self.ln_color = nn.LayerNorm(128)

        # Auxiliary head: predict the label-averaged color coordinate from pooled
        # BERT. Targets use ground-truth labels only during training; eval never
        # uses labels (no train/eval mismatch).
        self.color_head = nn.Linear(hidden, 3)
        self.color_loss_weight = color_loss_weight
        self.color_anchor_weight = color_anchor_weight

        # Direct color -> label classifier head (3 -> num_labels). Initialized so
        # its initial behavior is a dot product against the color map (same
        # geometry as a cosine-similarity bias, but each label has its own
        # free weights so it can refine its effective color direction).
        # Trained implicitly through the main BCE via the +scale*color_bias
        # addition to the logits; no separate discriminative BCE (that created
        # calibration coupling between logits_main and color_bias and broke
        # the bert_gnn ablation).
        self.color_to_label = nn.Linear(3, num_labels, bias=True)
        with torch.no_grad():
            self.color_to_label.weight.copy_(table)
            self.color_to_label.bias.zero_()

        concat_dim = hidden + 128
        self.gcn1 = nn.Linear(concat_dim, gcn_hidden)
        self.gcn2 = nn.Linear(gcn_hidden, gcn_hidden)
        self.gnn_dropout = nn.Dropout(gcn_dropout)
        self.gnn_classifier = nn.Linear(gcn_hidden, num_labels)

        self.adj_temperature = adj_temperature
        self.adj_topk = adj_topk
        self.use_residual = use_residual
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))

        # Learnable scale on the color logit-bias pathway (applied only when
        # ``use_color`` is true). Gives the color module a direct, non-redundant
        # classifier pathway: logit_ell += scale * color_to_label(color_head(pooled))_ell.
        # Initialized modestly (0.5) so the pathway opens up gradually rather
        # than aggressively trading precision for recall.
        self.color_logit_scale = nn.Parameter(torch.tensor(float(color_logit_scale_init)))

        # Teacher forcing is disabled by default to avoid train/eval mismatch.
        # Kept as an off-by-default option for ablation studies.
        self.color_teacher_prob = color_teacher_prob

        self.loss_type = loss_type
        if loss_type == "bce":
            self._loss = nn.BCEWithLogitsLoss()
        elif loss_type == "bce_weighted":
            if pos_weight is None:
                raise ValueError("loss_type='bce_weighted' requires pos_weight")
            if pos_weight_clip is not None and pos_weight_clip > 0:
                pos_weight = pos_weight.clamp(max=pos_weight_clip)
            self.register_buffer("pos_weight", pos_weight.clone())
            self._loss = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        elif loss_type == "asl":
            self._loss = AsymmetricLoss(
                gamma_pos=asl_gamma_pos,
                gamma_neg=asl_gamma_neg,
                clip=asl_clip,
            )
        elif loss_type == "focal":
            # Focal is a special case of ASL with symmetric gammas and no clip.
            self._loss = AsymmetricLoss(
                gamma_pos=focal_gamma,
                gamma_neg=focal_gamma,
                clip=0.0,
            )
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

    def _batch_adjacency(self, pooled: torch.Tensor) -> torch.Tensor:
        """
        Softmax similarity + self-loop (B, B). Optionally top-k sparsified to avoid
        spreading each node's features across unrelated batch members.
        """
        z = F.normalize(pooled, p=2, dim=-1)
        sim = torch.matmul(z, z.transpose(0, 1)) / self.adj_temperature

        b = sim.size(0)
        if self.adj_topk is not None and 0 < self.adj_topk < b:
            topk_vals, topk_idx = sim.topk(self.adj_topk, dim=-1)
            sparse = torch.full_like(sim, float("-inf"))
            sparse.scatter_(-1, topk_idx, topk_vals)
            sim = sparse

        adj = F.softmax(sim, dim=-1)
        eye = torch.eye(b, device=adj.device, dtype=adj.dtype)
        adj = adj + eye
        adj = adj / (adj.sum(dim=-1, keepdim=True) + 1e-8)
        return adj

    def _emotion_vectors(
        self,
        logits_bert: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Weighted sum over ``label_color_vectors``.

        If ``labels`` is provided and we're in training mode, each example uses the
        teacher-forced label distribution with probability ``color_teacher_prob``
        (and the predicted sigmoid distribution otherwise). This gives the color
        branch a signal that isn't just a deterministic copy of the BERT head.
        """
        probs = torch.sigmoid(logits_bert)
        mass = probs.sum(dim=-1, keepdim=True) + 1e-8
        w_pred = probs / mass

        if (
            self.training
            and labels is not None
            and self.color_teacher_prob > 0.0
        ):
            lab_mass = labels.sum(dim=-1, keepdim=True) + 1e-8
            w_lab = labels / lab_mass
            use_teacher = (
                torch.rand(labels.size(0), 1, device=labels.device)
                < self.color_teacher_prob
            ).float()
            w = use_teacher * w_lab + (1.0 - use_teacher) * w_pred
        else:
            w = w_pred

        return torch.matmul(w, self.label_color_vectors)

    def _color_aux_loss(
        self,
        pooled: torch.Tensor,
        labels: torch.Tensor,
        pred_color: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Auxiliary color reconstruction: predict the label-averaged color vector
        from pooled BERT, target = mean of ground-truth label colors. Uses the
        anchored (init) label colors as the target so the supervision reflects
        the hand-designed geometry, not transiently drifted values.
        """
        mass = labels.sum(dim=-1, keepdim=True).clamp(min=1.0)
        target = (labels @ self.label_color_vectors_init) / mass
        if pred_color is None:
            pred_color = self.color_head(pooled)
        return F.mse_loss(pred_color, target)

    def _color_anchor_loss(self) -> torch.Tensor:
        return F.mse_loss(self.label_color_vectors, self.label_color_vectors_init)

    def _color_logit_bias(self, pred_color: torch.Tensor) -> torch.Tensor:
        """
        Direct color -> label pathway: a small learned linear head maps the
        predicted 3-D sentence color to per-label logits. The head is
        initialized from ``label_color_vectors`` so its initial behavior is a
        dot product against the color map (same geometry as a cosine-similarity
        bias), but with free per-label weights it can refine each label's
        effective color direction. Added to the final logits with a learnable
        scale when ``use_color=True``; this is the component ablated by the
        ``bert_gnn`` branch.
        """
        return self.color_to_label(pred_color)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        bert_only: bool = False,
        use_color: bool = True,
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

        x_b = self.ln_bert(pooled)
        pred_color: Optional[torch.Tensor] = None
        if use_color:
            e_vec = self._emotion_vectors(logits_bert, labels=labels)
            c = self.color_proj(e_vec)
            x_c = self.ln_color(c)
            x = torch.cat([x_b, x_c], dim=-1)
            pred_color = self.color_head(pooled)
        else:
            # Same 896-dim layout as full model: BERT half + zeros (no color module).
            e_vec = None
            x = torch.cat(
                [x_b, pooled.new_zeros(pooled.size(0), 128)],
                dim=-1,
            )

        adj = self._batch_adjacency(pooled.detach())
        h = torch.matmul(adj, x)
        h = F.gelu(self.gcn1(h))
        h = self.gnn_dropout(h)
        h = torch.matmul(adj, h)
        h = F.gelu(self.gcn2(h))
        h = self.gnn_dropout(h)
        logits_gnn = self.gnn_classifier(h)

        if self.use_residual:
            logits = logits_gnn + self.residual_scale * logits_bert
        else:
            logits = logits_gnn

        # Direct color -> label pathway: only active when the color module is in use.
        color_bias: Optional[torch.Tensor] = None
        if use_color and pred_color is not None:
            color_bias = self._color_logit_bias(pred_color)
            logits = logits + self.color_logit_scale * color_bias

        loss = None
        loss_cls = None
        loss_color = None
        loss_anchor = None
        if labels is not None:
            loss_cls = self._loss(logits, labels)
            loss = loss_cls
            if use_color and self.color_loss_weight > 0.0:
                loss_color = self._color_aux_loss(pooled, labels, pred_color=pred_color)
                loss = loss + self.color_loss_weight * loss_color
            if use_color and self.color_anchor_weight > 0.0:
                loss_anchor = self._color_anchor_loss()
                loss = loss + self.color_anchor_weight * loss_anchor

        out_dict: Dict[str, Any] = {
            "loss": loss,
            "loss_cls": loss_cls,
            "loss_color": loss_color,
            "loss_anchor": loss_anchor,
            "logits": logits,
            "logits_bert": logits_bert,
            "logits_gnn": logits_gnn,
            "pooled": pooled,
        }
        if e_vec is not None:
            out_dict["emotion_vectors"] = e_vec
        if pred_color is not None:
            out_dict["pred_color"] = pred_color
        if color_bias is not None:
            out_dict["color_bias"] = color_bias
        return out_dict

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
    for p in model.color_head.parameters():
        p.requires_grad = False
    for p in model.color_to_label.parameters():
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
    model.color_logit_scale.requires_grad = False
    model.label_color_vectors.requires_grad = False


def freeze_for_joint(model: EmotionColorGNNBERT) -> None:
    """Freeze BERT + BERT head; train color projection, norms, GCN, classifier."""
    model.set_bert_trainable(False)
    for p in model.color_proj.parameters():
        p.requires_grad = True
    for p in model.color_head.parameters():
        p.requires_grad = True
    for p in model.color_to_label.parameters():
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
    model.color_logit_scale.requires_grad = True
    model.label_color_vectors.requires_grad = True


def unfreeze_bert(model: EmotionColorGNNBERT) -> None:
    model.set_bert_trainable(True)


def unfreeze_top_bert_layers(model: EmotionColorGNNBERT, n_layers: int) -> List[nn.Parameter]:
    """
    Unfreeze the top ``n_layers`` transformer blocks of ``model.bert`` (and the
    pooler if present) for fine-grained joint training with a small LR.
    Returns the list of newly-trainable parameters for use in a distinct optimizer
    param group.
    """
    if n_layers <= 0:
        return []
    encoder_layers = model.bert.encoder.layer
    top = encoder_layers[-n_layers:]
    params: List[nn.Parameter] = []
    for layer in top:
        for p in layer.parameters():
            p.requires_grad = True
            params.append(p)
    if hasattr(model.bert, "pooler") and model.bert.pooler is not None:
        for p in model.bert.pooler.parameters():
            p.requires_grad = True
            params.append(p)
    return params


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
    # Macro metrics: average per-label metrics (equal label weighting).
    tp_label = (pred * labels).sum(dim=0)
    fp_label = (pred * (1 - labels)).sum(dim=0)
    fn_label = ((1 - pred) * labels).sum(dim=0)
    precision_label = tp_label / (tp_label + fp_label + 1e-8)
    recall_label = tp_label / (tp_label + fn_label + 1e-8)
    f1_label = 2 * precision_label * recall_label / (precision_label + recall_label + 1e-8)
    macro_p = precision_label.mean()
    macro_r = recall_label.mean()
    macro_f1 = f1_label.mean()
    return {
        "macro_precision": float(macro_p.item()),
        "macro_recall": float(macro_r.item()),
        "macro_f1": float(macro_f1.item()),
        "micro_precision": float(micro_p.item()),
        "micro_recall": float(micro_r.item()),
        "micro_f1": float(micro_f1.item()),
    }


@torch.no_grad()
def tune_threshold(
    logits: torch.Tensor,
    labels: torch.Tensor,
    thresholds: Optional[List[float]] = None,
    objective: str = "micro_f1",
) -> Tuple[float, Dict[str, float]]:
    """
    Sweep decision thresholds on validation logits/labels and return the one
    that maximizes ``objective`` (default: ``micro_f1``) along with the full
    metrics dict at that threshold. Lowering the threshold trades precision
    for recall; tuning picks the per-model sweet spot instead of pinning 0.5.
    """
    if thresholds is None:
        thresholds = [round(0.05 * i, 2) for i in range(2, 13)]  # 0.10..0.60
    best_t = 0.5
    best_metrics: Dict[str, float] = multilabel_f1(logits, labels, threshold=best_t)
    best_score = best_metrics.get(objective, -1.0)
    for t in thresholds:
        m = multilabel_f1(logits, labels, threshold=float(t))
        if m.get(objective, -1.0) > best_score:
            best_score = m[objective]
            best_t = float(t)
            best_metrics = m
    return best_t, best_metrics


# ---------------------------------------------------------------------------
# 5. Training loops (Need adjustments)
# ---------------------------------------------------------------------------


def train_goemotions_warmup(
    bert_name: str = "bert-base-uncased",
    batch_size: int = 16,
    lr: float = 2e-5,
    max_length: int = 128,
    epochs: int = 4,
    device: Optional[torch.device] = None,
    color_map_path: Optional[str] = None,
    log_jsonl_path: Optional[str] = None,
    log_every: int = 200,
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    warmup_ratio: float = 0.1,
    model_kwargs: Optional[Dict[str, Any]] = None,
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

    mk: Dict[str, Any] = dict(model_kwargs or {})
    # Auto-compute pos_weight from the train split if the caller asked for it.
    if mk.get("loss_type") == "bce_weighted" and "pos_weight" not in mk:
        mk["pos_weight"] = compute_pos_weight(ds["train"])

    model = EmotionColorGNNBERT(
        bert_name=bert_name,
        color_map_path=color_map_path,
        **mk,
    ).to(device)
    freeze_gnn_for_warmup(model)

    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    param_groups = _build_param_groups(named, base_lr=lr, weight_decay=weight_decay)
    optimizer = torch.optim.AdamW(param_groups)

    total_steps = max(1, len(train_loader) * epochs)
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
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
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_grad_norm,
                )
            optimizer.step()
            scheduler.step()

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


def _build_param_groups(
    named_params: List[Tuple[str, nn.Parameter]],
    base_lr: float,
    weight_decay: float,
    bert_lr: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Split parameters into decay / no-decay groups and optionally a lower-LR BERT
    group. Excludes bias and LayerNorm weights from weight decay (standard practice).
    """
    no_decay_substrings = ("bias", "LayerNorm.weight", "layer_norm.weight", "ln_bert", "ln_color")

    def is_no_decay(name: str) -> bool:
        return any(s in name for s in no_decay_substrings)

    groups: Dict[Tuple[bool, bool], List[nn.Parameter]] = {
        (False, False): [],
        (False, True): [],
        (True, False): [],
        (True, True): [],
    }
    for name, p in named_params:
        if not p.requires_grad:
            continue
        nd = is_no_decay(name)
        is_bert = name.startswith("bert.")
        groups[(is_bert, nd)].append(p)

    out: List[Dict[str, Any]] = []
    if groups[(False, False)]:
        out.append({"params": groups[(False, False)], "lr": base_lr, "weight_decay": weight_decay})
    if groups[(False, True)]:
        out.append({"params": groups[(False, True)], "lr": base_lr, "weight_decay": 0.0})
    if bert_lr is not None and (groups[(True, False)] or groups[(True, True)]):
        if groups[(True, False)]:
            out.append({"params": groups[(True, False)], "lr": bert_lr, "weight_decay": weight_decay})
        if groups[(True, True)]:
            out.append({"params": groups[(True, True)], "lr": bert_lr, "weight_decay": 0.0})
    else:
        if groups[(True, False)]:
            out.append({"params": groups[(True, False)], "lr": base_lr, "weight_decay": weight_decay})
        if groups[(True, True)]:
            out.append({"params": groups[(True, True)], "lr": base_lr, "weight_decay": 0.0})
    return out


def train_goemotions_joint(
    model: EmotionColorGNNBERT,
    bert_name: str = "bert-base-uncased",
    batch_size: int = 16,
    lr: float = 1e-3,
    max_length: int = 128,
    epochs: int = 4,
    device: Optional[torch.device] = None,
    log_jsonl_path: Optional[str] = None,
    log_every: int = 200,
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    warmup_ratio: float = 0.1,
    n_top_bert_layers: int = 2,
    bert_lr: float = 2e-5,
    early_stop: bool = True,
) -> EmotionColorGNNBERT:
    """
    Joint training: BERT body frozen except the top ``n_top_bert_layers`` blocks
    (fine-tuned at ``bert_lr``). Trains color + GCN + classifier at ``lr`` with
    AdamW (weight decay 0.01, LayerNorm/bias excluded), linear warmup +
    decay schedule, and gradient clipping. Tracks best validation micro-F1 and
    restores those weights before returning if ``early_stop`` is set.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    freeze_for_joint(model)
    bert_params = unfreeze_top_bert_layers(model, n_top_bert_layers)
    bert_param_ids = {id(p) for p in bert_params}

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

    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    param_groups = _build_param_groups(
        named,
        base_lr=lr,
        weight_decay=weight_decay,
        bert_lr=bert_lr if bert_param_ids else None,
    )
    optimizer = torch.optim.AdamW(param_groups)

    total_steps = max(1, len(train_loader) * epochs)
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    best_f1 = -1.0
    best_threshold = 0.5
    best_state: Optional[Dict[str, torch.Tensor]] = None

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
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_grad_norm,
                )
            optimizer.step()
            scheduler.step()

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
        val_logits: List[torch.Tensor] = []
        val_labels: List[torch.Tensor] = []
        val_loss = 0.0
        with torch.no_grad():
            for step, batch in enumerate(val_loader):
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                y = batch["labels"].to(device)
                out = model(ids, mask, labels=y, bert_only=False)
                val_loss += out["loss"].item() * ids.size(0)
                val_logits.append(out["logits"].detach().cpu())
                val_labels.append(y.detach().cpu())
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
        logits_cat = torch.cat(val_logits)
        labels_cat = torch.cat(val_labels)
        m_default = multilabel_f1(logits_cat, labels_cat, threshold=0.5)
        best_t, m_tuned = tune_threshold(logits_cat, labels_cat, objective="micro_f1")
        print(
            f"[Joint epoch {epoch + 1}] val loss = {val_loss:.4f} | "
            f"t=0.50 P={m_default['micro_precision']:.4f} "
            f"R={m_default['micro_recall']:.4f} F1={m_default['micro_f1']:.4f} | "
            f"t*={best_t:.2f} P={m_tuned['micro_precision']:.4f} "
            f"R={m_tuned['micro_recall']:.4f} F1={m_tuned['micro_f1']:.4f}"
        )
        if m_tuned["micro_f1"] > best_f1:
            best_f1 = m_tuned["micro_f1"]
            best_threshold = best_t
            if early_stop:
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if early_stop and best_state is not None:
        print(
            f"[Joint] Restoring best weights (val micro-F1 = {best_f1:.4f} "
            f"at threshold {best_threshold:.2f})"
        )
        model.load_state_dict(best_state)

    # Stash the tuned threshold so downstream evaluation can reuse it.
    model.best_threshold = float(best_threshold)
    return model


def train_bert_only_baseline(
    bert_name: str = "bert-base-uncased",
    batch_size: int = 16,
    lr: float = 2e-5,
    max_length: int = 128,
    epochs: int = 4,
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
    threshold: Optional[float] = None,
    tune: bool = False,
) -> Dict[str, float]:
    """
    For EmotionColorGNNBERT, gnn_branch selects the forward path:
      - 'full' or 'bert_color_gnn': BERT + color concat + GCN + residual (trained model)
      - 'bert_gnn' or 'gnn_no_color': same GCN stack but no color features (zeros in the
        128-dim slot; fair ablation vs adding the color module)
      - 'bert_only': BERT head logits only (no GNN)
    For BERTOnlyBaseline, gnn_branch is ignored.
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
            if gnn_branch == "bert_only":
                out = model(ids, mask, labels=None, bert_only=True)
            elif gnn_branch in ("bert_gnn", "gnn_no_color"):
                out = model(ids, mask, labels=None, bert_only=False, use_color=False)
            else:
                out = model(ids, mask, labels=None, bert_only=False, use_color=True)
        else:
            out = model(ids, mask)
        all_logits.append(out["logits"].cpu())
        all_labels.append(y.cpu())
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    if tune:
        best_t, metrics = tune_threshold(logits, labels, objective="micro_f1")
        metrics["threshold"] = best_t
        return metrics
    if threshold is None:
        threshold = float(getattr(model, "best_threshold", 0.5))
    metrics = multilabel_f1(logits, labels, threshold=threshold)
    metrics["threshold"] = float(threshold)
    return metrics


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
    }
    if "emotion_vectors" in out:
        record["emotion_vectors"] = out["emotion_vectors"][0].detach().cpu().tolist()
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def run_full_pipeline(
    bert_name: str = "bert-base-uncased",
    batch_size_warmup: int = 16,
    batch_size_joint: int = 16,
    lr_warmup: float = 2e-5,
    lr_joint: float = 1e-3,
    max_length: int = 128,
    epochs_warmup: int = 4,
    epochs_joint: int = 4,
    device: Optional[torch.device] = None,
    color_map_path: Optional[str] = None,
    log_jsonl_path: Optional[str] = None,
    metrics_plot_path: Optional[str] = None,
    # ---- Training / architecture knobs (defaults reflect the refined framework) ----
    loss_type: str = "bce_weighted",
    asl_gamma_pos: float = 0.0,
    asl_gamma_neg: float = 4.0,
    asl_clip: float = 0.05,
    color_teacher_prob: float = 0.0,
    color_loss_weight: float = 0.1,
    color_anchor_weight: float = 1e-3,
    color_logit_scale_init: float = 0.5,
    gcn_dropout: float = 0.1,
    residual_scale_init: float = 0.5,
    adj_temperature: float = 1.0,
    adj_topk: Optional[int] = None,
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    warmup_ratio: float = 0.1,
    n_top_bert_layers: int = 2,
    bert_lr_joint: float = 2e-5,
    early_stop: bool = True,
) -> Tuple[EmotionColorGNNBERT, Dict[str, Dict[str, float]]]:
    """
    Warm-up BERT head, then joint GNN + color; evaluate BERT-ColorGNN vs BERT-GNN (no color).

    Defaults implement the refined framework:
      - ``loss_type='bce'``: keeps the Bayes-optimal threshold near 0.5 so a fixed
        0.5 decision threshold stays valid.
      - ``color_teacher_prob=0.0``: no teacher forcing, so train == eval for the
        color branch.
      - ``color_loss_weight=0.1``: auxiliary MSE loss on a ``pooled -> 3D`` head
        with the ground-truth label-color mixture as target. This is how the color
        map injects supervision without train/eval mismatch.
      - ``color_anchor_weight=1e-3``: gentle L2 pull on the trainable color
        vectors toward the ``COLOR_MAP.txt`` initialization.
      - ``color_logit_scale_init=0.5``: modest initial weight on the color
        logit-bias term so the color pathway opens gradually. The scale is a
        trainable scalar, so the model can grow it if the color path is useful.
      - ``adj_temperature=1.0`` / ``adj_topk=None``: batch graph defaults reverted;
        tune via kwargs if desired.
      - Joint phase still unfreezes the top ``n_top_bert_layers`` BERT blocks at
        ``bert_lr_joint`` and uses linear warmup + decay, weight decay, grad
        clipping, and best-F1 early stopping.

    If ``metrics_plot_path`` is set (e.g. ``logs/metrics.png``), saves a figure from
    ``metrics_viz`` (requires matplotlib).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_kwargs: Dict[str, Any] = {
        "adj_temperature": adj_temperature,
        "adj_topk": adj_topk,
        "color_teacher_prob": color_teacher_prob,
        "loss_type": loss_type,
        "asl_gamma_pos": asl_gamma_pos,
        "asl_gamma_neg": asl_gamma_neg,
        "asl_clip": asl_clip,
        "color_loss_weight": color_loss_weight,
        "color_anchor_weight": color_anchor_weight,
        "color_logit_scale_init": color_logit_scale_init,
        "gcn_dropout": gcn_dropout,
        "residual_scale_init": residual_scale_init,
    }

    model = train_goemotions_warmup(
        bert_name=bert_name,
        batch_size=batch_size_warmup,
        lr=lr_warmup,
        max_length=max_length,
        epochs=epochs_warmup,
        device=device,
        color_map_path=color_map_path,
        log_jsonl_path=log_jsonl_path,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        warmup_ratio=warmup_ratio,
        model_kwargs=model_kwargs,
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
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        warmup_ratio=warmup_ratio,
        n_top_bert_layers=n_top_bert_layers,
        bert_lr=bert_lr_joint,
        early_stop=early_stop,
    )

    metrics: Dict[str, Dict[str, float]] = {}
    # Tune the decision threshold per branch on the validation set; GoEmotions
    # is negative-dominated so a fixed 0.5 threshold systematically suppresses
    # recall. ``tune=True`` picks the micro-F1-maximizing threshold (swept from
    # 0.10 to 0.60 in 0.05 increments).
    metrics["bert_color_gnn"] = evaluate_model(
        model,
        split="validation",
        bert_name=bert_name,
        batch_size=32,
        max_length=max_length,
        device=device,
        gnn_branch="full",
        tune=True,
    )
    metrics["bert_gnn"] = evaluate_model(
        model,
        split="validation",
        bert_name=bert_name,
        batch_size=32,
        max_length=max_length,
        device=device,
        gnn_branch="bert_gnn",
        tune=True,
    )

    print("Validation metrics:", json.dumps(metrics, indent=2))

    if metrics_plot_path:
        try:
            from metrics_viz import save_pipeline_metrics_figure

            save_pipeline_metrics_figure(
                metrics,
                metrics_plot_path,
                suptitle="GoEmotions validation",
            )
            print(f"Saved metrics figure to {metrics_plot_path}")
        except ImportError as exc:
            print(f"Skipping metrics plot (install matplotlib): {exc}")

    return model, metrics
