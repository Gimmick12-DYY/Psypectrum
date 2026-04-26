"""
Add-on modules for the Psypectrum experiments suite.

Contents:
  - Per-label P/R/F1 metrics + per-label threshold tuning
  - Token-level NRC-VAD lookup encoder (TokenVADEncoder)
  - Label-graph construction + GCN (ML-GCN style, LabelGraphGCN)
  - Supervised contrastive loss for multi-label

Imported by palette_model.py (model assembly) and train_one.py (training/eval).
Kept in a separate module so palette_model.py stays focused on the core forward
pass and so each new component can be imported and unit-tested in isolation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Per-label metrics
# =============================================================================


@torch.no_grad()
def per_label_prf(
    logits: torch.Tensor,
    labels: torch.Tensor,
    thresholds,
) -> Dict[str, list]:
    """
    Per-label precision/recall/F1/support at the given threshold(s).

    Parameters
    ----------
    logits, labels : Tensor[N, C]
    thresholds : float | sequence | Tensor
        Either a scalar (applied to all labels) or one threshold per label.
    """
    C = logits.size(1)
    if isinstance(thresholds, (int, float)):
        thr = torch.full((C,), float(thresholds))
    else:
        thr = torch.as_tensor(thresholds, dtype=torch.float).view(-1)
        if thr.numel() == 1:
            thr = thr.expand(C).clone()
    thr = thr.to(logits.device)
    probs = torch.sigmoid(logits)
    pred = (probs >= thr.unsqueeze(0)).float()
    tp = (pred * labels).sum(dim=0)
    fp = (pred * (1 - labels)).sum(dim=0)
    fn = ((1 - pred) * labels).sum(dim=0)
    support = labels.sum(dim=0)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return {
        "precision": precision.detach().cpu().tolist(),
        "recall": recall.detach().cpu().tolist(),
        "f1": f1.detach().cpu().tolist(),
        "support": support.detach().cpu().long().tolist(),
        "thresholds": thr.detach().cpu().tolist(),
    }


@torch.no_grad()
def aggregate_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    thresholds,
) -> Dict[str, float]:
    """Macro (mean of per-label) and micro (pooled tp/fp/fn) at given thresholds."""
    C = logits.size(1)
    if isinstance(thresholds, (int, float)):
        thr = torch.full((C,), float(thresholds))
    else:
        thr = torch.as_tensor(thresholds, dtype=torch.float).view(-1)
        if thr.numel() == 1:
            thr = thr.expand(C).clone()
    thr = thr.to(logits.device)
    probs = torch.sigmoid(logits)
    pred = (probs >= thr.unsqueeze(0)).float()
    tp_l = (pred * labels).sum(dim=0)
    fp_l = (pred * (1 - labels)).sum(dim=0)
    fn_l = ((1 - pred) * labels).sum(dim=0)
    p_l = tp_l / (tp_l + fp_l + 1e-8)
    r_l = tp_l / (tp_l + fn_l + 1e-8)
    f1_l = 2 * p_l * r_l / (p_l + r_l + 1e-8)
    tp = tp_l.sum()
    fp = fp_l.sum()
    fn = fn_l.sum()
    micro_p = float((tp / (tp + fp + 1e-8)).item())
    micro_r = float((tp / (tp + fn + 1e-8)).item())
    micro_f1 = (
        2 * micro_p * micro_r / (micro_p + micro_r + 1e-8)
        if (micro_p + micro_r) > 0
        else 0.0
    )
    return {
        "macro_precision": float(p_l.mean().item()),
        "macro_recall": float(r_l.mean().item()),
        "macro_f1": float(f1_l.mean().item()),
        "macro_precision_std": float(p_l.std(unbiased=False).item()),
        "macro_recall_std": float(r_l.std(unbiased=False).item()),
        "macro_f1_std": float(f1_l.std(unbiased=False).item()),
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": micro_f1,
    }


@torch.no_grad()
def tune_thresholds_per_label(
    logits: torch.Tensor,
    labels: torch.Tensor,
    sweep: Optional[List[float]] = None,
) -> torch.Tensor:
    """
    Pick one threshold per label maximizing per-label F1 on the supplied set.
    Returns Tensor[C].
    """
    if sweep is None:
        sweep = [round(0.05 * i, 2) for i in range(1, 20)]  # 0.05 .. 0.95
    C = logits.size(1)
    probs = torch.sigmoid(logits)
    best_thr = torch.full((C,), 0.5, device=logits.device)
    best_f1 = torch.full((C,), -1.0, device=logits.device)
    for t in sweep:
        pred = (probs >= t).float()
        tp = (pred * labels).sum(dim=0)
        fp = (pred * (1 - labels)).sum(dim=0)
        fn = ((1 - pred) * labels).sum(dim=0)
        p = tp / (tp + fp + 1e-8)
        r = tp / (tp + fn + 1e-8)
        f1 = 2 * p * r / (p + r + 1e-8)
        better = f1 > best_f1
        best_thr = torch.where(better, torch.full_like(best_thr, float(t)), best_thr)
        best_f1 = torch.where(better, f1, best_f1)
    return best_thr.detach().cpu()


@torch.no_grad()
def tune_threshold_global(
    logits: torch.Tensor,
    labels: torch.Tensor,
    sweep: Optional[List[float]] = None,
    objective: str = "micro_f1",
) -> float:
    """Single-threshold sweep maximizing the given objective."""
    if sweep is None:
        sweep = [round(0.05 * i, 2) for i in range(1, 13)]  # 0.05 .. 0.60
    best_t = 0.5
    best_score = -1.0
    for t in sweep:
        m = aggregate_metrics(logits, labels, thresholds=float(t))
        s = m.get(objective, -1.0)
        if s > best_score:
            best_score = s
            best_t = float(t)
    return best_t


# =============================================================================
# Token-level NRC-VAD encoder
# =============================================================================


def build_token_vad_table(tokenizer, nrc_vad_path: str) -> torch.Tensor:
    """
    Build a [vocab_size, 3] lookup mapping each tokenizer id to mean NRC-VAD over
    every term whose tokenization includes that id.

    NRC-VAD-Lexicon-v2.1.txt format: TSV with header `term\\tvalence\\tarousal\\tdominance`.
    Multi-word terms ("a battery") are tokenized with the BERT tokenizer and the term's
    VAD is added to *each* of its subword ids; collisions are averaged.

    Tokens never seen in NRC remain zero (treated as no lexical signal).
    """
    size = len(tokenizer)
    table = torch.zeros(size, 3, dtype=torch.float32)
    counts = torch.zeros(size, dtype=torch.float32)
    with open(nrc_vad_path, "r", encoding="utf-8") as f:
        f.readline()  # header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            term = parts[0].strip()
            try:
                v = float(parts[1])
                a = float(parts[2])
                d = float(parts[3])
            except ValueError:
                continue
            ids = tokenizer.encode(term, add_special_tokens=False)
            if not ids:
                continue
            for tid in ids:
                if 0 <= tid < size:
                    table[tid, 0] += v
                    table[tid, 1] += a
                    table[tid, 2] += d
                    counts[tid] += 1
    table = table / counts.clamp(min=1).unsqueeze(-1)
    return table


class TokenVADEncoder(nn.Module):
    """
    Frozen NRC-VAD lookup over input ids -> per-sentence (mean+max)-pool.

    Output: [B, 6]  =  concat( mean over unmasked tokens, max over unmasked tokens ).
    Mean pooling captures average sentiment density; max captures the most extreme
    lexical hit. Both ignore PAD and special tokens via attention_mask.
    """

    def __init__(self, table: torch.Tensor):
        super().__init__()
        self.register_buffer("table", table)  # [V, 3]

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        vad = self.table[input_ids]  # [B, L, 3]
        mask = attention_mask.float().unsqueeze(-1)  # [B, L, 1]
        summed = (vad * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        mean = summed / denom  # [B, 3]
        bool_mask = attention_mask.bool().unsqueeze(-1).expand_as(vad)
        masked_for_max = vad.masked_fill(~bool_mask, float("-inf"))
        maxed = masked_for_max.max(dim=1).values  # [B, 3]
        maxed = torch.where(torch.isfinite(maxed), maxed, torch.zeros_like(maxed))
        return torch.cat([mean, maxed], dim=-1)  # [B, 6]


# =============================================================================
# Label-graph construction + GCN (ML-GCN style)
# =============================================================================


def build_label_cooccurrence_adjacency(
    label_matrix: torch.Tensor,
    threshold: float = 0.4,
    reweight: float = 0.25,
) -> torch.Tensor:
    """
    Co-occurrence adjacency over labels (Chen et al., ML-GCN, CVPR 2019).

    label_matrix : Tensor[N, C], multi-hot training labels.
    Returns      : Tensor[C, C], symmetric, self-loop included.
    """
    C = label_matrix.size(1)
    co = label_matrix.t() @ label_matrix  # [C, C]
    diag = co.diag().clamp(min=1.0)
    cond = co / diag.unsqueeze(1)  # P(j | i)
    cond = (cond + cond.t()) / 2  # symmetrize
    A = (cond >= threshold).float()
    A.fill_diagonal_(0.0)
    row_sum = A.sum(dim=1, keepdim=True).clamp(min=1.0)
    A = A * (reweight / row_sum) + torch.eye(C) * (1.0 - reweight)
    return A


def build_vad_adjacency(
    label_color_table: torch.Tensor,
    sigma: float = 0.5,
    top_k: int = 5,
) -> torch.Tensor:
    """
    Symmetric kNN graph on per-label VAD vectors with a Gaussian kernel weight.

    label_color_table : Tensor[C, 3]
    Returns            : Tensor[C, C], symmetric, self-loop, row-normalized.
    """
    C = label_color_table.size(0)
    dist = torch.cdist(label_color_table, label_color_table, p=2)
    sim = torch.exp(-dist.pow(2) / (2.0 * sigma * sigma))
    sim.fill_diagonal_(0.0)
    if top_k is not None and 0 < top_k < C:
        _, idx = sim.topk(top_k, dim=-1)
        keep = torch.zeros_like(sim, dtype=torch.bool).scatter_(-1, idx, True)
        sim = sim * keep.float()
    sim = (sim + sim.t()) / 2
    sim = sim + torch.eye(C)
    deg = sim.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return sim / deg


def combine_adjacencies(
    A_co: torch.Tensor,
    A_vad: Optional[torch.Tensor],
    alpha: float = 0.5,
) -> torch.Tensor:
    """Convex combination; if A_vad is None, returns A_co."""
    if A_vad is None:
        return A_co
    return alpha * A_co + (1.0 - alpha) * A_vad


class LabelGraphGCN(nn.Module):
    """
    Two-layer GCN over the [C, C] label adjacency. Output [C, out_dim] used as
    per-label classifier weights: ``logits = features @ output.T``.

    Label embeddings (initial node features) are learnable Parameters, initialized
    from VAD or any other prior provided by the caller.
    """

    def __init__(
        self,
        label_init: torch.Tensor,
        adj: torch.Tensor,
        hidden_dim: int = 512,
        out_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert label_init.size(0) == adj.size(0) == adj.size(1), (
            f"label_init {tuple(label_init.shape)} vs adj {tuple(adj.shape)} mismatch"
        )
        self.register_buffer("adj", adj)
        in_dim = label_init.size(1)
        self.label_emb = nn.Parameter(label_init.clone())
        self.gc1 = nn.Linear(in_dim, hidden_dim)
        self.gc2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self) -> torch.Tensor:
        h = self.label_emb
        h = self.adj @ h
        h = F.gelu(self.gc1(h))
        h = self.dropout(h)
        h = self.adj @ h
        h = self.gc2(h)
        return h  # [C, out_dim]


# =============================================================================
# Supervised contrastive loss for multi-label
# =============================================================================


def supervised_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    Khosla et al. (2020) SupCon adapted for multi-label: two examples are
    considered positive if their label sets share at least one label.

    features : Tensor[B, D] — l2-normalized inside.
    labels   : Tensor[B, C] multi-hot.
    """
    B = features.size(0)
    if B < 2:
        return features.new_zeros(())
    feat = F.normalize(features, dim=-1)
    sim = feat @ feat.t() / max(temperature, 1e-8)
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()  # numerical stability
    eye = torch.eye(B, device=features.device, dtype=torch.bool)
    overlap = (labels @ labels.t()) > 0  # share any label
    pos_mask = overlap & ~eye
    exp_sim = torch.exp(sim).masked_fill(eye, 0.0)
    denom = exp_sim.sum(dim=1, keepdim=True).clamp(min=1e-8)
    log_prob = sim - torch.log(denom)
    pos_count = pos_mask.float().sum(dim=1)
    has_pos = (pos_count > 0).float()
    pos_log_prob = (log_prob * pos_mask.float()).sum(dim=1) / pos_count.clamp(min=1.0)
    denom_examples = has_pos.sum().clamp(min=1.0)
    return -(pos_log_prob * has_pos).sum() / denom_examples


__all__ = [
    "per_label_prf",
    "aggregate_metrics",
    "tune_thresholds_per_label",
    "tune_threshold_global",
    "build_token_vad_table",
    "TokenVADEncoder",
    "build_label_cooccurrence_adjacency",
    "build_vad_adjacency",
    "combine_adjacencies",
    "LabelGraphGCN",
    "supervised_contrastive_loss",
]
