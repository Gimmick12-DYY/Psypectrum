"""
Configurations for the 10-experiment sweep. Each entry is a self-contained dict
consumed by ``train_one.py``. Common defaults live in ``DEFAULTS`` and per-run
overrides live in ``EXPERIMENTS``.

Run order matters: honest no-color twins are placed adjacent to their
with-color counterparts so a partial run still gives interpretable comparisons.
"""

from __future__ import annotations

from typing import Any, Dict


# Common defaults applied to every run unless overridden.
# Keep batch_size, max_length, epochs identical across runs so comparisons are fair.
DEFAULTS: Dict[str, Any] = {
    # Encoder / data
    "bert_name": "bert-base-uncased",
    "max_length": 128,
    "batch_size_warmup": 32,
    "batch_size_joint": 32,
    "epochs_warmup": 3,
    "epochs_joint": 5,
    # Optimization
    "lr_warmup": 2e-5,
    "lr_joint": 1e-3,
    "bert_lr_joint": 2e-5,
    "n_top_bert_layers": 2,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "warmup_ratio": 0.1,
    "early_stop": True,
    # Loss
    "loss_type": "bce_weighted",
    "asl_gamma_pos": 0.0,
    "asl_gamma_neg": 4.0,
    "asl_clip": 0.05,
    "pos_weight_clip": 3.0,
    # Color (defaults to current architecture)
    "disable_color": False,
    "use_logit_mix_vad": True,
    "use_token_vad": False,
    "color_loss_weight": 0.1,
    "color_anchor_weight": 1e-3,
    "color_logit_scale_init": 0.5,
    "color_teacher_prob": 0.0,
    "supcon_weight": 0.0,
    "supcon_temperature": 0.1,
    # GCN
    "gcn_type": "batch",
    "gcn_n_layers": 2,
    "gcn_internal_residual": False,
    "gcn_dropout": 0.1,
    "gcn_hidden": 896,
    "adj_temperature": 1.0,
    "adj_topk": None,
    "residual_scale_init": 0.5,
    "use_residual": True,
    # Label-graph (only used when gcn_type='label')
    "label_graph_hidden": 512,
    "label_graph_out": 512,
    "label_graph_use_vad_edges": True,
    "label_graph_co_threshold": 0.4,
    "label_graph_co_reweight": 0.25,
    "label_graph_vad_sigma": 0.5,
    "label_graph_vad_topk": 5,
    "label_graph_vad_alpha": 0.5,  # weight on co-occurrence vs VAD
    # Misc
    "seed": 42,
    "model_type": "full",  # 'full' | 'bert_only'
}


# Each experiment overrides a small slice of DEFAULTS. Keep diffs minimal so
# the report can describe each run with one sentence.
EXPERIMENTS: Dict[str, Dict[str, Any]] = {
    # ---- Group A: baselines & honest pairs ----
    "01_bert_only": {
        # Plain BERT pooled -> linear head (uses the existing BERTOnlyBaseline class).
        "model_type": "bert_only",
        "loss_type": "bce",
        "epochs_warmup": 0,
        "epochs_joint": 5,
    },
    "02_current_full": {
        # Reproduce current best (logit-mix VAD only, batch 2-layer GCN, with color).
        "use_logit_mix_vad": True,
        "use_token_vad": False,
        "gcn_type": "batch",
        "gcn_n_layers": 2,
    },
    "03_current_nocolor": {
        # Honest no-color twin of #02: same architecture but trained without color from scratch.
        "disable_color": True,
        "gcn_type": "batch",
        "gcn_n_layers": 2,
    },
    # ---- Group B: token-VAD (give color branch lexical info BERT doesn't expose) ----
    "04_tokenvad_full": {
        "use_logit_mix_vad": True,
        "use_token_vad": True,
        "gcn_type": "batch",
        "gcn_n_layers": 2,
    },
    "05_tokenvad_nocolor": {
        # Honest no-color twin of #04 (token-VAD has no effect when disable_color=True).
        "disable_color": True,
        "gcn_type": "batch",
        "gcn_n_layers": 2,
    },
    # ---- Group C: GCN architecture ----
    "06_tokenvad_sparse_gcn": {
        # 1-layer sparser batch graph; cures over-smoothing.
        "use_logit_mix_vad": True,
        "use_token_vad": True,
        "gcn_type": "batch",
        "gcn_n_layers": 1,
        "adj_topk": 4,
    },
    "07_labelgraph_full": {
        # ML-GCN style label-graph with VAD-induced edges + co-occurrence.
        "use_logit_mix_vad": True,
        "use_token_vad": True,
        "gcn_type": "label",
        "label_graph_use_vad_edges": True,
    },
    "08_labelgraph_nocolor": {
        # Label-graph with co-occurrence ONLY (no VAD edges, no color features) — honest twin of #07.
        "disable_color": True,
        "gcn_type": "label",
        "label_graph_use_vad_edges": False,
    },
    # ---- Group D: extras ----
    "09_tokenvad_contrastive": {
        # Add SupCon loss on color_head: forces 3D bottleneck to be discriminative.
        "use_logit_mix_vad": True,
        "use_token_vad": True,
        "gcn_type": "batch",
        "gcn_n_layers": 2,
        "supcon_weight": 0.1,
        "supcon_temperature": 0.1,
    },
    "10_tokenvad_asl": {
        # Same arch as #04 but ASL loss instead of weighted BCE.
        "use_logit_mix_vad": True,
        "use_token_vad": True,
        "gcn_type": "batch",
        "gcn_n_layers": 2,
        "loss_type": "asl",
    },
}


def merged(name: str) -> Dict[str, Any]:
    """Return DEFAULTS merged with the named experiment overrides."""
    if name not in EXPERIMENTS:
        raise KeyError(f"Unknown experiment: {name!r}. Known: {list(EXPERIMENTS)}")
    cfg = dict(DEFAULTS)
    cfg.update(EXPERIMENTS[name])
    cfg["name"] = name
    return cfg


def all_names() -> list:
    return list(EXPERIMENTS.keys())
