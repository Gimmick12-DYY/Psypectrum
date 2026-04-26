"""
Single-experiment trainer.

Usage:
    python train_one.py --config 04_tokenvad_full
    python train_one.py --config 04_tokenvad_full --smoke   # tiny smoke test

Reads a named config from ``experiments_config.py``, sets seeds, builds the right
model + dataloaders, runs warmup + joint phases, and writes:

    results/<name>/per_label_val.csv
    results/<name>/per_label_test.csv
    results/<name>/summary.json
    results/<name>/config.json
    checkpoints/<name>/best.pt
    checkpoints/<name>/last.pt

Resumable: if ``results/<name>/summary.json`` already exists, skips the run
unless ``--force`` is passed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# tqdm: required (in environment.yml); fall back gracefully if missing.
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(it, **kw):  # type: ignore
        return it

from datasets import load_dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from palette_model import (
    BERTOnlyBaseline,
    EmotionColorGNNBERT,
    GOEMOTIONS_LABELS,
    GoEmotionsDataset,
    NUM_LABELS,
    _build_param_groups,
    compute_pos_weight,
    default_color_map_path,
    freeze_for_joint,
    freeze_gnn_for_warmup,
    goemotions_collate_fn,
    load_color_map,
    unfreeze_top_bert_layers,
    _label_color_table,
)
from model_extensions import (
    aggregate_metrics,
    build_label_cooccurrence_adjacency,
    build_token_vad_table,
    build_vad_adjacency,
    combine_adjacencies,
    per_label_prf,
    tune_thresholds_per_label,
)
from experiments_config import merged


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
CACHE_DIR = PROJECT_ROOT / "cache"


# =============================================================================
# Utility helpers
# =============================================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(arg: Optional[str] = None) -> torch.device:
    if arg and arg != "auto":
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def cache_path_for(name: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / name


def load_or_build_token_vad_table(tokenizer, bert_name: str) -> torch.Tensor:
    """Cache the [V, 3] table per (tokenizer) so we don't rebuild every run."""
    safe_name = bert_name.replace("/", "_")
    p = cache_path_for(f"token_vad_{safe_name}.pt")
    if p.exists():
        return torch.load(p, map_location="cpu")
    nrc_path = PROJECT_ROOT / "NRC-VAD-Lexicon-v2.1.txt"
    if not nrc_path.exists():
        raise FileNotFoundError(
            f"Token-VAD requires {nrc_path}. Make sure it is committed and pulled."
        )
    table = build_token_vad_table(tokenizer, str(nrc_path))
    torch.save(table, p)
    return table


def build_label_matrix(hf_split) -> torch.Tensor:
    """[N, C] multi-hot label matrix from a HF split."""
    rows = []
    for row in hf_split:
        rows.append([float(row[lab]) for lab in GOEMOTIONS_LABELS])
    return torch.tensor(rows, dtype=torch.float32)


def maybe_subset(ds, n: Optional[int]):
    """Return a Subset of the first ``n`` examples; passthrough if n is None or larger."""
    if n is None:
        return ds
    if isinstance(ds, GoEmotionsDataset):
        n = min(n, len(ds))
        return Subset(ds, list(range(n)))
    return ds


# =============================================================================
# Loaders + model construction
# =============================================================================


def build_loaders(
    tokenizer,
    cfg: Dict[str, Any],
    smoke: bool,
) -> Tuple[DataLoader, DataLoader, DataLoader, Any]:
    ds = load_dataset("SetFit/go_emotions")
    train_ds = GoEmotionsDataset(ds["train"], tokenizer, cfg["max_length"])
    val_ds = GoEmotionsDataset(ds["validation"], tokenizer, cfg["max_length"])
    test_ds = GoEmotionsDataset(ds["test"], tokenizer, cfg["max_length"])
    if smoke:
        train_ds = maybe_subset(train_ds, 64)
        val_ds = maybe_subset(val_ds, 64)
        test_ds = maybe_subset(test_ds, 64)
    num_workers = 2 if not smoke else 0
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size_joint"],
        shuffle=True,
        num_workers=num_workers,
        collate_fn=goemotions_collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(cfg["batch_size_joint"], 32),
        shuffle=False,
        num_workers=num_workers,
        collate_fn=goemotions_collate_fn,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=max(cfg["batch_size_joint"], 32),
        shuffle=False,
        num_workers=num_workers,
        collate_fn=goemotions_collate_fn,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader, ds


def build_label_graph_inputs(
    cfg: Dict[str, Any],
    train_split,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (adj [C,C], label_init [C, init_dim])."""
    cmap = load_color_map(default_color_map_path())
    vad_table = _label_color_table(cmap)  # [C, 3]
    label_matrix = build_label_matrix(train_split)
    A_co = build_label_cooccurrence_adjacency(
        label_matrix,
        threshold=cfg["label_graph_co_threshold"],
        reweight=cfg["label_graph_co_reweight"],
    )
    if cfg["label_graph_use_vad_edges"]:
        A_vad = build_vad_adjacency(
            vad_table,
            sigma=cfg["label_graph_vad_sigma"],
            top_k=cfg["label_graph_vad_topk"],
        )
        A = combine_adjacencies(A_co, A_vad, alpha=cfg["label_graph_vad_alpha"])
    else:
        A = A_co
    # Initial label features = VAD vectors. The GCN expands them through gc1/gc2.
    return A, vad_table.clone()


def build_model(
    cfg: Dict[str, Any],
    tokenizer,
    train_split,
    device: torch.device,
) -> nn.Module:
    if cfg["model_type"] == "bert_only":
        model = BERTOnlyBaseline(bert_name=cfg["bert_name"], num_labels=NUM_LABELS)
        return model.to(device)

    model_kwargs: Dict[str, Any] = {
        "adj_temperature": cfg["adj_temperature"],
        "adj_topk": cfg["adj_topk"],
        "use_residual": cfg["use_residual"],
        "color_teacher_prob": cfg["color_teacher_prob"],
        "loss_type": cfg["loss_type"],
        "pos_weight_clip": cfg["pos_weight_clip"],
        "asl_gamma_pos": cfg["asl_gamma_pos"],
        "asl_gamma_neg": cfg["asl_gamma_neg"],
        "asl_clip": cfg["asl_clip"],
        "color_loss_weight": cfg["color_loss_weight"],
        "color_anchor_weight": cfg["color_anchor_weight"],
        "color_logit_scale_init": cfg["color_logit_scale_init"],
        "gcn_dropout": cfg["gcn_dropout"],
        "residual_scale_init": cfg["residual_scale_init"],
        "gcn_hidden": cfg["gcn_hidden"],
        # New flags:
        "disable_color": cfg["disable_color"],
        "use_logit_mix_vad": cfg["use_logit_mix_vad"],
        "use_token_vad": cfg["use_token_vad"],
        "gcn_type": cfg["gcn_type"],
        "gcn_n_layers": cfg["gcn_n_layers"],
        "gcn_internal_residual": cfg["gcn_internal_residual"],
        "label_graph_hidden": cfg["label_graph_hidden"],
        "label_graph_out": cfg["label_graph_out"],
        "supcon_weight": cfg["supcon_weight"],
        "supcon_temperature": cfg["supcon_temperature"],
    }
    if cfg["loss_type"] == "bce_weighted":
        model_kwargs["pos_weight"] = compute_pos_weight(train_split)
    if cfg["use_token_vad"] and not cfg["disable_color"]:
        model_kwargs["token_vad_table"] = load_or_build_token_vad_table(
            tokenizer, cfg["bert_name"]
        )
    if cfg["gcn_type"] == "label":
        adj, label_init = build_label_graph_inputs(cfg, train_split)
        model_kwargs["label_graph_adj"] = adj
        model_kwargs["label_graph_init"] = label_init

    model = EmotionColorGNNBERT(bert_name=cfg["bert_name"], **model_kwargs)
    return model.to(device)


# =============================================================================
# Training loops
# =============================================================================


def _epoch_progress_desc(prefix: str, epoch: int, total: int) -> str:
    return f"{prefix} {epoch + 1}/{total}"


def run_warmup_phase(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Dict[str, Any],
    device: torch.device,
    log_writer,
) -> None:
    """Train BERT + bert_head only (color/GCN frozen). Skipped if epochs_warmup == 0."""
    if cfg["epochs_warmup"] <= 0 or cfg["model_type"] == "bert_only":
        return
    assert isinstance(model, EmotionColorGNNBERT)
    freeze_gnn_for_warmup(model)
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    param_groups = _build_param_groups(
        named, base_lr=cfg["lr_warmup"], weight_decay=cfg["weight_decay"]
    )
    optimizer = torch.optim.AdamW(param_groups)
    total_steps = max(1, len(train_loader) * cfg["epochs_warmup"])
    warmup_steps = max(1, int(total_steps * cfg["warmup_ratio"]))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    for epoch in range(cfg["epochs_warmup"]):
        model.train()
        running = 0.0
        n = 0
        bar = tqdm(
            train_loader,
            desc=_epoch_progress_desc("warmup", epoch, cfg["epochs_warmup"]),
            mininterval=1.0,
            ncols=100,
        )
        for batch in bar:
            ids = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            y = batch["labels"].to(device, non_blocking=True)
            out = model(ids, mask, labels=y, bert_only=True)
            loss = out["loss"]
            optimizer.zero_grad()
            loss.backward()
            if cfg["max_grad_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    cfg["max_grad_norm"],
                )
            optimizer.step()
            scheduler.step()
            running += float(loss.item()) * ids.size(0)
            n += ids.size(0)
            bar.set_postfix(loss=running / max(n, 1))
        # Quick val loss
        model.eval()
        vloss = 0.0
        vn = 0
        with torch.no_grad():
            for batch in val_loader:
                ids = batch["input_ids"].to(device, non_blocking=True)
                mask = batch["attention_mask"].to(device, non_blocking=True)
                y = batch["labels"].to(device, non_blocking=True)
                out = model(ids, mask, labels=y, bert_only=True)
                vloss += float(out["loss"].item()) * ids.size(0)
                vn += ids.size(0)
        log_writer(
            f"[warmup epoch {epoch + 1}/{cfg['epochs_warmup']}] "
            f"train_loss={running / max(n, 1):.4f} val_loss={vloss / max(vn, 1):.4f}"
        )


def run_joint_phase(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Dict[str, Any],
    device: torch.device,
    log_writer,
) -> Dict[str, Any]:
    """Joint phase. Returns dict with best_state_dict, best_metrics, last_state_dict."""
    is_full = isinstance(model, EmotionColorGNNBERT)
    bert_param_ids: set = set()
    if is_full:
        freeze_for_joint(model)
        bert_params = unfreeze_top_bert_layers(model, cfg["n_top_bert_layers"])
        bert_param_ids = {id(p) for p in bert_params}
    else:
        # bert_only baseline: train all params at one LR
        for p in model.parameters():
            p.requires_grad = True

    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if is_full:
        param_groups = _build_param_groups(
            named,
            base_lr=cfg["lr_joint"],
            weight_decay=cfg["weight_decay"],
            bert_lr=cfg["bert_lr_joint"] if bert_param_ids else None,
        )
    else:
        param_groups = _build_param_groups(
            named, base_lr=cfg["lr_warmup"], weight_decay=cfg["weight_decay"]
        )
    optimizer = torch.optim.AdamW(param_groups)
    total_steps = max(1, len(train_loader) * cfg["epochs_joint"])
    warmup_steps = max(1, int(total_steps * cfg["warmup_ratio"]))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    best_f1 = -1.0
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch = -1

    bce_for_baseline = nn.BCEWithLogitsLoss()

    for epoch in range(cfg["epochs_joint"]):
        model.train()
        running = 0.0
        n = 0
        bar = tqdm(
            train_loader,
            desc=_epoch_progress_desc("joint", epoch, cfg["epochs_joint"]),
            mininterval=1.0,
            ncols=100,
        )
        for batch in bar:
            ids = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            y = batch["labels"].to(device, non_blocking=True)
            if is_full:
                out = model(ids, mask, labels=y, bert_only=False)
                loss = out["loss"]
            else:
                out = model(ids, mask)
                loss = bce_for_baseline(out["logits"], y)
            optimizer.zero_grad()
            loss.backward()
            if cfg["max_grad_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    cfg["max_grad_norm"],
                )
            optimizer.step()
            scheduler.step()
            running += float(loss.item()) * ids.size(0)
            n += ids.size(0)
            bar.set_postfix(loss=running / max(n, 1))
        # Eval at epoch end (val): default 0.5 micro-F1 for early-stop tracking.
        val_logits, val_labels = collect_logits(model, val_loader, device)
        val_metrics_05 = aggregate_metrics(val_logits, val_labels, thresholds=0.5)
        log_writer(
            f"[joint epoch {epoch + 1}/{cfg['epochs_joint']}] "
            f"train_loss={running / max(n, 1):.4f} "
            f"val(t=0.5) macro_f1={val_metrics_05['macro_f1']:.4f} "
            f"micro_f1={val_metrics_05['micro_f1']:.4f}"
        )
        if cfg["early_stop"] and val_metrics_05["micro_f1"] > best_f1:
            best_f1 = val_metrics_05["micro_f1"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    last_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if cfg["early_stop"] and best_state is not None:
        model.load_state_dict(best_state)
        log_writer(f"[joint] restored best state from epoch {best_epoch + 1} (val micro-F1={best_f1:.4f})")
    else:
        best_state = last_state
        best_epoch = cfg["epochs_joint"] - 1

    return {
        "best_state": best_state,
        "last_state": last_state,
        "best_epoch": best_epoch,
        "best_val_micro_f1_at_05": best_f1,
    }


@torch.no_grad()
def collect_logits(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run model over loader, return (logits[N,C], labels[N,C]) on CPU."""
    model.eval()
    logits_list = []
    labels_list = []
    for batch in loader:
        ids = batch["input_ids"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)
        y = batch["labels"]
        if isinstance(model, EmotionColorGNNBERT):
            out = model(ids, mask, labels=None, bert_only=False)
        else:
            out = model(ids, mask)
        logits_list.append(out["logits"].detach().cpu())
        labels_list.append(y)
    return torch.cat(logits_list, dim=0), torch.cat(labels_list, dim=0)


# =============================================================================
# Result writers
# =============================================================================


def write_per_label_csv(
    path: Path,
    per_label: Dict[str, list],
    macro: Dict[str, float],
) -> None:
    """Write a Demszky-table-style CSV: per-label P/R/F1/support/threshold + macro avg + std."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("label,precision,recall,f1,support,threshold\n")
        for i, lab in enumerate(GOEMOTIONS_LABELS):
            f.write(
                f"{lab},"
                f"{per_label['precision'][i]:.6f},"
                f"{per_label['recall'][i]:.6f},"
                f"{per_label['f1'][i]:.6f},"
                f"{per_label['support'][i]},"
                f"{per_label['thresholds'][i]:.4f}\n"
            )
        f.write(
            f"macro-average,{macro['macro_precision']:.6f},{macro['macro_recall']:.6f},{macro['macro_f1']:.6f},,\n"
        )
        f.write(
            f"std,{macro['macro_precision_std']:.6f},{macro['macro_recall_std']:.6f},{macro['macro_f1_std']:.6f},,\n"
        )


# =============================================================================
# Main
# =============================================================================


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Experiment name from experiments_config.EXPERIMENTS")
    parser.add_argument("--smoke", action="store_true", help="Tiny dataset, 1 epoch each phase")
    parser.add_argument("--force", action="store_true", help="Re-run even if results/<name>/summary.json exists")
    parser.add_argument("--device", default="auto", help="cuda | cpu | auto")
    parser.add_argument("--save-checkpoints", action="store_true", help="Save best.pt and last.pt")
    args = parser.parse_args()

    cfg = merged(args.config)
    name = cfg["name"]
    if args.smoke:
        cfg["epochs_warmup"] = min(cfg["epochs_warmup"], 1)
        cfg["epochs_joint"] = 1
        cfg["batch_size_warmup"] = 8
        cfg["batch_size_joint"] = 8
        name = f"smoke_{name}"
        cfg["name"] = name

    res_dir = RESULTS_DIR / name
    log_dir = LOGS_DIR / name
    ckpt_dir = CHECKPOINTS_DIR / name
    summary_path = res_dir / "summary.json"

    if summary_path.exists() and not args.force:
        print(f"[train_one] {name}: already complete ({summary_path}); use --force to re-run.")
        return 0

    res_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    if args.save_checkpoints:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    # File logger (in addition to stdout)
    log_path = log_dir / "train.log"
    log_file = open(log_path, "a", encoding="utf-8", buffering=1)

    def log_writer(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_file.write(line + "\n")

    set_seed(int(cfg["seed"]))
    device = get_device(args.device)
    log_writer(f"=== run: {name} ===")
    log_writer(f"device={device} cuda_available={torch.cuda.is_available()}")
    log_writer(f"cfg={json.dumps({k: v for k, v in cfg.items() if not isinstance(v, torch.Tensor)})}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["bert_name"])
    train_loader, val_loader, test_loader, ds = build_loaders(tokenizer, cfg, smoke=args.smoke)
    log_writer(
        f"data sizes: train={len(train_loader.dataset)} "
        f"val={len(val_loader.dataset)} test={len(test_loader.dataset)}"
    )

    t0 = time.time()
    model = build_model(cfg, tokenizer, ds["train"], device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_writer(f"model: total_params={n_params:,} trainable={n_trainable:,}")

    run_warmup_phase(model, train_loader, val_loader, cfg, device, log_writer)
    joint_out = run_joint_phase(model, train_loader, val_loader, cfg, device, log_writer)
    train_seconds = time.time() - t0
    log_writer(f"training done in {train_seconds:.1f}s")

    # ---- Evaluation ----
    val_logits, val_labels = collect_logits(model, val_loader, device)
    test_logits, test_labels = collect_logits(model, test_loader, device)

    # Tune per-label thresholds on validation, evaluate both splits with them.
    per_label_thresh = tune_thresholds_per_label(val_logits, val_labels)

    val_per_label_05 = per_label_prf(val_logits, val_labels, thresholds=0.5)
    val_per_label_t = per_label_prf(val_logits, val_labels, thresholds=per_label_thresh)
    test_per_label_05 = per_label_prf(test_logits, test_labels, thresholds=0.5)
    test_per_label_t = per_label_prf(test_logits, test_labels, thresholds=per_label_thresh)

    val_macro_05 = aggregate_metrics(val_logits, val_labels, thresholds=0.5)
    val_macro_t = aggregate_metrics(val_logits, val_labels, thresholds=per_label_thresh)
    test_macro_05 = aggregate_metrics(test_logits, test_labels, thresholds=0.5)
    test_macro_t = aggregate_metrics(test_logits, test_labels, thresholds=per_label_thresh)

    # Write per-label CSVs (Demszky-style table)
    write_per_label_csv(res_dir / "per_label_val_t05.csv", val_per_label_05, val_macro_05)
    write_per_label_csv(res_dir / "per_label_val_tuned.csv", val_per_label_t, val_macro_t)
    write_per_label_csv(res_dir / "per_label_test_t05.csv", test_per_label_05, test_macro_05)
    write_per_label_csv(res_dir / "per_label_test_tuned.csv", test_per_label_t, test_macro_t)

    summary = {
        "name": name,
        "config": cfg,
        "device": str(device),
        "train_seconds": train_seconds,
        "best_epoch_1based": joint_out["best_epoch"] + 1,
        "best_val_micro_f1_at_05_during_training": joint_out["best_val_micro_f1_at_05"],
        "val": {"t=0.5": val_macro_05, "tuned_per_label": val_macro_t},
        "test": {"t=0.5": test_macro_05, "tuned_per_label": test_macro_t},
        "per_label_thresholds": per_label_thresh.tolist(),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(res_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    log_writer(
        f"VAL  t=0.5  macro_f1={val_macro_05['macro_f1']:.4f} micro_f1={val_macro_05['micro_f1']:.4f} | "
        f"tuned   macro_f1={val_macro_t['macro_f1']:.4f} micro_f1={val_macro_t['micro_f1']:.4f}"
    )
    log_writer(
        f"TEST t=0.5  macro_f1={test_macro_05['macro_f1']:.4f} micro_f1={test_macro_05['micro_f1']:.4f} | "
        f"tuned   macro_f1={test_macro_t['macro_f1']:.4f} micro_f1={test_macro_t['micro_f1']:.4f}"
    )

    if args.save_checkpoints:
        torch.save(joint_out["best_state"], ckpt_dir / "best.pt")
        torch.save(joint_out["last_state"], ckpt_dir / "last.pt")
        log_writer(f"checkpoints saved to {ckpt_dir}")

    log_file.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
