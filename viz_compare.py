"""
After all experiments finish (or any subset), build comparison artifacts:

  results/comparison_summary.csv     # one row per run, val + test macro/micro
  results/comparison_bars.png        # grouped bars: macro & micro F1 per run
  results/per_label_heatmap.png      # 28 labels x runs, F1 on test (tuned)
  results/<run>/per_label_table.png  # Demszky-style P/R/F1 figure per run

Reads ``results/<run>/summary.json`` for each completed run.

Usage:
    python viz_compare.py
    python viz_compare.py --smoke      # only consider smoke_* runs
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from palette_model import GOEMOTIONS_LABELS

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"


# =============================================================================
# Load all completed runs
# =============================================================================


def load_summaries(smoke: bool) -> List[Dict[str, Any]]:
    summaries = []
    if not RESULTS_DIR.exists():
        return summaries
    for sub in sorted(RESULTS_DIR.iterdir()):
        if not sub.is_dir():
            continue
        if smoke and not sub.name.startswith("smoke_"):
            continue
        if not smoke and sub.name.startswith("smoke_"):
            continue
        sp = sub / "summary.json"
        if not sp.exists():
            continue
        with open(sp, "r", encoding="utf-8") as f:
            summary = json.load(f)
        summaries.append(summary)
    return summaries


def load_per_label(run_name: str, split: str, mode: str) -> List[Dict[str, Any]]:
    """mode: 't05' or 'tuned'."""
    p = RESULTS_DIR / run_name / f"per_label_{split}_{mode}.csv"
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# =============================================================================
# Comparison summary CSV
# =============================================================================


def write_comparison_csv(summaries: List[Dict[str, Any]], path: Path) -> None:
    fields = [
        "name",
        "val_t05_macro_p", "val_t05_macro_r", "val_t05_macro_f1",
        "val_t05_micro_p", "val_t05_micro_r", "val_t05_micro_f1",
        "val_tuned_macro_p", "val_tuned_macro_r", "val_tuned_macro_f1",
        "val_tuned_micro_p", "val_tuned_micro_r", "val_tuned_micro_f1",
        "test_t05_macro_p", "test_t05_macro_r", "test_t05_macro_f1",
        "test_t05_micro_p", "test_t05_micro_r", "test_t05_micro_f1",
        "test_tuned_macro_p", "test_tuned_macro_r", "test_tuned_macro_f1",
        "test_tuned_micro_p", "test_tuned_micro_r", "test_tuned_micro_f1",
        "best_epoch", "train_seconds",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in summaries:
            row = {
                "name": s["name"],
                "best_epoch": s.get("best_epoch_1based", ""),
                "train_seconds": f"{s.get('train_seconds', 0.0):.1f}",
            }
            for split in ("val", "test"):
                for mode_key, mode_label in (("t=0.5", "t05"), ("tuned_per_label", "tuned")):
                    m = s.get(split, {}).get(mode_key, {})
                    row[f"{split}_{mode_label}_macro_p"] = f"{m.get('macro_precision', 0):.4f}"
                    row[f"{split}_{mode_label}_macro_r"] = f"{m.get('macro_recall', 0):.4f}"
                    row[f"{split}_{mode_label}_macro_f1"] = f"{m.get('macro_f1', 0):.4f}"
                    row[f"{split}_{mode_label}_micro_p"] = f"{m.get('micro_precision', 0):.4f}"
                    row[f"{split}_{mode_label}_micro_r"] = f"{m.get('micro_recall', 0):.4f}"
                    row[f"{split}_{mode_label}_micro_f1"] = f"{m.get('micro_f1', 0):.4f}"
            w.writerow(row)
    print(f"Wrote {path}")


# =============================================================================
# Demszky-style per-label table (one figure per run)
# =============================================================================


def render_per_label_table(run_name: str, split: str, mode: str, title_suffix: str = "") -> None:
    rows = load_per_label(run_name, split, mode)
    if not rows:
        return
    out_path = RESULTS_DIR / run_name / f"per_label_table_{split}_{mode}.png"

    label_rows = [r for r in rows if r["label"] in set(GOEMOTIONS_LABELS)]
    summary_rows = [r for r in rows if r["label"] in ("macro-average", "std")]

    table_data = [["Emotion", "Precision", "Recall", "F1", "Support"]]
    for r in label_rows:
        table_data.append([
            r["label"],
            f"{float(r['precision']):.2f}",
            f"{float(r['recall']):.2f}",
            f"{float(r['f1']):.2f}",
            r["support"],
        ])
    for r in summary_rows:
        table_data.append([
            r["label"],
            f"{float(r['precision']):.2f}",
            f"{float(r['recall']):.2f}",
            f"{float(r['f1']):.2f}",
            "",
        ])

    n_rows = len(table_data)
    fig_h = 0.30 * n_rows + 1.0
    fig, ax = plt.subplots(figsize=(6.5, fig_h))
    ax.axis("off")
    table = ax.table(cellText=table_data, loc="center", cellLoc="center", colWidths=[0.32, 0.17, 0.17, 0.17, 0.17])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.18)
    # Style header
    for j in range(len(table_data[0])):
        cell = table[(0, j)]
        cell.set_text_props(fontweight="bold")
        cell.set_facecolor("#dddddd")
    # Style summary rows (last two)
    for i in (n_rows - 2, n_rows - 1):
        for j in range(len(table_data[0])):
            table[(i, j)].set_text_props(fontweight="bold")
            table[(i, j)].set_facecolor("#f2f2f2")
    title = f"{run_name} — {split} ({mode})"
    if title_suffix:
        title += f" {title_suffix}"
    ax.set_title(title, fontsize=11, pad=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


# =============================================================================
# Comparison bars across runs
# =============================================================================


def render_comparison_bars(summaries: List[Dict[str, Any]], path: Path) -> None:
    if not summaries:
        return
    names = [s["name"] for s in summaries]

    # Plot test macro F1 and micro F1, both at t=0.5 and tuned-per-label.
    metrics = [
        ("test t=0.5 macro F1", "test", "t=0.5", "macro_f1"),
        ("test tuned macro F1", "test", "tuned_per_label", "macro_f1"),
        ("test t=0.5 micro F1", "test", "t=0.5", "micro_f1"),
        ("test tuned micro F1", "test", "tuned_per_label", "micro_f1"),
    ]

    n_runs = len(names)
    n_metrics = len(metrics)
    width = 0.8 / max(n_metrics, 1)
    x = np.arange(n_runs, dtype=float)

    fig, ax = plt.subplots(figsize=(max(8, 1.1 * n_runs), 5.5))
    palette = plt.get_cmap("tab10").colors
    for mi, (label, split, mode, key) in enumerate(metrics):
        vals = [s[split][mode].get(key, float("nan")) for s in summaries]
        offset = (mi - (n_metrics - 1) / 2.0) * width
        bars = ax.bar(x + offset, vals, width, label=label, color=palette[mi % 10])
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, min(v + 0.005, 0.99), f"{v:.3f}",
                        ha="center", va="bottom", fontsize=7, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("F1 score")
    ax.set_title("Test-set F1 across experiments")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.92)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


# =============================================================================
# Per-label F1 heatmap (labels x runs)
# =============================================================================


def render_per_label_heatmap(summaries: List[Dict[str, Any]], path: Path) -> None:
    if not summaries:
        return
    names = [s["name"] for s in summaries]
    mat = np.full((len(GOEMOTIONS_LABELS), len(names)), np.nan, dtype=float)
    for ci, s in enumerate(summaries):
        rows = load_per_label(s["name"], "test", "tuned")
        if not rows:
            continue
        idx_by_label = {r["label"]: r for r in rows}
        for ri, lab in enumerate(GOEMOTIONS_LABELS):
            r = idx_by_label.get(lab)
            if r is not None:
                try:
                    mat[ri, ci] = float(r["f1"])
                except ValueError:
                    pass
    fig, ax = plt.subplots(figsize=(max(7, 0.85 * len(names)), 0.32 * len(GOEMOTIONS_LABELS) + 1.5))
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(GOEMOTIONS_LABELS)))
    ax.set_yticklabels(GOEMOTIONS_LABELS, fontsize=9)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if not np.isnan(v):
                color = "white" if v < 0.5 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color=color)
    ax.set_title("Per-label F1 on test (tuned thresholds)")
    fig.colorbar(im, ax=ax, shrink=0.6, label="F1")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


# =============================================================================
# Main
# =============================================================================


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Use smoke_* runs")
    args = parser.parse_args()

    summaries = load_summaries(smoke=args.smoke)
    if not summaries:
        print(f"No completed runs found under {RESULTS_DIR}.")
        return 1
    print(f"Found {len(summaries)} completed run(s): {[s['name'] for s in summaries]}")

    write_comparison_csv(summaries, RESULTS_DIR / "comparison_summary.csv")
    render_comparison_bars(summaries, RESULTS_DIR / "comparison_bars.png")
    render_per_label_heatmap(summaries, RESULTS_DIR / "per_label_heatmap.png")

    for s in summaries:
        for split in ("val", "test"):
            for mode in ("t05", "tuned"):
                render_per_label_table(s["name"], split, mode)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
