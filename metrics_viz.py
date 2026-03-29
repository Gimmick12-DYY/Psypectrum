"""
Figures for validation metrics (e.g. output of ``run_full_pipeline``).

Requires matplotlib (see environment.yml).
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

METRIC_KEYS: Tuple[str, ...] = ("micro_precision", "micro_recall", "micro_f1")
METRIC_DISPLAY: Dict[str, str] = {
    "micro_precision": "Micro precision",
    "micro_recall": "Micro recall",
    "micro_f1": "Micro F1",
}
BRANCH_ORDER: Tuple[str, ...] = ("bert_color_gnn", "bert_gnn")
BRANCH_DISPLAY: Dict[str, str] = {
    "bert_color_gnn": "BERT + Color GNN",
    "bert_gnn": "BERT + GNN (no color)",
}

# Matplotlib tab10, first colors (stable across versions).
TAB10: Tuple[str, ...] = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)


def _ordered_branches(metrics: Dict[str, Dict[str, float]]) -> List[str]:
    out = [k for k in BRANCH_ORDER if k in metrics]
    for k in sorted(metrics.keys()):
        if k not in out:
            out.append(k)
    return out


def plot_validation_metrics(
    metrics: Dict[str, Dict[str, float]],
    *,
    save_path: Optional[str] = None,
    show: bool = False,
    suptitle: Optional[str] = None,
    dpi: int = 150,
) -> Optional[plt.Figure]:
    """
    One figure with three panels:

    1. Grouped bar chart: each metric with one bar per model branch.
    2. Delta: ``bert_color_gnn − bert_gnn`` when both keys exist (gain from color).
    3. Heatmap of the same numbers for quick comparison.

    Parameters
    ----------
    metrics
        Nested dicts like ``{"bert_color_gnn": {"micro_precision": ..., ...}, ...}``.
    save_path
        If set, write PNG to this path (parent dirs are created).
    show
        If True, call ``plt.show()`` (blocking in scripts).
    """
    branches = _ordered_branches(metrics)
    if not branches:
        raise ValueError("metrics dict is empty")

    metric_keys = [k for k in METRIC_KEYS if any(k in metrics[b] for b in branches)]
    if not metric_keys:
        raise ValueError("No known metric keys found in metrics")

    fig = plt.figure(figsize=(9, 10), layout="constrained")
    gs = fig.add_gridspec(3, 1, height_ratios=[1.15, 0.75, 1.0])
    ax_bar = fig.add_subplot(gs[0, 0])
    ax_delta = fig.add_subplot(gs[1, 0])
    ax_hm = fig.add_subplot(gs[2, 0])

    n_m = len(metric_keys)
    n_b = len(branches)
    x = np.arange(n_m, dtype=float)
    width = 0.8 / max(n_b, 1)

    for bi, branch in enumerate(branches):
        vals = [metrics[branch].get(mk, float("nan")) for mk in metric_keys]
        offset = (bi - (n_b - 1) / 2.0) * width
        label = BRANCH_DISPLAY.get(branch, branch)
        bars = ax_bar.bar(
            x + offset,
            vals,
            width,
            label=label,
            color=TAB10[bi % len(TAB10)],
            edgecolor="white",
            linewidth=0.6,
        )
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax_bar.text(
                    b.get_x() + b.get_width() / 2.0,
                    min(v + 0.02, 0.99),
                    f"{v:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([METRIC_DISPLAY.get(mk, mk) for mk in metric_keys])
    ax_bar.set_ylabel("Score")
    ax_bar.set_ylim(0, 1.05)
    ax_bar.axhline(1.0, color="0.85", linewidth=0.8, zorder=0)
    ax_bar.legend(loc="lower right", framealpha=0.92)
    ax_bar.set_title("Validation metrics (micro, threshold 0.5)")

    if "bert_color_gnn" in metrics and "bert_gnn" in metrics:
        deltas = [
            metrics["bert_color_gnn"].get(mk, float("nan"))
            - metrics["bert_gnn"].get(mk, float("nan"))
            for mk in metric_keys
        ]
        colors = ["#2ca02c" if d >= 0 else "#d62728" for d in deltas]
        y_pos = np.arange(len(metric_keys))
        ax_delta.barh(y_pos, deltas, color=colors, edgecolor="white", linewidth=0.6)
        ax_delta.axvline(0, color="0.4", linewidth=0.9)
        ax_delta.set_yticks(y_pos)
        ax_delta.set_yticklabels([METRIC_DISPLAY.get(mk, mk) for mk in metric_keys])
        ax_delta.set_xlabel("Δ score (BERT+Color GNN minus BERT+GNN)")
        ax_delta.set_title("Effect of the color module")
        finite_abs = [abs(d) for d in deltas if not np.isnan(d)]
        xmax = max(finite_abs) if finite_abs else 0.05
        pad = max(0.008, xmax * 0.06)
        for i, d in enumerate(deltas):
            if np.isnan(d):
                continue
            ax_delta.text(
                d + (pad if d >= 0 else -pad),
                i,
                f"{d:+.4f}",
                va="center",
                ha="left" if d >= 0 else "right",
                fontsize=8,
            )
    else:
        ax_delta.text(0.5, 0.5, "Need both bert_color_gnn and bert_gnn for delta plot", ha="center", va="center", transform=ax_delta.transAxes, color="0.45")
        ax_delta.set_axis_off()

    mat = np.array([[metrics[b].get(mk, float("nan")) for mk in metric_keys] for b in branches])
    im = ax_hm.imshow(mat, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax_hm.set_xticks(np.arange(len(metric_keys)))
    ax_hm.set_xticklabels([METRIC_DISPLAY.get(mk, mk) for mk in metric_keys])
    ax_hm.set_yticks(np.arange(len(branches)))
    ax_hm.set_yticklabels([BRANCH_DISPLAY.get(b, b) for b in branches])
    for i in range(len(branches)):
        for j in range(len(metric_keys)):
            v = mat[i, j]
            if not np.isnan(v):
                ax_hm.text(j, i, f"{v:.3f}", ha="center", va="center", color="0.1" if v > 0.5 else "white", fontsize=10)
    ax_hm.set_title("Heatmap (same values)")
    fig.colorbar(im, ax=ax_hm, shrink=0.5, label="Score")

    if suptitle:
        fig.suptitle(suptitle, fontsize=12, fontweight="medium")

    if save_path:
        parent = os.path.dirname(os.path.abspath(save_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
        return fig
    if save_path:
        plt.close(fig)
        return None
    return fig


def save_pipeline_metrics_figure(
    metrics: Dict[str, Dict[str, float]],
    path: str,
    *,
    suptitle: Optional[str] = None,
    dpi: int = 150,
) -> None:
    """Convenience wrapper: save a single PNG and close the figure."""
    plot_validation_metrics(metrics, save_path=path, show=False, suptitle=suptitle, dpi=dpi)
