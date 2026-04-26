"""
CPU-only smoke test. Verifies:
  1. All new modules import cleanly
  2. model_extensions math is correct on synthetic data
  3. Each EmotionColorGNNBERT variant constructs and does forward+backward
     using a tiny mocked BertModel (no HF download)

Run:
    python smoke_test.py
"""

from __future__ import annotations

import sys
import traceback
from typing import Any, Dict

import torch
import torch.nn as nn

# ---- 1) Import sanity ----
print("[smoke] importing modules...")
import model_extensions  # noqa: F401
import experiments_config
import palette_model
import train_one  # noqa: F401
import viz_compare  # noqa: F401

print(f"  experiments_config has {len(experiments_config.EXPERIMENTS)} configs:",
      list(experiments_config.EXPERIMENTS))
print(f"  palette_model.NUM_LABELS = {palette_model.NUM_LABELS}")


# ---- 2) Per-label metrics + threshold tuning ----
print("\n[smoke] testing per-label metrics...")
torch.manual_seed(0)
N, C = 200, palette_model.NUM_LABELS
logits = torch.randn(N, C)
labels = (torch.rand(N, C) > 0.85).float()  # sparse multi-hot

per_label = model_extensions.per_label_prf(logits, labels, thresholds=0.5)
assert len(per_label["precision"]) == C
assert len(per_label["recall"]) == C
assert len(per_label["f1"]) == C
assert len(per_label["support"]) == C
assert len(per_label["thresholds"]) == C

agg = model_extensions.aggregate_metrics(logits, labels, thresholds=0.5)
for k in ("macro_precision", "macro_recall", "macro_f1",
          "micro_precision", "micro_recall", "micro_f1",
          "macro_precision_std", "macro_recall_std", "macro_f1_std"):
    assert k in agg, f"missing {k}"
print(f"  agg t=0.5: {agg}")

per_label_thr = model_extensions.tune_thresholds_per_label(logits, labels)
assert per_label_thr.shape == (C,)
agg_tuned = model_extensions.aggregate_metrics(logits, labels, thresholds=per_label_thr)
print(f"  agg tuned: macro_f1={agg_tuned['macro_f1']:.4f}  micro_f1={agg_tuned['micro_f1']:.4f}")
assert agg_tuned["macro_f1"] >= agg["macro_f1"] - 1e-6, "tuned should not be worse than t=0.5"


# ---- 3) Token-VAD encoder (synthetic table; skip lexicon parse) ----
print("\n[smoke] testing TokenVADEncoder...")
fake_table = torch.randn(100, 3)
enc = model_extensions.TokenVADEncoder(fake_table)
ids = torch.randint(0, 100, (4, 16))
mask = torch.ones_like(ids)
mask[:, 10:] = 0
out = enc(ids, mask)
assert out.shape == (4, 6), f"expected (4, 6) got {tuple(out.shape)}"
print(f"  output shape OK: {tuple(out.shape)}")


# ---- 4) Label-graph builders + GCN ----
print("\n[smoke] testing label-graph...")
fake_label_matrix = (torch.rand(500, C) > 0.85).float()
A_co = model_extensions.build_label_cooccurrence_adjacency(fake_label_matrix)
assert A_co.shape == (C, C)
vad_table = palette_model._label_color_table(palette_model.load_color_map(palette_model.default_color_map_path()))
A_vad = model_extensions.build_vad_adjacency(vad_table)
assert A_vad.shape == (C, C)
A = model_extensions.combine_adjacencies(A_co, A_vad, alpha=0.5)
assert A.shape == (C, C)

gcn = model_extensions.LabelGraphGCN(label_init=vad_table.clone(), adj=A, hidden_dim=32, out_dim=64)
W = gcn()
assert W.shape == (C, 64), f"expected ({C}, 64) got {tuple(W.shape)}"
print(f"  W shape OK: {tuple(W.shape)}")


# ---- 5) SupCon loss ----
print("\n[smoke] testing supcon loss...")
feats = torch.randn(8, 3, requires_grad=True)
labs = (torch.rand(8, C) > 0.7).float()
labs[0] = 0  # one example with no positives, triggers has_pos branch
loss = model_extensions.supervised_contrastive_loss(feats, labs, temperature=0.1)
loss.backward()
assert torch.isfinite(loss).item(), f"loss not finite: {loss.item()}"
assert feats.grad is not None
print(f"  loss={loss.item():.4f}, grad OK")


# ---- 6) EmotionColorGNNBERT construction + forward/backward, all variants ----
print("\n[smoke] testing EmotionColorGNNBERT variants with mocked BertModel...")
from transformers import BertConfig, BertModel
import transformers

# Tiny BERT config (no download). Match the embedding vocab to our token table.
SMALL_VOCAB = 64
SMALL_HIDDEN = 64
small_cfg = BertConfig(
    vocab_size=SMALL_VOCAB,
    hidden_size=SMALL_HIDDEN,
    num_hidden_layers=2,
    num_attention_heads=2,
    intermediate_size=128,
    max_position_embeddings=64,
    type_vocab_size=2,
)


class _FakeAutoModel:
    @staticmethod
    def from_pretrained(name, *args, **kwargs):
        return BertModel(small_cfg)


# Patch palette_model's reference to AutoModel
_orig_auto = palette_model.AutoModel
palette_model.AutoModel = _FakeAutoModel  # type: ignore

try:
    # Helper: synthetic batch
    B, L = 6, 16
    ids = torch.randint(0, SMALL_VOCAB, (B, L))
    mask = torch.ones_like(ids)
    y = (torch.rand(B, C) > 0.7).float()

    # pos_weight for bce_weighted variants
    pos_w = torch.ones(C) * 2.0

    # Token VAD lookup table sized to small vocab
    tok_vad_table = torch.randn(SMALL_VOCAB, 3)

    # Label graph inputs (real builders, real shapes)
    label_matrix = (torch.rand(200, C) > 0.85).float()
    A_co_small = model_extensions.build_label_cooccurrence_adjacency(label_matrix)
    A_vad_small = model_extensions.build_vad_adjacency(vad_table)
    A_small = model_extensions.combine_adjacencies(A_co_small, A_vad_small, 0.5)

    variants = [
        ("02_current_full", dict(
            disable_color=False, use_logit_mix_vad=True, use_token_vad=False,
            gcn_type="batch", gcn_n_layers=2, loss_type="bce_weighted", pos_weight=pos_w,
        )),
        ("03_current_nocolor", dict(
            disable_color=True, gcn_type="batch", gcn_n_layers=2,
            loss_type="bce_weighted", pos_weight=pos_w,
        )),
        ("04_tokenvad_full", dict(
            disable_color=False, use_logit_mix_vad=True, use_token_vad=True,
            token_vad_table=tok_vad_table, gcn_type="batch", gcn_n_layers=2,
            loss_type="bce_weighted", pos_weight=pos_w,
        )),
        ("05_tokenvad_nocolor", dict(
            disable_color=True, gcn_type="batch", gcn_n_layers=2,
            loss_type="bce_weighted", pos_weight=pos_w,
        )),
        ("06_tokenvad_sparse_gcn", dict(
            disable_color=False, use_logit_mix_vad=True, use_token_vad=True,
            token_vad_table=tok_vad_table, gcn_type="batch", gcn_n_layers=1,
            adj_topk=4, loss_type="bce_weighted", pos_weight=pos_w,
        )),
        ("07_labelgraph_full", dict(
            disable_color=False, use_logit_mix_vad=True, use_token_vad=True,
            token_vad_table=tok_vad_table, gcn_type="label",
            label_graph_adj=A_small, label_graph_init=vad_table.clone(),
            label_graph_hidden=64, label_graph_out=64,
            loss_type="bce_weighted", pos_weight=pos_w,
        )),
        ("08_labelgraph_nocolor", dict(
            disable_color=True, gcn_type="label",
            label_graph_adj=A_co_small, label_graph_init=vad_table.clone(),
            label_graph_hidden=64, label_graph_out=64,
            loss_type="bce_weighted", pos_weight=pos_w,
        )),
        ("09_tokenvad_contrastive", dict(
            disable_color=False, use_logit_mix_vad=True, use_token_vad=True,
            token_vad_table=tok_vad_table, gcn_type="batch", gcn_n_layers=2,
            supcon_weight=0.1, supcon_temperature=0.1,
            loss_type="bce_weighted", pos_weight=pos_w,
        )),
        ("10_tokenvad_asl", dict(
            disable_color=False, use_logit_mix_vad=True, use_token_vad=True,
            token_vad_table=tok_vad_table, gcn_type="batch", gcn_n_layers=2,
            loss_type="asl",
        )),
    ]

    for vname, kw in variants:
        print(f"  [{vname}]")
        try:
            model = palette_model.EmotionColorGNNBERT(bert_name="ignored", **kw)
        except Exception as e:
            print(f"    CONSTRUCT FAIL: {type(e).__name__}: {e}")
            traceback.print_exc()
            sys.exit(2)
        model.train()
        # Verify freeze functions don't crash for either phase
        try:
            palette_model.freeze_gnn_for_warmup(model)
            palette_model.freeze_for_joint(model)
        except Exception as e:
            print(f"    FREEZE FAIL: {e}")
            traceback.print_exc()
            sys.exit(2)
        # Joint-phase forward
        try:
            out = model(ids, mask, labels=y, bert_only=False)
        except Exception as e:
            print(f"    FORWARD FAIL: {e}")
            traceback.print_exc()
            sys.exit(2)
        assert out["logits"].shape == (B, C), f"expected ({B}, {C}) got {tuple(out['logits'].shape)}"
        assert torch.isfinite(out["loss"]).item(), f"loss not finite: {out['loss'].item()}"
        try:
            out["loss"].backward()
        except Exception as e:
            print(f"    BACKWARD FAIL: {e}")
            traceback.print_exc()
            sys.exit(2)
        # bert-only path (used in warmup)
        out_b = model(ids, mask, labels=y, bert_only=True)
        assert out_b["logits"].shape == (B, C)
        # use_color=False forward (still legal even when disable_color=False)
        out_nc = model(ids, mask, labels=y, bert_only=False, use_color=False)
        assert out_nc["logits"].shape == (B, C)
        print(f"    OK  loss={out['loss'].item():.4f} logits_shape={tuple(out['logits'].shape)}")

    # BERTOnlyBaseline
    print("  [01_bert_only / BERTOnlyBaseline]")
    baseline = palette_model.BERTOnlyBaseline(bert_name="ignored")
    baseline.train()
    out_bl = baseline(ids, mask)
    assert out_bl["logits"].shape == (B, C)
    bce = nn.BCEWithLogitsLoss()
    bl_loss = bce(out_bl["logits"], y)
    bl_loss.backward()
    print(f"    OK  loss={bl_loss.item():.4f}")

finally:
    palette_model.AutoModel = _orig_auto  # type: ignore

print("\n[smoke] ALL CHECKS PASSED")
