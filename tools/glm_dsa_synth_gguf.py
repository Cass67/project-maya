"""tools/glm_dsa_synth_gguf.py - a tiny, structurally complete synthetic glm-dsa (GLM-5.3) GGUF.

The glm-dsa counterpart of tools/glm_synth_gguf.py: the tensor set and metadata keys llama.cpp's
glm-dsa loader reads, dimensions scaled down, values from a seeded RNG.  Every path of
ref/glm_dsa.py is exercised: a dense lead layer, MoE + shared expert, roped MLA, full AND shared
indexer layers (the indexer types array), and an indexer top_k small enough that 24-token
sequences discard keys.  The indexer key width is 64, the smallest Walsh-Hadamard size llama.cpp
builds, so `--vocab` output loads in upstream llama.cpp for the parity check.

Usage:
    python tools/glm_dsa_synth_gguf.py --out glm-dsa-synth.gguf [--seed 4321] [--vocab]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from glm_synth_gguf import ALIGN, arr_f32, arr_i32, arr_str, kv_bool, kv_f32, kv_str, kv_u32, write_gguf  # noqa: E402

SMALL = dict(n_embd=128, n_vocab=512, n_layers=6, n_head=4, qk_nope=32, n_rot=16, v_head=32,
             q_lora=64, kv_lora=64, n_expert=16, n_exp_used=4, n_ff_exp=32, n_shared=1,
             n_ff_dense=192, dense_lead=1, idx_heads=4, idx_dim=64, idx_topk=8,
             idx_types=[1, 1, 0, 1, 0, 0], freq_base=10000.0, ctx=4096)


def build(seed: int = 4321, vocab: bool = False) -> tuple[list, list]:
    g = SMALL
    n_embd, n_vocab, n_layers, n_head = g["n_embd"], g["n_vocab"], g["n_layers"], g["n_head"]
    qk_nope, n_rot, v_head = g["qk_nope"], g["n_rot"], g["v_head"]
    k_mla = qk_nope + n_rot
    q_lora, kv_lora = g["q_lora"], g["kv_lora"]
    n_expert, n_ff_exp, n_shared = g["n_expert"], g["n_ff_exp"], g["n_shared"]
    idx_heads, idx_dim = g["idx_heads"], g["idx_dim"]
    rng = np.random.default_rng(seed)

    def w(*shape, scale=0.05):
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    def norm_w(n):
        return (1.0 + rng.standard_normal(n) * 0.02).astype(np.float32)

    a = "glm-dsa."
    md = [
        ("general.architecture", kv_str("glm-dsa")),
        ("general.name", kv_str("glm-dsa-synth")),
        ("general.alignment", kv_u32(ALIGN)),
        (a + "block_count", kv_u32(n_layers)),
        (a + "context_length", kv_u32(g["ctx"])),
        (a + "embedding_length", kv_u32(n_embd)),
        (a + "feed_forward_length", kv_u32(g["n_ff_dense"])),
        (a + "vocab_size", kv_u32(n_vocab)),
        (a + "attention.head_count", kv_u32(n_head)),
        (a + "attention.head_count_kv", kv_u32(1)),
        (a + "attention.key_length", kv_u32(kv_lora + n_rot)),
        (a + "attention.value_length", kv_u32(kv_lora)),
        (a + "attention.key_length_mla", kv_u32(k_mla)),
        (a + "attention.value_length_mla", kv_u32(v_head)),
        (a + "attention.q_lora_rank", kv_u32(q_lora)),
        (a + "attention.kv_lora_rank", kv_u32(kv_lora)),
        (a + "attention.layer_norm_rms_epsilon", kv_f32(1e-5)),
        (a + "rope.freq_base", kv_f32(g["freq_base"])),
        (a + "rope.dimension_count", kv_u32(n_rot)),
        (a + "expert_count", kv_u32(n_expert)),
        (a + "expert_used_count", kv_u32(g["n_exp_used"])),
        (a + "expert_group_count", kv_u32(1)),
        (a + "expert_group_used_count", kv_u32(1)),
        (a + "expert_gating_func", kv_u32(2)),
        (a + "expert_feed_forward_length", kv_u32(n_ff_exp)),
        (a + "expert_shared_count", kv_u32(n_shared)),
        (a + "expert_weights_scale", kv_f32(2.5)),
        (a + "expert_weights_norm", kv_bool(True)),
        (a + "leading_dense_block_count", kv_u32(g["dense_lead"])),
        (a + "nextn_predict_layers", kv_u32(0)),
        (a + "attention.indexer.head_count", kv_u32(idx_heads)),
        (a + "attention.indexer.key_length", kv_u32(idx_dim)),
        (a + "attention.indexer.top_k", kv_u32(g["idx_topk"])),
        (a + "attention.indexer.types", arr_i32(g["idx_types"])),
    ]
    if vocab:
        md += [
            ("tokenizer.ggml.model", kv_str("gpt2")),
            ("tokenizer.ggml.pre", kv_str("glm5")),
            ("tokenizer.ggml.tokens", arr_str(["tok%d" % i for i in range(n_vocab)])),
            ("tokenizer.ggml.scores", arr_f32([0.0] * n_vocab)),
            ("tokenizer.ggml.token_type", arr_i32([1] * n_vocab)),
            ("tokenizer.ggml.merges", arr_str([])),
        ]

    ts = [
        ("token_embd.weight", w(n_vocab, n_embd, scale=1.0)),
        ("output_norm.weight", norm_w(n_embd)),
        ("output.weight", w(n_vocab, n_embd)),
    ]
    for il in range(n_layers):
        p = f"blk.{il}."
        ts += [
            (p + "attn_norm.weight", norm_w(n_embd)),
            (p + "ffn_norm.weight", norm_w(n_embd)),
            (p + "attn_q_a_norm.weight", norm_w(q_lora)),
            (p + "attn_kv_a_norm.weight", norm_w(kv_lora)),
            (p + "attn_q_a.weight", w(q_lora, n_embd)),
            (p + "attn_q_b.weight", w(n_head * k_mla, q_lora, scale=0.2)),
            (p + "attn_kv_a_mqa.weight", w(kv_lora + n_rot, n_embd, scale=0.2)),
            (p + "attn_k_b.weight", w(n_head, kv_lora, qk_nope, scale=0.2)),
            (p + "attn_v_b.weight", w(n_head, v_head, kv_lora)),
            (p + "attn_output.weight", w(n_embd, n_head * v_head)),
            # every layer carries indexer weights (the real GGUF does); shared layers ignore them
            (p + "indexer.k_norm.weight", norm_w(idx_dim)),
            (p + "indexer.k_norm.bias", w(idx_dim)),
            (p + "indexer.proj.weight", w(idx_heads, n_embd, scale=0.5)),
            (p + "indexer.attn_k.weight", w(idx_dim, n_embd, scale=0.2)),
            (p + "indexer.attn_q_b.weight", w(idx_heads * idx_dim, q_lora, scale=0.2)),
        ]
        if il < g["dense_lead"]:
            ts += [
                (p + "ffn_gate.weight", w(g["n_ff_dense"], n_embd)),
                (p + "ffn_down.weight", w(n_embd, g["n_ff_dense"])),
                (p + "ffn_up.weight", w(g["n_ff_dense"], n_embd)),
            ]
        else:
            ts += [
                (p + "ffn_gate_inp.weight", w(n_expert, n_embd, scale=0.2)),
                (p + "exp_probs_b.bias", w(n_expert, scale=0.1)),
                (p + "ffn_gate_exps.weight", w(n_expert, n_ff_exp, n_embd)),
                (p + "ffn_down_exps.weight", w(n_expert, n_embd, n_ff_exp)),
                (p + "ffn_up_exps.weight", w(n_expert, n_ff_exp, n_embd)),
                (p + "ffn_gate_shexp.weight", w(n_ff_exp * n_shared, n_embd)),
                (p + "ffn_down_shexp.weight", w(n_embd, n_ff_exp * n_shared)),
                (p + "ffn_up_shexp.weight", w(n_ff_exp * n_shared, n_embd)),
            ]
    return md, ts


def selftest(path: Path) -> None:
    from gguf_reader import GGUFFile
    gf = GGUFFile(path)
    by = {t.name: t for t in gf.tensors}
    assert gf.metadata["general.architecture"] == "glm-dsa", "architecture key lost"
    assert gf.metadata["glm-dsa.attention.indexer.types"] == SMALL["idx_types"], "indexer types mangled"
    assert list(by["blk.1.attn_k_b.weight"].shape) == [32, 64, 4], f"k_b shape {by['blk.1.attn_k_b.weight'].shape}"
    assert list(by["blk.1.attn_kv_a_mqa.weight"].shape) == [128, 80], "kv_a_mqa is kv_lora + rope wide"
    assert list(by["blk.1.ffn_gate_exps.weight"].shape) == [128, 32, 16], "expert shape wrong"
    assert "blk.0.ffn_gate.weight" in by and "blk.0.ffn_gate_exps.weight" not in by, "dense lead wrong"
    print(f"selftest ok: {len(gf.tensors)} tensors, {path.stat().st_size / 1e6:.2f} MB")


def main() -> int:
    ap = argparse.ArgumentParser(description="write a tiny synthetic glm-dsa GGUF")
    ap.add_argument("--out", default="glm-dsa-synth.gguf")
    ap.add_argument("--seed", type=int, default=4321)
    ap.add_argument("--vocab", action="store_true",
                    help="add the tokenizer metadata upstream llama.cpp needs to LOAD the model")
    args = ap.parse_args()
    md, ts = build(args.seed, vocab=args.vocab)
    out = Path(args.out)
    write_gguf(out, md, ts)
    selftest(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
