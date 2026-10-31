"""tools/ml4_synth_gguf.py - a tiny, structurally complete synthetic mistral-large4 GGUF.

The mistral-large4 counterpart of tools/glm_dsa_synth_gguf.py: the tensor set and metadata keys
llama.cpp's mistral-large4 loader reads, dimensions scaled down, values from a seeded RNG.  Every
path of ref/ml4.py runs: a dense lead layer, MoE + shared expert, roped and NoPE layers, a layer
reading another's KV (kv_source_map), full and shared indexers, SSMax, YaRN with a ramp inside the
rotated dims and LeRoPE factors, the logit softcap, and an indexer top_k small enough that 24-token
sequences discard keys.  The indexer key width is 64, the smallest Hadamard llama.cpp builds, so
`--vocab` output loads in upstream llama.cpp for the parity check.

Usage:
    python tools/ml4_synth_gguf.py --out ml4-synth.gguf [--seed 4336] [--vocab]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from glm_synth_gguf import ALIGN, arr_f32, arr_i32, arr_str, kv_bool, kv_f32, kv_str, kv_u32, write_gguf  # noqa: E402

SMALL = dict(n_embd=128, n_vocab=512, n_layers=8, n_head=4, k_mla=32, n_rot=16, v_head=32,
             kv_lora=64, n_expert=16, n_exp_used=4, n_ff_exp=32, n_ff_dense=192, dense_lead=1,
             idx_heads=4, idx_dim=64, idx_topk=8,
             idx_types=[1, 1, 0, 1, 0, 1, 0, 0], rope_pattern=[1, 1, 1, 0, 1, 1, 1, 0],
             kv_source=[0, 1, 2, 3, 4, 5, 6, 3],
             freq_base=10000.0, yarn_factor=16.0, yarn_orig_ctx=64, softcap=30.0, ctx=4096)


def build(seed: int = 4336, vocab: bool = False) -> tuple[list, list]:
    g = SMALL
    n_embd, n_vocab, n_layers, n_head = g["n_embd"], g["n_vocab"], g["n_layers"], g["n_head"]
    k_mla, n_rot, v_head, kv_lora = g["k_mla"], g["n_rot"], g["v_head"], g["kv_lora"]
    n_expert, n_ff_exp = g["n_expert"], g["n_ff_exp"]
    idx_heads, idx_dim = g["idx_heads"], g["idx_dim"]
    rng = np.random.default_rng(seed)

    def w(*shape, scale=0.05):
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    def norm_w(n):
        return (1.0 + rng.standard_normal(n) * 0.02).astype(np.float32)

    a = "mistral-large4."
    md = [
        ("general.architecture", kv_str("mistral-large4")),
        ("general.name", kv_str("ml4-synth")),
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
        (a + "attention.kv_lora_rank", kv_u32(kv_lora)),
        (a + "attention.layer_norm_rms_epsilon", kv_f32(1e-5)),
        (a + "attention.layer_norm_epsilon", kv_f32(1e-6)),
        (a + "rope.freq_base", kv_f32(g["freq_base"])),
        (a + "rope.dimension_count", kv_u32(n_rot)),
        (a + "rope.scaling.type", kv_str("yarn")),
        (a + "rope.scaling.factor", kv_f32(g["yarn_factor"])),
        (a + "rope.scaling.yarn_beta_fast", kv_f32(32.0)),
        (a + "rope.scaling.yarn_beta_slow", kv_f32(1.0)),
        (a + "rope.scaling.yarn_log_multiplier", kv_f32(1.0)),
        (a + "rope.scaling.original_context_length", kv_u32(g["yarn_orig_ctx"])),
        (a + "expert_count", kv_u32(n_expert)),
        (a + "expert_used_count", kv_u32(g["n_exp_used"])),
        (a + "expert_group_count", kv_u32(1)),
        (a + "expert_group_used_count", kv_u32(1)),
        (a + "expert_gating_func", kv_u32(1)),
        (a + "expert_feed_forward_length", kv_u32(n_ff_exp)),
        (a + "expert_shared_count", kv_u32(1)),
        (a + "expert_weights_scale", kv_f32(1.0)),
        (a + "expert_weights_norm", kv_bool(True)),
        (a + "leading_dense_block_count", kv_u32(g["dense_lead"])),
        (a + "attention.indexer.head_count", kv_u32(idx_heads)),
        (a + "attention.indexer.key_length", kv_u32(idx_dim)),
        (a + "attention.indexer.top_k", kv_u32(g["idx_topk"])),
        (a + "attention.indexer.types", arr_i32(g["idx_types"])),
        (a + "attention.rope_pattern", arr_i32(g["rope_pattern"])),
        (a + "attention.kv_source_map", arr_i32(g["kv_source"])),
        (a + "final_logit_softcapping", kv_f32(g["softcap"])),
    ]
    if vocab:
        md += [
            ("tokenizer.ggml.model", kv_str("gpt2")),
            ("tokenizer.ggml.pre", kv_str("tekken")),
            ("tokenizer.ggml.tokens", arr_str(["tok%d" % i for i in range(n_vocab)])),
            ("tokenizer.ggml.scores", arr_f32([0.0] * n_vocab)),
            ("tokenizer.ggml.token_type", arr_i32([1] * n_vocab)),
            ("tokenizer.ggml.merges", arr_str([])),
        ]

    ts = [
        ("token_embd.weight", w(n_vocab, n_embd, scale=1.0)),
        ("output_norm.weight", norm_w(n_embd)),
        # large enough that the softcap bends the logits
        ("output.weight", w(n_vocab, n_embd, scale=2.0)),
        ("rope_freqs.weight", rng.uniform(0.5, 2.0, n_rot // 2).astype(np.float32)),
    ]
    for il in range(n_layers):
        p = f"blk.{il}."
        nope = k_mla - (n_rot if g["rope_pattern"][il] else 0)
        ts += [
            (p + "attn_norm.weight", norm_w(n_embd)),
            (p + "ffn_norm.weight", norm_w(n_embd)),
            (p + "attn_q.weight", w(n_head * k_mla, n_embd, scale=0.2)),
            (p + "attn_k_b.weight", w(n_head, kv_lora, nope, scale=0.2)),
            (p + "attn_v_b.weight", w(n_head, v_head, kv_lora)),
            (p + "attn_output.weight", w(n_embd, n_head * v_head)),
            (p + "attn_ssmax.weight", w(n_head, scale=0.3)),
            (p + "attn_ssmax.bias", (1.0 + w(n_head, scale=0.2)).astype(np.float32)),
        ]
        if g["kv_source"][il] == il:
            ts += [
                (p + "attn_kv_a_norm.weight", norm_w(kv_lora)),
                (p + "attn_kv_a_mqa.weight",
                 w(kv_lora + (n_rot if g["rope_pattern"][il] else 0), n_embd, scale=0.2)),
            ]
        if g["idx_types"][il]:
            ts += [
                (p + "indexer.q_norm.weight", norm_w(n_embd)),
                (p + "indexer.q_norm.bias", w(n_embd)),
                (p + "indexer.attn_q_b.weight", w(idx_heads * idx_dim, n_embd, scale=0.2)),
                (p + "indexer.attn_k.weight", w(idx_dim, n_embd, scale=0.2)),
                (p + "indexer.k_norm.weight", norm_w(idx_dim)),
                (p + "indexer.k_norm.bias", w(idx_dim)),
                (p + "indexer.proj.weight", w(idx_heads, n_embd, scale=0.5)),
                (p + "indexer.ssmax.weight", w(idx_heads, scale=0.3)),
                (p + "indexer.ssmax.bias", (1.0 + w(idx_heads, scale=0.2)).astype(np.float32)),
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
                (p + "ffn_gate_exps.weight", w(n_expert, n_ff_exp, n_embd)),
                (p + "ffn_down_exps.weight", w(n_expert, n_embd, n_ff_exp)),
                (p + "ffn_up_exps.weight", w(n_expert, n_ff_exp, n_embd)),
                (p + "ffn_gate_shexp.weight", w(n_ff_exp, n_embd)),
                (p + "ffn_down_shexp.weight", w(n_embd, n_ff_exp)),
                (p + "ffn_up_shexp.weight", w(n_ff_exp, n_embd)),
            ]
    return md, ts


def selftest(path: Path) -> None:
    from gguf_reader import GGUFFile
    gf = GGUFFile(path)
    by = {t.name: t for t in gf.tensors}
    assert gf.metadata["general.architecture"] == "mistral-large4", "architecture key lost"
    assert gf.metadata["mistral-large4.attention.kv_source_map"] == SMALL["kv_source"], "kv source map mangled"
    assert list(by["blk.1.attn_k_b.weight"].shape) == [16, 64, 4], f"k_b {by['blk.1.attn_k_b.weight'].shape}"
    assert list(by["blk.3.attn_k_b.weight"].shape) == [32, 64, 4], "NoPE k_b takes the whole head"
    assert list(by["blk.3.attn_kv_a_mqa.weight"].shape) == [128, 64], "NoPE kv_a_mqa has no rope part"
    assert "blk.7.attn_kv_a_mqa.weight" not in by, "a shared-KV layer has no kv_a"
    assert "blk.2.indexer.attn_k.weight" not in by, "a shared indexer layer has no indexer"
    print(f"selftest ok: {len(gf.tensors)} tensors, {path.stat().st_size / 1e6:.2f} MB")


def main() -> int:
    ap = argparse.ArgumentParser(description="write a tiny synthetic mistral-large4 GGUF")
    ap.add_argument("--out", default="ml4-synth.gguf")
    ap.add_argument("--seed", type=int, default=4336)
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
