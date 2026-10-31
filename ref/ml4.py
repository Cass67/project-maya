"""ref/ml4.py - the mistral-large4 (Mistral Large 4, 1T / 52B active) reference forward pass.

Phase 1 of the ML4 port, the spec its kernels are parity-tested against.  Written from llama.cpp
b12017's `src/models/mistral-large4.cpp` (the graph unsloth's GGUFs run on), conventions and
primitives of `ref/glm_dsa.py`, seam names of llama.cpp's `cb()`.

What differs from glm-dsa (GLM-5.3, `ref/glm_dsa.py`):

  * no q_lora: q = attn_q @ x (n_head x 128); the indexer q is attn_q_b @ layer_norm(x) (own
    q_norm with bias, eps attention.layer_norm_epsilon);
  * NoPE layers (rope_pattern 0, every 4th): no k_pe, the whole 128-wide q head goes through wk_b,
    and llama.cpp zero-pads q / k to kv_lora + n_rot - the score is q_abs . latent alone;
  * KV SHARING (kv_source_map): a layer whose source is another layer writes no KV and attends
    over the source's latent cache (it has no kv_a_mqa / kv_a_norm);
  * SSMax: q (and the indexer q, before its rope) x (s_h * log(pos + 1) + b_h) per head;
  * YaRN (freq_scale 1/factor, ramp over corr dims from n_ctx_orig) with LeRoPE rope_freqs
    dividing theta, magnitude cancelled (yarn_log_multiplier 1);
  * softmax router (expert_gating_func 1), top-k by prob, renormalized, no selection bias;
  * the final logits soft-capped: cap * tanh(z / cap).

Shared indexer layers (indexer types 0) reuse the previous full layer's selection, as in glm-dsa.

Usage:
    python ref/ml4.py MODEL.gguf [--tokens 1,2,3] [--dump-dir DIR]
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from glm import COVER, F32, layer_norm, rms_norm, silu, softmax, split_heads, top_k_desc  # noqa: E402
from glm_dsa import hadamard  # noqa: E402
from load import Gguf  # noqa: E402


class Ml4Config:
    """Every number a kernel depends on, from the mistral-large4.* GGUF keys (released values as
    defaults)."""

    def __init__(self, meta: dict):
        def u(key: str, default):
            v = meta.get("mistral-large4." + key)
            return default if v is None else v

        self.n_embd = int(u("embedding_length", 6144))
        self.n_layers = int(u("block_count", 72))
        self.n_head = int(u("attention.head_count", 64))
        self.n_head_k_mla = int(u("attention.key_length_mla", 128))
        self.n_head_v_mla = int(u("attention.value_length_mla", 128))
        self.n_rot = int(u("rope.dimension_count", 64))
        self.kv_lora_rank = int(u("attention.kv_lora_rank", 512))
        if u("attention.q_lora_rank", 0):
            raise NotImplementedError("mistral-large4 with a q_lora")
        self.rope_freq_base = float(u("rope.freq_base", 10000.0))
        factor = float(u("rope.scaling.factor", 1.0))
        self.yarn = u("rope.scaling.type", None) == "yarn" and factor != 1.0
        self.freq_scale = 1.0 / factor
        self.n_ctx_orig = int(u("rope.scaling.original_context_length", u("context_length", 1048576)))
        self.beta_fast = float(u("rope.scaling.yarn_beta_fast", 32.0))
        self.beta_slow = float(u("rope.scaling.yarn_beta_slow", 1.0))
        self.yarn_log_mul = float(u("rope.scaling.yarn_log_multiplier", 0.0))

        self.n_expert = int(u("expert_count", 128))
        self.n_expert_used = int(u("expert_used_count", 4))
        self.n_ff_exp = int(u("expert_feed_forward_length", 6144))
        self.n_layer_dense_lead = int(u("leading_dense_block_count", 1))
        self.expert_weights_scale = float(u("expert_weights_scale", 1.0))
        self.expert_weights_norm = bool(u("expert_weights_norm", True))
        if int(u("expert_gating_func", 1)) != 1:
            raise NotImplementedError("mistral-large4 routes with softmax gating (expert_gating_func 1)")

        self.idx_n_head = int(u("attention.indexer.head_count", 32))
        self.idx_key_dim = int(u("attention.indexer.key_length", 128))
        self.idx_top_k = int(u("attention.indexer.top_k", 2048))
        self.idx_full = [bool(x) for x in u("attention.indexer.types", [1] * self.n_layers)]
        if not self.idx_full[0]:
            raise ValueError("indexer types: layer 0 must be a full indexer")
        self.has_rope = [bool(x) for x in u("attention.rope_pattern", [1] * self.n_layers)]
        self.kv_src = [int(x) for x in u("attention.kv_source_map", list(range(self.n_layers)))]
        for il, s in enumerate(self.kv_src):
            if s > il or self.kv_src[s] != s or self.has_rope[s] != self.has_rope[il]:
                raise ValueError(f"invalid KV source layer {s} for layer {il}")

        self.norm_rms_eps = float(u("attention.layer_norm_rms_epsilon", 1e-5))
        self.norm_eps = float(u("attention.layer_norm_epsilon", 1e-6))
        self.softcap = float(u("final_logit_softcapping", 0.0))

    def has_kv(self, il: int) -> bool:
        return self.kv_src[il] == il

    def attn_factor(self) -> np.float32:
        """llama-context.cpp's yarn_attn_factor: mscale / mscale_all_dims (1 when the log
        multiplier is 1), divided by the magnitude ggml's rope_yarn multiplies back in."""
        if not self.yarn:
            return F32(1.0)
        f = F32(1.0 / self.freq_scale)

        def gm(m: float) -> np.float32:
            return F32(1.0) if f <= 1 else F32(F32(0.1) * F32(m) * F32(math.log(f)) + F32(1.0))

        a = gm(1.0) / gm(self.yarn_log_mul) if self.yarn_log_mul != 0.0 else gm(1.0)
        return F32(F32(a) * (F32(1.0) / (F32(1.0) + F32(0.1) * F32(math.log(f)))))


def rope_yarn(x: np.ndarray, pos: np.ndarray, c: Ml4Config, ff: np.ndarray) -> np.ndarray:
    """ggml_rope_ext NORM with YaRN and freq_factors, as ggml-cpu's rope cache: per pair i0, theta
    = pos * base^(-i0/n_rot) (stepped in f32), divided by ff[i0/2]; interp = freq_scale * theta,
    mixed back to theta by the ramp over corr dims; cos/sin scaled by attn_factor x (1 + 0.1
    ln(1/freq_scale)).  x is (dim, ..., T); the first n_rot features rotate."""
    n_rot = c.n_rot
    out = x.astype(F32, copy=True)
    theta_scale = F32(c.rope_freq_base) ** F32(-2.0 / n_rot)
    fs = F32(c.freq_scale)
    mscale = c.attn_factor()
    ext = F32(1.0) if c.yarn else F32(0.0)
    if c.yarn:
        def corr(beta: float) -> float:
            return n_rot * math.log(c.n_ctx_orig / (beta * 2 * math.pi)) / (2 * math.log(c.rope_freq_base))
        lo = max(0.0, math.floor(F32(corr(c.beta_fast))))
        hi = min(n_rot - 1.0, math.ceil(F32(corr(c.beta_slow))))
        mscale = F32(mscale * (F32(1.0) + F32(0.1) * F32(math.log(F32(1.0) / fs))))
    theta = pos.astype(F32)
    for i0 in range(0, n_rot, 2):
        t_ex = (theta / F32(ff[i0 // 2])).astype(F32)
        t = (fs * t_ex).astype(F32)
        if c.yarn:
            y = F32((i0 // 2 - lo) / max(0.001, hi - lo))
            ramp = F32(1 - min(1.0, max(0.0, float(y)))) * ext
            t = (t * (F32(1) - ramp) + t_ex * ramp).astype(F32)
        cs = (np.cos(t).astype(F32) * mscale).astype(F32)
        sn = (np.sin(t).astype(F32) * mscale).astype(F32)
        x0, x1 = x[i0], x[i0 + 1]
        out[i0] = x0 * cs - x1 * sn
        out[i0 + 1] = x0 * sn + x1 * cs
        theta = (theta * theta_scale).astype(F32)
    return out


def ssmax_pos(pos: np.ndarray) -> np.ndarray:
    """The SSMax position basis as llama.cpp computes it: the attn-temp input 1 + log(pos + 1)
    (double log, stored f32), then scale_bias(1, -1) in f32."""
    return (np.array([np.float32(math.log(math.floor(p) + 1.0) + 1.0) for p in pos], F32) - F32(1.0)).astype(F32)


class Ml4:
    def __init__(self, path: str | pathlib.Path):
        self.g = Gguf(str(path))
        self.cfg = Ml4Config(self.g.g.metadata)
        self.dumps: dict[str, np.ndarray] = {}
        self.hada = hadamard(self.cfg.idx_key_dim)
        self.ff = self.t("rope_freqs.weight") if "rope_freqs.weight" in self.g._by_name else \
            np.ones(self.cfg.n_rot // 2, F32)
        self.kv: dict[int, tuple[np.ndarray, np.ndarray | None]] = {}

    def t(self, name: str) -> np.ndarray:
        return self.g.tensor(name).astype(F32)

    def layer_t(self, il: int, base: str, suffix: str = "weight") -> np.ndarray:
        return self.t(f"blk.{il}.{base}.{suffix}")

    def cb(self, name: str, x: np.ndarray, il: int = -1) -> None:
        self.dumps[f"{name}-{il}" if il >= 0 else name] = np.array(x, copy=True)

    def ssmax(self, il: int, base: str, sp: np.ndarray) -> np.ndarray:
        s = self.layer_t(il, base)
        b = self.layer_t(il, base, "bias")
        return (sp[None, :] * s[:, None] + b[:, None]).astype(F32)         # (n_head, T)

    # ================================ the lightning indexer ================================

    def indexer(self, cur: np.ndarray, pos: np.ndarray, sp: np.ndarray, il: int) -> list[np.ndarray]:
        c = self.cfg
        T = cur.shape[1]
        qn = layer_norm(cur, self.layer_t(il, "indexer.q_norm"), self.layer_t(il, "indexer.q_norm", "bias"),
                        c.norm_eps)
        iq = split_heads(self.layer_t(il, "indexer.attn_q_b") @ qn, c.idx_key_dim, c.idx_n_head)
        iq = (iq * self.ssmax(il, "indexer.ssmax", sp)[None]).astype(F32)
        ik = layer_norm(self.layer_t(il, "indexer.attn_k") @ cur, self.layer_t(il, "indexer.k_norm"),
                        self.layer_t(il, "indexer.k_norm", "bias"), c.norm_eps)
        if c.has_rope[il]:
            iq = rope_yarn(iq, pos, c, self.ff)
            ik = rope_yarn(ik, pos, c, self.ff)
        iq = np.einsum("ij,jht->iht", self.hada, iq).astype(F32)
        ik = (self.hada @ ik).astype(F32)
        self.cb("indexer_q", iq, il)
        self.cb("indexer_k", ik, il)

        weights = (self.layer_t(il, "indexer.proj") @ cur) / F32(np.sqrt(c.idx_key_dim * c.idx_n_head))
        self.cb("indexer_weights", weights, il)
        kq = np.einsum("eht,ej->htj", iq.astype(np.float64), ik.astype(np.float64))
        score = (np.maximum(kq, 0.0) * weights[:, :, None].astype(np.float64)).sum(axis=0)   # (T, T)
        score[np.triu_indices(T, 1)] = -np.inf
        self.cb("indexer_score", score.T, il)

        sel = []
        for t in range(T):
            if t + 1 <= c.idx_top_k:
                sel.append(np.arange(t + 1, dtype=np.int64))
                continue
            row = score[t, :t + 1]
            sel.append(top_k_desc(row, c.idx_top_k))
            COVER["dsa_topk_discard_queries"] += 1
            ordered = np.sort(row)[::-1]
            COVER["dsa_topk_near_ties"] += int(ordered[c.idx_top_k - 1] - ordered[c.idx_top_k] < 1e-6)
        return sel

    # ================================ MLA (roped, NoPE, shared KV) ================================

    def attention(self, cur: np.ndarray, pos: np.ndarray, sp: np.ndarray, il: int, prev_sel):
        c = self.cfg
        T = cur.shape[1]
        rope = c.has_rope[il]
        n_rot = c.n_rot if rope else 0
        qk_nope = c.n_head_k_mla - n_rot

        if c.idx_full[il]:
            sel = self.indexer(cur, pos, sp, il)
        else:
            sel = prev_sel
            COVER["dsa_shared_indexer_layers"] += 1
        self.dumps[f"top_k-{il}"] = [s.copy() for s in sel]

        q = split_heads(self.layer_t(il, "attn_q") @ cur, c.n_head_k_mla, c.n_head)   # (128, n_head, T)
        wk_b = self.layer_t(il, "attn_k_b")                                            # (n_head, kv_lora, qk_nope)
        q_abs = np.einsum("hln,nht->lht", wk_b, q[:qk_nope]).astype(F32)
        self.cb("q_nope_absorbed", q_abs, il)
        q_pe = rope_yarn(q[qk_nope:], pos, c, self.ff) if rope else None

        if c.has_kv(il):
            kv_cmpr_pe = self.layer_t(il, "attn_kv_a_mqa") @ cur
            kv_cmpr = rms_norm(kv_cmpr_pe[:c.kv_lora_rank], self.layer_t(il, "attn_kv_a_norm"), c.norm_rms_eps)
            k_pe = rope_yarn(kv_cmpr_pe[c.kv_lora_rank:], pos, c, self.ff) if rope else None
            self.kv[il] = (kv_cmpr, k_pe)
            self.cb("kv_cmpr", kv_cmpr, il)
        else:
            kv_cmpr, k_pe = self.kv[c.kv_src[il]]
            COVER["ml4_shared_kv_layers"] += 1

        m = self.ssmax(il, "attn_ssmax", sp)                       # (n_head, T)
        q_abs = (q_abs * m[None]).astype(F32)
        if rope:
            q_pe = (q_pe * m[None]).astype(F32)
            self.cb("q_pe", q_pe, il)

        wv_b = self.layer_t(il, "attn_v_b")                        # (n_head, v_head, kv_lora)
        kq_scale = 1.0 / np.sqrt(c.n_head_k_mla)
        out = np.zeros((c.n_head * c.n_head_v_mla, T), dtype=F32)
        for t in range(T):
            cells = sel[t]
            lat = kv_cmpr[:, cells].astype(np.float64)
            s = q_abs[:, :, t].astype(np.float64).T @ lat
            if rope:
                s += q_pe[:, :, t].astype(np.float64).T @ k_pe[:, cells].astype(np.float64)
            s *= kq_scale
            p = np.exp(s - s.max(axis=1, keepdims=True))
            p = p / p.sum(axis=1, keepdims=True)
            ctx = lat @ p.T
            oh = np.einsum("hvl,lh->vh", wv_b.astype(np.float64), ctx)
            out[:, t] = oh.T.reshape(-1).astype(F32)
        self.cb("kqv_out", out, il)
        attn = self.layer_t(il, "attn_output") @ out
        self.cb("attn_out", attn, il)
        return attn, sel

    # ================================ the FFNs ================================

    def dense_ffn(self, cur: np.ndarray, il: int, kind: str = "") -> np.ndarray:
        gate = self.layer_t(il, f"ffn_gate{kind}") @ cur
        up = self.layer_t(il, f"ffn_up{kind}") @ cur
        return (self.layer_t(il, f"ffn_down{kind}") @ (silu(gate) * up)).astype(F32)

    def moe_ffn(self, cur: np.ndarray, il: int) -> np.ndarray:
        """build_moe_ffn, softmax gating: top-k of the probs, those probs renormalized (clamp
        2**-14) x expert_weights_scale; plus the shared expert."""
        c = self.cfg
        probs = softmax(self.layer_t(il, "ffn_gate_inp") @ cur, axis=0)
        self.cb("ffn_moe_probs", probs, il)
        T = cur.shape[1]
        sel_all = np.empty((c.n_expert_used, T), dtype=np.int64)
        w_all = np.empty((c.n_expert_used, T), dtype=F32)
        for t in range(T):
            sel = top_k_desc(probs[:, t], c.n_expert_used)
            w = probs[sel, t]
            if c.expert_weights_norm:
                w = w / max(float(w.sum()), 6.103515625e-5)
            ordered = np.sort(probs[:, t])[::-1]
            COVER["moe_selection_near_ties"] += int(ordered[c.n_expert_used - 1] - ordered[c.n_expert_used] < 1e-4)
            sel_all[:, t] = sel
            w_all[:, t] = (w * F32(c.expert_weights_scale)).astype(F32)

        moe = np.zeros((c.n_embd, T), dtype=np.float64)
        for e in np.unique(sel_all):
            gate = self.g.expert(f"blk.{il}.ffn_gate_exps.weight", int(e))
            up = self.g.expert(f"blk.{il}.ffn_up_exps.weight", int(e))
            down = self.g.expert(f"blk.{il}.ffn_down_exps.weight", int(e))
            rows, toks = np.nonzero(sel_all == e)
            x = cur[:, toks]
            moe[:, toks] += (down @ (silu(gate @ x) * (up @ x))) * w_all[rows, toks][None, :]
        moe = moe.astype(F32)
        self.dumps[f"ffn_moe_topk-{il}"] = sel_all
        self.dumps[f"ffn_moe_weights-{il}"] = w_all
        self.cb("ffn_moe_out", moe, il)
        shexp = self.dense_ffn(cur, il, "_shexp")
        self.cb("ffn_shexp", shexp, il)
        return (moe + shexp).astype(F32)

    # ================================ the trunk ================================

    def forward(self, tokens: list[int]) -> np.ndarray:
        """One teacher-forced pass from position 0; returns (vocab, T) float32 logits."""
        c = self.cfg
        x = self.g.rows("token_embd.weight", tokens).T.astype(F32)
        pos = np.arange(len(tokens), dtype=np.int64)
        sp = ssmax_pos(pos)
        self.kv = {}
        prev_sel = None
        for il in range(c.n_layers):
            cur = rms_norm(x, self.layer_t(il, "attn_norm"), c.norm_rms_eps)
            self.cb("attn_norm", cur, il)
            attn, prev_sel = self.attention(cur, pos, sp, il, prev_sel)
            x = (x + attn).astype(F32)
            self.cb("ffn_inp", x, il)
            cur = rms_norm(x, self.layer_t(il, "ffn_norm"), c.norm_rms_eps)
            cur = self.dense_ffn(cur, il) if il < c.n_layer_dense_lead else self.moe_ffn(cur, il)
            self.cb("ffn_out", cur, il)
            x = (x + cur).astype(F32)
            self.cb("l_out", x, il)
        cur = rms_norm(x, self.t("output_norm.weight"), c.norm_rms_eps)
        self.cb("result_norm", cur)
        logits = self.t("output.weight") @ cur
        if c.softcap:
            logits = (np.tanh(logits / F32(c.softcap)) * F32(c.softcap)).astype(F32)
        self.cb("result_output", logits)
        return logits


def main() -> int:
    ap = argparse.ArgumentParser(description="mistral-large4 reference forward pass")
    ap.add_argument("model", help="a mistral-large4 GGUF (real, any shard, or tools/ml4_synth_gguf.py's)")
    ap.add_argument("--tokens", default="1,2,3,4,5,6,7,8", help="comma-separated token ids")
    ap.add_argument("--dump-dir", default=None, help="write every cb() seam here as .npy")
    ap.add_argument("--logits-out", default=None, help="write the (T, vocab) f32 logits here (.npy)")
    args = ap.parse_args()

    m = Ml4(args.model)
    c = m.cfg
    print(f"mistral-large4: n_embd {c.n_embd}, layers {c.n_layers} ({sum(c.idx_full)} full indexers, "
          f"{sum(c.has_rope)} roped, {sum(not c.has_kv(i) for i in range(c.n_layers))} shared KV), "
          f"heads {c.n_head} k {c.n_head_k_mla}/v {c.n_head_v_mla} kv {c.kv_lora_rank}, "
          f"experts {c.n_expert} x top{c.n_expert_used}, indexer {c.idx_n_head}x{c.idx_key_dim} top {c.idx_top_k}")
    tokens = [int(x) for x in args.tokens.split(",")]
    logits = m.forward(tokens)
    if not np.isfinite(logits).all():
        print("ml4.py: NON-FINITE LOGITS")
        return 1
    print(f"logits {logits.shape}: mean {logits.mean():.6f} std {logits.std():.6f} "
          f"argmax per token: {[int(i) for i in logits.argmax(axis=0)]}")
    if args.logits_out:
        np.save(args.logits_out, np.ascontiguousarray(logits.T))
    if args.dump_dir:
        d = pathlib.Path(args.dump_dir)
        d.mkdir(parents=True, exist_ok=True)
        for name, arr in m.dumps.items():
            if isinstance(arr, np.ndarray):
                np.save(d / f"{name}.npy", arr)
        print(f"dumps -> {d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
