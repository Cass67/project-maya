"""ref/glm_dsa.py - the glm-dsa (GLM-5.3, 744B) reference forward pass, one op at a time.

Phase 1 of the GLM-5.3 port: the specification the engine's glm-dsa kernels are parity-tested
against.  It is written from llama.cpp's `src/models/glm-dsa.cpp` (the graph unsloth's GGUFs run
on) with the released config's values as defaults, and keeps `ref/glm.py`'s conventions - and
its ggml primitives - so seams compare op by op:

  * activations are (features, tokens); weights come from `ref/load.py` as (out, in), y = W @ x;
  * every intermediate the engine needs as a parity seam is recorded by `cb()` under the name
    llama.cpp's `cb()` uses for the same node.

What differs from glm5-next (GLM-5.3-Flash, `ref/glm.py`):

  * a plain pre-norm residual (no mHC streams), no KDA layers, no SwiGLU clamps;
  * ROPED MLA: q splits into qk_nope (absorbed through wk_b into the latent) and qk_rope;
    kv_a_mqa yields [latent | k_pe]; the score is q_abs . latent + q_pe . k_pe, scaled by
    1/sqrt(qk_nope + qk_rope); V is the normed latent, expanded through wv_b after the softmax;
  * the lightning indexer scores TOKENS (no k-pool): q from the normed q_a, k from
    layer_norm(attn_k @ x), both roped on their first n_rot dims ([rope | nope] layout) and
    rotated by an orthonormal Walsh-Hadamard matrix, score[t, j] = sum_h relu(q_h . k_j) * w_h
    over visible j <= t, w prescaled by 1/sqrt(key_dim * n_head), top_k tokens selected;
  * SHARED indexer layers (indexer types 0) reuse the previous full layer's selection;
  * rope is ggml's NORM type (adjacent pairs), theta stepped in f32 like ggml's rope cache.

The selection feeds the attention as an unmask: cells not in the top_k are -inf, then the causal
mask applies - so while t + 1 <= top_k every visible token is attended (llama-graph.cpp
build_attn_mha's kq_mask_top_k).

The NextN (MTP) block is not part of the trunk and is not implemented here.

Usage:
    python ref/glm_dsa.py MODEL.gguf [--tokens 1,2,3] [--dump-dir DIR]
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from glm import COVER, F32, layer_norm, rms_norm, sigmoid, silu, split_heads, top_k_desc  # noqa: E402
from load import Gguf  # noqa: E402

# llama.cpp's built-in indexer pattern for GLM >= 5.2 GGUFs without attention.indexer.types
# (glm-dsa.cpp GLM_5_2_DEFAULT_INDEXER_TYPES): layers 0 and 1, then every 4th from 2 are full
GLM_5_2_INDEXER_TYPES = [1, 1] + [1, 0, 0, 0] * 19


class GlmDsaConfig:
    """Every number a kernel depends on, from the GGUF metadata (glm-dsa.* keys) with the released
    GLM-5.3 values as defaults."""

    def __init__(self, meta: dict):
        def u(key: str, default):
            v = meta.get("glm-dsa." + key)
            return default if v is None else v

        self.n_embd = int(u("embedding_length", 6144))
        self.n_nextn = int(u("nextn_predict_layers", 0))
        self.n_layers = int(u("block_count", 79)) - self.n_nextn     # the trunk; NextN blocks follow it
        self.n_head = int(u("attention.head_count", 64))
        self.n_head_k_mla = int(u("attention.key_length_mla", 256))
        self.n_head_v_mla = int(u("attention.value_length_mla", 256))
        self.n_rot = int(u("rope.dimension_count", 64))
        self.qk_nope = self.n_head_k_mla - self.n_rot
        self.q_lora_rank = int(u("attention.q_lora_rank", 2048))
        self.kv_lora_rank = int(u("attention.kv_lora_rank", 512))
        _kl = u("attention.key_length", None)
        if _kl is not None and int(_kl) != self.kv_lora_rank + self.n_rot:
            raise ValueError(f"attention.key_length {int(_kl)} != kv_lora_rank + n_rot "
                             f"{self.kv_lora_rank + self.n_rot}")
        self.rope_freq_base = float(u("rope.freq_base", 8000000.0))
        scaling = u("rope.scaling.type", None)
        if scaling not in (None, "none") or float(u("rope.scaling.factor", 1.0)) != 1.0:
            raise NotImplementedError("rope scaling (YaRN) is not implemented; GLM-5.3 ships without it")

        self.n_expert = int(u("expert_count", 256))
        self.n_expert_used = int(u("expert_used_count", 8))
        self.n_ff_exp = int(u("expert_feed_forward_length", 2048))
        self.n_expert_shared = int(u("expert_shared_count", 1))
        self.n_layer_dense_lead = int(u("leading_dense_block_count", 3))
        self.n_ff = int(u("feed_forward_length", 12288))
        self.expert_weights_scale = float(u("expert_weights_scale", 2.5))
        self.expert_weights_norm = bool(u("expert_weights_norm", True))
        if int(u("expert_gating_func", 2)) != 2:
            raise NotImplementedError("glm-dsa routes with sigmoid gating (expert_gating_func 2)")

        self.idx_n_head = int(u("attention.indexer.head_count", 32))
        self.idx_key_dim = int(u("attention.indexer.key_length", 128))
        self.idx_top_k = int(u("attention.indexer.top_k", 2048))
        if self.n_rot > self.idx_key_dim or self.idx_key_dim & (self.idx_key_dim - 1):
            raise ValueError(f"indexer key_length {self.idx_key_dim}: must be a power of 2 >= n_rot")
        idx_types = u("attention.indexer.types", None)
        if idx_types is None:
            # llama.cpp's fallback: GLM 5 / 5.1 (context < 1M) index every layer, 5.2+ the pattern
            idx_types = ([1] * self.n_layers if int(u("context_length", 1048576)) < 1048576
                         else GLM_5_2_INDEXER_TYPES[:self.n_layers])
        self.idx_full = [bool(int(x)) for x in idx_types][:self.n_layers]
        if not self.idx_full[0]:
            raise ValueError("indexer types: layer 0 must be a full indexer (nothing to share)")

        self.norm_rms_eps = float(u("attention.layer_norm_rms_epsilon", 1e-5))
        # the indexer key's LayerNorm: glm-dsa never reads layer_norm_epsilon and llama_hparams is
        # value-initialized, so llama.cpp runs it with eps 0 - matched here unless the key is present
        self.norm_eps = float(u("attention.layer_norm_epsilon", 0.0))


def rope_norm(x: np.ndarray, pos: np.ndarray, n_rot: int, base: float) -> np.ndarray:
    """ggml_rope_ext, LLAMA_ROPE_TYPE_NORM, no YaRN (ext_factor 0, attn_factor 1): adjacent pairs
    (2i, 2i+1) of the first n_rot features rotate by pos * base^(-2i/n_rot); the rest pass through.
    theta is stepped like ggml's rope cache - theta *= base^(-2/n_rot) in f32 - so long positions
    round the same way.  x is (dim, ..., T); pos is (T,)."""
    out = x.astype(F32, copy=True)
    theta_scale = np.float32(base) ** np.float32(-2.0 / n_rot)
    theta = pos.astype(F32)
    for i in range(0, n_rot, 2):
        c = np.cos(theta).astype(F32)
        s = np.sin(theta).astype(F32)
        x0, x1 = x[i], x[i + 1]
        out[i] = x0 * c - x1 * s
        out[i + 1] = x0 * s + x1 * c
        theta = (theta * theta_scale).astype(F32)
    return out


def hadamard(n: int) -> np.ndarray:
    """llama-kv-cache.cpp's ggml_gen_hadamard: the orthonormal Sylvester Walsh-Hadamard matrix
    (H @ H == I).  Orthonormal, so it leaves every q . k unchanged - it exists for a quantized
    indexer-key cache, and the reference applies it so its seams match llama.cpp's."""
    h = np.array([[1.0]])
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return (h / np.sqrt(n)).astype(F32)


class GlmDsa:
    def __init__(self, path: str | pathlib.Path):
        self.g = Gguf(str(path))
        self.cfg = GlmDsaConfig(self.g.g.metadata)
        self.dumps: dict[str, np.ndarray] = {}
        self.hada = hadamard(self.cfg.idx_key_dim)

    def t(self, name: str) -> np.ndarray:
        return self.g.tensor(name).astype(F32)

    def layer_t(self, il: int, base: str, suffix: str = "weight") -> np.ndarray:
        return self.t(f"blk.{il}.{base}.{suffix}")

    def cb(self, name: str, x: np.ndarray, il: int = -1) -> None:
        self.dumps[f"{name}-{il}" if il >= 0 else name] = np.array(x, copy=True)

    # ================================ the lightning indexer ================================

    def indexer(self, cur: np.ndarray, qr: np.ndarray, pos: np.ndarray, il: int) -> list[np.ndarray]:
        """A full indexer layer: per query t, the selected key positions (descending score), or
        every visible position while t + 1 <= top_k."""
        c = self.cfg
        T = cur.shape[1]
        iq = split_heads(self.layer_t(il, "indexer.attn_q_b") @ qr, c.idx_key_dim, c.idx_n_head)
        iq = rope_norm(iq, pos, c.n_rot, c.rope_freq_base)
        ik = layer_norm(self.layer_t(il, "indexer.attn_k") @ cur, self.layer_t(il, "indexer.k_norm"),
                        self.layer_t(il, "indexer.k_norm", "bias"), c.norm_eps)
        ik = rope_norm(ik, pos, c.n_rot, c.rope_freq_base)
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
            s = top_k_desc(row, c.idx_top_k)
            sel.append(s)
            COVER["dsa_topk_discard_queries"] += 1
            ordered = np.sort(row)[::-1]
            COVER["dsa_topk_near_ties"] += int(ordered[c.idx_top_k - 1] - ordered[c.idx_top_k] < 1e-6)
        return sel

    # ================================ roped MLA ================================

    def attention(self, cur: np.ndarray, pos: np.ndarray, il: int, prev_sel):
        """build_attn with the indexer's selection.  Returns ((n_embd, T), the selection used)."""
        c = self.cfg
        T = cur.shape[1]
        qr = rms_norm(self.layer_t(il, "attn_q_a") @ cur, self.layer_t(il, "attn_q_a_norm"), c.norm_rms_eps)
        self.cb("qr", qr, il)

        if c.idx_full[il]:
            sel = self.indexer(cur, qr, pos, il)
        else:
            sel = prev_sel
            COVER["dsa_shared_indexer_layers"] += 1
        self.dumps[f"top_k-{il}"] = [s.copy() for s in sel]

        q = split_heads(self.layer_t(il, "attn_q_b") @ qr, c.n_head_k_mla, c.n_head)   # (k_mla, n_head, T)
        q_nope, q_pe = q[:c.qk_nope], q[c.qk_nope:]
        q_pe = rope_norm(q_pe, pos, c.n_rot, c.rope_freq_base)
        self.cb("q_pe", q_pe, il)

        kv_cmpr_pe = self.layer_t(il, "attn_kv_a_mqa") @ cur                            # (kv_lora + rope, T)
        k_pe = rope_norm(kv_cmpr_pe[c.kv_lora_rank:], pos, c.n_rot, c.rope_freq_base)  # NOT normed
        kv_cmpr = rms_norm(kv_cmpr_pe[:c.kv_lora_rank], self.layer_t(il, "attn_kv_a_norm"), c.norm_rms_eps)
        self.cb("k_pe", k_pe, il)
        self.cb("kv_cmpr", kv_cmpr, il)

        wk_b = self.layer_t(il, "attn_k_b")                      # (n_head, kv_lora, qk_nope)
        q_abs = np.einsum("hln,nht->lht", wk_b, q_nope).astype(F32)
        self.cb("q_nope_absorbed", q_abs, il)

        wv_b = self.layer_t(il, "attn_v_b")                      # (n_head, v_head, kv_lora)
        kq_scale = 1.0 / np.sqrt(c.n_head_k_mla)
        out = np.zeros((c.n_head * c.n_head_v_mla, T), dtype=F32)
        for t in range(T):
            cells = sel[t]
            lat = kv_cmpr[:, cells].astype(np.float64)           # (kv_lora, n_sel)
            s = (q_abs[:, :, t].astype(np.float64).T @ lat
                 + q_pe[:, :, t].astype(np.float64).T @ k_pe[:, cells].astype(np.float64)) * kq_scale
            p = np.exp(s - s.max(axis=1, keepdims=True))
            p = p / p.sum(axis=1, keepdims=True)                 # (n_head, n_sel)
            ctx = lat @ p.T                                      # (kv_lora, n_head)
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
        """build_moe_ffn: sigmoid probs (unbiased, the weights), + exp_probs_b for the top-k
        selection only, renormalized (clamp 2**-14), x expert_weights_scale; plus the shared expert.
        Experts are dequantized one at a time (ref/load.py's lazy path), so the real model runs."""
        c = self.cfg
        logits = self.layer_t(il, "ffn_gate_inp") @ cur
        probs = sigmoid(logits)
        sel_scores = probs + self.layer_t(il, "exp_probs_b", "bias")[:, None]
        self.cb("ffn_moe_probs", probs, il)

        T = cur.shape[1]
        sel_all = np.empty((c.n_expert_used, T), dtype=np.int64)
        w_all = np.empty((c.n_expert_used, T), dtype=F32)
        for t in range(T):
            sel = top_k_desc(sel_scores[:, t], c.n_expert_used)
            w = probs[sel, t]
            if c.expert_weights_norm:
                w = w / max(float(w.sum()), 6.103515625e-5)
            ordered = np.sort(sel_scores[:, t])[::-1]
            COVER["moe_selection_near_ties"] += int(
                ordered[c.n_expert_used - 1] - ordered[c.n_expert_used] < 1e-4)
            sel_all[:, t] = sel
            w_all[:, t] = (w * F32(c.expert_weights_scale)).astype(F32)

        moe = np.zeros((c.n_embd, T), dtype=np.float64)
        for e in np.unique(sel_all):
            gate = self.g.expert(f"blk.{il}.ffn_gate_exps.weight", int(e))
            up = self.g.expert(f"blk.{il}.ffn_up_exps.weight", int(e))
            down = self.g.expert(f"blk.{il}.ffn_down_exps.weight", int(e))
            rows, toks = np.nonzero(sel_all == e)
            x = cur[:, toks]
            y = down @ (silu(gate @ x) * (up @ x))
            moe[:, toks] += y * w_all[rows, toks][None, :]
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
        x = self.g.rows("token_embd.weight", tokens).T.astype(F32)    # (n_embd, T)
        pos = np.arange(len(tokens), dtype=np.int64)
        prev_sel = None
        for il in range(c.n_layers):
            cur = rms_norm(x, self.layer_t(il, "attn_norm"), c.norm_rms_eps)
            self.cb("attn_norm", cur, il)
            attn, prev_sel = self.attention(cur, pos, il, prev_sel)
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
        self.cb("result_output", logits)
        return logits


def main() -> int:
    ap = argparse.ArgumentParser(description="glm-dsa reference forward pass")
    ap.add_argument("model", help="a glm-dsa GGUF (real, any shard, or tools/glm_dsa_synth_gguf.py's)")
    ap.add_argument("--tokens", default="1,2,3,4,5,6,7,8", help="comma-separated token ids")
    ap.add_argument("--dump-dir", default=None, help="write every cb() seam here as .npy")
    ap.add_argument("--logits-out", default=None, help="write the (T, vocab) f32 logits here (.npy)")
    args = ap.parse_args()

    m = GlmDsa(args.model)
    c = m.cfg
    print(f"glm-dsa: n_embd {c.n_embd}, layers {c.n_layers} ({sum(c.idx_full)} full indexers), "
          f"heads {c.n_head} mla {c.qk_nope}+{c.n_rot}/v {c.n_head_v_mla} kv {c.kv_lora_rank}, "
          f"experts {c.n_expert} x top{c.n_expert_used}, indexer {c.idx_n_head}x{c.idx_key_dim} "
          f"top {c.idx_top_k}")
    tokens = [int(x) for x in args.tokens.split(",")]
    logits = m.forward(tokens)
    if not np.isfinite(logits).all():
        print("glm_dsa.py: NON-FINITE LOGITS")
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
