"""tools/test_ml4_oracle.py - the mistral-large4 reference forward pass on a synthetic model.

Runs tools/ml4_synth_gguf.py's tiny GGUF through ref/ml4.py and checks:

  * PARITY WITH LLAMA.CPP: data/ml4-synth-llama-logprobs.npy holds upstream llama.cpp's full
    log-softmax at the last position of every prefix of TOKENS (llama-server b12017, --device
    none, f32 KV, -fa off, /completion n_probs 512, the --vocab variant of the same seed); the
    oracle must match it to f32 noise - and wrong readings (no top-k, shared layers re-indexing,
    no SSMax, no YaRN, no softcap, own KV on the shared layer) must NOT;
  * the indexer selection (as glm-dsa's) and the shared-KV layer reading its source's latent;
  * MoE weights renormalize to expert_weights_scale; logits finite, under the softcap; causality.

No GPU, no network: `python -m unittest tools.test_ml4_oracle`.
"""
from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "ref"))

import numpy as np  # noqa: E402

import ml4_synth_gguf  # noqa: E402
import ml4  # noqa: E402
from ml4 import Ml4, rope_yarn  # noqa: E402

TOKENS = [i * 7 % 512 for i in range(1, 25)]
LLAMA_LOGPROBS = ROOT / "data" / "ml4-synth-llama-logprobs.npy"


def log_softmax(logits: np.ndarray) -> np.ndarray:
    z = logits.astype(np.float64) - logits.max(axis=0, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=0, keepdims=True))


class Ml4OracleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="maya-ml4-oracle-")
        cls.gguf = pathlib.Path(cls.tmp.name) / "ml4-synth.gguf"
        md, ts = ml4_synth_gguf.build()
        ml4_synth_gguf.write_gguf(cls.gguf, md, ts)
        cls.model = Ml4(cls.gguf)
        cls.cfg = cls.model.cfg
        cls.logits = cls.model.forward(TOKENS)
        cls.llama = np.load(LLAMA_LOGPROBS).astype(np.float64).T

    @classmethod
    def tearDownClass(cls):
        cls.model.g.close()
        cls.tmp.cleanup()

    def _gap(self, logits: np.ndarray) -> np.ndarray:
        return np.abs(log_softmax(logits) - self.llama).max(axis=0)

    def _variant(self, patch) -> np.ndarray:
        m = Ml4(self.gguf)
        patch(m)
        try:
            return self._gap(m.forward(TOKENS))
        finally:
            m.g.close()

    def test_config_read_from_metadata(self):
        c = self.cfg
        self.assertEqual((c.n_embd, c.n_layers, c.n_head, c.n_head_k_mla, c.n_rot), (128, 8, 4, 32, 16))
        self.assertEqual(c.idx_full, [True, True, False, True, False, True, False, False])
        self.assertEqual(c.has_rope, [True, True, True, False, True, True, True, False])
        self.assertEqual(c.kv_src, [0, 1, 2, 3, 4, 5, 6, 3])
        self.assertTrue(c.yarn)
        self.assertAlmostEqual(float(c.attn_factor()) * (1 + 0.1 * np.log(16.0)), 1.0, places=6)

    def test_matches_llama_cpp(self):
        gap = self._gap(self.logits)
        self.assertLess(float(gap.max()), 1e-4, f"per-position gap {gap}")
        np.testing.assert_array_equal(self.logits.argmax(axis=0), self.llama.argmax(axis=0))

    def test_fixture_rejects_wrong_readings(self):
        def no_ssmax(m):
            m.ssmax = lambda il, base, sp: np.ones((m.cfg.idx_n_head if "indexer" in base else m.cfg.n_head,
                                                    sp.shape[0]), np.float32)

        def own_kv(m):
            m.cfg.kv_src = list(range(m.cfg.n_layers))
            m.g._by_name["blk.7.attn_kv_a_mqa.weight"] = m.g._by_name["blk.3.attn_kv_a_mqa.weight"]
            m.g._by_name["blk.7.attn_kv_a_norm.weight"] = m.g._by_name["blk.3.attn_kv_a_norm.weight"]

        for name, patch, from_pos in (
                ("attend all", lambda m: setattr(m.cfg, "idx_top_k", 10 ** 9), self.cfg.idx_top_k),
                ("no ssmax", no_ssmax, 1),
                ("no yarn", lambda m: setattr(m.cfg, "yarn", False), 1),
                ("no softcap", lambda m: setattr(m.cfg, "softcap", 0.0), 0),
                ("own kv on the shared layer", own_kv, 0)):
            gap = self._variant(patch)
            self.assertLess(float(gap[:from_pos].max(initial=0.0)), 1e-4, f"{name}: must agree before {from_pos}")
            self.assertGreater(float(gap[from_pos:].max()), 0.05, f"{name}: fixture cannot tell")

    def test_selection(self):
        K = self.cfg.idx_top_k
        prev = None
        for il in range(self.cfg.n_layers):
            sel = self.model.dumps[f"top_k-{il}"]
            if self.cfg.idx_full[il]:
                for t, s in enumerate(sel):
                    if t + 1 <= K:
                        np.testing.assert_array_equal(np.sort(s), np.arange(t + 1))
                    else:
                        self.assertEqual(len(set(s.tolist())), K, f"layer {il} t {t}")
                        self.assertLessEqual(int(s.max()), t, f"layer {il} t {t}: future key selected")
            else:
                for a, b in zip(sel, prev, strict=True):
                    np.testing.assert_array_equal(a, b)
            prev = sel
        self.assertGreater(ml4.COVER["ml4_shared_kv_layers"], 0)

    def test_moe_weights_renormalize_to_scale(self):
        for il in range(self.cfg.n_layer_dense_lead, self.cfg.n_layers):
            w = self.model.dumps[f"ffn_moe_weights-{il}"]
            np.testing.assert_allclose(w.sum(axis=0), self.cfg.expert_weights_scale, atol=1e-5)

    def test_logits_finite_capped_and_causal(self):
        self.assertEqual(self.logits.shape, (512, len(TOKENS)))
        self.assertTrue(np.isfinite(self.logits).all())
        self.assertLessEqual(float(np.abs(self.logits).max()), self.cfg.softcap)
        m = Ml4(self.gguf)
        prefix = m.forward(TOKENS[:12])
        m.g.close()
        np.testing.assert_allclose(prefix, self.logits[:, :12], atol=3e-5, rtol=3e-5)

    def test_rope_identity_at_zero(self):
        x = np.random.default_rng(0).standard_normal((16, 3)).astype(np.float32)
        np.testing.assert_allclose(rope_yarn(x, np.zeros(3, np.int64), self.cfg, self.model.ff), x, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
