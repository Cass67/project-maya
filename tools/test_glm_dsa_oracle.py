"""tools/test_glm_dsa_oracle.py - the glm-dsa (GLM-5.3) reference forward pass on a synthetic model.

Runs tools/glm_dsa_synth_gguf.py's tiny glm-dsa GGUF through ref/glm_dsa.py and checks:

  * PARITY WITH LLAMA.CPP: data/glm-dsa-synth-llama-logprobs.npy holds upstream llama.cpp's
    full log-softmax at the last position of every prefix of TOKENS (llama-server, CPU, f32 KV,
    rocmmain image @ 5aeac37, the --vocab variant of the same seed); the oracle must match it
    to f32 noise - and two wrong readings (no top-k, shared layers re-indexing) must NOT, so the
    fixture keeps proving it can tell;
  * the indexer selection: every visible key while t + 1 <= top_k, then exactly top_k distinct
    causal keys; shared layers reuse the previous full layer's selection;
  * MoE weights renormalize to expert_weights_scale; logits finite; causality (prefix logits
    unchanged by later tokens); the Hadamard is orthonormal; rope is the identity at position 0.

No GPU, no network: `python -m unittest tools.test_glm_dsa_oracle`.
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

import glm_dsa_synth_gguf  # noqa: E402
from glm_dsa import GlmDsa, hadamard, rope_norm  # noqa: E402

TOKENS = [i * 7 % 512 for i in range(1, 25)]
LLAMA_LOGPROBS = ROOT / "data" / "glm-dsa-synth-llama-logprobs.npy"


def log_softmax(logits: np.ndarray) -> np.ndarray:
    z = logits.astype(np.float64) - logits.max(axis=0, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=0, keepdims=True))


class GlmDsaOracleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="maya-glm-dsa-oracle-")
        cls.gguf = pathlib.Path(cls.tmp.name) / "glm-dsa-synth.gguf"
        md, ts = glm_dsa_synth_gguf.build()
        glm_dsa_synth_gguf.write_gguf(cls.gguf, md, ts)
        cls.model = GlmDsa(cls.gguf)
        cls.cfg = cls.model.cfg
        cls.logits = cls.model.forward(TOKENS)
        cls.llama = np.load(LLAMA_LOGPROBS).astype(np.float64).T          # (vocab, T)

    @classmethod
    def tearDownClass(cls):
        cls.model.g.close()
        cls.tmp.cleanup()

    def _llama_gap(self, logits: np.ndarray) -> np.ndarray:
        return np.abs(log_softmax(logits) - self.llama).max(axis=0)       # per position

    def test_config_read_from_metadata(self):
        c = self.cfg
        self.assertEqual((c.n_embd, c.n_layers, c.n_head), (128, 6, 4))
        self.assertEqual((c.qk_nope, c.n_rot, c.n_head_v_mla, c.kv_lora_rank), (32, 16, 32, 64))
        self.assertEqual(c.idx_full, [True, True, False, True, False, False])
        self.assertEqual((c.idx_top_k, c.idx_key_dim), (8, 64))
        self.assertEqual(c.norm_eps, 0.0)

    def test_matches_llama_cpp(self):
        gap = self._llama_gap(self.logits)
        self.assertLess(float(gap.max()), 1e-5, f"per-position gap {gap}")
        np.testing.assert_array_equal(self.logits.argmax(axis=0), self.llama.argmax(axis=0))

    def test_fixture_rejects_wrong_readings(self):
        for name, patch in (("attend all", lambda c: setattr(c, "idx_top_k", 10 ** 9)),
                            ("re-index shared", lambda c: setattr(c, "idx_full", [True] * c.n_layers))):
            m = GlmDsa(self.gguf)
            patch(m.cfg)
            gap = self._llama_gap(m.forward(TOKENS))
            m.g.close()
            self.assertLess(float(gap[:self.cfg.idx_top_k].max()), 1e-5, f"{name}: no discard yet, must agree")
            self.assertGreater(float(gap[self.cfg.idx_top_k:].max()), 0.1, f"{name}: fixture cannot tell")

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

    def test_moe_weights_renormalize_to_scale(self):
        for il in range(self.cfg.n_layer_dense_lead, self.cfg.n_layers):
            w = self.model.dumps[f"ffn_moe_weights-{il}"]
            np.testing.assert_allclose(w.sum(axis=0), self.cfg.expert_weights_scale, atol=1e-5)

    def test_logits_finite_and_causal(self):
        self.assertEqual(self.logits.shape, (512, len(TOKENS)))
        self.assertTrue(np.isfinite(self.logits).all())
        m = GlmDsa(self.gguf)
        prefix = m.forward(TOKENS[:12])
        m.g.close()
        np.testing.assert_allclose(prefix, self.logits[:, :12], atol=3e-6, rtol=3e-6)

    def test_primitives(self):
        h = hadamard(64)
        np.testing.assert_allclose(h @ h, np.eye(64), atol=1e-6)
        x = np.random.default_rng(0).standard_normal((16, 3)).astype(np.float32)
        np.testing.assert_array_equal(rope_norm(x, np.zeros(3, np.int64), 16, 1e4), x)


if __name__ == "__main__":
    unittest.main()
