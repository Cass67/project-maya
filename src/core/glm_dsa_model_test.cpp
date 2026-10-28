// src/core/glm_dsa_model_test.cpp - the glm-dsa (GLM-5.3) runner vs ref/glm_dsa.py on the synthetic model.
//
// data/glm-dsa-synth.gguf (tools/glm_dsa_synth_gguf.py, seed 4321) and data/glm-dsa-synth-dumps (ref/glm_dsa.py
// --dump-raw-dir over TOKENS; the oracle itself matches llama.cpp, tools/test_glm_dsa_oracle.py).  24 tokens
// against an indexer top_k of 8, so from position 8 on the selection discards keys, and layers 2, 4 and 5 reuse
// the previous full layer's selection.  Two gates:
//
//   1. EVERY PREFIX: each prefix its own sequence (reset first); its last logits within f32 tolerance of the
//      oracle's column, and the same argmax.
//   2. STREAMING: one token per call on a persistent state reproduces the batched call's logits.
#include "strata/core/glm_model.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

void require(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "glm_dsa_model_test: %s\n", what.c_str());
        std::exit(1);
    }
}

double max_abs(const float* a, const float* b, size_t n) {
    double m = 0.0;
    for (size_t i = 0; i < n; ++i) m = std::max(m, (double) std::fabs(a[i] - b[i]));
    return m;
}

size_t argmax(const float* v, size_t n) { return (size_t) (std::max_element(v, v + n) - v); }

}  // namespace

// --pack DIR IDS OUT: a REAL pack (MAYA_GLM_SLOW=1) fed the comma-separated token ids one at a time; every
// position's logits (n_vocab f32) appended to OUT - for the real-model check against ref/glm_dsa.py and llama.cpp
static int probe_pack(const std::string& dir, const std::string& ids, const std::string& out) {
    std::vector<int32_t> tokens;
    for (size_t at = 0; at < ids.size();) {
        const size_t c = ids.find(',', at);
        tokens.push_back((int32_t) std::stol(ids.substr(at, c - at)));
        at = c == std::string::npos ? ids.size() : c + 1;
    }
    strata::core::Glm5Model model;
    std::string err;
    require(model.load_pack_env(dir, 256, err), "load_pack: " + err);
    FILE* f = std::fopen(out.c_str(), "wb");
    require(f != nullptr, "cannot write " + out);
    for (int32_t tok : tokens) {
        std::vector<float> lg;
        require(model.forward({tok}, lg, err), "forward: " + err);
        std::fwrite(lg.data(), sizeof(float), lg.size(), f);
        std::printf("pos %lld token %d argmax %zu\n", (long long) model.position() - 1, tok, argmax(lg.data(), lg.size()));
        std::fflush(stdout);
    }
    std::fclose(f);
    return 0;
}

int main(int argc, char** argv) {
    if (argc == 5 && std::string(argv[1]) == "--pack") return probe_pack(argv[2], argv[3], argv[4]);
    if (argc < 2 || std::string(argv[1]) != "--selftest") {
        std::fprintf(stderr, "glm_dsa_model_test: run with --selftest (the CTest form)\n");
        return 2;
    }
    const std::string gguf = argc > 2 ? argv[2] : "data/glm-dsa-synth.gguf";
    const std::string dumps = argc > 3 ? argv[3] : "data/glm-dsa-synth-dumps";
    std::vector<int32_t> tokens;
    for (int i = 1; i <= 24; ++i) tokens.push_back(i * 7 % 512);   // TOKENS in tools/test_glm_dsa_oracle.py

    strata::core::Glm5Model model;
    std::string err;
    require(model.load(gguf, 64, err), "load: " + err);
    require(model.geometry().dsa, "the synthetic did not load as glm-dsa");
    const size_t V = (size_t) model.geometry().n_vocab;
    const size_t T = tokens.size();

    std::vector<float> oracle(V * T);   // (vocab, T) ggml order: element (v, t) at v + V*t
    FILE* f = std::fopen((dumps + "/result_output.f32").c_str(), "rb");
    require(f != nullptr, "cannot open " + dumps + "/result_output.f32 (run from the repo root)");
    require(std::fread(oracle.data(), sizeof(float), oracle.size(), f) == oracle.size(), "short oracle dump");
    std::fclose(f);

    // ---- 1. every prefix vs the oracle
    double worst = 0.0;
    std::vector<float> last;
    for (size_t t = 1; t <= T; ++t) {
        model.reset();
        std::vector<float> lg;
        require(model.forward(std::vector<int32_t>(tokens.begin(), tokens.begin() + (long) t), lg, err),
                "forward: " + err);
        const float* col = oracle.data() + V * (t - 1);
        const double d = max_abs(lg.data(), col, V);
        worst = std::max(worst, d);
        require(d < 2e-3, "prefix " + std::to_string(t) + " diverges from ref/glm_dsa.py (" + std::to_string(d) + ")");
        require(argmax(lg.data(), V) == argmax(col, V), "prefix " + std::to_string(t) + " argmax differs");
        last = std::move(lg);
    }

    // ---- 2. streaming vs the batched call
    model.reset();
    std::vector<float> streamed;
    for (int32_t tok : tokens) require(model.forward({tok}, streamed, err), "streamed forward: " + err);
    const double d_stream = max_abs(streamed.data(), last.data(), V);
    require(d_stream < 1e-5, "streamed and batched logits differ (" + std::to_string(d_stream) + ")");

    std::printf("glm_dsa_model_test: %zu prefixes match ref/glm_dsa.py (max |dlogit| %.2e), streaming exact (%.1e)\n",
                T, worst, d_stream);
    return 0;
}
