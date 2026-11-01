// Real DSpark launch/replay probe without loading the trillion-parameter trunk.
// Synthetic captured features test execution and reset determinism, not draft quality.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/glm_dflash.hpp"
#include "strata/core/glm_model.hpp"
#include "strata/kernels/glm_fast.hpp"
#include "strata/kernels/sampler.hpp"
#include "ggml.h"
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

static void require(bool ok, const std::string& message) {
    if (!ok) { std::fprintf(stderr, "ml4_dspark_probe: %s\n", message.c_str()); std::exit(1); }
}

// Negative control for the window-head fix, using captured real target logits.
// Argmax is monotone under the cap, but temperature sampling need not agree.
static int softcap_probe(const char* path, int vocab) {
    require(vocab > 0, "invalid vocabulary size");
    const size_t bytes = std::filesystem::file_size(path);
    require(bytes > 0 && bytes % ((size_t) vocab * sizeof(float)) == 0, "logit row size");
    const int rows = (int) (bytes / ((size_t) vocab * sizeof(float)));
    require(rows <= 256, "head probe is limited to 256 captured rows");
    std::vector<float> capped(bytes / sizeof(float)), raw(capped.size());
    FILE* file = std::fopen(path, "rb");
    require(file && std::fread(capped.data(), 1, bytes, file) == bytes, "read captured logits");
    std::fclose(file);
    for (size_t i = 0; i < raw.size(); ++i) {
        require(std::isfinite(capped[i]) && std::abs(capped[i]) <= 30.0f, "invalid softcapped logit");
        raw[i] = (float) (30.0 * std::atanh(std::clamp(capped[i] / 30.0, -1.0 + 1e-7, 1.0 - 1e-7)));
    }
    int greedy_matches = 0;
    for (int r = 0; r < rows; ++r) {
        const auto a = capped.begin() + (size_t) r * vocab, b = raw.begin() + (size_t) r * vocab;
        greedy_matches += std::max_element(a, a + vocab) - a == std::max_element(b, b + vocab) - b;
    }
    require(cudaSetDevice(0) == cudaSuccess, "head probe device");
    float *device_capped = nullptr, *device_raw = nullptr, *device_fixed = nullptr;
    int *picks = nullptr;
    require(cudaMalloc(&device_capped, bytes) == cudaSuccess && cudaMalloc(&device_raw, bytes) == cudaSuccess &&
            cudaMalloc(&device_fixed, bytes) == cudaSuccess && cudaMalloc(&picks, 3 * rows * sizeof(int)) == cudaSuccess,
            "head probe allocations");
    require(cudaMemcpy(device_capped, capped.data(), bytes, cudaMemcpyHostToDevice) == cudaSuccess &&
            cudaMemcpy(device_raw, raw.data(), bytes, cudaMemcpyHostToDevice) == cudaSuccess &&
            cudaMemcpy(device_fixed, raw.data(), bytes, cudaMemcpyHostToDevice) == cudaSuccess, "head probe upload");
    strata::kernels::SamplerParams sp;
    sp.seed = 4336; sp.temperature = 1.0f; sp.greedy = false;
    strata::kernels::glmf::softcap(device_fixed, (int) raw.size(), 30.0f, nullptr);
    strata::kernels::sample_tokens(device_capped, rows, vocab, nullptr, 0, sp, picks, nullptr);
    strata::kernels::sample_tokens(device_raw, rows, vocab, nullptr, 0, sp, picks + rows, nullptr);
    strata::kernels::sample_tokens(device_fixed, rows, vocab, nullptr, 0, sp, picks + 2 * rows, nullptr);
    std::vector<int> output(3 * rows);
    require(cudaMemcpy(output.data(), picks, output.size() * sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess,
            "head probe download");
    int wrong_raw = 0, wrong_fixed = 0;
    for (int r = 0; r < rows; ++r) {
        wrong_raw += output[r] != output[rows + r];
        wrong_fixed += output[r] != output[2 * rows + r];
    }
    cudaFree(device_capped); cudaFree(device_raw); cudaFree(device_fixed); cudaFree(picks);
    std::printf("SOFTCAP_HEAD rows=%d greedy_matches=%d uncapped_sample_differences=%d fixed_sample_differences=%d\n",
                rows, greedy_matches, wrong_raw, wrong_fixed);
    require(greedy_matches == rows && wrong_raw > 0 && wrong_fixed == 0, "softcap sampling negative control");
    std::printf("ml4_dspark_probe: softcap sampling PASS\n");
    return 0;
}

int main(int argc, char** argv) {
    if (argc == 4 && std::string(argv[1]) == "--softcap") return softcap_probe(argv[2], std::atoi(argv[3]));
    if (argc < 3) {
        std::fprintf(stderr, "usage: ml4_dspark_probe DSpark.gguf TARGET_SHARD_DIR [device]\n");
        return 2;
    }
    const int dev = argc > 3 ? std::atoi(argv[3]) : 0;
    require(cudaSetDevice(dev) == cudaSuccess, "set device");
    std::vector<std::filesystem::path> paths;
    for (const auto& entry : std::filesystem::directory_iterator(argv[2])) {
        const std::string name = entry.path().filename().string();
        if (name.rfind("Mistral-Large-4-1T-A52B-", 0) == 0 && entry.path().extension() == ".gguf")
            paths.push_back(entry.path());
    }
    std::sort(paths.begin(), paths.end());
    std::vector<std::unique_ptr<strata::GgufFile>> shards;
    const strata::TensorInfo *embedding = nullptr, *output = nullptr;
    const strata::GgufFile *embedding_file = nullptr, *output_file = nullptr;
    for (const auto& path : paths) {
        shards.emplace_back(new strata::GgufFile(path.string()));
        const auto* g = shards.back().get();
        if (const auto* t = g->find("token_embd.weight")) { embedding = t; embedding_file = g; }
        if (const auto* t = g->find("output.weight")) { output = t; output_file = g; }
        if (embedding && output) break;
    }
    require(embedding && output && embedding->shape.size() == 2 && output->shape == embedding->shape,
            "target embedding/head missing or inconsistent");
    const int E = (int) embedding->shape[0], V = (int) embedding->shape[1];
    std::string err;
    strata::core::GlmDflash draft;
    require(draft.load(argv[1], dev, err), err);
    require(draft.bind(E, V, 72, err), err);
    std::printf("%s\n", draft.describe().c_str());
    const size_t bytes = (size_t) ggml_row_size((ggml_type) output->type, E) * V;
    void* weights = nullptr;
    require(cudaMalloc(&weights, bytes) == cudaSuccess, "head allocation");
    require(cudaMemcpy(weights, output_file->tensor_data(*output), bytes, cudaMemcpyHostToDevice) == cudaSuccess,
            "head upload");
    strata::core::WSlot head;
    head.type = (int) output->type;
    head.q = weights;
    if (head.type == 0) head.f32 = (const float*) weights;
    head.n_in = E; head.n_out = V;
    const auto stream = draft.stream();
    require(stream != nullptr, "draft stream");
    constexpr int context = 14;
    const auto inject = [&]() {
        draft.reset();
        const int K = (int) draft.target_layers().size();
        for (int p = 0; p < context; ++p) for (int k = 0; k < K; ++k) {
            float* row = draft.stage_row(p, k);
            for (int e = 0; e < E; ++e) row[e] = (((p + 1) * 17 + (k + 3) * e) % 101 - 50) * 0.001f;
            draft.staged(p, k);
        }
        require(draft.inject(0, context, stream, err), err);
    };
    for (int n = 1; n < draft.block(); ++n) {
        std::vector<int32_t> expected;
        for (int pass = 0; pass < 2; ++pass) {
            inject();
            std::vector<int32_t> tokens;
            const bool ok = draft.draft(1, context, n, 0.0f, head, embedding_file->tensor_data(*embedding),
                                        (int) embedding->type, stream, tokens, err);
            require(ok, err);
            require(tokens.size() == (size_t) n, "draft count");
            for (int token : tokens) require(token >= 0 && token < V, "draft token out of range");
            if (pass == 0) expected = tokens;
            else require(tokens == expected, "reset/replay changed the drafts");
            std::printf("draft size=%d pass=%d ids=", n, pass);
            for (int token : tokens) std::printf("%d,", token);
            std::printf("\n"); std::fflush(stdout);
        }
    }
    require(strata::kernels::glmf::launch_errors() == 0, "sticky launch failure");
    cudaStreamSynchronize(stream);
    cudaFree(weights);
    std::printf("ml4_dspark_probe: PASS (synthetic features; no draft-quality claim)\n");
    return 0;
}
