// src/core/glm_pack_test.cpp - the pack-backed Glm5Model against the REAL UD-IQ1_S weights.
//
//   glm_pack_test <pack_dir> <n_tokens> [out.bin]
//
// Teacher-forces token ids 1..n (streaming, one forward per token) through the pack-backed
// runner and writes the (T, vocab) float32 logits.  The cross-check is llama.cpp on the same
// GGUF (llm_logits_dump) - docs/GLM5-FLASH.md §9.
#include "strata/platform/numa_balance.hpp"
#include "strata/core/glm_model.hpp"
#include "strata/kernels/sampler.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#if defined(__linux__)
#include <dlfcn.h>
#endif
#include <string>
#include <vector>
#include "glm_cpu_decode_bench.hpp"

// Optional profiler control; library is loaded only for an explicit decode trace.
struct DecodeTrace {
    using Control = int (*)(uint64_t);
    using Push = int (*)(const char*);
    using Pop = int (*)();
    Control pause = nullptr, resume = nullptr;
    Push push = nullptr; Pop pop = nullptr;
    DecodeTrace() {
        if (!getenv("GLM_TEST_ROCPROF")) return;
#if defined(__linux__)
        void* lib = dlopen("librocprofiler-sdk-roctx.so", RTLD_NOW | RTLD_GLOBAL);
        if (lib) {
            pause = (Control) dlsym(lib, "roctxProfilerPause");
            resume = (Control) dlsym(lib, "roctxProfilerResume");
            push = (Push) dlsym(lib, "roctxRangePushA");
            pop = (Pop) dlsym(lib, "roctxRangePop");
        }
#endif
        if (!pause || !resume || !push || !pop || pause(0) != 0) {
            std::fprintf(stderr, "decode trace: profiler controls unavailable\n"); std::exit(2);
        }
    }
    void begin(const char* label) { if (resume) { if (resume(0)) std::exit(2); push(label); } }
    void end() { if (pause) { pop(); if (pause(0)) std::exit(2); } }
};

int main(int argc, char** argv) {
    strata::platform::numa_local_policy();   // (before any thread: out of the kernel's NUMA balancing)
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <pack_dir> <n_tokens> [out.bin]\n", argv[0]);
        return 2;
    }
    const std::string pack = argv[1];
    const int n_tok = std::atoi(argv[2]);
    const char* out_path = argc > 3 ? argv[3] : nullptr;

    // The model's optional seam writers open files directly. Prepare the directory
    // before the first forward, so a fresh diagnostic run cannot silently lose dumps.
    if (const char* cb = getenv("GLM_CB_DIR"); cb != nullptr && cb[0]) {
        std::error_code ec;
        std::filesystem::create_directories(cb, ec);
        if (ec) {
            std::fprintf(stderr, "glm_pack_test: cannot create seam directory: %s\n", ec.message().c_str());
            return 1;
        }
    }

    DecodeTrace trace;
    strata::core::Glm5Model model;
    std::string err;
    const int64_t max_ctx = getenv("GLM_TEST_CTX") ? std::atoll(getenv("GLM_TEST_CTX")) : 4096;
    if (!model.load_pack_env(pack, max_ctx, err)) {
        std::fprintf(stderr, "glm_pack_test: %s\n", err.c_str());
        return 1;
    }
    const size_t V = (size_t) model.geometry().n_vocab;
    std::fprintf(stderr, "glm_pack_test: loaded (vocab %zu, layers %d)\n", V, model.geometry().n_layers);

    if (getenv("STRATA_GLM_TEST_PIPE")) {
        // ONE forward for all tokens: with STRATA_GLM_SPLIT this exercises the cross-token pipeline,
        // and the single output row must be BITWISE equal to the serial run's LAST row
        std::vector<int32_t> all((size_t) n_tok);
        for (int t = 0; t < n_tok; ++t) all[(size_t) t] = (int32_t) ((t % 511) + 1);
        std::vector<float> lg;
        if (!model.forward(all, lg, err)) {
            std::fprintf(stderr, "glm_pack_test: pipe forward: %s\n", err.c_str());
            return 1;
        }
        if (out_path) {
            FILE* f = std::fopen(out_path, "wb");
            if (f) {
                std::fwrite(lg.data(), sizeof(float), lg.size(), f);
                std::fclose(f);
            }
        }
        int am = 0;
        for (size_t v = 1; v < V; ++v)
            if (lg[v] > lg[(size_t) am]) am = (int) v;
        std::printf("pipe: last-position argmax %d (%.6f)\n", am, lg[(size_t) am]);
        return 0;
    }

    // GLM_TEST_IDS=<file>: teacher-force those ids (comma/space separated) instead of 1..n - real text
    // is what the fast-vs-reference comparison needs (synthetic ids sit in degenerate near-ties)
    std::vector<int32_t> ids;
    if (const char* fp = getenv("GLM_TEST_IDS")) {
        FILE* f = std::fopen(fp, "r");
        if (f) {
            long v = 0;
            int c = 0;
            bool in = false;
            while ((c = std::fgetc(f)) != EOF) {
                if (c >= '0' && c <= '9') { v = v * 10 + (c - '0'); in = true; }
                else if (in) { ids.push_back((int32_t) v); v = 0; in = false; }
            }
            if (in) ids.push_back((int32_t) v);
            std::fclose(f);
        }
    }
    // A fast-prefilled, snapshot-restored full-model comparison at arbitrary context depth.
    if (const char* prefix = getenv("GLM_TEST_CPU_AB_PREFIX")) {
        const int k = std::atoi(prefix);
        const int count = getenv("GLM_TEST_DECODE_COMPARE") ? std::atoi(getenv("GLM_TEST_DECODE_COMPARE")) : 400;
        const char* threads = getenv("GLM_TEST_LANE_THREADS");
        if (k + count >= max_ctx || !glm_cpu_decode_bench(model, ids, k, count, threads ? threads : "64", err)) {
            std::fprintf(stderr, "CPU_AB failed: %s\n", err.c_str()); return 1;
        }
        std::fprintf(stderr, "CPU_AB PASS\n"); return 0;
    }
    // GLM_TEST_WARM=N: run the sequence N times first (then reset) - the fast path's expert tiers fill, so the
    // recorded pass runs its experts where a warm decode would (VRAM), not on the cold CPU path
    if (const char* wv = getenv("GLM_TEST_WARM")) {
        for (int r = 0; r < std::atoi(wv); ++r) {
            for (int t = 0; t < n_tok; ++t) {
                std::vector<float> lg;
                const int32_t tok = !ids.empty() ? ids[(size_t) t % ids.size()] : (int32_t) ((t % 511) + 1);
                if (!model.forward({tok}, lg, err)) {
                    std::fprintf(stderr, "glm_pack_test: warm forward at %d: %s\n", t, err.c_str());
                    return 1;
                }
            }
            model.reset();
        }
        std::fprintf(stderr, "glm_pack_test: warmed (%s passes)\n", wv);
    }
    std::vector<float> all;
    std::vector<int> draft_ids;
    const int compare_prefill = getenv("GLM_TEST_COMPARE_PREFILL") ? std::atoi(getenv("GLM_TEST_COMPARE_PREFILL")) : 0;
    const auto serial_start = std::chrono::steady_clock::now();
    double serial_prefix_ms = 0;
    const auto stash_seams = [](const char* name) {
        const char* cb = getenv("GLM_CB_DIR");
        if (cb == nullptr || !cb[0]) return;
        const std::filesystem::path root(cb), out = root / name;
        std::filesystem::create_directories(out);
        for (const auto& entry : std::filesystem::directory_iterator(root))
            if (entry.is_regular_file() && entry.path().extension() == ".f32")
                std::filesystem::copy_file(entry.path(), out / entry.path().filename(), std::filesystem::copy_options::overwrite_existing);
    };
    // GLM_TEST_PREFILL=k: the first k positions through the batched prompt path (model.prefill), the rest through
    // the token path as usual - the recorded rows then start at position k (compare against a plain run's tail)
    int t_first = 0;
    if (const char* pv = getenv("GLM_TEST_PREFILL")) {
        const int k = std::min(std::atoi(pv), getenv("GLM_TEST_DUMP") ? n_tok : n_tok - 1);
        if (k > 0) {
            std::vector<int32_t> pre((size_t) k);
            for (int t = 0; t < k; ++t) pre[(size_t) t] = !ids.empty() ? ids[(size_t) t % ids.size()] : (int32_t) ((t % 511) + 1);
            const int32_t nxt = k < n_tok ? (!ids.empty() ? ids[(size_t) k % ids.size()] : (int32_t) ((k % 511) + 1)) : -1;
            if (!model.prefill(pre, err, nxt)) {
                std::fprintf(stderr, "glm_pack_test: prefill of %d: %s\n", k, err.empty() ? "unavailable" : err.c_str());
                return 1;
            }
            t_first = k;
            std::fprintf(stderr, "glm_pack_test: prefilled %d positions\n", k);
        }
    }
    // GLM_TEST_MTP=1: after every position t the NextN block drafts t + 2 from (h_t, token t + 1); the next position's
    // argmax is what a greedy speculative decode would compare it with
    const bool mtp_probe = getenv("GLM_TEST_MTP") != nullptr && model.has_mtp();
    int draft_prev = -1, mtp_n = 0, mtp_ok = 0;
    // Reuse a separately validated teacher-forced artifact for long-prefix F2
    // probes. This is not a fresh serial timing or a same-configuration A/B.
    const char* frozen_serial = getenv("GLM_TEST_SERIAL_LOGITS");
    if (frozen_serial != nullptr) {
        if (compare_prefill <= 0 || t_first != 0 || mtp_probe || getenv("GLM_CB_DIR")) {
            std::fprintf(stderr, "frozen serial: only supported for a comparison without MTP/seam dumps\n");
            return 2;
        }
        FILE* file = std::fopen(frozen_serial, "rb");
        all.resize((size_t) n_tok * V);
        const bool valid = file && std::fread(all.data(), sizeof(float), all.size(), file) == all.size() &&
                           std::fgetc(file) == EOF &&
                           std::all_of(all.begin(), all.end(), [](float v) { return std::isfinite(v); });
        if (file) std::fclose(file);
        if (!valid) {
            std::fprintf(stderr, "frozen serial: wrong shape, nonfinite values or unreadable file\n");
            return 1;
        }
        std::fprintf(stderr, "frozen serial: %d rows loaded; caller must verify model/token identity\n", n_tok);
    }
    for (int t = t_first; frozen_serial == nullptr && t < n_tok; ++t) {
        std::vector<float> lg;
        const int32_t tok = !ids.empty() ? ids[(size_t) t % ids.size()] : (int32_t) ((t % 511) + 1);
        if (!model.forward({tok}, lg, err)) {
            std::fprintf(stderr, "glm_pack_test: forward at %d: %s\n", t, err.c_str());
            return 1;
        }
        all.insert(all.end(), lg.begin(), lg.end());
        int am = 0;
        for (size_t v = 1; v < V; ++v)
            if (lg[v] > lg[am]) am = (int) v;
        if (mtp_probe) {
            static const int mtp_from = getenv("GLM_TEST_MTP_FROM") ? std::atoi(getenv("GLM_TEST_MTP_FROM")) : 0;
            if (draft_prev >= 0 && t >= mtp_from) {
                ++mtp_n;
                mtp_ok += draft_prev == am;
            }
            draft_prev = -1;
            if (t + 1 < n_tok) {
                const int32_t nxt = !ids.empty() ? ids[(size_t) (t + 1) % ids.size()] : (int32_t) (((t + 1) % 511) + 1);
                draft_prev = model.mtp_draft(nxt, err);
                if (draft_prev < 0 && !err.empty()) {
                    std::fprintf(stderr, "glm_pack_test: mtp at %d: %s\n", t, err.c_str());
                    return 1;
                }
                draft_ids.push_back(draft_prev);
            }
        }
        std::printf("pos %d tok %d argmax %d (%.6f)\n", t, tok, am, lg[(size_t) am]);
        if (t + 1 == compare_prefill) {
            serial_prefix_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - serial_start).count();
            stash_seams("serial");
        }
    }
    // A single load compares the new prompt path with the validated token path,
    // including the continuation and NextN drafts after the prefilled prefix.
    if (compare_prefill > 0) {
        const int k = compare_prefill;
        if (t_first != 0 || k >= n_tok) return 2;
        const auto token_at = [&](int t) { return !ids.empty() ? ids[(size_t) t % ids.size()] : (int32_t) ((t % 511) + 1); };
        std::vector<int32_t> prefix;
        for (int t = 0; t < k; ++t) prefix.push_back(token_at(t));
        model.reset();
        const auto start = std::chrono::steady_clock::now();
        if (!model.prefill(prefix, err, token_at(k))) {
            std::fprintf(stderr, "compare prefill: %s\n", err.empty() ? "batched prefill unavailable" : err.c_str());
            return 1;
        }
        stash_seams("batch");
        // A paired decode benchmark can reuse this append-only prefix instead of
        // recomputing thousands of serial prompt tokens for every variant. The
        // first continuation token is replayed after restore to rebuild the head.
        const bool decode_snapshot = getenv("GLM_TEST_DECODE_SNAPSHOT") != nullptr;
        if (decode_snapshot && (!model.geometry().dsa || model.geometry().idx_kpool != 1 || !model.snapshot_save())) {
            std::fprintf(stderr, "decode snapshot: requires an append-only DSA prefix and a valid snapshot\n");
            return 1;
        }
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        if (frozen_serial != nullptr)
            std::printf("compare prefill: %d tokens; frozen serial reference (no serial timing); batch %.1f ms / %.2f tok/s\n",
                        k, ms, k * 1000.0 / ms);
        else
            std::printf("compare prefill: %d tokens serial %.1f ms / %.2f tok/s; batch %.1f ms / %.2f tok/s\n",
                        k, serial_prefix_ms, k * 1000.0 / serial_prefix_ms, ms, k * 1000.0 / ms);
        double worst = 0, top20_sum = 0, top20_max = 0;
        int mismatches = 0, draft_mismatches = 0;
        std::vector<float> batch_logits;
        for (int t = k; t < n_tok; ++t) {
            std::vector<float> lg;
            if (!model.forward({token_at(t)}, lg, err)) return 1;
            if (out_path) batch_logits.insert(batch_logits.end(), lg.begin(), lg.end());
            const float* ref = all.data() + (size_t) t * V;
            const int a = (int) (std::max_element(lg.begin(), lg.end()) - lg.begin());
            const int b = (int) (std::max_element(ref, ref + V) - ref);
            mismatches += a != b;
            for (size_t v = 0; v < V; ++v) {
                if (!std::isfinite(lg[v])) return 1;
                worst = std::max(worst, (double) std::fabs(lg[v] - ref[v]));
            }
            std::vector<size_t> ranked(V);
            for (size_t v = 0; v < V; ++v) ranked[v] = v;
            std::partial_sort(ranked.begin(), ranked.begin() + 20, ranked.end(), [&](size_t a, size_t b) { return ref[a] > ref[b]; });
            const auto lse = [&](const float* x) {
                const double mx = *std::max_element(x, x + V);
                double sum = 0;
                for (size_t v = 0; v < V; ++v) sum += std::exp((double) x[v] - mx);
                return mx + std::log(sum);
            };
            const double lr = lse(ref), lb = lse(lg.data());
            for (int j = 0; j < 20; ++j) {
                const size_t v = ranked[(size_t) j];
                const double delta = std::fabs((lg[v] - lb) - (ref[v] - lr));
                top20_sum += delta; top20_max = std::max(top20_max, delta);
            }
            if (mtp_probe && t + 1 < n_tok) {
                const int d = model.mtp_draft(token_at(t + 1), err);
                if (d < 0) return 1;
                draft_mismatches += d != draft_ids[(size_t) t];
                std::printf("compare draft position=%d serial=%d batch=%d\n", t, draft_ids[(size_t) t], d);
            }
        }
        std::printf("compare prefill: continuation=%d max |dlogit|=%.6f argmax mismatches=%d MTP mismatches=%d\n",
                    n_tok - k, worst, mismatches, draft_mismatches);
        std::printf("compare prefill: top20 mean |dlogp|=%.6f max=%.6f\n", top20_sum / (20.0 * (n_tok - k)), top20_max);
        // Keep both products even when the parity gate fails, so an independent
        // oracle can distinguish numerical drift from an implementation error.
        // The serial file contains n_tok rows; .prefill.bin contains only rows k..n_tok.
        if (out_path) {
            const auto save = [](const std::string& path, const std::vector<float>& rows) {
                FILE* f = std::fopen(path.c_str(), "wb");
                if (!f) return false;
                const bool written = std::fwrite(rows.data(), sizeof(float), rows.size(), f) == rows.size();
                const bool closed = std::fclose(f) == 0;
                return written && closed;
            };
            if (!save(out_path, all) || !save(std::string(out_path) + ".prefill.bin", batch_logits)) {
                std::fprintf(stderr, "compare prefill: cannot save oracle comparison logits\n");
                return 1;
            }
        }
        if (const char* sweep = getenv("GLM_TEST_CPU_SWEEP")) {
            if (!decode_snapshot) { std::fprintf(stderr, "CPU sweep requires snapshot\n"); return 2; }
            const int n = 64;
            model.set_host_logits(false);
            std::vector<int32_t> reference;
            const auto run = [&](const char* phase, int threads, int trial, int count) -> double {
                model.set_follow({}); model.set_pcie_share(-1);
                if (!model.set_cpu_lane_threads(threads) || model.cpu_lane_threads() != threads) return -1;
                std::vector<float> lg;
                if (!model.snapshot_restore() || !model.forward({token_at(k)}, lg, err)) return -1;
                if (!reference.empty()) model.set_follow(reference);
                const auto before = model.fast_stats();
                const auto begin = std::chrono::steady_clock::now();
                std::vector<int32_t> ids;
                strata::kernels::SamplerParams sp; sp.temperature = 0; sp.greedy = true;
                for (int i = 0; i < count; ++i) {
                    const int pick = model.sample_token(sp, err);
                    if (pick < 0) return -1;
                    const int tok = model.forced(pick); ids.push_back(tok);
                    if (i + 1 < count && !model.forward({tok}, lg, err)) return -1;
                }
                const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
                const auto after = model.fast_stats();
                if (reference.empty()) reference = ids;
                std::printf("CPU_SWEEP phase=%s threads=%d trial=%d tokens=%d wall_s=%.6f tok_s=%.6f cpu_experts_output=%.6f cpu_ms_output=%.6f gpu_hits_output=%.6f misses_output=%.6f promotions=%llu disk_reads=%llu target_differences=%zu\n",
                    phase, threads, trial, count, seconds, count/seconds,
                    double(after.cpu_experts-before.cpu_experts)/count, double(after.cpu_ms-before.cpu_ms)/count,
                    double(after.hits-before.hits)/count, double(after.misses-before.misses)/count,
                    (unsigned long long)(after.promotions-before.promotions),
                    (unsigned long long)(after.disk_reads-before.disk_reads), model.follow_differ());
                std::fflush(stdout);
                return count/seconds;
            };
            // Populate via the actual adaptive policy before freezing the shared workload's placement.
            if (run("adaptive-warm", 40, 0, 128) < 0 ||
                !model.diagnostic_expert_cache(-1, err) || run("frozen-reference", 40, 0, 128) < 0) return 1;
            int best = 40; double best_ratio = 1.0;
            int trial = 0;
            for (const char* q = sweep; *q;) {
                const int th = std::atoi(q);
                if (th > 0) {
                    const double a = run("frozen-control", 40, trial, n);
                    const double b = run("frozen-candidate", th, trial, n);
                    const double c = run("frozen-control-after", 40, trial, n);
                    if (a <= 0 || b <= 0 || c <= 0) return 1;
                    const double ratio = b / ((a+c)*0.5);
                    std::printf("CPU_SWEEP_PAIR threads=%d trial=%d ratio=%.6f\n", th, trial, ratio);
                    std::fflush(stdout);
                    if (ratio > best_ratio) { best = th; best_ratio = ratio; }
                    ++trial;
                }
                while (*q && *q != ',') ++q;
                if (*q == ',') ++q;
            }
            if (!model.diagnostic_expert_cache(-1, err)) return 1;
            // Recheck the selected count on the actual adaptive path; placement may adapt in these trials.
            if (!model.diagnostic_expert_cache(0, err)) return 1;
            for (int t = 0; t < 4; ++t)
                if (run("adaptive-finalist", t == 1 || t == 2 ? best : 40, t, 128) < 0) return 1;
            std::printf("CPU_SWEEP_DONE best_threads=%d frozen_ratio=%.6f\n", best, best_ratio);
            std::fflush(stdout);
            model.set_follow({}); model.set_cpu_lane_threads(40);
            return 0; // Performance diagnostic, independent of the pre-existing raw-prefill gate.
        }
        if (getenv("GLM_TEST_PROFILE")) {
            if (!decode_snapshot || !out_path) { std::fprintf(stderr, "decode profile requires snapshot/output\n"); return 2; }
            const int count = getenv("GLM_TEST_DECODE_COMPARE") ? std::max(64, std::atoi(getenv("GLM_TEST_DECODE_COMPARE"))) : 128;
            std::vector<int32_t> reference;
            model.set_host_logits(false);
            const auto phase = [&](const char* name, bool spec, bool tracing, int n) -> bool {
                model.set_follow({}); model.set_pcie_share(-1);
                std::vector<float> lg;
                if (!model.snapshot_restore() || !model.forward({token_at(k)}, lg, err)) return false;
                if (!reference.empty()) model.set_follow(reference);
                model.fast_profile(tracing);
                const std::string phase_path = std::string(out_path) + ".phase";
                FILE* marker = std::fopen(phase_path.c_str(), "w");
                if (!marker) return false;
                std::fprintf(marker, "%s\n", name); std::fclose(marker);
                std::fprintf(stderr, "PROFILE_BEGIN phase=%s traced=%d tokens=%d\n", name, tracing, n);
                std::fflush(stderr);
                const auto st0 = model.fast_stats();
                const uint64_t proposals = model.spec_steps_, accepts = model.spec_hits_;
                strata::kernels::SamplerParams sp; sp.temperature = 0; sp.greedy = true;
                std::vector<int32_t> tokens;
                if (tracing) trace.begin(name);
                const auto start = std::chrono::steady_clock::now();
                if (spec) {
                    int64_t produced = 0;
                    if (!model.decode_dflash(sp, n, 2, 0.0f, [&](int tok) { tokens.push_back(tok); return true; }, produced, err)) return false;
                } else {
                    for (int i = 0; i < n; ++i) {
                        const int pick = model.sample_token(sp, err);
                        if (pick < 0) return false;
                        const int tok = model.forced(pick); tokens.push_back(tok);
                        if (i + 1 < n && !model.forward({tok}, lg, err)) return false;
                    }
                }
                model.win_pipe_drain();
                const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                if (tracing) trace.end();
                model.fast_profile(false, tracing ? name : nullptr, n);
                const auto st1 = model.fast_stats();
                if (reference.empty()) reference = tokens;
                std::printf("PROFILE_RESULT phase=%s traced=%d tokens=%zu wall_s=%.6f tok_s=%.6f cpu_experts_output=%.6f cpu_ms_output=%.6f gpu_hit_experts_output=%.6f misses_output=%.6f disk_reads=%llu promotions=%llu proposals=%llu accepts=%llu target_differences=%zu\n",
                            name, tracing, tokens.size(), seconds, tokens.size()/seconds,
                            (double)(st1.cpu_experts-st0.cpu_experts)/n, (st1.cpu_ms-st0.cpu_ms)/n,
                            (double)(st1.hits-st0.hits)/n, (double)(st1.misses-st0.misses)/n,
                            (unsigned long long)(st1.disk_reads-st0.disk_reads), (unsigned long long)(st1.promotions-st0.promotions),
                            (unsigned long long)(model.spec_steps_-proposals), (unsigned long long)(model.spec_hits_-accepts), model.follow_differ());
                std::fflush(stdout);
                return tokens.size() == (size_t) n;
            };
            if (!phase("serial-warm", false, false, count) || !phase("serial-trace", false, true, 64) ||
                !phase("serial-measure", false, false, count)) return 1;
            if (model.dflash_ready() && (!phase("dflash-warm", true, false, count) ||
                !phase("dflash-trace", true, true, 64) || !phase("dflash-measure", true, false, count))) return 1;
            FILE* marker=std::fopen((std::string(out_path)+".phase").c_str(),"w");
            if (marker) { std::fprintf(marker,"done\n"); std::fclose(marker); }
            model.set_follow({});
            return 0; // profiling is not the independent prefill numerical gate
        }
        int sampled_failures = 0;
        if (getenv("GLM_TEST_DECODE_COMPARE")) {
            const int count = std::max(16, std::atoi(getenv("GLM_TEST_DECODE_COMPARE")));
            // Verify the actual sampled target distribution, not only argmax:
            // a missing softcap is monotone and can pass every greedy check.
            if (getenv("GLM_TEST_SAMPLED_WINDOWS")) {
                if (!decode_snapshot) {
                    std::fprintf(stderr, "sampled windows: requires GLM_TEST_DECODE_SNAPSHOT\n");
                    return 1;
                }
                constexpr int N = 16, T = 3;
                const char* sample_pcie = getenv("GLM_TEST_SAMPLE_PCIE");
                if (sample_pcie) model.set_pcie_share(std::atof(sample_pcie));
                std::fprintf(stderr, "sampled windows: pcie_share=%.6f\n", model.pcie_share());
                strata::kernels::SamplerParams sp;
                sp.seed = 4336; sp.temperature = 1.0f; sp.greedy = false;
                std::vector<int> expected;
                std::vector<float> serial_rows, window_rows;
                std::vector<float> lg;
                if (!model.snapshot_restore()) return 1;
                for (int i = 0; i < N; ++i) {
                    if (!model.forward({token_at(k + i)}, lg, err)) return 1;
                    sp.counter = (uint64_t) i;
                    const int pick = model.sample_token(sp, err);
                    if (pick < 0) return 1;
                    expected.push_back(pick);
                    serial_rows.insert(serial_rows.end(), lg.begin(), lg.end());
                }
                for (bool piped : {false, true}) {
                    if (!model.snapshot_restore()) return 1;
                    int checked = 0, windows = 0, failures = 0;
                    for (int i = 0; i < N;) {
                        const int n = std::min(T, N - i), keep = 1 + windows % n;
                        int32_t tokens[T], picks[T];
                        for (int j = 0; j < n; ++j) tokens[j] = token_at(k + i + j);
                        sp.counter = (uint64_t) i;
                        const bool ok = piped ? model.window_pipe(tokens, n, picks, err, &sp)
                                              : model.window(tokens, n, picks, err, &sp);
                        if (!ok || !model.window_logits(window_rows, err) || !model.window_commit(keep, err)) {
                            std::fprintf(stderr, "sampled windows: %s\n", err.c_str()); return 1;
                        }
                        for (int j = 0; j < n; ++j) {
                            ++checked;
                            if (picks[j] != expected[(size_t) i + j]) {
                                ++sampled_failures;
                                if (failures++ > 0) continue; // retain the first mismatch artifact per mode
                                std::fprintf(stderr, "sampled windows: %s row %d pick %d expected %d\n",
                                             piped ? "pipelined" : "batched", i + j, picks[j], expected[(size_t) i + j]);
                                const float* a = serial_rows.data() + (size_t) (i + j) * V;
                                const float* b = window_rows.data() + (size_t) j * V;
                                double max_diff = 0;
                                int arg_a = 0, arg_b = 0;
                                for (int v = 0; v < V; ++v) {
                                    if (!std::isfinite(a[v]) || !std::isfinite(b[v])) return 1;
                                    max_diff = std::max(max_diff, std::abs((double) a[v] - b[v]));
                                    if (a[v] > a[arg_a]) arg_a = v;
                                    if (b[v] > b[arg_b]) arg_b = v;
                                }
                                std::fprintf(stderr, "sampled windows: max_abs_dlogit=%.9f argmax=%d/%d window_start=%d keep=%d\n",
                                             max_diff, arg_a, arg_b, i, keep);
                                if (out_path) {
                                    const std::string base = std::string(out_path) + ".sampled-" + (piped ? "pipe" : "batch");
                                    FILE* f = std::fopen((base + ".serial.bin").c_str(), "wb");
                                    if (!f) return 1;
                                    const bool saved_a = std::fwrite(a, sizeof(float), V, f) == (size_t) V;
                                    const bool closed_a = std::fclose(f) == 0;
                                    f = std::fopen((base + ".window.bin").c_str(), "wb");
                                    if (!f) return 1;
                                    const bool saved_b = std::fwrite(b, sizeof(float), V, f) == (size_t) V;
                                    const bool closed_b = std::fclose(f) == 0;
                                    if (!saved_a || !closed_a || !saved_b || !closed_b) return 1;
                                }
                                if (!getenv("GLM_TEST_DIAGNOSTIC_CONTINUE")) return 1;
                            }
                        }
                        i += keep; ++windows;
                    }
                    std::printf("SAMPLED_WINDOW %s checks=%d matches=%d context=%d seed=4336 temperature=1 %s\n",
                                piped ? "pipelined" : "batched", checked, checked - failures, k, failures ? "FAIL" : "PASS");
                    std::fflush(stdout);
                }
            }
            if (getenv("GLM_TEST_DFLASH_MATRIX")) {
                if (!decode_snapshot || !model.dflash_ready() || !out_path) {
                    std::fprintf(stderr, "DFlash matrix requires snapshot, drafter and output path\n"); return 2;
                }
                const auto save_rows = [&](const std::string& path, const std::vector<float>& rows) {
                    FILE* file = std::fopen(path.c_str(), "wb");
                    if (!file) return false;
                    const bool written = std::fwrite(rows.data(), sizeof(float), rows.size(), file) == rows.size();
                    const bool closed = std::fclose(file) == 0;
                    return written && closed;
                };
                const auto diagnostic_setting = [](const char* key, const std::string& value) {
#ifdef _WIN32
                    _putenv_s(key, value.c_str());
#else
                    setenv(key, value.c_str(), 1);
#endif
                };
                const auto diagnostic_prefix = [&](const std::string& value) {
                    diagnostic_setting("STRATA_GLM_DFLASH_DIFF_PREFIX", value);
                };
                bool matrix_ok = true;
                struct Policy { const char* name; double share; int count; bool sampled = false; };
                const Policy policies[] = {{"gpu", 1.0, std::min(count, 64)},
                                           {"hybrid", 0.5, count}, {"startup", -1.0, count},
                                           {"gpu-sampled", 1.0, std::min(count, 64), true},
                                           {"startup-sampled", -1.0, std::min(count, 64), true}};
                std::vector<int32_t> cache_control[2];
                const bool cache_experiment = getenv("GLM_TEST_CACHE_WARM") != nullptr;
                for (int cache_pass = 0; cache_pass < (cache_experiment ? 2 : 1); ++cache_pass) {
                const std::string cache_label = cache_experiment ? (cache_pass ? ".warmed" : ".cold") : "";
                if (cache_experiment) {
                    if (cache_pass) {
                        // Only prompt tokens train the cache: replay the known suffix, never the measured continuation.
                        const int warm = std::max(1, std::min(k, std::atoi(getenv("GLM_TEST_CACHE_WARM"))));
                        std::vector<float> warm_logits;
                        const bool all_cpu = getenv("GLM_TEST_CACHE_WARM_ALL_CPU") != nullptr;
                        model.set_follow({}); model.set_pcie_share(all_cpu ? 0.0 : -1.0);
                        if (!model.snapshot_restore() || !model.diagnostic_expert_cache(1, err)) return 1;
                        const auto warm_start = std::chrono::steady_clock::now();
                        for (int t = k - warm; t < k; ++t)
                            if (!model.forward({token_at(t)}, warm_logits, err)) return 1;
                        // Drain/freeze before restoring the measured routing plan. Include this in warmup cost.
                        if (!model.diagnostic_expert_cache(-1, err)) return 1;
                        model.set_pcie_share(-1);
                        std::printf("CACHE_WARM prompt_suffix=%d all_cpu=%d wall_s=%.6f\n", warm, all_cpu,
                            std::chrono::duration<double>(std::chrono::steady_clock::now()-warm_start).count());
                    }
                    if (!model.diagnostic_expert_cache(-1, err)) { std::fprintf(stderr, "cache freeze: %s\n", err.c_str()); return 1; }
                }
                for (const auto& policy : policies) {
                    const char* selection = getenv("GLM_TEST_MATRIX_POLICIES");
                    if (selection) {
                        const std::string list = std::string(",") + selection + ",";
                        if (list.find(std::string(",") + policy.name + ",") == std::string::npos) continue;
                    } else if (policy.sampled) continue;
                    std::vector<int32_t> reference;
                    std::vector<float> reference_rows;
                    const std::string base = std::string(out_path) + cache_label + "." + policy.name;
                    const auto trial = [&](const char* name, bool spec, bool force) -> bool {
                        model.set_pcie_share(policy.share);
                        model.set_follow({});
                        std::vector<float> lg;
                        if (!model.snapshot_restore() || !model.forward({token_at(k)}, lg, err)) return false;
                        if (force) model.set_follow(reference);
                        const std::string artifact = base + "." + name;
                        diagnostic_prefix(spec ? artifact : "");
                        diagnostic_setting("STRATA_GLM_DFLASH_DIFF_IDS", spec ? base + ".serial.ids" : "");
                        strata::kernels::SamplerParams sp;
                        sp.seed = 4336; sp.temperature = policy.sampled ? 1.0f : 0.0f; sp.greedy = !policy.sampled;
                        std::vector<int32_t> generated;
                        bool saved_difference = false;
                        const auto cache_stats0 = model.fast_stats();
                        const auto begin = std::chrono::steady_clock::now();
                        if (spec) {
                            int64_t produced = 0;
                            if (!model.decode_dflash(sp, policy.count, 7, 0.0f,
                                [&](int tok) { generated.push_back(tok); return true; }, produced, err)) return false;
                            if (produced != policy.count) { err = "DFlash matrix: short decode"; return false; }
                        } else {
                            for (int i = 0; i < policy.count; ++i) {
                                const int pick = model.sample_token(sp, err);
                                if (pick < 0) return false;
                                if (reference.empty()) reference_rows.insert(reference_rows.end(), lg.begin(), lg.end());
                                if (!reference.empty() && !saved_difference && pick != reference[(size_t) i]) {
                                    if (!save_rows(artifact + ".window.bin", lg)) return false;
                                    FILE* file = std::fopen((artifact + ".index.txt").c_str(), "w");
                                    if (!file) return false;
                                    std::fprintf(file, "%d %d %d\n", i, reference[(size_t) i], pick);
                                    if (std::fclose(file) != 0) return false;
                                    saved_difference = true;
                                }
                                const int tok = model.forced(pick);
                                generated.push_back(tok);
                                ++sp.counter;
                                if (i + 1 < policy.count && !model.forward({tok}, lg, err)) return false;
                            }
                        }
                        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
                        const auto cache_stats1 = model.fast_stats();
                        if (cache_experiment) {
                            std::printf("CACHE_TRIAL cache=%s policy=%s trial=%s tok_s=%.6f cpu_experts_output=%.6f gpu_hits_output=%.6f misses_output=%.6f promotions=%llu disk_reads=%llu\n",
                                cache_pass ? "warmed" : "cold", policy.name, name, generated.size()/seconds,
                                double(cache_stats1.cpu_experts-cache_stats0.cpu_experts)/generated.size(),
                                double(cache_stats1.hits-cache_stats0.hits)/generated.size(),
                                double(cache_stats1.misses-cache_stats0.misses)/generated.size(),
                                (unsigned long long)(cache_stats1.promotions-cache_stats0.promotions),
                                (unsigned long long)(cache_stats1.disk_reads-cache_stats0.disk_reads));
                            if (!model.diagnostic_expert_cache(-1, err)) return false;
                        }
                        diagnostic_prefix("");
                        diagnostic_setting("STRATA_GLM_DFLASH_DIFF_IDS", "");
                        FILE* ids_file = std::fopen((artifact + ".ids").c_str(), "w");
                        if (!ids_file) return false;
                        for (int tok : generated) std::fprintf(ids_file, "%d\n", tok);
                        if (std::fclose(ids_file) != 0) return false;
                        const bool baseline = reference.empty();
                        if (baseline) {
                            reference = generated;
                            if (!save_rows(base + ".serial.bin", reference_rows)) return false;
                        }
                        const bool match = generated == reference;
                        const size_t differing = model.follow_differ();
                        const bool passed = match && differing == 0;
                        matrix_ok = matrix_ok && passed;
                        size_t first = 0;
                        while (first < generated.size() && first < reference.size() && generated[first] == reference[first]) ++first;
                        std::printf("DFLASH_MATRIX policy=%s share=%.6f trial=%s forced=%d tokens=%zu tok_s=%.3f sequence_match=%d target_differences=%zu first_sequence_difference=%lld %s\n",
                                    (cache_label + "." + policy.name).c_str(), model.pcie_share(), name, force, generated.size(), generated.size() / seconds,
                                    match, differing, match ? -1LL : (long long) first, passed ? "PASS" : "FAIL");
                        std::fflush(stdout);
                        return true; // collect every trial; mismatches retain a failing final result
                    };
                    if (!trial("serial", false, false) || !trial("dflash", true, true) ||
                        !trial("serial-repeat", false, true) || !trial("dflash-repeat", true, true) ||
                        !trial("serial-unforced", false, false) || !trial("dflash-unforced", true, false)) {
                        diagnostic_prefix("");
                        diagnostic_setting("STRATA_GLM_DFLASH_DIFF_IDS", "");
                        std::fprintf(stderr, "DFlash matrix execution: %s\n", err.c_str()); return 1;
                    }
                    if (cache_experiment) {
                        if (!cache_pass) cache_control[policy.sampled ? 1 : 0] = reference;
                        else {
                            const auto& matched = cache_control[policy.sampled ? 1 : 0];
                            if (matched.empty()) return 2;
                            model.set_follow({}); model.set_pcie_share(policy.share);
                            std::vector<float> matched_logits;
                            if (!model.snapshot_restore() || !model.forward({token_at(k)}, matched_logits, err)) return 1;
                            model.set_follow(matched);
                            const auto st0 = model.fast_stats();
                            const auto start = std::chrono::steady_clock::now();
                            strata::kernels::SamplerParams sp; sp.temperature = 0; sp.greedy = true;
                            for (size_t i = 0; i < matched.size(); ++i) {
                                const int pick = model.sample_token(sp, err);
                                if (pick < 0) return 1;
                                const int tok = model.forced(pick);
                                if (i + 1 < matched.size() && !model.forward({tok}, matched_logits, err)) return 1;
                            }
                            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
                            const auto st1 = model.fast_stats();
                            std::printf("CACHE_MATCHED policy=%s tokens=%zu tok_s=%.6f gpu_hits_output=%.6f cpu_experts_output=%.6f target_differences=%zu\n",
                                policy.name, matched.size(), matched.size()/seconds,
                                double(st1.hits-st0.hits)/matched.size(), double(st1.cpu_experts-st0.cpu_experts)/matched.size(), model.follow_differ());
                            model.set_follow({});
                        }
                    }
                }
                } // cache experiment passes
                model.set_follow({}); model.set_pcie_share(-1);
                std::printf("DFLASH_MATRIX overall=%s prefill_raw_gate=%s\n", matrix_ok ? "PASS" : "FAIL",
                            mismatches == 0 && worst < 1.0 ? "PASS" : "FAIL");
                std::fflush(stdout);
                return matrix_ok && sampled_failures == 0 && mismatches == 0 && draft_mismatches == 0 && worst < 1.0 ? 0 : 1;
            }
            model.set_pcie_share(-1);
            std::vector<int> serial_tokens;
            const auto bench = [&](const char* name, bool spec, int threads, double pcie) -> bool {
                model.set_cpu_lane_threads(threads);
                model.set_pcie_share(pcie);
                std::vector<float> lg;
                if (decode_snapshot) {
                    if (!model.snapshot_restore() || !model.forward({token_at(k)}, lg, err)) {
                        if (err.empty()) err = "decode prefix restore failed";
                        return false;
                    }
                } else {
                    model.reset();
                    std::vector<int32_t> head(prefix.begin(), prefix.end() - 1);
                    if (!model.forward(head, lg, err)) return false;
                    if (!model.forward({prefix.back()}, lg, err)) return false;
                }
                model.set_follow(spec ? std::vector<int32_t>(serial_tokens.begin(), serial_tokens.end()) : std::vector<int32_t>{});
                strata::kernels::SamplerParams sp;
                sp.temperature = 0; sp.greedy = true;
                std::vector<int> generated;
                const auto before = model.fast_stats();
                const uint64_t offers = model.spec_steps_, accepts = model.spec_hits_;
                const auto begin = std::chrono::steady_clock::now();
                if (spec && std::strncmp(name, "dfl", 3) == 0) {   // the DFlash2 drafter (decode_dflash: 7 drafts, p_min 0)
                    int64_t produced = 0;
                    if (!model.decode_dflash(sp, count, 7, 0.0f, [&](int tok) { generated.push_back(tok); return true; }, produced, err)) return false;
                } else if (spec && std::strncmp(name, "win", 3) == 0) {   // the MTP verify window (decode_mtp)
                    int64_t produced = 0;
                    if (!model.decode_mtp(sp, count, [&](int tok) { generated.push_back(tok); return true; }, produced, err)) return false;
                } else if (spec) {
                    int64_t produced = 0;
                    if (!model.decode_spec(sp, count, [&](int tok) { generated.push_back(tok); return true; }, produced, err)) return false;
                } else {
                    for (int i = 0; i < count; ++i) {
                        const int tok = model.sample_token(sp, err);
                        if (tok < 0) return false;
                        generated.push_back(tok);
                        if (i + 1 < count && !model.forward({tok}, lg, err)) return false;
                    }
                    if (serial_tokens.empty()) serial_tokens = generated;
                }
                const double wall = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
                const auto after = model.fast_stats();
                const double n = (double) generated.size();
                std::printf("DECODE %s tokens=%zu tok_s=%.3f ms_tok=%.3f cpu_experts_tok=%.2f cpu_ms_tok=%.3f misses_tok=%.2f drafts=%llu accepted=%llu matches_serial=%d target_pick_differences=%zu\n",
                            name, generated.size(), n * 1000 / wall, wall / n,
                            (after.cpu_experts - before.cpu_experts) / n, (after.cpu_ms - before.cpu_ms) / n,
                            (after.misses - before.misses) / n, (unsigned long long) (model.spec_steps_ - offers),
                            (unsigned long long) (model.spec_hits_ - accepts), serial_tokens.empty() || generated == serial_tokens, model.follow_differ());
                std::fflush(stdout);
                if (!serial_tokens.empty() && (generated != serial_tokens || model.follow_differ() != 0)) {
                    err = "paired decode differs from the serial target picks";
                    return false;
                }
                return true;
            };
            if (const char* lt = getenv("GLM_TEST_LANE_THREADS")) {   // "40,44,56": the MTP decode, every miss on the CPU
                const bool plane_ab = getenv("GLM_TEST_CPU_PLANES_AB") != nullptr;
                if (plane_ab) model.set_cpu_plane_cache(false);
                if (!bench("no-mtp", false, 0, -1)) return 1;
                for (const char* q = lt; *q;) {
                    const int th = std::atoi(q);
                    const std::string nm = "lane-" + std::to_string(th);
                    if (th > 0 && plane_ab) {
                        for (int trial = 0; trial < 4; ++trial) {
                            const bool on = trial == 1 || trial == 2;
                            model.set_cpu_plane_cache(on);
                            const std::string ab = "plane-" + std::string(on ? "on-" : "off-") + std::to_string(th) + "-" + std::to_string(trial);
                            if (!bench(ab.c_str(), true, th, 0.0)) {
                                std::fprintf(stderr, "plane A/B: %s\n", err.c_str()); return 1;
                            }
                        }
                    } else if (th > 0 && !bench(nm.c_str(), true, th, 0.0)) {
                        std::fprintf(stderr, "decode comparison: %s\n", err.c_str());
                        return 1;
                    }
                    while (*q && *q != ',') ++q;
                    if (*q == ',') ++q;
                }
            } else if (model.dflash_ready()) {   // (STRATA_GLM_DFLASH: the drafter against the token path)
                if (!bench("no-mtp", false, 0, -1) || !bench("dflash", true, 0, -1) ||
                    !bench("dflash-repeat", true, 0, -1) || !bench("no-mtp-repeat", false, 0, -1)) {
                    std::fprintf(stderr, "decode comparison: %s\n", err.c_str());
                    return 1;
                }
            } else if (model.mtp_ready()) {   // (STRATA_GLM_SPLIT_MTP=1 on a split: the window against the token path and spec)
                if (!bench("no-mtp", false, 0, -1) || !bench("win-mtp", true, 0, -1) || !bench("mtp-default", true, 0, -1) ||
                    !bench("win-mtp-repeat", true, 0, -1)) {
                    std::fprintf(stderr, "decode comparison: %s\n", err.c_str());
                    return 1;
                }
            } else if (!bench("no-mtp", false, 0, -1) || !bench("mtp-default", true, 0, -1) ||
                !bench("mtp-cpu20", true, 20, -1) || !bench("mtp-cpu10", true, 10, -1) ||
                !bench("mtp-pcie0", true, 0, 0.0) || !bench("mtp-pcie0.5", true, 0, 0.5) ||
                !bench("mtp-default-repeat", true, 0, -1)) {
                std::fprintf(stderr, "decode comparison: %s\n", err.c_str());
                return 1;
            }
        }
        return sampled_failures == 0 && mismatches == 0 && draft_mismatches == 0 && worst < 1.0 ? 0 : 1;
    }
    // GLM_TEST_TIME=<n>: then n more tokens token by token, twice: the time a token costs (with and without a DFlash
    // drafter loaded, STRATA_GLM_DFLASH, the same forward)
    if (const char* tv = getenv("GLM_TEST_TIME")) {
        const int nt = std::max(1, std::atoi(tv));
        int at = (int) model.position();
        for (int round = 0; round < 2; ++round) {
            const auto st0 = model.fast_stats();
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < nt; ++i, ++at) {
                std::vector<float> lg;
                const int32_t tok = !ids.empty() ? ids[(size_t) at % ids.size()] : (int32_t) ((at % 511) + 1);
                if (!model.forward({tok}, lg, err)) {
                    std::fprintf(stderr, "glm_pack_test: timed forward at %d: %s\n", at, err.c_str());
                    return 1;
                }
            }
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / nt;
            const auto st1 = model.fast_stats();
            std::printf("token time: %6.2f ms a token | cpu lane %6.1f experts, %5.2f ms a token | vram misses %.1f\n",
                        ms, (double) (st1.cpu_experts - st0.cpu_experts) / nt, (st1.cpu_ms - st0.cpu_ms) / nt,
                        (double) (st1.misses - st0.misses) / nt);
        }
    }
    // GLM_TEST_WINDOW=T: the same ids again through verify windows of T tokens (Glm5Model::window), every window
    // position's pick against the token-at-a-time argmax recorded above - with the CPU lane off (STRATA_GLM_CPU_LANE=0:
    // the CPU's experts are not the GPU's arithmetic) they must agree bit for bit.  The commits keep 1, 2, .. T tokens in
    // turn, so the rejected-token path (the recurrent state restored and the kept tokens replayed) runs as often as the
    // full one; the next window starts after the kept ones.
    if (const char* wt = getenv("GLM_TEST_WINDOW")) {
        const int T = std::max(1, std::min(strata::core::Glm5Model::kWinMax, std::atoi(wt)));
        std::vector<int> am((size_t) n_tok, -1);
        for (int t = t_first; t < n_tok; ++t) {
            const float* lg = all.data() + (size_t) (t - t_first) * V;
            int a = 0;
            for (size_t v = 1; v < V; ++v)
                if (lg[v] > lg[a]) a = (int) v;
            am[(size_t) t] = a;
        }
        model.reset();
        const auto id_at = [&](int t) { return !ids.empty() ? ids[(size_t) t % ids.size()] : (int32_t) ((t % 511) + 1); };
        // GLM_TEST_WINDOW_PIPE=1: the pipelined window (window_pipe) instead of the batched one
        static const bool pipe = getenv("GLM_TEST_WINDOW_PIPE") != nullptr && std::atoi(getenv("GLM_TEST_WINDOW_PIPE")) != 0;
        const auto win = [&](const int32_t* tk, int n, int32_t* o) {
            return pipe ? model.window_pipe(tk, n, o, err) : model.window(tk, n, o, err);
        };
        for (int t = 0; t < t_first; ++t) {   // the prefix the recorded rows start after
            std::vector<float> lg;
            if (!model.forward({id_at(t)}, lg, err)) {
                std::fprintf(stderr, "glm_pack_test: window prefix at %d: %s\n", t, err.c_str());
                return 1;
            }
        }
        int checked = 0, wrong = 0, windows = 0;
        // GLM_TEST_WINDOW_EARLY=1 (with GLM_TEST_WINDOW_PIPE): the DFlash decode's rounds - the picks read as they come,
        // the window stopped after `keep` of them (the later tokens cancelled while in flight), committed behind them,
        // and the next window started without waiting for the devices
        static const bool early = pipe && getenv("GLM_TEST_WINDOW_EARLY") != nullptr;
        for (int t = t_first; t < n_tok && early;) {
            const int n = std::min(T, n_tok - t);
            const int keep = 1 + windows % n;
            int32_t out[strata::core::Glm5Model::kWinMax];
            bool ok = model.win_pipe_begin(id_at(t), err, nullptr, windows == 0);
            for (int i = 1; i < n && ok; ++i) ok = model.win_pipe_push(id_at(t + i), err);
            for (int i = 0; i < keep && ok; ++i) ok = model.win_pipe_pick(i, out[i], err);
            if (ok && keep < n) model.win_pipe_cancel_from(keep);
            if (!ok || !model.win_pipe_settle(err) || !model.window_commit(keep, err)) {
                std::fprintf(stderr, "glm_pack_test: early window at %d: %s\n", t, err.c_str());
                return 1;
            }
            for (int i = 0; i < keep; ++i) {
                ++checked;
                if (out[i] != am[(size_t) (t + i)]) {
                    ++wrong;
                    std::printf("window: pos %d (row %d of %d, kept %d) picks %d, token-at-a-time %d\n", t + i, i, n,
                                keep, out[i], am[(size_t) (t + i)]);
                }
            }
            ++windows;
            t += keep;
        }
        for (int t = t_first; t < n_tok && !early;) {
            const int n = std::min(T, n_tok - t);
            int32_t tok[strata::core::Glm5Model::kWinMax], out[strata::core::Glm5Model::kWinMax];
            for (int i = 0; i < n; ++i) tok[i] = id_at(t + i);
            if (!win(tok, n, out)) {
                std::fprintf(stderr, "glm_pack_test: window at %d: %s\n", t, err.c_str());
                return 1;
            }
            for (int i = 0; i < n; ++i) {
                ++checked;
                if (out[i] != am[(size_t) (t + i)]) {
                    ++wrong;
                    const float* lg = all.data() + (size_t) (t + i - t_first) * V;
                    std::printf("window: pos %d (row %d of %d) picks %d, token-at-a-time %d (its logits %.6f vs %.6f)\n",
                                t + i, i, n, out[i], am[(size_t) (t + i)], lg[out[i]], lg[am[(size_t) (t + i)]]);
                }
            }
            const int keep = 1 + windows % n;
            if (!model.window_commit(keep, err)) {
                std::fprintf(stderr, "glm_pack_test: commit at %d: %s\n", t, err.c_str());
                return 1;
            }
            ++windows;
            t += keep;
        }
        if (early) model.win_pipe_drain();   // the last window's cancelled tokens done
        std::printf("window: %d windows of up to %d, %d of %d picks the token-at-a-time ones\n", windows, T,
                    checked - wrong, checked);
        // GLM_TEST_WINDOW_TIME=<n>: then n tokens token by token and n in full windows of T (every token kept), twice
        // over: the time a token costs each way, from where the check left the sequence
        if (const char* tv = getenv("GLM_TEST_WINDOW_TIME")) {
            const int nt = std::max(T, std::atoi(tv));
            int at = (int) model.position();
            using clk = std::chrono::steady_clock;
            const auto seg = [&](int Tw) -> bool {
                const auto st0 = model.fast_stats();
                const auto t0 = clk::now();
                int done = 0;
                if (Tw == 0) {
                    for (; done < nt; ++done) {
                        std::vector<float> lg;
                        if (!model.forward({id_at(at + done)}, lg, err)) return false;
                    }
                } else {
                    while (done + Tw <= nt) {
                        int32_t tok[strata::core::Glm5Model::kWinMax], out[strata::core::Glm5Model::kWinMax];
                        for (int i = 0; i < Tw; ++i) tok[i] = id_at(at + done + i);
                        if (!win(tok, Tw, out) || !model.window_commit(Tw, err)) return false;
                        done += Tw;
                    }
                }
                const double ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count() / std::max(1, done);
                const auto st1 = model.fast_stats();
                at += done;
                std::printf("window time: %-18s %6.2f ms a token | cpu lane %6.1f experts, %5.2f ms a token\n",
                            Tw == 0 ? "token by token" : ("windows of " + std::to_string(Tw)).c_str(), ms,
                            (double) (st1.cpu_experts - st0.cpu_experts) / std::max(1, done),
                            (st1.cpu_ms - st0.cpu_ms) / std::max(1, done));
                return true;
            };
            // GLM_TEST_WINDOW_TS=1,2,4,8: those window sizes in each round (default: T), each against token by token
            // (a window of 1 is window(), token by token is forward())
            std::vector<int> tws;
            if (const char* ts = getenv("GLM_TEST_WINDOW_TS")) {
                for (const char* c = ts; *c;) {
                    const int v = std::atoi(c);
                    if (v >= 1 && v <= strata::core::Glm5Model::kWinMax) tws.push_back(v);
                    while (*c && *c != ',') ++c;
                    if (*c == ',') ++c;
                }
            }
            if (tws.empty()) tws.push_back(T);
            for (int round = 0; round < 2; ++round)
                // (GLM_TEST_WINDOW_NOTOKEN=1: the windows only - a profile of them alone)
                for (const int Tw : [&] {
                         std::vector<int> v;
                         if (getenv("GLM_TEST_WINDOW_NOTOKEN") == nullptr) v.push_back(0);
                         v.insert(v.end(), tws.begin(), tws.end());
                         return v;
                     }()) {
                    model.win_prof_report();   // (the per-layer split of the window segments below, STRATA_GLM_WIN_PROF)
                    if (!seg(Tw)) {
                        std::fprintf(stderr, "glm_pack_test: timed: %s\n", err.c_str());
                        return 1;
                    }
                }
            model.win_prof_report();
        }
        if (wrong > 0) return 1;
    }
    // GLM_TEST_DUMP=<path>: the sequence state after the run (Glm5Model::dump_state), for per-layer comparisons
    if (const char* dp = getenv("GLM_TEST_DUMP"))
        if (!model.dump_state(dp)) std::fprintf(stderr, "glm_pack_test: state dump failed\n");
    if (out_path) {
        FILE* f = std::fopen(out_path, "wb");
        if (f) {
            std::fwrite(all.data(), sizeof(float), all.size(), f);
            std::fclose(f);
        }
    }
    if (mtp_probe)
        std::printf("mtp: draft == next argmax %d/%d (%.1f%%)\n", mtp_ok, mtp_n, 100.0 * mtp_ok / std::max(1, mtp_n));
    std::fprintf(stderr, "glm_pack_test: PASS (%d tokens streamed)\n", n_tok);
    // GLM_TEST_GEN=N: then N greedy tokens on from the teacher-forced ones (the plain decode's speed)
    if (const char* gv = getenv("GLM_TEST_GEN")) {
        strata::kernels::SamplerParams sp;
        sp.temperature = 0;
        sp.greedy = true;
        const int count = std::atoi(gv);
        std::vector<float> lg;
        std::string gen;
        const auto begin = std::chrono::steady_clock::now();
        if (model.dflash_ready()) {   // (STRATA_GLM_DFLASH: the drafter's decode)
            int64_t produced = 0;
            if (!model.decode_dflash(sp, count, -1, -1.0f, [&](int tok) {
                    gen += (gen.empty() ? "" : ",") + std::to_string(tok);
                    return true;
                }, produced, err)) {
                std::fprintf(stderr, "gen: %s\n", err.c_str());
                return 1;
            }
        } else {
            for (int i = 0; i < count; ++i) {
                const int tok = model.sample_token(sp, err);
                if (tok < 0 || !model.forward({tok}, lg, err)) {
                    std::fprintf(stderr, "gen: %s\n", err.c_str());
                    return 1;
                }
                gen += (i ? "," : "") + std::to_string(tok);
            }
        }
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
        std::printf("GEN tokens=%d tok_s=%.3f ms_tok=%.1f ids=%s\n", count, count * 1000.0 / ms, ms / count, gen.c_str());
    }
    return 0;
}
