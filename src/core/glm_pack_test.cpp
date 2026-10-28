// src/core/glm_pack_test.cpp - the pack-backed Glm5Model against the REAL UD-IQ1_S weights.
//
//   glm_pack_test <pack_dir> <n_tokens> [out.bin]
//
// Teacher-forces token ids 1..n (streaming, one forward per token) through the pack-backed
// runner and writes the (T, vocab) float32 logits.  The cross-check is llama.cpp on the same
// GGUF (llm_logits_dump) - docs/GLM5-FLASH.md §9.
#include "strata/core/glm_model.hpp"
#include "strata/kernels/sampler.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

int main(int argc, char** argv) {
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

    strata::core::Glm5Model model;
    std::string err;
    if (!model.load_pack_env(pack, 4096, err)) {
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
    for (int t = t_first; t < n_tok; ++t) {
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
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
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
        if (getenv("GLM_TEST_DECODE_COMPARE")) {
            const int count = std::max(16, std::atoi(getenv("GLM_TEST_DECODE_COMPARE")));
            std::vector<int> serial_tokens;
            const auto bench = [&](const char* name, bool spec, int threads, double pcie) -> bool {
                model.set_cpu_lane_threads(threads);
                model.set_pcie_share(pcie);
                model.reset();
                std::vector<int32_t> head(prefix.begin(), prefix.end() - 1);
                std::vector<float> lg;
                if (!model.forward(head, lg, err)) return false;   // diagnostic does not depend on experimental prefill
                if (!model.forward({prefix.back()}, lg, err)) return false;
                model.set_follow(spec ? std::vector<int32_t>(serial_tokens.begin(), serial_tokens.end()) : std::vector<int32_t>{});
                strata::kernels::SamplerParams sp;
                sp.temperature = 0; sp.greedy = true;
                std::vector<int> generated;
                const auto before = model.fast_stats();
                const uint64_t offers = model.spec_steps_, accepts = model.spec_hits_;
                const auto begin = std::chrono::steady_clock::now();
                if (spec) {
                    int64_t produced = 0;
                    if (!model.decode_spec(sp, count, [&](int tok) { generated.push_back(tok); return true; }, produced, err)) return false;
                } else {
                    for (int i = 0; i < count; ++i) {
                        const int tok = model.sample_token(sp, err);
                        if (tok < 0) return false;
                        generated.push_back(tok);
                        if (i + 1 < count && !model.forward({tok}, lg, err)) return false;
                    }
                    serial_tokens = generated;
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
                return true;
            };
            if (!bench("no-mtp", false, 0, -1) || !bench("mtp-default", true, 0, -1) ||
                !bench("mtp-cpu20", true, 20, -1) || !bench("mtp-cpu10", true, 10, -1) ||
                !bench("mtp-pcie0", true, 0, 0.0) || !bench("mtp-pcie0.5", true, 0, 0.5) ||
                !bench("mtp-default-repeat", true, 0, -1)) {
                std::fprintf(stderr, "decode comparison: %s\n", err.c_str());
                return 1;
            }
        }
        return mismatches == 0 && draft_mismatches == 0 && worst < 1.0 ? 0 : 1;
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
    return 0;
}
