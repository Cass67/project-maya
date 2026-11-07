// src/ds4/ds4_pdl.cuh - programmatic dependent launch (compute capability 9.0 on): a kernel launched with pdl_launch
// may be put on the device while the kernel before it in its stream still runs - its blocks wait in pdl_wait() until
// that kernel has ended and its writes are visible - so the launch's own latency hides behind the previous kernel
// instead of sitting between the two.  Every kernel pdl_launch starts calls pdl_wait() before it touches memory a
// kernel before it wrote or reads; pdl_trigger() (right after the wait) lets the next one be put on the device in turn.
// One kernel at most waits behind a running one.  A kernel that waits for a flag another stream, device or the host
// sets triggers only once that wait is over (or not at all): the blocks of the kernel behind it would otherwise sit on
// the multiprocessors while it spins, and the kernel that sets its flag on another stream of the same device could
// find none free - a deadlock (the prompt path's all-reduces).  Elsewhere (older GPUs, HIP, DS4_PDL=0): a plain
// launch, and the two calls do nothing.  The arithmetic is untouched: the same kernels in the same order.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <functional>
#include <tuple>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <utility>
#include <vector>

namespace strata::ds4 {

// DS4_SEG_GRAPH (HIP): while a segment records on a stream, a launch on it is kept - a closure, plus the kernel, its
// shape and every argument's bytes mixed into a hash - instead of issued.  The segment's end replays the graph captured
// from the same hash, or captures one (ds4_model.cu, seg_end).  A launch is ~6-9 us of host time with four threads
// launching (#841 diagnostics), a node of a replayed graph ~0.1 us: a decode token's ~1500 short kernels a GPU were
// launch-bound.  The hash covers every byte the kernels receive, so a changed argument is a new graph, never a stale one.
// The launches are kept as bytes in a per-thread arena (kernel, shape, stream and the argument tuple) with a thunk that
// issues them - no allocation a launch once the arena has grown - and hashed a word at a time.
struct SegRec {
    struct Op {
        void (*fn)(const uint8_t*);
        size_t off;
    };
    cudaStream_t s = nullptr;
    uint64_t h = 1469598103934665603ull;
    std::vector<uint8_t> arena;
    std::vector<Op> ops;
    void reset(cudaStream_t st) {
        s = st;
        h = 1469598103934665603ull;
        arena.clear();
        ops.clear();
    }
    void mix(const void* p, size_t n) {
        const unsigned char* b = static_cast<const unsigned char*>(p);
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            uint64_t w;
            std::memcpy(&w, b + i, 8);
            h = (h ^ w) * 0x9E3779B97F4A7C15ull;
            h ^= h >> 29;
        }
        for (; i < n; ++i) h = (h ^ b[i]) * 1099511628211ull;
    }
    template <typename Pack>
    void keep(const Pack& p) {
        static_assert(std::is_trivially_destructible_v<Pack>, "a kept launch is bytes");
        const size_t off = (arena.size() + 15) & ~(size_t) 15;
        arena.resize(off + sizeof(Pack));
        std::memcpy(arena.data() + off, (const void*) &p, sizeof(Pack));
        ops.push_back({[](const uint8_t* b) {
                           const Pack& q = *reinterpret_cast<const Pack*>(b);
                           std::apply([&](const auto&... v) { q.kernel<<<q.grid, q.block, q.smem, q.s>>>(v...); }, q.t);
                       },
                       off});
    }
    void replay() const {
        for (const Op& o : ops) o.fn(arena.data() + o.off);
    }
};
inline SegRec& seg_tls() {
    static thread_local SegRec r = [] {
        SegRec x;
        x.arena.reserve(1 << 16);
        x.ops.reserve(256);
        return x;
    }();
    return r;
}
inline thread_local SegRec* g_seg = nullptr;
// DS4_SEG_PROBE: each launch on the probed stream as (kernel, hash of its shape and arguments), issued as usual
struct SegProbe {
    cudaStream_t s = nullptr;
    std::vector<std::pair<const void*, uint64_t>> ops;
};
inline thread_local SegProbe* g_probe = nullptr;

template <typename... P, typename... A>
inline bool seg_take(void (*kernel)(P...), dim3 grid, dim3 block, size_t smem, cudaStream_t s, A&&... args) {
#if defined(STRATA_USE_HIP)
    if (SegProbe* pb = g_probe; pb != nullptr && s == pb->s) {
        SegRec h;
        h.mix(&grid, sizeof grid);
        h.mix(&block, sizeof block);
        h.mix(&smem, sizeof smem);
        std::tuple<std::remove_cv_t<P>...> t(args...);
        std::apply([&h](const auto&... v) { (h.mix(&v, sizeof v), ...); }, t);
        pb->ops.push_back({(const void*) kernel, h.h});
    }
    SegRec* r = g_seg;
    if (r == nullptr || s != r->s) return false;
    r->mix(&kernel, sizeof kernel);
    r->mix(&grid, sizeof grid);
    r->mix(&block, sizeof block);
    r->mix(&smem, sizeof smem);
    struct Pack {
        void (*kernel)(P...);
        dim3 grid, block;
        size_t smem;
        cudaStream_t s;
        std::tuple<std::remove_cv_t<P>...> t;   // (as the kernel takes them)
    };
    const Pack pk{kernel, grid, block, smem, s, std::tuple<std::remove_cv_t<P>...>(std::forward<A>(args)...)};
    std::apply([r](const auto&... v) { (r->mix(&v, sizeof v), ...); }, pk.t);
    r->keep(pk);
    return true;
#else
    (void) kernel, (void) grid, (void) block, (void) smem, (void) s, ((void) args, ...);
    return false;
#endif
}

/// DS4_LAUNCH_PAD_US=<n> (a test): every direct launch through these helpers is followed by n us of host spin - how much
/// decode depends on launch cost
inline void launch_pad() {
    static const int us = [] {
        const char* v = std::getenv("DS4_LAUNCH_PAD_US");
        return v != nullptr ? std::atoi(v) : 0;
    }();
    if (us <= 0) return;
    const auto t0 = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - t0 < std::chrono::microseconds(us)) {
    }
}

/// A plain launch that a recording segment can take (the kernels launched with <<< inside a segment).
template <typename... P, typename... A>
inline void plain_launch(void (*kernel)(P...), dim3 grid, dim3 block, size_t smem, cudaStream_t s, A&&... args) {
    if (seg_take(kernel, grid, block, smem, s, args...)) return;
    kernel<<<grid, block, smem, s>>>(std::forward<A>(args)...);
    launch_pad();
}

__device__ __forceinline__ void pdl_wait() {
#if !defined(STRATA_USE_HIP) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}
__device__ __forceinline__ void pdl_trigger() {
#if !defined(STRATA_USE_HIP) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

/// Whether the current device takes programmatic launches: compute capability 9.0 or later, unless DS4_PDL=0.
/// (MAYA_DS4_IQ_ONLY: ds4_iq.cu built into another engine without the rest of maya-ds4 - maya-mm3 and its tests: their
/// launches as before)
#if defined(MAYA_DS4_IQ_ONLY)
inline bool pdl_on() { return false; }
#else
bool pdl_on();
#endif

template <typename... P, typename... A>
inline void pdl_launch(void (*kernel)(P...), dim3 grid, dim3 block, size_t smem, cudaStream_t s, A&&... args) {
    if (seg_take(kernel, grid, block, smem, s, args...)) return;
#if !defined(STRATA_USE_HIP)
    if (pdl_on()) {
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = grid;
        cfg.blockDim = block;
        cfg.dynamicSmemBytes = smem;
        cfg.stream = s;
        cudaLaunchAttribute at[1];
        at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        at[0].val.programmaticStreamSerializationAllowed = 1;
        cfg.attrs = at;
        cfg.numAttrs = 1;
        (void) cudaLaunchKernelEx(&cfg, kernel, std::forward<A>(args)...);
        return;
    }
#endif
    kernel<<<grid, block, smem, s>>>(std::forward<A>(args)...);
    launch_pad();
}

}  // namespace strata::ds4
