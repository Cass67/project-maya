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
#include <type_traits>
#include <utility>
#include <vector>

namespace strata::ds4 {

// DS4_SEG_GRAPH (HIP): while a segment records on a stream, a launch on it is kept - a closure, plus the kernel, its
// shape and every argument's bytes mixed into a hash - instead of issued.  The segment's end replays the graph captured
// from the same hash, or captures one (ds4_model.cu, seg_end).  A launch is ~6-9 us of host time with four threads
// launching (#841 diagnostics), a node of a replayed graph ~0.1 us: a decode token's ~1500 short kernels a GPU were
// launch-bound.  The hash covers every byte the kernels receive, so a changed argument is a new graph, never a stale one.
struct SegRec {
    cudaStream_t s = nullptr;
    uint64_t h = 1469598103934665603ull;
    std::vector<std::function<void()>> ops;
    void mix(const void* p, size_t n) {
        const unsigned char* b = static_cast<const unsigned char*>(p);
        for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ull;
    }
};
inline thread_local SegRec* g_seg = nullptr;

template <typename... P, typename... A>
inline bool seg_take(void (*kernel)(P...), dim3 grid, dim3 block, size_t smem, cudaStream_t s, A&&... args) {
#if defined(STRATA_USE_HIP)
    SegRec* r = g_seg;
    if (r == nullptr || s != r->s) return false;
    r->mix(&kernel, sizeof kernel);
    r->mix(&grid, sizeof grid);
    r->mix(&block, sizeof block);
    r->mix(&smem, sizeof smem);
    std::tuple<std::remove_cv_t<P>...> t(std::forward<A>(args)...);   // (as the kernel takes them)
    std::apply([r](const auto&... v) { (r->mix(&v, sizeof v), ...); }, t);
    r->ops.push_back([kernel, grid, block, smem, s, t]() {
        std::apply([&](const auto&... v) { kernel<<<grid, block, smem, s>>>(v...); }, t);
    });
    return true;
#else
    (void) kernel, (void) grid, (void) block, (void) smem, (void) s, ((void) args, ...);
    return false;
#endif
}

/// A plain launch that a recording segment can take (the kernels launched with <<< inside a segment).
template <typename... P, typename... A>
inline void plain_launch(void (*kernel)(P...), dim3 grid, dim3 block, size_t smem, cudaStream_t s, A&&... args) {
    if (seg_take(kernel, grid, block, smem, s, args...)) return;
    kernel<<<grid, block, smem, s>>>(std::forward<A>(args)...);
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
}

}  // namespace strata::ds4
