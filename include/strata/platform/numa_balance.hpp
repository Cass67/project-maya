// include/strata/platform/numa_balance.hpp - the GLM engine out of the kernel's NUMA balancing (maya-ds4's #562).
//
// kernel.numa_balancing=1 scans memory whose policy migrates on fault: every thread then takes hinting faults - the
// CPU lane's workers on their shared scratch, each phase as slow as its straggler (a 2-socket Xeon: 3k faults/s
// decoding; maya-ds4 +10% decode).  MPOL_LOCAL as the task policy, before any thread (they inherit it), allocates as
// the default does but takes the process out of the scan; the RAM tier's own mbind (interleaved) is unchanged.  A
// policy the user set (numactl around the engine) is left as it is.  STRATA_GLM_NUMA_BALANCE=1: left to the kernel.
// (A container needs SYS_NICE for set_mempolicy.)
#pragma once

#if defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#endif

namespace strata::platform {

inline void numa_local_policy() {
#if defined(__linux__)
    if (const char* nb = std::getenv("STRATA_GLM_NUMA_BALANCE"); nb != nullptr && std::atoi(nb) != 0) return;
    int mode = 0;
    if (syscall(SYS_get_mempolicy, &mode, nullptr, 0, nullptr, 0) == 0 && mode != 0) return;
    if (syscall(SYS_set_mempolicy, 4 /* MPOL_LOCAL */, nullptr, 0) != 0)
        std::fprintf(stderr, "glm: set_mempolicy(MPOL_LOCAL) failed (%s): NUMA balancing still scans the engine\n",
                     std::strerror(errno));
#endif
}

}  // namespace strata::platform
