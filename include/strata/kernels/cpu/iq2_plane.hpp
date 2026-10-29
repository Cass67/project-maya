#pragma once
#include <cstddef>
#include <cstdint>

namespace strata::kernels::cpu {
// Experimental lossless CPU-only IQ2_XS layout; GPU/storage weights retain their GGUF form.
bool iq2_plane_available() noexcept;
size_t iq2_plane_row_bytes(int n) noexcept;
bool iq2_plane_repack(const uint8_t* src, uint8_t* dst, int rows);
void iq2_plane_gu_rows(const uint8_t* gate, const uint8_t* up, const void* act, float* ff,
                       int r0, int r1, float limit);
size_t iq3_nibble_row_bytes(int n) noexcept;
bool iq3_nibble_repack(const uint8_t* src, uint8_t* dst, int rows, int n);
void iq3_nibble_down_rows(const uint8_t* packed, const void* act, float* out, int r0, int r1, int n);
}
