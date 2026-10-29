// Lossless IQ2_XS magnitude/sign planes. Arithmetic follows ggml's AVX2 dot operation order.
#include "strata/kernels/cpu/iq2_plane.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"
namespace strata::kernels::cpu {
namespace {
constexpr int E = 6144, FF = 2048, NB = E / QK_K;   // a gate/up row: 24 blocks

// the repacked block: magnitudes (2 chunks of 128 values, Q2_K layout: byte j of a chunk holds values j, 32+j,
// 64+j, 96+j in bit pairs 0-1 .. 6-7; code 0/1/2 = grid byte 8/25/43), sign bytes (one per 8 values: ksigns, i.e.
// the 7 stored bits + parity), then ggml's scales and d unchanged
struct alignas(2) BlockP {
    uint8_t mag[64];
    uint8_t sgn[32];
    uint8_t scales[8];
    uint16_t d;
};
static_assert(sizeof(BlockP) == 106);

static inline float hsum_float_8(const __m256 x) {   // ggml's (arch/x86)
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

static inline __m128i get_scale_shuffle(int i) {
    static const uint8_t k_shuffle[128] = {
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
        4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7,
        8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11,
        12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15};
    return _mm_loadu_si128((const __m128i*) k_shuffle + i);
}

// ggml_vec_dot_iq2_xs_q8_K's AVX2 path with only the grid bytes (from the magnitude plane) and the sign bytes (from
// the sign plane) loaded differently: the same int8 lanes, the same maddubs / madd / adds, one fmadd a block
static float dot_p(const BlockP* x, const block_q8_K* y) {
    const __m256i mone = _mm256_set1_epi8(1);
    static const uint8_t bit_selector_mask_bytes[32] = {
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80};
    const __m256i bit_selector_mask = _mm256_loadu_si256((const __m256i*) bit_selector_mask_bytes);
    // sign byte g of the chunk's 16 -> the 8 values of group g: q8_k reads groups 4(k-1) .. 4(k-1)+3 (lane 0: the first
    // two, lane 1: the last two - _mm256_shuffle_epi8 works within each 128-bit lane)
    const __m256i sh1 = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
    const __m256i sh2 = _mm256_add_epi8(sh1, _mm256_set1_epi8(4));
    const __m256i sh3 = _mm256_add_epi8(sh1, _mm256_set1_epi8(8));
    const __m256i sh4 = _mm256_add_epi8(sh1, _mm256_set1_epi8(12));
    const __m256i lut = _mm256_setr_epi8(8, 25, 43, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         8, 25, 43, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    const __m256i m3 = _mm256_set1_epi8(3);
    const __m128i m4 = _mm_set1_epi8(0xf);
    const __m128i m1 = _mm_set1_epi8(1);
    __m256 accumf = _mm256_setzero_ps();
    for (int i = 0; i < NB; ++i) {
        const float d = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(x[i].d))) * y[i].d;
        const int8_t* q8 = y[i].qs;
        uint64_t aux64;
        std::memcpy(&aux64, x[i].scales, 8);
        __m128i stmp = _mm_set1_epi64x((long long) aux64);
        stmp = _mm_unpacklo_epi8(_mm_and_si128(stmp, m4), _mm_and_si128(_mm_srli_epi16(stmp, 4), m4));
        const __m128i scales = _mm_add_epi8(_mm_slli_epi16(stmp, 1), m1);
        __m256i sumi1 = _mm256_setzero_si256();
        __m256i sumi2 = _mm256_setzero_si256();
        for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 4) {
            const int chunk = ib32 / 4;
            const __m256i mg = _mm256_loadu_si256((const __m256i*) (x[i].mag + chunk * 32));
            const __m256i q2_1 = _mm256_shuffle_epi8(lut, _mm256_and_si256(mg, m3));
            const __m256i q2_2 = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(mg, 2), m3));
            const __m256i q2_3 = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(mg, 4), m3));
            const __m256i q2_4 = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(mg, 6), m3));
            const __m256i sg = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*) (x[i].sgn + chunk * 16)));
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i*) q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i*) q8); q8 += 32;
            const __m256i q8_3 = _mm256_loadu_si256((const __m256i*) q8); q8 += 32;
            const __m256i q8_4 = _mm256_loadu_si256((const __m256i*) q8); q8 += 32;
            __m256i signs;
            signs = _mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(sg, sh1), bit_selector_mask), bit_selector_mask);
            const __m256i q8s_1 = _mm256_sign_epi8(q8_1, _mm256_or_si256(signs, mone));
            signs = _mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(sg, sh2), bit_selector_mask), bit_selector_mask);
            const __m256i q8s_2 = _mm256_sign_epi8(q8_2, _mm256_or_si256(signs, mone));
            signs = _mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(sg, sh3), bit_selector_mask), bit_selector_mask);
            const __m256i q8s_3 = _mm256_sign_epi8(q8_3, _mm256_or_si256(signs, mone));
            signs = _mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(sg, sh4), bit_selector_mask), bit_selector_mask);
            const __m256i q8s_4 = _mm256_sign_epi8(q8_4, _mm256_or_si256(signs, mone));
            const __m256i dot1 = _mm256_maddubs_epi16(q2_1, q8s_1);
            const __m256i dot2 = _mm256_maddubs_epi16(q2_2, q8s_2);
            const __m256i dot3 = _mm256_maddubs_epi16(q2_3, q8s_3);
            const __m256i dot4 = _mm256_maddubs_epi16(q2_4, q8s_4);
            const __m256i sc1 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle(ib32 + 0)));
            const __m256i sc2 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle(ib32 + 1)));
            const __m256i sc3 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle(ib32 + 2)));
            const __m256i sc4 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle(ib32 + 3)));
            sumi1 = _mm256_add_epi32(sumi1, _mm256_madd_epi16(dot1, sc1));
            sumi2 = _mm256_add_epi32(sumi2, _mm256_madd_epi16(dot2, sc2));
            sumi1 = _mm256_add_epi32(sumi1, _mm256_madd_epi16(dot3, sc3));
            sumi2 = _mm256_add_epi32(sumi2, _mm256_madd_epi16(dot4, sc4));
        }
        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accumf);
    }
    return 0.125f * hsum_float_8(accumf);
}


struct alignas(2) NibbleBlock { uint8_t qs[128], scales[8]; uint16_t d; };
static_assert(sizeof(NibbleBlock)==138);

template<int TYPE> static void nibble_repack(const uint8_t* raw,uint8_t* packed,int k,int rows) {
    const int nb=k/256;
    constexpr int width=TYPE==17?8:4, grids=TYPE==17?512:256;
    static const auto magnitudes=[] {
        std::array<uint64_t,grids> lut{};
        for(int i=0;i<grids;++i) for(int j=0;j<width;++j) {
            uint8_t v;
            if constexpr(TYPE==17) v=((const uint8_t*)(iq2xs_grid+i))[j];
            else v=((const uint8_t*)(iq3xxs_grid+i))[j];
            int c;
            if constexpr(TYPE==17) c=v==8?0:v==25?1:v==43?2:-1;
            else c=v==62?7:(v>=4 && v<=52 && (v-4)%8==0)?(v-4)/8:-1;
            if(c<0)std::abort();
            lut[i]|=uint64_t(c)<<(8*j);
        }
        return lut;
    }();
    static const auto signs=[] {
        std::array<uint64_t,128> lut{};
        for(int i=0;i<128;++i)for(int j=0;j<8;++j)lut[i]|=uint64_t(((ksigns_iq2xs[i]>>j)&1)*8)<<(8*j);
        return lut;
    }();
    for(int r=0;r<rows;++r)for(int b=0;b<nb;++b){
        auto& out=((NibbleBlock*)packed)[(size_t)r*nb+b];std::memset(&out,0,sizeof out);
        for(int g=0;g<32;++g){
            uint64_t v;
            if constexpr(TYPE==17){
                const auto& x=((const block_iq2_xs*)raw)[(size_t)r*nb+b];out.d=x.d;
                std::memcpy(out.scales,x.scales,8);
                uint16_t q=x.qs[g];v=magnitudes[q&511]|signs[q>>9];
            }else{
                const auto& x=((const block_iq3_xxs*)raw)[(size_t)r*nb+b];out.d=x.d;
                uint32_t sas;std::memcpy(&sas,x.qs+64+4*(g/4),4);
                const int scale=sas>>28;
                out.scales[(g/4)/2]|=uint8_t(scale<<(4*((g/4)%2)));
                v=magnitudes[x.qs[2*g]]|(magnitudes[x.qs[2*g+1]]<<32)|signs[(sas>>(7*(g%4)))&127];
            }
            auto* q=out.qs+(g/8)*32+(g%4)*8;
            uint64_t before;std::memcpy(&before,q,8);before|=v<<(4*((g/4)%2));std::memcpy(q,&before,8);
        }
    }
}

template<int TYPE> static float nibble_dot(const uint8_t* row,const void* activation,int k) {
    const auto* x=(const NibbleBlock*)row;
    const auto* y=(const block_q8_K*)activation;
    const __m128i lut128=TYPE==17 ? _mm_setr_epi8(8,25,43,0,0,0,0,0,-8,-25,-43,0,0,0,0,0)
                                : _mm_setr_epi8(4,12,20,28,36,44,52,62,-4,-12,-20,-28,-36,-44,-52,-62);
    const auto lut=_mm256_broadcastsi128_si256(lut128), mask=_mm256_set1_epi8(15);
    auto acc=_mm256_setzero_ps();
    for(int b=0;b<k/256;++b){
        auto s1=_mm256_setzero_si256(),s2=_mm256_setzero_si256();
        for(int j=0;j<8;++j){
            auto q=_mm256_loadu_si256((const __m256i*)(x[b].qs+(j/2)*32));
            if(j%2)q=_mm256_srli_epi16(q,4);
            auto weights=_mm256_shuffle_epi8(lut,_mm256_and_si256(q,mask));
            auto a=_mm256_sign_epi8(_mm256_loadu_si256((const __m256i*)(y[b].qs+32*j)),weights);
            auto dot=_mm256_maddubs_epi16(_mm256_abs_epi8(weights),a);
            __m256i scales;
            if constexpr(TYPE==17){
                int v=x[b].scales[j];
                scales=_mm256_set_m128i(_mm_set1_epi16(2*(v>>4)+1),_mm_set1_epi16(2*(v&15)+1));
            }else{
                int v=(x[b].scales[j/2]>>(4*(j%2)))&15;scales=_mm256_set1_epi16(2*v+1);
            }
            auto prod=_mm256_madd_epi16(dot,scales);
            if(j%2)s2=_mm256_add_epi32(s2,prod);else s1=_mm256_add_epi32(s1,prod);
        }
        float d=_mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(x[b].d)))*y[b].d;
        acc=_mm256_fmadd_ps(_mm256_set1_ps(d),_mm256_cvtepi32_ps(_mm256_add_epi32(s1,s2)),acc);
    }
    return (TYPE==17?0.125f:0.25f)*hsum_float_8(acc);
}

const std::array<uint64_t,512>& magnitude_codes() {
    static const auto table = [] {
        std::array<uint64_t,512> t{};
        for (int i=0; i<512; ++i) {
            const auto* grid = reinterpret_cast<const uint8_t*>(iq2xs_grid+i);
            for (int j=0; j<8; ++j) {
                const int code = grid[j]==8 ? 0 : grid[j]==25 ? 1 : grid[j]==43 ? 2 : -1;
                if (code<0) std::abort();
                t[i] |= uint64_t(code) << (8*j);
            }
        }
        return t;
    }();
    return table;
}
} // namespace
bool iq2_plane_available() noexcept { return cpu_avx2_ok(); }
size_t iq2_plane_row_bytes(int n) noexcept { return n==E ? sizeof(BlockP)*NB : 0; }
bool iq2_plane_repack(const uint8_t* src,uint8_t* dst,int rows) {
    const auto& lut=magnitude_codes();
    for(int r=0;r<rows;++r) {
        auto* x=reinterpret_cast<const block_iq2_xs*>(src)+r*NB;
        auto* y=reinterpret_cast<BlockP*>(dst)+r*NB;
        for(int b=0;b<NB;++b) {
            auto& o=y[b]; std::memset(&o,0,sizeof o); o.d=x[b].d; std::memcpy(o.scales,x[b].scales,8);
            for(int g=0;g<32;++g) {
                uint16_t q=x[b].qs[g]; o.sgn[g]=ksigns_iq2xs[q>>9];
                auto* mag=o.mag+(g/16)*32+(g%4)*8;
                uint64_t v; std::memcpy(&v,mag,8); v |= lut[q&511] << (2*((g/4)%4)); std::memcpy(mag,&v,8);
            }
        }
    }
    return true;
}
void iq2_plane_gu_rows(const uint8_t* gate,const uint8_t* up,const void* act,float* ff,int r0,int r1,float limit) {
    const auto* a=static_cast<const block_q8_K*>(act);
    const auto* g=reinterpret_cast<const BlockP*>(gate);
    const auto* u=reinterpret_cast<const BlockP*>(up);
    for(int r=r0;r<r1;++r) {
        float gv=std::fmin(dot_p(g+r*NB,a),limit);
        float uv=std::fmin(std::fmax(dot_p(u+r*NB,a),-limit),limit);
        ff[r]=(gv/(1.f+std::exp(-gv)))*uv;
    }
}
size_t iq3_nibble_row_bytes(int n) noexcept { return n > 0 && n%256==0 ? sizeof(NibbleBlock)*(n/256) : 0; }
bool iq3_nibble_repack(const uint8_t* src,uint8_t* dst,int rows,int n) {
    if (!iq3_nibble_row_bytes(n)) return false;
    nibble_repack<18>(src,dst,n,rows); return true;
}
void iq3_nibble_down_rows(const uint8_t* packed,const void* act,float* out,int r0,int r1,int n) {
    const size_t row=iq3_nibble_row_bytes(n);
    for(int r=r0;r<r1;++r) out[r]=nibble_dot<18>(packed+row*r,act,n);
}
} // namespace strata::kernels::cpu
#else
namespace strata::kernels::cpu {
size_t iq3_nibble_row_bytes(int) noexcept { return 0; }
bool iq3_nibble_repack(const uint8_t*,uint8_t*,int,int) { return false; }
void iq3_nibble_down_rows(const uint8_t*,const void*,float*,int,int,int) {}
bool iq2_plane_available() noexcept { return false; }
size_t iq2_plane_row_bytes(int) noexcept { return 0; }
bool iq2_plane_repack(const uint8_t*,uint8_t*,int) { return false; }
void iq2_plane_gu_rows(const uint8_t*,const uint8_t*,const void*,float*,int,int,float) {}
}
#endif
