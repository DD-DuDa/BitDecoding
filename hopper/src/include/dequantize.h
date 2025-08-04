#pragma once

#include <cute/tensor.hpp>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>


namespace quant {

using namespace cute;

// Instances of `Vec` are used to organize groups of >>registers<<, as needed for instance as inputs to tensor core
// operations. Consequently, all corresponding index accesses must be compile-time constants, which is why we
// extensively use `#pragma unroll` throughout the kernel code to guarantee this.
template <typename T, int n>
struct Vec {
  T elems[n];
  __device__ T& operator[](int i) {
    return elems[i];
  }
};


using I4 = Vec<int, 4>;

// Matrix fragments for tensor core instructions; their precise layout is documented here: 
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-fragments-for-mma-m16n8k16-with-floating-point-type
using FragA = Vec<half2, 4>;
using FragB = Vec<half2, 8>;
using FragC = Vec<float, 4>;
using FragS = Vec<half2, 1>; // quantization scales


// Lookup-table based 3-input logical operation; explicitly used for dequantization as the compiler does not seem to
// automatically recognize it in all cases. 
template <int lut>
__device__ inline int lop3(int a, int b, int c) {
  int res;
  asm volatile(
    "lop3.b32 %0, %1, %2, %3, %4;\n"
    : "=r"(res) : "r"(a), "r"(b), "r"(c), "n"(lut)
  );
  return res;
}

// Efficiently dequantize an int32 value into a full B-fragment of 4 fp16 values.
// We mostly follow the strategy in the link below, with some small changes:
// https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h
__device__ inline FragA lop3_dequant(int q) {
    const int LO = 0x000f000f;
    const int HI = 0x00f000f0;
    const int EX = 0x64006400;

    // Shift right by 8 to now consider elt_45 and elt_67. Issue first to hide RAW dependency if we issue
    // immediately before required.
    const uint32_t top_i4s = q >> 8;

    // Guarantee that the `(a & b) | c` operations are LOP3s.
    int lo_1 = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);        // 0,4
    int hi_1 = lop3<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);        // 1,5
    int lo_2 = lop3<(0xf0 & 0xcc) | 0xaa>(top_i4s, LO, EX);  // 2,6
    int hi_2 = lop3<(0xf0 & 0xcc) | 0xaa>(top_i4s, HI, EX);  // 3,7

    // We want signed int4 outputs, hence we fuse the `-8` symmetric zero point directly into `SUB` and `ADD`.
    const int SUB = 0x64006400; // 0x64086408
    const int MUL = 0x2c002c00; // {1/16, 1/16}
    const int ADD = 0xd400d400; // 0xd480d480

    FragA frag_a;
    frag_a[0] = __hsub2(
        *reinterpret_cast<half2*>(&lo_1),
        *reinterpret_cast<const half2*>(&SUB)
    ); // 0,4
    frag_a[1] = __hfma2(
        *reinterpret_cast<half2*>(&hi_1),
        *reinterpret_cast<const half2*>(&MUL), *reinterpret_cast<const half2*>(&ADD)
    ); // 1,5
    frag_a[2] = __hsub2(
        *reinterpret_cast<half2*>(&lo_2),
        *reinterpret_cast<const half2*>(&SUB)
    ); // 2,6
    frag_a[3] = __hfma2(
        *reinterpret_cast<half2*>(&hi_2),
        *reinterpret_cast<const half2*>(&MUL), *reinterpret_cast<const half2*>(&ADD)
    ); // 3,7

    return frag_a;
}


// Efficiently dequantize an int32 value into a full B-fragment of 4 fp16 values.
// We mostly follow the strategy in the link below, with some small changes:
// https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h
__device__ inline FragB lop3_dequant_2bit(int q) {
    const int LO = 0x00030003;
    const int HI = 0x00300030;
    const int EX = 0x64006400;

    // Shift right by 8 to now consider elt_45 and elt_67. Issue first to hide RAW dependency if we issue
    // immediately before required.
    const uint32_t top_i4s = q >> 8;

    // Guarantee that the `(a & b) | c` operations are LOP3s.
    int lo_1_a = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);            // 0,8
    int lo_1_b = lop3<(0xf0 & 0xcc) | 0xaa>(q >> 2, LO, EX);       // 1,9
    int hi_1_a = lop3<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);            // 2,10
    int hi_1_b = lop3<(0xf0 & 0xcc) | 0xaa>(q >> 2, HI, EX);       // 3,11
    int lo_2_a = lop3<(0xf0 & 0xcc) | 0xaa>(top_i4s, LO, EX);      // 4,12
    int lo_2_b = lop3<(0xf0 & 0xcc) | 0xaa>(top_i4s >> 2, LO, EX); // 5,13
    int hi_2_a = lop3<(0xf0 & 0xcc) | 0xaa>(top_i4s, HI, EX);      // 6,14
    int hi_2_b = lop3<(0xf0 & 0xcc) | 0xaa>(top_i4s >> 2, HI, EX); // 7,15


    // int hi_2 = lop3<(0xf0 & 0xcc) | 0xaa>(top_i4s, HI, EX);  // 3,7

    // We want signed int4 outputs, hence we fuse the `-8` symmetric zero point directly into `SUB` and `ADD`.
    const int SUB = 0x64006400; // {1024, 1024} 0x64086408 
    const int MUL = 0x2c002c00; // {1/16, 1/16}
    const int ADD = 0xd400d400; // {-64, -64} 0xd480d480 

    FragB frag_b;
    frag_b[0] = __hsub2(
        *reinterpret_cast<half2*>(&lo_1_a),
        *reinterpret_cast<const half2*>(&SUB)
    ); // 0,8
    frag_b[1] = __hsub2(
        *reinterpret_cast<half2*>(&lo_1_b),
        *reinterpret_cast<const half2*>(&SUB)
    ); // 1,9
    frag_b[2] = __hfma2(
        *reinterpret_cast<half2*>(&hi_1_a),
        *reinterpret_cast<const half2*>(&MUL), *reinterpret_cast<const half2*>(&ADD)
    ); // 2,10
    frag_b[3] = __hfma2(
        *reinterpret_cast<half2*>(&hi_1_b),
        *reinterpret_cast<const half2*>(&MUL), *reinterpret_cast<const half2*>(&ADD)
    ); // 3,11
    frag_b[4] = __hsub2(
        *reinterpret_cast<half2*>(&lo_2_a),
        *reinterpret_cast<const half2*>(&SUB)
    ); // 4,12
    frag_b[5] = __hsub2(
        *reinterpret_cast<half2*>(&lo_2_b),
        *reinterpret_cast<const half2*>(&SUB)
    ); // 5,13
    frag_b[6] = __hfma2(
        *reinterpret_cast<half2*>(&hi_2_a),
        *reinterpret_cast<const half2*>(&MUL), *reinterpret_cast<const half2*>(&ADD)
    ); // 6,14
    frag_b[7] = __hfma2(
        *reinterpret_cast<half2*>(&hi_2_b),
        *reinterpret_cast<const half2*>(&MUL), *reinterpret_cast<const half2*>(&ADD)
    ); // 7,15

    return frag_b;
}



//////////////////////////////////////////////////////////////////////////////
// Loading params
//////////////////////////////////////////////////////////////////////////////

template <typename Tensor0, typename Tensor1, typename Tensor2>
__forceinline__ __device__  
void 
load_params_Kchannel(
    Tensor0 & scales,
    Tensor1 & zeros,
    Tensor2 const& params,
    int tidx,
    int i,
    const int num_params
) {
    CUTE_UNROLL
    for (int m = 0; m < size<1>(scales); ++m) {
        CUTE_UNROLL
        for (int j = 0; j < size<0>(scales); ++j) {
            // seems no one can know why is this offset ... 
            scales(j, m, i) = params(m * num_params + j % num_params, 0  + 8 * i + 4 * (j / num_params) + tidx % 4);
            zeros(j, m, i)  = params(m * num_params + j % num_params, 64 + 8 * i + 4 * (j / num_params) + tidx % 4);
        }
    }
}

template <typename Tensor0_g, typename Tensor1_g, typename Tensor2_g>
__forceinline__ __device__  
void 
load_params_Ktensor(
    Tensor0_g & scales,
    Tensor1_g & zeros,
    Tensor2_g const& params,
    int tidx,
    const int num_params
) {
    CUTE_UNROLL
    for (int j = 0; j < size<0>(scales); ++j) {
        scales(j) = params(128 * (j / num_params / 2) + 0  + 32 * ((j / num_params) % 2) + tidx / 4, j % num_params);
        zeros(j)  = params(128 * (j / num_params / 2) + 64 + 32 * ((j / num_params) % 2) + tidx / 4, j % num_params);
        // scales(j) = params(0 + 32 * (j / num_params) + tidx / 4, j % num_params);
        // zeros(j)  = params(64 + 32 * (j / num_params) + tidx / 4, j % num_params);
    }

    // CUTE_UNROLL
    // for (int j = 0; j < size<0>(scales); ++j) {
    //     params(0  + 32 * (j / num_params) + threadIdx.x / 4, j % num_params) = scales(j);
    //     params(64 + 32 * (j / num_params) + threadIdx.x / 4, j % num_params) = zeros(j);
    // }
}

template <int num_bits, typename Tensor0, typename Tensor1, typename Tensor2>
__forceinline__ __device__  
void 
load_params_Vtensor(
    Tensor0 & scales,
    Tensor1 & zeros,
    Tensor2 const& params,
    int tidx,
    int i,
    const int num_params
) {
    const int num_params_2 = num_bits == 2 ? num_params / 2 : num_params;
    CUTE_UNROLL
    for (int j = 0; j < size<0>(scales); ++j) {
        // seems no one can know why is this offset ... 
        scales(j, i) = params(128 * (i / 8) + 0  + 8 * (i % 8) + 4 * (j / num_params_2) + tidx % 4, j % num_params_2);
        zeros(j, i)  = params(128 * (i / 8) + 64 + 8 * (i % 8) + 4 * (j / num_params_2) + tidx % 4, j % num_params_2);
    }
}

//////////////////////////////////////////////////////////////////////////////
// Dequantization
//////////////////////////////////////////////////////////////////////////////


template<int num_bits,
         class SourceEngine, class SourceLayout,  
         class TargetEngine, class TargetLayout,
         class ScaleEngine,  class ScaleLayout,
         class ZeroEngine,   class ZeroLayout>
struct dequant_kc_vt;

template<class SourceEngine, class SourceLayout,  
         class TargetEngine, class TargetLayout,
         class ScaleEngine,  class ScaleLayout,
         class ZeroEngine,   class ZeroLayout>
struct dequant_kc_vt<2, SourceEngine, SourceLayout, TargetEngine, TargetLayout, ScaleEngine, ScaleLayout, ZeroEngine, ZeroLayout> {
    static constexpr int num_bits = 2;
    CUTE_DEVICE static 
    void apply(cute::Tensor<SourceEngine   , SourceLayout   > const& source,  
               cute::Tensor<TargetEngine   , TargetLayout   > const& target,
               cute::Tensor<ScaleEngine    , ScaleLayout    > const& scales,
               cute::Tensor<ZeroEngine     , ZeroLayout     > const& zeros,
               const int num_params) {
        using TQ  = cute::uint16_t;
        using TQ2 = cute::uint32_t;
        using T   = typename TargetEngine::value_type;
        using T2  = __half2;
        const int num_params_ = num_params / 2;                // TODO: only for g128
        const int pack_num    = 4 / num_params_;               // TODO: check 4
        
        // vectorize the source and target
        auto scales_vec  = cute::recast<T2>(scales);
        auto zeros_vec   = cute::recast<T2>(zeros);
        auto source_vec  = cute::recast<TQ2>(source);
        auto target_vec  = cute::recast<T2>(target);

        const int channel_stride = size<0>(source_vec);
        
        CUTE_UNROLL
        for (int i = 0; i < cute::size<0>(source_vec); ++i) {

            CUTE_UNROLL
            for (int p = 0; p < cute::size<1>(source_vec); ++p) {
                auto src_crd = cute::make_coord(i, p);
                auto src_raw = source_vec(src_crd);
                auto src_val = lop3_dequant_2bit(src_raw);

                CUTE_UNROLL
                for (int j = 0; j < size<1>(target_vec); ++j) {
                    target_vec(i, j) = __hfma2(src_val[j], scales_vec(i + j / pack_num * channel_stride), zeros_vec(i + j / pack_num * channel_stride));
                }

                // target_vec(i,0) = __hfma2(src_val[0], scales_vec(i), zeros_vec(i));
                // target_vec(i,1) = __hfma2(src_val[1], scales_vec(i + 1 / pack_num * channel_stride), zeros_vec(i + 1 / pack_num * channel_stride));
                // target_vec(i,2) = __hfma2(src_val[2], scales_vec(i + 2 / pack_num * channel_stride), zeros_vec(i + 2 / pack_num * channel_stride));
                // target_vec(i,3) = __hfma2(src_val[3], scales_vec(i + 3 / pack_num * channel_stride), zeros_vec(i + 3 / pack_num * channel_stride));
                // target_vec(i,4) = __hfma2(src_val[4], scales_vec(i + 4 / pack_num * channel_stride), zeros_vec(i + 4 / pack_num * channel_stride));
                // target_vec(i,5) = __hfma2(src_val[5], scales_vec(i + 5 / pack_num * channel_stride), zeros_vec(i + 5 / pack_num * channel_stride));
                // target_vec(i,6) = __hfma2(src_val[6], scales_vec(i + 6 / pack_num * channel_stride), zeros_vec(i + 6 / pack_num * channel_stride));
                // target_vec(i,7) = __hfma2(src_val[7], scales_vec(i + 7 / pack_num * channel_stride), zeros_vec(i + 7 / pack_num * channel_stride));

                // target_vec(i,0) = __hfma2(src_val[0], scales_vec(0), zeros_vec(0));
                // target_vec(i,1) = __hfma2(src_val[1], scales_vec(0), zeros_vec(0));
                // target_vec(i,2) = __hfma2(src_val[2], scales_vec(0), zeros_vec(0));
                // target_vec(i,3) = __hfma2(src_val[3], scales_vec(0), zeros_vec(0));
                // target_vec(i,4) = __hfma2(src_val[4], scales_vec(0), zeros_vec(0));
                // target_vec(i,5) = __hfma2(src_val[5], scales_vec(0), zeros_vec(0));
                // target_vec(i,6) = __hfma2(src_val[6], scales_vec(0), zeros_vec(0));
                // target_vec(i,7) = __hfma2(src_val[7], scales_vec(0), zeros_vec(0));

                // target_vec(i,0) = src_val[0];
                // target_vec(i,1) = src_val[1];
                // target_vec(i,2) = src_val[2];
                // target_vec(i,3) = src_val[3];
                // target_vec(i,4) = src_val[4];
                // target_vec(i,5) = src_val[5];
                // target_vec(i,6) = src_val[6];
                // target_vec(i,7) = src_val[7];
            }
        }
    }
};

template<class SourceEngine, class SourceLayout,  
         class TargetEngine, class TargetLayout,
         class ScaleEngine,  class ScaleLayout,
         class ZeroEngine,   class ZeroLayout>
struct dequant_kc_vt<4, SourceEngine, SourceLayout, TargetEngine, TargetLayout, ScaleEngine, ScaleLayout, ZeroEngine, ZeroLayout> {
    static constexpr int num_bits = 4;
    CUTE_DEVICE static 
    void apply(cute::Tensor<SourceEngine   ,SourceLayout   > const& source,  
               cute::Tensor<TargetEngine   ,TargetLayout   > const& target,
               cute::Tensor<ScaleEngine    ,ScaleLayout    > const& scales,
               cute::Tensor<ZeroEngine     ,ZeroLayout     > const& zeros,
               const int num_params) {
        using TQ  = cute::uint16_t;
        using TQ2 = cute::uint32_t;
        using T   = typename TargetEngine::value_type;
        using T2  = __half2;
        const int pack_num            = 4 / num_params;
        
        // vectorize the source and target
        auto scales_vec  = cute::recast<T2>(scales);
        auto zeros_vec   = cute::recast<T2>(zeros);
        auto source_vec  = cute::recast<TQ2>(source);
        auto target_vec  = cute::recast<T2>(target);

        const int channel_stride = cute::size<0>(source_vec);
        const int scales_stride  = cute::size<0>(scales_vec);

        CUTE_UNROLL
        for (int i = 0; i < cute::size<0>(source_vec); ++i)     // 2
        {
            CUTE_UNROLL
            for (int p = 0; p < cute::size<1>(source_vec); ++p) // 1
            {
                auto src_crd = cute::make_coord(i, p);
                auto src_raw = source_vec(src_crd);
                auto src_val = lop3_dequant(src_raw);

                auto col_offset = p * num_bits;

                auto tgt0_crd        = cute::make_coord(i, col_offset + 0);
                auto tgt1_crd        = cute::make_coord(i, col_offset + 1);
                auto tgt2_crd        = cute::make_coord(i, col_offset + 2);
                auto tgt3_crd        = cute::make_coord(i, col_offset + 3);

                // TODO: hard code for now 2
                int params_crd = i;

                target_vec(tgt0_crd) = __hfma2(src_val[0], scales_vec(params_crd + p * scales_stride), zeros_vec(params_crd + p * scales_stride));
                target_vec(tgt1_crd) = __hfma2(src_val[1], scales_vec(params_crd + 1 / pack_num * channel_stride + p * scales_stride), zeros_vec(params_crd + 1 / pack_num * channel_stride + p * scales_stride));
                target_vec(tgt2_crd) = __hfma2(src_val[2], scales_vec(params_crd + 2 / pack_num * channel_stride + p * scales_stride), zeros_vec(params_crd + 2 / pack_num * channel_stride + p * scales_stride));
                target_vec(tgt3_crd) = __hfma2(src_val[3], scales_vec(params_crd + 3 / pack_num * channel_stride + p * scales_stride), zeros_vec(params_crd + 3 / pack_num * channel_stride + p * scales_stride));

                // target_vec(tgt0_crd) = src_val[0];
                // target_vec(tgt1_crd) = src_val[1];
                // target_vec(tgt2_crd) = src_val[2];
                // target_vec(tgt3_crd) = src_val[3];
            }
        }
    }
};

template <int num_bits,
          class SourceEngine, class SourceLayout,  
          class TargetEngine, class TargetLayout,
          class ScaleEngine,  class ScaleLayout,
          class ZeroEngine,   class ZeroLayout>  
CUTE_DEVICE  
void  
dequant_Kchannel_Vtensor(
    cute::Tensor<SourceEngine   , SourceLayout   > const& source,  
    cute::Tensor<TargetEngine   , TargetLayout   > const& target,
    cute::Tensor<ScaleEngine    , ScaleLayout    > const& scales_vec,
    cute::Tensor<ZeroEngine     , ZeroLayout    >  const& zeros_vec,
    const int num_params=1
) {  
    dequant_kc_vt<num_bits, SourceEngine, SourceLayout, TargetEngine, TargetLayout, ScaleEngine, ScaleLayout, ZeroEngine, ZeroLayout>::apply(source, target, scales_vec, zeros_vec, num_params);
}

template <class SourceEngine, class SourceLayout,  
          class TargetEngine, class TargetLayout,
          typename TensorParamsG1, typename TensorParamsG2>
CUTE_DEVICE  
void  
dequantize_Ktensor(  
    cute::Tensor<SourceEngine   , SourceLayout   > const& source_,  
    cute::Tensor<TargetEngine   , TargetLayout   > & target_,
    TensorParamsG1 & scales_k_g_vec,
    TensorParamsG2 & zeros_k_g_vec,
    int num_bits,
    int group_size,
    int ii
) {  
    using TQ  = cute::uint16_t;
    using TQ2 = cute::uint32_t;
    using T   = typename TargetEngine::value_type;
    using T2  = __half2;

    // vectorize the source and target
    auto source = source_(_,_,_,_0{});
    auto target = target_(_,_,_,_0{});

    static constexpr int kNumBits    = 4;
    const int num_params             = 128 / group_size;
    const int ki                     = size<2>(target) / num_params;

    auto scales_k_g  = cute::recast<T>(scales_k_g_vec);
    auto zeros_k_g   = cute::recast<T>(zeros_k_g_vec);
    auto source_vec  = cute::recast<TQ2>(source);
    auto target_vec  = cute::recast<T2>(target);

    const int tile_j = size<2>(target) != size<2>(source) ? 2 : 1;

    CUTE_UNROLL
    for (int i = 0; i < cute::size<0>(source_vec); ++i)
    {
        auto src_crd = cute::make_coord(0, 0, 0);
        for (int p = 0; p < tile_j; ++p) {
            src_crd = tile_j == 1 ? cute::make_coord(i, 0, ii) : cute::make_coord(i, 0, 8 * (ii / 4) + ii % 4 + p * 4);
            auto src_raw = source_vec(src_crd);
            auto src_val = lop3_dequant(src_raw);

            auto col_offset = p * kNumBits;

            auto tgt0_crd      = cute::make_coord(i, col_offset + 0, ii);
            auto tgt1_crd      = cute::make_coord(i, col_offset + 1, ii);
            auto tgt2_crd      = cute::make_coord(i, col_offset + 2, ii);
            auto tgt3_crd      = cute::make_coord(i, col_offset + 3, ii);

            // Create half2 values for scales and zeros
            half2 scales_k_g_0 = __half2half2(__half(scales_k_g(ii / ki + col_offset * num_params + 0 * num_params)));
            half2 scales_k_g_1 = __half2half2(__half(scales_k_g(ii / ki + col_offset * num_params + 1 * num_params)));
            half2 scales_k_g_2 = __half2half2(__half(scales_k_g(ii / ki + col_offset * num_params + 2 * num_params)));
            half2 scales_k_g_3 = __half2half2(__half(scales_k_g(ii / ki + col_offset * num_params + 3 * num_params)));

            half2 zeros_k_g_0 = __half2half2(__half(zeros_k_g(ii / ki + col_offset * num_params + 0 * num_params)));
            half2 zeros_k_g_1 = __half2half2(__half(zeros_k_g(ii / ki + col_offset * num_params + 1 * num_params)));
            half2 zeros_k_g_2 = __half2half2(__half(zeros_k_g(ii / ki + col_offset * num_params + 2 * num_params)));
            half2 zeros_k_g_3 = __half2half2(__half(zeros_k_g(ii / ki + col_offset * num_params + 3 * num_params)));

            target_vec(tgt0_crd) = __hfma2(src_val[0], scales_k_g_0, zeros_k_g_0);
            target_vec(tgt1_crd) = __hfma2(src_val[1], scales_k_g_1, zeros_k_g_1);
            target_vec(tgt2_crd) = __hfma2(src_val[2], scales_k_g_2, zeros_k_g_2);
            target_vec(tgt3_crd) = __hfma2(src_val[3], scales_k_g_3, zeros_k_g_3);

            // target_vec(tgt0_crd) = src_val[0];
            // target_vec(tgt1_crd) = src_val[1];
            // target_vec(tgt2_crd) = src_val[2];
            // target_vec(tgt3_crd) = src_val[3];
        }
        

    }
    
}  

}  // namespace quant