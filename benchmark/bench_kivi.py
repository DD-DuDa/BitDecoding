import torch
import math
import triton

# Add KIVI path to import the modules
import sys
sys.path.insert(0, 'KIVI')

from quant.new_pack import triton_quantize_and_pack_along_last_dim
from quant.matmul import cuda_bmm_fA_qB_outer


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["seq_len"],
        x_vals=[2**i for i in range(10, 18, 1)],
        line_arg='provider',
        line_vals=['kivi-int4'],
        line_names=['KIVI INT4'],
        styles=[('green', '-')],
        ylabel="Time (ms)",
        plot_name="KIVI INT4 Attention Benchmark",
        args={},
    )
)
def benchmark(seq_len, provider):
    torch.random.manual_seed(0)
    device = "cuda"
    dtype = torch.float16

    batch_size = 1
    nheads = 32
    nheads_k = 8
    d = 128
    group_size = 128
    bits = 4

    # Query: shape (batch_size, nheads, 1, d) - single token decoding
    q = torch.randn(batch_size, nheads, 1, d, device=device, dtype=dtype)

    # K cache: shape (batch_size, nheads_k, seq_len, d)
    # Since seq_len % 128 == 0, we quantize all of K cache
    k_cache = torch.randn(batch_size, nheads_k, seq_len, d, device=device, dtype=dtype)

    # V cache: shape (batch_size, nheads_k, seq_len, d)
    v_cache = torch.randn(batch_size, nheads_k, seq_len, d, device=device, dtype=dtype)

    # Quantize K cache: need to transpose to (batch_size, nheads_k, d, seq_len) for quantization
    # triton_quantize_and_pack_along_last_dim expects (B, nh, D, T) and quantizes along T
    k_cache_trans = k_cache.transpose(2, 3).contiguous()  # (batch_size, nheads_k, d, seq_len)
    k_quant, k_scale, k_mn = triton_quantize_and_pack_along_last_dim(k_cache_trans, group_size, bits)
    # k_quant: (batch_size, nheads_k, d, seq_len // (32 // bits))
    # k_scale, k_mn: (batch_size, nheads_k, d, seq_len // group_size)

    # Quantize V cache: (batch_size, nheads_k, seq_len, d)
    # For V, we quantize along the last dim (d), so we need to reshape
    # Actually looking at the code, V is quantized differently - along the head_dim
    # But for simplicity, let's use the same approach as K (transpose and quantize)
    v_quant, v_scale, v_mn = triton_quantize_and_pack_along_last_dim(v_cache, group_size, bits)
    # v_quant: (batch_size, nheads_k, seq_len, d // (32 // bits))
    # v_scale, v_mn: (batch_size, nheads_k, seq_len, d // group_size)

    quantiles = [0.5, 0.2, 0.8]

    if provider == 'kivi-int4':
        def kivi_attention():
            # Step 1: Q @ K^T using cuda_bmm_fA_qB_outer
            # q: (batch_size, nheads, 1, d)
            # k_quant: (batch_size, nheads_k, d, seq_len // feat_per_int)
            # k_scale, k_mn: (batch_size, nheads_k, d, seq_len // group_size)
            attn_weights = cuda_bmm_fA_qB_outer(group_size, q, k_quant, k_scale, k_mn, bits)
            # attn_weights: (batch_size, nheads, 1, seq_len)

            # Step 2: Scale and softmax
            attn_weights = attn_weights / math.sqrt(d)
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(dtype)

            # Step 3: attn_weights @ V using cuda_bmm_fA_qB_outer
            # attn_weights: (batch_size, nheads, 1, seq_len)
            # v_quant: (batch_size, nheads_k, seq_len, d // feat_per_int)
            # v_scale, v_mn: (batch_size, nheads_k, seq_len, d // group_size)
            output = cuda_bmm_fA_qB_outer(group_size, attn_weights, v_quant, v_scale, v_mn, bits)
            # output: (batch_size, nheads, 1, d)

            return output

        ms, min_ms, max_ms = triton.testing.do_bench(
            kivi_attention,
            quantiles=quantiles
        )

    return ms, min_ms, max_ms


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size"],
        x_vals=[8, 32, 128],
        line_arg='provider',
        line_vals=['kivi-int4'],
        line_names=['KIVI INT4'],
        styles=[('orange', '-')],
        ylabel="Time (ms)",
        plot_name="KIVI INT4 Batch Attention Benchmark",
        args={'seq_len': 8192},
    )
)
def benchmark_batch_decode(batch_size, provider, seq_len=8192):
    torch.random.manual_seed(0)
    device = "cuda"
    dtype = torch.float16

    nheads = 32
    nheads_k = 8
    d = 128
    group_size = 128
    bits = 4

    # Query: shape (batch_size, nheads, 1, d) - single token decoding
    q = torch.randn(batch_size, nheads, 1, d, device=device, dtype=dtype)

    # K cache: shape (batch_size, nheads_k, seq_len, d)
    # Since seq_len % 128 == 0, we quantize all of K cache
    k_cache = torch.randn(batch_size, nheads_k, seq_len, d, device=device, dtype=dtype)

    # V cache: shape (batch_size, nheads_k, seq_len, d)
    v_cache = torch.randn(batch_size, nheads_k, seq_len, d, device=device, dtype=dtype)

    # Quantize K cache: need to transpose to (batch_size, nheads_k, d, seq_len) for quantization
    # triton_quantize_and_pack_along_last_dim expects (B, nh, D, T) and quantizes along T
    k_cache_trans = k_cache.transpose(2, 3).contiguous()  # (batch_size, nheads_k, d, seq_len)
    k_quant, k_scale, k_mn = triton_quantize_and_pack_along_last_dim(k_cache_trans, group_size, bits)
    # k_quant: (batch_size, nheads_k, d, seq_len // (32 // bits))
    # k_scale, k_mn: (batch_size, nheads_k, d, seq_len // group_size)

    # Quantize V cache: (batch_size, nheads_k, seq_len, d)
    # For V, we quantize along the last dim (d), so we need to reshape
    # Actually looking at the code, V is quantized differently - along the head_dim
    # But for simplicity, let's use the same approach as K (transpose and quantize)
    v_quant, v_scale, v_mn = triton_quantize_and_pack_along_last_dim(v_cache, group_size, bits)
    # v_quant: (batch_size, nheads_k, seq_len, d // (32 // bits))
    # v_scale, v_mn: (batch_size, nheads_k, seq_len, d // group_size)

    quantiles = [0.5, 0.2, 0.8]

    if provider == 'kivi-int4':
        def kivi_attention():
            # Step 1: Q @ K^T using cuda_bmm_fA_qB_outer
            # q: (batch_size, nheads, 1, d)
            # k_quant: (batch_size, nheads_k, d, seq_len // feat_per_int)
            # k_scale, k_mn: (batch_size, nheads_k, d, seq_len // group_size)
            attn_weights = cuda_bmm_fA_qB_outer(group_size, q, k_quant, k_scale, k_mn, bits)
            # attn_weights: (batch_size, nheads, 1, seq_len)

            # Step 2: Scale and softmax
            attn_weights = attn_weights / math.sqrt(d)
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(dtype)

            # Step 3: attn_weights @ V using cuda_bmm_fA_qB_outer
            # attn_weights: (batch_size, nheads, 1, seq_len)
            # v_quant: (batch_size, nheads_k, seq_len, d // feat_per_int)
            # v_scale, v_mn: (batch_size, nheads_k, seq_len, d // group_size)
            output = cuda_bmm_fA_qB_outer(group_size, attn_weights, v_quant, v_scale, v_mn, bits)
            # output: (batch_size, nheads, 1, d)

            return output

        ms, min_ms, max_ms = triton.testing.do_bench(
            kivi_attention,
            quantiles=quantiles
        )

    return ms, min_ms, max_ms


if __name__ == "__main__":
    benchmark.run(show_plots=True, print_data=True)
    # benchmark_batch_decode.run(show_plots=True, print_data=True)
