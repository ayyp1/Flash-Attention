import torch
import torch.nn.functional as F
import triton

from flash_attention import TritonAttention

DEVICE = "cuda"

# ponytail: no flash-attn/fp8/warp-specialize comparison — this repo's TritonAttention doesn't implement them.
configs = [
    triton.testing.Benchmark(
        x_names=["SEQ_LEN"],
        x_vals=[512, 1024, 2048, 4096],
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "Torch (SDPA)"],
        styles=[("red", "-"), ("blue", "-")],
        ylabel="TFLOPS",
        plot_name=f"attn-{mode}-causal={causal}",
        args={"BATCH": 4, "NUM_HEADS": 16, "HEAD_DIM": 64, "causal": causal, "mode": mode},
    )
    for mode in ["fwd", "bwd"]
    for causal in [True, False]
]


@triton.testing.perf_report(configs)
def bench_flash_attention(BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM, causal, mode, provider, device=DEVICE):
    assert mode in ["fwd", "bwd"]
    dtype = torch.float16
    requires_grad = mode == "bwd"
    q = torch.randn((BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device=device, requires_grad=requires_grad)
    k = torch.randn((BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device=device, requires_grad=requires_grad)
    v = torch.randn((BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device=device, requires_grad=requires_grad)
    sm_scale = 1.0 / (HEAD_DIM**0.5)

    if provider == "triton":
        fn = lambda: TritonAttention.apply(q, k, v, causal, sm_scale)
    else:
        fn = lambda: F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=sm_scale)

    if mode == "bwd":
        o = fn()
        do = torch.randn_like(o)
        fn = lambda: o.backward(do, retain_graph=True)

    ms = triton.testing.do_bench(fn)
    flops = 2.0 * BATCH * NUM_HEADS * SEQ_LEN * SEQ_LEN * HEAD_DIM * 2
    if causal:
        flops *= 0.5
    if mode == "bwd":
        flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
    return flops * 1e-12 / (ms * 1e-3)


def _sanity_check():
    """Smallest check that fails if the triton/torch dispatch (fwd output + bwd grads) above is wrong."""
    B, H, S, D = 1, 2, 128, 64
    scale = 1.0 / (D**0.5)
    for causal in (True, False):
        q = torch.randn(B, H, S, D, dtype=torch.float16, device=DEVICE, requires_grad=True)
        k = torch.randn(B, H, S, D, dtype=torch.float16, device=DEVICE, requires_grad=True)
        v = torch.randn(B, H, S, D, dtype=torch.float16, device=DEVICE, requires_grad=True)
        do = torch.randn(B, H, S, D, dtype=torch.float16, device=DEVICE)

        tri_out = TritonAttention.apply(q, k, v, causal, scale)
        tri_out.backward(do)
        tri_dq, tri_dk, tri_dv = q.grad.clone(), k.grad.clone(), v.grad.clone()
        q.grad = k.grad = v.grad = None

        ref_out = F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=scale)
        ref_out.backward(do)
        ref_dq, ref_dk, ref_dv = q.grad.clone(), k.grad.clone(), v.grad.clone()

        assert torch.allclose(tri_out, ref_out, atol=1e-2, rtol=0), f"fwd mismatch, causal={causal}"
        assert torch.allclose(tri_dq, ref_dq, atol=1e-2, rtol=0), f"dQ mismatch, causal={causal}"
        assert torch.allclose(tri_dk, ref_dk, atol=1e-2, rtol=0), f"dK mismatch, causal={causal}"
        assert torch.allclose(tri_dv, ref_dv, atol=1e-2, rtol=0), f"dV mismatch, causal={causal}"
    print("sanity check passed")


if __name__ == "__main__":
    _sanity_check()
    bench_flash_attention.run(print_data=True, save_path=".", show_plots=True)
