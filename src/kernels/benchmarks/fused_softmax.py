import torch
import triton

from kernels.ops.fused_softmax import naive_softmax, softmax

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[128 * i for i in range(2, 100)],
        line_arg="provider",
        line_vals=["triton", "torch", "naive"],
        line_names=["Triton", "PyTorch", "Naive"],
        styles=[("blue", "-"), ("green", "-"), ("red", "-")],
        ylabel="GB/s",
        plot_name="softmax-perf",
        args={"M": 4096},
    )
)
def benchmark(N: int, M: int, provider: str):
    x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
    stream = getattr(torch, DEVICE.type).Stream()
    getattr(torch, DEVICE.type).set_stream(stream)

    if provider == "torch":
        ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=1))
    elif provider == "triton":
        ms = triton.testing.do_bench(lambda: softmax(x))
    elif provider == "naive":
        ms = triton.testing.do_bench(lambda: naive_softmax(x))
    else:
        raise ValueError(f"unknown provider: {provider}")

    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)


if __name__ == "__main__":
    benchmark.run(print_data=True, show_plots=True, save_path="outputs")
