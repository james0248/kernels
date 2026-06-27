import torch
import triton

from kernels.ops.vector_addition import add

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],
        x_vals=[2**i for i in range(12, 28, 1)],
        x_log=True,
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "PyTorch"],
        styles=[("blue", "-"), ("red", "-")],
        ylabel="GB/s",
        plot_name="vector-add-perf",
        args={},
    )
)
def benchmark(size: int, provider: str):
    x = torch.randn(size, device=DEVICE, dtype=torch.float32)
    y = torch.randn(size, device=DEVICE, dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]

    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: x + y, quantiles=quantiles
        )  # pyright: ignore[reportGeneralTypeIssues]
    elif provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: add(x, y), quantiles=quantiles
        )  # pyright: ignore[reportGeneralTypeIssues]
    else:
        raise ValueError(f"unknown provider: {provider}")

    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)

    return gbps(ms), gbps(min_ms), gbps(max_ms)


if __name__ == "__main__":
    benchmark.run(print_data=True, show_plots=True)
