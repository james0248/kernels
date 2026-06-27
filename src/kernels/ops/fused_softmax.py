import torch

import triton
import triton.language as tl
from triton.runtime import driver

DEVICE = driver.active.get_active_torch_device()


def naive_softmax(x: torch.Tensor) -> torch.Tensor:
    x_max = x.max(dim=1).values
    z = x - x_max[:, None]
    numerator = z.exp()
    denominator = numerator.sum(dim=1)
    softmax = numerator / denominator[:, None]
    return softmax


@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr,
):
    row_start = tl.program_id(0)
    row_steps = tl.num_programs(0)

    for row_offset in tl.range(row_start, n_rows, row_steps, num_stages=num_stages):
        row_start_ptr = input_ptr + row_offset * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols

        row = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float("inf"))
        row_max = tl.max(row)
        z = row - row_max
        numerator = tl.exp(z)
        denominator = tl.sum(numerator)
        softmax = numerator / denominator

        output_row_start_ptr = output_ptr + row_offset * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax, mask=mask)


properties = driver.active.utils.get_device_properties(DEVICE.index)
NUM_SM = properties["multiprocessor_count"]  # total number of SMs in the device
NUM_REGS = properties["max_num_regs"]  # total number of registers in the device
SMEM_SIZE = properties["max_shared_mem"]  # total amount of shared memory per SM
WARP_SIZE = properties["warpSize"]  # number of threads per warp


def softmax(x: torch.Tensor) -> torch.Tensor:
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    num_warps = 8  # warps per program
    num_stages = 4 if SMEM_SIZE > 200_000 else 2  # use more stages for larger shared memory

    y = torch.empty_like(x)

    kernel = softmax_kernel.warmup(
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_rows,
        n_cols,
        BLOCK_SIZE,
        num_stages=num_stages,
        num_warps=num_warps,
        grid=(1,),
    )

    kernel._init_handles()
    n_regs = kernel.n_regs  # number of registers per thread
    size_smem = kernel.metadata.shared  # amount of shared memory per program

    # occupancy = max number of programs per SM
    occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
    occupancy = min(occupancy, SMEM_SIZE // size_smem)

    # total number of programs on the device
    num_programs = NUM_SM * occupancy
    num_programs = min(num_programs, n_rows)

    kernel[(num_programs, 1, 1)](y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE, num_stages)
    return y


def validate():
    torch.manual_seed(0)
    x = torch.randn(1823, 781, device=DEVICE)
    y_triton = softmax(x)
    y_torch = torch.softmax(x, axis=1)
    assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)


if __name__ == "__main__":
    validate()
