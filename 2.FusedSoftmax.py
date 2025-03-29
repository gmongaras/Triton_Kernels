import torch

import triton
import triton.language as tl
from triton.runtime import driver

DEVICE = torch.device("cuda:0")




def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"
def is_cdna():
    return is_hip() and triton.runtime.driver.active.get_current_target().arch in ('gfx940', 'gfx941', 'gfx942',
                                                                                   'gfx90a', 'gfx908')



def naive_softmax(x):
    """
    Row-wise softmax of X using native PyTorch

    Subtract the maximum element to avoid overflows.
    Softmax is invariant to shift factors so this works.
    """
    # X of shape (M, N)
    # Get max for each row (M)
    # Read MN elements from DRAM to SRAM; Write M elements to DRAM
    x_max = x.max(dim=-1)[0]

    # Subtract the max element from each row
    # Read MN + M elements; Write MN elements
    z = x - x_max[:, None]
    
    # Exponentiate each value
    # Read MN elements; Write MN elements
    numerator = torch.exp(z)

    # Get the denominator, the sum of each row
    # Read MN elements; Write M elements
    denominator = numerator.sum(-1)

    # Divide the numerator by the denominator
    # Read MN + M elements; Write MN elements
    ret = numerator / denominator[:, None]

    # Total: 
    #     Read 5MN + 2M elements
    #     Wrote 3MN + 2M elements
    return ret




# Theoretically, we should be able to get a 4x speed increase
# if we fuse the operations into a single read and write
# (8MN + 4M) / 2MN
# Note that this assumes we can load the entire softmax row into SRAM
@triton.jit
def softmax_kernel(
        output_ptr,
        input_ptr,
        input_row_stride,
        output_row_stride,
        n_rows,
        n_cols,
        BLOCK_SIZE: tl.constexpr,
        num_stages: tl.constexpr
    ):

    # Starting row of this thread block / program
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)

    # Iterate over all rows
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        # The stride representas how much we need to increase the pointer 
        # to advance to the next row
        # input_row_stride is the size of each row
        # row_idx is the row index
        # (row_idx * input_row_stride) is the location of the current row
        row_start_ptr = input_ptr + row_idx * input_row_stride

        # The block size is the next power of two greater than n_cols,
        # so we can fit each row in a single block
        # Offsets of all columns within this row, within this block,
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets

        # Load the row into SRAM, using a mask since BLOCKS_ZIE may be > than n_cols
        mask = col_offsets < n_cols # Don't go out of bounds
        # Pad where the mask is False
        row = tl.load(input_ptrs, mask=mask, other=-float("inf"))

        # Subtract the max for numerical stability
        row_minus_max = row - tl.max(row, axis=0)

        # Note that exponentiation in Triton is fast, but approximate
        # Essentially __expf in CUDA
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator

        # Write back output to DRAM
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)



properties = driver.active.utils.get_device_properties(DEVICE.index)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
target = triton.runtime.driver.active.get_current_target()
kernels = {}

# Helper function to queue kernels
def softmax(x):
    n_rows, n_cols = x.shape

    # The block size of each loop iteration is the smallest power
    # of two greater than the numebr of columns i `x`
    # Note that this value is expected to fit into SRAM
    # and the row is not going to be tiled.
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # Trick:
    # We can ask the compiler to use more threads per row by
    # increasing the number of warps (`num_warps`) over which each row is distributed.
    # This value will be auto-tuned in  a more natural way in the next file
    # Note that in CUDA, each warp has 32 threads
    num_warps = 8

    # Number of software pipelining stages
    num_stages = 4 if SIZE_SMEM > 200_000 else 2

    # Allocate output
    y = torch.empty_like(x)

    # Pre-compile kernel to get register usage and computer thread occupancy.
    kernel = softmax_kernel.warmup(
        x, 
        y, 
        x.stride(0), 
        y.stride(0),
        n_rows,
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_stages=num_stages,
        num_warps=num_warps,
        grid=(1, )
    )
    kernel._init_handles()
    n_regs = kernel.n_regs # Number of registers
    size_smem = kernel.metadata.shared
    if is_hip():
        # NUM_REGS represents the number of regular purpose registers. On CDNA architectures this is half of all registers available.
        # However, this is not always the case. In most cases all registers can be used as regular purpose registers.
        # ISA SECTION (3.6.4 for CDNA3)
        # VGPRs are allocated out of two pools: regular VGPRs and accumulation VGPRs. Accumulation VGPRs are used
        # with matrix VALU instructions, and can also be loaded directly from memory. A wave may have up to 512 total
        # VGPRs, 256 of each type. When a wave has fewer than 512 total VGPRs, the number of each type is flexible - it is
        # not required to be equal numbers of both types.
        if is_cdna():
            NUM_GPRS = NUM_REGS * 2

        # MAX_NUM_THREADS represents maximum number of resident threads per multi-processor.
        # When we divide this number with WARP_SIZE we get maximum number of waves that can
        # execute on a CU (multi-processor)  in parallel.
        MAX_NUM_THREADS = properties["max_threads_per_sm"]
        max_num_waves = MAX_NUM_THREADS // WARP_SIZE
        occupancy = min(NUM_GPRS // WARP_SIZE // n_regs, max_num_waves) // num_warps
    else:
        occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
    occupancy = min(occupancy, SIZE_SMEM // size_smem)
    num_programs = NUM_SM * occupancy

    # Num programs / num thread blocks
    # Number of tiles we are splitting the rows into
    num_programs = min(num_programs, n_rows)

    # Create a number of persistent programs / thread blocks
    # NOTE: BLOCK_SIZE and num_stages not passed because they're static
    kernel[(num_programs, 1, 1)](y, x, x.stride(0), y.stride(0), n_rows, n_cols)
    return y




torch.manual_seed(0)
x = torch.randn(1823, 781, device=DEVICE)
y_triton = softmax(x)
y_torch = x.softmax(-1)
assert torch.allclose(y_triton, y_torch,)
# print(y_triton, y_torch)



@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # argument names to use as an x-axis for the plot
        x_vals=[128 * i for i in range(2, 100)],  # different possible values for `x_name`
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        line_vals=['triton', 'torch'],  # possible values for `line_arg``
        line_names=[
            "Triton",
            "Torch",
        ],  # label name for the lines
        styles=[('blue', '-'), ('green', '-')],  # line styles
        ylabel="GB/s",  # label name for the y-axis
        plot_name="softmax-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={'M': 4096},  # values for function arguments not in `x_names` and `y_name`
    ))
def benchmark(M, N, provider):
    x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
    stream = getattr(torch, DEVICE.type).Stream()
    getattr(torch, DEVICE.type).set_stream(stream)
    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1))
    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: softmax(x))
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)


benchmark.run(show_plots=True, print_data=True, save_path="./")