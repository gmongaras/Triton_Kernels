import torch
import triton
import triton.language as tl
from triton.runtime import driver
import math

DEVICE = torch.device("cuda:0")










@triton.jit
def add_kernel(
        output_ptr,
        a_ptr,
        b_ptr,
        num_blocks_N,
        num_blocks_d,
        N,
        d,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_d: tl.constexpr,
    ):

    # Get block index
    block_idx_n = tl.program_id(0)
    block_idx_d = tl.program_id(1)

    # Get start of this block
    block_start_N = block_idx_n * BLOCK_SIZE_N
    block_start_d = block_idx_d * BLOCK_SIZE_d

    # Get offsets for this block
    offsets_N = (block_start_N + tl.arange(0, BLOCK_SIZE_N)) * d
    offsets_d = block_start_d + tl.arange(0, BLOCK_SIZE_d)
    offsets = offsets_N[:, None] + offsets_d[None, :]

    # Mask - where are the offsets out of bounds?
    mask_N = block_start_N + tl.arange(0, BLOCK_SIZE_N) < N
    mask_d = offsets_d < d
    mask = mask_N[:, None] & mask_d[None, :]

    # Load the data
    a = tl.load(a_ptr + offsets, mask = mask)
    b = tl.load(b_ptr + offsets, mask = mask)

    # Add the data
    out = a + b

    # Save the data to the output vector
    tl.store(output_ptr + offsets, out, mask = mask)









# Helper function to queue kernels
def add(x, y):
    # Get the size of the vectors
    N, d = x.shape
    assert x.shape[0] == y.shape[0]
    assert x.shape[1] == y.shape[1]

    # Get the number of blocks and block size
    BLOCK_SIZE_N = BLOCK_SIZE_d = 32
    num_blocks_N = math.ceil(N / BLOCK_SIZE_N)
    num_blocks_d = math.ceil(d / BLOCK_SIZE_d)

    # Create output tensor
    output = torch.empty_like(x)

    # # kernel
    # kernel = add_kernel.warmup(
    #     output,
    #     x,
    #     y,
    #     n_blocks,
    #     BLOCK_SIZE,

    #     # num_warps=num_warps,
    #     grid=(1, )
    # )
    # kernel._init_handles()
    # n_regs = kernel.n_regs # Number of registers
    # size_smem = kernel.metadata.shared

    add_kernel[(BLOCK_SIZE_N, BLOCK_SIZE_d, 1)](output, x, y, num_blocks_N, num_blocks_d, N, d, BLOCK_SIZE_N, BLOCK_SIZE_d)

    return output






torch.manual_seed(0)
x = torch.randn(256, 781, device=DEVICE)
y = torch.randn(256, 781, device=DEVICE)
y_triton = add(x, y)
y_torch = x + y
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
        plot_name="vector-add-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={"M": 781},  # values for function arguments not in `x_names` and `y_name`
    ))
def benchmark(M, N, provider):
    x = torch.randn(N, M, device=DEVICE, dtype=torch.float32)
    y = torch.randn(N, M, device=DEVICE, dtype=torch.float32)
    stream = getattr(torch, DEVICE.type).Stream()
    getattr(torch, DEVICE.type).set_stream(stream)
    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: x + y)
    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: add(x, y))
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)



benchmark.run(show_plots=True, print_data=True, save_path="./")