import torch

import triton
import triton.language as tl


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"

DEVICE = torch.device("cuda:0")



# Kernel to add two vectors
@triton.jit
def add_kernel(
        x_ptr, # Pointer to x vector
        y_ptr, # Pointer to y vector
        output_ptr, # Pointer to output vector
        n_elements, # Size of vectors
        BLOCK_SIZE: tl.constexpr, # Number of elements in each thread block
        # `constexpr` so it can be used for shape values
    ):

    # Identify the thread block here
    # Tirton calls each thread block "programs"
    # Axis is zero since we are launching a 1d grid
    pid = tl.program_id(axis=0)

    # Each program processes a part of the input vector
    # For example, if the vector is of length 256 and block size of 64,
    # The blocks access elements [0:64, 64:128, 128:192, 192:256]
    # Note that offsets is a list of pointers:
    block_start = pid * BLOCK_SIZE # Start of block
    offsets = block_start + tl.arange(0, BLOCK_SIZE) # Get offsets for each element in the block
    
    # Mask to guard memory operations so they don't go out-of-bounds.
    mask = offsets < n_elements

    # Load x and y from DRAM to SRAM, masking out elements that are out of bounds
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # Add the vectors
    output = x + y

    # Write x + y back to DRAM
    tl.store(output_ptr + offsets, output, mask=mask)



# Helper function to allocate output tensor
# and launch kernel
def add(
        x: torch.Tensor,
        y: torch.Tensor
    ):

    # Preallocate the output
    output = torch.empty_like(x)
    assert x.device == DEVICE and \
           y.device == DEVICE and \
           output.device == DEVICE
    n_elements = output.numel()

    # The SPMD launch grid denotes the number of kernel instances running in parallel.
    # It is the same as a  CUDA lanch grid
    # It can either be Tuple[int] or Callable(metaparams) -> Tuple[int]
    # Here, we make a 1d grid where the grid size is the number of blockss
    # (Num elements total, block size)
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )

    # NOTE:
    #  - Each torch.Tensor is implicitly converted into a pointer to the first element
    #  - `triton.jit`'ed functions can be indexed with a launch grid to obtain a callable GPU kernel
    #  - Don't forget to pass meta-parameters as keyword arguments.
    add_kernel[grid](
        x,
        y,
        output,
        n_elements,
        BLOCK_SIZE=1024
    )

    # Return a handle to the output.
    # Since torch.cuda.synchronize() hasn't been called, 
    # the kernel is still running asynchronously
    return output




# Run the kernel
torch.manual_seed(0)
size = 98432
x = torch.rand(size, device=DEVICE)
y = torch.rand(size, device=DEVICE)
output_torch = x + y
output_triton = add(x, y)
print(output_torch, output_triton)





# Benchmarking against normal torch
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"], # ARgument names to use as x-axis for plot
        x_vals=[2**i for i in range(12, 28, 1)], # Possible values of x
        x_log=True, # x axis logarithmic
        line_arg="provider", # Name of different lines in the plot.
        line_vals=["triton", "torch"], # Values for the "line_arg"
        line_names=["Triton", "Torch"], # Labels for each line
        styles=[("blue", "-"), ("green", "-")],
        ylabel="GB/s",
        plot_name="vector-add-performance",
        args={},
    ))
def benchmark(size, provider):
    x = torch.rand(size, device=DEVICE, dtype=torch.float32)
    y = torch.rand(size, device=DEVICE, dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y), quantiles=quantiles)
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)
benchmark.run(print_data=True, show_plots=True, save_path="./")