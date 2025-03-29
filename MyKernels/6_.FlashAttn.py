import math
import cupy as cp
import torch
import triton
import triton.language as tl

DEVICE = torch.device("cuda:0")

def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"

def is_hip_mi200():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == 'hip' and target.arch == 'gfx90a'


def get_cuda_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),

        # Good config for fp8 inputs.
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4)
    ]


def get_hip_autotune_config():
    return [
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 1, 'waves_per_eu': 2},
            num_warps=4, num_stages=2),
        triton.Config(
            {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 4, 'waves_per_eu': 2},
            num_warps=8, num_stages=2),
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 1, 'waves_per_eu': 2},
            num_warps=8, num_stages=2),
        triton.Config(
            {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'waves_per_eu': 3},
            num_warps=4, num_stages=2),
        triton.Config(
            {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 1, 'waves_per_eu': 8},
            num_warps=4, num_stages=2),
    ]


def get_autotune_config():
    if is_cuda():
        return get_cuda_autotune_config()
    else:
        return get_hip_autotune_config()
    




@triton.jit
def _flash_attn(
        # Pointers to matrices
        Q_ptr, # (N, d)
        K_ptr, # (N, d)
        V_ptr, # (N, d)
        l_ptr, # (N)
        m_ptr, # (N)
        O_ptr, # (N, d)
        # Matrix dimensions
        N, d: tl.constexpr,
        # Block sizes
        B_c: tl.constexpr, B_r: tl.constexpr,
        # Number of blocks
        T_c, T_r
    ):
    
    # -----------------------------------------------------------
    # Map the program ids `pid` to the block C it should compute.
    pid_c = tl.program_id(axis=0)
    pid_r = tl.program_id(axis=1)
    # Starting index on c and r
    c_start_idx = pid_c * B_c * d
    r_start_idx = pid_r * B_r * d
    r_start_idx_no_dim = pid_r * B_r
    # index for each part of the dimension
    dims = tl.arange(0, d)
    # Q/O pointers of shape (B_r, d). Each row is incremented by a value of d
    QO_ptrs = r_start_idx + (tl.arange(0, B_r)*d)[:, None] + dims[None, :]
    # l and m pointers are of shape (B_r).
    lm_ptrs = r_start_idx_no_dim + tl.arange(0, B_r)
    # K/V pointers of shape (B_c, d). Each row is incremented by a value of d
    KV_ptrs = c_start_idx + (tl.arange(0, B_c)*d)[:, None] + dims[None, :]
    # Used for masking. No index value should be larger than N*d
    largest_idx = N * d

    # Load in the data
    Q = tl.load(Q_ptr + QO_ptrs, mask=QO_ptrs < largest_idx, other=0.0)
    O = tl.load(O_ptr + QO_ptrs, mask=QO_ptrs < largest_idx, other=0.0)
    K = tl.load(K_ptr + KV_ptrs, mask=KV_ptrs < largest_idx, other=0.0)
    V = tl.load(V_ptr + KV_ptrs, mask=KV_ptrs < largest_idx, other=0.0)
    l = tl.load(l_ptr + lm_ptrs, mask=lm_ptrs < N, other=0.0)
    m = tl.load(m_ptr + lm_ptrs, mask=lm_ptrs < N, other=0.0)

    # Compute inner product (B_r, B_c)
    S = tl.dot(Q, K.permute(1, 0))
    # Compute the row-wise max (B_r)
    m_tilde = tl.max(S, axis=-1)
    # Exponentiate the inner product, but subtract the max for stability (B_r, B_c)
    P_tilde = tl.exp(S - m_tilde[:, None])
    # Row-wise sumfor denominator (B_r)
    l_tilde = tl.sum(P_tilde, axis=-1)

    # Get the max between the current row max up to block j
    # and the max of this block. (B_r)
    m_new = tl.maximum(m, m_tilde)
    # Update denominator (B_r)
    l_new = tl.exp(m - m_new) * l + tl.exp(m_tilde - m_new) * l_tilde

    # Write to output
    tl.store(
        O_ptr + QO_ptrs, 
        (1/l_new[:, None])*(l_new[:, None] * O + tl.exp(m - m_new)[:, None]*tl.dot(P_tilde, V)), 
        mask=QO_ptrs < largest_idx)
    tl.store(l_ptr + lm_ptrs, l_new, mask=lm_ptrs < N)
    tl.store(m_ptr + lm_ptrs, m_new, mask=lm_ptrs < N)



def flash_attn(Q, K, V, dtype=torch.float32):
    # Check constraints
    assert Q.shape[0] == K.shape[0] == V.shape[0], "One of the dimensions does not match"
    assert Q.shape[1] == K.shape[1] == V.shape[1], "One of the dimensions does not match"
    assert Q.is_contiguous(), "Matrix Q must be contiguous"
    assert K.is_contiguous(), "Matrix K must be contiguous"
    assert V.is_contiguous(), "Matrix V must be contiguous"

    # Get dimensions
    N, d = Q.shape

    # Get SRAM size
    device = cp.cuda.Device(Q.device.index)
    M = device.attributes["MaxSharedMemoryPerBlock"]

    # get block sizes
    # A block will first tile the column dimension, then the row dimension. That is, first we
    # allocate blocks over the entirety of a single row and allocate this block over multiple
    # rows if we have space to do so.
    B_c = math.ceil(M / (4*d)) # Blocks over the columns (N) - keys and values
    B_c = 2**math.floor(math.log2(B_c)) # Nearest power of 2
    # B_c = max(B_c, 16) # Cannot be smaller than 16
    B_r = min(B_c, d) # Blocks over the rows (N) - output, queries, denominators (l), and max values (m)

    # Initialize output (N, d), denominator (N), and max values (N) for each row
    O = torch.zeros((N, d), device=Q.device, dtype=dtype)
    l = torch.zeros((N), device=Q.device, dtype=dtype)
    m = torch.zeros((N), device=Q.device, dtype=dtype)

    # We need to get the number of blocks over the (queries, denominators, outputs, and maxes) and (keys and values)
    T_r = math.ceil(N / B_r) # Number of blocks over the rows - output, queries, denominators (l), and max values (m)
    T_c = math.ceil(N / B_c) # Number of blocks over the columns - keys and values

    # 2D launch kernel over blocks of size (B_c, B_r)
    grid = (T_c, T_r)

    Q_ = Q.clone()
    K_ = K.clone()
    V_ = V.clone()

    # Launch kernel
    _flash_attn[grid](
        Q, K, V,
        l, m,
        O,
        N, d,
        B_c, B_r,
        T_c, T_r
    )

    return O










torch.manual_seed(0)
N = 512
d = 1024
Q = torch.randn((N, d), device=DEVICE)
K = torch.randn_like(Q)
V = torch.randn_like(Q)

O_ = (Q @ K.mT).softmax(-1) @ V
O = flash_attn(Q, K, V, dtype=torch.float32)





torch.manual_seed(0)
a = torch.randn((512, 512), device=DEVICE, dtype=torch.float16)
b = torch.randn((512, 512), device=DEVICE, dtype=torch.float16)
triton_output = matmul(a, b)
torch_output = torch.matmul(a, b)
print(f"triton_output_with_fp16_inputs={triton_output}")
print(f"torch_output_with_fp16_inputs={torch_output}")
# Bigger tolerance for AMD MI200 devices.
# MI200 devices use reduced precision fp16 and bf16 and flush input and
# output denormal values to zero. Detailed info is at: https://pytorch.org/docs/stable/notes/numerical_accuracy.html#reduced-precision-fp16-and-bf16-gemms-and-convolutions-on-amd-instinct-mi200-devices
rtol = 1e-2 if is_hip_mi200() else 0
if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=rtol):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")

TORCH_HAS_FP8 = hasattr(torch, "float8_e5m2")
if TORCH_HAS_FP8 and is_cuda():
    torch.manual_seed(0)
    a = torch.randn((512, 512), device=DEVICE, dtype=torch.float16)
    b = torch.randn((512, 512), device=DEVICE, dtype=torch.float16)
    a = a.to(torch.float8_e5m2)
    # pre-transpose b for efficiency.
    b = b.T
    b = b.to(torch.float8_e5m2)
    triton_output = matmul(a, b)
    torch_output = torch.matmul(a.to(torch.float16), b.to(torch.float16))
    print(f"triton_output_with_fp8_inputs={triton_output}")
    print(f"torch_output_with_fp8_inputs={torch_output}")
    if torch.allclose(triton_output, torch_output, atol=0.125, rtol=0):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")










ref_lib = 'cuBLAS' if is_cuda() else 'rocBLAS'

configs = []
for fp8_inputs in [False, True]:
    if fp8_inputs and (not TORCH_HAS_FP8 or not is_cuda()):
        continue
    configs.append(
        triton.testing.Benchmark(
            x_names=["M", "N", "K"],  # Argument names to use as an x-axis for the plot
            x_vals=[128 * i for i in range(2, 33)],  # Different possible values for `x_name`
            line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
            # Possible values for `line_arg`
            # Don't compare to cublas for fp8 cases as torch.matmul doesn't support fp8 at the moment.
            line_vals=["triton"] if fp8_inputs else [ref_lib.lower(), "triton"],  # Label name for the lines
            line_names=["Triton"] if fp8_inputs else [ref_lib, "Triton"],  # Line styles
            styles=[("green", "-"), ("blue", "-")],
            ylabel="TFLOPS",  # Label name for the y-axis
            plot_name="matmul-performance-" +
            ("fp16" if not fp8_inputs else "fp8"),  # Name for the plot, used also as a file name for saving the plot.
            args={"fp8_inputs": fp8_inputs},
        ))


@triton.testing.perf_report(configs)
def benchmark(M, N, K, provider, fp8_inputs):
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
    if TORCH_HAS_FP8 and fp8_inputs:
        a = a.to(torch.float8_e5m2)
        b = b.T
        b = b.to(torch.float8_e5m2)
    quantiles = [0.5, 0.2, 0.8]
    if provider == ref_lib.lower():
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b), quantiles=quantiles)
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


benchmark.run(show_plots=True, print_data=True, save_path="./")