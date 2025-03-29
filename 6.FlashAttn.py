import pytest
import torch
import triton.tools.experimental_descriptor

import triton
import triton.language as tl

# ENABLE_LHS_TO_TMEM is an experimental environment variable for Blackwell.
# If it is set to 1 it can improve performance of Blackwell attention. However,
# it defaults to 0 as it is known to cause correctness issues outside of the
# _attn_fwd_tma kernel below.

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def supports_tma():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9


HAS_TMA_DESC = "nv_tma_desc_type" in dir(tl)

if HAS_TMA_DESC:
    print("TMA benchmarks will be running with experimental grid constant TMA descriptor.", )
else:
    print("TMA benchmarks will be running without grid constant TMA descriptor.", )

# TmaAutoTuneHelper used in htyu's PR #5622
class TmaAutoTuneHelper:

    # duck typing wrapper to implement the same interface as TmaDescKernelParam in Triton PR #4498
    class KernelParamWrapper:

        def __init__(self, desc):
            self.desc = desc

        def tma_desc_cpu_ptr(self):
            return self.desc.data_ptr()

    TMA_SIZE = 128

    def __init__(self):
        self.fill_1d_tma_descriptor_inner = (triton.runtime.driver.active.utils.fill_1d_tma_descriptor)
        self.fill_2d_tma_descriptor_inner = (triton.runtime.driver.active.utils.fill_2d_tma_descriptor)
        if HAS_TMA_DESC:
            self.descriptors = {}
        else:
            self.cuda_descriptors = {}

    # Call this method outside of the lambda function for grid size
    def init_tma_descriptor(self, name):
        if HAS_TMA_DESC:
            self.descriptors[name] = torch.empty(TmaAutoTuneHelper.TMA_SIZE, device="cpu", dtype=torch.int8)
        else:
            self.cuda_descriptors[name] = torch.empty(TmaAutoTuneHelper.TMA_SIZE, device="cuda", dtype=torch.int8)

    # Call this method inside the lambda function for grid size
    def fill_1d_tma_descriptor(self, name, ptr, dim, block_dim, element_size):
        if HAS_TMA_DESC:
            desc_x = self.descriptors[name]
            assert desc_x.data_ptr() % 64 == 0
            self.fill_1d_tma_descriptor_inner(ptr, dim, block_dim, element_size, desc_x.data_ptr())
        else:
            desc_x = self.cuda_descriptors[name]
            buf_x = torch.empty_like(desc_x, device="cpu", pin_memory=True)
            self.fill_1d_tma_descriptor_inner(ptr, dim, block_dim, element_size, buf_x.data_ptr())
            desc_x.copy_(buf_x, non_blocking=True)

    # Call this method inside the lambda function for grid size
    def fill_2d_tma_descriptor(self, name, ptr, dim1, dim0, block_dim1, block_dim0, element_size):
        if HAS_TMA_DESC:
            desc_x = self.descriptors[name]
            assert desc_x.data_ptr() % 64 == 0
            self.fill_2d_tma_descriptor_inner(ptr, dim1, dim0, block_dim1, block_dim0, element_size, desc_x.data_ptr())
        else:
            desc_x = self.cuda_descriptors[name]
            buf_x = torch.empty_like(desc_x, device="cpu", pin_memory=True)
            self.fill_2d_tma_descriptor_inner(ptr, dim1, dim0, block_dim1, block_dim0, element_size, buf_x.data_ptr())
            desc_x.copy_(buf_x, non_blocking=True)

    def get_tma_descriptor_kernel_param(self, name):
        if HAS_TMA_DESC:
            assert self.descriptors[name] is not None
            return self.KernelParamWrapper(self.descriptors[name])
        else:
            assert self.cuda_descriptors[name] is not None
            return self.cuda_descriptors[name]
        























# Function used to calculate the inner iteration over
# the entire row.
@triton.jit
def _attn_fwd_inner(
        acc, # (B_r) Acc is the attention output.
        l_i, # (B_r) l_i is the denominator for the ith column. Basically, accumulating the denominator in the row.
        m_i, # (B_r) m_i is similar to l_i, but accumulates the row max for stable exponentials
        q, # (B_r, d) Query for the entire row
        K_block_ptr, # (B_c, d) Pointer to the K block to compute
        V_block_ptr, # (B_c, d) Pointer to the V block to compute
        start_m, # ???
        qk_scale, # Scale for inner prod before softmax
        BLOCK_M: tl.constexpr, # Block size over columns
        HEAD_DIM: tl.constexpr, # dimension of each head
        BLOCK_N: tl.constexpr, # Block size over rows
        STAGE: tl.constexpr, # Stage 1 are noncausal blocks (off diag). Stage 2 is causal (diag).
        offs_m: tl.constexpr, # ???
        offs_n: tl.constexpr, # ???
        N_CTX: tl.constexpr, # ???
        fp8_v: tl.constexpr, # Save the output in float 8?
    ):

    # Range of values handled by this stage
    # Stage 1 are blocks that are noncausal. That is,
    # those blocks not on the diagonal and do not need masking
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    # Stage 2 are blocks that are causal. These are on the
    # diagonal and need masking
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    else: # causal = False
        lo, hi = 0, N_CTX

    # Move the K and V to the first row in the block
    V_block_ptr = tl.advance(K_block_ptr, (0, lo))
    K_block_ptr = tl.advance(V_block_ptr, (lo, 0))

    # loop over the rows (k, v) in blocks of size N, updating
    # the accumulators
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        # Compute inner product
        k = tl.load(K_block_ptr)
        qk = tl.dot(q, k) # (B_r, B_c)

        # We need to do masking if we are in a causal block
        if STAGE == 2:
            # Get causal mask
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            # Scale and mask
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            # Get new max value along this row
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            # Subtract max
            qk -= m_ij[:, None]
        else:
            # Get new rowmax
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            # Scale and subtract row max
            qk = qk * qk_scale - m_ij[:, None]
        
        # Exponentiate
        p = tl.math.exp2(qk)
        # Get row sum for this block
        l_ij = tl.sum(p, 1)
        # Update m_i and l_i for this row
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij

        # update the output by the new max value
        acc = acc * alpha[:, None]

        # Inner product with the values.
        # Add the results to the current output
        v = tl.load(V_block_ptr)
        if fp8_v:
            p = p.to(tl.float8e5)
        else:
            p = p.to(tl.float16)
        acc = tl.dot(p, v, acc)

        # Update the (m_i) max and move the V/K pointers
        # to the next block in the row.
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))

    return acc, l_i, m_i



# We don't run auto-tuning every time to keep the tutorial fast. Keeping
# the code below and commenting out the equivalent parameters is convenient for
# re-tuning.
configs = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w) \
    for BM in [64, 128]\
    for BN in [32, 64]\
    for s in ([1] if is_hip() else [3, 4, 7])\
    for w in [4, 8]\
]


def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    if BLOCK_M * BLOCK_N < 128 * 128 and conf.num_warps == 8:
        return False
    return True





@triton.autotune(list(filter(keep, configs)), key=["N_CTX", "HEAD_DIM"])
@triton.jit
def _attn_fwd(Q, K, V, sm_scale, M, Out, # Inputs, scale, mask, output
              stride_qz, stride_qh, stride_qm, stride_qk, # query strides for each dim in the tensor (batch, head, seq (m), dim)
              stride_kz, stride_kh, stride_kn, stride_kk, # key strides for each dim in the tensor (batch, head, seq (n), dim)
              stride_vz, stride_vh, stride_vk, stride_vn, # value strides for each dim in the tensor  (batch, head, dim, seq (n))
              stride_oz, stride_oh, stride_om, stride_on, # output strides for each dim in the tensor (batch, head, seq (m), seq (n))
              Z, H, N_CTX, # Z - batch size, H - num heads, N_CTX - ???
              HEAD_DIM: tl.constexpr, # Dimension of each head
              BLOCK_M: tl.constexpr, # Number of rows in each block (query/output)
              BLOCK_N: tl.constexpr, # Number of columns in each block (keys/values)
              STAGE: tl.constexpr # STAGE=1 for causal block (diag), STAGE=2 for noncausal block (offdiag)
              ):
    
    tl.static_assert(BLOCK_N <= HEAD_DIM)

    # First dim of grid - Which block is currently being operated on?
    start_m = tl.program_id(0)
    # Second dim of grid - Iterates over the batch (z) and head dim (h)
    off_hz = tl.program_id(1)
    off_z = off_hz // H # batch offset
    off_h = off_hz % H # head offset
    # Get base offset. This is the offset for the current
    # head and batch
    qkv_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

    # Block pointers
    Q_block_ptr = tl.make_block_ptr(
        base = Q + qkv_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    v_order: tl.constexpr = (0, 1) if V.dtype.element_ty == tl.float8e5 else (1, 0)
    V_block_ptr = tl.make_block_ptr(
        base=V + qkv_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=v_order,
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qkv_offset,
        shape=(HEAD_DIM, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + qkv_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )

    # Initialize offsets within this (M, N) block
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    # Initialize pointers for m (cur row max) and l (cur row denom)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    # Intialize output matrix for the current output batch
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    
    # Load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504 # 1/log(2)

    # Load q, it will stay in sram through the row iteration
    q = tl.load(Q_block_ptr)

    # Stage 1: off-band / off-diagonal
    # For caual = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1 and _attn_fwd_inner gets 3 as its STAGE
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,
                                        start_m, qk_scale,
                                        BLOCK_M, HEAD_DIM, BLOCK_N,
                                        4 - STAGE, offs_m, offs_n, N_CTX, V.dtype.elemtn_ty == tl.float8e5
                                        )
    # Stage 2: on-band / on-diagonal
    if STAGE & 2:
        # Barrier makes it easier for the compiler to
        # schedule the two inner loops independently
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,  #
                                        start_m, qk_scale,  #
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        2, offs_m, offs_n, N_CTX, V.dtype.element_ty == tl.float8e5  #
                                        )
    
    # Divide output by the denominator
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    # Store max and output
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))