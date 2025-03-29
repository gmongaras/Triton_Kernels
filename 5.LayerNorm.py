# layernorm defined by
# y = ((x - mean[x])/sqrt(Var[x] + eps)) * w + b
# where w and b are learnable d-dim vectors



import torch

import triton
import triton.language as tl


try:
    # This is https://github.com/NVIDIA/apex, NOT the apex on PyPi, so it
    # should not be added to extras_require in setup.py.
    import apex
    HAS_APEX = True
except ModuleNotFoundError:
    HAS_APEX = False

DEVICE = torch.device("cuda")


# Basic forward kernel
@triton.jit
def _layer_norm_fwd_fused(
        X, # Pointer to the input
        Y, # Pointer to the output
        W, # Pointer to the scale weight vector
        B, # Pointer to the shift weight vector
        Mean, # Pointer to the mean of X
        Rstd, # Pointer to the 1/std of X
        stride, # How to increase the pointer by when moving by 1 row
        N, # Number of columns in X
        eps, # Epsilon to avoid division by zero
        BLOCK_SIZE: tl.constexpr
    ):

    # Our thread block will compute the output for an entire row.
    # Each block performs on its own row.

    # Map the program/block to the row of X and Y to compute
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride



    # Compute mean of X in float32 over the entire row
    mean = 0
    # _mean is of size BLOCK_SIZE so we don't need thread syncs.
    # each thread writes to its spot in the block
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    # Iterate over thw row in blocks
    for off in range(0, N, BLOCK_SIZE):
        # Get the offsets within the block
        cols = off + tl.arange(0, BLOCK_SIZE)
        # Load in the data within the block
        a = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        # Accumulate the sum for each thread
        _mean += a
    # Sum the values each thread accumulated and take the mean
    mean = tl.sum(_mean, axis=0) / N

    


    # Compute the variance of X in float32 over the entire row
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    # Iterate over the entire dimension in blocks
    for off in range(0, N, BLOCK_SIZE):
        # Get the indices for all threads in this block
        cols = off + tl.arange(0, BLOCK_SIZE)
        # Load in the data
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        # Subtract mean (x - \mu)
        x = tl.where(cols < N, x - mean, 0.)
        # Square and sum. Each thread writes to its own
        # place into the _var shared memory
        # (x - \mu) ** 2
        _var += x * x
    # Sum over all the accumulations from the threads
    # and get the variance
    # (\sum_{over dim} (x - \mu) ** 2) / N
    var = tl.sum(_var, axis=0) / N
    # Calculate the inverse standard deviation
    rstd = 1 / tl.sqrt(var + eps)

    # Write mean and rstd to output
    var = tl.store(Mean + row, mean)
    rstd = tl.store(Rstd + row, rstd)


    # Normalize X and apply the linear transformation
    # via block-wise iteration over the row
    for off in range(0, N, BLOCK_SIZE):
        # Load in data within this block
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)

        # Normalize x
        x_hat = (x - mean) * rstd

        # Linear transform
        y = x_hat * w + b

        # Write output
        tl.store(Y + cols, y, mask=mask)







# The backward pass looks like the following:
# \grad_x = 1/sigma * (
#     \grad_y * w
#     - (1/N)*x_hat @ (\grad_y * w) * x
#     - (1 / N) * \grad_y @ w
# )
# where c1 = (1/N)*x_hat @ (\grad_y * w)
#       c2 = (1 / N) * \grad_y @ w
# Then we can get the paramater gradients by:
# \grad_w = \grad_y * x_hat
# \grad_b = \grad_y
# 
# x is easy to deal with, but the grads of w and b are used throughout
# every kernel launch. Since we need to su mup the gradients,
# we need to use a parallel reduction strategy across
# rows / across thread blocks. This is done via a buffer
# that accumulates partial \grad_w and \grad)b across rows.
# Then we use a parallel reduction algorithm to sum them up.
@triton.jit
def _layer_norm_bwd_dx_fused(
        DX, # Pointer to the input gradient
        DY, # Pointer to the output gradient
        DW, # Pointer to the partial sum of weights gradient
        DB, # Pointer to the partial sum of biases gradient
        X, # Pointer to the input
        W, # Pointer to the weights
        Mean, # Pointer to the mean
        Rstd, # Pointer to the 1/std
        Lock, # Pointer to the lock for parallel reduction
        stride, # How much to increase the pointer by when moving by 1 row
        N, # Number of columns in X
        GROUP_SIZE_M: tl.constexpr, # Col group size for parallel reduction
        BLOCK_SIZE_N: tl.constexpr, # Block size along the row
    ):

    # Each thrad block will compute an entire row

    # Map the program id / thread block id to the number
    # of elements of X, DX, and DY it will compute
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N
    X += row * stride
    DY += row * stride
    DX += row * stride

    # Offset locks and offset the weights/biases gradie pointer
    # for parallel reductioon. Essentially, what group
    # will this block be writing to?
    lock_id = row % GROUP_SIZE_M
    Lock += lock_id
    Count = Lock + GROUP_SIZE_M
    DW = DW + lock_id * N + cols
    DB = DB + lock_id * N + cols

    # Load data from the input, output grad,
    # weights, mean, and std into SRAM
    x = tl.load(X, cols, mask=mask, other=0).to(tl.float32)
    dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    mean = tl.load(Mean + row)
    rstd = tl.load(Rstd + row)

    # Compute dx
    xhat = (x - mean) * rstd
    wdy = w * dy
    xhat = tl.where(mask, xhat, 0.)
    wdy = tl.where(mask, wdy, 0.)
    c1 = tl.sum(xhat * wdy, axis=0) / N
    c2 = tl.sum(wdy, axis=0) / N
    dx = (wdy - (xhat * c1 + c2)) * rstd

    # Write dx
    tl.store(DX + cols, dx, mask=mask)

    # Accumulate partial sums for dw and db
    partial_dw = (dy * xhat).to(w.dtype)
    partial_db = (dy).to(w.dtype)
    # Wait for lock to be ready
    # https://en.wikipedia.org/wiki/Compare-and-swap
    # (wait until value is 0. When it is, change it to 1)
    while tl.atomic_cas(Lock, 0, 1) == 1:
        pass
    # How many times have we stored data into
    # the group?
    count = tl.load(Count)

    # First store does not accumulate values as
    # there is nothing to sum.
    if count == 0:
        # Change value to 1
        tl.atomic_xchg(Count, 1)
    # Any store after the first loads from global
    # memory, and adds it to the current partial sum
    else:
        partial_dw += tl.load(DW, mask=mask)
        partial_db += tl.load(DB, mask=mask)
    # Write to global memory
    tl.store(DW, partial_dw, mask=mask)
    tl.store(DB, partial_db, mask=mask)
    # Release the lock
    tl.atomic_xchg(Lock, 0)




# Kernel used to do the reduction of the partial DW and DB
# which are currently in groups (group_size, d).
# and we want them to be in a single group (d).
@triton.jit
def _layer_norm_bwd_dwdb(
        DW, # Pointer to the partial sum of weight gradients
        DB, # Pointer to the partial sum of biase gradients
        FINAL_DW, # Pointer to the final, accumulated, weight gradient
        FINAL_DB, # Pointer to the final, accumualted, bias gradient
        M, # GROUP_SIZE_M
        N, # Number of columns per row
        BLOCK_SIZE_M: tl.constexpr, # Block over cols
        BLOCK_SIZE_N: tl.constexpr # Block over rows
    ):

    # Map the program id / thread block id to the 
    # elements of DW and DB it should compute
    pid = tl.program_id(0)
    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    db = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Iterate through all rows of DW and DB to sum the partial sums
    # iterate block-wise
    for i in range(0, M, BLOCK_SIZE_M):
        # Get the rows in this block
        rows = i + tl.arange(0, BLOCK_SIZE_M)
        # Get group-wise masks
        mask = (rows[:, None] < M) &  (cols[None, :] < N)
        # Get offsets
        offs = rows[:, None] * N + cols[None, :]
        # Load in the data from the global grouped gradients
        # and reduce them into the partial dw and db
        dw += tl.load(DW + offs, mask=mask, other=0.)
        db += tl.load(DB + offs, mask=mask, other=0.)
    
    # Write the final sums to the output
    sum_dw = tl.sum(dw, axis=0)
    sum_db = tl.sum(db, axis=0)
    tl.store(FINAL_DW + cols, sum_dw, mask=cols < N)
    tl.store(FINAL_DB + cols, sum_db, mask=cols < N)






# Torch implementation
class LayerNorm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, normalized_shape, weight, bias, eps):
        # allocate output
        y = torch.empty_like(x)
        # reshape input data into 2D tensor
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        mean = torch.empty((M, ), dtype=torch.float32, device=x.device)
        rstd = torch.empty((M, ), dtype=torch.float32, device=x.device)
        # Less than 64KB per feature: enqueue fused kernel
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > BLOCK_SIZE:
            raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
        # heuristics for number of warps
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
        # enqueue kernel
        _layer_norm_fwd_fused[(M, )](  #
            x_arg, y, weight, bias, mean, rstd,  #
            x_arg.stride(0), N, eps,  #
            BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps, num_ctas=1)
        ctx.save_for_backward(x, weight, bias, mean, rstd)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, dy):
        x, w, b, m, v = ctx.saved_tensors
        # heuristics for amount of parallel reduction stream for DW/DB
        N = w.shape[0]
        GROUP_SIZE_M = 64
        if N <= 8192: GROUP_SIZE_M = 96
        if N <= 4096: GROUP_SIZE_M = 128
        if N <= 1024: GROUP_SIZE_M = 256
        # allocate output
        locks = torch.zeros(2 * GROUP_SIZE_M, dtype=torch.int32, device=w.device)
        _dw = torch.zeros((GROUP_SIZE_M, N), dtype=x.dtype, device=w.device)
        _db = torch.zeros((GROUP_SIZE_M, N), dtype=x.dtype, device=w.device)
        dw = torch.empty((N, ), dtype=w.dtype, device=w.device)
        db = torch.empty((N, ), dtype=w.dtype, device=w.device)
        dx = torch.empty_like(dy)
        # enqueue kernel using forward pass heuristics
        # also compute partial sums for DW and DB
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        _layer_norm_bwd_dx_fused[(M, )](  #
            dx, dy, _dw, _db, x, w, m, v, locks,  #
            x_arg.stride(0), N,  #
            BLOCK_SIZE_N=ctx.BLOCK_SIZE,  #
            GROUP_SIZE_M=GROUP_SIZE_M,  #
            num_warps=ctx.num_warps)
        grid = lambda meta: [triton.cdiv(N, meta['BLOCK_SIZE_N'])]
        # accumulate partial sums in separate kernel
        _layer_norm_bwd_dwdb[grid](
            _dw, _db, dw, db, min(GROUP_SIZE_M, M), N,  #
            BLOCK_SIZE_M=32,  #
            BLOCK_SIZE_N=128, num_ctas=1)
        return dx, None, dw, db, None


layer_norm = LayerNorm.apply














# Benchmark 
def test_layer_norm(M, N, dtype, eps=1e-5, device=DEVICE):
    # create data
    x_shape = (M, N)
    w_shape = (x_shape[-1], )
    weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    dy = .1 * torch.randn_like(x)
    x.requires_grad_(True)
    # forward pass
    y_tri = layer_norm(x, w_shape, weight, bias, eps)
    y_ref = torch.nn.functional.layer_norm(x, w_shape, weight, bias, eps).to(dtype)
    # backward pass (triton)
    y_tri.backward(dy, retain_graph=True)
    dx_tri, dw_tri, db_tri = [_.grad.clone() for _ in [x, weight, bias]]
    x.grad, weight.grad, bias.grad = None, None, None
    # backward pass (torch)
    y_ref.backward(dy, retain_graph=True)
    dx_ref, dw_ref, db_ref = [_.grad.clone() for _ in [x, weight, bias]]
    # compare
    assert torch.allclose(y_tri, y_ref, atol=1e-2, rtol=0)
    assert torch.allclose(dx_tri, dx_ref, atol=1e-2, rtol=0)
    assert torch.allclose(db_tri, db_ref, atol=1e-2, rtol=0)
    assert torch.allclose(dw_tri, dw_ref, atol=1e-2, rtol=0)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[512 * i for i in range(2, 32)],
        line_arg='provider',
        line_vals=['triton', 'torch'] + (['apex'] if HAS_APEX else []),
        line_names=['Triton', 'Torch'] + (['Apex'] if HAS_APEX else []),
        styles=[('blue', '-'), ('green', '-'), ('orange', '-')],
        ylabel='GB/s',
        plot_name='layer-norm-backward',
        args={'M': 4096, 'dtype': torch.float16, 'mode': 'backward'},
    ))
def bench_layer_norm(M, N, dtype, provider, mode='backward', eps=1e-5, device=DEVICE):
    # create data
    x_shape = (M, N)
    w_shape = (x_shape[-1], )
    weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    dy = .1 * torch.randn_like(x)
    x.requires_grad_(True)
    quantiles = [0.5, 0.2, 0.8]

    def y_fwd():

        if provider == "triton":
            return layer_norm(x, w_shape, weight, bias, eps)  # noqa: F811, E704

        if provider == "torch":
            return torch.nn.functional.layer_norm(x, w_shape, weight, bias, eps)  # noqa: F811, E704

        if provider == "apex":
            apex_layer_norm = (apex.normalization.FusedLayerNorm(w_shape).to(x.device).to(x.dtype))
            return apex_layer_norm(x)  # noqa: F811, E704

    # forward pass
    if mode == 'forward':
        gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
        ms, min_ms, max_ms = triton.testing.do_bench(y_fwd, quantiles=quantiles, rep=500)
    # backward pass
    if mode == 'backward':
        y = y_fwd()
        gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)  # noqa: F811, E704
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: y.backward(dy, retain_graph=True), quantiles=quantiles,
                                                     grad_to_none=[x], rep=500)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


test_layer_norm(1151, 8192, torch.float16)
bench_layer_norm.run(save_path='.', print_data=True)