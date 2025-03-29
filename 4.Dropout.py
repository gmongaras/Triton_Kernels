import tabulate
import torch

import triton
import triton.language as tl

DEVICE = torch.device("cuda")


@triton.jit
def _dropout(
        x_ptr, # Pointer to the input
        x_keep_ptr, # Pointer to a mask of 0s and 1s. 0 to dropout.
        output_ptr, # Pointer to the output
        n_elements, # Number of elements in the input, x, tensor
        p, # Probability that an element, `x` is change to zero
        BLOCK_SIZE: tl.constexpr,
    ):

    # Get the id of this program along the 0th axis
    # within the kernel launch
    pid = tl.program_id(axis=0)

    # Where does this thread block start?
    block_start = pid * BLOCK_SIZE

    # Offsets for every element in the block
    # so we know which elements this thread block
    # needs to dropout from the original vector
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Mask the elements so we don't go out of bounds
    mask = offsets < n_elements

    # Load in the data
    x = tl.load(x_ptr + offsets, mask=mask)
    x_keep = tl.load(x_keep_ptr + offsets, mask=mask)

    # For elements we keep, we divide them by a
    # Let's say we have 10 elements, with p=0.2 we
    # expect 2 to be droped out and 8 to be kept.
    # (1-p)=0.8. Thus we scale each elements by 10/8=5/4.
    # This increases the magnitude to be, on average, the
    # same as before dropout
    output = tl.where(x_keep, x / (1-p), 0.0)

    # Write the output
    tl.store(output_ptr + offsets, output, mask=mask)





def dropout(x, x_keep, p):
    output = torch.empty_like(x)
    assert x.is_contiguous()
    n_elements = x.numel()

    # 1D grid of n_elements broken into "BLOCK_SIZE"
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )

    # Launch the dropout kernel
    _dropout[grid](x, x_keep, output, n_elements, p, BLOCK_SIZE=1024)

    return output



# Input tensor
x = torch.randn(size=(10, ), device=DEVICE)
# Dropout mask
p = 0.5
x_keep = (torch.rand(size=(10, ), device=DEVICE) > p).to(torch.int32)
# 
output = dropout(x, x_keep=x_keep, p=p)
print(tabulate.tabulate([
    ["input"] + x.tolist(),
    ["keep mask"] + x_keep.tolist(),
    ["output"] + output.tolist()
]))


# Problems:
# 1. The mask needs to be stored
# 2. Checkpointing is weird as we need to preserve RNG state







@triton.jit
def _seeded_dropout(
        x_ptr, # Pointer to the input
        # x_keep_ptr, - No mask pointer
        output_ptr, # Pointer to the output
        n_elements, # Number of elements in the input, x, tensor
        p, # Probability that an element, `x` is change to zero
        seed, # RNG seed for dropout
        BLOCK_SIZE: tl.constexpr,
    ):

    # Computer the memory offsets to be handled by this thread block / program instance
    pid = tl.program_id(axis = 0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Load data from input
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)

    # Get random mask using seed
    random = tl.rand(seed, offsets)
    x_keep = random > p

    # Dropout and scale
    output = tl.where(x_keep, x / (1-p), 0.0)

    # Write back
    tl.store(output_ptr + offsets, output, mask=mask)






def seeded_dropout(x, p, seed):
    output = torch.empty_like(x)
    assert x.is_contiguous()
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    _seeded_dropout[grid](x, output, n_elements, p, seed, BLOCK_SIZE=1024)
    return output




x = torch.randn(size=(10, ), device=DEVICE)
# Compare this to the baseline - dropout mask is never instantiated!
output = seeded_dropout(x, p=0.5, seed=123)
output2 = seeded_dropout(x, p=0.5, seed=123)
output3 = seeded_dropout(x, p=0.5, seed=512)

print(
    tabulate.tabulate([
        ["input"] + x.tolist(),
        ["output (seed = 123)"] + output.tolist(),
        ["output (seed = 123)"] + output2.tolist(),
        ["output (seed = 512)"] + output3.tolist(),
    ]))
