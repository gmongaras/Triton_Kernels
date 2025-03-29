import tabulate
import torch

import triton
import triton.language as tl

DEVICE = torch.device("cuda")



@triton.jit
def _seeded_dropout(
        x_ptr, # Pointer to the input
        # x_keep_ptr, - No mask pointer
        output_ptr, # Pointer to the output
        n_elements, # Number of elements in the input, x, tensor
        n_el, # Number of elements per seed
        probs, # Probability that an element, `x` is change to zero
        seeds, # RNG seed for dropout
        BLOCK_SIZE: tl.constexpr,
    ):

    # Compute the memory offsets to be handled by this thread block / program instance
    pid_data, pid_seed = tl.program_id(axis = 0), tl.program_id(axis = 1)
    row_offset = n_el * pid_seed
    col_offset = pid_data * BLOCK_SIZE
    block_start = row_offset + col_offset
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Get seed and prob
    seed = tl.load(seeds + pid_seed)
    p = tl.load(probs + pid_seed)

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






def seeded_dropout(x, probs, seeds):
    # Output matrix
    output = torch.empty_like(x)
    assert x.is_contiguous()
    n_elements = x.numel()
    n_el = x.shape[1]
    n_seeds = seeds.numel()
    
    # Create 2D grid - One axis over the seeds (rows) and the other over the data (columns)
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), triton.cdiv(n_seeds, 1))
    _seeded_dropout[grid](x, output, n_elements, n_el, probs, seeds, BLOCK_SIZE=1024)
    return output



# Random input
x = torch.randn(size=(3, 3), device=DEVICE)

# Random seeds
seeds = torch.randint(low=0, high=100000, size=(3, ), device=DEVICE)

# Probabilities
probs = torch.tensor([
    0.0,
    0.99,
    0.0
], device=DEVICE)

# 
output = seeded_dropout(x, probs=probs, seeds=seeds)

print(
    tabulate.tabulate([
        ["input"] + x.tolist(),
        ["output (seed = 123)"] + output.tolist(),
    ]))

print(x)
print(output)
