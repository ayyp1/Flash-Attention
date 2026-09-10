# Flash Attention

A simplified Triton implementation of Flash Attention with forward and backward passes.

For the math behind it, check out my blog: [Understanding Flash Attention](https://ashishkpokharel.com.np/blogs/flash-attention/)

## What's here

- `flash_attention.py` — Triton kernels for forward and backward pass
- `bench_flash_attention.py` — benchmarks comparing against PyTorch SDPA across sequence lengths
- `Triton_basics/` — notebooks used to learn Triton before writing the kernel
  - `Vector_Addition.ipynb` — vector addition kernel + memory bandwidth benchmarks
  - `fused-softmax-ipynb.ipynb` — fused softmax kernel

## Requirements

- CUDA GPU
- PyTorch
- [Triton](https://github.com/openai/triton)

## Run

```bash
# sanity check + benchmarks (saves plots to current dir)
python bench_flash_attention.py
```

## Note

Current benchmarks were run on a Tesla T4. Planning to run on an H100 or similar GPU later and post results for comparison.
