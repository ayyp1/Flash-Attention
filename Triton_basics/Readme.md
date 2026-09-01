## Overview

This folder contains small Triton projects and experiments completed while learning the framework, prior to implementing the Flash Attention kernel.


###Vector Addition
 
See [Vector_Addition.ipynb](./Vector_Addition.ipynb) for full benchmark.
 
##### Results
 
- Triton and Torch GPU are identical → our kernel matches hand-tuned CUDA.
- Both flatten at ~247 GB/s = the memory bus maxed out; can't go faster.
- Small sizes are slow for everyone: launch overhead > actual work.
  Below ~16K elements the CPU actually wins.
- CPU rises to 51 then crashes to 5 GB/s: small tensors fit in cache
  (fake speed), big ones show real RAM bandwidth.
- ~50x GPU win at scale, purely from higher memory bandwidth. The addition
  itself was never the bottleneck.
 
