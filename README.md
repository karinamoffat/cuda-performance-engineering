# CUDA Matrix Computation & GPU Performance Analysis

A CUDA-based parallel computing project focused on **GPU architecture exploration, memory transfer optimization, and high-performance matrix multiplication**. This project evaluates how computation patterns, memory movement, and thread-level parallelism impact performance on modern GPUs.

## Overview

This project explores core principles of **GPU computing and parallel programming** using CUDA:

- Querying and understanding GPU hardware capabilities
- Measuring host ↔ device memory transfer performance
- Implementing and benchmarking matrix multiplication on CPU vs GPU
- Designing parallel kernels using CUDA thread hierarchies
- Optimizing performance using **tiling and shared memory**

The work was developed and tested on an **NVIDIA GeForce RTX 3060 Ti**, providing real-world insights into GPU execution behavior.

---

## Tech Stack

- **CUDA C/C++**
- **NVIDIA CUDA Runtime API**
- **GPU Architecture (SMs, warps, memory hierarchy)**
- **Parallel Computing**
- **Performance Profiling & Benchmarking**

---

## Key Features

### 1. GPU Hardware Introspection
- Queried device-level properties using CUDA runtime APIs:
  - Streaming Multiprocessors (SMs)
  - Warp size
  - Global, shared, and constant memory
  - Register limits
  - Thread/block/grid constraints

- Demonstrates understanding of **hardware-aware programming** and GPU execution limits

---

### 2. Memory Transfer Benchmarking
- Measured **host-to-device** and **device-to-host** transfer times across:
  - 256 × 256
  - 512 × 512
  - 1024 × 1024 matrices

- Key insight:
  - Device → host transfers are significantly slower due to memory retrieval overhead

- Highlights:
  - Understanding of **PCIe bottlenecks**
  - Tradeoffs between computation and data movement

---

### 3. CPU vs GPU Matrix Multiplication
- Implemented matrix multiplication on:
  - CPU (baseline)
  - GPU (CUDA kernel)

- Tested performance with:
  - and without data transfer overhead

- Key insight:
  - GPU is not always faster when:
    - parallelism is underutilized (e.g. 1 thread per block)
    - data transfer dominates compute time

---

### 4. Parallel Kernel Design (Thread-Level Parallelism)
- Designed CUDA kernels where:
  - each thread computes a single output element
  - thread indexing maps directly to matrix coordinates

- Demonstrates:
  - **grid/block/thread hierarchy understanding**
  - mapping mathematical operations to parallel execution

---

### 5. Tiled Matrix Multiplication (Performance Optimization)
- Implemented optimized matrix multiplication using:
  - **shared memory tiling**
  - cooperative thread loading
  - synchronization (`__syncthreads()`)

- Key improvements:
  - reduced global memory access
  - improved memory locality
  - significantly faster computation at larger tile sizes

- Observed:
  - performance improves as tile size increases (within shared memory limits)
  - tiled implementation outperforms naive GPU approach

---

### 6. GPU Occupancy & Resource Analysis
- Analyzed:
  - number of registers per thread
  - shared memory usage
  - blocks per SM
  - total active threads (~40k concurrent threads)

- Demonstrates:
  - understanding of **GPU occupancy and scheduling constraints**

---

## Performance Insights

Key takeaways from the project:

- GPU acceleration is **workload-dependent**
- Memory transfer can dominate runtime for small problems
- Parallelism must be fully utilized to achieve speedup
- Shared memory is critical for high-performance kernels
- Tiling significantly reduces memory bandwidth bottlenecks
- Larger matrix sizes benefit more from GPU execution

