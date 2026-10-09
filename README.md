# sfLU

Standalone CUDA C++17 real-f64 batched dense LU with partial row pivoting,
followed by a solve for one RHS per factor. This fresh MIT mathematical
implementation depends only on the CUDA runtime. It contains no copied
proprietary solidSF or private rvllm solver code.

`src/kernel.cuh` provides `sflu::factor(n, batch, matrices, factors, pivots,
results, relative_pivot_tolerance, stream)` and `sflu::solve(n, batch,
factors, pivots, factor_results, rhs, solutions, results, stream)`.
Matrices and factors are contiguous row-major f64 `n*n` blocks. Lower
entries hold unit-diagonal L multipliers; diagonal/upper entries hold U.
Each zero-based pivot records the row swapped with row k at elimination
step k. Applying those swaps in order gives `P A = L U`. Vectors hold n
entries per system. Results and pivots are device arrays.

The explicit domain is `1 <= n <= 32` and `1 <= batch <= 65535`.
Factorization uses one 256-thread CTA per matrix and keeps its working
matrix in shared memory with padded rows. Pivot search uses a warp reduction
with the lowest row chosen on ties; trailing updates use all eight warps. A solve uses one
warp per system, with shuffle broadcasts for triangular updates.
The implementation is bounded to small dense matrices and does not call
cuDSS or cuSOLVER.

`sflu::Result` reports `Success`, `Singular`, `InvalidInput` or `NonFinite`,
the first failed pivot index where applicable, and minimum observed pivot
magnitude. A pivot is rejected when its absolute magnitude is at most
`relative_pivot_tolerance * max(abs(A))`, using the original matrix scale.
The tolerance must be finite in `[0,1)`; zero rejects only an exact zero.
Numerical near-singularity is therefore an explicit caller-selected policy.
Non-finite factor inputs are rejected. A failed factor cannot become a
successful solve: its status propagates, and the solution is filled with NaNs.

Exact in-place matrices/factors and exact in-place RHS/solutions are
supported. Other partial buffer overlaps are unsupported. All arrays must
have the documented extents and valid device residency. Host checks validate
dimensions, null pointers and pivot tolerance; factor and solve status is
produced asynchronously. Calls use the supplied stream and return
`cudaGetLastError()` without synchronization. The caller must check stream
completion, read status, and validate a problem-specific residual before
relying on an ill-conditioned solve.

Required targets: H100/H200 `sm_90`, B200 `sm_100`, B300/GB300 `sm_103` and GB10
`sm_121`. Portable CUDA core math is used; f64 performance varies by device.
Use a toolkit with support for the selected architecture. Compile results
and numerical hardware qualification must be reported separately.

`tests/check.cu` produces one JSON object and exits zero only for all-pass
cases. Independent scalar products verify `P A = L U`, manufactured solutions
and original-system residuals. Tests include pivoting, batched matrices,
the order-32 boundary, odd-order row permutations and partial batches, exact
in-place factor/RHS/result buffers, singular-status propagation, non-finite
input/RHS, invalid pivots and invalid domains/tolerances. Parent tooling
builds and qualifies the package.
Build contract: `make -j1 ARCHS='90 100 103 121' BUILD_DIR=<durable-path>`
and `make test` (runs `BUILD_DIR/check`). Direct compile:
`nvcc -std=c++17 -O3 -arch=sm_121 tests/check.cu -o
<retained-staging-path>/check`, within approved resource limits.

## GB10 qualification and measured performance

The frozen benchmark source `e661bfaf` passed all 10 package correctness
cases on GB10. The test binary contains verified `sm_90`, `sm_100`, `sm_103`
and `sm_121` cubins plus `compute_90` PTX. Runtime correctness and performance
on H100, H200, B200, B300 and GB300 remain unmeasured.

Timings below are CUDA-event medians of seven alternating native/reference
samples on one nonblocking stream, after warmup. Setup, allocations, layout
conversion, transfers and destructive-input resets are excluded. Numerical
status and independent residual/solution gates passed before and after
timing. These results apply to the stated workload; other sizes, matrices,
batch counts and devices require separate measurements.

| Workload | sfLU | cuSOLVER | Reference / sfLU |
| --- | ---: | ---: | ---: |
| 32-by-32 real FP64 partial-pivot factorization plus one RHS solve, batch 1 | 64.512 microseconds | 86.880 microseconds | 1.3467x |

The reference is `cusolverDnDgetrf` followed by `cusolverDnDgetrs`, using the
same mathematical matrix and RHS. Native relative pivot tolerance is
`1e-12`; this matrix is well-conditioned. Independent original-system
relative residual and manufactured-solution error gates require errors at
most `1e-9`.

## Standalone build

CUDA 13.0 or newer is required for the full target matrix.

```sh
make -j1
make test
# A single device family can be selected explicitly:
make ARCHS=90 BUILD_DIR=build-hopper
```

Default targets: H100/H200 (`sm_90`), B200 (`sm_100`), B300/GB300 (`sm_103`),
GB10 (`sm_121`), plus portable Hopper PTX. Do not add `--use_fast_math`;
that changes the numerical contract. Compile success and numerical
qualification are reported separately from performance benchmarking.

## Published source and GB10 measurements

The standalone source repository is [solidsf-inc/sfLU](https://github.com/solidsf-inc/sfLU). The [solidSF CUDA kernel catalog](https://solidsf.com/kernels) contains the formal report, all measured workloads and raw timing samples.

| Workload | SF median us | Reference median us | Reference / SF |
| --- | ---: | ---: | ---: |
| lu solve f64; 32 x 32 | 64.512 | 86.880 | 1.347x |

References: cuSOLVER cusolverDnDgetrf plus cusolverDnDgetrs.

These are seven-sample medians of alternating CUDA-event measurements on one GB10 at stock clocks. Setup, allocations, transfers, plan creation and destructive-input resets are excluded; stream timing can include API enqueue gaps. Ratios above one mean SF is faster; near-unity medians do not establish a meaningful advantage. No aggregate or universal speedup is claimed.

All 10 package numerical/API cases passed on GB10. All four cubin targets and compute_90 PTX were verified. H100, H200, B200, B300 and GB300 hardware runtime and performance remain unmeasured.
