// SPDX-License-Identifier: MIT
#pragma once
#include <cuda_runtime.h>
#include <cmath>

namespace sflu {
constexpr int max_order = 32;
constexpr int max_batch = 65535;
enum class Status : int { Success, Singular, InvalidInput, NonFinite };
struct Result { Status status; int pivot_index; double minimum_pivot; };

static __global__ void factor_kernel(int n, const double* matrices, double* factors,
    int* pivots, Result* results, double tolerance) {
    __shared__ double a[max_order * (max_order + 1)], reductions[8], threshold, minimum;
    __shared__ int status, selected, failed_pivot;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const size_t base = static_cast<size_t>(blockIdx.x) * n * n;
    int* permutation = pivots + static_cast<size_t>(blockIdx.x) * n;
    double scale = 0.0;
    bool finite = true;
    for (int index = tid; index < n * n; index += 256) {
        const double value = matrices[base + index];
        a[(index / n) * (max_order + 1) + index % n] = value;
        finite &= isfinite(value);
        scale = fmax(scale, fabs(value));
    }
    const int invalid = __syncthreads_or(!finite);
    for (int offset = 16; offset > 0; offset >>= 1) scale = fmax(scale, __shfl_down_sync(0xffffffff, scale, offset));
    if (lane == 0) reductions[warp] = scale;
    __syncthreads();
    if (tid == 0) {
        scale = 0.0;
        for (int i = 0; i < 8; ++i) scale = fmax(scale, reductions[i]);
        threshold = tolerance * scale;
        status = static_cast<int>(invalid ? Status::InvalidInput : Status::Success);
        failed_pivot = -1; minimum = INFINITY;
    }
    __syncthreads();
    for (int k = 0; k < n; ++k) {
        if (tid < 32) {
            double magnitude = tid >= k && tid < n ? fabs(a[tid * (max_order + 1) + k]) : -1.0;
            int row = tid;
            for (int offset = 16; offset > 0; offset >>= 1) {
                const double other = __shfl_down_sync(0xffffffff, magnitude, offset);
                const int other_row = __shfl_down_sync(0xffffffff, row, offset);
                if (other > magnitude || (other == magnitude && other_row < row)) { magnitude = other; row = other_row; }
            }
            if (tid == 0) {
                selected = row; permutation[k] = row;
                if (status == static_cast<int>(Status::Success)) {
                    minimum = fmin(minimum, magnitude);
                    if (!isfinite(magnitude)) { status = static_cast<int>(Status::NonFinite); failed_pivot = k; }
                    else if (!(magnitude > threshold)) { status = static_cast<int>(Status::Singular); failed_pivot = k; }
                }
            }
        }
        __syncthreads();
        if (status != static_cast<int>(Status::Success)) break;
        if (tid < n && selected != k) {
            const double old = a[k * (max_order + 1) + tid]; a[k * (max_order + 1) + tid] = a[selected * (max_order + 1) + tid]; a[selected * (max_order + 1) + tid] = old;
        }
        __syncthreads();
        if (tid > k && tid < n) a[tid * (max_order + 1) + k] /= a[k * (max_order + 1) + k];
        __syncthreads();
        const int remaining = n - k - 1;
        for (int index = tid; index < remaining * remaining; index += 256) {
            const int i = k + 1 + index / remaining, j = k + 1 + index % remaining;
            a[i * (max_order + 1) + j] = fma(-a[i * (max_order + 1) + k], a[k * (max_order + 1) + j], a[i * (max_order + 1) + j]);
        }
        __syncthreads();
    }
    finite = true;
    for (int index = tid; index < n * n; index += 256) {
        const double value = a[(index / n) * (max_order + 1) + index % n];
        factors[base + index] = value;
        finite &= isfinite(value);
    }
    const int nonfinite = __syncthreads_or(!finite);
    if (tid == 0) {
        if (status == static_cast<int>(Status::Success) && nonfinite) status = static_cast<int>(Status::NonFinite);
        results[blockIdx.x] = {static_cast<Status>(status), failed_pivot, minimum};
    }
}

static __global__ void solve_kernel(int n, int batch, const double* factors, const int* pivots,
    const Result* factor_results, const double* rhs, double* solutions, Result* results) {
    __shared__ double x[4][max_order];
    __shared__ int status[4];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int system = blockIdx.x * 4 + warp;
    if (system >= batch) return;
    const Result factor_result = factor_results[system];
    const double* a = factors + static_cast<size_t>(system) * n * n;
    const int* permutation = pivots + static_cast<size_t>(system) * n;
    double value = lane < n ? rhs[static_cast<size_t>(system) * n + lane] : 0.0;
    const int pivot = lane < n ? permutation[lane] : lane;
    const bool invalid = lane < n && (!isfinite(value) || pivot < lane || pivot >= n);
    const unsigned invalid_mask = __ballot_sync(0xffffffff, invalid);
    if (lane == 0) status[warp] = static_cast<int>(factor_result.status == Status::Success && invalid_mask ? Status::InvalidInput : factor_result.status);
    x[warp][lane] = value;
    __syncwarp();
    if (status[warp] == static_cast<int>(Status::Success)) {
        for (int k = 0; k < n; ++k) {
            const int selected = __shfl_sync(0xffffffff, pivot, k);
            const double current = x[warp][lane];
            const double first = __shfl_sync(0xffffffff, current, k), second = __shfl_sync(0xffffffff, current, selected);
            x[warp][lane] = lane == k ? second : (lane == selected ? first : current);
            __syncwarp();
        }
        value = x[warp][lane];
        for (int k = 0; k < n; ++k) {
            const double solved = __shfl_sync(0xffffffff, value, k);
            if (lane > k && lane < n) value = fma(-a[lane * n + k], solved, value);
        }
        for (int k = n - 1; k >= 0; --k) {
            if (lane == k) value /= a[k * n + k];
            const double solved = __shfl_sync(0xffffffff, value, k);
            if (lane < k) value = fma(-a[lane * n + k], solved, value);
        }
        const unsigned nonfinite = __ballot_sync(0xffffffff, lane < n && !isfinite(value));
        if (lane == 0 && nonfinite) status[warp] = static_cast<int>(Status::NonFinite);
        __syncwarp();
    }
    if (lane < n) solutions[static_cast<size_t>(system) * n + lane] = status[warp] == static_cast<int>(Status::Success) ? value : nan("");
    if (lane == 0) results[system] = {static_cast<Status>(status[warp]), factor_result.pivot_index, factor_result.minimum_pivot};
}

inline cudaError_t factor(int n, int batch, const double* matrices, double* factors, int* pivots,
    Result* results, double relative_pivot_tolerance = 1e-12, cudaStream_t stream = nullptr) {
    if (n <= 0 || n > max_order || batch <= 0 || batch > max_batch || !matrices || !factors || !pivots || !results ||
        !std::isfinite(relative_pivot_tolerance) || relative_pivot_tolerance < 0.0 || relative_pivot_tolerance >= 1.0)
        return cudaErrorInvalidValue;
    factor_kernel<<<batch, 256, 0, stream>>>(n, matrices, factors, pivots, results, relative_pivot_tolerance);
    return cudaGetLastError();
}

inline cudaError_t solve(int n, int batch, const double* factors, const int* pivots,
    const Result* factor_results, const double* rhs, double* solutions, Result* results, cudaStream_t stream = nullptr) {
    if (n <= 0 || n > max_order || batch <= 0 || batch > max_batch || !factors || !pivots || !factor_results || !rhs || !solutions || !results)
        return cudaErrorInvalidValue;
    const unsigned blocks = static_cast<unsigned>((static_cast<long long>(batch) + 3) / 4);
    solve_kernel<<<blocks, 128, 0, stream>>>(n, batch, factors, pivots, factor_results, rhs, solutions, results);
    return cudaGetLastError();
}
}
