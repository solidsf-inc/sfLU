// SPDX-License-Identifier: MIT
#include "../src/kernel.cuh"
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

static void cuda_check(cudaError_t error) { if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error)); }
template<class T> struct Device {
    T* p = nullptr; size_t n;
    explicit Device(size_t count) : n(count) { cuda_check(cudaMalloc(reinterpret_cast<void**>(&p), n * sizeof(T))); }
    ~Device() { cudaFree(p); }
    void put(const std::vector<T>& data) { if (data.size() != n) throw std::runtime_error("upload size"); cuda_check(cudaMemcpy(p, data.data(), n * sizeof(T), cudaMemcpyHostToDevice)); }
    std::vector<T> get() { std::vector<T> data(n); cuda_check(cudaMemcpy(data.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost)); return data; }
};
struct Case { std::string name; bool pass; };

static bool verify(int n, int batch, const std::vector<double>& a, const std::vector<double>& factors,
    const std::vector<int>& pivots, const std::vector<double>& x, const std::vector<double>& b, const std::vector<double>& known) {
    bool pass = true;
    for (int system = 0; system < batch; ++system) {
        const int base = system * n * n;
        std::vector<double> pa(a.begin() + base, a.begin() + base + n * n);
        for (int k = 0; k < n; ++k) for (int j = 0; j < n; ++j) std::swap(pa[k * n + j], pa[pivots[system * n + k] * n + j]);
        for (int i = 0; i < n; ++i) {
            double ax = 0.0;
            for (int j = 0; j < n; ++j) {
                ax += a[base + i * n + j] * x[system * n + j];
                double product = 0.0;
                for (int k = 0; k < n; ++k) {
                    const double l = i == k ? 1.0 : (i > k ? factors[base + i * n + k] : 0.0);
                    const double u = k <= j ? factors[base + k * n + j] : 0.0;
                    product += l * u;
                }
                pass &= std::isfinite(product) && std::abs(product - pa[i * n + j]) < 1e-9 * (1.0 + std::abs(pa[i * n + j]));
            }
            pass &= std::isfinite(x[system * n + i]) && std::abs(ax - b[system * n + i]) < 1e-9 * (1.0 + std::abs(b[system * n + i]));
            pass &= std::abs(x[system * n + i] - known[system * n + i]) < 1e-9;
        }
    }
    return pass;
}

int main() {
    std::vector<Case> cases; auto record = [&](const char* name, bool pass) { cases.push_back({name, pass}); };
    cudaStream_t stream = nullptr;
    try {
        cuda_check(cudaStreamCreate(&stream));
        auto run = [&](int n, int batch, const std::vector<double>& a, const std::vector<double>& known) {
            std::vector<double> b(n * batch, 0.0);
            for (int s = 0; s < batch; ++s) for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j)
                b[s * n + i] += a[s * n * n + i * n + j] * known[s * n + j];
            Device<double> d_a(a.size()), d_f(a.size()), d_b(b.size()), d_x(b.size());
            Device<int> d_p(n * batch); Device<sflu::Result> d_r(batch), d_s(batch);
            d_a.put(a); d_b.put(b);
            cuda_check(sflu::factor(n, batch, d_a.p, d_f.p, d_p.p, d_r.p, 1e-12, stream));
            cuda_check(sflu::solve(n, batch, d_f.p, d_p.p, d_r.p, d_b.p, d_x.p, d_s.p, stream));
            cuda_check(cudaStreamSynchronize(stream));
            bool pass = true; for (const auto& result : d_s.get()) pass &= result.status == sflu::Status::Success;
            return pass && verify(n, batch, a, d_f.get(), d_p.get(), d_x.get(), b, known);
        };
        record("batched-partial-pivot-reconstruction-and-solve", run(3, 2,
            {0,2,1, 1,-2,-3, 2,3,1, 2,0,0, 0,3,0, 0,0,4}, {1,-2,0.5, -1,2,3}));
        const int n = sflu::max_order;
        std::vector<double> a(n * n, 0.0), known(n);
        for (int i = 0; i < n; ++i) { known[i] = (i % 7 - 3) * 0.25; for (int j = 0; j < n; ++j) a[i * n + j] = i == j ? n + 2.0 : (i + j) % 5 - 2.0; }
        record("maximum-order-vs-manufactured-reference", run(n, 1, a, known));
        record("single-order", run(1, 1, {2.5}, {-1.25}));
        const int tail_n = 17, tail_batch = 5;
        std::vector<double> tail_a(tail_n * tail_n * tail_batch), tail_known(tail_n * tail_batch);
        for (int system = 0; system < tail_batch; ++system) for (int i = 0; i < tail_n; ++i) {
            tail_known[system * tail_n + i] = 0.125 * (i - 8);
            const int source_row = (i + 7 + system) % tail_n;
            for (int j = 0; j < tail_n; ++j) tail_a[system * tail_n * tail_n + i * tail_n + j] = source_row == j ? 25.0 : 0.1 * ((source_row + j) % 5 - 2);
        }
        record("odd-order-row-permutations-and-partial-warp-batch", run(tail_n, tail_batch, tail_a, tail_known));
        Device<double> d_a(9), d_f(9), d_b(3), d_x(3); Device<int> d_p(3); Device<sflu::Result> d_r(1), d_s(1);
        const std::vector<double> inplace_a = {0,2,1, 1,-2,-3, 2,3,1}, inplace_known = {1,-2,0.5}, inplace_b = {-3.5,3.5,-3.5};
        d_a.put(inplace_a); d_b.put(inplace_b);
        cuda_check(sflu::factor(3, 1, d_a.p, d_a.p, d_p.p, d_r.p, 1e-12, stream));
        cuda_check(sflu::solve(3, 1, d_a.p, d_p.p, d_r.p, d_b.p, d_b.p, d_r.p, stream));
        cuda_check(cudaStreamSynchronize(stream));
        record("exact-inplace-factor-rhs-and-result", d_r.get()[0].status == sflu::Status::Success && verify(3, 1, inplace_a, d_a.get(), d_p.get(), d_b.get(), inplace_b, inplace_known));
        d_a.put({1,0,0, 0,2,0, 0,0,3}); d_b.put({1,std::numeric_limits<double>::quiet_NaN(),3});
        cuda_check(sflu::factor(3, 1, d_a.p, d_f.p, d_p.p, d_r.p, 1e-12, stream));
        cuda_check(sflu::solve(3, 1, d_f.p, d_p.p, d_r.p, d_b.p, d_x.p, d_s.p, stream));
        cuda_check(cudaStreamSynchronize(stream));
        bool rejected_rhs = d_s.get()[0].status == sflu::Status::InvalidInput;
        for (double value : d_x.get()) rejected_rhs &= std::isnan(value);
        record("warp-solve-rejects-nonfinite-rhs", rejected_rhs);
        d_b.put({1,2,3}); d_p.put({0,0,2});
        cuda_check(sflu::solve(3, 1, d_f.p, d_p.p, d_r.p, d_b.p, d_x.p, d_s.p, stream));
        cuda_check(cudaStreamSynchronize(stream));
        bool rejected_pivot = d_s.get()[0].status == sflu::Status::InvalidInput;
        for (double value : d_x.get()) rejected_pivot &= std::isnan(value);
        record("warp-solve-rejects-invalid-pivots", rejected_pivot);
        d_a.put({1,2,3, 2,4,6, 0,1,1}); d_b.put({1,2,3});
        cuda_check(sflu::factor(3, 1, d_a.p, d_f.p, d_p.p, d_r.p, 1e-12, stream));
        cuda_check(sflu::solve(3, 1, d_f.p, d_p.p, d_r.p, d_b.p, d_x.p, d_s.p, stream));
        cuda_check(cudaStreamSynchronize(stream));
        const auto singular = d_s.get()[0]; bool nan_solution = true; for (double value : d_x.get()) nan_solution &= std::isnan(value);
        record("singular-factor-propagates-to-solve", singular.status == sflu::Status::Singular && singular.pivot_index >= 0 && nan_solution);
        d_a.put({1,0,0, 0,std::numeric_limits<double>::quiet_NaN(),0, 0,0,1});
        cuda_check(sflu::factor(3, 1, d_a.p, d_f.p, d_p.p, d_r.p, 1e-12, stream)); cuda_check(cudaStreamSynchronize(stream));
        record("nonfinite-input-status", d_r.get()[0].status == sflu::Status::InvalidInput);
        record("reject-domain-and-tolerance",
            sflu::factor(33, 1, d_a.p, d_f.p, d_p.p, d_r.p) == cudaErrorInvalidValue &&
            sflu::factor(3, 0, d_a.p, d_f.p, d_p.p, d_r.p) == cudaErrorInvalidValue &&
            sflu::factor(3, 1, d_a.p, d_f.p, d_p.p, d_r.p, -1.0) == cudaErrorInvalidValue);
    } catch (const std::exception& error) { std::fprintf(stderr, "%s\n", error.what()); record("cuda-runtime", false); }
    if (stream) cudaStreamDestroy(stream);
    bool pass = !cases.empty(); for (const auto& item : cases) pass &= item.pass;
    std::printf("{\"package\":\"sfLU\",\"status\":\"%s\",\"cases\":[", pass ? "PASS" : "FAIL");
    for (size_t i = 0; i < cases.size(); ++i) std::printf("%s{\"name\":\"%s\",\"status\":\"%s\"}", i ? "," : "", cases[i].name.c_str(), cases[i].pass ? "PASS" : "FAIL");
    std::puts("]}"); return pass ? 0 : 1;
}
