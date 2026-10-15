/*
 * (C) Copyright 2026 QUICK contributors. All rights reserved.
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. See https://mozilla.org/MPL/2.0/.
 *
 * Independent FP64 volume-density and reaction-potential quadrature.
 * These entry points own their storage and never modify the published
 * ESP, XC, one-electron, or density buffers in QUICK's GPU simulation.
 */
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <vector>
#include <cstddef>
#include "gpu_solvation_atomic.h"

namespace {
const int batch_size = 2048;
struct Arena {
    int error;
    cublasHandle_t blas;
    bool have_blas;
    std::vector<void *> memory;
    Arena() : error(0), have_blas(false) {
        have_blas = cublasCreate(&blas) == CUBLAS_STATUS_SUCCESS;
        if (!have_blas) error = 53;
    }
    ~Arena() {
        release();
    }
    void release() {
        for (std::size_t i = 0; i < memory.size(); ++i) check(cudaFree(memory[i]));
        memory.clear();
        if (have_blas) {
            check_blas(cublasDestroy(blas));
            have_blas = false;
        }
    }
    bool check(cudaError_t status) {
        if (status == cudaSuccess) return true;
        if (!error) std::fprintf(stderr, "QUICK solvent CUDA: %s\n", cudaGetErrorString(status));
        error = 53;
        return false;
    }
    bool check_blas(cublasStatus_t status) {
        if (status == CUBLAS_STATUS_SUCCESS) return true;
        std::fprintf(stderr, "QUICK solvent cuBLAS status: %d\n", int(status));
        error = 53;
        return false;
    }
    template <typename T> T *allocate(std::size_t n) {
        if (error) return NULL;
        T *p = NULL;
        if (!check(cudaMalloc(reinterpret_cast<void **>(&p), n * sizeof(T)))) return NULL;
        memory.push_back(p);
        return p;
    }
    template <typename T> T *upload(const T *h, std::size_t n) {
        T *p = allocate<T>(n);
        if (p) check(cudaMemcpy(p, h, n * sizeof(T), cudaMemcpyHostToDevice));
        return p;
    }
};
struct Basis {
    int nb, na, mp;
    int *nprim, *angular, *center;
    double *exponents, *coefficients, *xyz;
};
Basis upload_basis(Arena &a, int nb, int na, int mp, const int *nc,
        const int *angular, const int *center, const double *exponents,
        const double *coefficients, const double *xyz) {
    Basis b;
    b.nb = nb; b.na = na; b.mp = mp;
    b.nprim = a.upload(nc, nb);
    b.angular = a.upload(angular, 3 * std::size_t(nb));
    b.center = a.upload(center, nb);
    b.exponents = a.upload(exponents, std::size_t(mp) * nb);
    b.coefficients = a.upload(coefficients, std::size_t(mp) * nb);
    b.xyz = a.upload(xyz, 3 * std::size_t(na));
    return b;
}
__device__ inline double power(double x, int n) {
    double result = 1.0;
    for (int i = 0; i < n; ++i) result *= x;
    return result;
}
template <bool derivatives>
__global__ void basis_kernel(Basis b, int n, const double *points,
        double *phi, double *derivative) {
    std::size_t id = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    std::size_t total = std::size_t(n) * b.nb;
    for (; id < total; id += std::size_t(blockDim.x) * gridDim.x) {
        int mu = int(id % b.nb), point = int(id / b.nb), atom = b.center[mu] - 1;
        double delta[3], poly[3];
        int angular[3];
        double r2 = 0.0;
        for (int k = 0; k < 3; ++k) {
            delta[k] = points[3 * point + k] - b.xyz[3 * atom + k];
            angular[k] = b.angular[3 * mu + k];
            poly[k] = power(delta[k], angular[k]);
            r2 += delta[k] * delta[k];
        }
        double value = 0.0, grad[3] = {0.0, 0.0, 0.0};
        for (int p = 0; p < b.nprim[mu]; ++p) {
            int index = p + b.mp * mu;
            double alpha = b.exponents[index];
            double factor = b.coefficients[index] * exp(-alpha * r2);
            value += factor * poly[0] * poly[1] * poly[2];
            if (derivatives) for (int k = 0; k < 3; ++k) {
                double lower = angular[k] ? angular[k] * power(delta[k], angular[k] - 1) : 0.0;
                grad[k] += factor * (lower - 2.0 * alpha * delta[k] * poly[k])
                    * poly[(k + 1) % 3] * poly[(k + 2) % 3];
            }
        }
        phi[id] = value;
        if (derivatives) for (int k = 0; k < 3; ++k) derivative[id + k * total] = grad[k];
    }
}
__global__ void density_kernel(int nb, int n, const double *phi, const double *dphi,
        const double *dphi_density, double *rho, double *gradient) {
    for (int point = blockIdx.x * blockDim.x + threadIdx.x; point < n;
            point += blockDim.x * gridDim.x) {
        double value = 0.0, grad[3] = {0.0, 0.0, 0.0};
        for (int mu = 0; mu < nb; ++mu) {
            std::size_t index = mu + std::size_t(nb) * point;
            double t = dphi_density[index];
            value += phi[index] * t;
            for (int k = 0; k < 3; ++k) grad[k] += 2.0 * dphi[index + std::size_t(k) * nb * n] * t;
        }
        rho[point] = value;
        for (int k = 0; k < 3; ++k) gradient[3 * point + k] = grad[k];
    }
}
__global__ void weight_basis_kernel(int nb, int n, const double *phi, const double *eta, double *weighted) {
    std::size_t total = std::size_t(nb) * n;
    for (std::size_t index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
            index < total; index += std::size_t(blockDim.x) * gridDim.x)
        weighted[index] = eta[index / nb] * phi[index];
}
__global__ void gradient_kernel(Basis b, int n, const int *parent, const double *eta,
        const double *dphi, const double *density_phi, double *gradient) {
    std::size_t total = std::size_t(b.nb) * n;
    for (std::size_t index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
            index < total; index += std::size_t(blockDim.x) * gridDim.x) {
        int mu = int(index % b.nb), point = int(index / b.nb);
        int atom = b.center[mu] - 1, moving_parent = parent[point] - 1;
        if (atom == moving_parent) continue;
        double factor = 2.0 * eta[point] * density_phi[index];
        for (int k = 0; k < 3; ++k) {
            double value = factor * dphi[index + std::size_t(k) * total];
            quick_atomic_add_fp64(gradient + 3 * atom + k, -value);
            quick_atomic_add_fp64(gradient + 3 * moving_parent + k, value);
        }
    }
}
void evaluate(Arena &arena, Basis b, int n, const double *points, double *phi, double *dphi) {
    int blocks = int((std::size_t(b.nb) * n + 255) / 256);
    if (blocks > 65535) blocks = 65535;
    if (dphi) basis_kernel<true><<<blocks, 256>>>(b, n, points, phi, dphi);
    else basis_kernel<false><<<blocks, 256>>>(b, n, points, phi, NULL);
    arena.check(cudaGetLastError());
}
bool dimensions(int nb, int na, int mp, int np, int *ierr) {
    if (*ierr) return false;
    if (nb <= 0 || na <= 0 || mp <= 0 || np <= 0) { *ierr = 53; return false; }
    return true;
}
} // namespace

extern "C" void gpu_solvation_density_(const int *nb, const int *na, const int *mp,
        const int *nc, const int *angular, const int *center, const double *exponents,
        const double *coefficients, const double *xyz, const int *np, const double *points,
        const double *density, double *rho, double *gradient, int *ierr) {
    *ierr = 0;
    if (!dimensions(*nb, *na, *mp, *np, ierr)) return;
    Arena arena;
    Basis b = upload_basis(arena, *nb, *na, *mp, nc, angular, center, exponents, coefficients, xyz);
    double *d = arena.upload(density, std::size_t(*nb) * *nb);
    double *p = arena.allocate<double>(3 * batch_size);
    double *phi = arena.allocate<double>(std::size_t(*nb) * batch_size);
    double *dp = arena.allocate<double>(3 * std::size_t(*nb) * batch_size);
    double *product = arena.allocate<double>(std::size_t(*nb) * batch_size);
    double *r = arena.allocate<double>(batch_size), *g = arena.allocate<double>(3 * batch_size);
    const double one = 1.0, zero = 0.0;
    for (int start = 0; !arena.error && start < *np; start += batch_size) {
        int n = (*np - start < batch_size) ? *np - start : batch_size;
        arena.check(cudaMemcpy(p, points + 3 * std::size_t(start), 3 * std::size_t(n) * sizeof(double), cudaMemcpyHostToDevice));
        if (arena.error) break;
        evaluate(arena, b, n, p, phi, dp);
        if (arena.error) break;
        arena.check_blas(cublasDgemm(arena.blas, CUBLAS_OP_N, CUBLAS_OP_N, *nb, n, *nb,
            &one, d, *nb, phi, *nb, &zero, product, *nb));
        if (arena.error) break;
        density_kernel<<<(n + 127) / 128, 128>>>(*nb, n, phi, dp, product, r, g);
        arena.check(cudaGetLastError());
        if (arena.error) break;
        arena.check(cudaMemcpy(rho + start, r, n * sizeof(double), cudaMemcpyDeviceToHost));
        if (arena.error) break;
        arena.check(cudaMemcpy(gradient + 3 * std::size_t(start), g, 3 * std::size_t(n) * sizeof(double), cudaMemcpyDeviceToHost));
    }
    arena.release();
    *ierr = arena.error;
}

extern "C" void gpu_solvation_local_matrix_(const int *nb, const int *na, const int *mp,
        const int *nc, const int *angular, const int *center, const double *exponents,
        const double *coefficients, const double *xyz, const int *np, const double *points,
        const double *eta, double *matrix, int *ierr) {
    *ierr = 0;
    if (!dimensions(*nb, *na, *mp, *np, ierr)) return;
    Arena arena;
    Basis b = upload_basis(arena, *nb, *na, *mp, nc, angular, center, exponents, coefficients, xyz);
    double *p = arena.allocate<double>(3 * batch_size), *w = arena.allocate<double>(batch_size);
    double *phi = arena.allocate<double>(std::size_t(*nb) * batch_size);
    double *weighted = arena.allocate<double>(std::size_t(*nb) * batch_size);
    double *result = arena.allocate<double>(std::size_t(*nb) * *nb);
    const double one = 1.0;
    if (!arena.error) arena.check(cudaMemset(result, 0, std::size_t(*nb) * *nb * sizeof(double)));
    for (int start = 0; !arena.error && start < *np; start += batch_size) {
        int n = (*np - start < batch_size) ? *np - start : batch_size;
        arena.check(cudaMemcpy(p, points + 3 * std::size_t(start), 3 * std::size_t(n) * sizeof(double), cudaMemcpyHostToDevice));
        arena.check(cudaMemcpy(w, eta + start, n * sizeof(double), cudaMemcpyHostToDevice));
        if (arena.error) break;
        evaluate(arena, b, n, p, phi, NULL);
        if (arena.error) break;
        weight_basis_kernel<<<(*nb * n + 255) / 256, 256>>>(*nb, n, phi, w, weighted);
        arena.check(cudaGetLastError());
        if (arena.error) break;
        arena.check_blas(cublasDgemm(arena.blas, CUBLAS_OP_N, CUBLAS_OP_T, *nb, *nb, n,
            &one, weighted, *nb, phi, *nb, &one, result, *nb));
    }
    if (!arena.error) arena.check(cudaMemcpy(matrix, result, std::size_t(*nb) * *nb * sizeof(double), cudaMemcpyDeviceToHost));
    arena.release();
    *ierr = arena.error;
}

extern "C" void gpu_solvation_local_gradient_(const int *nb, const int *na, const int *mp,
        const int *nc, const int *angular, const int *center, const double *exponents,
        const double *coefficients, const double *xyz, const int *np, const double *points,
        const double *eta, const double *density, const int *parent, double *gradient, int *ierr) {
    *ierr = 0;
    if (!dimensions(*nb, *na, *mp, *np, ierr)) return;
    Arena arena;
    Basis b = upload_basis(arena, *nb, *na, *mp, nc, angular, center, exponents, coefficients, xyz);
    double *d = arena.upload(density, std::size_t(*nb) * *nb);
    double *p = arena.allocate<double>(3 * batch_size), *w = arena.allocate<double>(batch_size);
    int *parent_d = arena.allocate<int>(batch_size);
    double *phi = arena.allocate<double>(std::size_t(*nb) * batch_size);
    double *dp = arena.allocate<double>(3 * std::size_t(*nb) * batch_size);
    double *product = arena.allocate<double>(std::size_t(*nb) * batch_size);
    double *result = arena.allocate<double>(3 * std::size_t(*na));
    const double one = 1.0, zero = 0.0;
    if (!arena.error) arena.check(cudaMemset(result, 0, 3 * std::size_t(*na) * sizeof(double)));
    for (int start = 0; !arena.error && start < *np; start += batch_size) {
        int n = (*np - start < batch_size) ? *np - start : batch_size;
        arena.check(cudaMemcpy(p, points + 3 * std::size_t(start), 3 * std::size_t(n) * sizeof(double), cudaMemcpyHostToDevice));
        arena.check(cudaMemcpy(w, eta + start, n * sizeof(double), cudaMemcpyHostToDevice));
        arena.check(cudaMemcpy(parent_d, parent + start, n * sizeof(int), cudaMemcpyHostToDevice));
        if (arena.error) break;
        evaluate(arena, b, n, p, phi, dp);
        if (arena.error) break;
        arena.check_blas(cublasDgemm(arena.blas, CUBLAS_OP_N, CUBLAS_OP_N, *nb, n, *nb,
            &one, d, *nb, phi, *nb, &zero, product, *nb));
        if (arena.error) break;
        gradient_kernel<<<(*nb * n + 255) / 256, 256>>>(b, n, parent_d, w, dp, product, result);
        arena.check(cudaGetLastError());
    }
    if (!arena.error) arena.check(cudaMemcpy(gradient, result, 3 * std::size_t(*na) * sizeof(double), cudaMemcpyDeviceToHost));
    arena.release();
    *ierr = arena.error;
}
