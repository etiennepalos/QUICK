/*
 * (C) Copyright 2026 QUICK contributors. All rights reserved.
 * This Source Code Form is subject to the Mozilla Public License, v. 2.0.
 *
 * Native FP64 second spatial derivative of the electronic electric field:
 * H_ijk(C) = d^2 E_i(C)/(dC_j dC_k) = -d^3 V(C)/(dC_i dC_j dC_k).
 * Coordinates are bohr. Original normalized Cartesian primitive coefficients
 * and the spin-summed density are supplied by the Fortran caller. This file
 * owns its temporary device storage and never changes published ESP/EFG paths.
 */
#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>
#include "gpu_solvation_atomic.h"

namespace {
const int input_error = 55;
const int backend_error = 56;
const int max_angular = 3;
const int max_pair_degree = 2 * max_angular;
const int max_order = max_pair_degree + 3;
const int coulomb_slots = (max_order + 1) * (max_order + 2) * (max_order + 3) / 6;
const double pi = 3.14159265358979323846264338327950288;
std::atomic<int> launches(0);

struct Arena {
    int error;
    std::vector<void *> allocations;
    Arena() : error(0) {}
    ~Arena() { release(); }
    bool check(cudaError_t status) {
        if (status == cudaSuccess) return true;
        if (!error) std::fprintf(stderr, "QUICK field-Hessian CUDA: %s\n", cudaGetErrorString(status));
        error = backend_error;
        return false;
    }
    template <typename T> T *allocate(std::size_t count) {
        if (error) return NULL;
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) { error = backend_error; return NULL; }
        T *pointer = NULL;
        if (!check(cudaMalloc(reinterpret_cast<void **>(&pointer), count * sizeof(T)))) return NULL;
        try { allocations.push_back(pointer); }
        catch (const std::bad_alloc &) { check(cudaFree(pointer)); error = backend_error; return NULL; }
        return pointer;
    }
    template <typename T> T *upload(const T *host, std::size_t count) {
        T *pointer = allocate<T>(count);
        if (pointer) check(cudaMemcpy(pointer, host, count * sizeof(T), cudaMemcpyHostToDevice));
        return pointer;
    }
    void release() {
        while (!allocations.empty()) {
            void *pointer = allocations.back();
            allocations.pop_back();
            check(cudaFree(pointer));
        }
    }
};

struct Basis {
    int nb, mp;
    int *nc, *angular, *center, *owned;
    double *exponents, *coefficients, *xyz, *density;
};

bool valid_request(int nb, int na, int mp, const int *nc, const int *angular,
        const int *center, const double *exponents, const double *coefficients,
        const double *xyz, const double *density, int np, const double *points,
        const int *owned, double cutoff, const double *output, bool &work) {
    if (nb < 1 || na < 1 || mp < 1 || np < 1 || !std::isfinite(cutoff) || cutoff < 0.0 ||
            !nc || !angular || !center || !exponents || !coefficients || !xyz || !density || !owned ||
            (np && (!points || !output))) return false;
    const std::size_t limit = std::numeric_limits<std::size_t>::max() / sizeof(double);
    std::size_t matrix_size = std::size_t(nb) * nb, primitive_size = std::size_t(mp) * nb;
    if (matrix_size > limit || primitive_size > limit || 27 * std::size_t(np) > limit) return false;
    std::size_t pairs = std::size_t(nb) * (std::size_t(nb) + 1) / 2;
    if (np && pairs > std::numeric_limits<std::size_t>::max() / std::size_t(np)) return false;
    work = false;
    for (int mu = 0; mu < nb; ++mu) {
        if (nc[mu] < 1 || nc[mu] > mp || center[mu] < 1 || center[mu] > na ||
                (owned[mu] != 0 && owned[mu] != 1)) return false;
        int degree = 0;
        for (int axis = 0; axis < 3; ++axis) {
            int power = angular[3 * std::size_t(mu) + axis];
            if (power < 0 || power > max_angular) return false;
            degree += power;
        }
        if (degree > max_angular) return false;
        for (int primitive = 0; primitive < nc[mu]; ++primitive) {
            std::size_t index = primitive + std::size_t(mp) * mu;
            if (!std::isfinite(exponents[index]) || exponents[index] <= 0.0 ||
                    !std::isfinite(coefficients[index])) return false;
        }
        for (int nu = 0; nu <= mu; ++nu) {
            double left = density[mu + std::size_t(nb) * nu], right = density[nu + std::size_t(nb) * mu];
            if (!std::isfinite(left) || !std::isfinite(right) || std::fabs(left - right) > 1.0e-10) return false;
            if (owned[mu] && right != 0.0) work = true;
        }
    }
    for (std::size_t index = 0; index < 3 * std::size_t(na); ++index)
        if (!std::isfinite(xyz[index])) return false;
    for (std::size_t index = 0; index < 3 * std::size_t(np); ++index)
        if (!std::isfinite(points[index])) return false;
    for (std::size_t index = 0; index < 27 * std::size_t(np); ++index)
        if (!std::isfinite(output[index])) return false;
    return true;
}

// Accurate Boys values without the legacy order-dependent small-T fit.
// For T<=32, seed F_order using an all-positive convergent series, then use
// stable downward recursion. Above32, erf gives F0 and upward recursion is
// well conditioned through order9. Unrepresentable auxiliaries are rejected.
__device__ __forceinline__ bool boys_values(int order, double t, double *boys) {
    if (!isfinite(t) || t < 0.0) return false;
    double exponential = exp(-t);
    if (t <= 32.0) {
        double term = 1.0 / (2 * order + 1), sum = term;
        bool converged = false;
        for (int k = 1; k <= 256; ++k) {
            term *= t / (order + k + 0.5);
            sum += term;
            if (term <= 1.0e-16 * sum) { converged = true; break; }
        }
        if (!converged || !isfinite(sum)) return false;
        boys[order] = exponential * sum;
        for (int m = order; m > 0; --m)
            boys[m - 1] = (2.0 * t * boys[m] + exponential) / (2 * m - 1);
    } else {
        boys[0] = 0.5 * sqrt(pi / t) * erf(sqrt(t));
        for (int m = 1; m <= order; ++m)
            boys[m] = ((2 * m - 1) * boys[m - 1] - exponential) / (2.0 * t);
    }
    for (int m = 0; m <= order; ++m)
        if (!isfinite(boys[m]) || boys[m] <= 0.0) return false;
    return true;
}

// Gaussian product coefficients for Lambda_t=(d/dP)^t exp[-g*(x-P)^2].
// The exp[-ab/g*(A-B)^2] factor is kept outside the three axis expansions.
__device__ __forceinline__ void hermite_coefficients(int la, int lb, double pa,
        double pb, double inverse_two_g, double *coefficients) {
    for (int t = 0; t <= max_pair_degree; ++t) coefficients[t] = 0.0;
    coefficients[0] = 1.0;
    int degree = 0;
    for (int side = 0; side < 2; ++side) {
        int count = side == 0 ? la : lb;
        double shift = side == 0 ? pa : pb;
        for (int angular = 0; angular < count; ++angular) {
            double next[max_pair_degree + 1];
            for (int t = 0; t <= degree + 1; ++t) {
                double value = t <= degree ? shift * coefficients[t] : 0.0;
                int lower = t > 0 ? t - 1 : 0;
                int upper = t < max_pair_degree ? t + 1 : max_pair_degree;
                if (t > 0) value += inverse_two_g * coefficients[lower];
                if (t + 1 <= degree) value += (t + 1) * coefficients[upper];
                next[t] = value;
            }
            ++degree;
            for (int t = 0; t <= degree; ++t) coefficients[t] = next[t];
        }
    }
}

// Compact degree-major indexing for all triples t+u+v<=9 (220 slots).
// Guard inactive lower powers as well so no invalid address is ever formed.
__device__ __forceinline__ int r_index(int t, int u, int v) {
    if (t < 0 || u < 0 || v < 0) return 0;
    int degree = t + u + v;
    return degree * (degree + 1) * (degree + 2) / 6
        + t * (degree + 1) - t * (t - 1) / 2 + u;
}

__device__ __forceinline__ bool coulomb_derivatives(int order, double g,
        const double *pc, const double *boys, double r[2][coulomb_slots]) {
    double factors[max_order + 1];
    factors[0] = 1.0;
    for (int m = 1; m <= order; ++m) {
        factors[m] = factors[m - 1] * (-2.0 * g);
        if (!isfinite(factors[m]) || factors[m] == 0.0) return false;
    }
    // R_(t+1,u,v)^m=t*R_(t-1,u,v)^(m+1)+PCx*R_(t,u,v)^(m+1).
    // Two n slices suffice. Every referenced slot belongs to the previous
    // slice's completed degree range; final R^0 is always in r[0].
    for (int m = order; m >= 0; --m) {
        double *current = r[m & 1], *previous = r[(m + 1) & 1];
        current[0] = factors[m] * boys[m];
        if (!isfinite(current[0])) return false;
        for (int degree = 1; degree <= order - m; ++degree)
            for (int t = 0; t <= degree; ++t)
                for (int u = 0; u <= degree - t; ++u) {
                    int v = degree - t - u;
                    double value;
                    if (t > 0) {
                        value = pc[0] * previous[r_index(t - 1, u, v)];
                        if (t > 1) value += (t - 1) * previous[r_index(t - 2, u, v)];
                    } else if (u > 0) {
                        value = pc[1] * previous[r_index(t, u - 1, v)];
                        if (u > 1) value += (u - 1) * previous[r_index(t, u - 2, v)];
                    } else {
                        value = pc[2] * previous[r_index(t, u, v - 1)];
                        if (v > 1) value += (v - 1) * previous[r_index(t, u, v - 2)];
                    }
                    if (!isfinite(value)) return false;
                    current[r_index(t, u, v)] = value;
                }
    }
    return true;
}

__device__ __forceinline__ void pair_indices(std::size_t pair, int &mu, int &nu) {
    mu = int((sqrt(8.0 * double(pair) + 1.0) - 1.0) * 0.5);
    while (std::size_t(mu) * (std::size_t(mu) + 1) / 2 > pair) --mu;
    while ((std::size_t(mu) + 1) * (std::size_t(mu) + 2) / 2 <= pair) ++mu;
    nu = int(pair - std::size_t(mu) * (std::size_t(mu) + 1) / 2);
}

__global__ void field_hessian_kernel(Basis basis, std::size_t pairs, int np,
        const double *points, double cutoff, double *unique, int *bad_result) {
    std::size_t total = pairs * std::size_t(np);
    for (std::size_t work = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
            work < total; work += std::size_t(gridDim.x) * blockDim.x) {
        int point = int(work / pairs), mu, nu;
        pair_indices(work % pairs, mu, nu);
        if (!basis.owned[mu]) continue;
        // Match the CPU reference's upper triangle, including accepted
        // roundoff-sized asymmetry. Ownership still refers to higher mu.
        double density = basis.density[nu + std::size_t(basis.nb) * mu];
        if (mu != nu) density *= 2.0;
        if (density == 0.0) continue;
        const int *la = basis.angular + 3 * mu, *lb = basis.angular + 3 * nu;
        const double *a = basis.xyz + 3 * (basis.center[mu] - 1);
        const double *b = basis.xyz + 3 * (basis.center[nu] - 1);
        const double *c = points + 3 * std::size_t(point);
        double r2 = 0.0;
        int order = 3;
        for (int axis = 0; axis < 3; ++axis) {
            double distance = b[axis] - a[axis];
            r2 += distance * distance;
            order += la[axis] + lb[axis];
        }
        double values[10] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        for (int ia = 0; ia < basis.nc[mu]; ++ia)
            for (int ib = 0; ib < basis.nc[nu]; ++ib) {
                std::size_t ai = ia + std::size_t(basis.mp) * mu, bi = ib + std::size_t(basis.mp) * nu;
                double alpha = basis.exponents[ai], beta = basis.exponents[bi], g = alpha + beta;
                if (!isfinite(g) || g <= 0.0) { atomicExch(bad_result, 1); continue; }
                double inverse_g = 1.0 / g;
                double gaussian = exp(-alpha * beta * r2 * inverse_g);
                // Preserve the existing opf screening, without density weight.
                double screened = gaussian * sqrt(pi * inverse_g) * pi * inverse_g
                    * basis.coefficients[bi] * basis.coefficients[ai];
                if (!isfinite(screened)) { atomicExch(bad_result, 1); continue; }
                if (fabs(screened) <= cutoff) continue;
                double prefactor = 2.0 * pi * inverse_g * gaussian
                    * basis.coefficients[bi] * basis.coefficients[ai];
                if (!isfinite(prefactor)) { atomicExch(bad_result, 1); continue; }
                double pc[3], coefficients[3][max_pair_degree + 1], tvalue = 0.0;
                for (int axis = 0; axis < 3; ++axis) {
                    double distance = b[axis] - a[axis];
                    double pa = beta * inverse_g * distance, pb = -alpha * inverse_g * distance;
                    pc[axis] = (a[axis] - c[axis]) + pa;
                    tvalue += pc[axis] * pc[axis];
                    hermite_coefficients(la[axis], lb[axis], pa, pb, 0.5 * inverse_g, coefficients[axis]);
                }
                double boys[max_order + 1], r[2][coulomb_slots];
                if (!boys_values(order, g * tvalue, boys) || !coulomb_derivatives(order, g, pc, boys, r)) {
                    atomicExch(bad_result, 1); continue;
                }
                for (int t = 0; t <= la[0] + lb[0]; ++t)
                    for (int u = 0; u <= la[1] + lb[1]; ++u)
                        for (int v = 0; v <= la[2] + lb[2]; ++v) {
                            double coefficient = prefactor * coefficients[0][t] * coefficients[1][u] * coefficients[2][v];
                            // R differentiates P. Three derivatives at C reverse
                            // the sign: H_e=+D*I_C'''=-D*sum(E*R_(t+3,...)).
                            // Fixed xxx,xxy,xxz,xyy,xyz,xzz,yyy,yyz,yzz,zzz
                            // order matches the original nested i/j/k loops.
                            // Constant offsets avoid runtime component-index
                            // branches and let the compiler share r_index work.
                            values[0] -= coefficient * r[0][r_index(t + 3, u, v)];
                            values[1] -= coefficient * r[0][r_index(t + 2, u + 1, v)];
                            values[2] -= coefficient * r[0][r_index(t + 2, u, v + 1)];
                            values[3] -= coefficient * r[0][r_index(t + 1, u + 2, v)];
                            values[4] -= coefficient * r[0][r_index(t + 1, u + 1, v + 1)];
                            values[5] -= coefficient * r[0][r_index(t + 1, u, v + 2)];
                            values[6] -= coefficient * r[0][r_index(t, u + 3, v)];
                            values[7] -= coefficient * r[0][r_index(t, u + 2, v + 1)];
                            values[8] -= coefficient * r[0][r_index(t, u + 1, v + 2)];
                            values[9] -= coefficient * r[0][r_index(t, u, v + 3)];
                        }
            }
        for (int component = 0; component < 10; ++component) {
            double contribution = density * values[component];
            if (!isfinite(contribution)) { atomicExch(bad_result, 1); continue; }
            quick_atomic_add_fp64(unique + component + 10 * std::size_t(point), contribution);
        }
    }
}

__device__ __forceinline__ int unique_component(int i, int j, int k) {
    if (i > j) { int tmp = i; i = j; j = tmp; }
    if (j > k) { int tmp = j; j = k; k = tmp; }
    if (i > j) { int tmp = i; i = j; j = tmp; }
    if (i == 0) return j == 0 ? k : (j == 1 ? k + 2 : 5);
    if (i == 1) return j == 1 ? k + 5 : 8;
    return 9;
}

__global__ void expand_hessian_kernel(std::size_t total, const double *unique,
        double *output, int *bad_result) {
    for (std::size_t index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
            index < total; index += std::size_t(gridDim.x) * blockDim.x) {
        int tensor = int(index % 27), i = tensor % 3, j = tensor / 3 % 3, k = tensor / 9;
        double value = output[index] + unique[unique_component(i, j, k) + 10 * (index / 27)];
        if (!isfinite(value)) { atomicExch(bad_result, 1); continue; }
        output[index] = value;
    }
}

int block_count(std::size_t total) {
    std::size_t blocks = total / 128 + (total % 128 != 0);
    return blocks > 65535 ? 65535 : int(blocks);
}
} // namespace

extern "C" void gpu_field_hessian_(const int *nb, const int *na, const int *maxprim,
        const int *nc, const int *angular, const int *centers, const double *exponents,
        const double *coefficients, const double *atoms, const double *density,
        const int *npoints, const double *points, const int *owned, const double *cutoff,
        double *output, int *ierr) {
    if (!ierr) return;
    *ierr = 0;
    bool work = false;
    if (!nb || !na || !maxprim || !npoints || !cutoff) { *ierr = input_error; return; }
    if (*nb < 1 || *na < 1 || *maxprim < 1 || *npoints < 1) { *ierr = 46; return; }
    if (!valid_request(*nb, *na, *maxprim, nc, angular, centers, exponents, coefficients,
                atoms, density, *npoints, points, owned, *cutoff, output, work)) {
        *ierr = input_error; return;
    }
    if (!work) return;
    try {
        Arena arena;
        Basis basis;
        basis.nb = *nb; basis.mp = *maxprim;
        basis.nc = arena.upload(nc, *nb);
        basis.angular = arena.upload(angular, 3 * std::size_t(*nb));
        basis.center = arena.upload(centers, *nb);
        basis.owned = arena.upload(owned, *nb);
        basis.exponents = arena.upload(exponents, std::size_t(*maxprim) * *nb);
        basis.coefficients = arena.upload(coefficients, std::size_t(*maxprim) * *nb);
        basis.xyz = arena.upload(atoms, 3 * std::size_t(*na));
        basis.density = arena.upload(density, std::size_t(*nb) * *nb);
        double *device_points = arena.upload(points, 3 * std::size_t(*npoints));
        std::size_t output_size = 27 * std::size_t(*npoints), unique_size = 10 * std::size_t(*npoints);
        double *device_output = arena.upload(output, output_size), *unique = arena.allocate<double>(unique_size);
        int *device_bad = arena.allocate<int>(1), host_bad = 0;
        if (!arena.error) arena.check(cudaMemset(device_bad, 0, sizeof(int)));
        if (!arena.error) arena.check(cudaMemset(unique, 0, unique_size * sizeof(double)));
        if (!arena.error) {
            std::size_t pairs = std::size_t(*nb) * (std::size_t(*nb) + 1) / 2;
            field_hessian_kernel<<<block_count(pairs * std::size_t(*npoints)), 128>>>(basis, pairs,
                *npoints, device_points, *cutoff, unique, device_bad);
            if (arena.check(cudaGetLastError())) ++launches;
        }
        if (!arena.error) {
            expand_hessian_kernel<<<block_count(output_size), 128>>>(output_size, unique, device_output, device_bad);
            arena.check(cudaGetLastError());
        }
        std::vector<double> result(output_size);
        if (!arena.error) arena.check(cudaMemcpy(&host_bad, device_bad, sizeof(int), cudaMemcpyDeviceToHost));
        // Scientific auxiliaries outside the representable FP64 domain are
        // numerical-input failures; CUDA/allocation failures use backend56.
        if (host_bad) arena.error = input_error;
        if (!arena.error) arena.check(cudaMemcpy(result.data(), device_output, output_size * sizeof(double), cudaMemcpyDeviceToHost));
        if (!arena.error)
            for (std::size_t index = 0; index < output_size; ++index)
                if (!std::isfinite(result[index])) { arena.error = input_error; break; }
        arena.release();
        *ierr = arena.error;
        if (!*ierr) std::copy(result.begin(), result.end(), output);
    } catch (const std::bad_alloc &) { *ierr = backend_error; }
      catch (const std::length_error &) { *ierr = backend_error; }
}

// Number of primary electronic AO-pair evaluation launches; the inexpensive
// symmetry-expansion launch is deliberately excluded from this dispatch proof.
extern "C" void gpu_field_hessian_launch_count_(int *count) {
    if (count) *count = launches.load();
}
