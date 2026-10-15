/*
 * (C) Copyright 2026 QUICK contributors. All rights reserved.
 * This Source Code Form is subject to the Mozilla Public License, v. 2.0.
 *
 * Native FP64 uniform-field AO moments and electronic nuclear gradients.
 * Normalized Cartesian coefficients are supplied by the Fortran caller.
 * All coordinates/origins are bohr. Independent temporary storage leaves
 * the published ESP/XC/OEI simulation and density buffers unchanged.
 */
#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <new>
#include <vector>
#include "gpu_solvation_atomic.h"

namespace {
const int error_code = 53;
const int max_angular = 3;
const int max_power = max_angular + 1;
const double pi = 3.14159265358979323846264338327950288;
std::atomic<int> matrix_launches(0), gradient_launches(0);

struct Arena {
    int error;
    std::vector<void *> allocations;
    Arena() : error(0) {}
    ~Arena() { release(); }
    bool check(cudaError_t status) {
        if (status == cudaSuccess) return true;
        if (!error) std::fprintf(stderr, "QUICK external-field CUDA: %s\n", cudaGetErrorString(status));
        error = error_code;
        return false;
    }
    template <typename T> T *allocate(std::size_t count) {
        if (error) return NULL;
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) { error = error_code; return NULL; }
        T *pointer = NULL;
        if (!check(cudaMalloc(reinterpret_cast<void **>(&pointer), count * sizeof(T)))) return NULL;
        try { allocations.push_back(pointer); }
        catch (const std::bad_alloc &) { check(cudaFree(pointer)); error = error_code; return NULL; }
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
    int nb, na, mp;
    int *nc, *angular, *center, *owned;
    double *exponents, *coefficients, *xyz;
};

Basis upload_basis(Arena &arena, int nb, int na, int mp, const int *nc,
        const int *angular, const int *center, const double *exponents,
        const double *coefficients, const double *xyz, const int *owned) {
    Basis basis;
    basis.nb = nb; basis.na = na; basis.mp = mp;
    basis.nc = arena.upload(nc, nb);
    basis.angular = arena.upload(angular, 3 * std::size_t(nb));
    basis.center = arena.upload(center, nb);
    basis.owned = arena.upload(owned, nb);
    basis.exponents = arena.upload(exponents, std::size_t(mp) * nb);
    basis.coefficients = arena.upload(coefficients, std::size_t(mp) * nb);
    basis.xyz = arena.upload(xyz, 3 * std::size_t(na));
    return basis;
}

bool valid_request(int nb, int na, int mp, const int *nc, const int *angular,
        const int *center, const double *exponents, const double *coefficients,
        const double *xyz, const double *field, const double *origin, const int *owned,
        double cutoff, int *ierr) {
    if (nb < 1 || na < 1 || mp < 1 || !std::isfinite(cutoff) || cutoff < 0.0) {
        *ierr = error_code; return false;
    }
    if (std::size_t(nb) * nb > std::numeric_limits<std::size_t>::max() / sizeof(double) ||
            std::size_t(mp) * nb > std::numeric_limits<std::size_t>::max() / sizeof(double)) {
        *ierr = error_code; return false;
    }
    for (int mu = 0; mu < nb; ++mu) {
        int degree = 0;
        if (nc[mu] < 1 || nc[mu] > mp || center[mu] < 1 || center[mu] > na ||
                (owned[mu] != 0 && owned[mu] != 1)) { *ierr = error_code; return false; }
        for (int axis = 0; axis < 3; ++axis) {
            int power = angular[3 * std::size_t(mu) + axis];
            if (power < 0 || power > max_angular) { *ierr = error_code; return false; }
            degree += power;
        }
        // Explicit rejection, never a CPU fallback or angular truncation.
        if (degree > max_angular) { *ierr = error_code; return false; }
        for (int primitive = 0; primitive < nc[mu]; ++primitive) {
            std::size_t index = primitive + std::size_t(mp) * mu;
            if (!std::isfinite(exponents[index]) || exponents[index] <= 0.0 ||
                    !std::isfinite(coefficients[index])) { *ierr = error_code; return false; }
        }
    }
    for (std::size_t index = 0; index < 3 * std::size_t(na); ++index)
        if (!std::isfinite(xyz[index])) { *ierr = error_code; return false; }
    for (int axis = 0; axis < 3; ++axis)
        if (!std::isfinite(field[axis]) || !std::isfinite(origin[axis])) { *ierr = error_code; return false; }
    return true;
}

bool have_work(int nb, const int *owned, const double *field) {
    if (field[0] == 0.0 && field[1] == 0.0 && field[2] == 0.0) return false;
    for (int mu = 0; mu < nb; ++mu) if (owned[mu]) return true;
    return false;
}

// Normalized one-dimensional overlap factors omit exp(-mu*AB^2)*sqrt(pi/g).
// The moment follows integration by parts:
// M_ab = (P-origin)*S_ab + [a*S_(a-1,b)+b*S_(a,b-1)]/(2*g).
// Variants: unchanged, A+, A-, B+, B-. No shifted Gaussian renormalization.
template <bool Gradient>
__device__ __forceinline__ void axis_factors(int la, int lb, double pa, double pb,
        double po, double inverse_two_g, double *overlap, double *moment) {
    double table[max_power + 1][max_power + 1];
    for (int a = 0; a <= max_power; ++a)
        for (int b = 0; b <= max_power; ++b) table[a][b] = 0.0;
    int amax = la + int(Gradient), bmax = lb + int(Gradient);
    table[0][0] = 1.0;
    for (int b = 1; b <= bmax; ++b)
        table[0][b] = pb * table[0][b - 1] + (b > 1 ? (b - 1) * inverse_two_g * table[0][b - 2] : 0.0);
    for (int a = 1; a <= amax; ++a)
        for (int b = 0; b <= bmax; ++b)
            table[a][b] = pa * table[a - 1][b]
                + (a > 1 ? (a - 1) * inverse_two_g * table[a - 2][b] : 0.0)
                + (b ? b * inverse_two_g * table[a - 1][b - 1] : 0.0);
    for (int variant = 0; variant < (Gradient ? 5 : 1); ++variant) {
        int a = la + int(variant == 1) - int(variant == 2);
        int b = lb + int(variant == 3) - int(variant == 4);
        overlap[variant] = 0.0;
        moment[variant] = 0.0;
        if (a < 0 || b < 0) continue;
        overlap[variant] = table[a][b];
        moment[variant] = po * table[a][b]
            + inverse_two_g * ((a ? a * table[a - 1][b] : 0.0) + (b ? b * table[a][b - 1] : 0.0));
    }
}

__device__ __forceinline__ double field_moment(const double overlap[3][5],
        const double moment[3][5], const double *field, int shifted_axis, int variant) {
    double value = 0.0;
    for (int component = 0; component < 3; ++component) {
        if (field[component] == 0.0) continue;
        double term = field[component];
        for (int axis = 0; axis < 3; ++axis) {
            int selected = axis == shifted_axis ? variant : 0;
            term *= axis == component ? moment[axis][selected] : overlap[axis][selected];
        }
        value += term;
    }
    return value;
}

__device__ __forceinline__ void pair_indices(std::size_t pair, int &mu, int &nu) {
    nu = int((sqrt(8.0 * double(pair) + 1.0) - 1.0) * 0.5);
    while (std::size_t(nu) * (std::size_t(nu) + 1) / 2 > pair) --nu;
    while ((std::size_t(nu) + 1) * (std::size_t(nu) + 2) / 2 <= pair) ++nu;
    mu = int(pair - std::size_t(nu) * (std::size_t(nu) + 1) / 2);
}

template <bool Gradient>
__global__ void external_field_kernel(Basis basis, const double *field, const double *origin,
        double cutoff, const double *density, double *matrix, double *gradient, int *bad_result) {
    std::size_t pairs = std::size_t(basis.nb) * (std::size_t(basis.nb) + 1) / 2;
    for (std::size_t pair = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
            pair < pairs; pair += std::size_t(gridDim.x) * blockDim.x) {
        int mu, nu;
        pair_indices(pair, mu, nu);
        if (!basis.owned[mu]) continue;
        int ia = basis.center[mu] - 1, ja = basis.center[nu] - 1;
        const int *la = basis.angular + 3 * mu, *lb = basis.angular + 3 * nu;
        const double *a = basis.xyz + 3 * ia, *b = basis.xyz + 3 * ja;
        double weight = 1.0;
        if (Gradient) {
            weight = density[nu + std::size_t(basis.nb) * mu];
            if (mu != nu) weight *= 2.0;
            if (weight == 0.0) continue;
        }
        double r2 = 0.0;
        for (int axis = 0; axis < 3; ++axis) {
            double distance = a[axis] - b[axis];
            r2 += distance * distance;
        }
        double value = 0.0, da[3] = {0.0, 0.0, 0.0}, db[3] = {0.0, 0.0, 0.0};
        for (int icon = 0; icon < basis.nc[mu]; ++icon)
            for (int jcon = 0; jcon < basis.nc[nu]; ++jcon) {
                std::size_t ai = icon + std::size_t(basis.mp) * mu;
                std::size_t bi = jcon + std::size_t(basis.mp) * nu;
                double alpha = basis.exponents[ai], beta = basis.exponents[bi], g = alpha + beta;
                if (!isfinite(g) || g <= 0.0) { atomicExch(bad_result, 1); continue; }
                double inverse_g = 1.0 / g;
                // Identical primitive screening expression and threshold as CPU opf.
                double prefactor = exp(-alpha * beta * r2 * inverse_g) * sqrt(pi * inverse_g)
                    * pi * inverse_g * basis.coefficients[bi] * basis.coefficients[ai];
                if (!isfinite(prefactor)) { atomicExch(bad_result, 1); continue; }
                if (fabs(prefactor) <= cutoff) continue;
                double overlap[3][5], moment[3][5];
                for (int axis = 0; axis < 3; ++axis) {
                    double distance = b[axis] - a[axis];
                    double pa = beta * inverse_g * distance, pb = -alpha * inverse_g * distance;
                    axis_factors<Gradient>(la[axis], lb[axis], pa, pb, a[axis] - origin[axis] + pa,
                        0.5 * inverse_g, overlap[axis], moment[axis]);
                }
                if (Gradient) for (int axis = 0; axis < 3; ++axis) {
                    double left = 2.0 * alpha * field_moment(overlap, moment, field, axis, 1);
                    if (la[axis]) left -= la[axis] * field_moment(overlap, moment, field, axis, 2);
                    double right = 2.0 * beta * field_moment(overlap, moment, field, axis, 3);
                    if (lb[axis]) right -= lb[axis] * field_moment(overlap, moment, field, axis, 4);
                    da[axis] += prefactor * left;
                    db[axis] += prefactor * right;
                }
                else value += prefactor * field_moment(overlap, moment, field, -1, 0);
            }
        if (Gradient) for (int axis = 0; axis < 3; ++axis) {
            double left = weight * da[axis], right = weight * db[axis];
            if (!isfinite(left) || !isfinite(right)) { atomicExch(bad_result, 1); continue; }
            if (ia == ja) quick_atomic_add_fp64(gradient + 3 * ia + axis, left + right);
            else {
                quick_atomic_add_fp64(gradient + 3 * ia + axis, left);
                quick_atomic_add_fp64(gradient + 3 * ja + axis, right);
            }
        }
        else {
            std::size_t index = nu + std::size_t(basis.nb) * mu;
            double total = matrix[index] + value;
            if (!isfinite(total)) { atomicExch(bad_result, 1); continue; }
            matrix[index] = total;
        }
    }
}

int block_count(std::size_t total) {
    std::size_t blocks = total / 128 + (total % 128 != 0);
    return blocks > 65535 ? 65535 : int(blocks);
}

// Both entry points are additive. Host results are committed only after all
// device work, finite checks, transfers and frees succeed. No persistent GPU
// state, asynchronous lifetime, shared OEI buffer or CPU fallback is involved.
template <bool Gradient>
void evaluate(const int *nb, const int *na, const int *mp, const int *nc, const int *angular,
        const int *center, const double *exponents, const double *coefficients, const double *xyz,
        const double *field, const double *origin, const int *owned, const double *cutoff,
        const double *density, double *output, int *ierr) {
    *ierr = 0;
    if (!valid_request(*nb, *na, *mp, nc, angular, center, exponents, coefficients,
            xyz, field, origin, owned, *cutoff, ierr)) return;
    if (!have_work(*nb, owned, field)) return;
    std::size_t matrix_size = std::size_t(*nb) * *nb;
    std::size_t output_size = Gradient ? 3 * std::size_t(*na) : matrix_size;
    for (std::size_t index = 0; index < output_size; ++index)
        if (!std::isfinite(output[index])) { *ierr = error_code; return; }
    if (Gradient) {
        for (int mu = 0; mu < *nb; ++mu)
            for (int nu = 0; nu <= mu; ++nu) {
                double left = density[mu + std::size_t(*nb) * nu], right = density[nu + std::size_t(*nb) * mu];
                if (!std::isfinite(left) || !std::isfinite(right) || fabs(left - right) > 1.0e-10) {
                    *ierr = error_code; return;
                }
            }
    }
    try {
        Arena arena;
        Basis basis = upload_basis(arena, *nb, *na, *mp, nc, angular, center, exponents, coefficients, xyz, owned);
        double *device_field = arena.upload(field, 3), *device_origin = arena.upload(origin, 3);
        double *device_density = Gradient ? arena.upload(density, matrix_size) : NULL;
        double *device_output = arena.upload(output, output_size);
        int *device_bad = arena.allocate<int>(1), host_bad = 0;
        if (!arena.error) arena.check(cudaMemset(device_bad, 0, sizeof(int)));
        if (!arena.error) {
            std::size_t pairs = std::size_t(*nb) * (std::size_t(*nb) + 1) / 2;
            external_field_kernel<Gradient><<<block_count(pairs), 128>>>(basis, device_field, device_origin,
                *cutoff, device_density, Gradient ? NULL : device_output, Gradient ? device_output : NULL, device_bad);
            if (arena.check(cudaGetLastError())) {
                if (Gradient) ++gradient_launches;
                else ++matrix_launches;
            }
        }
        std::vector<double> result(output_size);
        if (!arena.error) arena.check(cudaMemcpy(&host_bad, device_bad, sizeof(int), cudaMemcpyDeviceToHost));
        if (host_bad) arena.error = error_code;
        if (!arena.error) arena.check(cudaMemcpy(result.data(), device_output, output_size * sizeof(double), cudaMemcpyDeviceToHost));
        if (!arena.error)
            for (std::size_t index = 0; index < output_size; ++index)
                if (!std::isfinite(result[index])) { arena.error = error_code; break; }
        arena.release();
        *ierr = arena.error;
        if (!*ierr) std::copy(result.begin(), result.end(), output);
    } catch (const std::bad_alloc &) { *ierr = error_code; }
}
} // namespace

extern "C" void gpu_external_field_matrix_(const int *nb, const int *na, const int *mp,
        const int *nc, const int *angular, const int *center, const double *exponents,
        const double *coefficients, const double *xyz, const double *field, const double *origin,
        const int *owned, const double *cutoff, double *matrix, int *ierr) {
    evaluate<false>(nb, na, mp, nc, angular, center, exponents, coefficients, xyz,
        field, origin, owned, cutoff, NULL, matrix, ierr);
}

extern "C" void gpu_external_field_gradient_(const int *nb, const int *na, const int *mp,
        const int *nc, const int *angular, const int *center, const double *exponents,
        const double *coefficients, const double *xyz, const double *field, const double *origin,
        const int *owned, const double *cutoff, const double *density, double *gradient, int *ierr) {
    evaluate<true>(nb, na, mp, nc, angular, center, exponents, coefficients, xyz,
        field, origin, owned, cutoff, density, gradient, ierr);
}

extern "C" void gpu_external_field_launch_counts_(int *matrix, int *gradient) {
    *matrix = matrix_launches.load();
    *gradient = gradient_launches.load();
}
