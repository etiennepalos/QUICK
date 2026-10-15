/*
 * (C) Copyright 2026 QUICK contributors. All rights reserved.
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. See https://mozilla.org/MPL/2.0/.
 *
 * Independent FP64 Coulomb AO operators for continuum solvation. No QUICK
 * ESP/XC/external-charge buffers, screening settings or simulation objects
 * are changed. The Fortran caller owns MPI distribution and rank selection.
 */
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>
#include "gpu_solvation_atomic.h"

namespace {
const int error_code = 54;
const int max_angular = 3;
const int max_degree = 2 * max_angular + 1;
const int max_r = (max_degree + 1) * (max_degree + 2) * (max_degree + 3) / 6;
const double pi = 3.14159265358979323846264338327950288;
std::atomic<long long> matrix_launches(0), gradient_launches(0);
std::atomic<long long> cache_hits(0), cache_builds(0), cache_gemvs(0);
std::atomic<long long> direct_matrix_launches(0), cache_fallbacks(0);
const std::size_t hard_cache_limit = std::size_t(8) * 1024 * 1024 * 1024;
std::size_t configured_cache_limit = hard_cache_limit;

bool check_cuda(cudaError_t status, int &error) {
    if (status == cudaSuccess) return true;
    if (!error) std::fprintf(stderr, "QUICK continuum Coulomb CUDA: %s\n", cudaGetErrorString(status));
    error = error_code;
    return false;
}

bool check_blas(cublasStatus_t status, int &error) {
    if (status == CUBLAS_STATUS_SUCCESS) return true;
    if (!error) std::fprintf(stderr, "QUICK continuum Coulomb cuBLAS status: %d\n", int(status));
    error = error_code;
    return false;
}

// Only an explicit allocation failure is a benign reason to select the
// direct CUDA operator. Never clear an unrelated asynchronous CUDA error.
bool clear_allocation_error(int &error) {
    cudaError_t pending = cudaGetLastError();
    if (pending == cudaSuccess || pending == cudaErrorMemoryAllocation) return true;
    return check_cuda(pending, error);
}

template <typename T>
bool identical(const std::vector<T> &saved, const T *value, std::size_t count) {
    return saved.size() == count && (count == 0 || std::memcmp(saved.data(), value, count * sizeof(T)) == 0);
}

struct CoulombCache {
    int device, nb, na, mp, np;
    std::size_t pairs, t_bytes, allocated_bytes;
    double *t, *charges, *packed, *matrix;
    cublasHandle_t blas;
    bool have_blas;
    std::vector<void *> allocations;
    std::vector<int> nc, angular, center;
    std::vector<double> exponents, coefficients, xyz, points;

    CoulombCache() : device(-1), nb(0), na(0), mp(0), np(0), pairs(0), t_bytes(0),
        allocated_bytes(0), t(NULL), charges(NULL), packed(NULL), matrix(NULL), have_blas(false) {}
    ~CoulombCache() { release(); }

    int release() {
        int error = 0;
        // This path is also called by default-disabled QUICK jobs: empty
        // cleanup must not initialize CUDA or query any GPU state.
        if (!allocations.empty() || have_blas) {
            int previous;
            if (!check_cuda(cudaGetDevice(&previous), error)) return error;
            if (previous != device && !check_cuda(cudaSetDevice(device), error)) return error;
            if (have_blas) check_blas(cublasDestroy(blas), error);
            have_blas = false;
            while (!allocations.empty()) {
                void *p = allocations.back();
                allocations.pop_back();
                check_cuda(cudaFree(p), error);
            }
            if (previous != device) check_cuda(cudaSetDevice(previous), error);
        }
        device = -1; nb = na = mp = np = 0;
        pairs = t_bytes = allocated_bytes = 0;
        t = charges = packed = matrix = NULL;
        nc.clear(); angular.clear(); center.clear();
        exponents.clear(); coefficients.clear(); xyz.clear(); points.clear();
        return error;
    }

    bool matches(int current_device, int nbas, int natom, int maxprim, int npoints,
            const int *ncontract, const int *ang, const int *centers, const double *expo,
            const double *coeff, const double *atoms, const double *cavity) const {
        if (!t || device != current_device || nb != nbas || na != natom || mp != maxprim || np != npoints)
            return false;
        if (!identical(nc, ncontract, nb) || !identical(angular, ang, 3 * std::size_t(nb)) ||
                !identical(center, centers, nb) || !identical(xyz, atoms, 3 * std::size_t(na)) ||
                !identical(points, cavity, 3 * std::size_t(np))) return false;
        std::size_t offset = 0;
        for (int mu = 0; mu < nb; ++mu) {
            std::size_t count = nc[mu];
            if (std::memcmp(exponents.data() + offset, expo + std::size_t(mp) * mu, count * sizeof(double)) ||
                    std::memcmp(coefficients.data() + offset, coeff + std::size_t(mp) * mu, count * sizeof(double)))
                return false;
            offset += count;
        }
        return true;
    }

    void remember(int current_device, int nbas, int natom, int maxprim, int npoints,
            const int *ncontract, const int *ang, const int *centers, const double *expo,
            const double *coeff, const double *atoms, const double *cavity) {
        device = current_device; nb = nbas; na = natom; mp = maxprim; np = npoints;
        nc.assign(ncontract, ncontract + nb);
        angular.assign(ang, ang + 3 * std::size_t(nb));
        center.assign(centers, centers + nb);
        xyz.assign(atoms, atoms + 3 * std::size_t(na));
        points.assign(cavity, cavity + 3 * std::size_t(np));
        for (int mu = 0; mu < nb; ++mu) {
            exponents.insert(exponents.end(), expo + std::size_t(mp) * mu, expo + std::size_t(mp) * mu + nc[mu]);
            coefficients.insert(coefficients.end(), coeff + std::size_t(mp) * mu, coeff + std::size_t(mp) * mu + nc[mu]);
        }
    }

    // 1=allocated, 0=benign memory shortage, -1=other CUDA error.
    int allocate(double *&pointer, std::size_t count, int &error) {
        cudaError_t status = cudaMalloc(reinterpret_cast<void **>(&pointer), count * sizeof(double));
        if (status == cudaErrorMemoryAllocation) {
            pointer = NULL;
            return clear_allocation_error(error) ? 0 : -1;
        }
        if (!check_cuda(status, error)) { pointer = NULL; return -1; }
        allocations.push_back(pointer);
        allocated_bytes += count * sizeof(double);
        return 1;
    }
};

CoulombCache reverse_cache;

struct Arena {
    int error;
    cudaError_t failure;
    bool quiet_allocation_failure;
    std::vector<void *> allocations;
    explicit Arena(bool quiet_allocation = false) : error(0), failure(cudaSuccess),
        quiet_allocation_failure(quiet_allocation) {}
    ~Arena() { release(); }
    bool check(cudaError_t status) {
        if (status == cudaSuccess) return true;
        if (!error) failure = status;
        if (!error && !(quiet_allocation_failure && status == cudaErrorMemoryAllocation))
            std::fprintf(stderr, "QUICK continuum Coulomb CUDA: %s\n", cudaGetErrorString(status));
        error = error_code;
        return false;
    }
    template <typename T> T *allocate(std::size_t count) {
        if (error) return NULL;
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            error = error_code;
            return NULL;
        }
        T *p = NULL;
        if (!check(cudaMalloc(reinterpret_cast<void **>(&p), count * sizeof(T)))) return NULL;
        allocations.push_back(p);
        return p;
    }
    template <typename T> T *upload(const T *host, std::size_t count) {
        T *p = allocate<T>(count);
        if (p) check(cudaMemcpy(p, host, count * sizeof(T), cudaMemcpyHostToDevice));
        return p;
    }
    void release() {
        while (!allocations.empty()) {
            void *p = allocations.back();
            allocations.pop_back();
            check(cudaFree(p));
        }
    }
};

struct Basis {
    int nb, na, mp;
    int *nc, *angular, *center;
    double *exponents, *coefficients, *xyz;
};

Basis upload_basis(Arena &arena, int nb, int na, int mp, const int *nc,
        const int *angular, const int *center, const double *exponents,
        const double *coefficients, const double *xyz) {
    Basis basis;
    basis.nb = nb; basis.na = na; basis.mp = mp;
    basis.nc = arena.upload(nc, nb);
    basis.angular = arena.upload(angular, 3 * std::size_t(nb));
    basis.center = arena.upload(center, nb);
    basis.exponents = arena.upload(exponents, std::size_t(mp) * nb);
    basis.coefficients = arena.upload(coefficients, std::size_t(mp) * nb);
    basis.xyz = arena.upload(xyz, 3 * std::size_t(na));
    return basis;
}

bool valid_request(int nb, int na, int mp, int np, const int *nc,
        const int *angular, const int *center, const double *exponents,
        const double *coefficients, const double *xyz, const double *points,
        const double *charges, int *ierr) {
    if (*ierr) return false;
    if (nb < 1 || na < 1 || mp < 1 || np < 0) { *ierr = error_code; return false; }
    for (int mu = 0; mu < nb; ++mu) {
        int degree = 0;
        if (nc[mu] < 1 || nc[mu] > mp || center[mu] < 1 || center[mu] > na) {
            *ierr = error_code; return false;
        }
        for (int axis = 0; axis < 3; ++axis) {
            int l = angular[3 * mu + axis];
            if (l < 0 || l > max_angular) { *ierr = error_code; return false; }
            degree += l;
        }
        // No CPU fallback or silent angular-momentum truncation.
        if (degree > max_angular) { *ierr = error_code; return false; }
        for (int primitive = 0; primitive < nc[mu]; ++primitive) {
            std::size_t index = primitive + std::size_t(mp) * mu;
            if (!std::isfinite(exponents[index]) || exponents[index] <= 0.0 ||
                    !std::isfinite(coefficients[index])) { *ierr = error_code; return false; }
        }
    }
    for (std::size_t index = 0; index < 3 * std::size_t(na); ++index)
        if (!std::isfinite(xyz[index])) { *ierr = error_code; return false; }
    for (int point = 0; point < np; ++point) {
        if (!std::isfinite(charges[point])) { *ierr = error_code; return false; }
        for (int axis = 0; axis < 3; ++axis)
            if (!std::isfinite(points[3 * std::size_t(point) + axis])) { *ierr = error_code; return false; }
    }
    std::size_t pairs = std::size_t(nb) * (std::size_t(nb) + 1) / 2;
    if (std::size_t(np) > std::numeric_limits<std::size_t>::max() / pairs) {
        *ierr = error_code; return false;
    }
    return true;
}

__device__ __forceinline__ int r_index(int tx, int ty, int tz, int degree) {
    int remaining = degree - tx;
    int count = (degree + 1) * (degree + 2) * (degree + 3) / 6;
    int tail = (remaining + 1) * (remaining + 2) * (remaining + 3) / 6;
    return count - tail + ty * (remaining + 1) - ty * (ty - 1) / 2 + tz;
}

__device__ __forceinline__ void boys_values(int degree, double t, double *values) {
    double damping = exp(-t);
    double threshold = (double(degree) + 20.0 > 30.0) ? double(degree) + 20.0 : 30.0;
    if (t < threshold) {
        double term = 1.0 / double(2 * degree + 1), total = term;
        for (int k = 1; k <= 1000; ++k) {
            term *= (2.0 * t) / double(2 * degree + 2 * k + 1);
            total += term;
            if (term <= 2.220446049250313080847263336181640625e-16 * total) break;
        }
        values[degree] = damping * total;
        for (int n = degree - 1; n >= 0; --n)
            values[n] = (2.0 * t * values[n + 1] + damping) / double(2 * n + 1);
    } else {
        values[0] = 0.5 * sqrt(pi / t) * erf(sqrt(t));
        for (int n = 1; n <= degree; ++n)
            values[n] = (double(2 * n - 1) * values[n - 1] - damping) / (2.0 * t);
    }
}

// E_t^(la,lb) excludes the Gaussian product exponential.  Only the requested
// angular pair is retained, reducing thread-local storage for the gradients.
__device__ __forceinline__ void hermite_coefficients(int la, int lb, double exponent,
        double pa, double pb, double *coeff) {
    double work[max_degree + 1];
    for (int t = 0; t <= max_degree; ++t) coeff[t] = 0.0;
    coeff[0] = 1.0;
    int degree = 0;
    for (int step = 0; step < la + lb; ++step) {
        double displacement = step < la ? pa : pb;
        for (int t = 0; t <= degree + 1; ++t) {
            double lower = t ? coeff[t - 1] / (2.0 * exponent) : 0.0;
            double same = t <= degree ? displacement * coeff[t] : 0.0;
            double higher = t < degree ? double(t + 1) * coeff[t + 1] : 0.0;
            work[t] = lower + same + higher;
        }
        ++degree;
        for (int t = 0; t <= degree; ++t) coeff[t] = work[t];
    }
}

__device__ __forceinline__ double contract(const int *la, const int *lb,
        const double coeff[3][5][max_degree + 1], const double *r,
        int degree, int axis, int variation) {
    int limits[3] = {la[0] + lb[0], la[1] + lb[1], la[2] + lb[2]};
    if (axis >= 0) limits[axis] += (variation == 1 || variation == 3) ? 1 : -1;
    int vx = axis == 0 ? variation : 0;
    int vy = axis == 1 ? variation : 0;
    int vz = axis == 2 ? variation : 0;
    double value = 0.0;
    for (int tx = 0; tx <= limits[0]; ++tx) {
        double ex = coeff[0][vx][tx];
        for (int ty = 0; ty <= limits[1]; ++ty) {
            double exy = ex * coeff[1][vy][ty];
            for (int tz = 0; tz <= limits[2]; ++tz)
                value += exy * coeff[2][vz][tz] * r[r_index(tx, ty, tz, degree)];
        }
    }
    return value;
}

template <bool Gradient>
__device__ __forceinline__ void primitive(double alpha, double beta, const int *la,
        const int *lb, const double *a, const double *b, const double *point,
        double &value, double *da, double *db) {
    int degree = la[0] + la[1] + la[2] + lb[0] + lb[1] + lb[2] + (Gradient ? 1 : 0);
    double exponent = alpha + beta, reduced = alpha * beta / exponent;
    double product[3], pc[3], ab2 = 0.0, pc2 = 0.0;
    double coeff[3][5][max_degree + 1];
    for (int axis = 0; axis < 3; ++axis) {
        double ab = a[axis] - b[axis];
        ab2 += ab * ab;
        product[axis] = (alpha * a[axis] + beta * b[axis]) / exponent;
        pc[axis] = product[axis] - point[axis];
        pc2 += pc[axis] * pc[axis];
    }
    double prefactor = 2.0 * pi / exponent * exp(-reduced * ab2);
    value = 0.0;
    if (Gradient) for (int axis = 0; axis < 3; ++axis) { da[axis] = 0.0; db[axis] = 0.0; }
    if (prefactor == 0.0) return;
    for (int axis = 0; axis < 3; ++axis) {
        double pa = product[axis] - a[axis], pb = product[axis] - b[axis];
        hermite_coefficients(la[axis], lb[axis], exponent, pa, pb, coeff[axis][0]);
        if (Gradient) {
            hermite_coefficients(la[axis] + 1, lb[axis], exponent, pa, pb, coeff[axis][1]);
            if (la[axis]) hermite_coefficients(la[axis] - 1, lb[axis], exponent, pa, pb, coeff[axis][2]);
            hermite_coefficients(la[axis], lb[axis] + 1, exponent, pa, pb, coeff[axis][3]);
            if (lb[axis]) hermite_coefficients(la[axis], lb[axis] - 1, exponent, pa, pb, coeff[axis][4]);
        }
    }
    double boys[max_degree + 1], base[max_degree + 1];
    boys_values(degree, exponent * pc2, boys);
    double factor = 1.0;
    for (int n = 0; n <= degree; ++n) { base[n] = factor * boys[n]; factor *= -2.0 * exponent; }
    // A compact Cartesian simplex and two adjacent Boys-order layers are
    // sufficient. Only indices of total degree <= degree-n are accessed.
    double layers[2][max_r];
    double *previous = layers[0], *current = layers[1];
    previous[0] = base[degree];
    for (int n = degree - 1; n >= 0; --n) {
        current[0] = base[n];
        for (int tx = 0; tx <= degree - n; ++tx)
            for (int ty = 0; ty <= degree - n - tx; ++ty)
                for (int tz = 0; tz <= degree - n - tx - ty; ++tz) {
                    int index = r_index(tx, ty, tz, degree);
                    if (tx) {
                        current[index] = pc[0] * previous[r_index(tx - 1, ty, tz, degree)];
                        if (tx > 1) current[index] += double(tx - 1) * previous[r_index(tx - 2, ty, tz, degree)];
                    } else if (ty) {
                        current[index] = pc[1] * previous[r_index(tx, ty - 1, tz, degree)];
                        if (ty > 1) current[index] += double(ty - 1) * previous[r_index(tx, ty - 2, tz, degree)];
                    } else if (tz) {
                        current[index] = pc[2] * previous[r_index(tx, ty, tz - 1, degree)];
                        if (tz > 1) current[index] += double(tz - 1) * previous[r_index(tx, ty, tz - 2, degree)];
                    }
                }
        double *swap = previous; previous = current; current = swap;
    }
    value = prefactor * contract(la, lb, coeff, previous, degree, -1, 0);
    if (Gradient) for (int axis = 0; axis < 3; ++axis) {
        da[axis] = 2.0 * alpha * contract(la, lb, coeff, previous, degree, axis, 1);
        if (la[axis]) da[axis] -= double(la[axis]) * contract(la, lb, coeff, previous, degree, axis, 2);
        db[axis] = 2.0 * beta * contract(la, lb, coeff, previous, degree, axis, 3);
        if (lb[axis]) db[axis] -= double(lb[axis]) * contract(la, lb, coeff, previous, degree, axis, 4);
        da[axis] *= prefactor;
        db[axis] *= prefactor;
    }
}

__device__ __forceinline__ void pair_indices(std::size_t pair, int &mu, int &nu) {
    mu = int((sqrt(8.0 * double(pair) + 1.0) - 1.0) * 0.5);
    while (std::size_t(mu) * (mu + 1) / 2 > pair) --mu;
    while (std::size_t(mu + 1) * (mu + 2) / 2 <= pair) ++mu;
    nu = int(pair - std::size_t(mu) * (mu + 1) / 2);
}

template <bool Gradient>
__global__ void coulomb_kernel(Basis basis, int np, const double *points, const double *charges,
        const double *density, double *pairs_out, double *atomic_gradient, double *point_gradient) {
    std::size_t pairs = std::size_t(basis.nb) * (basis.nb + 1) / 2;
    std::size_t total = pairs * np;
    for (std::size_t index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
            index < total; index += std::size_t(gridDim.x) * blockDim.x) {
        std::size_t pair = index % pairs;
        int point = int(index / pairs), mu, nu;
        if (charges[point] == 0.0) continue;
        pair_indices(pair, mu, nu);
        int ia = basis.center[mu] - 1, ja = basis.center[nu] - 1;
        double weight = 1.0;
        if (Gradient) {
            weight = density[mu + std::size_t(basis.nb) * nu];
            if (mu != nu) weight += density[nu + std::size_t(basis.nb) * mu];
            if (weight == 0.0) continue;
        }
        double value = 0.0, da[3] = {0.0, 0.0, 0.0}, db[3] = {0.0, 0.0, 0.0};
        for (int icon = 0; icon < basis.nc[mu]; ++icon)
            for (int jcon = 0; jcon < basis.nc[nu]; ++jcon) {
                std::size_t ai = icon + std::size_t(basis.mp) * mu;
                std::size_t bi = jcon + std::size_t(basis.mp) * nu;
                double coefficient = basis.coefficients[ai] * basis.coefficients[bi];
                if (coefficient == 0.0) continue;
                double primitive_value, primitive_da[3], primitive_db[3];
                primitive<Gradient>(basis.exponents[ai], basis.exponents[bi], basis.angular + 3 * mu,
                    basis.angular + 3 * nu, basis.xyz + 3 * ia, basis.xyz + 3 * ja,
                    points + 3 * std::size_t(point), primitive_value, primitive_da, primitive_db);
                if (Gradient) for (int axis = 0; axis < 3; ++axis) {
                    da[axis] += coefficient * primitive_da[axis];
                    db[axis] += coefficient * primitive_db[axis];
                }
                else value += coefficient * primitive_value;
            }
        if (Gradient) for (int axis = 0; axis < 3; ++axis) {
            double ga = weight * charges[point] * da[axis];
            double gb = weight * charges[point] * db[axis];
            quick_atomic_add_fp64(atomic_gradient + 3 * ia + axis, ga);
            quick_atomic_add_fp64(atomic_gradient + 3 * ja + axis, gb);
            quick_atomic_add_fp64(point_gradient + 3 * std::size_t(point) + axis, -(ga + gb));
        }
        else quick_atomic_add_fp64(pairs_out + pair, charges[point] * value);
    }
}

// Each column contains the same positive, fully contracted T integrals as
// the direct kernel, for one cavity node. Distinct threads own each value;
// no atomics, density, reaction charges or screening enter the cache build.
__global__ void build_cache_kernel(Basis basis, int np, const double *points, double *t) {
    std::size_t pairs = std::size_t(basis.nb) * (basis.nb + 1) / 2;
    std::size_t total = pairs * np;
    for (std::size_t index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
            index < total; index += std::size_t(gridDim.x) * blockDim.x) {
        int mu, nu;
        pair_indices(index % pairs, mu, nu);
        int point = int(index / pairs), ia = basis.center[mu] - 1, ja = basis.center[nu] - 1;
        double value = 0.0;
        for (int icon = 0; icon < basis.nc[mu]; ++icon)
            for (int jcon = 0; jcon < basis.nc[nu]; ++jcon) {
                std::size_t ai = icon + std::size_t(basis.mp) * mu;
                std::size_t bi = jcon + std::size_t(basis.mp) * nu;
                double coefficient = basis.coefficients[ai] * basis.coefficients[bi];
                if (coefficient == 0.0) continue;
                double primitive_value, da[3], db[3];
                primitive<false>(basis.exponents[ai], basis.exponents[bi], basis.angular + 3 * mu,
                    basis.angular + 3 * nu, basis.xyz + 3 * ia, basis.xyz + 3 * ja,
                    points + 3 * std::size_t(point), primitive_value, da, db);
                value += coefficient * primitive_value;
            }
        t[index] = value;
    }
}

__global__ void expand_matrix(int nb, const double *pairs, double *matrix) {
    std::size_t total = std::size_t(nb) * nb;
    for (std::size_t index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
            index < total; index += std::size_t(gridDim.x) * blockDim.x) {
        int mu = int(index % nb), nu = int(index / nb);
        int high = mu > nu ? mu : nu, low = mu > nu ? nu : mu;
        matrix[index] = pairs[std::size_t(high) * (high + 1) / 2 + low];
    }
}

int block_count(std::size_t total, int threads) {
    std::size_t count = total / threads + (total % threads != 0);
    return count > 65535 ? 65535 : int(count);
}

bool abandon_cache(int *ierr) {
    int error = reverse_cache.release();
    if (error) { *ierr = error; return true; }
    ++cache_fallbacks;
    return false;
}

// true means a cached result or a hard error; false selects the unchanged
// direct CUDA operator. The latter is limited to size/budget decisions or
// explicitly identified allocation failures, never numerical/CUDA faults.
bool cached_matrix(int nb, int na, int mp, int np, const int *nc, const int *angular,
        const int *center, const double *exponents, const double *coefficients,
        const double *xyz, const double *points, const double *charges, double *matrix, int *ierr) {
    int device;
    if (!check_cuda(cudaGetDevice(&device), *ierr)) return true;
    bool hit = reverse_cache.matches(device, nb, na, mp, np, nc, angular, center,
        exponents, coefficients, xyz, points);
    if (!hit) {
        *ierr = reverse_cache.release();
        if (*ierr) return true;
        std::size_t pairs = std::size_t(nb) * (std::size_t(nb) + 1) / 2;
        if (!configured_cache_limit || pairs > std::size_t(std::numeric_limits<int>::max()) ||
                pairs > configured_cache_limit / sizeof(double) / std::size_t(np))
            return abandon_cache(ierr);
        std::size_t free_bytes, total_bytes;
        if (!check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), *ierr)) return true;
        std::size_t budget = configured_cache_limit < free_bytes / 4 ? configured_cache_limit : free_bytes / 4;
        std::size_t t_bytes = pairs * std::size_t(np) * sizeof(double);
        std::size_t workspace_bytes = (pairs + std::size_t(nb) * nb + std::size_t(np)) * sizeof(double);
        if (t_bytes > budget || workspace_bytes > budget - t_bytes) return abandon_cache(ierr);
        reverse_cache.device = device;
        cublasStatus_t blas_status = cublasCreate(&reverse_cache.blas);
        if (blas_status == CUBLAS_STATUS_ALLOC_FAILED) {
            if (!clear_allocation_error(*ierr)) return true;
            return abandon_cache(ierr);
        }
        if (!check_blas(blas_status, *ierr)) return true;
        reverse_cache.have_blas = true;
        reverse_cache.pairs = pairs;
        reverse_cache.t_bytes = t_bytes;
        int allocation = reverse_cache.allocate(reverse_cache.t, pairs * np, *ierr);
        if (allocation > 0) allocation = reverse_cache.allocate(reverse_cache.charges, np, *ierr);
        if (allocation > 0) allocation = reverse_cache.allocate(reverse_cache.packed, pairs, *ierr);
        if (allocation > 0) allocation = reverse_cache.allocate(reverse_cache.matrix, std::size_t(nb) * nb, *ierr);
        if (allocation < 0) return true;
        if (allocation == 0) return abandon_cache(ierr);
        Arena arena(true);
        Basis basis = upload_basis(arena, nb, na, mp, nc, angular, center, exponents, coefficients, xyz);
        double *p = arena.upload(points, 3 * std::size_t(np));
        if (arena.error) {
            if (arena.failure != cudaErrorMemoryAllocation) { *ierr = arena.error; return true; }
            if (!clear_allocation_error(*ierr)) return true;
            arena.error = 0;
            arena.failure = cudaSuccess;
            arena.release();
            if (arena.error) { *ierr = arena.error; return true; }
            return abandon_cache(ierr);
        }
        build_cache_kernel<<<block_count(pairs * np, 64), 64>>>(basis, np, p, reverse_cache.t);
        arena.check(cudaGetLastError());
        if (!arena.error) arena.check(cudaDeviceSynchronize());
        arena.release();
        if (arena.error) { *ierr = arena.error; return true; }
        reverse_cache.remember(device, nb, na, mp, np, nc, angular, center, exponents, coefficients, xyz, points);
        ++cache_builds;
    }
    if (!check_cuda(cudaMemcpy(reverse_cache.charges, charges, np * sizeof(double), cudaMemcpyHostToDevice), *ierr)) return true;
    const double one = 1.0, zero = 0.0;
    cublasStatus_t blas_status = cublasDgemv(reverse_cache.blas, CUBLAS_OP_N,
        int(reverse_cache.pairs), np, &one, reverse_cache.t, int(reverse_cache.pairs),
        reverse_cache.charges, 1, &zero, reverse_cache.packed, 1);
    if (blas_status == CUBLAS_STATUS_ALLOC_FAILED) {
        if (!clear_allocation_error(*ierr)) return true;
        return abandon_cache(ierr);
    }
    if (!check_blas(blas_status, *ierr)) return true;
    expand_matrix<<<block_count(std::size_t(nb) * nb, 256), 256>>>(nb, reverse_cache.packed, reverse_cache.matrix);
    if (!check_cuda(cudaGetLastError(), *ierr)) return true;
    if (!check_cuda(cudaMemcpy(matrix, reverse_cache.matrix, std::size_t(nb) * nb * sizeof(double), cudaMemcpyDeviceToHost), *ierr)) return true;
    ++cache_gemvs;
    if (hit) ++cache_hits;
    // Preserve the existing matrix-evaluation counter. The explicit cache
    // counters distinguish T construction, GEMV reuse and direct kernels.
    ++matrix_launches;
    return true;
}
} // namespace

extern "C" void gpu_solvation_coulomb_matrix_(const int *nb, const int *na, const int *mp,
        const int *nc, const int *angular, const int *center, const double *exponents,
        const double *coefficients, const double *xyz, const int *np, const double *points,
        const double *charges, double *matrix, int *ierr) {
    if (!valid_request(*nb, *na, *mp, *np, nc, angular, center, exponents, coefficients,
            xyz, points, charges, ierr)) return;
    if (*np == 0) {
        *ierr = reverse_cache.release();
        for (std::size_t index = 0; index < std::size_t(*nb) * *nb; ++index) matrix[index] = 0.0;
        return;
    }
    if (cached_matrix(*nb, *na, *mp, *np, nc, angular, center, exponents,
            coefficients, xyz, points, charges, matrix, ierr)) return;
    Arena arena;
    Basis basis = upload_basis(arena, *nb, *na, *mp, nc, angular, center, exponents, coefficients, xyz);
    double *p = arena.upload(points, 3 * std::size_t(*np));
    double *q = arena.upload(charges, *np);
    std::size_t pairs = std::size_t(*nb) * (std::size_t(*nb) + 1) / 2;
    double *pair_result = arena.allocate<double>(pairs);
    double *result = arena.allocate<double>(std::size_t(*nb) * *nb);
    if (!arena.error) arena.check(cudaMemset(pair_result, 0, pairs * sizeof(double)));
    if (!arena.error) {
        coulomb_kernel<false><<<block_count(pairs * *np, 64), 64>>>(basis, *np, p, q, NULL, pair_result, NULL, NULL);
        if (arena.check(cudaGetLastError())) { ++matrix_launches; ++direct_matrix_launches; }
    }
    if (!arena.error) {
        expand_matrix<<<block_count(std::size_t(*nb) * *nb, 256), 256>>>(*nb, pair_result, result);
        arena.check(cudaGetLastError());
    }
    if (!arena.error) arena.check(cudaMemcpy(matrix, result, std::size_t(*nb) * *nb * sizeof(double), cudaMemcpyDeviceToHost));
    arena.release();
    *ierr = arena.error;
}

extern "C" void gpu_solvation_coulomb_gradient_(const int *nb, const int *na, const int *mp,
        const int *nc, const int *angular, const int *center, const double *exponents,
        const double *coefficients, const double *xyz, const int *np, const double *points,
        const double *charges, const double *density, double *atomic_gradient,
        double *point_gradient, int *ierr) {
    if (!valid_request(*nb, *na, *mp, *np, nc, angular, center, exponents, coefficients,
            xyz, points, charges, ierr)) return;
    for (std::size_t index = 0; index < std::size_t(*nb) * *nb; ++index)
        if (!std::isfinite(density[index])) { *ierr = error_code; return; }
    if (*np == 0) {
        for (std::size_t index = 0; index < 3 * std::size_t(*na); ++index) atomic_gradient[index] = 0.0;
        return;
    }
    Arena arena;
    Basis basis = upload_basis(arena, *nb, *na, *mp, nc, angular, center, exponents, coefficients, xyz);
    double *p = arena.upload(points, 3 * std::size_t(*np)), *q = arena.upload(charges, *np);
    double *d = arena.upload(density, std::size_t(*nb) * *nb);
    double *ga = arena.allocate<double>(3 * std::size_t(*na));
    double *gc = arena.allocate<double>(3 * std::size_t(*np));
    if (!arena.error) arena.check(cudaMemset(ga, 0, 3 * std::size_t(*na) * sizeof(double)));
    if (!arena.error) arena.check(cudaMemset(gc, 0, 3 * std::size_t(*np) * sizeof(double)));
    if (!arena.error) {
        std::size_t pairs = std::size_t(*nb) * (std::size_t(*nb) + 1) / 2;
        coulomb_kernel<true><<<block_count(pairs * *np, 64), 64>>>(basis, *np, p, q, d, NULL, ga, gc);
        if (arena.check(cudaGetLastError())) ++gradient_launches;
    }
    if (!arena.error) arena.check(cudaMemcpy(atomic_gradient, ga, 3 * std::size_t(*na) * sizeof(double), cudaMemcpyDeviceToHost));
    if (!arena.error) arena.check(cudaMemcpy(point_gradient, gc, 3 * std::size_t(*np) * sizeof(double), cudaMemcpyDeviceToHost));
    arena.release();
    *ierr = arena.error;
}

extern "C" void gpu_solvation_coulomb_counters_(long long *matrix, long long *gradient, const int *reset) {
    if (*reset) {
        *matrix = matrix_launches.exchange(0);
        *gradient = gradient_launches.exchange(0);
    } else {
        *matrix = matrix_launches.load();
        *gradient = gradient_launches.load();
    }
}

extern "C" void gpu_solvation_coulomb_cache_info_(long long *hits, long long *builds,
        long long *bytes, long long *gemvs, long long *direct_calls, long long *fallbacks,
        long long *allocated_bytes) {
    *hits = cache_hits.load(); *builds = cache_builds.load();
    *bytes = static_cast<long long>(reverse_cache.t_bytes);
    *allocated_bytes = static_cast<long long>(reverse_cache.allocated_bytes);
    *gemvs = cache_gemvs.load(); *direct_calls = direct_matrix_launches.load();
    *fallbacks = cache_fallbacks.load();
}

extern "C" void gpu_solvation_coulomb_cache_clear_(int *ierr, const int *reset) {
    // Cleanup must also run on an earlier scientific error, preserving the
    // first error for the caller while releasing this optional allocation.
    int error = reverse_cache.release();
    if (!*ierr) *ierr = error;
    if (*reset) {
        cache_hits = 0; cache_builds = 0; cache_gemvs = 0;
        direct_matrix_launches = 0; cache_fallbacks = 0;
        matrix_launches = 0; gradient_launches = 0;
        configured_cache_limit = hard_cache_limit;
    }
}

extern "C" void gpu_solvation_coulomb_cache_set_limit_(const long long *bytes, int *ierr) {
    if (*ierr) return;
    if (*bytes < 0) { *ierr = error_code; return; }
    *ierr = reverse_cache.release();
    if (*ierr) return;
    configured_cache_limit = static_cast<unsigned long long>(*bytes) < hard_cache_limit
        ? std::size_t(*bytes) : hard_cache_limit;
}
