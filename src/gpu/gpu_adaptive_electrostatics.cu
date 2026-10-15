/*
 * (C) Copyright 2026 QUICK contributors. All rights reserved.
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Independent FP64 query of a fixed electronic scalar-potential cover.
 * Compact normalized weights and ALL their first/second derivatives enter
 * V, E=-grad(V), and G=-Hess(V). Classical sources and direct H are handled
 * by the Fortran caller. Anchors/whole-support majorants are supplied, not
 * constructed here. This file never changes a published direct ESP path.
 *
 * Fortran column-major arrays: va(na), ea(3,na), ga(3,3,na), ha(3,3,3,na),
 * majorants(2,na), optional anchor_errors(4,na), points(3,np), outputs v(np),
 * e(3,np), g(3,3,np), budgets(3,np), where na=nside^3, z varies fastest.
 * Outputs are overwritten only after the whole transaction succeeds.
 * Budgets are real-arithmetic Taylor/PoU inequalities evaluated in FP64;
 * they are not total floating-point error certificates. Missing anchor
 * errors are unavailable, not a declaration of exact numerical anchors.
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

namespace {
const int dimension_error = 46;
const int input_error = 55;
const int backend_error = 56;
std::atomic<int> launches(0);
std::atomic<long long> live_bytes(0), last_bytes(0);

struct Allocation {
    void *pointer;
    std::size_t bytes;
};

struct Arena {
    int error;
    long long allocated;
    std::vector<Allocation> allocations;
    Arena() : error(0), allocated(0) {}
    ~Arena() { release(); last_bytes.store(allocated); }
    bool check(cudaError_t status) {
        if (status == cudaSuccess) return true;
        if (!error) std::fprintf(stderr, "QUICK adaptive-query CUDA: %s\n", cudaGetErrorString(status));
        error = backend_error;
        return false;
    }
    template <typename T> T *allocate(std::size_t count) {
        if (error) return NULL;
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(T) ||
                count > std::size_t(std::numeric_limits<long long>::max() - allocated) / sizeof(T)) {
            error = backend_error;
            return NULL;
        }
        std::size_t bytes = count * sizeof(T);
        T *pointer = NULL;
        if (!check(cudaMalloc(reinterpret_cast<void **>(&pointer), bytes))) return NULL;
        allocated += static_cast<long long>(bytes);
        live_bytes.fetch_add(static_cast<long long>(bytes));
        try { allocations.push_back(Allocation{pointer, bytes}); }
        catch (const std::bad_alloc &) {
            if (check(cudaFree(pointer))) live_bytes.fetch_sub(static_cast<long long>(bytes));
            error = backend_error;
            return NULL;
        }
        return pointer;
    }
    template <typename T> T *upload(const T *host, std::size_t count) {
        T *pointer = allocate<T>(count);
        if (pointer) check(cudaMemcpy(pointer, host, count * sizeof(T), cudaMemcpyHostToDevice));
        return pointer;
    }
    void release() {
        while (!allocations.empty()) {
            Allocation item = allocations.back();
            allocations.pop_back();
            if (check(cudaFree(item.pointer))) live_bytes.fetch_sub(static_cast<long long>(item.bytes));
        }
    }
};

bool finite_values(const double *values, std::size_t count, bool nonnegative = false) {
    if (!values) return false;
    for (std::size_t k = 0; k < count; ++k)
        if (!std::isfinite(values[k]) || (nonnegative && values[k] < 0.0)) return false;
    return true;
}

bool valid_request(int ns, int degree, const double *lower, const double *upper,
        const double *va, const double *ea, const double *ga, const double *ha,
        const double *majorants, int has_errors, const double *anchor_errors,
        int np, const double *points, double *v, double *e, double *g, double *budgets,
        double *width) {
    if (!v || !e || !g || !budgets || (has_errors != 0 && has_errors != 1)) return false;
    std::size_t na = std::size_t(ns) * ns * ns;
    const std::size_t limit = std::numeric_limits<std::size_t>::max() / sizeof(double);
    if (na > limit / 27 || std::size_t(np) > limit / 9) return false;
    if (!finite_values(lower, 3) || !finite_values(upper, 3) ||
            !finite_values(va, na) || !finite_values(ea, 3 * na) ||
            !finite_values(ga, 9 * na) || !finite_values(ha, 27 * na) ||
            !finite_values(majorants, 2 * na, true) ||
            !finite_values(points, 3 * std::size_t(np)) ||
            (has_errors && !finite_values(anchor_errors, 4 * na, true))) return false;
    for (int axis = 0; axis < 3; ++axis) {
        if (upper[axis] <= lower[axis]) return false;
        width[axis] = (upper[axis] - lower[axis]) / ns;
        if (!std::isfinite(width[axis]) || width[axis] <= 0.0 ||
                !std::isfinite(1.0 / (width[axis] * width[axis]))) return false;
        for (int point = 0; point < np; ++point)
            if (points[3 * std::size_t(point) + axis] < lower[axis] ||
                    points[3 * std::size_t(point) + axis] > upper[axis]) return false;
    }
    // Reject inconsistent derivative tensors. This is the same global-scale
    // tolerance as the CPU cover; no symmetry projection hides bad anchors.
    double gscale = 1.0, hscale = 1.0;
    for (std::size_t k = 0; k < 9 * na; ++k) gscale = std::max(gscale, std::fabs(ga[k]));
    for (std::size_t k = 0; k < 27 * na; ++k) hscale = std::max(hscale, std::fabs(ha[k]));
    double gtol = 64.0 * std::numeric_limits<double>::epsilon() * gscale;
    double htol = 64.0 * std::numeric_limits<double>::epsilon() * hscale;
    for (std::size_t a = 0; a < na; ++a)
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                if (std::fabs(ga[i + 3 * j + 9 * a] - ga[j + 3 * i + 9 * a]) > gtol) return false;
                for (int k = 0; k < 3; ++k) {
                    double h = ha[i + 3 * j + 9 * k + 27 * a];
                    if (std::fabs(h - ha[j + 3 * i + 9 * k + 27 * a]) > htol ||
                            std::fabs(h - ha[i + 3 * k + 9 * j + 27 * a]) > htol) return false;
                }
            }
    return degree == 2 || degree == 3;
}

struct Cover {
    int nside, degree, has_errors;
    double lower[3], width[3];
    const double *va, *ea, *ga, *ha, *majorants, *anchor_errors, *points;
};

// One independent query per thread. The eight compact patches are local;
// there are no density contractions, inter-point reductions, or atomics in
// scientific arithmetic. Grid-striding keeps launch dimensions bounded.
__global__ void query_kernel(Cover cover, int np, double *v, double *e, double *g,
        double *budgets, int *overlap, int *bad) {
    for (std::size_t point = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
            point < std::size_t(np); point += std::size_t(blockDim.x) * gridDim.x) {
        double q[8] = {}, dq[8][3] = {}, ddq[8][9] = {};
        double p[8] = {}, ep[8][3] = {}, gp[8][9] = {}, remainder[8][3] = {};
        int base[3];
        for (int i = 0; i < 3; ++i)
            base[i] = static_cast<int>(floor((cover.points[3 * point + i] - cover.lower[i]) / cover.width[i] - 0.5));
        int patch = 0;
        for (int ix = 0; ix < 2; ++ix)
            for (int iy = 0; iy < 2; ++iy)
                for (int iz = 0; iz < 2; ++iz, ++patch) {
                    int ijk[3] = {base[0] + ix, base[1] + iy, base[2] + iz};
                    if (ijk[0] < 0 || ijk[1] < 0 || ijk[2] < 0 ||
                            ijk[0] >= cover.nside || ijk[1] >= cover.nside || ijk[2] >= cover.nside) continue;
                    std::size_t a = (std::size_t(ijk[0]) * cover.nside + ijk[1]) * cover.nside + ijk[2];
                    double u[3], t[3], psi[3], first[3], second[3];
                    bool active = true;
                    for (int i = 0; i < 3; ++i) {
                        double center = cover.lower[i] + (ijk[i] + 0.5) * cover.width[i];
                        u[i] = cover.points[3 * point + i] - center;
                        t[i] = u[i] / cover.width[i];
                        if (fabs(t[i]) >= 1.0 || !isfinite(t[i])) active = false;
                        double z = 1.0 - t[i] * t[i];
                        psi[i] = z * z * z * z;
                        first[i] = -8.0 * t[i] * z * z * z / cover.width[i];
                        second[i] = 8.0 * z * z * (7.0 * t[i] * t[i] - 1.0) /
                            (cover.width[i] * cover.width[i]);
                    }
                    if (!active) continue;
                    q[patch] = psi[0] * psi[1] * psi[2];
                    for (int i = 0; i < 3; ++i) {
                        dq[patch][i] = first[i];
                        ddq[patch][i + 3 * i] = second[i];
                        for (int axis = 0; axis < 3; ++axis) {
                            if (axis == i) continue;
                            dq[patch][i] *= psi[axis];
                            ddq[patch][i + 3 * i] *= psi[axis];
                        }
                        for (int j = i + 1; j < 3; ++j) {
                            int other = 3 - i - j;
                            ddq[patch][i + 3 * j] = first[i] * first[j] * psi[other];
                            ddq[patch][j + 3 * i] = ddq[patch][i + 3 * j];
                        }
                    }
                    double gu[3] = {}, hu[9] = {}, huu[3] = {};
                    for (int i = 0; i < 3; ++i)
                        for (int j = 0; j < 3; ++j) {
                            double gij = cover.ga[i + 3 * j + 9 * a];
                            gu[i] += gij * u[j];
                            gp[patch][i + 3 * j] = gij;
                            if (cover.degree == 3)
                                for (int k = 0; k < 3; ++k)
                                    hu[i + 3 * j] += cover.ha[i + 3 * j + 9 * k + 27 * a] * u[k];
                        }
                    p[patch] = cover.va[a];
                    for (int i = 0; i < 3; ++i) {
                        p[patch] -= cover.ea[i + 3 * a] * u[i];
                        p[patch] -= 0.5 * u[i] * gu[i];
                        ep[patch][i] = cover.ea[i + 3 * a] + gu[i];
                    }
                    if (cover.degree == 3) {
                        for (int i = 0; i < 3; ++i)
                            for (int j = 0; j < 3; ++j) {
                                huu[i] += hu[i + 3 * j] * u[j];
                                gp[patch][i + 3 * j] += hu[i + 3 * j];
                            }
                        for (int i = 0; i < 3; ++i) {
                            p[patch] -= huu[i] * u[i] / 6.0;
                            ep[patch][i] += 0.5 * huu[i];
                        }
                    }
                    double r2 = 0.0;
                    for (int i = 0; i < 3; ++i) r2 += u[i] * u[i];
                    double r = sqrt(r2), m = cover.majorants[cover.degree - 2 + 2 * a];
                    if (cover.degree == 2) {
                        remainder[patch][0] = m * r * r * r / 6.0;
                        remainder[patch][1] = m * r2 / 2.0;
                        remainder[patch][2] = m * r;
                    } else {
                        remainder[patch][0] = m * r2 * r2 / 24.0;
                        remainder[patch][1] = m * r * r * r / 6.0;
                        remainder[patch][2] = m * r2 / 2.0;
                    }
                    if (cover.has_errors) {
                        const double *delta = cover.anchor_errors + 4 * a;
                        remainder[patch][0] += delta[0] + delta[1] * r + delta[2] * r2 / 2.0;
                        remainder[patch][1] += delta[1] + delta[2] * r;
                        remainder[patch][2] += delta[2];
                        if (cover.degree == 3) {
                            remainder[patch][0] += delta[3] * r * r * r / 6.0;
                            remainder[patch][1] += delta[3] * r2 / 2.0;
                            remainder[patch][2] += delta[3] * r;
                        }
                    }
                }
        double s = 0.0, ds[3] = {}, dds[9] = {};
        int active = 0, reference = 0;
        for (patch = 0; patch < 8; ++patch) {
            s += q[patch];
            if (q[patch] > 0.0) ++active;
            if (q[patch] > q[reference]) reference = patch;
            for (int i = 0; i < 3; ++i) ds[i] += dq[patch][i];
            for (int ij = 0; ij < 9; ++ij) dds[ij] += ddq[patch][ij];
        }
        const double min_sum = 0.031676352024078369140625 * (1.0 - 1.0e-12); // (3/4)^12
        if (!isfinite(s) || s < min_sum) { atomicExch(bad, 1); continue; }
        atomicMax(overlap, active);
        double value = p[reference], field[3] = {}, gradient[9] = {}, bound[3] = {};
        for (patch = 0; patch < 8; ++patch) {
            double weight = q[patch] / s, dw[3], ddw[9], dn2 = 0.0, ddn2 = 0.0;
            double shifted = p[patch] - p[reference];
            value += weight * shifted;
            for (int i = 0; i < 3; ++i) {
                dw[i] = (dq[patch][i] - weight * ds[i]) / s;
                dn2 += dw[i] * dw[i];
                field[i] += weight * ep[patch][i] - shifted * dw[i];
            }
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j) {
                    int ij = i + 3 * j;
                    ddw[ij] = (ddq[patch][ij] - weight * dds[ij] - dw[i] * ds[j] - ds[i] * dw[j]) / s;
                    ddn2 += ddw[ij] * ddw[ij];
                    gradient[ij] += weight * gp[patch][ij] + dw[i] * ep[patch][j] +
                        ep[patch][i] * dw[j] - shifted * ddw[ij];
                }
            double dn = sqrt(dn2), ddn = sqrt(ddn2);
            for (int i = 0; i < 3; ++i) bound[i] += weight * remainder[patch][i];
            bound[1] += dn * remainder[patch][0];
            bound[2] += 2.0 * dn * remainder[patch][1] + ddn * remainder[patch][0];
        }
        bool finite = isfinite(value);
        for (int i = 0; i < 3; ++i) finite = finite && isfinite(field[i]) && isfinite(bound[i]);
        for (int ij = 0; ij < 9; ++ij) finite = finite && isfinite(gradient[ij]);
        if (!finite) { atomicExch(bad, 1); continue; }
        v[point] = value;
        for (int i = 0; i < 3; ++i) { e[3 * point + i] = field[i]; budgets[3 * point + i] = bound[i]; }
        for (int ij = 0; ij < 9; ++ij) g[9 * point + ij] = gradient[ij];
    }
}
} // namespace

extern "C" void gpu_adaptive_electrostatics_(const int *nside, const int *degree,
        const double *lower, const double *upper, const double *va, const double *ea,
        const double *ga, const double *ha, const double *majorants, const int *has_anchor_errors,
        const double *anchor_errors, const int *npoints, const double *points,
        double *v, double *e, double *g, double *budgets, int *max_overlap, int *ierr) {
    last_bytes.store(0);
    if (!ierr) return;
    *ierr = dimension_error;
    if (max_overlap) *max_overlap = 0;
    if (!nside || !degree || !has_anchor_errors || !npoints || !max_overlap ||
            *nside < 1 || *nside > 1024 || *npoints < 1 || (*degree != 2 && *degree != 3)) return;
    if (std::size_t(*nside) * *nside * *nside > std::size_t(std::numeric_limits<int>::max())) return;
    double width[3];
    *ierr = input_error;
    if (!valid_request(*nside, *degree, lower, upper, va, ea, ga, ha, majorants,
            *has_anchor_errors, anchor_errors, *npoints, points, v, e, g, budgets, width)) return;
    try {
        Arena arena;
        std::size_t na = std::size_t(*nside) * *nside * *nside, np = *npoints;
        Cover cover;
        cover.nside = *nside; cover.degree = *degree; cover.has_errors = *has_anchor_errors;
        for (int i = 0; i < 3; ++i) { cover.lower[i] = lower[i]; cover.width[i] = width[i]; }
        cover.va = arena.upload(va, na);
        cover.ea = arena.upload(ea, 3 * na);
        cover.ga = arena.upload(ga, 9 * na);
        cover.ha = arena.upload(ha, 27 * na);
        cover.majorants = arena.upload(majorants, 2 * na);
        cover.anchor_errors = *has_anchor_errors ? arena.upload(anchor_errors, 4 * na) : NULL;
        cover.points = arena.upload(points, 3 * np);
        double *dv = arena.allocate<double>(np), *de = arena.allocate<double>(3 * np);
        double *dg = arena.allocate<double>(9 * np), *db = arena.allocate<double>(3 * np);
        int *device_bad = arena.allocate<int>(1), *device_overlap = arena.allocate<int>(1);
        if (!arena.error) arena.check(cudaMemset(device_bad, 0, sizeof(int)));
        if (!arena.error) arena.check(cudaMemset(device_overlap, 0, sizeof(int)));
        if (!arena.error) {
            unsigned int blocks = static_cast<unsigned int>(std::min<std::size_t>((np + 127) / 128, 65535));
            query_kernel<<<blocks, 128>>>(cover, *npoints, dv, de, dg, db, device_overlap, device_bad);
            if (arena.check(cudaGetLastError())) ++launches;
        }
        int host_bad = 0, host_overlap = 0;
        if (!arena.error) arena.check(cudaMemcpy(&host_bad, device_bad, sizeof(int), cudaMemcpyDeviceToHost));
        if (host_bad) arena.error = input_error;
        if (!arena.error) arena.check(cudaMemcpy(&host_overlap, device_overlap, sizeof(int), cudaMemcpyDeviceToHost));
        std::vector<double> values(np), fields(3 * np), gradients(9 * np), bounds(3 * np);
        if (!arena.error) arena.check(cudaMemcpy(values.data(), dv, np * sizeof(double), cudaMemcpyDeviceToHost));
        if (!arena.error) arena.check(cudaMemcpy(fields.data(), de, 3 * np * sizeof(double), cudaMemcpyDeviceToHost));
        if (!arena.error) arena.check(cudaMemcpy(gradients.data(), dg, 9 * np * sizeof(double), cudaMemcpyDeviceToHost));
        if (!arena.error) arena.check(cudaMemcpy(bounds.data(), db, 3 * np * sizeof(double), cudaMemcpyDeviceToHost));
        if (!arena.error && (!finite_values(values.data(), np) || !finite_values(fields.data(), 3 * np) ||
                !finite_values(gradients.data(), 9 * np) || !finite_values(bounds.data(), 3 * np, true))) arena.error = input_error;
        arena.release();
        *ierr = arena.error;
        if (!*ierr) {
            std::copy(values.begin(), values.end(), v);
            std::copy(fields.begin(), fields.end(), e);
            std::copy(gradients.begin(), gradients.end(), g);
            std::copy(bounds.begin(), bounds.end(), budgets);
            *max_overlap = host_overlap;
        }
    } catch (const std::bad_alloc &) { *ierr = backend_error; }
      catch (const std::length_error &) { *ierr = backend_error; }
}

extern "C" void gpu_adaptive_electrostatics_launch_count_(int *count) {
    if (count) *count = launches.load();
}

// live counts currently owned device bytes; last records total bytes acquired
// by the latest completed call, including inputs, outputs and status words.
// Successful calls release every allocation before committing host outputs.
extern "C" void gpu_adaptive_electrostatics_allocation_bytes_(long long *live, long long *last) {
    if (live) *live = live_bytes.load();
    if (last) *last = last_bytes.load();
}
