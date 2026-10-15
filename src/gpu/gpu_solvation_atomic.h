/*
 * (C) Copyright 2026 QUICK contributors. All rights reserved.
 * This Source Code Form is subject to the Mozilla Public License, v. 2.0.
 * Feature-local FP64 addition, including QUICK's older CUDA targets.
 */
#ifndef QUICK_SOLVATION_ATOMIC_H
#define QUICK_SOLVATION_ATOMIC_H

__device__ static inline double quick_atomic_add_fp64(double *address, double value) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
    unsigned long long *bits = reinterpret_cast<unsigned long long *>(address);
    unsigned long long old = *bits, assumed;
    do {
        assumed = old;
        old = atomicCAS(bits, assumed,
            __double_as_longlong(value + __longlong_as_double(assumed)));
        // Integer comparison also terminates for a NaN bit pattern.
    } while (assumed != old);
    return __longlong_as_double(old);
#else
    return atomicAdd(address,value);
#endif
}
#endif
