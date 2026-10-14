/*
 * (C) Copyright 2026 QUICK contributors
 * All rights reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#ifndef __QUICK_GPU_OEPROP_HERMITE_H_
#define __QUICK_GPU_OEPROP_HERMITE_H_

// EFIELD/EFG Hermite evaluator for s/p/d shell pairs. The positive Boys
// auxiliaries include Xcoeff_oei, so the normalized Hermite expansion starts
// at E_0 = 1. Since R differentiates with respect to P and PC = P-C,
// E_i = dI/dC_i has a minus sign; G_ij = dE_i/dC_j has a plus sign.
// The existing recursive evaluator remains the higher-angular-momentum path.
//
// Each thread reuses its existing, interleaved store/store2 slices. At most
// 7^4 = 2401 Coulomb slots and 3*3*3*5 = 135 coefficient slots are addressed,
// within the allocated STOREDIM_XL^2 >= 84^2 slots per thread. Every value
// read below is written for this primitive before use; no clearing is needed.
static_assert(2401 <= STOREDIM_XL * STOREDIM_XL, "OEPROP Hermite scratch exceeds store");

__device__ static inline int oeprop_hermite_r_index(int t, int u, int v, int m)
{
    return ((m * 7 + t) * 7 + u) * 7 + v;
}

// Keep the generated EFIELD polynomial bytes unchanged and guard its entry
// before evaluating large Cartesian powers.
#define oeprop_hermite_coulomb_efield_direct oeprop_hermite_coulomb_efield_direct_unchecked
#include "gpu_oeprop_efield_direct.h"
#undef oeprop_hermite_coulomb_efield_direct

#if !defined(USE_LEGACY_ATOMICS)
#include "gpu_oeprop_efg_direct.h"
#endif

__device__ static inline int oeprop_hermite_e_index(int axis, int a, int b, int t)
{
    return ((axis * 3 + a) * 3 + b) * 5 + t;
}

__device__ static inline void oeprop_hermite_coulomb_reference(int order,
        QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble g,
        QUICKDouble * const YVerticalTemp, QUICKDouble * const store)
{
    QUICKDouble factor = 1.0;
    for (int m = 0; m <= order; m++) {
        LOCSTORE(store, oeprop_hermite_r_index(0, 0, 0, m), 0, 1, 1)
            = factor * VY(0, 0, m);
        factor *= -2.0 * g;
    }

    // R_(t+1,u,v)^m = t R_(t-1,u,v)^(m+1) + PCx R_(t,u,v)^(m+1).
    // Descending auxiliary order makes all right-hand sides available.
    for (int m = order - 1; m >= 0; m--) {
        for (int t = 0; t <= order - m; t++) {
            for (int u = 0; u <= order - m - t; u++) {
                for (int v = 0; v <= order - m - t - u; v++) {
                    QUICKDouble value;
                    if (t > 0) {
                        value = PCx * LOCSTORE(store,
                                oeprop_hermite_r_index(t - 1, u, v, m + 1), 0, 1, 1);
                        if (t > 1) {
                            value += (t - 1) * LOCSTORE(store,
                                    oeprop_hermite_r_index(t - 2, u, v, m + 1), 0, 1, 1);
                        }
                    } else if (u > 0) {
                        value = PCy * LOCSTORE(store,
                                oeprop_hermite_r_index(t, u - 1, v, m + 1), 0, 1, 1);
                        if (u > 1) {
                            value += (u - 1) * LOCSTORE(store,
                                    oeprop_hermite_r_index(t, u - 2, v, m + 1), 0, 1, 1);
                        }
                    } else if (v > 0) {
                        value = PCz * LOCSTORE(store,
                                oeprop_hermite_r_index(t, u, v - 1, m + 1), 0, 1, 1);
                        if (v > 1) {
                            value += (v - 1) * LOCSTORE(store,
                                    oeprop_hermite_r_index(t, u, v - 2, m + 1), 0, 1, 1);
                        }
                    } else {
                        continue;
                    }
                    LOCSTORE(store, oeprop_hermite_r_index(t, u, v, m), 0, 1, 1) = value;
                }
            }
        }
    }
}

// PC is in bohr and g is in bohr^-2. Bounds select GPU arithmetic only;
// they never screen contributions or select a CPU fallback.
// Reference arithmetic above is retained for legacy EFG and exceptional ranges.
// Positive binary64 magnitudes have the same ordering as their unsigned bits.
// NaNs sort above infinity and retain the original polynomial routing.
__device__ static inline bool oeprop_hermite_efield_reference_range(QUICKDouble pc)
{
    QUICKULL magnitude_bits = (QUICKULL) __double_as_longlong(pc) & 0x7fffffffffffffffULL;
    const QUICKULL lower_bits = (QUICKULL) __double_as_longlong(1.0e-50);
    const QUICKULL upper_bits = (QUICKULL) __double_as_longlong(1.0e50);
    return (magnitude_bits != 0 && magnitude_bits < lower_bits)
        || (magnitude_bits > upper_bits && magnitude_bits <= 0x7ff0000000000000ULL);
}

__device__ static inline void oeprop_hermite_coulomb_efield_direct(int order,
        QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble g,
        QUICKDouble * const YVerticalTemp, QUICKDouble * const store)
{
    if (oeprop_hermite_efield_reference_range(PCx)
            || oeprop_hermite_efield_reference_range(PCy)
            || oeprop_hermite_efield_reference_range(PCz)) {
        oeprop_hermite_coulomb_reference(order, PCx, PCy, PCz, g, YVerticalTemp, store);
        return;
    }
    oeprop_hermite_coulomb_efield_direct_unchecked(order, PCx, PCy, PCz, g,
            YVerticalTemp, store);
}

#if !defined(USE_LEGACY_ATOMICS)
__device__ static inline bool oeprop_hermite_efg_direct_domain(
        QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble g)
{
    QUICKDouble ax = fabs(PCx), ay = fabs(PCy), az = fabs(PCz);
    return g >= 1.0e-20 && g <= 1.0e20
        && (ax == 0.0 || (ax >= 1.0e-20 && ax <= 1.0e20))
        && (ay == 0.0 || (ay >= 1.0e-20 && ay <= 1.0e20))
        && (az == 0.0 || (az >= 1.0e-20 && az <= 1.0e20));
}
#endif

__device__ static inline void oeprop_hermite_coulomb(int order,
        QUICKDouble PCx, QUICKDouble PCy, QUICKDouble PCz, QUICKDouble g,
        QUICKDouble * const YVerticalTemp, QUICKDouble * const store)
{
#if !defined(USE_LEGACY_ATOMICS)
    if (order >= 2 && order <= 6
            && oeprop_hermite_efg_direct_domain(PCx, PCy, PCz, g)) {
        oeprop_hermite_coulomb_efg_direct(order, PCx, PCy, PCz, g, YVerticalTemp, store);
        return;
    }
#endif
    oeprop_hermite_coulomb_reference(order, PCx, PCy, PCz, g, YVerticalTemp, store);
}

__device__ static inline void oeprop_hermite_coefficients(int I, int J,
        QUICKDouble PAx, QUICKDouble PAy, QUICKDouble PAz,
        QUICKDouble PBx, QUICKDouble PBy, QUICKDouble PBz,
        QUICKDouble TwoZetaInv, QUICKDouble * const store2)
{
    for (int axis = 0; axis < 3; axis++) {
        QUICKDouble PA = (axis == 0) ? PAx : ((axis == 1) ? PAy : PAz);
        QUICKDouble PB = (axis == 0) ? PBx : ((axis == 1) ? PBy : PBz);
        for (int a = 0; a <= I; a++) {
            for (int b = 0; b <= J; b++) {
                if (a + b == 0) {
                    LOCSTORE(store2, oeprop_hermite_e_index(axis, 0, 0, 0), 0, 1, 1) = 1.0;
                    continue;
                }
                int previous_a = (a > 0) ? a - 1 : a;
                int previous_b = (a > 0) ? b : b - 1;
                int degree = a + b;
                QUICKDouble shift = (a > 0) ? PA : PB;
                for (int t = 0; t <= degree; t++) {
                    QUICKDouble value = 0.0;
                    if (t < degree) {
                        value = shift * LOCSTORE(store2,
                                oeprop_hermite_e_index(axis, previous_a, previous_b, t), 0, 1, 1);
                    }
                    if (t > 0) {
                        value += TwoZetaInv * LOCSTORE(store2,
                                oeprop_hermite_e_index(axis, previous_a, previous_b, t - 1), 0, 1, 1);
                    }
                    if (t + 1 < degree) {
                        value += (t + 1) * LOCSTORE(store2,
                                oeprop_hermite_e_index(axis, previous_a, previous_b, t + 1), 0, 1, 1);
                    }
                    LOCSTORE(store2, oeprop_hermite_e_index(axis, a, b, t), 0, 1, 1) = value;
                }
            }
        }
    }
}

__device__ static inline void addint_oeprop_efield_hermite(unsigned int I, unsigned int J,
        unsigned int II, unsigned int JJ, unsigned int ipoint, unsigned int totalpoint,
        QUICKDouble * const store, QUICKDouble * const store2)
{
    int III1 = LOC2(devSim.Qsbasis, II, I, devSim.nshell, 4);
    int III2 = LOC2(devSim.Qfbasis, II, I, devSim.nshell, 4);
    int JJJ1 = LOC2(devSim.Qsbasis, JJ, J, devSim.nshell, 4);
    int JJJ2 = LOC2(devSim.Qfbasis, JJ, J, devSim.nshell, 4);

#if !defined(USE_LEGACY_ATOMICS)
    QUICKDouble field_x = 0.0, field_y = 0.0, field_z = 0.0;
#endif

    for (int III = III1; III <= III2; III++) {
        int lx1 = (int) LOC2(devSim.KLMN, 0, III - 1, 3, devSim.nbasis);
        int ly1 = (int) LOC2(devSim.KLMN, 1, III - 1, 3, devSim.nbasis);
        int lz1 = (int) LOC2(devSim.KLMN, 2, III - 1, 3, devSim.nbasis);
        for (int JJJ = MAX(III, JJJ1); JJJ <= JJJ2; JJJ++) {
            int lx2 = (int) LOC2(devSim.KLMN, 0, JJJ - 1, 3, devSim.nbasis);
            int ly2 = (int) LOC2(devSim.KLMN, 1, JJJ - 1, 3, devSim.nbasis);
            int lz2 = (int) LOC2(devSim.KLMN, 2, JJJ - 1, 3, devSim.nbasis);
            QUICKDouble DENSEJI = (QUICKDouble) LOC2(devSim.dense, JJJ - 1, III - 1,
                    devSim.nbasis, devSim.nbasis);
            if (devSim.is_oshell) {
                DENSEJI += (QUICKDouble) LOC2(devSim.denseb, JJJ - 1, III - 1,
                        devSim.nbasis, devSim.nbasis);
            }
            QUICKDouble prefactor = ((III != JJJ) ? 2.0 : 1.0) * DENSEJI
                * devSim.cons[III - 1] * devSim.cons[JJJ - 1];
            QUICKDouble ex = 0.0, ey = 0.0, ez = 0.0;

            for (int t = 0; t <= lx1 + lx2; t++) {
                QUICKDouble Ex = LOCSTORE(store2, oeprop_hermite_e_index(0, lx1, lx2, t), 0, 1, 1);
                for (int u = 0; u <= ly1 + ly2; u++) {
                    QUICKDouble ExEy = Ex * LOCSTORE(store2,
                            oeprop_hermite_e_index(1, ly1, ly2, u), 0, 1, 1);
                    for (int v = 0; v <= lz1 + lz2; v++) {
                        QUICKDouble coefficient = ExEy * LOCSTORE(store2,
                                oeprop_hermite_e_index(2, lz1, lz2, v), 0, 1, 1);
                        ex -= coefficient * LOCSTORE(store, oeprop_hermite_r_index(t + 1, u, v, 0), 0, 1, 1);
                        ey -= coefficient * LOCSTORE(store, oeprop_hermite_r_index(t, u + 1, v, 0), 0, 1, 1);
                        ez -= coefficient * LOCSTORE(store, oeprop_hermite_r_index(t, u, v + 1, 0), 0, 1, 1);
                    }
                }
            }

#if defined(USE_LEGACY_ATOMICS)
            // Keep per-pair fixed-point quantization unchanged.
            OEPROP_ATOMIC_ADD(&LOC2(devSim.efield_electronic, 0, ipoint, 3, totalpoint),
                    &LOC2(devSim.efield_electronicULL, 0, ipoint, 3, totalpoint), prefactor * ex);
            OEPROP_ATOMIC_ADD(&LOC2(devSim.efield_electronic, 1, ipoint, 3, totalpoint),
                    &LOC2(devSim.efield_electronicULL, 1, ipoint, 3, totalpoint), prefactor * ey);
            OEPROP_ATOMIC_ADD(&LOC2(devSim.efield_electronic, 2, ipoint, 3, totalpoint),
                    &LOC2(devSim.efield_electronicULL, 2, ipoint, 3, totalpoint), prefactor * ez);
#else
            field_x += prefactor * ex;
            field_y += prefactor * ey;
            field_z += prefactor * ez;
#endif
        }
    }
#if !defined(USE_LEGACY_ATOMICS)
    // Sum Cartesian pairs locally, then update the component-fast E(3,n)
    // array once per primitive. Density and normalization factors stay per pair.
    OEPROP_ATOMIC_ADD(&LOC2(devSim.efield_electronic, 0, ipoint, 3, totalpoint),
            &LOC2(devSim.efield_electronicULL, 0, ipoint, 3, totalpoint), field_x);
    OEPROP_ATOMIC_ADD(&LOC2(devSim.efield_electronic, 1, ipoint, 3, totalpoint),
            &LOC2(devSim.efield_electronicULL, 1, ipoint, 3, totalpoint), field_y);
    OEPROP_ATOMIC_ADD(&LOC2(devSim.efield_electronic, 2, ipoint, 3, totalpoint),
            &LOC2(devSim.efield_electronicULL, 2, ipoint, 3, totalpoint), field_z);
#endif
}

__device__ static inline void addint_oeprop_efg_hermite(unsigned int I, unsigned int J,
        unsigned int II, unsigned int JJ, unsigned int ipoint, unsigned int totalpoint,
        QUICKDouble * const store, QUICKDouble * const store2)
{
    int III1 = LOC2(devSim.Qsbasis, II, I, devSim.nshell, 4);
    int III2 = LOC2(devSim.Qfbasis, II, I, devSim.nshell, 4);
    int JJJ1 = LOC2(devSim.Qsbasis, JJ, J, devSim.nshell, 4);
    int JJJ2 = LOC2(devSim.Qfbasis, JJ, J, devSim.nshell, 4);

#if !defined(USE_LEGACY_ATOMICS)
    QUICKDouble tensor_xx = 0.0, tensor_xy = 0.0, tensor_xz = 0.0;
    QUICKDouble tensor_yy = 0.0, tensor_yz = 0.0, tensor_zz = 0.0;
#endif

    for (int III = III1; III <= III2; III++) {
        int lx1 = (int) LOC2(devSim.KLMN, 0, III - 1, 3, devSim.nbasis);
        int ly1 = (int) LOC2(devSim.KLMN, 1, III - 1, 3, devSim.nbasis);
        int lz1 = (int) LOC2(devSim.KLMN, 2, III - 1, 3, devSim.nbasis);
        for (int JJJ = MAX(III, JJJ1); JJJ <= JJJ2; JJJ++) {
            int lx2 = (int) LOC2(devSim.KLMN, 0, JJJ - 1, 3, devSim.nbasis);
            int ly2 = (int) LOC2(devSim.KLMN, 1, JJJ - 1, 3, devSim.nbasis);
            int lz2 = (int) LOC2(devSim.KLMN, 2, JJJ - 1, 3, devSim.nbasis);
            QUICKDouble DENSEJI = (QUICKDouble) LOC2(devSim.dense, JJJ - 1, III - 1,
                    devSim.nbasis, devSim.nbasis);
            if (devSim.is_oshell) {
                DENSEJI += (QUICKDouble) LOC2(devSim.denseb, JJJ - 1, III - 1,
                        devSim.nbasis, devSim.nbasis);
            }
            QUICKDouble prefactor = ((III != JJJ) ? 2.0 : 1.0) * DENSEJI
                * devSim.cons[III - 1] * devSim.cons[JJJ - 1];
            QUICKDouble xx = 0.0, xy = 0.0, xz = 0.0;
            QUICKDouble yy = 0.0, yz = 0.0, zz = 0.0;

            for (int t = 0; t <= lx1 + lx2; t++) {
                QUICKDouble Ex = LOCSTORE(store2, oeprop_hermite_e_index(0, lx1, lx2, t), 0, 1, 1);
                for (int u = 0; u <= ly1 + ly2; u++) {
                    QUICKDouble ExEy = Ex * LOCSTORE(store2,
                            oeprop_hermite_e_index(1, ly1, ly2, u), 0, 1, 1);
                    for (int v = 0; v <= lz1 + lz2; v++) {
                        QUICKDouble coefficient = ExEy * LOCSTORE(store2,
                                oeprop_hermite_e_index(2, lz1, lz2, v), 0, 1, 1);
                        xx += coefficient * LOCSTORE(store, oeprop_hermite_r_index(t + 2, u, v, 0), 0, 1, 1);
                        xy += coefficient * LOCSTORE(store, oeprop_hermite_r_index(t + 1, u + 1, v, 0), 0, 1, 1);
                        xz += coefficient * LOCSTORE(store, oeprop_hermite_r_index(t + 1, u, v + 1, 0), 0, 1, 1);
                        yy += coefficient * LOCSTORE(store, oeprop_hermite_r_index(t, u + 2, v, 0), 0, 1, 1);
                        yz += coefficient * LOCSTORE(store, oeprop_hermite_r_index(t, u + 1, v + 1, 0), 0, 1, 1);
                        zz += coefficient * LOCSTORE(store, oeprop_hermite_r_index(t, u, v + 2, 0), 0, 1, 1);
                    }
                }
            }

#if defined(USE_LEGACY_ATOMICS)
            // Keep per-pair fixed-point quantization and symmetric updates unchanged.
            OEPROP_ATOMIC_ADD(&LOC2(devSim.efg_electronic, 0, ipoint, 9, totalpoint),
                    &LOC2(devSim.efg_electronicULL, 0, ipoint, 9, totalpoint), prefactor * xx);
            OEPROP_ATOMIC_ADD(&LOC2(devSim.efg_electronic, 1, ipoint, 9, totalpoint),
                    &LOC2(devSim.efg_electronicULL, 1, ipoint, 9, totalpoint), prefactor * xy);
            OEPROP_ATOMIC_ADD(&LOC2(devSim.efg_electronic, 2, ipoint, 9, totalpoint),
                    &LOC2(devSim.efg_electronicULL, 2, ipoint, 9, totalpoint), prefactor * xz);
            OEPROP_ATOMIC_ADD(&LOC2(devSim.efg_electronic, 3, ipoint, 9, totalpoint),
                    &LOC2(devSim.efg_electronicULL, 3, ipoint, 9, totalpoint), prefactor * xy);
            OEPROP_ATOMIC_ADD(&LOC2(devSim.efg_electronic, 4, ipoint, 9, totalpoint),
                    &LOC2(devSim.efg_electronicULL, 4, ipoint, 9, totalpoint), prefactor * yy);
            OEPROP_ATOMIC_ADD(&LOC2(devSim.efg_electronic, 5, ipoint, 9, totalpoint),
                    &LOC2(devSim.efg_electronicULL, 5, ipoint, 9, totalpoint), prefactor * yz);
            OEPROP_ATOMIC_ADD(&LOC2(devSim.efg_electronic, 6, ipoint, 9, totalpoint),
                    &LOC2(devSim.efg_electronicULL, 6, ipoint, 9, totalpoint), prefactor * xz);
            OEPROP_ATOMIC_ADD(&LOC2(devSim.efg_electronic, 7, ipoint, 9, totalpoint),
                    &LOC2(devSim.efg_electronicULL, 7, ipoint, 9, totalpoint), prefactor * yz);
            OEPROP_ATOMIC_ADD(&LOC2(devSim.efg_electronic, 8, ipoint, 9, totalpoint),
                    &LOC2(devSim.efg_electronicULL, 8, ipoint, 9, totalpoint), prefactor * zz);
#else
            tensor_xx += prefactor * xx;
            tensor_xy += prefactor * xy;
            tensor_xz += prefactor * xz;
            tensor_yy += prefactor * yy;
            tensor_yz += prefactor * yz;
            tensor_zz += prefactor * zz;
#endif
        }
    }
#if !defined(USE_LEGACY_ATOMICS)
    // Sum Cartesian pairs locally, retaining per-pair density/normalization.
    // Update component-fast G(3,3,n) once per primitive, including both
    // symmetric off-diagonal locations without an extra tensor factor of two.
    OEPROP_ATOMIC_ADD(&LOC2(devSim.efg_electronic, 0, ipoint, 9, totalpoint),
            &LOC2(devSim.efg_electronicULL, 0, ipoint, 9, totalpoint), tensor_xx);
    OEPROP_ATOMIC_ADD(&LOC2(devSim.efg_electronic, 1, ipoint, 9, totalpoint),
            &LOC2(devSim.efg_electronicULL, 1, ipoint, 9, totalpoint), tensor_xy);
    OEPROP_ATOMIC_ADD(&LOC2(devSim.efg_electronic, 2, ipoint, 9, totalpoint),
            &LOC2(devSim.efg_electronicULL, 2, ipoint, 9, totalpoint), tensor_xz);
    OEPROP_ATOMIC_ADD(&LOC2(devSim.efg_electronic, 3, ipoint, 9, totalpoint),
            &LOC2(devSim.efg_electronicULL, 3, ipoint, 9, totalpoint), tensor_xy);
    OEPROP_ATOMIC_ADD(&LOC2(devSim.efg_electronic, 4, ipoint, 9, totalpoint),
            &LOC2(devSim.efg_electronicULL, 4, ipoint, 9, totalpoint), tensor_yy);
    OEPROP_ATOMIC_ADD(&LOC2(devSim.efg_electronic, 5, ipoint, 9, totalpoint),
            &LOC2(devSim.efg_electronicULL, 5, ipoint, 9, totalpoint), tensor_yz);
    OEPROP_ATOMIC_ADD(&LOC2(devSim.efg_electronic, 6, ipoint, 9, totalpoint),
            &LOC2(devSim.efg_electronicULL, 6, ipoint, 9, totalpoint), tensor_xz);
    OEPROP_ATOMIC_ADD(&LOC2(devSim.efg_electronic, 7, ipoint, 9, totalpoint),
            &LOC2(devSim.efg_electronicULL, 7, ipoint, 9, totalpoint), tensor_yz);
    OEPROP_ATOMIC_ADD(&LOC2(devSim.efg_electronic, 8, ipoint, 9, totalpoint),
            &LOC2(devSim.efg_electronicULL, 8, ipoint, 9, totalpoint), tensor_zz);
#endif
}

template<int derivative_order>
__device__ static inline void iclass_oeprop_hermite(unsigned int I, unsigned int J,
        unsigned int II, unsigned int JJ, unsigned int ipoint, unsigned int totalpoint,
        unsigned int totalatom, QUICKDouble * const YVerticalTemp,
        QUICKDouble * const store, QUICKDouble * const store2)
{
    static_assert(derivative_order == 1 || derivative_order == 2,
            "OEPROP Hermite evaluator supports first and second derivatives");
    QUICKDouble Ax = LOC2(devSim.allxyz, 0, devSim.katom[II] - 1, 3, totalatom);
    QUICKDouble Ay = LOC2(devSim.allxyz, 1, devSim.katom[II] - 1, 3, totalatom);
    QUICKDouble Az = LOC2(devSim.allxyz, 2, devSim.katom[II] - 1, 3, totalatom);
    QUICKDouble Bx = LOC2(devSim.allxyz, 0, devSim.katom[JJ] - 1, 3, totalatom);
    QUICKDouble By = LOC2(devSim.allxyz, 1, devSim.katom[JJ] - 1, 3, totalatom);
    QUICKDouble Bz = LOC2(devSim.allxyz, 2, devSim.katom[JJ] - 1, 3, totalatom);
    int kPrimI = devSim.kprim[II];
    int kPrimJ = devSim.kprim[JJ];
    int kStartI = devSim.kstart[II] - 1;
    int kStartJ = devSim.kstart[JJ] - 1;
    // Match the CPU derivative shell's Boys seed, including combined SP shells.
    // The small-argument approximation depends on this auxiliary order.
    int boys_order = devSim.Qfinal[II] + devSim.Qfinal[JJ] + 2;

    for (int i = 0; i < kPrimI * kPrimJ; i++) {
        int JJJ = i / kPrimI;
        int III = i - kPrimI * JJJ;
        int ii_start = devSim.prim_start[II];
        int jj_start = devSim.prim_start[JJ];
        QUICKDouble Zeta = LOC2(devSim.expoSum, ii_start + III, jj_start + JJJ,
                devSim.prim_total, devSim.prim_total);
        QUICKDouble Px = LOC2(devSim.weightedCenterX, ii_start + III, jj_start + JJJ,
                devSim.prim_total, devSim.prim_total);
        QUICKDouble Py = LOC2(devSim.weightedCenterY, ii_start + III, jj_start + JJJ,
                devSim.prim_total, devSim.prim_total);
        QUICKDouble Pz = LOC2(devSim.weightedCenterZ, ii_start + III, jj_start + JJJ,
                devSim.prim_total, devSim.prim_total);
        QUICKDouble Xcoeff_oei = LOC4(devSim.Xcoeff_oei, kStartI + III, kStartJ + JJJ,
                I - devSim.Qstart[II], J - devSim.Qstart[JJ], devSim.jbasis, devSim.jbasis, 2, 2);
        if (abs(Xcoeff_oei) > devSim.coreIntegralCutoff) {
            QUICKDouble Cx = LOC2(devSim.extpointxyz, 0, ipoint, 3, totalpoint);
            QUICKDouble Cy = LOC2(devSim.extpointxyz, 1, ipoint, 3, totalpoint);
            QUICKDouble Cz = LOC2(devSim.extpointxyz, 2, ipoint, 3, totalpoint);
            int order = I + J + derivative_order;
            // R values still use the actual Cartesian order; lower-angular
            // SP channels retain the shell-wide Boys seed used on the CPU.
            FmT(boys_order, Zeta * (SQR(Px - Cx) + SQR(Py - Cy) + SQR(Pz - Cz)), YVerticalTemp);
            for (int m = 0; m <= order; m++) {
                VY(0, 0, m) *= Xcoeff_oei;
            }
            if (derivative_order == 1) {
                oeprop_hermite_coulomb_efield_direct(order, Px - Cx, Py - Cy, Pz - Cz,
                        Zeta, YVerticalTemp, store);
            } else {
                oeprop_hermite_coulomb(order, Px - Cx, Py - Cy, Pz - Cz,
                        Zeta, YVerticalTemp, store);
            }
            oeprop_hermite_coefficients(I, J, Px - Ax, Py - Ay, Pz - Az,
                    Px - Bx, Py - By, Pz - Bz, 1.0 / (2.0 * Zeta), store2);
            if (derivative_order == 1) {
                addint_oeprop_efield_hermite(I, J, II, JJ, ipoint, totalpoint, store, store2);
            } else {
                addint_oeprop_efg_hermite(I, J, II, JJ, ipoint, totalpoint, store, store2);
            }
        }
    }
}

#endif
