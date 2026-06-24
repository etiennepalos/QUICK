/*
   !---------------------------------------------------------------------!
   ! Copyright (C) 2020-2021 Merz lab                                    !
   ! Copyright (C) 2020-2021 Götz lab                                    !
   !                                                                     !
   ! This Source Code Form is subject to the terms of the Mozilla Public !
   ! License, v. 2.0. If a copy of the MPL was not distributed with this !
   ! file, You can obtain one at http://mozilla.org/MPL/2.0/.            !
   !_____________________________________________________________________!

   !---------------------------------------------------------------------!
   ! This source file contains functions required for QUICK one electron !
   ! integral computation.                                               !
   !---------------------------------------------------------------------!
*/

#if !defined(__QUICK_GPU_OEPROP_H_)
#define __QUICK_GPU_OEPROP_H_

#undef FMT_NAME
#define FMT_NAME FmT
#include "gpu_fmt.h"


__device__ static inline void addint_oeprop(unsigned int I, unsigned int J, unsigned int II, unsigned int JJ,
        unsigned int ipoint, QUICKDouble * const store2)
{
    // obtain the start and final basis function indices for given shells II and JJ for
    // contribution into correct location in Fock matrix.
    int III1 = LOC2(devSim.Qsbasis, II, I, devSim.nshell, 4);
    int III2 = LOC2(devSim.Qfbasis, II, I, devSim.nshell, 4);
    int JJJ1 = LOC2(devSim.Qsbasis, JJ, J, devSim.nshell, 4);
    int JJJ2 = LOC2(devSim.Qfbasis, JJ, J, devSim.nshell, 4);

    for (int III = III1; III <= III2; III++) {
        // devTrans maps a basis function with certain angular momentum to store2 array. Get the correct indices now.
        int i = (int) LOC3(devTrans,
                LOC2(devSim.KLMN, 0, III - 1, 3, devSim.nbasis),
                LOC2(devSim.KLMN, 1, III - 1, 3, devSim.nbasis),
                LOC2(devSim.KLMN, 2, III - 1, 3, devSim.nbasis),
                TRANSDIM, TRANSDIM, TRANSDIM);
        for (int JJJ = MAX(III,JJJ1); JJJ <= JJJ2; JJJ++) {
        // devTrans maps a basis function with certain angular momentum to store2 array. Get the correct indices now.
            int j = (int) LOC3(devTrans, 
                    LOC2(devSim.KLMN, 0, JJJ - 1, 3, devSim.nbasis),
                    LOC2(devSim.KLMN, 1, JJJ - 1, 3, devSim.nbasis),
                    LOC2(devSim.KLMN, 2, JJJ - 1, 3, devSim.nbasis),
                    TRANSDIM, TRANSDIM, TRANSDIM);

            // multiply the integral value by normalization constants.
            QUICKDouble dense_sym_factor;
            if (III != JJJ) {
                dense_sym_factor = 2.0;
            } else {
                dense_sym_factor = 1.0;
            }
            QUICKDouble DENSEJI = (QUICKDouble) LOC2(devSim.dense, JJJ - 1, III - 1, devSim.nbasis, devSim.nbasis);
            if (devSim.is_oshell) {
                DENSEJI = DENSEJI + (QUICKDouble) LOC2(devSim.denseb, JJJ - 1, III - 1, devSim.nbasis, devSim.nbasis);
            }
            QUICKDouble Y = dense_sym_factor * DENSEJI * devSim.cons[III - 1] * devSim.cons[JJJ - 1]
                * LOCSTORE(store2, i - 1, j - 1, STOREDIM, STOREDIM);

#if defined(USE_LEGACY_ATOMICS)
            GPUATOMICADD(&devSim.esp_electronicULL[ipoint], Y, OSCALE);
#else
            atomicAdd(&devSim.esp_electronic[ipoint], Y);
#endif
        }
    }
}

__device__ static inline QUICKDouble oeprop_aux_value(QUICKDouble * const YVerticalTemp, int m)
{
    return VY(0, 0, m);
}

__device__ static QUICKDouble oeprop_att_recurse(int i, int j, int k, int ii, int jj, int kk,
        int m, QUICKDouble * const YVerticalTemp,
        QUICKDouble Ax, QUICKDouble Ay, QUICKDouble Az,
        QUICKDouble Bx, QUICKDouble By, QUICKDouble Bz,
        QUICKDouble Cx, QUICKDouble Cy, QUICKDouble Cz,
        QUICKDouble Px, QUICKDouble Py, QUICKDouble Pz, QUICKDouble g)
{
    if (i + j + k + ii + jj + kk == 0) {
        return oeprop_aux_value(YVerticalTemp, m);
    }

    int iexponents[6] = {i, j, k, ii, jj, kk};
    QUICKDouble center[12];
    center[6] = Cx;
    center[7] = Cy;
    center[8] = Cz;
    center[9] = Px;
    center[10] = Py;
    center[11] = Pz;

    int ilownum = 300;
    int ilowex = 300;
    for (int L = 0; L < 6; L++) {
        if (iexponents[L] < ilowex && iexponents[L] != 0) {
            ilowex = iexponents[L];
            ilownum = L;
        }
    }

    if (ilownum <= 2) {
        center[0] = Ax;
        center[1] = Ay;
        center[2] = Az;
        center[3] = Bx;
        center[4] = By;
        center[5] = Bz;
    } else {
        center[3] = Ax;
        center[4] = Ay;
        center[5] = Az;
        center[0] = Bx;
        center[1] = By;
        center[2] = Bz;
        iexponents[3] = i;
        iexponents[4] = j;
        iexponents[5] = k;
        iexponents[0] = ii;
        iexponents[1] = jj;
        iexponents[2] = kk;
        ilownum -= 3;
    }

    iexponents[ilownum] -= 1;
    QUICKDouble attrec = 0.0;

    QUICKDouble PA = center[9 + ilownum] - center[ilownum];
    if (PA != 0.0) {
        attrec += PA * oeprop_att_recurse(iexponents[0], iexponents[1], iexponents[2],
                iexponents[3], iexponents[4], iexponents[5], m, YVerticalTemp,
                center[0], center[1], center[2], center[3], center[4], center[5],
                center[6], center[7], center[8], center[9], center[10], center[11], g);
    }

    QUICKDouble PC = center[9 + ilownum] - center[6 + ilownum];
    if (PC != 0.0) {
        attrec -= PC * oeprop_att_recurse(iexponents[0], iexponents[1], iexponents[2],
                iexponents[3], iexponents[4], iexponents[5], m + 1, YVerticalTemp,
                center[0], center[1], center[2], center[3], center[4], center[5],
                center[6], center[7], center[8], center[9], center[10], center[11], g);
    }

    if (iexponents[ilownum] != 0) {
        QUICKDouble coeff = ((QUICKDouble) iexponents[ilownum]) / (2.0 * g);
        iexponents[ilownum] -= 1;
        attrec += coeff * (
                oeprop_att_recurse(iexponents[0], iexponents[1], iexponents[2],
                    iexponents[3], iexponents[4], iexponents[5], m, YVerticalTemp,
                    center[0], center[1], center[2], center[3], center[4], center[5],
                    center[6], center[7], center[8], center[9], center[10], center[11], g)
                - oeprop_att_recurse(iexponents[0], iexponents[1], iexponents[2],
                    iexponents[3], iexponents[4], iexponents[5], m + 1, YVerticalTemp,
                    center[0], center[1], center[2], center[3], center[4], center[5],
                    center[6], center[7], center[8], center[9], center[10], center[11], g));
        iexponents[ilownum] += 1;
    }

    if (iexponents[ilownum + 3] != 0) {
        QUICKDouble coeff = ((QUICKDouble) iexponents[ilownum + 3]) / (2.0 * g);
        iexponents[ilownum + 3] -= 1;
        attrec += coeff * (
                oeprop_att_recurse(iexponents[0], iexponents[1], iexponents[2],
                    iexponents[3], iexponents[4], iexponents[5], m, YVerticalTemp,
                    center[0], center[1], center[2], center[3], center[4], center[5],
                    center[6], center[7], center[8], center[9], center[10], center[11], g)
                - oeprop_att_recurse(iexponents[0], iexponents[1], iexponents[2],
                    iexponents[3], iexponents[4], iexponents[5], m + 1, YVerticalTemp,
                    center[0], center[1], center[2], center[3], center[4], center[5],
                    center[6], center[7], center[8], center[9], center[10], center[11], g));
        iexponents[ilownum + 3] += 1;
    }

    return attrec;
}

__device__ static QUICKDouble oeprop_deriv_recurse(int i, int j, int k, int ii, int jj, int kk,
        int idx, int idy, int idz, int m, QUICKDouble * const YVerticalTemp,
        QUICKDouble Ax, QUICKDouble Ay, QUICKDouble Az,
        QUICKDouble Bx, QUICKDouble By, QUICKDouble Bz,
        QUICKDouble Cx, QUICKDouble Cy, QUICKDouble Cz,
        QUICKDouble Px, QUICKDouble Py, QUICKDouble Pz, QUICKDouble g)
{
    if (idx + idy + idz == 0) {
        return oeprop_att_recurse(i, j, k, ii, jj, kk, m, YVerticalTemp,
                Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, Px, Py, Pz, g);
    } else if (i + j + k + ii + jj + kk == 0) {
        if (idx == 2) {
            return -2.0 * g * oeprop_att_recurse(i, j, k, ii, jj, kk, m + 1, YVerticalTemp,
                    Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, Px, Py, Pz, g)
                + 4.0 * g * g * (Px - Cx) * (Px - Cx)
                * oeprop_att_recurse(i, j, k, ii, jj, kk, m + 2, YVerticalTemp,
                    Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, Px, Py, Pz, g);
        } else if (idy == 2) {
            return -2.0 * g * oeprop_att_recurse(i, j, k, ii, jj, kk, m + 1, YVerticalTemp,
                    Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, Px, Py, Pz, g)
                + 4.0 * g * g * (Py - Cy) * (Py - Cy)
                * oeprop_att_recurse(i, j, k, ii, jj, kk, m + 2, YVerticalTemp,
                    Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, Px, Py, Pz, g);
        } else if (idz == 2) {
            return -2.0 * g * oeprop_att_recurse(i, j, k, ii, jj, kk, m + 1, YVerticalTemp,
                    Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, Px, Py, Pz, g)
                + 4.0 * g * g * (Pz - Cz) * (Pz - Cz)
                * oeprop_att_recurse(i, j, k, ii, jj, kk, m + 2, YVerticalTemp,
                    Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, Px, Py, Pz, g);
        } else if (idx == 1 && idy == 1) {
            return 4.0 * g * g * (Px - Cx) * (Py - Cy)
                * oeprop_att_recurse(i, j, k, ii, jj, kk, m + 2, YVerticalTemp,
                    Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, Px, Py, Pz, g);
        } else if (idx == 1 && idz == 1) {
            return 4.0 * g * g * (Px - Cx) * (Pz - Cz)
                * oeprop_att_recurse(i, j, k, ii, jj, kk, m + 2, YVerticalTemp,
                    Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, Px, Py, Pz, g);
        } else if (idy == 1 && idz == 1) {
            return 4.0 * g * g * (Py - Cy) * (Pz - Cz)
                * oeprop_att_recurse(i, j, k, ii, jj, kk, m + 2, YVerticalTemp,
                    Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, Px, Py, Pz, g);
        } else if (idx == 1) {
            return 2.0 * g * (Px - Cx)
                * oeprop_att_recurse(i, j, k, ii, jj, kk, m + 1, YVerticalTemp,
                    Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, Px, Py, Pz, g);
        } else if (idy == 1) {
            return 2.0 * g * (Py - Cy)
                * oeprop_att_recurse(i, j, k, ii, jj, kk, m + 1, YVerticalTemp,
                    Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, Px, Py, Pz, g);
        } else if (idz == 1) {
            return 2.0 * g * (Pz - Cz)
                * oeprop_att_recurse(i, j, k, ii, jj, kk, m + 1, YVerticalTemp,
                    Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, Px, Py, Pz, g);
        }
    }

    int iexponents[6] = {i, j, k, ii, jj, kk};
    QUICKDouble center[12];
    center[6] = Cx;
    center[7] = Cy;
    center[8] = Cz;
    center[9] = Px;
    center[10] = Py;
    center[11] = Pz;

    int ilownum = 300;
    int ilowex = 300;
    for (int L = 0; L < 6; L++) {
        if (iexponents[L] < ilowex && iexponents[L] != 0) {
            ilowex = iexponents[L];
            ilownum = L;
        }
    }

    if (ilownum <= 2) {
        center[0] = Ax;
        center[1] = Ay;
        center[2] = Az;
        center[3] = Bx;
        center[4] = By;
        center[5] = Bz;
    } else {
        center[3] = Ax;
        center[4] = Ay;
        center[5] = Az;
        center[0] = Bx;
        center[1] = By;
        center[2] = Bz;
        iexponents[3] = i;
        iexponents[4] = j;
        iexponents[5] = k;
        iexponents[0] = ii;
        iexponents[1] = jj;
        iexponents[2] = kk;
        ilownum -= 3;
    }

    iexponents[ilownum] -= 1;
    QUICKDouble derivrec = 0.0;

    QUICKDouble PA = center[9 + ilownum] - center[ilownum];
    if (PA != 0.0) {
        derivrec += PA * oeprop_deriv_recurse(iexponents[0], iexponents[1], iexponents[2],
                iexponents[3], iexponents[4], iexponents[5], idx, idy, idz, m, YVerticalTemp,
                center[0], center[1], center[2], center[3], center[4], center[5],
                center[6], center[7], center[8], center[9], center[10], center[11], g);
    }

    QUICKDouble PC = center[9 + ilownum] - center[6 + ilownum];
    if (PC != 0.0) {
        derivrec -= PC * oeprop_deriv_recurse(iexponents[0], iexponents[1], iexponents[2],
                iexponents[3], iexponents[4], iexponents[5], idx, idy, idz, m + 1, YVerticalTemp,
                center[0], center[1], center[2], center[3], center[4], center[5],
                center[6], center[7], center[8], center[9], center[10], center[11], g);
    }

    if (iexponents[ilownum] != 0) {
        QUICKDouble coeff = ((QUICKDouble) iexponents[ilownum]) / (2.0 * g);
        iexponents[ilownum] -= 1;
        derivrec += coeff * (
                oeprop_deriv_recurse(iexponents[0], iexponents[1], iexponents[2],
                    iexponents[3], iexponents[4], iexponents[5], idx, idy, idz, m, YVerticalTemp,
                    center[0], center[1], center[2], center[3], center[4], center[5],
                    center[6], center[7], center[8], center[9], center[10], center[11], g)
                - oeprop_deriv_recurse(iexponents[0], iexponents[1], iexponents[2],
                    iexponents[3], iexponents[4], iexponents[5], idx, idy, idz, m + 1, YVerticalTemp,
                    center[0], center[1], center[2], center[3], center[4], center[5],
                    center[6], center[7], center[8], center[9], center[10], center[11], g));
        iexponents[ilownum] += 1;
    }

    if (iexponents[ilownum + 3] != 0) {
        QUICKDouble coeff = ((QUICKDouble) iexponents[ilownum + 3]) / (2.0 * g);
        iexponents[ilownum + 3] -= 1;
        derivrec += coeff * (
                oeprop_deriv_recurse(iexponents[0], iexponents[1], iexponents[2],
                    iexponents[3], iexponents[4], iexponents[5], idx, idy, idz, m, YVerticalTemp,
                    center[0], center[1], center[2], center[3], center[4], center[5],
                    center[6], center[7], center[8], center[9], center[10], center[11], g)
                - oeprop_deriv_recurse(iexponents[0], iexponents[1], iexponents[2],
                    iexponents[3], iexponents[4], iexponents[5], idx, idy, idz, m + 1, YVerticalTemp,
                    center[0], center[1], center[2], center[3], center[4], center[5],
                    center[6], center[7], center[8], center[9], center[10], center[11], g));
        iexponents[ilownum + 3] += 1;
    }

    if (ilownum == 0 && idx > 0) {
        derivrec += ((QUICKDouble) idx) * oeprop_deriv_recurse(iexponents[0], iexponents[1], iexponents[2],
                iexponents[3], iexponents[4], iexponents[5], idx - 1, idy, idz, m + 1, YVerticalTemp,
                center[0], center[1], center[2], center[3], center[4], center[5],
                center[6], center[7], center[8], center[9], center[10], center[11], g);
    } else if (ilownum == 1 && idy > 0) {
        derivrec += ((QUICKDouble) idy) * oeprop_deriv_recurse(iexponents[0], iexponents[1], iexponents[2],
                iexponents[3], iexponents[4], iexponents[5], idx, idy - 1, idz, m + 1, YVerticalTemp,
                center[0], center[1], center[2], center[3], center[4], center[5],
                center[6], center[7], center[8], center[9], center[10], center[11], g);
    } else if (ilownum == 2 && idz > 0) {
        derivrec += ((QUICKDouble) idz) * oeprop_deriv_recurse(iexponents[0], iexponents[1], iexponents[2],
                iexponents[3], iexponents[4], iexponents[5], idx, idy, idz - 1, m + 1, YVerticalTemp,
                center[0], center[1], center[2], center[3], center[4], center[5],
                center[6], center[7], center[8], center[9], center[10], center[11], g);
    }

    return derivrec;
}

#if defined(USE_LEGACY_ATOMICS)
#define OEPROP_ATOMIC_ADD(double_address, ull_address, value) GPUATOMICADD((ull_address), (value), OSCALE)
#else
#define OEPROP_ATOMIC_ADD(double_address, ull_address, value) atomicAdd((double_address), (value))
#endif

__device__ static inline void addint_oeprop_derivatives(unsigned int I, unsigned int J,
        unsigned int II, unsigned int JJ, unsigned int ipoint, unsigned int totalpoint,
        bool do_efield, bool do_efg,
        QUICKDouble Ax, QUICKDouble Ay, QUICKDouble Az,
        QUICKDouble Bx, QUICKDouble By, QUICKDouble Bz,
        QUICKDouble Cx, QUICKDouble Cy, QUICKDouble Cz,
        QUICKDouble Px, QUICKDouble Py, QUICKDouble Pz, QUICKDouble g,
        QUICKDouble * const YVerticalTemp)
{
    int III1 = LOC2(devSim.Qsbasis, II, I, devSim.nshell, 4);
    int III2 = LOC2(devSim.Qfbasis, II, I, devSim.nshell, 4);
    int JJJ1 = LOC2(devSim.Qsbasis, JJ, J, devSim.nshell, 4);
    int JJJ2 = LOC2(devSim.Qfbasis, JJ, J, devSim.nshell, 4);

    for (int III = III1; III <= III2; III++) {
        int lx1 = (int) LOC2(devSim.KLMN, 0, III - 1, 3, devSim.nbasis);
        int ly1 = (int) LOC2(devSim.KLMN, 1, III - 1, 3, devSim.nbasis);
        int lz1 = (int) LOC2(devSim.KLMN, 2, III - 1, 3, devSim.nbasis);

        for (int JJJ = MAX(III, JJJ1); JJJ <= JJJ2; JJJ++) {
            int lx2 = (int) LOC2(devSim.KLMN, 0, JJJ - 1, 3, devSim.nbasis);
            int ly2 = (int) LOC2(devSim.KLMN, 1, JJJ - 1, 3, devSim.nbasis);
            int lz2 = (int) LOC2(devSim.KLMN, 2, JJJ - 1, 3, devSim.nbasis);

            QUICKDouble dense_sym_factor = (III != JJJ) ? 2.0 : 1.0;
            QUICKDouble DENSEJI = (QUICKDouble) LOC2(devSim.dense, JJJ - 1, III - 1, devSim.nbasis, devSim.nbasis);
            if (devSim.is_oshell) {
                DENSEJI = DENSEJI + (QUICKDouble) LOC2(devSim.denseb, JJJ - 1, III - 1, devSim.nbasis, devSim.nbasis);
            }

            QUICKDouble prefactor = dense_sym_factor * DENSEJI
                * devSim.cons[III - 1] * devSim.cons[JJJ - 1];

            if (do_efield) {
                QUICKDouble ex = prefactor * oeprop_deriv_recurse(lx1, ly1, lz1, lx2, ly2, lz2,
                        1, 0, 0, 0, YVerticalTemp, Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, Px, Py, Pz, g);
                QUICKDouble ey = prefactor * oeprop_deriv_recurse(lx1, ly1, lz1, lx2, ly2, lz2,
                        0, 1, 0, 0, YVerticalTemp, Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, Px, Py, Pz, g);
                QUICKDouble ez = prefactor * oeprop_deriv_recurse(lx1, ly1, lz1, lx2, ly2, lz2,
                        0, 0, 1, 0, YVerticalTemp, Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, Px, Py, Pz, g);

                OEPROP_ATOMIC_ADD(&LOC2(devSim.efield_electronic, 0, ipoint, 3, totalpoint),
                        &LOC2(devSim.efield_electronicULL, 0, ipoint, 3, totalpoint), ex);
                OEPROP_ATOMIC_ADD(&LOC2(devSim.efield_electronic, 1, ipoint, 3, totalpoint),
                        &LOC2(devSim.efield_electronicULL, 1, ipoint, 3, totalpoint), ey);
                OEPROP_ATOMIC_ADD(&LOC2(devSim.efield_electronic, 2, ipoint, 3, totalpoint),
                        &LOC2(devSim.efield_electronicULL, 2, ipoint, 3, totalpoint), ez);
            }

            if (do_efg) {
                QUICKDouble xx = prefactor * oeprop_deriv_recurse(lx1, ly1, lz1, lx2, ly2, lz2,
                        2, 0, 0, 0, YVerticalTemp, Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, Px, Py, Pz, g);
                QUICKDouble xy = prefactor * oeprop_deriv_recurse(lx1, ly1, lz1, lx2, ly2, lz2,
                        1, 1, 0, 0, YVerticalTemp, Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, Px, Py, Pz, g);
                QUICKDouble xz = prefactor * oeprop_deriv_recurse(lx1, ly1, lz1, lx2, ly2, lz2,
                        1, 0, 1, 0, YVerticalTemp, Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, Px, Py, Pz, g);
                QUICKDouble yy = prefactor * oeprop_deriv_recurse(lx1, ly1, lz1, lx2, ly2, lz2,
                        0, 2, 0, 0, YVerticalTemp, Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, Px, Py, Pz, g);
                QUICKDouble yz = prefactor * oeprop_deriv_recurse(lx1, ly1, lz1, lx2, ly2, lz2,
                        0, 1, 1, 0, YVerticalTemp, Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, Px, Py, Pz, g);
                QUICKDouble zz = prefactor * oeprop_deriv_recurse(lx1, ly1, lz1, lx2, ly2, lz2,
                        0, 0, 2, 0, YVerticalTemp, Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, Px, Py, Pz, g);

                OEPROP_ATOMIC_ADD(&LOC2(devSim.efg_electronic, 0, ipoint, 9, totalpoint),
                        &LOC2(devSim.efg_electronicULL, 0, ipoint, 9, totalpoint), xx);
                OEPROP_ATOMIC_ADD(&LOC2(devSim.efg_electronic, 1, ipoint, 9, totalpoint),
                        &LOC2(devSim.efg_electronicULL, 1, ipoint, 9, totalpoint), xy);
                OEPROP_ATOMIC_ADD(&LOC2(devSim.efg_electronic, 2, ipoint, 9, totalpoint),
                        &LOC2(devSim.efg_electronicULL, 2, ipoint, 9, totalpoint), xz);
                OEPROP_ATOMIC_ADD(&LOC2(devSim.efg_electronic, 3, ipoint, 9, totalpoint),
                        &LOC2(devSim.efg_electronicULL, 3, ipoint, 9, totalpoint), xy);
                OEPROP_ATOMIC_ADD(&LOC2(devSim.efg_electronic, 4, ipoint, 9, totalpoint),
                        &LOC2(devSim.efg_electronicULL, 4, ipoint, 9, totalpoint), yy);
                OEPROP_ATOMIC_ADD(&LOC2(devSim.efg_electronic, 5, ipoint, 9, totalpoint),
                        &LOC2(devSim.efg_electronicULL, 5, ipoint, 9, totalpoint), yz);
                OEPROP_ATOMIC_ADD(&LOC2(devSim.efg_electronic, 6, ipoint, 9, totalpoint),
                        &LOC2(devSim.efg_electronicULL, 6, ipoint, 9, totalpoint), xz);
                OEPROP_ATOMIC_ADD(&LOC2(devSim.efg_electronic, 7, ipoint, 9, totalpoint),
                        &LOC2(devSim.efg_electronicULL, 7, ipoint, 9, totalpoint), yz);
                OEPROP_ATOMIC_ADD(&LOC2(devSim.efg_electronic, 8, ipoint, 9, totalpoint),
                        &LOC2(devSim.efg_electronicULL, 8, ipoint, 9, totalpoint), zz);
            }
        }
    }
}

__device__ static inline void iclass_oeprop_derivatives(unsigned int I, unsigned int J,
        unsigned int II, unsigned int JJ, unsigned int ipoint, unsigned int totalpoint,
        unsigned int totalatom, bool do_efield, bool do_efg, QUICKDouble * const YVerticalTemp)
{
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

    for (int i = 0; i < kPrimI * kPrimJ ; ++i) {
        int JJJ = (int) i / kPrimI;
        int III = (int) i - kPrimI * JJJ;

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

            FmT(I + J + 2, Zeta * (SQR(Px - Cx) + SQR(Py - Cy) + SQR(Pz - Cz)), YVerticalTemp);

            // Positive auxiliary integrals are used here. EFIELD is dI/dC and
            // EFG is dE_i/dC_j for the electronic charge density, matching the
            // CPU OEPROP convention. ESP remains the only negative auxiliary path.
            for (int n = 0; n <= I + J + 2; n++) {
                VY(0, 0, n) *= Xcoeff_oei;
            }

            addint_oeprop_derivatives(I, J, II, JJ, ipoint, totalpoint, do_efield, do_efg,
                    Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, Px, Py, Pz, Zeta, YVerticalTemp);
        }
    }
}


__device__ static inline void iclass_oeprop(unsigned int I, unsigned int J, unsigned int II, unsigned int JJ,
        unsigned int ipoint, unsigned int totalpoint, unsigned int totalatom,
        QUICKDouble * const YVerticalTemp, QUICKDouble * const store, QUICKDouble * const store2)
{
    /*
       kAtom A, B  is the coresponding atom for shell II, JJ
       and be careful with the index difference between Fortran and C++,
       Fortran starts array index with 1 and C++ starts 0.
       Ai, Bi, Ci are the coordinates for atom katomA, katomB, katomC,
       which means they are corrosponding coorinates for shell II, JJ and nuclei.
   */
    QUICKDouble Ax = LOC2(devSim.allxyz, 0, devSim.katom[II] - 1, 3, totalatom);
    QUICKDouble Ay = LOC2(devSim.allxyz, 1, devSim.katom[II] - 1, 3, totalatom);
    QUICKDouble Az = LOC2(devSim.allxyz, 2, devSim.katom[II] - 1, 3, totalatom);

    QUICKDouble Bx = LOC2(devSim.allxyz, 0, devSim.katom[JJ] - 1, 3, totalatom);
    QUICKDouble By = LOC2(devSim.allxyz, 1, devSim.katom[JJ] - 1, 3, totalatom);
    QUICKDouble Bz = LOC2(devSim.allxyz, 2, devSim.katom[JJ] - 1, 3, totalatom);

    /*
       kPrimI and kPrimJ indicates the number of primitives in shell II and JJ.
       kStartI, J indicates the starting guassian function for shell II, JJ.
       We retrieve from global memory and save them to register to avoid multiple retrieve.
   */
    int kPrimI = devSim.kprim[II];
    int kPrimJ = devSim.kprim[JJ];

    int kStartI = devSim.kstart[II] - 1;
    int kStartJ = devSim.kstart[JJ] - 1;

    /*
       Store array holds contracted integral values computed using VRR algorithm.
       See J. Chem. Phys. 1986, 84, 3963−3974 for theoretical details.
    */
    // initialize store2 array
    for (int i = Sumindex[J]; i < Sumindex[J + 2]; ++i) {
        for (int j = Sumindex[I]; j < Sumindex[I + 2]; ++j) {
            if (i < STOREDIM && j < STOREDIM) {
                LOCSTORE(store2, j, i, STOREDIM, STOREDIM) = 0.0;
            }
        }
    }

    for (int i = 0; i < kPrimI * kPrimJ ; ++i) {
        int JJJ = (int) i / kPrimI;
        int III = (int) i - kPrimI * JJJ;

        /*
           In the following comments, we have I, J, K, L denote the primitive gaussian function we use, and
           for example, expo(III, ksumtype(II)) stands for the expo for the IIIth primitive guassian function for II shell,
           we use I to express the corresponding index.
           Zeta = expo(I)+expo(J)
           --->                --->
           ->     expo(I) * xyz (I) + expo(J) * xyz(J)
           P  = ---------------------------------------
           expo(I) + expo(J)
           Those two are pre-calculated in CPU stage.

        */
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

        // get Xcoeff, which is a product of overlap prefactor and contraction coefficients
        QUICKDouble Xcoeff_oei = LOC4(devSim.Xcoeff_oei, kStartI + III, kStartJ + JJJ,
                I - devSim.Qstart[II], J - devSim.Qstart[JJ], devSim.jbasis, devSim.jbasis, 2, 2);

        if (abs(Xcoeff_oei) > devSim.coreIntegralCutoff) {
            QUICKDouble Cx = LOC2(devSim.extpointxyz, 0, ipoint, 3, totalpoint);
            QUICKDouble Cy = LOC2(devSim.extpointxyz, 1, ipoint, 3, totalpoint);
            QUICKDouble Cz = LOC2(devSim.extpointxyz, 2, ipoint, 3, totalpoint);

            FmT(I + J, Zeta * (SQR(Px - Cx) + SQR(Py - Cy) + SQR(Pz - Cz)), YVerticalTemp);

            // compute all auxilary integrals and store
            for (int n = 0; n <= I + J; n++) {
                VY(0, 0, n) *= -1.0 * Xcoeff_oei;
            }

            // decompose all attraction integrals to their auxilary integrals through VRR scheme.
            OEint_vertical(I, J,
#if defined(DEBUG_OEI)
                    II, JJ,
#endif
                    Px - Ax, Py - Ay, Pz - Az,
                    Px - Bx, Py - By, Pz - Bz,
                    Px - Cx, Py - Cy, Pz - Cz,
                    1.0 / (2.0 * Zeta), store, YVerticalTemp);

            // sum up primitive integral contributions
            for (int i = Sumindex[J]; i < Sumindex[J + 2]; ++i) {
                for (int j = Sumindex[I]; j < Sumindex[I + 2]; ++j) {
                    if (i < STOREDIM && j < STOREDIM) {
                        LOCSTORE(store2, j, i, STOREDIM, STOREDIM) +=  LOCSTORE(store, j, i, STOREDIM, STOREDIM);
                    }
                }
            }
        }
    }

    // retrive computed integral values from store array and update the Fock matrix
    addint_oeprop(I, J, II, JJ, ipoint, store2);
}


__global__ void getOEPROP_kernel()
{
    unsigned int offset = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int totalThreads = blockDim.x * gridDim.x;
    unsigned int jshell = devSim.Qshell;
    unsigned int totalatom = devSim.natom + devSim.nextatom;
    QUICKULL totalpoint = devSim.nextpoint;
    QUICKULL ncalcs = (QUICKULL) (jshell * jshell * totalpoint);

    for (QUICKULL i = offset; i < ncalcs; i += totalThreads) {
        // use the global index to obtain shell pair. Note that here we obtain
        // a couple of indices that helps us to obtain
        // shell number (ii and jj) and quantum numbers (iii, jjj).
        // For each shell pair, we are going over all the external points before
        // moving to the next shell pair.
        unsigned int idx = (unsigned int) (i / totalpoint);

#if defined(MPIV_GPU)
        if (devSim.mpi_boeicompute[idx] > 0) {
#endif
            unsigned int ipoint = (unsigned int) (i - idx * totalpoint);

            int II = devSim.sorted_OEICutoffIJ[idx].x;
            int JJ = devSim.sorted_OEICutoffIJ[idx].y;

            // get the shell numbers of selected shell pair
            int ii = devSim.sorted_Q[II];
            int jj = devSim.sorted_Q[JJ];

            // Only choose the unique shell pairs
            if (jj >= ii) {
                // get the quantum number (or angular momentum of shells, s=0, p=1 and so on.)
                int iii = devSim.sorted_Qnumber[II];
                int jjj = devSim.sorted_Qnumber[JJ];

                // compute coulomb attraction for the selected shell pair.
                iclass_oeprop(iii, jjj, ii, jj, ipoint, totalpoint, totalatom, devSim.YVerticalTemp + offset,
                        devSim.store + offset, devSim.store2 + offset);
            }
#if defined(MPIV_GPU)
        }
#endif
    }
}

__global__ void getOEPROP_efield_kernel()
{
    unsigned int offset = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int totalThreads = blockDim.x * gridDim.x;
    unsigned int jshell = devSim.Qshell;
    unsigned int totalatom = devSim.natom + devSim.nextatom;
    QUICKULL totalpoint = devSim.nextpoint;
    QUICKULL ncalcs = (QUICKULL) (jshell * jshell * totalpoint);

    for (QUICKULL i = offset; i < ncalcs; i += totalThreads) {
        unsigned int idx = (unsigned int) (i / totalpoint);

#if defined(MPIV_GPU)
        if (devSim.mpi_boeicompute[idx] > 0) {
#endif
            unsigned int ipoint = (unsigned int) (i - idx * totalpoint);

            int II = devSim.sorted_OEICutoffIJ[idx].x;
            int JJ = devSim.sorted_OEICutoffIJ[idx].y;
            int ii = devSim.sorted_Q[II];
            int jj = devSim.sorted_Q[JJ];

            if (jj >= ii) {
                int iii = devSim.sorted_Qnumber[II];
                int jjj = devSim.sorted_Qnumber[JJ];

                iclass_oeprop_derivatives(iii, jjj, ii, jj, ipoint, totalpoint, totalatom,
                        true, false, devSim.YVerticalTemp + offset);
            }
#if defined(MPIV_GPU)
        }
#endif
    }
}

__global__ void getOEPROP_efg_kernel()
{
    unsigned int offset = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int totalThreads = blockDim.x * gridDim.x;
    unsigned int jshell = devSim.Qshell;
    unsigned int totalatom = devSim.natom + devSim.nextatom;
    QUICKULL totalpoint = devSim.nextpoint;
    QUICKULL ncalcs = (QUICKULL) (jshell * jshell * totalpoint);

    for (QUICKULL i = offset; i < ncalcs; i += totalThreads) {
        unsigned int idx = (unsigned int) (i / totalpoint);

#if defined(MPIV_GPU)
        if (devSim.mpi_boeicompute[idx] > 0) {
#endif
            unsigned int ipoint = (unsigned int) (i - idx * totalpoint);

            int II = devSim.sorted_OEICutoffIJ[idx].x;
            int JJ = devSim.sorted_OEICutoffIJ[idx].y;
            int ii = devSim.sorted_Q[II];
            int jj = devSim.sorted_Q[JJ];

            if (jj >= ii) {
                int iii = devSim.sorted_Qnumber[II];
                int jjj = devSim.sorted_Qnumber[JJ];

                iclass_oeprop_derivatives(iii, jjj, ii, jj, ipoint, totalpoint, totalatom,
                        false, true, devSim.YVerticalTemp + offset);
            }
#if defined(MPIV_GPU)
        }
#endif
    }
}


#endif
