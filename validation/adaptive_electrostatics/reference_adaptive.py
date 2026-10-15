"""Independent ordinary-FP mathematical pilot, not a production/certified API.

Gaussian coefficients include original primitive/Cartesian normalization.
All nonzero density/primitive contributions are retained. Derivatives are
of the ELECTRONIC potential at fixed geometry and fixed real symmetric D.
"""
# (C) Copyright 2026 QUICK contributors. Mozilla Public License, v. 2.0.
from __future__ import annotations

from itertools import product
from math import factorial, pi, sqrt
from functools import lru_cache

import numpy as np
from scipy.special import hyp1f1


@lru_cache(None)
def components(rank):
    return tuple((x, y, rank-x-y) for x in range(rank, -1, -1)
                 for y in range(rank-x, -1, -1))


def multiplicities(rank):
    return np.array([factorial(rank) / np.prod([factorial(x) for x in c])
                     for c in components(rank)])


def boys(order, t):
    t = np.atleast_1d(t)
    return np.array([hyp1f1(m+.5, m+1.5, -t)/(2*m+1) for m in range(order+1)])


def hermite_coefficients(a, b, pa, pb, inverse_two_p, positive=False):
    e = np.zeros(a+b+1)
    e[0] = 1.
    for raising in range(a+b):
        displacement = pa if raising < a else pb
        if positive:
            displacement = abs(displacement)
        work = np.zeros_like(e)
        for t in range(raising+2):
            if t > 0:
                work[t] += inverse_two_p*e[t-1]
            if t <= raising:
                work[t] += displacement*e[t]
            if t < raising:
                work[t] += (t+1)*e[t+1]
        e = work
    return e


@lru_cache(None)
def slots(order):
    entries = tuple(c for rank in range(order+1) for c in components(rank))
    return entries, {c: i for i, c in enumerate(entries)}


def r_layer(order, p, q, positive=False, tmin=None):
    """Two Cartesian layers only: F10 needs 2*286 slots per point/cell.

    For a cell, q is qmax and tmin=p*distance(P,box)^2. All operations in
    the majorant recurrence are nonnegative; this is NOT outward rounding.
    """
    q = np.atleast_2d(q)
    entries, index = slots(order)
    t = p*np.sum(q*q, axis=1) if tmin is None else np.atleast_1d(tmin)
    f = boys(order, t)
    previous = np.zeros((len(entries), len(q)))
    sign = 1 if positive else -1
    for m in range(order, -1, -1):
        current = np.zeros_like(previous)
        current[0] = (sign*2*p)**m*f[m]
        for rank in range(1, order-m+1):
            for powers in components(rank):
                axis = next(i for i in range(3) if powers[i])
                lower = list(powers)
                lower[axis] -= 1
                value = q[:, axis]*previous[index[tuple(lower)]]
                if powers[axis] > 1:
                    lower[axis] -= 1
                    value = value+(powers[axis]-1)*previous[index[tuple(lower)]]
                current[index[powers]] = value
        previous = current
    return previous


def primitive_normalization(exponent, angular):
    def odd_factorial(n):
        return np.prod(np.arange(n, 0, -2)) if n > 0 else 1
    return (2*exponent/pi)**.75*sqrt((4*exponent)**sum(angular) /
        np.prod([odd_factorial(2*l-1) for l in angular]))


def prepare_pairs(aos, density):
    density = np.asarray(density)
    if density.shape != (len(aos), len(aos)) or not np.allclose(density, density.T, rtol=0, atol=1e-14):
        raise ValueError('Require a finite real symmetric full AO density')
    if not np.all(np.isfinite(density)):
        raise ValueError('Nonfinite density')
    pairs = []
    for mu, a in enumerate(aos):
        for nu, b in enumerate(aos[:mu+1]):
            weight = density[nu, mu]*(1 if mu == nu else 2)
            if weight == 0:
                continue  # Exact zero in the given floating-point input.
            ca, cb = np.array(a['center']), np.array(b['center'])
            for alpha, da in a['primitives']:
                for beta, db in b['primitives']:
                    if da == 0 or db == 0:
                        continue
                    p = alpha+beta
                    displacement = cb-ca
                    pa, pb = beta/p*displacement, -alpha/p*displacement
                    k = np.exp(-alpha*beta/p*np.dot(displacement, displacement))
                    if k == 0:
                        raise FloatingPointError('Underflow is an unknown numerical omission, not zero budget')
                    h = [hermite_coefficients(x, y, u, v, .5/p)
                         for x, y, u, v in zip(a['angular'], b['angular'], pa, pb)]
                    hpos = [hermite_coefficients(x, y, u, v, .5/p, True)
                            for x, y, u, v in zip(a['angular'], b['angular'], pa, pb)]
                    terms = [(t, u, v) for t in range(len(h[0])) for u in range(len(h[1])) for v in range(len(h[2]))]
                    prefactor = 2*pi/p*k*da*db*weight
                    if prefactor == 0 or not np.isfinite(prefactor):
                        raise FloatingPointError('Unknown prefactor underflow/overflow')
                    pairs.append({'p': p, 'P': ca+pa, 'L': sum(a['angular'])+sum(b['angular']),
                                  'prefactor': prefactor,
                                  'terms': terms,
                                  'coefficient': np.array([h[0][t]*h[1][u]*h[2][v] for t, u, v in terms]),
                                  'positive_coefficient': np.array([hpos[0][t]*hpos[1][u]*hpos[2][v] for t, u, v in terms])})
    return pairs


def electronic_jet(pairs, points, rank=4):
    """Packed d^r V_el for r=0..rank, without any primitive screening."""
    points = np.atleast_2d(points)
    values = [np.zeros((len(points), len(components(r)))) for r in range(rank+1)]
    for pair in pairs:
        order = pair['L']+rank
        _, index = slots(order)
        r0 = r_layer(order, pair['p'], pair['P']-points)
        for r in range(rank+1):
            for c, eta in enumerate(components(r)):
                ids = [index[tuple(t[i]+eta[i] for i in range(3))] for t in pair['terms']]
                values[r][:, c] += (-1)**(r+1)*pair['prefactor']*(pair['coefficient'] @ r0[ids])
    return values


def cell_majorants(pairs, lower, upper, rank=4, coefficient_mode='positive'):
    """Full-density real-arithmetic component inequalities, evaluated FP64.

    No anchor/numerical error bound is implied. Positive coefficient
    recurrence is the default; |signed E| can be compared for tightness.
    """
    lower, upper = np.array(lower), np.array(upper)
    if np.any(lower > upper):
        raise ValueError('Invalid cell')
    out = [np.zeros(len(components(r))) for r in range(rank+1)]
    for pair in pairs:
        qlo, qhi = pair['P']-upper, pair['P']-lower
        qmax = np.maximum(abs(qlo), abs(qhi))
        qmin = np.where((qlo <= 0) & (qhi >= 0), 0., np.minimum(abs(qlo), abs(qhi)))
        tmin = pair['p']*np.dot(qmin, qmin)
        order = pair['L']+rank
        _, index = slots(order)
        r0 = r_layer(order, pair['p'], qmax, True, tmin)[:, 0]
        coeff = pair['positive_coefficient'] if coefficient_mode == 'positive' else abs(pair['coefficient'])
        for r in range(rank+1):
            for c, eta in enumerate(components(r)):
                ids = [index[tuple(t[i]+eta[i] for i in range(3))] for t in pair['terms']]
                out[r][c] += abs(pair['prefactor'])*np.dot(coeff, r0[ids])
    if any(not np.all(np.isfinite(v)) for v in out):
        raise FloatingPointError('Nonfinite majorant')
    return out


def tensor(packed, rank):
    packed = np.atleast_2d(packed)
    if rank == 0:
        return packed[:, 0]
    result = np.empty((len(packed),)+(3,)*rank)
    indices = {c: i for i, c in enumerate(components(rank))}
    for axes in product(range(3), repeat=rank):
        eta = tuple(axes.count(i) for i in range(3))
        result[(slice(None),)+axes] = packed[:, indices[eta]]
    return result


def packed_norm(packed, rank):
    return np.sqrt(np.sum(np.asarray(packed)**2*multiplicities(rank), axis=-1))


def validate_error_mode(mode):
    if mode == 'certified':
        raise NotImplementedError('Validated Boys/arithmetic/anchor intervals are unavailable')
    if mode not in ('indicator', 'analytic_majorant'):
        raise ValueError('Unknown error mode')
    return {'mode': mode, 'certified': False,
            'total_numerical_error_budget_available': False}


def gaussian_fourth(points, exponent):
    """Independent closed radial fourth derivative of positive normalized I.

    A normalized s AO has density (2a/pi)^(3/2)*exp(-2a*r^2).
    The electronic potential has the opposite sign.
    """
    u = np.atleast_2d(points)
    p = 2*exponent
    t = p*np.sum(u*u, axis=1)
    f2, f3, f4 = boys(4, t)[2:5]
    prefactor = 2*pi/p*(p/pi)**1.5
    result = np.empty((len(u), 3, 3, 3, 3))
    for i, j, k, l in product(range(3), repeat=4):
        pair_terms = ((i == j)*u[:, k]*u[:, l] + (i == k)*u[:, j]*u[:, l]
                      + (i == l)*u[:, j]*u[:, k] + (j == k)*u[:, i]*u[:, l]
                      + (j == l)*u[:, i]*u[:, k] + (k == l)*u[:, i]*u[:, j])
        delta = (i == j)*(k == l)+(i == k)*(j == l)+(i == l)*(j == k)
        result[:, i, j, k, l] = prefactor*(16*p**4*u[:, i]*u[:, j]*u[:, k]*u[:, l]*f4
                                            - 8*p**3*pair_terms*f3+4*p*p*delta*f2)
    return result


class FixedCover:
    """Uniform leaves, compact support h=leaf width, at most8 active patches.

    Only the real-arithmetic Taylor/POU truncation budget is reported. The
    anchor/arithmetic budget remains explicitly unavailable, never zero.
    """
    def __init__(self, pairs, lower, upper, nside):
        self.lower, self.upper = np.array(lower), np.array(upper)
        self.nside = nside
        self.h = (self.upper-self.lower)/nside
        self.centers = np.array([self.lower+(np.array(i)+.5)*self.h
                                 for i in product(range(nside), repeat=3)])
        self.jet = electronic_jet(pairs, self.centers, 3)
        self.v = tensor(self.jet[0], 0)
        self.e, self.g, self.H = [-tensor(self.jet[r], r) for r in range(1, 4)]
        self.majorants = np.array([[packed_norm(v, rank) for rank, v in enumerate(
            cell_majorants(pairs, center-self.h, center+self.h)) if rank in (3, 4)]
            for center in self.centers])

    def evaluate(self, points, degree, anchor_errors=None):
        points = np.atleast_2d(points)
        if degree not in (2, 3) or np.any(points < self.lower) or np.any(points > self.upper):
            raise ValueError('Require covered points and degree2/3')
        initial = np.floor((points-self.lower)/self.h-.5).astype(int)
        ijk = initial[:, None, :]+np.array(list(product((0, 1), repeat=3)))[None, :, :]
        valid = np.all((ijk >= 0) & (ijk < self.nside), axis=-1)
        ids = np.clip(ijk, 0, self.nside-1)
        ids = (ids[:, :, 0]*self.nside+ids[:, :, 1])*self.nside+ids[:, :, 2]
        u = points[:, None, :]-self.centers[ids]
        t = u/self.h
        inside = (abs(t) < 1)
        psi = np.where(inside, (1-t*t)**4, 0.)
        first = np.where(inside, -8*t*(1-t*t)**3, 0.)/self.h
        second = np.where(inside, 8*(1-t*t)**2*(7*t*t-1), 0.)/self.h**2
        q = np.prod(psi, axis=-1)*valid
        dq = np.zeros(q.shape+(3,))
        ddq = np.zeros(q.shape+(3, 3))
        for i in range(3):
            others = [j for j in range(3) if j != i]
            dq[:, :, i] = first[:, :, i]*np.prod(psi[:, :, others], axis=-1)*valid
            ddq[:, :, i, i] = second[:, :, i]*np.prod(psi[:, :, others], axis=-1)*valid
            for j in range(i):
                k = 3-i-j
                ddq[:, :, i, j] = first[:, :, i]*first[:, :, j]*psi[:, :, k]*valid
                ddq[:, :, j, i] = ddq[:, :, i, j]
        s = np.sum(q, axis=1)
        if np.any(s <= 0):
            raise ValueError('Uncovered query')
        ds, dds = np.sum(dq, axis=1), np.sum(ddq, axis=1)
        w = q/s[:, None]
        dw = (dq-w[:, :, None]*ds[:, None, :])/s[:, None, None]
        ddw = (ddq-w[:, :, None, None]*dds[:, None, :, :]
               - np.einsum('pai,pj->paij', dw, ds)-np.einsum('pi,paj->paij', ds, dw))/s[:, None, None, None]
        ea, ga, ha = self.e[ids], self.g[ids], self.H[ids]
        local_v = self.v[ids]-np.einsum('pai,pai->pa', ea, u)-.5*np.einsum('pai,paij,paj->pa', u, ga, u)
        local_e = ea+np.einsum('paij,paj->pai', ga, u)
        local_g = ga.copy()
        if degree == 3:
            local_v -= np.einsum('paijk,pai,paj,pak->pa', ha, u, u, u)/6
            local_e += .5*np.einsum('paijk,paj,pak->pai', ha, u, u)
            local_g += np.einsum('paijk,pak->paij', ha, u)
        v = np.sum(w*local_v, axis=1)
        e = np.einsum('pa,pai->pi', w, local_e)-np.einsum('pa,pai->pi', local_v, dw)
        g = (np.einsum('pa,paij->pij', w, local_g)+np.einsum('pai,paj->pij', dw, local_e)
             + np.einsum('pai,paj->pij', local_e, dw)-np.einsum('pa,paij->pij', local_v, ddw))
        radius = np.linalg.norm(self.h)
        m = self.majorants[ids, degree-2]
        if degree == 2:
            bv, be, bg = m*radius**3/6, m*radius**2/2, m*radius
        else:
            bv, be, bg = m*radius**4/24, m*radius**3/6, m*radius**2/2
        dnorm, ddnorm = np.linalg.norm(dw, axis=-1), np.linalg.norm(ddw, axis=(-2, -1))
        budgets = np.column_stack((np.sum(w*bv, axis=1),
            np.sum(w*be+dnorm*bv, axis=1), np.sum(w*bg+2*dnorm*be+ddnorm*bv, axis=1)))
        conditional = None
        if anchor_errors is not None:
            supplied = np.asarray(anchor_errors)
            if (supplied.shape != (len(self.centers), 4) or not np.all(np.isfinite(supplied))
                    or np.any(supplied < 0)):
                raise ValueError('Require explicit finite nonnegative dV,dE,dG,dH per anchor')
            da = supplied[ids]
            dv = da[:, :, 0]+radius*da[:, :, 1]+radius**2*da[:, :, 2]/2
            de = da[:, :, 1]+radius*da[:, :, 2]
            dg = da[:, :, 2].copy()
            if degree == 3:
                dv += radius**3*da[:, :, 3]/6
                de += radius**2*da[:, :, 3]/2
                dg += radius*da[:, :, 3]
            conditional = budgets+np.column_stack((np.sum(w*dv, axis=1),
                np.sum(w*de+dnorm*dv, axis=1),
                np.sum(w*dg+2*dnorm*de+ddnorm*dv, axis=1)))
        return {'V': v, 'E': e, 'G': g, 'truncation_majorant': budgets,
                'conditional_truncation_plus_supplied_anchor_budget': conditional,
                'numeric_error_budget': None, 'total_error_bound': None,
                'certified': False, 'active_limit': 8, 'minimum_denominator': float(np.min(s)),
                'weight_sum_error': float(np.max(abs(np.sum(w, axis=1)-1))),
                'weight_gradient_sum_error': float(np.max(abs(np.sum(dw, axis=1)))),
                'weight_hessian_sum_error': float(np.max(abs(np.sum(ddw, axis=1))))}
