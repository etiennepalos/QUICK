!
! (C) Copyright 2026 QUICK contributors
! All rights reserved.
! This Source Code Form is subject to the terms of the Mozilla Public
! License, v. 2.0. If a copy of the MPL was not distributed with this
! file, You can obtain one at https://mozilla.org/MPL/2.0/.
!
#include "util.fh"

! Independent Coulomb AO operators for the solvation interface.  These
! routines read the normalized Cartesian QUICK basis and geometry, but do
! not change Hcore, external charges, OEPROP buffers or screening settings.
module quick_solvation_coulomb_module
   use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
   use, intrinsic :: iso_c_binding, only: c_int64_t
   use quick_basis_module, only: quick_basis, nbasis, itype, ncontract, aexp, dcoeff
   use quick_molspec_module, only: natom, xyz
#if defined(CUDA) || defined(CUDA_MPIV)
   use quick_method_module, only: quick_method
#endif
   implicit none
   private

   double precision, parameter :: COULOMB_PI = 3.14159265358979323846264338327950288d0
   integer, parameter, public :: QUICK_SOLVATION_COULOMB_ERROR = 54
   public :: solvation_coulomb_matrix, solvation_coulomb_gradient
   public :: solvation_primitive_coulomb
   public :: solvation_coulomb_gpu_counters
   public :: clear_solvation_coulomb_cache, solvation_coulomb_cache_info
   public :: solvation_coulomb_cache_set_limit

#if defined(CUDA) || defined(CUDA_MPIV)
   interface
      subroutine gpu_solvation_coulomb_matrix(nb, na, maxprim, nc, ang, centers, expo, coeff, &
         atoms, np, points, charges, matrix, ierr)
         implicit none
         integer, intent(in) :: nb, na, maxprim, np, nc(nb), ang(3, nb), centers(nb)
         double precision, intent(in) :: expo(maxprim, nb), coeff(maxprim, nb), atoms(3, na)
         double precision, intent(in) :: points(3, np), charges(np)
         double precision, intent(out) :: matrix(nb, nb)
         integer, intent(inout) :: ierr
      end subroutine gpu_solvation_coulomb_matrix
      subroutine gpu_solvation_coulomb_gradient(nb, na, maxprim, nc, ang, centers, expo, coeff, &
         atoms, np, points, charges, density, atomic_gradient, point_gradient, ierr)
         implicit none
         integer, intent(in) :: nb, na, maxprim, np, nc(nb), ang(3, nb), centers(nb)
         double precision, intent(in) :: expo(maxprim, nb), coeff(maxprim, nb), atoms(3, na)
         double precision, intent(in) :: points(3, np), charges(np), density(nb, nb)
         double precision, intent(out) :: atomic_gradient(3, na), point_gradient(3, np)
         integer, intent(inout) :: ierr
      end subroutine gpu_solvation_coulomb_gradient
      subroutine gpu_solvation_coulomb_counters(matrix_calls, gradient_calls, reset)
         import c_int64_t
         implicit none
         integer(c_int64_t), intent(out) :: matrix_calls, gradient_calls
         integer, intent(in) :: reset
      end subroutine gpu_solvation_coulomb_counters
      subroutine gpu_solvation_coulomb_cache_info(hits, builds, bytes, gemvs, direct_calls, fallbacks, allocated_bytes)
         import c_int64_t
         implicit none
         integer(c_int64_t), intent(out) :: hits, builds, bytes, gemvs, direct_calls, fallbacks, allocated_bytes
      end subroutine gpu_solvation_coulomb_cache_info
      subroutine gpu_solvation_coulomb_cache_clear(ierr, reset_statistics)
         implicit none
         integer, intent(inout) :: ierr
         integer, intent(in) :: reset_statistics
      end subroutine gpu_solvation_coulomb_cache_clear
      subroutine gpu_solvation_coulomb_cache_set_limit(bytes, ierr)
         import c_int64_t
         implicit none
         integer(c_int64_t), intent(in) :: bytes
         integer, intent(inout) :: ierr
      end subroutine gpu_solvation_coulomb_cache_set_limit
   end interface
#endif

contains

   ! matrix_mu,nu = sum_c q_c * integral chi_mu(r)*chi_nu(r)/|r-C_c| dr.
   ! The integral is positive for s-type primitives and a positive q.
   ! No electronic-charge sign or one-half energy factor is included.
   subroutine solvation_coulomb_matrix(points, charges, matrix, ierr, message)
      implicit none
      double precision, intent(in) :: points(:, :), charges(:)
      double precision, intent(out) :: matrix(:, :)
      integer, intent(inout) :: ierr
      character(len=*), intent(out) :: message
      integer :: ibas, jbas, icon, jcon, ic, ia, ja
      double precision :: element, value, coefficient
      double precision :: center_a(3), center_b(3)

      matrix = 0.0d0
      message = ''
      if (ierr /= 0) return
      call validate_coulomb_input(points, charges, ierr, message)
      if (ierr /= 0) return
      if (size(matrix, 1) /= nbasis .or. size(matrix, 2) /= nbasis) then
         call coulomb_error(ierr, message, 'Invalid continuum Coulomb AO matrix dimension.')
         return
      endif
#if defined(CUDA) || defined(CUDA_MPIV)
      if (quick_method%bGPU) then
         call gpu_solvation_coulomb_matrix(nbasis, natom, size(aexp, 1), ncontract, itype, &
            quick_basis%ncenter, aexp, dcoeff, xyz, size(charges), points, charges, matrix, ierr)
         if (ierr /= 0) then
            message = 'CUDA continuum Coulomb matrix failed (Cartesian s/p/d/f support).'
         else if (.not. all(ieee_is_finite(matrix))) then
            call coulomb_error(ierr, message, 'Non-finite CUDA continuum Coulomb matrix.')
         endif
         return
      endif
#endif
      do ibas = 1, nbasis
         ia = quick_basis%ncenter(ibas)
         center_a = xyz(:, ia)
         do jbas = ibas, nbasis
            ja = quick_basis%ncenter(jbas)
            center_b = xyz(:, ja)
            element = 0.0d0
            do icon = 1, ncontract(ibas)
               do jcon = 1, ncontract(jbas)
                  coefficient = dcoeff(icon, ibas) * dcoeff(jcon, jbas)
                  if (coefficient == 0.0d0) cycle
                  do ic = 1, size(charges)
                     if (charges(ic) == 0.0d0) cycle
                     call solvation_primitive_coulomb(aexp(icon, ibas), aexp(jcon, jbas), &
                        itype(:, ibas), itype(:, jbas), center_a, center_b, points(:, ic), value)
                     element = element + coefficient * charges(ic) * value
                  enddo
               enddo
            enddo
            matrix(ibas, jbas) = element
            matrix(jbas, ibas) = element
         enddo
      enddo
      if (.not. all(ieee_is_finite(matrix))) &
         call coulomb_error(ierr, message, 'Non-finite continuum Coulomb AO matrix.')
   end subroutine solvation_coulomb_matrix

   ! Differentiate sum_mu,nu P_mu,nu * matrix_mu,nu, keeping P and q fixed.
   ! atomic_gradient differentiates AO centers; point_gradient differentiates
   ! the explicit source positions.  Both are +dE/dR and are separate outputs
   ! so a caller with moving cavity nodes can avoid counting that motion twice.
   subroutine solvation_coulomb_gradient(points, charges, density, atomic_gradient, point_gradient, ierr, message)
      implicit none
      double precision, intent(in) :: points(:, :), charges(:), density(:, :)
      double precision, intent(out) :: atomic_gradient(:, :), point_gradient(:, :)
      integer, intent(inout) :: ierr
      character(len=*), intent(out) :: message
      integer :: ibas, jbas, icon, jcon, ic, ia, ja
      double precision :: value, coefficient, weight
      double precision :: center_a(3), center_b(3), derivative_a(3), derivative_b(3)

      atomic_gradient = 0.0d0
      point_gradient = 0.0d0
      message = ''
      if (ierr /= 0) return
      call validate_coulomb_input(points, charges, ierr, message)
      if (ierr /= 0) return
      if (size(density, 1) /= nbasis .or. size(density, 2) /= nbasis .or. &
         size(atomic_gradient, 1) /= 3 .or. size(atomic_gradient, 2) /= natom .or. &
         size(point_gradient, 1) /= 3 .or. size(point_gradient, 2) /= size(charges)) then
         call coulomb_error(ierr, message, 'Invalid continuum Coulomb density or gradient dimension.')
         return
      endif
      if (.not. all(ieee_is_finite(density))) then
         call coulomb_error(ierr, message, 'Non-finite continuum Coulomb density.')
         return
      endif
#if defined(CUDA) || defined(CUDA_MPIV)
      if (quick_method%bGPU) then
         call gpu_solvation_coulomb_gradient(nbasis, natom, size(aexp, 1), ncontract, itype, &
            quick_basis%ncenter, aexp, dcoeff, xyz, size(charges), points, charges, density, &
            atomic_gradient, point_gradient, ierr)
         if (ierr /= 0) then
            message = 'CUDA continuum Coulomb gradient failed (Cartesian s/p/d/f support).'
         else if (.not. all(ieee_is_finite(atomic_gradient)) .or. &
            .not. all(ieee_is_finite(point_gradient))) then
            call coulomb_error(ierr, message, 'Non-finite CUDA continuum Coulomb gradients.')
         endif
         return
      endif
#endif
      do ibas = 1, nbasis
         ia = quick_basis%ncenter(ibas)
         center_a = xyz(:, ia)
         do jbas = ibas, nbasis
            ja = quick_basis%ncenter(jbas)
            center_b = xyz(:, ja)
            weight = density(ibas, jbas)
            if (ibas /= jbas) weight = weight + density(jbas, ibas)
            if (weight == 0.0d0) cycle
            do icon = 1, ncontract(ibas)
               do jcon = 1, ncontract(jbas)
                  coefficient = weight * dcoeff(icon, ibas) * dcoeff(jcon, jbas)
                  if (coefficient == 0.0d0) cycle
                  do ic = 1, size(charges)
                     if (charges(ic) == 0.0d0) cycle
                     call solvation_primitive_coulomb(aexp(icon, ibas), aexp(jcon, jbas), &
                        itype(:, ibas), itype(:, jbas), center_a, center_b, points(:, ic), &
                        value, derivative_a, derivative_b)
                     atomic_gradient(:, ia) = atomic_gradient(:, ia) + coefficient * charges(ic) * derivative_a
                     atomic_gradient(:, ja) = atomic_gradient(:, ja) + coefficient * charges(ic) * derivative_b
                     ! Translational invariance is exact for every primitive.
                     point_gradient(:, ic) = point_gradient(:, ic) &
                        - coefficient * charges(ic) * (derivative_a + derivative_b)
                  enddo
               enddo
            enddo
         enddo
      enddo
      if (.not. all(ieee_is_finite(atomic_gradient)) .or. .not. all(ieee_is_finite(point_gradient))) &
         call coulomb_error(ierr, message, 'Non-finite continuum Coulomb gradients.')
   end subroutine solvation_coulomb_gradient

   ! Actual GPU matrix evaluations and direct gradient kernel launches.
   ! Cached matrix evaluations launch GEMV and symmetric expansion instead
   ! of the direct integral kernel; cache_info distinguishes those paths.
   ! CPU builds return zero; reset clears these counters after returning them.
   subroutine solvation_coulomb_gpu_counters(matrix_calls, gradient_calls, reset)
      implicit none
      integer(c_int64_t), intent(out) :: matrix_calls, gradient_calls
      logical, intent(in), optional :: reset
#if defined(CUDA) || defined(CUDA_MPIV)
      integer :: reset_flag

      reset_flag = 0
      if (present(reset)) then
         if (reset) reset_flag = 1
      endif
      call gpu_solvation_coulomb_counters(matrix_calls, gradient_calls, reset_flag)
#else
      matrix_calls = 0_c_int64_t
      gradient_calls = 0_c_int64_t
#endif
   end subroutine solvation_coulomb_gpu_counters

   ! Release the optional geometry cache before QUICK tears down its GPU.
   ! Empty cleanup performs no CUDA queries or initialization. Geometry
   ! changes pass reset_statistics=.false. to retain whole-job counters.
   subroutine clear_solvation_coulomb_cache(ierr, reset_statistics)
      implicit none
      integer, intent(inout) :: ierr
      logical, intent(in), optional :: reset_statistics
#if defined(CUDA) || defined(CUDA_MPIV)
      integer :: reset_flag

      reset_flag = 1
      if (present(reset_statistics)) then
         if (.not. reset_statistics) reset_flag = 0
      endif
      call gpu_solvation_coulomb_cache_clear(ierr, reset_flag)
#endif
   end subroutine clear_solvation_coulomb_cache

   ! bytes is the resident positive T matrix; allocated_bytes optionally
   ! includes its charge, packed-result and full-matrix workspaces. A
   ! zero-byte cache and positive direct_calls establish direct CUDA mode.
   subroutine solvation_coulomb_cache_info(hits, builds, bytes, gemvs, direct_calls, fallbacks, allocated_bytes)
      implicit none
      integer(c_int64_t), intent(out) :: hits, builds, bytes, gemvs, direct_calls, fallbacks
      integer(c_int64_t), intent(out), optional :: allocated_bytes
      integer(c_int64_t) :: total_bytes

#if defined(CUDA) || defined(CUDA_MPIV)
      call gpu_solvation_coulomb_cache_info(hits, builds, bytes, gemvs, direct_calls, fallbacks, total_bytes)
#else
      hits = 0_c_int64_t
      builds = 0_c_int64_t
      bytes = 0_c_int64_t
      gemvs = 0_c_int64_t
      direct_calls = 0_c_int64_t
      fallbacks = 0_c_int64_t
      total_bytes = 0_c_int64_t
#endif
      if (present(allocated_bytes)) allocated_bytes = total_bytes
   end subroutine solvation_coulomb_cache_info

   ! Runtime/validation control only: 0 selects direct CUDA, positive limits
   ! are capped at 8 GiB and one quarter of free device memory. Changing the
   ! limit invalidates resident T but preserves statistics. No CPU fallback.
   subroutine solvation_coulomb_cache_set_limit(bytes, ierr)
      implicit none
      integer(c_int64_t), intent(in) :: bytes
      integer, intent(inout) :: ierr

      if (ierr /= 0) return
      if (bytes < 0_c_int64_t) then
         ierr = QUICK_SOLVATION_COULOMB_ERROR
         return
      endif
#if defined(CUDA) || defined(CUDA_MPIV)
      call gpu_solvation_coulomb_cache_set_limit(bytes, ierr)
#endif
   end subroutine solvation_coulomb_cache_set_limit

   ! Unnormalized primitive Cartesian attraction integral.  Original
   ! contraction/normalization coefficients must be retained when using its
   ! center derivatives; the raised/lowered polynomial is not renormalized.
   subroutine solvation_primitive_coulomb(alpha, beta, angular_a, angular_b, center_a, center_b, &
      point, value, derivative_a, derivative_b)
      implicit none
      double precision, intent(in) :: alpha, beta, center_a(3), center_b(3), point(3)
      integer, intent(in) :: angular_a(3), angular_b(3)
      double precision, intent(out) :: value
      double precision, intent(out), optional :: derivative_a(3), derivative_b(3)
      integer :: degree, n, tx, ty, tz, axis, raised_a(3), raised_b(3), lowered(3)
      double precision :: exponent, reduced, prefactor, factor, parameter_t, product(3), pc(3)
      double precision, allocatable :: ecoeff(:, :, :, :), r(:, :, :, :), boys(:)

      degree = sum(angular_a) + sum(angular_b)
      if (present(derivative_a) .or. present(derivative_b)) degree = degree + 1
      allocate(ecoeff(0:degree, 0:maxval(angular_a) + 1, 0:maxval(angular_b) + 1, 3), &
         r(0:degree, 0:degree, 0:degree, 0:degree), boys(0:degree))
      exponent = alpha + beta
      reduced = alpha * beta / exponent
      product = (alpha * center_a + beta * center_b) / exponent
      prefactor = 2.0d0 * COULOMB_PI / exponent * exp(-reduced * sum((center_a - center_b)**2))
      value = 0.0d0
      if (present(derivative_a)) derivative_a = 0.0d0
      if (present(derivative_b)) derivative_b = 0.0d0
      if (prefactor == 0.0d0) return
      do axis = 1, 3
         call hermite_coefficients(angular_a(axis) + 1, angular_b(axis) + 1, degree, &
            exponent, product(axis) - center_a(axis), product(axis) - center_b(axis), &
            ecoeff(:, :, :, axis))
      enddo
      pc = product - point
      parameter_t = exponent * sum(pc**2)
      call boys_values(degree, parameter_t, boys)
      r = 0.0d0
      factor = 1.0d0
      do n = 0, degree
         r(0, 0, 0, n) = factor * boys(n)
         factor = factor * (-2.0d0 * exponent)
      enddo
      do n = degree - 1, 0, -1
         do tx = 0, degree - n
            do ty = 0, degree - n - tx
               do tz = 0, degree - n - tx - ty
                  if (tx > 0) then
                     r(tx, ty, tz, n) = pc(1) * r(tx - 1, ty, tz, n + 1)
                     if (tx > 1) r(tx, ty, tz, n) = r(tx, ty, tz, n) &
                        + dble(tx - 1) * r(tx - 2, ty, tz, n + 1)
                  else if (ty > 0) then
                     r(tx, ty, tz, n) = pc(2) * r(tx, ty - 1, tz, n + 1)
                     if (ty > 1) r(tx, ty, tz, n) = r(tx, ty, tz, n) &
                        + dble(ty - 1) * r(tx, ty - 2, tz, n + 1)
                  else if (tz > 0) then
                     r(tx, ty, tz, n) = pc(3) * r(tx, ty, tz - 1, n + 1)
                     if (tz > 1) r(tx, ty, tz, n) = r(tx, ty, tz, n) &
                        + dble(tz - 1) * r(tx, ty, tz - 2, n + 1)
                  endif
               enddo
            enddo
         enddo
      enddo
      value = prefactor * hermite_contraction(angular_a, angular_b, ecoeff, r)
      do axis = 1, 3
         if (present(derivative_a)) then
            raised_a = angular_a
            raised_a(axis) = raised_a(axis) + 1
            derivative_a(axis) = 2.0d0 * alpha * hermite_contraction(raised_a, angular_b, ecoeff, r)
            if (angular_a(axis) > 0) then
               lowered = angular_a
               lowered(axis) = lowered(axis) - 1
               derivative_a(axis) = derivative_a(axis) &
                  - dble(angular_a(axis)) * hermite_contraction(lowered, angular_b, ecoeff, r)
            endif
            derivative_a(axis) = prefactor * derivative_a(axis)
         endif
         if (present(derivative_b)) then
            raised_b = angular_b
            raised_b(axis) = raised_b(axis) + 1
            derivative_b(axis) = 2.0d0 * beta * hermite_contraction(angular_a, raised_b, ecoeff, r)
            if (angular_b(axis) > 0) then
               lowered = angular_b
               lowered(axis) = lowered(axis) - 1
               derivative_b(axis) = derivative_b(axis) &
                  - dble(angular_b(axis)) * hermite_contraction(angular_a, lowered, ecoeff, r)
            endif
            derivative_b(axis) = prefactor * derivative_b(axis)
         endif
      enddo
   end subroutine solvation_primitive_coulomb

   subroutine hermite_coefficients(la, lb, degree, exponent, pa, pb, coeff)
      implicit none
      integer, intent(in) :: la, lb, degree
      double precision, intent(in) :: exponent, pa, pb
      double precision, intent(out) :: coeff(0:, 0:, 0:)
      integer :: i, j, t
      double precision :: lower, higher

      coeff = 0.0d0
      coeff(0, 0, 0) = 1.0d0
      do j = 0, lb
         do i = 0, la
            if (i + j == 0 .or. i + j > degree) cycle
            do t = 0, i + j
               lower = 0.0d0
               higher = 0.0d0
               if (i > 0) then
                  if (t > 0) lower = coeff(t - 1, i - 1, j) / (2.0d0 * exponent)
                  if (t < degree) higher = dble(t + 1) * coeff(t + 1, i - 1, j)
                  coeff(t, i, j) = lower + pa * coeff(t, i - 1, j) + higher
               else
                  if (t > 0) lower = coeff(t - 1, i, j - 1) / (2.0d0 * exponent)
                  if (t < degree) higher = dble(t + 1) * coeff(t + 1, i, j - 1)
                  coeff(t, i, j) = lower + pb * coeff(t, i, j - 1) + higher
               endif
            enddo
         enddo
      enddo
   end subroutine hermite_coefficients

   function hermite_contraction(angular_a, angular_b, coeff, r) result(value)
      implicit none
      integer, intent(in) :: angular_a(3), angular_b(3)
      double precision, intent(in) :: coeff(0:, 0:, 0:, :), r(0:, 0:, 0:, 0:)
      double precision :: value, ex, exy
      integer :: tx, ty, tz

      value = 0.0d0
      do tx = 0, angular_a(1) + angular_b(1)
         ex = coeff(tx, angular_a(1), angular_b(1), 1)
         do ty = 0, angular_a(2) + angular_b(2)
            exy = ex * coeff(ty, angular_a(2), angular_b(2), 2)
            do tz = 0, angular_a(3) + angular_b(3)
               value = value + exy * coeff(tz, angular_a(3), angular_b(3), 3) * r(tx, ty, tz, 0)
            enddo
         enddo
      enddo
   end function hermite_contraction

   ! A positive-term series avoids cancellation at small/intermediate T.
   ! Its highest-order value seeds a stable downward recurrence.  Above
   ! degree+20, erf(T**0.5) and upward recurrence are well-conditioned.
   subroutine boys_values(degree, parameter_t, values)
      implicit none
      integer, intent(in) :: degree
      double precision, intent(in) :: parameter_t
      double precision, intent(out) :: values(0:degree)
      double precision :: term, total, damping
      integer :: k, n

      damping = exp(-parameter_t)
      if (parameter_t < max(30.0d0, dble(degree) + 20.0d0)) then
         term = 1.0d0 / dble(2 * degree + 1)
         total = term
         do k = 1, 1000
            term = term * (2.0d0 * parameter_t) / dble(2 * degree + 2 * k + 1)
            total = total + term
            if (term <= epsilon(total) * total) exit
         enddo
         values(degree) = damping * total
         do n = degree - 1, 0, -1
            values(n) = (2.0d0 * parameter_t * values(n + 1) + damping) / dble(2 * n + 1)
         enddo
      else
         values(0) = 0.5d0 * sqrt(COULOMB_PI / parameter_t) * erf(sqrt(parameter_t))
         do n = 1, degree
            values(n) = (dble(2 * n - 1) * values(n - 1) - damping) / (2.0d0 * parameter_t)
         enddo
      endif
   end subroutine boys_values

   subroutine validate_coulomb_input(points, charges, ierr, message)
      implicit none
      double precision, intent(in) :: points(:, :), charges(:)
      integer, intent(inout) :: ierr
      character(len=*), intent(out) :: message

      message = ''
      if (size(points, 1) /= 3 .or. size(points, 2) /= size(charges)) then
         call coulomb_error(ierr, message, 'Invalid continuum Coulomb point dimensions.')
         return
      endif
      if (.not. allocated(xyz) .or. .not. allocated(itype) .or. .not. allocated(ncontract) .or. &
         .not. allocated(aexp) .or. .not. allocated(dcoeff) .or. .not. allocated(quick_basis%ncenter)) then
         call coulomb_error(ierr, message, 'The QUICK basis is not initialized for continuum Coulomb integrals.')
         return
      endif
      if (nbasis < 1 .or. natom < 1) then
         call coulomb_error(ierr, message, 'The QUICK basis or geometry is empty.')
         return
      endif
      if (.not. all(ieee_is_finite(points)) .or. .not. all(ieee_is_finite(charges))) &
         call coulomb_error(ierr, message, 'Non-finite continuum Coulomb point data.')
   end subroutine validate_coulomb_input

   subroutine coulomb_error(ierr, message, detail)
      implicit none
      integer, intent(inout) :: ierr
      character(len=*), intent(out) :: message
      character(len=*), intent(in) :: detail

      ierr = QUICK_SOLVATION_COULOMB_ERROR
      message = detail
   end subroutine coulomb_error

end module quick_solvation_coulomb_module
