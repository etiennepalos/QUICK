!
! (C) Copyright 2026 QUICK contributors
! All rights reserved.
! This Source Code Form is subject to the terms of the Mozilla Public
! License, v. 2.0. If a copy of the MPL was not distributed with this
! file, You can obtain one at https://mozilla.org/MPL/2.0/.
!
#include "util.fh"

! Native ddX ownership and solver interface.  The host supplies the actual
! electronic density projection, not fitted atomic charges.  All positions
! and radii are in bohr; electrostatic quantities and energies are in a.u.
module quick_ddx_backend_module
   use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
#ifdef QUICK_DDX
   use omp_lib, only: omp_get_max_threads, omp_set_num_threads
   use ddx, only: ddinit, ddx_type, ddx_state_type, ddx_error_type, &
      ddx_electrostatics_type, allocate_state, setup, fill_guess, &
      fill_guess_adjoint, solve, solve_adjoint, ddx_energy => energy, &
      solvation_force_terms, reset_error, ddproject_cav, dbasis
   use ddx_multipolar_solutes, only: multipole_psi
   use ddx_harmonics, only: ylmbas
#endif
   implicit none
   private

   integer, parameter, public :: QUICK_DDX_ERROR = 53
#ifdef QUICK_DDX
   logical, parameter, public :: quick_ddx_available = .true.
#else
   logical, parameter, public :: quick_ddx_available = .false.
#endif
   double precision, parameter :: DDX_PI = 3.14159265358979323846264338327950288d0

   type, public :: quick_ddx_backend_type
      integer :: model = 0, nsph = 0, nlm = 0, ncav = 0, lmax = 0
      double precision :: scaling = 1.0d0, tolerance = 1.0d-10
      logical :: initialized = .false.
      double precision, allocatable :: cavity(:, :), nuclear_psi(:, :)
      ! One-based sphere index for each cavity node, in native ddX ordering.
      integer, allocatable :: cavity_sphere(:)
#ifdef QUICK_DDX
      type(ddx_type), allocatable, private :: data
      type(ddx_state_type), allocatable, private :: state
      type(ddx_error_type), private :: error
#endif
   end type quick_ddx_backend_type

   public :: ddx_backend_initialize, ddx_backend_finalize
   public :: ddx_backend_scaled_ylm, ddx_backend_solve
   public :: ddx_backend_cavity_gradient

contains

   subroutine ddx_backend_initialize(ctx, model, natom, xyz, radii, charges, &
      epsilon, lmax, nlebedev, tolerance, ierr, message)
      implicit none
      type(quick_ddx_backend_type), intent(inout) :: ctx
      integer, intent(in) :: model, natom, lmax, nlebedev
      double precision, intent(in) :: xyz(3, natom), radii(natom), charges(natom)
      double precision, intent(in) :: epsilon, tolerance
      integer, intent(inout) :: ierr
      character(len=*), intent(out) :: message
#ifdef QUICK_DDX
      double precision :: multipoles(1, natom)
      integer :: isph, first, last, status, host_threads
#endif

      message = ''
      if (ierr /= 0) return
      call ddx_backend_finalize(ctx)
      if (.not. quick_ddx_available) then
         call backend_error(ierr, message, 'QUICK was built without QUICK_DDX support.')
         return
      endif
      if (model /= 1 .and. model /= 2) then
         call backend_error(ierr, message, 'Only ddCOSMO (1) and ddPCM (2) are supported.')
         return
      endif
      if (natom < 1 .or. lmax < 0 .or. lmax > 30 .or. nlebedev < 1) then
         call backend_error(ierr, message, 'Invalid ddX cavity or harmonic dimensions.')
         return
      endif
      if (.not. all(ieee_is_finite(xyz)) .or. .not. all(ieee_is_finite(radii)) .or. &
         .not. all(ieee_is_finite(charges)) .or. .not. ieee_is_finite(epsilon) .or. &
         .not. ieee_is_finite(tolerance)) then
         call backend_error(ierr, message, 'Non-finite ddX model input.')
         return
      endif
      if (any(radii <= 0.0d0) .or. epsilon <= 1.0d0 .or. tolerance <= 0.0d0 .or. tolerance >= 1.0d0) then
         call backend_error(ierr, message, 'ddX needs positive radii, epsilon > 1 and 0 < tolerance < 1.')
         return
      endif
#ifdef QUICK_DDX
      allocate(ctx%data, ctx%state, stat=status)
      if (status /= 0) then
         call backend_error(ierr, message, 'Cannot allocate ddX model and state.')
         call ddx_backend_finalize(ctx)
         return
      endif
      call reset_error(ctx%error)
      ! Pass every option with a ddinit SAVE-initialized local explicitly.
      ! This prevents previous host jobs from changing the defaults here.
      ! Forces allocate the native derivative intermediates also used by the
      ! adjoint solve.  The host calls the library on its master rank only.
      host_threads = omp_get_max_threads()
      call ddinit(model, natom, xyz, radii, epsilon, ctx%data, ctx%error, &
         force=1, kappa=0.0d0, eta=0.1d0, shift=merge(-1.0d0, 0.0d0, model == 1), &
         lmax=lmax, ngrid=nlebedev, incore=0, maxiter=200, jacobi_ndiis=20, &
         enable_fmm=1, pm=max(8, lmax), pl=max(8, lmax), nproc=1, &
         logfile=repeat(' ', 255), switching=0)
      call omp_set_num_threads(host_threads)
      call collect_ddx_error(ctx, ierr, message)
      if (ierr /= 0) then
         call ddx_backend_finalize(ctx)
         return
      endif
      call allocate_state(ctx%data%params, ctx%data%constants, ctx%state, ctx%error)
      call collect_ddx_error(ctx, ierr, message)
      if (ierr /= 0) then
         call ddx_backend_finalize(ctx)
         return
      endif
      ctx%model = model
      ctx%nsph = natom
      ctx%nlm = ctx%data%constants%nbasis
      ctx%ncav = ctx%data%constants%ncav
      ctx%lmax = lmax
      ctx%tolerance = tolerance
      ctx%scaling = 1.0d0
      ! Use the conventional C-PCM/COSMO finite-dielectric scaling used by
      ! the Psi4/ddX QM interface.  PCM already contains epsilon in its solve.
      if (model == 1) ctx%scaling = (epsilon - 1.0d0) / epsilon
      allocate(ctx%cavity(3, ctx%ncav), ctx%cavity_sphere(ctx%ncav), &
         ctx%nuclear_psi(ctx%nlm, natom), stat=status)
      if (status /= 0) then
         call backend_error(ierr, message, 'Cannot allocate ddX host interface arrays.')
         call ddx_backend_finalize(ctx)
         return
      endif
      ctx%cavity = ctx%data%constants%ccav
      do isph = 1, natom
         first = ctx%data%constants%icav_ia(isph)
         last = ctx%data%constants%icav_ia(isph + 1) - 1
         ctx%cavity_sphere(first:last) = isph
      enddo
      multipoles(1, :) = charges / sqrt(4.0d0 * DDX_PI)
      call multipole_psi(ctx%data%params, multipoles, 0, ctx%nuclear_psi)
      ctx%initialized = .true.
#endif
   end subroutine ddx_backend_initialize

   subroutine ddx_backend_finalize(ctx)
      implicit none
      type(quick_ddx_backend_type), intent(inout) :: ctx

      ! Native ddX types own allocatable components, so derived-type
      ! deallocation also handles partially initialized models safely.
#ifdef QUICK_DDX
      if (allocated(ctx%state)) deallocate(ctx%state)
      if (allocated(ctx%data)) deallocate(ctx%data)
      call reset_error(ctx%error)
#endif
      if (allocated(ctx%cavity)) deallocate(ctx%cavity)
      if (allocated(ctx%cavity_sphere)) deallocate(ctx%cavity_sphere)
      if (allocated(ctx%nuclear_psi)) deallocate(ctx%nuclear_psi)
      ctx%model = 0
      ctx%nsph = 0
      ctx%nlm = 0
      ctx%ncav = 0
      ctx%lmax = 0
      ctx%scaling = 1.0d0
      ctx%tolerance = 1.0d-10
      ctx%initialized = .false.
   end subroutine ddx_backend_finalize

   ! B_lm^i(r) = [4*pi/(2*l+1)] (|r-R_i|/a_i)^l Y_lm(r-R_i).
   ! The host supplies its atom partition in the density integration; it
   ! is not included in B.  grad_ylm differentiates r with R_i held fixed.
   subroutine ddx_backend_scaled_ylm(ctx, point, sphere, ylm, ierr, message, grad_ylm)
      implicit none
      type(quick_ddx_backend_type), intent(in) :: ctx
      double precision, intent(in) :: point(3)
      integer, intent(in) :: sphere
      double precision, intent(out) :: ylm(:)
      integer, intent(inout) :: ierr
      character(len=*), intent(out) :: message
      double precision, intent(out), optional :: grad_ylm(:, :)
#ifdef QUICK_DDX
      double precision :: delta(3), direction(3), radius, ratio, rho, ctheta, stheta, cphi, sphi
      double precision :: angular(ctx%nlm), dangular(3, ctx%nlm), vplm(ctx%nlm)
      double precision :: vcos(ctx%lmax + 1), vsin(ctx%lmax + 1)
      integer :: l, m, ind
#endif

      ylm = 0.0d0
      if (present(grad_ylm)) grad_ylm = 0.0d0
      message = ''
      if (ierr /= 0) return
      if (.not. ctx%initialized) then
         call backend_error(ierr, message, 'The ddX model is not initialized.')
         return
      endif
      if (sphere < 1 .or. sphere > ctx%nsph .or. size(ylm) /= ctx%nlm) then
         call backend_error(ierr, message, 'Invalid ddX sphere or harmonic output dimension.')
         return
      endif
      if (present(grad_ylm)) then
         if (size(grad_ylm, 1) /= 3 .or. size(grad_ylm, 2) /= ctx%nlm) then
            call backend_error(ierr, message, 'Invalid ddX harmonic gradient dimension.')
            return
         endif
      endif
      if (.not. all(ieee_is_finite(point))) then
         call backend_error(ierr, message, 'Non-finite ddX harmonic point.')
         return
      endif
#ifdef QUICK_DDX
      delta = point - ctx%data%params%csph(:, sphere)
      radius = norm2(delta)
      if (radius == 0.0d0) then
         ! The solid harmonics are regular at the center, including their
         ! gradients.  Native angular-only evaluation is undefined there.
         ylm(1) = sqrt(4.0d0 * DDX_PI)
         if (present(grad_ylm) .and. ctx%lmax >= 1) then
            ratio = sqrt(4.0d0 * DDX_PI / 3.0d0) / ctx%data%params%rsph(sphere)
            grad_ylm(2, 2) = ratio
            grad_ylm(3, 3) = ratio
            grad_ylm(1, 4) = ratio
         endif
         return
      endif
      direction = delta / radius
      if (present(grad_ylm)) then
         call dbasis(ctx%data%params, ctx%data%constants, direction, &
            angular, dangular, vplm, vcos, vsin)
      else
         call ylmbas(delta, rho, ctheta, stheta, cphi, sphi, ctx%lmax, &
            ctx%data%constants%vscales(1:ctx%nlm), angular, vplm, vcos, vsin)
      endif
      do l = 0, ctx%lmax
         ratio = ctx%data%constants%v4pi2lp1(l + 1) * &
            (radius / ctx%data%params%rsph(sphere))**l
         do m = -l, l
            ind = l * (l + 1) + m + 1
            ylm(ind) = ratio * angular(ind)
            if (present(grad_ylm)) grad_ylm(:, ind) = ratio / radius * &
               (dangular(:, ind) + dble(l) * angular(ind) * direction)
         enddo
      enddo
      if (.not. all(ieee_is_finite(ylm))) &
         call backend_error(ierr, message, 'Non-finite ddX solid harmonics.')
      if (present(grad_ylm)) then
         if (.not. all(ieee_is_finite(grad_ylm))) &
            call backend_error(ierr, message, 'Non-finite ddX solid harmonic gradients.')
      endif
#endif
   end subroutine ddx_backend_scaled_ylm

   ! total_phi contains electronic AND nuclear potential at ctx%cavity.
   ! electronic_psi contains negative electronic charge only; this wrapper
   ! adds nuclear_psi exactly once.  Outputs x and xi are unscaled ddX
   ! coefficients.  With positive T(C)=integral chi_mu*chi_nu/|r-C|,
   ! the AO operator is -0.5*scaling*integral chi_mu*chi_nu*B*x
   ! +0.5*scaling*sum_c xi_c*T(C_c).  B includes the host atom partition.
   subroutine ddx_backend_solve(ctx, total_phi, electronic_psi, energy, x, xi, ierr, message)
      implicit none
      type(quick_ddx_backend_type), intent(inout) :: ctx
      double precision, intent(in) :: total_phi(:), electronic_psi(:, :)
      double precision, intent(out) :: energy, x(:, :), xi(:)
      integer, intent(inout) :: ierr
      character(len=*), intent(out) :: message
#ifdef QUICK_DDX
      type(ddx_electrostatics_type) :: electrostatics
      double precision :: psi(ctx%nlm, ctx%nsph), unscaled_energy
      logical :: have_primal, have_adjoint
      integer :: host_threads
#endif

      energy = 0.0d0
      x = 0.0d0
      xi = 0.0d0
      message = ''
      if (ierr /= 0) return
      if (.not. ctx%initialized) then
         call backend_error(ierr, message, 'The ddX model is not initialized.')
         return
      endif
      if (size(total_phi) /= ctx%ncav .or. size(electronic_psi, 1) /= ctx%nlm .or. &
         size(electronic_psi, 2) /= ctx%nsph .or. size(x, 1) /= ctx%nlm .or. &
         size(x, 2) /= ctx%nsph .or. size(xi) /= ctx%ncav) then
         call backend_error(ierr, message, 'Invalid ddX electrostatics or reaction coefficient dimension.')
         return
      endif
      if (.not. all(ieee_is_finite(total_phi)) .or. .not. all(ieee_is_finite(electronic_psi))) then
         call backend_error(ierr, message, 'Non-finite ddX electronic input.')
         return
      endif
#ifdef QUICK_DDX
      call reset_error(ctx%error)
      have_primal = ctx%state%solved
      have_adjoint = ctx%state%adjoint_solved
      psi = electronic_psi + ctx%nuclear_psi
      electrostatics%phi_cav = total_phi
      electrostatics%do_phi = .true.
      host_threads = omp_get_max_threads()
      call omp_set_num_threads(1)
      call setup(ctx%data%params, ctx%data%constants, ctx%data%workspace, &
         ctx%state, electrostatics, psi, ctx%error)
      if (.not. have_primal) call fill_guess(ctx%data%params, ctx%data%constants, &
         ctx%data%workspace, ctx%state, ctx%tolerance, ctx%error)
      if (.not. have_adjoint) call fill_guess_adjoint(ctx%data%params, ctx%data%constants, &
         ctx%data%workspace, ctx%state, ctx%tolerance, ctx%error)
      call collect_ddx_error(ctx, ierr, message)
      if (ierr /= 0) then
         call omp_set_num_threads(host_threads)
         return
      endif
      call solve(ctx%data%params, ctx%data%constants, ctx%data%workspace, &
         ctx%state, ctx%tolerance, ctx%error)
      call collect_ddx_error(ctx, ierr, message)
      if (ierr /= 0) then
         call omp_set_num_threads(host_threads)
         return
      endif
      call solve_adjoint(ctx%data%params, ctx%data%constants, ctx%data%workspace, &
         ctx%state, ctx%tolerance, ctx%error)
      call collect_ddx_error(ctx, ierr, message)
      if (ierr /= 0) then
         call omp_set_num_threads(host_threads)
         return
      endif
      call ddx_energy(ctx%data%params, ctx%data%constants, ctx%data%workspace, &
         ctx%state, unscaled_energy, ctx%error)
      call omp_set_num_threads(host_threads)
      call collect_ddx_error(ctx, ierr, message)
      if (ierr /= 0) return
      energy = ctx%scaling * unscaled_energy
      x = ctx%state%xs
      ! q=s for COSMO; PCM q=s-4*pi*y/(epsilon-1).  Projecting s for PCM
      ! would produce an incorrect AO response even though its energy works.
      call ddproject_cav(ctx%data%params, ctx%data%constants, ctx%state%q, xi)
      if (.not. ieee_is_finite(energy) .or. .not. all(ieee_is_finite(x)) .or. &
         .not. all(ieee_is_finite(xi))) then
         call backend_error(ierr, message, 'Non-finite ddX energy or reaction coefficients.')
      endif
#endif
   end subroutine ddx_backend_solve

   ! This is ONLY the geometrical/non-specific term in the ddX gradient
   ! decomposition.  Host electronic-basis, partition, nuclear-charge and
   ! RHS-specific derivative terms must be added before using a total gradient.
   ! ddX calls its output "force", but tests/force.f90 compares it directly
   ! with [E(R+h)-E(R-h)]/(2*h).  Return +dE/dR, already dielectric-scaled.
   ! total_field is the physical electric field E=-grad(phi).
   subroutine ddx_backend_cavity_gradient(ctx, total_field, forces, ierr, message)
      implicit none
      type(quick_ddx_backend_type), intent(inout) :: ctx
      double precision, intent(in) :: total_field(:, :)
      double precision, intent(out) :: forces(:, :)
      integer, intent(inout) :: ierr
      character(len=*), intent(out) :: message
#ifdef QUICK_DDX
      type(ddx_electrostatics_type) :: electrostatics
      integer :: host_threads
#endif

      forces = 0.0d0
      message = ''
      if (ierr /= 0) return
      if (.not. ctx%initialized) then
         call backend_error(ierr, message, 'The ddX model is not initialized.')
         return
      endif
      if (size(total_field, 1) /= 3 .or. size(total_field, 2) /= ctx%ncav .or. &
         size(forces, 1) /= 3 .or. size(forces, 2) /= ctx%nsph) then
         call backend_error(ierr, message, 'Invalid ddX field or force dimensions.')
         return
      endif
      if (.not. all(ieee_is_finite(total_field))) then
         call backend_error(ierr, message, 'Non-finite ddX cavity field.')
         return
      endif
#ifdef QUICK_DDX
      if (.not. ctx%state%solved .or. .not. ctx%state%adjoint_solved) then
         call backend_error(ierr, message, 'ddX forces require converged primal and adjoint solutions.')
         return
      endif
      call reset_error(ctx%error)
      electrostatics%e_cav = total_field
      electrostatics%do_e = .true.
      host_threads = omp_get_max_threads()
      call omp_set_num_threads(1)
      call solvation_force_terms(ctx%data%params, ctx%data%constants, ctx%data%workspace, &
         ctx%state, electrostatics, forces, ctx%error)
      call omp_set_num_threads(host_threads)
      call collect_ddx_error(ctx, ierr, message)
      if (ierr /= 0) return
      forces = ctx%scaling * forces
      if (.not. all(ieee_is_finite(forces))) &
         call backend_error(ierr, message, 'Non-finite ddX cavity forces.')
#endif
   end subroutine ddx_backend_cavity_gradient

   subroutine backend_error(ierr, message, detail)
      implicit none
      integer, intent(inout) :: ierr
      character(len=*), intent(out) :: message
      character(len=*), intent(in) :: detail

      ierr = QUICK_DDX_ERROR
      message = detail
   end subroutine backend_error

#ifdef QUICK_DDX
   subroutine collect_ddx_error(ctx, ierr, message)
      implicit none
      type(quick_ddx_backend_type), intent(in) :: ctx
      integer, intent(inout) :: ierr
      character(len=*), intent(out) :: message

      message = ''
      if (ctx%error%flag /= 0) call backend_error(ierr, message, trim(ctx%error%message))
   end subroutine collect_ddx_error
#endif

end module quick_ddx_backend_module
