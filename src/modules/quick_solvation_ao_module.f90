! (C) Copyright 2026 QUICK contributors
! All rights reserved.
!
! This Source Code Form is subject to the terms of the Mozilla Public
! License, v. 2.0. If a copy of the MPL was not distributed with this
! file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "util.fh"

module quick_solvation_ao_module
   use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
   use quick_basis_module, only: nbasis, ncontract, itype, aexp, dcoeff, quick_basis
   use quick_molspec_module, only: natom, xyz
#if defined(CUDA) || defined(CUDA_MPIV)
   use quick_method_module, only: quick_method
#endif
   implicit none
   private

   integer, parameter :: ao_block_size = 128
   integer, parameter :: ao_error = 53
   public :: solvation_density, solvation_local_matrix, solvation_local_gradient

#if defined(CUDA) || defined(CUDA_MPIV)
   interface
      subroutine gpu_solvation_density(nb, na, maxprim, nc, ang, centers, expo, coeff, &
         atoms, np, points, density, rho, grad_rho, ierr)
         implicit none
         integer, intent(in) :: nb, na, maxprim, np, nc(nb), ang(3, nb), centers(nb)
         double precision, intent(in) :: expo(maxprim, nb), coeff(maxprim, nb), atoms(3, na)
         double precision, intent(in) :: points(3, np), density(nb, nb)
         double precision, intent(out) :: rho(np), grad_rho(3, np)
         integer, intent(out) :: ierr
      end subroutine gpu_solvation_density
      subroutine gpu_solvation_local_matrix(nb, na, maxprim, nc, ang, centers, expo, coeff, &
         atoms, np, points, eta, matrix, ierr)
         implicit none
         integer, intent(in) :: nb, na, maxprim, np, nc(nb), ang(3, nb), centers(nb)
         double precision, intent(in) :: expo(maxprim, nb), coeff(maxprim, nb), atoms(3, na)
         double precision, intent(in) :: points(3, np), eta(np)
         double precision, intent(out) :: matrix(nb, nb)
         integer, intent(out) :: ierr
      end subroutine gpu_solvation_local_matrix
      subroutine gpu_solvation_local_gradient(nb, na, maxprim, nc, ang, centers, expo, coeff, &
         atoms, np, points, eta, density, parent, gradient, ierr)
         implicit none
         integer, intent(in) :: nb, na, maxprim, np, nc(nb), ang(3, nb), centers(nb), parent(np)
         double precision, intent(in) :: expo(maxprim, nb), coeff(maxprim, nb), atoms(3, na)
         double precision, intent(in) :: points(3, np), eta(np), density(nb, nb)
         double precision, intent(out) :: gradient(3, na)
         integer, intent(out) :: ierr
      end subroutine gpu_solvation_local_gradient
   end interface
#endif

contains

   ! density is the symmetric TOTAL positive electron-number AO density,
   ! including both spins. No additional restricted-spin factor is applied.
   ! points and atomic coordinates are in bohr. grad_rho is a spatial
   ! derivative, not a derivative with respect to an AO center or a nucleus.
   subroutine solvation_density(points, density, rho, grad_rho, ierr)
      implicit none
      double precision, intent(in) :: points(:, :), density(:, :)
      double precision, intent(out) :: rho(:), grad_rho(:, :)
      integer, intent(out) :: ierr
      double precision, allocatable :: phi(:, :), dphi(:, :, :), pphi(:, :)
      integer :: first, nblock, ip, k, np, alloc_status

      ierr = 0
      rho = 0.d0
      grad_rho = 0.d0
      np = size(points, 2)
      if (.not. valid_ao_request(points) .or. size(rho) /= np .or. &
         size(grad_rho, 1) /= 3 .or. size(grad_rho, 2) /= np) then
         ierr = ao_error
         return
      endif
      if (.not. valid_density(density)) then
         ierr = ao_error
         return
      endif
#if defined(CUDA) || defined(CUDA_MPIV)
      if (quick_method%bGPU) then
         call gpu_solvation_density(nbasis, natom, size(aexp, 1), ncontract, itype, &
            quick_basis%ncenter, aexp, dcoeff, xyz, np, points, density, rho, grad_rho, ierr)
         if (ierr == 0) then
            if (.not. all(ieee_is_finite(rho)) .or. .not. all(ieee_is_finite(grad_rho))) ierr = ao_error
         endif
         return
      endif
#endif
      allocate(phi(nbasis, ao_block_size), dphi(3, nbasis, ao_block_size), &
         pphi(nbasis, ao_block_size), stat=alloc_status)
      if (alloc_status /= 0) then
         ierr = ao_error
         return
      endif
      do first = 1, np, ao_block_size
         nblock = min(ao_block_size, np - first + 1)
         call evaluate_ao_block(points(:, first:first+nblock-1), phi(:, :nblock), dphi(:, :, :nblock))
         call dgemm('N', 'N', nbasis, nblock, nbasis, 1.d0, density, nbasis, phi, nbasis, &
            0.d0, pphi, nbasis)
         do ip = 1, nblock
            rho(first+ip-1) = dot_product(phi(:, ip), pphi(:, ip))
            do k = 1, 3
               grad_rho(k, first+ip-1) = 2.d0 * dot_product(dphi(k, :, ip), pphi(:, ip))
            enddo
         enddo
      enddo
      if (.not. all(ieee_is_finite(rho)) .or. .not. all(ieee_is_finite(grad_rho))) ierr = ao_error
   end subroutine solvation_density


   ! matrix(mu,nu) = sum_g weighted_potential(g) chi_mu(g) chi_nu(g).
   ! The caller supplies every quadrature weight, sign and physical prefactor.
   subroutine solvation_local_matrix(points, weighted_potential, matrix, ierr)
      implicit none
      double precision, intent(in) :: points(:, :), weighted_potential(:)
      double precision, intent(out) :: matrix(:, :)
      integer, intent(out) :: ierr
      double precision, allocatable :: phi(:, :), dphi(:, :, :), weighted_phi(:, :)
      integer :: first, nblock, ip, np, alloc_status

      ierr = 0
      matrix = 0.d0
      np = size(points, 2)
      if (.not. valid_ao_request(points) .or. size(weighted_potential) /= np .or. &
         size(matrix, 1) /= nbasis .or. size(matrix, 2) /= nbasis) then
         ierr = ao_error
         return
      endif
      if (.not. all(ieee_is_finite(weighted_potential))) then
         ierr = ao_error
         return
      endif
#if defined(CUDA) || defined(CUDA_MPIV)
      if (quick_method%bGPU) then
         call gpu_solvation_local_matrix(nbasis, natom, size(aexp, 1), ncontract, itype, &
            quick_basis%ncenter, aexp, dcoeff, xyz, np, points, weighted_potential, matrix, ierr)
         if (ierr == 0 .and. .not. all(ieee_is_finite(matrix))) ierr = ao_error
         return
      endif
#endif
      allocate(phi(nbasis, ao_block_size), dphi(3, nbasis, ao_block_size), &
         weighted_phi(nbasis, ao_block_size), stat=alloc_status)
      if (alloc_status /= 0) then
         ierr = ao_error
         return
      endif
      do first = 1, np, ao_block_size
         nblock = min(ao_block_size, np - first + 1)
         call evaluate_ao_block(points(:, first:first+nblock-1), phi(:, :nblock), dphi(:, :, :nblock))
         do ip = 1, nblock
            weighted_phi(:, ip) = weighted_potential(first+ip-1) * phi(:, ip)
         enddo
         call dgemm('N', 'T', nbasis, nbasis, nblock, 1.d0, weighted_phi, nbasis, phi, nbasis, &
            1.d0, matrix, nbasis)
      enddo
      if (.not. all(ieee_is_finite(matrix))) ierr = ao_error
   end subroutine solvation_local_matrix


   ! Fixed-density derivative of sum_g eta(g)*rho(g), with each point moving
   ! with parent(g). eta already contains weights and is held constant here.
   ! Its partition-weight derivative is a SEPARATE caller-owned contribution.
   ! Both AO-center motion and parent-grid motion are included, so the sum
   ! of these atomic gradients vanishes up to roundoff.
   subroutine solvation_local_gradient(points, parent, weighted_potential, density, atomic_gradient, ierr)
      implicit none
      double precision, intent(in) :: points(:, :), weighted_potential(:), density(:, :)
      integer, intent(in) :: parent(:)
      double precision, intent(out) :: atomic_gradient(:, :)
      integer, intent(out) :: ierr
      double precision, allocatable :: phi(:, :), dphi(:, :, :), pphi(:, :)
      double precision :: contribution(3), parent_contribution(3), eta
      integer :: first, nblock, ip, ibas, icenter, np, alloc_status

      ierr = 0
      atomic_gradient = 0.d0
      np = size(points, 2)
      if (.not. valid_ao_request(points) .or. size(weighted_potential) /= np .or. &
         size(parent) /= np .or. size(atomic_gradient, 1) /= 3 .or. size(atomic_gradient, 2) /= natom) then
         ierr = ao_error
         return
      endif
      if (.not. valid_density(density) .or. any(parent < 1) .or. any(parent > natom) .or. &
         .not. all(ieee_is_finite(weighted_potential))) then
         ierr = ao_error
         return
      endif
#if defined(CUDA) || defined(CUDA_MPIV)
      if (quick_method%bGPU) then
         call gpu_solvation_local_gradient(nbasis, natom, size(aexp, 1), ncontract, itype, &
            quick_basis%ncenter, aexp, dcoeff, xyz, np, points, weighted_potential, density, parent, &
            atomic_gradient, ierr)
         if (ierr == 0 .and. .not. all(ieee_is_finite(atomic_gradient))) ierr = ao_error
         return
      endif
#endif
      allocate(phi(nbasis, ao_block_size), dphi(3, nbasis, ao_block_size), &
         pphi(nbasis, ao_block_size), stat=alloc_status)
      if (alloc_status /= 0) then
         ierr = ao_error
         return
      endif
      do first = 1, np, ao_block_size
         nblock = min(ao_block_size, np - first + 1)
         call evaluate_ao_block(points(:, first:first+nblock-1), phi(:, :nblock), dphi(:, :, :nblock))
         call dgemm('N', 'N', nbasis, nblock, nbasis, 1.d0, density, nbasis, phi, nbasis, &
            0.d0, pphi, nbasis)
         do ip = 1, nblock
            eta = weighted_potential(first+ip-1)
            parent_contribution = 0.d0
            do ibas = 1, nbasis
               contribution = 2.d0 * eta * pphi(ibas, ip) * dphi(:, ibas, ip)
               icenter = quick_basis%ncenter(ibas)
               atomic_gradient(:, icenter) = atomic_gradient(:, icenter) - contribution
               parent_contribution = parent_contribution + contribution
            enddo
            icenter = parent(first+ip-1)
            atomic_gradient(:, icenter) = atomic_gradient(:, icenter) + parent_contribution
         enddo
      enddo
      if (.not. all(ieee_is_finite(atomic_gradient))) ierr = ao_error
   end subroutine solvation_local_gradient


   logical function valid_ao_request(points)
      implicit none
      double precision, intent(in) :: points(:, :)

      valid_ao_request = .false.
      if (nbasis < 1 .or. natom < 1 .or. size(points, 1) /= 3 .or. size(points, 2) < 1) return
      if (.not. allocated(ncontract) .or. .not. allocated(itype) .or. .not. allocated(aexp) .or. &
         .not. allocated(dcoeff) .or. .not. allocated(quick_basis%ncenter) .or. .not. allocated(xyz)) return
      if (size(ncontract) /= nbasis .or. size(itype, 1) /= 3 .or. size(itype, 2) /= nbasis .or. &
         size(aexp, 2) /= nbasis .or. any(shape(aexp) /= shape(dcoeff)) .or. &
         size(quick_basis%ncenter) /= nbasis .or. size(xyz, 1) /= 3 .or. size(xyz, 2) /= natom) return
      if (any(ncontract < 1) .or. any(ncontract > size(aexp, 1)) .or. any(itype < 0) .or. &
         any(quick_basis%ncenter < 1) .or. any(quick_basis%ncenter > natom)) return
      if (.not. all(ieee_is_finite(points)) .or. .not. all(ieee_is_finite(xyz))) return
      valid_ao_request = .true.
   end function valid_ao_request


   logical function valid_density(density)
      implicit none
      double precision, intent(in) :: density(:, :)

      valid_density = .false.
      if (size(density, 1) /= nbasis .or. size(density, 2) /= nbasis) return
      if (.not. all(ieee_is_finite(density))) return
      if (maxval(abs(density - transpose(density))) > 1.d-10) return
      valid_density = .true.
   end function valid_density


   ! Evaluate normalized contracted Cartesian Gaussians directly. These are
   ! the same normalized dcoeff/aexp data used by the existing CPU density
   ! surface evaluator, with analytic spatial derivatives and no screening.
   subroutine evaluate_ao_block(points, phi, dphi)
      implicit none
      double precision, intent(in) :: points(:, :)
      double precision, intent(out) :: phi(:, :), dphi(:, :, :)
      double precision :: displacement(3), powers(3), lower(3), radial, exponent_radial, term, rsq
      integer :: ip, ibas, iprim, k, icenter

      do ip = 1, size(points, 2)
         do ibas = 1, nbasis
            icenter = quick_basis%ncenter(ibas)
            displacement = points(:, ip) - xyz(:, icenter)
            rsq = dot_product(displacement, displacement)
            do k = 1, 3
               powers(k) = displacement(k)**itype(k, ibas)
               lower(k) = 0.d0
               if (itype(k, ibas) > 0) lower(k) = &
                  dble(itype(k, ibas)) * displacement(k)**(itype(k, ibas)-1)
            enddo
            radial = 0.d0
            exponent_radial = 0.d0
            do iprim = 1, ncontract(ibas)
               term = dcoeff(iprim, ibas) * exp(-aexp(iprim, ibas)*rsq)
               radial = radial + term
               exponent_radial = exponent_radial + aexp(iprim, ibas)*term
            enddo
            phi(ibas, ip) = radial * product(powers)
            dphi(1, ibas, ip) = (radial*lower(1) - 2.d0*displacement(1)*exponent_radial*powers(1)) &
               * powers(2)*powers(3)
            dphi(2, ibas, ip) = (radial*lower(2) - 2.d0*displacement(2)*exponent_radial*powers(2)) &
               * powers(1)*powers(3)
            dphi(3, ibas, ip) = (radial*lower(3) - 2.d0*displacement(3)*exponent_radial*powers(3)) &
               * powers(1)*powers(2)
         enddo
      enddo
   end subroutine evaluate_ao_block

end module quick_solvation_ao_module
