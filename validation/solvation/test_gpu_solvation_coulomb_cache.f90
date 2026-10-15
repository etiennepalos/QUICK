!
! (C) Copyright 2026 QUICK contributors. All rights reserved.
! This Source Code Form is subject to the terms of the Mozilla Public
! License, v. 2.0. If a copy of the MPL was not distributed with this
! file, You can obtain one at https://mozilla.org/MPL/2.0/.
!
#include "util.fh"
! Standalone regression client. Link against the matching CUDA QUICK
! library/modules. CPU reference and direct CUDA remain independent paths.
program test_gpu_solvation_coulomb_cache
   use, intrinsic :: iso_c_binding, only: c_int64_t
   use quick_solvation_coulomb_module
   use quick_basis_module, only: quick_basis, nbasis, itype, ncontract, aexp, dcoeff
   use quick_molspec_module, only: natom, xyz
   use quick_method_module, only: quick_method
   implicit none
   integer :: mu, i, j, k, l, ierr, cases, saved_angular(3)
   integer(c_int64_t), parameter :: full_limit = 4_c_int64_t * 1024 * 1024 * 1024
   integer(c_int64_t) :: hits, builds, bytes, gemvs, direct, fallbacks, allocated_bytes, nm, ng
   integer(c_int64_t) :: expected_hits, expected_builds, expected_gemvs, expected_direct, expected_fallbacks
   double precision :: points(3, 5), charges(5), density(20, 20), largest_error
   double precision :: reference(20, 20), device(20, 20), ga_ref(3, 2), gc_ref(3, 5)
   double precision :: ga(3, 2), gc(3, 5), error, translation
   character(len=1024) :: message

   ierr = 0
   cases = 0
   largest_error = 0.0d0
   call clear_solvation_coulomb_cache(ierr)
   call check_error('empty initial cleanup')
   call zero_expected()
   call stats('empty initial cleanup', 0_c_int64_t)
   natom = 2
   nbasis = 20
   allocate(xyz(3, 2), quick_basis%ncenter(20), itype(3, 20), ncontract(20), &
      aexp(3, 20), dcoeff(3, 20))
   xyz(:, 1) = (/ -0.2d0, 0.3d0, 0.7d0 /)
   xyz(:, 2) = (/ 0.9d0, -0.4d0, 0.2d0 /)
   mu = 0
   do l = 0, 3
      do i = 0, l
         do j = 0, l - i
            k = l - i - j
            mu = mu + 1
            itype(:, mu) = (/ i, j, k /)
         enddo
      enddo
   enddo
   if (mu /= 20) stop 1
   aexp = 7.0d0
   dcoeff = 0.0d0
   do mu = 1, 20
      quick_basis%ncenter(mu) = mod(mu, 2) + 1
      ncontract(mu) = 2
      aexp(1:2, mu) = (/ 0.4d0 + 0.02d0 * mu, 1.3d0 + 0.01d0 * mu /)
      dcoeff(1:2, mu) = (/ 0.7d0 - 0.01d0 * mu, -0.1d0 + 0.005d0 * mu /)
      do j = 1, 20
         density(mu, j) = 0.01d0 * sin(dble(3 * mu + j))
      enddo
      density(mu, mu) = 0.2d0 + 0.001d0 * mu
   enddo
   points(:, 1) = (/ -0.8d0, 1.2d0, 0.4d0 /)
   points(:, 2) = xyz(:, 2)
   points(:, 3) = (/ 25.0d0, -10.0d0, 8.0d0 /)
   points(:, 4) = (/ 0.0d0, 0.0d0, 0.0d0 /)
   points(:, 5) = (/ 0.7d0, 0.2d0, -0.4d0 /)
   charges = (/ 0.3d0, -0.2d0, 0.5d0, -0.1d0, -0.8d0 /)

   call solvation_coulomb_cache_set_limit(0_c_int64_t, ierr)
   call check_case('forced direct GPU', 'direct', 5)
   call clear_solvation_coulomb_cache(ierr)
   call zero_expected()
   call check_case('first T build', 'build', 5)
   charges = charges * (-0.7d0)
   call check_case('changed charges reuse T', 'hit', 5)
   density = 1.1d0 * density
   call check_case('changed density reuse T', 'hit', 5)
   aexp(3, 1) = 9.0d0
   dcoeff(3, 1) = -3.0d0
   call check_case('inactive primitive change reuse T', 'hit', 5)

   quick_method%bGPU = .false.
   call solvation_coulomb_gradient(points, charges, density, ga_ref, gc_ref, ierr, message)
   call check_error('CPU non-symmetric density gradient')
   quick_method%bGPU = .true.
   call solvation_coulomb_gradient(points, charges, density, ga, gc, ierr, message)
   call check_error('GPU direct gradient with resident T')
   error = max(maxval(abs(ga - ga_ref)), maxval(abs(gc - gc_ref)))
   translation = maxval(abs(sum(ga, dim=2) + sum(gc, dim=2)))
   if (error > 1.0d-11 .or. translation > 1.0d-11) stop 2
   largest_error = max(largest_error, error)
   call solvation_coulomb_gpu_counters(nm, ng)
   if (ng /= 1) stop 3
   call stats('gradient leaves T reusable', 210_c_int64_t * 5 * 8)

   ncontract(1) = 1
   call check_case('changed primitive count', 'build', 5)
   ncontract(1) = 2
   call check_case('restored primitive count', 'build', 5)
   aexp(1, 1) = aexp(1, 1) + 0.03d0
   call check_case('changed active exponent', 'build', 5)
   dcoeff(2, 4) = dcoeff(2, 4) + 0.02d0
   call check_case('changed active coefficient', 'build', 5)
   xyz(2, 1) = xyz(2, 1) + 0.04d0
   call check_case('changed nuclear geometry', 'build', 5)
   points(3, 4) = points(3, 4) + 0.07d0
   call check_case('changed cavity coordinate', 'build', 5)
   quick_basis%ncenter(5) = 3 - quick_basis%ncenter(5)
   call check_case('changed AO center mapping', 'build', 5)
   saved_angular = itype(:, 4)
   itype(:, 4) = (/ 0, 1, 0 /)
   call check_case('changed angular powers', 'build', 5)
   itype(:, 4) = saved_angular
   call check_case('restored angular powers', 'build', 5)
   call check_case('changed cavity count', 'build', 4)

   quick_method%bGPU = .true.
   call solvation_coulomb_matrix(points(:, 1:0), charges(1:0), device, ierr, message)
   call check_error('empty cavity invalidation')
   if (maxval(abs(device)) /= 0.0d0) stop 4
   call stats('empty cavity invalidation', 0_c_int64_t)
   call check_case('restored nonempty cavity', 'build', 5)
   call clear_solvation_coulomb_cache(ierr, reset_statistics=.false.)
   call check_error('geometry cleanup')
   call stats('geometry cleanup retains counters', 0_c_int64_t)
   call check_case('rebuild after geometry cleanup', 'build', 5)

   call solvation_coulomb_cache_set_limit(1_c_int64_t, ierr)
   call check_case('oversized cache direct GPU fallback', 'direct', 5)
   call solvation_coulomb_cache_set_limit(full_limit, ierr)
   call check_case('cache restored after fallback', 'build', 5)
   call solvation_coulomb_cache_set_limit(-1_c_int64_t, ierr)
   if (ierr /= QUICK_SOLVATION_COULOMB_ERROR) stop 5
   ierr = 0
   call check_case('rejected limit retains valid cache', 'hit', 5)

   ierr = 17
   call clear_solvation_coulomb_cache(ierr, reset_statistics=.false.)
   if (ierr /= 17) stop 6
   ierr = 0
   call stats('error cleanup preserves first error and counters', 0_c_int64_t)
   call clear_solvation_coulomb_cache(ierr)
   call zero_expected()
   call stats('job cleanup resets counters', 0_c_int64_t)
   call solvation_coulomb_gpu_counters(nm, ng)
   if (nm /= 0 .or. ng /= 0) stop 7
   call check_case('second job first build', 'build', 5)
   call check_case('second job repeated matrix', 'hit', 5)
   call clear_solvation_coulomb_cache(ierr)
   call zero_expected()
   call stats('second job cleanup', 0_c_int64_t)
   call clear_solvation_coulomb_cache(ierr)
   call check_error('repeated empty cleanup')
   print '(a,i0,a,es16.8)', 'PASS CUDA Coulomb cache: ', cases, ' matrix cases; max CPU/GPU error=', largest_error
   print '(a)', 'Device-switch case requires a second CUDA device; this driver uses the current device.'
contains
   subroutine zero_expected()
      expected_hits = 0
      expected_builds = 0
      expected_gemvs = 0
      expected_direct = 0
      expected_fallbacks = 0
   end subroutine zero_expected

   subroutine check_error(label)
      character(len=*), intent(in) :: label
      if (ierr /= 0) then
         print *, 'FAIL ', label, ': ', ierr, trim(message)
         stop 8
      endif
   end subroutine check_error

   subroutine stats(label, expected_bytes)
      character(len=*), intent(in) :: label
      integer(c_int64_t), intent(in) :: expected_bytes
      call solvation_coulomb_cache_info(hits, builds, bytes, gemvs, direct, fallbacks, allocated_bytes)
      if (hits /= expected_hits .or. builds /= expected_builds .or. bytes /= expected_bytes .or. &
         gemvs /= expected_gemvs .or. direct /= expected_direct .or. fallbacks /= expected_fallbacks) then
         print *, 'FAIL cache statistics: ', label
         print *, 'actual ', hits, builds, bytes, gemvs, direct, fallbacks
         print *, 'expect ', expected_hits, expected_builds, expected_bytes, expected_gemvs, expected_direct, &
            expected_fallbacks
         stop 9
      endif
      if (expected_bytes == 0 .and. allocated_bytes /= 0) stop 10
      if (expected_bytes > 0 .and. allocated_bytes <= expected_bytes) stop 11
   end subroutine stats

   subroutine check_case(label, mode, np)
      character(len=*), intent(in) :: label, mode
      integer, intent(in) :: np
      integer(c_int64_t) :: expected_bytes
      quick_method%bGPU = .false.
      call solvation_coulomb_matrix(points(:, 1:np), charges(1:np), reference, ierr, message)
      call check_error('CPU '//label)
      quick_method%bGPU = .true.
      call solvation_coulomb_matrix(points(:, 1:np), charges(1:np), device, ierr, message)
      call check_error('GPU '//label)
      error = maxval(abs(reference - device))
      largest_error = max(largest_error, error)
      if (error > 1.0d-11 .or. maxval(abs(device - transpose(device))) /= 0.0d0) then
         print *, 'FAIL matrix ', label, error
         stop 12
      endif
      expected_bytes = 210_c_int64_t * np * 8
      select case (mode)
      case ('build')
         expected_builds = expected_builds + 1
         expected_gemvs = expected_gemvs + 1
      case ('hit')
         expected_hits = expected_hits + 1
         expected_gemvs = expected_gemvs + 1
      case ('direct')
         expected_direct = expected_direct + 1
         expected_fallbacks = expected_fallbacks + 1
         expected_bytes = 0
      case default
         stop 13
      end select
      call stats(label, expected_bytes)
      call solvation_coulomb_gpu_counters(nm, ng)
      if (nm /= expected_gemvs + expected_direct) stop 14
      cases = cases + 1
      print '(a,a,a,es12.4)', 'PASS ', label, ': matrix error=', error
   end subroutine check_case
end program test_gpu_solvation_coulomb_cache
