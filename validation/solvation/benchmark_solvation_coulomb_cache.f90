!
! (C) Copyright 2026 QUICK contributors. All rights reserved.
! This Source Code Form is subject to the terms of the Mozilla Public
! License, v. 2.0. If a copy of the MPL was not distributed with this
! file, You can obtain one at https://mozilla.org/MPL/2.0/.
!
#include "util.fh"
! Developer profiling client; link against a CUDA QUICK build and its
! matching Fortran modules. Run with exclusive device access for timings.
program benchmark_solvation_coulomb_cache
   use, intrinsic :: iso_fortran_env, only: int64
   use, intrinsic :: iso_c_binding, only: c_int64_t
   use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
   use quick_api_module, only: setQuickJob, getQuickEnergy, deleteQuickJob
   use quick_method_module, only: quick_method
   use quick_basis_module, only: nbasis
   use quick_molspec_module, only: quick_molspec, atoms_bohr => xyz
   use quick_calculated_module, only: quick_qm_struct
   use quick_constants_module, only: BOHRS_TO_A
   use quick_ddx_backend_module
   use quick_solvation_grid_module
   use quick_solvation_ao_module, only: solvation_density
   use quick_solvation_coulomb_module
   use quick_oeproperties_module, only: compute_esp_values
   implicit none

   type(quick_ddx_backend_type) :: backend
   type(quick_solvation_grid_type) :: grid
   character(len=512) :: config, geometry_file, output_prefix, message, comment
   character(len=256) :: keywd
   character(len=80) :: jobname
   character(len=4) :: symbol
   integer :: repetitions, cavity_model, cavity_lmax, cavity_nlebedev, volume_nrad, volume_nang
   integer :: max_cavity_points, n_atoms, atom, unit, output_unit, meta_unit, ierr, cleanup_error
   integer :: npoints, point, repetition
   logical :: controlled_direct, repeat_gradients
   integer(int64) :: started, finished, rate, npairs, cached_bytes
   integer(c_int64_t) :: matrix_calls, gradient_calls, cache_limit_bytes
   integer(c_int64_t) :: cache_hits, cache_builds, cache_bytes, cache_gemvs, direct_calls, cache_fallbacks, allocated_bytes
   double precision :: dielectric, radius_scale, solver_tolerance, energy, solvent_energy, seconds
   double precision :: electron_count, matrix_error, gradient_error, esp_error, controlled_error
   double precision :: external_charges(4, 0)
   integer, allocatable :: atomic_numbers(:)
   double precision, allocatable :: coords(:, :), radii(:), density(:, :), rho(:), grad_rho(:, :)
   double precision, allocatable :: psi(:, :), ylm(:), phi(:), x(:, :), xi(:), points(:, :), charges(:)
   double precision, allocatable :: matrix(:, :), matrix_first(:, :), grad(:, :), point_grad(:, :)
   double precision, allocatable :: grad_first(:, :), point_grad_first(:, :), esp(:), esp_first(:)

   namelist /coulomb_benchmark/ geometry_file, output_prefix, keywd, repetitions, &
      cavity_model, cavity_lmax, cavity_nlebedev, volume_nrad, volume_nang, &
      dielectric, radius_scale, solver_tolerance, max_cavity_points, cache_limit_bytes, controlled_direct, repeat_gradients

   geometry_file = ''
   output_prefix = 'water20-coulomb'
   keywd = 'DFT PBE D3 BASIS=DEF2-SVP ENERGY CUTOFF=1.0D-12 XCCUTOFF=1.0D-10 '// &
      'BASISCUTOFF=1.0D-10 DENSERMS=1.0D-8 SCF=100 CHARGE=0'
   repetitions = 3
   cavity_model = 2
   cavity_lmax = 4
   cavity_nlebedev = 110
   volume_nrad = 50
   volume_nang = 110
   dielectric = 78.3553d0
   radius_scale = 1.2d0
   solver_tolerance = 1.0d-10
   max_cavity_points = 0
   cache_limit_bytes = 8_c_int64_t * 1024 * 1024 * 1024
   controlled_direct = .true.
   repeat_gradients = .false.
   ierr = 0
   message = ''
   jobname = 'water_cluster_coulomb_benchmark'
   if (command_argument_count() /= 1) then
      print *, 'Usage: benchmark-solvation-coulomb-cache benchmark.nml'
      stop 1
   endif
   call get_command_argument(1, config)
   open(newunit=unit, file=trim(config), status='old', action='read', iostat=ierr)
   call check_error('Open benchmark config')
   read(unit, nml=coulomb_benchmark, iostat=ierr)
   close(unit)
   call check_error('Read benchmark config')
   if (repetitions < 1 .or. max_cavity_points < 0 .or. cache_limit_bytes < 0 .or. &
      len_trim(geometry_file) == 0) stop 2
   open(newunit=unit, file=trim(geometry_file), status='old', action='read', iostat=ierr)
   call check_error('Open XYZ geometry')
   read(unit, *, iostat=ierr) n_atoms
   call check_error('Read XYZ atom count')
   read(unit, '(A)', iostat=ierr) comment
   call check_error('Read XYZ comment')
   allocate(atomic_numbers(n_atoms), coords(3, n_atoms), radii(n_atoms))
   do atom = 1, n_atoms
      read(unit, *, iostat=ierr) symbol, coords(:, atom)
      call check_error('Read XYZ atom')
      select case(trim(symbol))
      case('H', 'h')
         atomic_numbers(atom) = 1
         radii(atom) = 1.20d0 * radius_scale / BOHRS_TO_A
      case('O', 'o')
         atomic_numbers(atom) = 8
         radii(atom) = 1.52d0 * radius_scale / BOHRS_TO_A
      case default
         print *, 'This water-cluster benchmark only accepts H and O.'
         stop 3
      end select
   enddo
   close(unit)
   open(newunit=output_unit, file=trim(output_prefix)//'-timings.csv', status='new', action='write', iostat=ierr)
   call check_error('Create timing CSV')
   write(output_unit, '(A)') 'phase,repetition,wall_seconds,max_abs_value,repeat_error,gpu_matrix_evaluations,'// &
      'gpu_gradient_launches,cache_hits,cache_builds,cache_T_bytes,cache_gemvs,direct_matrix_launches,'// &
      'cache_fallbacks,cache_allocated_bytes'
   open(newunit=meta_unit, file=trim(output_prefix)//'-metadata.dat', status='new', action='write', iostat=ierr)
   call check_error('Create metadata')
   write(meta_unit, '(A)') 'profile_only_not_scientific_acceptance true'
   write(meta_unit, '(A)') 'density gas_phase_converged_actual_QM_density'
   write(meta_unit, '(A)') 'timing_scope end_to_end_operator_including_required_allocations_uploads_and_downloads'
   write(meta_unit, '(A)') 'cache_policy exact_FP64_T_reverse_operator_only_no_forward_ESP_changes'
   write(meta_unit, '(A,1X,I0)') 'requested_cache_limit_bytes', cache_limit_bytes
   write(meta_unit, '(A,1X,L1)') 'controlled_direct_reference_requested', controlled_direct
   write(meta_unit, '(A,1X,L1)') 'repeat_direct_gradients_each_matrix_repetition', repeat_gradients
   write(meta_unit, '(A)') 'cache_limit_upper_bound min_requested_8GiB_quarter_free_VRAM_including_owned_workspace'
   write(meta_unit, '(A)') 'matrix_counter_semantics successful_evaluations_not_literal_kernel_launches'
   write(meta_unit, '(A,1X,A)') 'geometry_file', trim(geometry_file)
   write(meta_unit, '(A,1X,A)') 'keywords', trim(keywd)

   call setQuickJob(jobname, keywd, n_atoms, atomic_numbers, .false., ierr)
   call check_error('Set QUICK job')
   if (.not. quick_method%bGPU) then
      print *, 'This profiling client requires an active CUDA QUICK backend.'
      stop 4
   endif
   if (quick_method%ddx) then
      print *, 'The initial SCF must be gas phase; omit DDX from keywd.'
      stop 5
   endif
   call solvation_coulomb_gpu_counters(matrix_calls, gradient_calls, .true.)
   call system_clock(started, rate)
   call getQuickEnergy(coords, 0, external_charges, energy, ierr)
   call check_error('Gas-phase density SCF')
   call system_clock(finished)
   if (.not. quick_method%scf_conv .or. .not. ieee_is_finite(energy)) stop 6
   call record('initial_scf', 0, abs(energy), 0.0d0)
   write(meta_unit, '(A,1X,I0)') 'atoms', n_atoms
   write(meta_unit, '(A,1X,I0)') 'basis_functions', nbasis
   write(meta_unit, '(A,1X,ES26.17E3)') 'gas_phase_energy', energy

   call system_clock(started)
   call ddx_backend_initialize(backend, cavity_model, n_atoms, atoms_bohr, radii, quick_molspec%chg, &
      dielectric, cavity_lmax, cavity_nlebedev, solver_tolerance, ierr, message)
   call check_error('Build native ddX cavity')
   call system_clock(finished)
   call record('cavity_build', 0, dble(backend%ncav), 0.0d0)
   call system_clock(started)
   call build_solvation_grid(n_atoms, atoms_bohr, volume_nrad, volume_nang, grid, ierr)
   call check_error('Build volume quadrature')
   call system_clock(finished)
   call record('volume_grid', 0, dble(grid%npoints), 0.0d0)
   allocate(density(nbasis, nbasis), rho(grid%npoints), grad_rho(3, grid%npoints), &
      psi(backend%nlm, n_atoms), ylm(backend%nlm), phi(backend%ncav), &
      x(backend%nlm, n_atoms), xi(backend%ncav))
   density = quick_qm_struct%dense
   if (quick_method%unrst) density = density + quick_qm_struct%denseb
   call system_clock(started)
   call solvation_density(grid%points, density, rho, grad_rho, ierr)
   call check_error('Evaluate actual volume density on GPU')
   call system_clock(finished)
   call record('volume_density', 0, maxval(abs(rho)), 0.0d0)
   electron_count = sum(grid%weight * rho)
   call system_clock(started)
   psi = 0.0d0
   do point = 1, grid%npoints
      call ddx_backend_scaled_ylm(backend, grid%points(:, point), grid%parent(point), ylm, ierr, message)
      call check_error('Density projection harmonic')
      psi(:, grid%parent(point)) = psi(:, grid%parent(point)) - grid%weight(point) * rho(point) * ylm
   enddo
   call system_clock(finished)
   call record('density_projection', 0, maxval(abs(psi)), 0.0d0)
   call system_clock(started)
   call compute_esp_values(backend%ncav, backend%cavity, phi)
   call system_clock(finished)
   call record('full_cavity_esp', 0, maxval(abs(phi)), 0.0d0)
   call system_clock(started)
   call ddx_backend_solve(backend, phi, psi, solvent_energy, x, xi, ierr, message)
   call check_error('Solve actual-density native ddX primal and adjoint')
   call system_clock(finished)
   call record('ddx_solve', 0, abs(solvent_energy), 0.0d0)

   npoints = backend%ncav
   if (max_cavity_points > 0) npoints = min(npoints, max_cavity_points)
   points = backend%cavity(:, :npoints)
   charges = 0.5d0 * backend%scaling * xi(:npoints)
   allocate(matrix(nbasis, nbasis), matrix_first(nbasis, nbasis), grad(3, n_atoms), &
      point_grad(3, npoints), grad_first(3, n_atoms), point_grad_first(3, npoints), &
      esp(npoints), esp_first(npoints))
   npairs = int(nbasis, int64) * (int(nbasis, int64) + 1_int64) / 2_int64
   cached_bytes = 8_int64 * npairs * int(npoints, int64)
   write(meta_unit, '(A,1X,I0)') 'cavity_points_total', backend%ncav
   write(meta_unit, '(A,1X,I0)') 'cavity_points_timed', npoints
   write(meta_unit, '(A,1X,L1)') 'complete_cavity_timed', npoints == backend%ncav
   write(meta_unit, '(A,1X,I0)') 'volume_points', grid%npoints
   write(meta_unit, '(A,1X,ES26.17E3)') 'volume_electrons', electron_count
   write(meta_unit, '(A,1X,ES26.17E3)') 'fixed_density_solvation_energy', solvent_energy
   write(meta_unit, '(A,1X,I0)') 'unique_ao_pairs', npairs
   write(meta_unit, '(A,1X,I0)') 'hypothetical_fp64_packed_T_bytes', cached_bytes
   write(meta_unit, '(A,1X,F16.8)') 'hypothetical_fp64_packed_T_GiB', dble(cached_bytes) / 1024.0d0**3
   write(meta_unit, '(A,1X,ES26.17E3)') 'reaction_charge_l1', sum(abs(charges))
   flush(meta_unit)
   call clear_solvation_coulomb_cache(ierr)
   call check_error('Clear optional Coulomb cache')
   call solvation_coulomb_cache_set_limit(cache_limit_bytes, ierr)
   call check_error('Set optional Coulomb cache budget')
   do repetition = 0, repetitions
      call system_clock(started)
      call compute_esp_values(npoints, points, esp)
      call system_clock(finished)
      if (repetition == 0) esp_first = esp
      esp_error = maxval(abs(esp - esp_first))
      call record('esp_forward', repetition, maxval(abs(esp)), esp_error)

      call system_clock(started)
      call solvation_coulomb_matrix(points, charges, matrix, ierr, message)
      call check_error('GPU Coulomb AO matrix')
      call system_clock(finished)
      if (repetition == 0) matrix_first = matrix
      matrix_error = maxval(abs(matrix - matrix_first))
      call record('coulomb_matrix', repetition, maxval(abs(matrix)), matrix_error)

      if (repetition == 0 .or. repeat_gradients) then
         call system_clock(started)
         call solvation_coulomb_gradient(points, charges, density, grad, point_grad, ierr, message)
         call check_error('GPU Coulomb analytic gradients')
         call system_clock(finished)
         if (repetition == 0) then
            grad_first = grad
            point_grad_first = point_grad
         endif
         gradient_error = max(maxval(abs(grad - grad_first)), maxval(abs(point_grad - point_grad_first)))
         call record('coulomb_gradient', repetition, max(maxval(abs(grad)), maxval(abs(point_grad))), gradient_error)
      endif
      if (.not. all(ieee_is_finite(matrix)) .or. .not. all(ieee_is_finite(grad)) .or. &
         .not. all(ieee_is_finite(point_grad))) stop 7
   enddo
   call solvation_coulomb_gpu_counters(matrix_calls, gradient_calls)
   if (matrix_calls /= repetitions + 1 .or. gradient_calls /= merge(repetitions + 1, 1, repeat_gradients)) stop 8
   write(meta_unit, '(A,2(1X,I0))') 'coulomb_GPU_matrix_evaluations_and_direct_gradient_launches', matrix_calls, gradient_calls
   call solvation_coulomb_cache_info(cache_hits, cache_builds, cache_bytes, cache_gemvs, direct_calls, &
      cache_fallbacks, allocated_bytes)
   write(meta_unit, '(A,7(1X,I0))') 'cache_hits_builds_Tbytes_gemvs_direct_fallbacks_allocatedbytes', &
      cache_hits, cache_builds, cache_bytes, cache_gemvs, direct_calls, cache_fallbacks, allocated_bytes
   if (cache_gemvs + direct_calls /= matrix_calls) stop 11
   if (cache_bytes > 0 .and. (cache_builds /= 1 .or. cache_hits /= repetitions)) stop 12
   ! A direct CUDA control uses the exact same geometry, basis, density and
   ! reaction charges already resident in this client. No second SCF, ddX
   ! solve or input preparation can change the comparison.
   if (controlled_direct .and. cache_bytes == 0) then
      print *, 'Controlled cache/direct profiling requires a resident T cache; check capacity and free VRAM.'
      stop 15
   endif
   if (controlled_direct .and. cache_bytes > 0) then
      call solvation_coulomb_cache_set_limit(0_c_int64_t, ierr)
      call check_error('Select direct CUDA control')
      controlled_error = 0.0d0
      do repetition = 0, repetitions
         call system_clock(started)
         call solvation_coulomb_matrix(points, charges, matrix, ierr, message)
         call check_error('Direct CUDA control matrix')
         call system_clock(finished)
         matrix_error = maxval(abs(matrix - matrix_first))
         controlled_error = max(controlled_error, matrix_error)
         call record('coulomb_matrix_direct_control', repetition, maxval(abs(matrix)), matrix_error)
         if (matrix_error > 1.0d-10 .or. .not. all(ieee_is_finite(matrix))) stop 13
      enddo
      call solvation_coulomb_cache_info(cache_hits, cache_builds, cache_bytes, cache_gemvs, direct_calls, &
         cache_fallbacks, allocated_bytes)
      if (cache_gemvs /= repetitions + 1 .or. direct_calls /= repetitions + 1 .or. cache_bytes /= 0) stop 14
      write(meta_unit, '(A,1X,ES26.17E3)') 'controlled_direct_vs_cached_max_matrix_error', controlled_error
      write(meta_unit, '(A,7(1X,I0))') 'final_cache_hits_builds_Tbytes_gemvs_direct_fallbacks_allocatedbytes', &
         cache_hits, cache_builds, cache_bytes, cache_gemvs, direct_calls, cache_fallbacks, allocated_bytes
   endif
   write(meta_unit, '(A,1X,ES26.17E3)') 'matrix_symmetry_error', maxval(abs(matrix - transpose(matrix)))
   write(meta_unit, '(A,1X,ES26.17E3)') 'gradient_translation_error', &
      maxval(abs(sum(grad, dim=2) + sum(point_grad, dim=2)))
   write(meta_unit, '(A,1X,ES26.17E3)') 'density_matrix_contraction', sum(density * matrix)
   call ddx_backend_finalize(backend)
   call destroy_solvation_grid(grid)
   cleanup_error = 0
   call deleteQuickJob(cleanup_error)
   if (cleanup_error /= 0) stop 9
   close(output_unit)
   close(meta_unit)
contains
   subroutine check_error(context)
      implicit none
      character(len=*), intent(in) :: context
      if (ierr /= 0) then
         print *, trim(context), ': ierr=', ierr, ' ', trim(message)
         stop 10
      endif
   end subroutine check_error

   subroutine record(phase, iteration, maximum, repeat_error)
      implicit none
      character(len=*), intent(in) :: phase
      integer, intent(in) :: iteration
      double precision, intent(in) :: maximum, repeat_error
      seconds = dble(finished - started) / dble(rate)
      call solvation_coulomb_gpu_counters(matrix_calls, gradient_calls)
      call solvation_coulomb_cache_info(cache_hits, cache_builds, cache_bytes, cache_gemvs, direct_calls, &
         cache_fallbacks, allocated_bytes)
      write(output_unit, '(A,",",I0,3(",",ES26.17E3),9(",",I0))') &
         trim(phase), iteration, seconds, maximum, repeat_error, matrix_calls, gradient_calls, &
         cache_hits, cache_builds, cache_bytes, cache_gemvs, direct_calls, cache_fallbacks, allocated_bytes
      flush(output_unit)
      print '(A,1X,I0,1X,F12.6,A)', trim(phase), iteration, seconds, ' s'
      flush(6)
   end subroutine record
end program benchmark_solvation_coulomb_cache
