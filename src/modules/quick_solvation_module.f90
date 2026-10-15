! (C) Copyright 2026 QUICK contributors. All rights reserved.
! This Source Code Form is subject to the Mozilla Public License, v. 2.0.
#include "util.fh"

! Variational continuous-density QM/ddX coupling. Default-disabled calls
! return before accessing or changing any published QUICK calculation state.
module quick_solvation_module
   use iso_fortran_env, only: int64
   use quick_ddx_backend_module
   use quick_solvation_grid_module
   use quick_solvation_ao_module
   use quick_solvation_coulomb_module
   use quick_method_module, only: quick_method, quick_method_type
   use quick_molspec_module, only: quick_molspec, xyz, natom
   use quick_basis_module, only: nbasis
   use quick_calculated_module, only: quick_qm_struct
   use quick_oeproperties_module, only: compute_esp_values, compute_efield_values
   use quick_constants_module, only: BOHRS_TO_A
   use quick_mpi_module, only: master
#ifdef MPIV
   use mpi
   use quick_mpi_module, only: quick_comm, quick_mpi_error, bMPI
#endif
   implicit none
   private
   public :: add_solvation_operator, add_solvation_gradient, finalize_solvation
   public :: get_solvation_contributions
   public :: solvation_energy, solvation_electron_count, solvation_updates
   public :: solvation_gpu_volume_density, solvation_gpu_volume_matrix, solvation_gpu_volume_gradient
   public :: solvation_wall_seconds, solvation_timing_labels
   character(len=24), parameter :: solvation_timing_labels(14) = [character(len=24) :: &
      'cavity_model', 'volume_grid', 'solid_harmonics', 'cavity_esp', 'volume_density', &
      'density_projection', 'ddx_solve', 'local_matrix', 'coulomb_matrix', 'cavity_efield', &
      'ddx_gradient', 'coulomb_gradient', 'volume_gradient', 'partition_gradient']
   double precision, save :: solvation_wall_seconds(14) = 0.d0
   type(quick_ddx_backend_type), save :: backend
   type(quick_solvation_grid_type), save :: grid
   type(quick_method_type), save :: cached_method
   double precision, allocatable, save :: cached_xyz(:, :), cavity(:, :), scaled_y(:, :)
   double precision, allocatable, save :: cached_nuclear_charges(:)
   double precision, allocatable, save :: x(:, :), xi(:), density(:, :), rho(:), rho_grad(:, :)
   double precision, allocatable, save :: volume_potential(:), potential_matrix(:, :)
   logical, save :: initialized = .false.
   double precision, save :: solvation_energy = 0.d0, solvation_electron_count = 0.d0
   integer, save :: solvation_updates = 0
   integer, save :: solvation_gpu_volume_density = 0, solvation_gpu_volume_matrix = 0
   integer, save :: solvation_gpu_volume_gradient = 0

contains
   double precision function wall_time()
      implicit none
      integer(int64) :: count, rate
      call system_clock(count,rate)
      wall_time = real(count,kind(1.d0))/real(rate,kind(1.d0))
   end function wall_time

   subroutine record_phase(phase,start)
      implicit none
      integer, intent(in) :: phase
      double precision, intent(in) :: start
      solvation_wall_seconds(phase) = solvation_wall_seconds(phase) + wall_time()-start
   end subroutine record_phase

   subroutine finalize_solvation(reset_statistics)
      implicit none
      logical, intent(in), optional :: reset_statistics
      logical :: clear_statistics
      integer :: ierr

      clear_statistics = .true.
      if (present(reset_statistics)) clear_statistics = reset_statistics
      ierr = 0
      call clear_solvation_coulomb_cache(ierr,reset_statistics=clear_statistics)
      if (ierr /= 0) call solvent_failure(ierr,'Cannot release the ddX Coulomb geometry cache.')
      call ddx_backend_finalize(backend)
      call destroy_solvation_grid(grid)
      if (allocated(cached_xyz)) deallocate(cached_xyz)
      if (allocated(cached_nuclear_charges)) deallocate(cached_nuclear_charges)
      if (allocated(cavity)) deallocate(cavity)
      if (allocated(scaled_y)) deallocate(scaled_y)
      if (allocated(x)) deallocate(x)
      if (allocated(xi)) deallocate(xi)
      if (allocated(density)) deallocate(density)
      if (allocated(rho)) deallocate(rho)
      if (allocated(rho_grad)) deallocate(rho_grad)
      if (allocated(volume_potential)) deallocate(volume_potential)
      if (allocated(potential_matrix)) deallocate(potential_matrix)
      initialized = .false.
      solvation_energy = 0.d0
      solvation_electron_count = 0.d0
      if (clear_statistics) then
         solvation_wall_seconds = 0.d0
         solvation_updates = 0
         solvation_gpu_volume_density = 0
         solvation_gpu_volume_matrix = 0
         solvation_gpu_volume_gradient = 0
      endif
   end subroutine finalize_solvation

   logical function geometry_unchanged()
      implicit none
      geometry_unchanged = .false.
      if (.not.initialized .or. .not.allocated(cached_xyz)) return
      if (.not.allocated(cached_nuclear_charges)) return
      if (size(cached_xyz,2) /= natom) return
      if (size(cached_nuclear_charges) /= natom) return
      if (any(cached_xyz /= xyz)) return
      if (any(cached_nuclear_charges /= quick_molspec%chg(1:natom))) return
      if (size(density,1) /= nbasis) return
      if (cached_method%ddx_model /= quick_method%ddx_model .or. &
          cached_method%ddx_lmax /= quick_method%ddx_lmax .or. &
          cached_method%ddx_nlebedev /= quick_method%ddx_nlebedev .or. &
          cached_method%ddx_nrad /= quick_method%ddx_nrad .or. &
          cached_method%ddx_nang /= quick_method%ddx_nang .or. &
          cached_method%ddx_epsilon /= quick_method%ddx_epsilon .or. &
          cached_method%ddx_radius_scale /= quick_method%ddx_radius_scale .or. &
          cached_method%ddx_tolerance /= quick_method%ddx_tolerance) return
      geometry_unchanged = .true.
   end function geometry_unchanged

   subroutine prepare_solvation()
      use quick_files_module, only: ioutfile
      implicit none
      double precision :: radii(natom), start
      integer :: ierr, npoint, i
      character(len=512) :: message
      ! Source compatibility must be checked on EVERY API evaluation,
      ! including calls that retain the nuclear geometry and its cache.
      if (quick_molspec%nextatom /= 0) &
         call solvent_failure(53,'ddX external point-charge sources are not implemented.')
      if (geometry_unchanged()) return
      ! Moving nuclei invalidate the geometry cache, while counters cover
      ! the entire job, including every geometry in an MD trajectory.
      call finalize_solvation(reset_statistics=.false.)
      ierr = 0
      cached_xyz = xyz
      cached_nuclear_charges = quick_molspec%chg(1:natom)
      cached_method = quick_method
      if (master) then
         start = wall_time()
         call cavity_radii(radii)
         call ddx_backend_initialize(backend, quick_method%ddx_model, natom, xyz, radii, &
            quick_molspec%chg, quick_method%ddx_epsilon, quick_method%ddx_lmax, &
            quick_method%ddx_nlebedev, quick_method%ddx_tolerance, ierr, message)
         if (ierr /= 0) call solvent_failure(ierr,message)
         cavity = backend%cavity
         call record_phase(1,start)
         start = wall_time()
         call build_solvation_grid(natom,xyz,quick_method%ddx_nrad,quick_method%ddx_nang,grid,ierr)
         if (ierr /= 0) call solvent_failure(ierr,'Cannot construct the ddX volume quadrature.')
         call record_phase(2,start)
         start = wall_time()
         allocate(scaled_y(backend%nlm,grid%npoints))
         do i = 1, grid%npoints
            call ddx_backend_scaled_ylm(backend,grid%points(:,i),grid%parent(i),scaled_y(:,i),ierr,message)
            if (ierr /= 0) call solvent_failure(ierr,message)
         enddo
         call record_phase(3,start)
         allocate(x(backend%nlm,natom),xi(backend%ncav),density(nbasis,nbasis), &
            rho(grid%npoints),rho_grad(3,grid%npoints),volume_potential(grid%npoints), &
            potential_matrix(nbasis,nbasis))
         npoint = backend%ncav
         write(ioutfile,'(/,A,I0,A,I0,A,I0)') ' DDX model=',quick_method%ddx_model, &
            ' cavity points=',npoint,' volume points=',grid%npoints
         write(ioutfile,'(A,F14.7,A,ES12.4)') ' DDX dielectric=',quick_method%ddx_epsilon, &
            ' solver tolerance=',quick_method%ddx_tolerance
         if (quick_method%external_efield) write(ioutfile,'(A)') &
            ' DDX_FIELD=SOLUTE: applied perturbation acts on the QM solute; dielectric responds to its density.'
      endif
#ifdef MPIV
      call MPI_BCAST(npoint,1,mpi_integer,0,quick_comm,quick_mpi_error)
      if (.not.master) allocate(cavity(3,npoint),density(nbasis,nbasis))
      call MPI_BCAST(cavity,3*npoint,mpi_double_precision,0,quick_comm,quick_mpi_error)
#endif
      initialized = .true.
   end subroutine prepare_solvation

   subroutine update_solvation()
      implicit none
      double precision, allocatable :: phi(:), psi(:, :), coulomb_matrix(:, :)
      double precision :: start
      integer :: ierr, i
      character(len=512) :: message
      call prepare_solvation()
      allocate(phi(size(cavity,2)))
      ! OEPROP owns its usual job-local collective. All ranks participate;
      ! only the master contracts the volume density and runs ddX.
      start = wall_time()
      call compute_esp_values(size(cavity,2),cavity,phi)
      call record_phase(4,start)
      if (.not.master) return
      density = quick_qm_struct%dense
      if (quick_method%unrst) density = density + quick_qm_struct%denseb
      start = wall_time()
      call solvation_density(grid%points,density,rho,rho_grad,ierr)
      call record_phase(5,start)
      if (ierr /= 0) call solvent_failure(ierr,'Cannot evaluate the ddX continuous electronic density.')
#if defined(CUDA) || defined(CUDA_MPIV)
      if (quick_method%bGPU) solvation_gpu_volume_density = solvation_gpu_volume_density + 1
#endif
      solvation_electron_count = sum(grid%weight*rho)
      allocate(psi(backend%nlm,natom),coulomb_matrix(nbasis,nbasis))
      start = wall_time()
      psi = 0.d0
      do i = 1,grid%npoints
         psi(:,grid%parent(i)) = psi(:,grid%parent(i)) - grid%weight(i)*rho(i)*scaled_y(:,i)
      enddo
      call record_phase(6,start)
      ierr = 0
      start = wall_time()
      call ddx_backend_solve(backend,phi,psi,solvation_energy,x,xi,ierr,message)
      call record_phase(7,start)
      if (ierr /= 0) call solvent_failure(ierr,message)
      start = wall_time()
      do i = 1,grid%npoints
         volume_potential(i) = -0.5d0*backend%scaling*dot_product(scaled_y(:,i),x(:,grid%parent(i)))
      enddo
      call solvation_local_matrix(grid%points,grid%weight*volume_potential,potential_matrix,ierr)
      call record_phase(8,start)
      if (ierr /= 0) call solvent_failure(ierr,'Cannot evaluate the ddX local reaction-potential AO matrix.')
#if defined(CUDA) || defined(CUDA_MPIV)
      if (quick_method%bGPU) solvation_gpu_volume_matrix = solvation_gpu_volume_matrix + 1
#endif
      start = wall_time()
      call solvation_coulomb_matrix(cavity,0.5d0*backend%scaling*xi,coulomb_matrix,ierr,message)
      call record_phase(9,start)
      if (ierr /= 0) call solvent_failure(ierr,message)
      potential_matrix = potential_matrix + coulomb_matrix
      solvation_updates = solvation_updates + 1
   end subroutine update_solvation

   ! Collective electronic-density interface for variational validation and
   ! host integrations. It does not change o, ob, Eel or perform an SCF.
   subroutine get_solvation_contributions(matrix,energy)
      implicit none
      double precision, intent(out) :: matrix(nbasis,nbasis), energy
      matrix = 0.d0
      energy = 0.d0
      if (.not.quick_method%ddx) return
      if (quick_method%ddx_epsilon == 1.d0) return
      call update_solvation()
      if (master) then
         matrix = potential_matrix
         energy = solvation_energy
      endif
   end subroutine get_solvation_contributions

   ! Called AFTER the ordinary operator has been saved and MPI-reduced.
   ! oSave remains the gas-phase incremental operator; the solvent sees
   ! full restored P, and its variational energy is added exactly once.
   subroutine add_solvation_operator()
      implicit none
      if (.not.quick_method%ddx) return
      if (quick_method%ddx_epsilon == 1.d0) return
      call update_solvation()
      if (.not.master) return
      quick_qm_struct%o = quick_qm_struct%o + potential_matrix
      if (quick_method%unrst) quick_qm_struct%ob = quick_qm_struct%ob + potential_matrix
      quick_qm_struct%Eel = quick_qm_struct%Eel + solvation_energy
   end subroutine add_solvation_operator

   subroutine add_solvation_gradient()
      implicit none
      double precision, allocatable :: field(:, :), grad(:, :), tmp(:, :), point_grad(:, :)
      double precision :: dp(3,natom), delta(3), radius, q, prefactor, start
      integer :: ierr, i, a
      character(len=512) :: message
      if (.not.quick_method%ddx) return
      if (quick_method%ddx_epsilon == 1.d0) return
      ! Refresh at the retained final density without another SCF. Its
      ! energy is not re-added and its AO matrix is not put into oSave.
      call update_solvation()
      allocate(field(3,size(cavity,2)))
      start = wall_time()
      call compute_efield_values(size(cavity,2),cavity,field)
      call record_phase(10,start)
      if (.not.master) return
      allocate(grad(3,natom),tmp(3,natom),point_grad(3,size(cavity,2)))
      ierr = 0
      start = wall_time()
      call ddx_backend_cavity_gradient(backend,field,grad,ierr,message)
      call record_phase(11,start)
      if (ierr /= 0) call solvent_failure(ierr,message)
      ! Positive-T convention: this is the derivative of the electronic
      ! Phi part only. Native ddX already includes cavity-node motion.
      start = wall_time()
      call solvation_coulomb_gradient(cavity,0.5d0*backend%scaling*xi,density,tmp,point_grad,ierr,message)
      call record_phase(12,start)
      if (ierr /= 0) call solvent_failure(ierr,message)
      grad = grad + tmp
      ! Nuclear Phi source derivative at fixed cavity coordinates.
      do i = 1,size(xi)
         q = -0.5d0*backend%scaling*xi(i)
         do a = 1,natom
            delta = xyz(:,a)-cavity(:,i)
            radius = norm2(delta)
            if (radius == 0.d0) call solvent_failure(53,'A ddX cavity node coincides with a nucleus.')
            grad(:,a) = grad(:,a)-q*quick_molspec%chg(a)*delta/radius**3
         enddo
      enddo
      ! Parent-following volume nodes make their solid harmonics constant.
      start = wall_time()
      call solvation_local_gradient(grid%points,grid%parent,grid%weight*volume_potential,density,tmp,ierr)
      call record_phase(13,start)
      if (ierr /= 0) call solvent_failure(ierr,'Cannot differentiate the ddX volume AO integrals.')
#if defined(CUDA) || defined(CUDA_MPIV)
      if (quick_method%bGPU) solvation_gpu_volume_gradient = solvation_gpu_volume_gradient + 1
#endif
      grad = grad + tmp
      start = wall_time()
      do i = 1,grid%npoints
         call solvation_grid_partition_derivative(grid,natom,xyz,i,dp,ierr)
         if (ierr /= 0) call solvent_failure(ierr,'Cannot differentiate the ddX volume partition.')
         prefactor = grid%base_weight(i)*rho(i)*volume_potential(i)
         grad = grad + prefactor*dp
      enddo
      call record_phase(14,start)
      quick_qm_struct%gradient = quick_qm_struct%gradient + reshape(grad,[3*natom])
   end subroutine add_solvation_gradient

   subroutine solvent_failure(ierr,message)
      use quick_files_module, only: ioutfile
      implicit none
      integer, intent(in) :: ierr
      character(len=*), intent(in) :: message
      integer :: status
      if (master) then
         call PrtErr(ioutfile,message)
         write(ioutfile,'(A,I0)') ' DDX error code: ',ierr
      endif
      status = max(1,abs(ierr))
#ifdef MPIV
      ! A failed master-only solver cannot enter collective cleanup while
      ! the other job ranks wait for the ordinary SCF/gradient reduction.
      if (bMPI) call MPI_ABORT(quick_comm,status,quick_mpi_error)
#endif
      call quick_exit(OUTFILEHANDLE,status)
   end subroutine solvent_failure

   subroutine cavity_radii(radii)
      implicit none
      double precision, intent(out) :: radii(natom)
      double precision :: bondi(118)
      integer :: a,z
      ! Same published Bondi values as the existing MKS surface builder;
      ! copied data avoids changing its private implementation or defaults.
      data bondi/ &
    1.20d0,                                                                                                                1.40d0, &
    1.82d0, 0.0d0,                                                                       0.0d0,1.70d0,1.55d0,1.52d0,1.47d0,1.54d0, &
    2.27d0,1.73d0,                                                                       0.0d0,2.10d0,1.80d0,1.80d0,1.75d0,1.88d0, &
    2.75d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0,1.63d0,1.40d0,1.39d0,1.87d0, 0.0d0,1.85d0,1.90d0,1.85d0,2.02d0, &
     0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0,1.63d0,1.72d0,1.58d0,1.93d0,2.17d0, 0.0d0,2.06d0,1.98d0,2.16d0, &
     0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, &
     0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0,1.75d0,1.66d0,1.55d0,1.96d0,2.02d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, &
     0.0d0,1.86d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, &
     0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0/
      do a = 1,natom
         z = nint(quick_molspec%chg(a))
         if (z < 1 .or. z > size(bondi)) call solvent_failure(53,'Invalid ddX cavity element.')
         if (bondi(z) <= 0.d0) call solvent_failure(53,'No Bondi radius is available for a ddX cavity element.')
         radii(a) = bondi(z)*quick_method%ddx_radius_scale/BOHRS_TO_A
      enddo
   end subroutine cavity_radii
end module quick_solvation_module
