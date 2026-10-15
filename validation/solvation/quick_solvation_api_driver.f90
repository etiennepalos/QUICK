#include "util.fh"
!
! (C) Copyright 2026 QUICK contributors. All rights reserved.
! This Source Code Form is subject to the Mozilla Public License, v. 2.0.
!
program quick_solvation_api_driver
   use iso_c_binding, only: c_int64_t
   use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
   use quick_api_module, only: setQuickJob, getQuickEnergy, getQuickEnergyGradients, deleteQuickJob, quick_api
   use quick_method_module, only: quick_method
   use quick_molspec_module, only: quick_molspec
   use quick_basis_module, only: nbasis
   use quick_calculated_module, only: quick_qm_struct
   use quick_solvation_module, only: get_solvation_contributions, solvation_energy, solvation_electron_count, &
      solvation_updates, solvation_gpu_volume_density, solvation_gpu_volume_matrix, solvation_gpu_volume_gradient, &
      solvation_wall_seconds, solvation_timing_labels
   use quick_solvation_coulomb_module, only: solvation_coulomb_cache_info
#ifdef MPIV
   use mpi
   use quick_api_module, only: setQuickMPI
   use quick_mpi_module, only: quick_comm, quick_comm_rank, quick_mpi_error
#endif
   implicit none

   character(len=300) :: filename, raw_keyword
   character(len=256) :: keyword
   character(len=80) :: jobname
   character(len=32) :: mode
   integer :: ierr, cleanup_error, unit, output_unit, natoms, atom, nelectrons
   integer(c_int64_t) :: cache_data(7)
   integer, allocatable :: atomic_numbers(:)
   double precision, allocatable :: coords(:,:), gradients(:,:)
   double precision :: energy, point_charges(4,0), charge_gradients(3,0)
   logical :: is_master, want_gradient, converged

   ierr = 0
   is_master = .true.
   output_unit = 0
#ifdef MPIV
   call MPI_INIT(quick_mpi_error)
   call setQuickMPI(MPI_COMM_WORLD,ierr)
   is_master = (quick_comm_rank == 0)
#endif
   if (command_argument_count() /= 2) then
      if (is_master) write(*,'(A)') 'Usage: solvation-api-driver energy|gradient|frozen input.dat'
      ierr = 49
   endif
   if (ierr == 0) then
      call get_command_argument(1,mode)
      call get_command_argument(2,filename)
      if (trim(mode) /= 'energy' .and. trim(mode) /= 'gradient' .and. trim(mode) /= 'frozen') ierr = 49
   endif
   if (ierr == 0) then
      open(newunit=unit,file=trim(filename),status='old',action='read',iostat=ierr)
      if (ierr == 0) then
         read(unit,'(A)',iostat=ierr) raw_keyword
         if (ierr == 0) then
            if (len_trim(raw_keyword) > len(keyword)) then
               ierr = 49
            else
               keyword = raw_keyword
            endif
         endif
         if (ierr == 0) read(unit,*,iostat=ierr) natoms
         if (ierr == 0) then
            if (natoms < 1) ierr = 49
         endif
         if (ierr == 0) allocate(atomic_numbers(natoms),coords(3,natoms),gradients(3,natoms),stat=ierr)
         if (ierr == 0) then
            do atom=1,natoms
               read(unit,*,iostat=ierr) atomic_numbers(atom),coords(:,atom)
               if (ierr /= 0) exit
            enddo
         endif
         close(unit)
      endif
   endif
   call synchronize_error(ierr)
   if (ierr == 0) then
      want_gradient = (trim(mode) == 'gradient')
      jobname = 'continuum_api'
      call setQuickJob(jobname,keyword,natoms,atomic_numbers,.false.,ierr)
      call synchronize_error(ierr)
      if (ierr == 0) then
         if (want_gradient) then
            call getQuickEnergyGradients(coords,0,point_charges,energy,gradients,charge_gradients,ierr)
         else
            call getQuickEnergy(coords,0,point_charges,energy,ierr)
         endif
         converged = merge(quick_method%uscf_conv,quick_method%scf_conv,quick_method%unrst)
         if (ierr == 0 .and. (.not.converged .or. .not.ieee_is_finite(energy))) ierr = 53
         if (ierr == 0 .and. want_gradient) then
            if (.not.all(ieee_is_finite(gradients))) ierr = 53
         endif
         call synchronize_error(ierr)
         if (ierr == 0 .and. is_master) then
            open(newunit=output_unit,file='result.dat',status='new',action='write',iostat=ierr)
            if (ierr == 0) then
               nelectrons = quick_molspec%nelec
               if (quick_method%unrst) nelectrons = nelectrons + quick_molspec%nelecb
               write(output_unit,'(A,1X,ES26.17E3)') 'energy',energy
               write(output_unit,'(A,1X,L1)') 'scf_converged',converged
               write(output_unit,'(A,1X,I0)') 'basis_functions',nbasis
               write(output_unit,'(A,1X,I0)') 'nominal_electrons',nelectrons
               write(output_unit,'(A,1X,ES26.17E3)') 'volume_electrons',solvation_electron_count
               write(output_unit,'(A,1X,ES26.17E3)') 'solvation_energy',solvation_energy
               write(output_unit,'(A,1X,I0)') 'solvation_updates',solvation_updates
               write(output_unit,'(A,3(1X,I0))') 'gpu_volume_counters',solvation_gpu_volume_density, &
                  solvation_gpu_volume_matrix,solvation_gpu_volume_gradient
               write(output_unit,'(A,1X,I0)') 'api_steps',quick_api%step
               call solvation_coulomb_cache_info(cache_data(1),cache_data(2),cache_data(3), &
                  cache_data(4),cache_data(5),cache_data(6),cache_data(7))
               write(output_unit,'(A,7(1X,I0))') 'gpu_coulomb_cache',cache_data
               do atom=1,size(solvation_wall_seconds)
                  write(output_unit,'(3A,1X,ES26.17E3)') 'phase_',trim(solvation_timing_labels(atom)), &
                     '_seconds',solvation_wall_seconds(atom)
               enddo
               if (want_gradient) then
                  do atom=1,natoms
                     write(output_unit,'(A,1X,I0,3(1X,ES26.17E3))') 'gradient',atom,gradients(:,atom)
                  enddo
               endif
            endif
         endif
         call synchronize_error(ierr)
         if (ierr == 0 .and. trim(mode) == 'frozen') call check_frozen_density(output_unit,ierr)
         if (is_master .and. output_unit /= 0) close(output_unit)
      endif
      cleanup_error = 0
      call deleteQuickJob(cleanup_error)
      if (ierr == 0) ierr = cleanup_error
   endif
   call synchronize_error(ierr)
   if (ierr /= 0 .and. is_master) write(*,'(A,I0)') 'Solvation API driver failed, ierr=',ierr
#ifdef MPIV
   call MPI_FINALIZE(quick_mpi_error)
#endif
   if (ierr /= 0) stop 1

contains

   subroutine check_frozen_density(out,status)
      implicit none
      integer, intent(in) :: out
      integer, intent(inout) :: status
      double precision, parameter :: steps(3) = (/1.d-3,3.d-4,1.d-4/)
      double precision, allocatable :: saved_alpha(:,:), saved_beta(:,:), saved_o(:,:), saved_ob(:,:)
      double precision, allocatable :: base_matrix(:,:), matrix(:,:), perturbation(:,:)
      double precision :: saved_eel, saved_etot, base_energy, plus_energy, minus_energy, restored_energy
      double precision :: expected, derivative, norm, fock_error, state_error
      integer :: saved_step, i, j, direction, spin, step

      saved_alpha = quick_qm_struct%dense
      saved_o = quick_qm_struct%o
      if (quick_method%unrst) then
         saved_beta = quick_qm_struct%denseb
         saved_ob = quick_qm_struct%ob
      endif
      saved_eel = quick_qm_struct%Eel
      saved_etot = quick_qm_struct%Etot
      saved_step = quick_api%step
      allocate(base_matrix(nbasis,nbasis),matrix(nbasis,nbasis),perturbation(nbasis,nbasis))
      call get_solvation_contributions(base_matrix,base_energy)
      if (is_master) then
         write(out,'(A,1X,ES26.17E3)') 'frozen_base_energy',base_energy
         write(out,'(A,1X,ES26.17E3)') 'potential_symmetry_error',maxval(abs(base_matrix-transpose(base_matrix)))
      endif
      do spin=1,merge(2,1,quick_method%unrst)
         do direction=1,2
            perturbation = 0.d0
            do i=1,nbasis
               do j=1,nbasis
                  if (direction == 1) then
                     if (i == j) perturbation(i,j) = 1.d0
                  else
                     perturbation(i,j) = 0.5d0*(cos(dble(i+2*j))+cos(dble(j+2*i)))
                  endif
               enddo
            enddo
            norm = sqrt(sum(perturbation*perturbation))
            perturbation = perturbation/norm
            if (is_master) expected = sum(base_matrix*perturbation)
            do step=1,size(steps)
               quick_qm_struct%dense = saved_alpha
               if (quick_method%unrst) quick_qm_struct%denseb = saved_beta
               if (spin == 1) then
                  quick_qm_struct%dense = saved_alpha + steps(step)*perturbation
               else
                  quick_qm_struct%denseb = saved_beta + steps(step)*perturbation
               endif
               call get_solvation_contributions(matrix,plus_energy)
               quick_qm_struct%dense = saved_alpha
               if (quick_method%unrst) quick_qm_struct%denseb = saved_beta
               if (spin == 1) then
                  quick_qm_struct%dense = saved_alpha - steps(step)*perturbation
               else
                  quick_qm_struct%denseb = saved_beta - steps(step)*perturbation
               endif
               call get_solvation_contributions(matrix,minus_energy)
               if (is_master) then
                  derivative = (plus_energy-minus_energy)/(2.d0*steps(step))
                  write(out,'(A,2(1X,I0),3(1X,ES26.17E3))') 'density_derivative',spin,direction, &
                     steps(step),derivative,expected
                  if (.not.ieee_is_finite(derivative) .or. .not.ieee_is_finite(expected)) status = 53
               endif
            enddo
         enddo
      enddo
      quick_qm_struct%dense = saved_alpha
      if (quick_method%unrst) quick_qm_struct%denseb = saved_beta
      call get_solvation_contributions(matrix,restored_energy)
      if (is_master) then
         fock_error = maxval(abs(quick_qm_struct%o-saved_o))
         if (quick_method%unrst) fock_error = max(fock_error,maxval(abs(quick_qm_struct%ob-saved_ob)))
         state_error = max(abs(quick_qm_struct%Eel-saved_eel),abs(quick_qm_struct%Etot-saved_etot))
         write(out,'(A,1X,ES26.17E3)') 'density_restored_energy_error',abs(restored_energy-base_energy)
         write(out,'(A,1X,ES26.17E3)') 'density_restored_potential_error',maxval(abs(matrix-base_matrix))
         write(out,'(A,1X,ES26.17E3)') 'fock_unchanged_error',fock_error
         write(out,'(A,1X,ES26.17E3)') 'energy_state_unchanged_error',state_error
         write(out,'(A,2(1X,I0))') 'frozen_api_steps',saved_step,quick_api%step
         write(out,'(A,1X,I0)') 'final_solvation_updates',solvation_updates
      endif
      call synchronize_error(status)
   end subroutine check_frozen_density

   subroutine synchronize_error(status)
      implicit none
      integer, intent(inout) :: status
#ifdef MPIV
      integer :: global_status
#endif
      status = abs(status)
#ifdef MPIV
      call MPI_ALLREDUCE(status,global_status,1,MPI_INTEGER,MPI_MAX,quick_comm,quick_mpi_error)
      status = global_status
#endif
   end subroutine synchronize_error

end program quick_solvation_api_driver
