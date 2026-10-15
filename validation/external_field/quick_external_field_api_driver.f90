#include "util.fh"
!
! (C) Copyright 2026 QUICK contributors. All rights reserved.
! This Source Code Form is subject to the terms of the Mozilla Public
! License, v. 2.0. If a copy of the MPL was not distributed with this
! file, You can obtain one at http://mozilla.org/MPL/2.0/.
!
program quick_external_field_api_driver
   use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
   use quick_api_module, only: setQuickJob, getQuickEnergy, getQuickEnergyGradients, deleteQuickJob
   use quick_method_module, only: quick_method
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
   integer :: ierr, cleanup_error, unit, natoms, ncharges, atom, charge
   integer, allocatable :: atomic_numbers(:)
   double precision, allocatable :: coords(:,:), charges(:,:), gradients(:,:), charge_gradients(:,:)
   double precision :: energy
   logical :: is_master, want_gradient, converged

   ierr = 0
   is_master = .true.
#ifdef MPIV
   call MPI_INIT(quick_mpi_error)
   call setQuickMPI(MPI_COMM_WORLD,ierr)
   is_master = (quick_comm_rank == 0)
#endif
   if (command_argument_count() /= 2) then
      if (is_master) write(*,'(A)') 'Usage: field-api-driver energy|gradient input.dat'
      ierr = 49
   endif
   if (ierr == 0) then
      call get_command_argument(1,mode)
      call get_command_argument(2,filename)
      if (trim(mode) /= 'energy' .and. trim(mode) /= 'gradient') ierr = 49
   endif
   if (ierr == 0) then
      ! Read-only input is identical on every participating rank. Only the
      ! QUICK master writes the result, after the collective API returns.
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
         if (ierr == 0) read(unit,*,iostat=ierr) natoms,ncharges
         if (ierr == 0) then
            if (natoms < 1 .or. ncharges < 1) ierr = 49
         endif
         if (ierr == 0) then
            allocate(atomic_numbers(natoms),coords(3,natoms),charges(4,ncharges), &
               gradients(3,natoms),charge_gradients(3,ncharges),stat=ierr)
         endif
         if (ierr == 0) then
            do atom=1,natoms
               read(unit,*,iostat=ierr) atomic_numbers(atom),coords(:,atom)
               if (ierr /= 0) exit
            enddo
         endif
         if (ierr == 0) then
            do charge=1,ncharges
               read(unit,*,iostat=ierr) charges(:,charge)
               if (ierr /= 0) exit
            enddo
         endif
         close(unit)
      endif
   endif
   call synchronize_error(ierr)
   if (ierr == 0) then
      want_gradient = (trim(mode) == 'gradient')
      jobname = 'external_field_api'
      call setQuickJob(jobname,keyword,natoms,atomic_numbers,.false.,ierr)
      call synchronize_error(ierr)
      if (ierr == 0) then
         if (want_gradient) then
            call getQuickEnergyGradients(coords,ncharges,charges,energy,gradients,charge_gradients,ierr)
         else
            call getQuickEnergy(coords,ncharges,charges,energy,ierr)
         endif
         converged = merge(quick_method%uscf_conv,quick_method%scf_conv,quick_method%unrst)
         if (ierr == 0 .and. (.not. converged .or. .not. ieee_is_finite(energy))) ierr = 53
         if (ierr == 0 .and. want_gradient) then
            if (.not. all(ieee_is_finite(gradients)) .or. .not. all(ieee_is_finite(charge_gradients))) ierr = 53
         endif
         call synchronize_error(ierr)
         if (ierr == 0 .and. is_master) then
            open(newunit=unit,file='result.dat',status='new',action='write',iostat=ierr)
            if (ierr == 0) then
               write(unit,'(A,1X,ES26.17E3)') 'energy',energy
               write(unit,'(A,1X,L1)') 'scf_converged',converged
               if (want_gradient) then
                  do atom=1,natoms
                     write(unit,'(A,1X,I0,3(1X,ES26.17E3))') 'qm_gradient',atom,gradients(:,atom)
                  enddo
                  do charge=1,ncharges
                     write(unit,'(A,1X,I0,3(1X,ES26.17E3))') 'charge_gradient',charge,charge_gradients(:,charge)
                  enddo
               endif
               close(unit)
            endif
         endif
         call synchronize_error(ierr)
      endif
      cleanup_error = 0
      call deleteQuickJob(cleanup_error)
      if (ierr == 0) ierr = cleanup_error
   endif
   call synchronize_error(ierr)
   if (ierr /= 0 .and. is_master) write(*,'(A,I0)') 'Field API driver failed, ierr=',ierr
#ifdef MPIV
   call MPI_FINALIZE(quick_mpi_error)
#endif
   if (ierr /= 0) stop 1

contains

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

end program quick_external_field_api_driver
