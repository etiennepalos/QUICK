#include "util.fh"
!---------------------------------------------------------------------!
! Copyright (C) 2026 QUICK contributors                               !
! All rights reserved.                                                !
!                                                                     !
! This Source Code Form is subject to the terms of the Mozilla Public !
! License, v. 2.0. If a copy of the MPL was not distributed with this !
! file, You can obtain one at http://mozilla.org/MPL/2.0/.            !
!_____________________________________________________________________!

!---------------------------------------------------------------------!
! Optional QUICK-MBX API smoke test.                                  !
!                                                                     !
! This target is built only when QUICK is configured with -DMBX=TRUE. !
! It exercises a small CPU energy-only QM/MB-pol coupling path through !
! the public QUICK API.  The runtime working directory must contain    !
! an MBX JSON file named mbx.json.                                    !
!_____________________________________________________________________!

program quick_mbx_api_smoke

   use quick_api_module, only: setQuickJob, getQuickEnergy, deleteQuickJob
   use quick_api_module, only: setQuickMBXWaterSystem, clearQuickMBXSystem
#ifdef MPIV
   use mpi
   use quick_api_module, only: setQuickMPI
#endif

   implicit none

   integer, parameter :: natom_qm = 3
   integer, parameter :: natom_mbx_one = 3
   integer, parameter :: natom_mbx_two = 6

   integer :: ierr
   integer :: atomic_numbers(natom_qm)
   double precision :: qm_xyz_ang(3,natom_qm)
   double precision :: mbx_xyz_one(3,natom_mbx_one)
   double precision :: mbx_xyz_two(3,natom_mbx_two)
#ifdef MPIV
   integer :: mpierror, mpirank, mpisize
   logical :: master
#else
   logical :: master
#endif

   ierr = 0
   master = .true.

#ifdef MPIV
   call MPI_INIT(mpierror)
   call MPI_COMM_RANK(MPI_COMM_WORLD,mpirank,mpierror)
   call MPI_COMM_SIZE(MPI_COMM_WORLD,mpisize,mpierror)
   master = (mpirank == 0)
   call setQuickMPI(mpirank,mpisize,ierr)
   call stop_on_error(ierr)
#endif

   atomic_numbers = (/8,1,1/)

   qm_xyz_ang(:,1) = (/-1.58972425d0, 1.04337922d0, -0.08780840d0/)
   qm_xyz_ang(:,2) = (/-0.63591971d0, 0.97898520d0,  0.00000000d0/)
   qm_xyz_ang(:,3) = (/-1.90066280d0, 1.74501050d0, -0.66454990d0/)

   mbx_xyz_one(:,1) = (/ 1.64924507d0, 1.08594656d0,  0.00000000d0/)
   mbx_xyz_one(:,2) = (/ 2.60878026d0, 1.09587704d0, -0.02817115d0/)
   mbx_xyz_one(:,3) = (/ 1.33830653d0, 1.78757784d0,  0.57674150d0/)

   mbx_xyz_two(:,1:3) = mbx_xyz_one(:,1:3)
   mbx_xyz_two(:,4) = (/-0.61315209d0, 2.46976336d0,  2.07005086d0/)
   mbx_xyz_two(:,5) = (/ 0.34684791d0, 2.46976336d0,  2.07005086d0/)
   mbx_xyz_two(:,6) = (/-0.93360667d0, 3.37469919d0,  2.07005086d0/)

   call run_smoke_case('QUICK_MBX_SMOKE_MONOMER_ENERGY_AU',1,mbx_xyz_one)
   call run_smoke_case('QUICK_MBX_SMOKE_DIMER_ENERGY_AU',2,mbx_xyz_two)

#ifdef MPIV
   call MPI_FINALIZE(mpierror)
#endif

contains

   subroutine run_smoke_case(label,nwaters,mbx_xyz_ang)
      implicit none

      character(len=*), intent(in) :: label
      integer, intent(in) :: nwaters
      double precision, intent(in) :: mbx_xyz_ang(:,:)

      double precision :: energy
      double precision, allocatable :: no_point_charges(:,:)
      character(len=80) :: fname
      character(len=256) :: keywd

      if (nwaters == 1) then
         fname = 'quick_mbx_api_smoke_1w'
      else
         fname = 'quick_mbx_api_smoke_2w'
      endif
      keywd = 'HF BASIS=STO-3G CUTOFF=1.0D-10 DENSERMS=1.0D-6 MBX_QMMM'

      call setQuickJob(fname, keywd, natom_qm, atomic_numbers, .false., ierr)
      call stop_on_error(ierr)

      call setQuickMBXWaterSystem(nwaters, mbx_xyz_ang, 'mbx.json', ierr)
      call stop_on_error(ierr)

      allocate(no_point_charges(4,0), stat=ierr)
      call stop_on_error(ierr)

      call getQuickEnergy(qm_xyz_ang, 0, no_point_charges, energy, ierr)
      call stop_on_error(ierr)

      if (master) then
         write(*,'(A,1X,F24.16)') trim(label), energy
      endif

      if (allocated(no_point_charges)) deallocate(no_point_charges)
      call clearQuickMBXSystem(ierr)
      call stop_on_error(ierr)
      call deleteQuickJob(ierr)
      call stop_on_error(ierr)
   end subroutine run_smoke_case

   subroutine stop_on_error(ierr_in)
      implicit none

      integer, intent(in) :: ierr_in

      if (ierr_in /= 0) then
         write(*,'(A,1X,I0)') 'QUICK_MBX_SMOKE_ERROR', ierr_in
#ifdef MPIV
         call MPI_ABORT(MPI_COMM_WORLD,ierr_in,mpierror)
#else
         stop 1
#endif
      endif
   end subroutine stop_on_error

end program quick_mbx_api_smoke
