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
! Small PBE0-D3/MB-pol water-dimer scan inspired by Lambros et al.     !
! Figure 1. Monomer A is kept as the donor water from the MBX          !
! 001_mbpol example; monomer B is translated along the A--B O--O axis. !
! The program prints QUICK-only dimer energies and QUICK-MBX hybrid    !
! energies on the same XYZ points for post-processing with MBX.        !
!_____________________________________________________________________!

program quick_mbx_dimer_scan

   use quick_api_module, only: setQuickJob, getQuickEnergy, deleteQuickJob
   use quick_api_module, only: setQuickMBXWaterSystem, clearQuickMBXSystem

   implicit none

   integer, parameter :: natom_water = 3
   integer, parameter :: natom_dimer = 6
   integer, parameter :: nscan = 20
   double precision, parameter :: distances(nscan) = (/ &
      2.50d0, 2.60d0, 2.70d0, 2.80d0, 2.90d0, &
      3.00d0, 3.10d0, 3.20d0, 3.30d0, 3.40d0, &
      3.50d0, 3.60d0, 3.70d0, 3.80d0, 3.90d0, &
      4.00d0, 4.20d0, 4.40d0, 4.70d0, 5.00d0/)

   integer :: ierr, i, idx, nrun
   integer :: water_atomic_numbers(natom_water)
   integer :: dimer_atomic_numbers(natom_dimer)
   double precision :: monomer_a_xyz(3,natom_water)
   double precision :: monomer_b_ref_xyz(3,natom_water)
   double precision :: monomer_b_xyz(3,natom_water)
   double precision :: dimer_xyz(3,natom_dimer)
   double precision :: qm_a_au, qm_b_au, qm_dimer_au, hybrid_ab_au
   double precision, allocatable :: no_point_charges(:,:)
   character(len=256) :: keywd_qm, keywd_mbx

   ierr = 0
   water_atomic_numbers = (/8,1,1/)
   dimer_atomic_numbers = (/8,1,1,8,1,1/)

   monomer_a_xyz(:,1) = (/-1.58972425d0, 1.04337922d0, -0.08780840d0/)
   monomer_a_xyz(:,2) = (/-0.63591971d0, 0.97898520d0,  0.00000000d0/)
   monomer_a_xyz(:,3) = (/-1.90066280d0, 1.74501050d0, -0.66454990d0/)

   monomer_b_ref_xyz(:,1) = (/ 1.64924507d0, 1.08594656d0,  0.00000000d0/)
   monomer_b_ref_xyz(:,2) = (/ 2.60878026d0, 1.09587704d0, -0.02817115d0/)
   monomer_b_ref_xyz(:,3) = (/ 1.33830653d0, 1.78757784d0,  0.57674150d0/)

   keywd_qm = 'DFT PBE0 D3 BASIS=AUG-CC-PVTZ CUTOFF=1.0D-9 XCCUTOFF=1.0D-8 BASISCUTOFF=1.0D-8 DENSERMS=1.0D-6'
   keywd_mbx = trim(keywd_qm)//' MBX_QMMM'
   nrun = get_scan_limit(nscan)

   allocate(no_point_charges(4,0), stat=ierr)
   call stop_on_error(ierr)

   qm_a_au = run_qm_water('quick_mbx_scan_qm_monomer_a', monomer_a_xyz, keywd_qm, no_point_charges)
   qm_b_au = run_qm_water('quick_mbx_scan_qm_monomer_b', monomer_b_ref_xyz, keywd_qm, no_point_charges)

   write(*,'(A)') '# QUICK-MBX Lambros-style A/B water-dimer scan'
   write(*,'(A)') '# method=PBE0-D3/AUG-CC-PVTZ for QUICK, MB-pol for MBX'
   write(*,'(A)') '# monomer A is QUICK in the hybrid; monomer B is MBX/MB-pol in the hybrid'
   write(*,'(A,I0,A,I0)') '# running ', nrun, ' of ', nscan, ' configured scan points'
   write(*,'(A)') 'distance_ang,qm_dimer_au,qm_a_iso_au,qm_b_iso_au,hybrid_ab_au'

   do i = 1, nrun
      idx = scan_index(i, nrun, nscan)
      call make_monomer_b_at_distance(distances(idx), monomer_a_xyz, monomer_b_ref_xyz, monomer_b_xyz)
      dimer_xyz(:,1:natom_water) = monomer_a_xyz(:,1:natom_water)
      dimer_xyz(:,natom_water+1:natom_dimer) = monomer_b_xyz(:,1:natom_water)

      qm_dimer_au = run_qm_dimer('quick_mbx_scan_qm_dimer', idx, dimer_xyz, keywd_qm, no_point_charges)
      hybrid_ab_au = run_qm_mbx_point('quick_mbx_scan_hybrid_ab', idx, monomer_a_xyz, monomer_b_xyz, keywd_mbx, &
         no_point_charges)

      write(*,'(F8.4,",",F24.16,",",F24.16,",",F24.16,",",F24.16)') distances(idx), qm_dimer_au, qm_a_au, &
         qm_b_au, hybrid_ab_au
   enddo

   if (allocated(no_point_charges)) deallocate(no_point_charges)

contains

   function run_qm_water(fname, qm_xyz_ang, keywd, no_point_charges) result(energy)
      implicit none

      character(len=*), intent(in) :: fname, keywd
      double precision, intent(in) :: qm_xyz_ang(3,natom_water)
      double precision, intent(in) :: no_point_charges(:,:)
      double precision :: energy

      call setQuickJob(fname, keywd, natom_water, water_atomic_numbers, .false., ierr)
      call stop_on_error(ierr)
      call getQuickEnergy(qm_xyz_ang, 0, no_point_charges, energy, ierr)
      call stop_on_error(ierr)
      call deleteQuickJob(ierr)
      call stop_on_error(ierr)
   end function run_qm_water

   function run_qm_dimer(prefix, idx, qm_xyz_ang, keywd, no_point_charges) result(energy)
      implicit none

      character(len=*), intent(in) :: prefix, keywd
      integer, intent(in) :: idx
      double precision, intent(in) :: qm_xyz_ang(3,natom_dimer)
      double precision, intent(in) :: no_point_charges(:,:)
      double precision :: energy
      character(len=80) :: fname

      write(fname,'(A,"_",I0)') trim(prefix), idx
      call setQuickJob(fname, keywd, natom_dimer, dimer_atomic_numbers, .false., ierr)
      call stop_on_error(ierr)
      call getQuickEnergy(qm_xyz_ang, 0, no_point_charges, energy, ierr)
      call stop_on_error(ierr)
      call deleteQuickJob(ierr)
      call stop_on_error(ierr)
   end function run_qm_dimer

   function run_qm_mbx_point(prefix, idx, qm_xyz_ang, mbx_xyz_ang, keywd, no_point_charges) result(energy)
      implicit none

      character(len=*), intent(in) :: prefix, keywd
      integer, intent(in) :: idx
      double precision, intent(in) :: qm_xyz_ang(3,natom_water)
      double precision, intent(in) :: mbx_xyz_ang(3,natom_water)
      double precision, intent(in) :: no_point_charges(:,:)
      double precision :: energy
      character(len=80) :: fname

      write(fname,'(A,"_",I0)') trim(prefix), idx
      call setQuickJob(fname, keywd, natom_water, water_atomic_numbers, .false., ierr)
      call stop_on_error(ierr)
      call setQuickMBXWaterSystem(1, mbx_xyz_ang, 'mbx.json', ierr)
      call stop_on_error(ierr)
      call getQuickEnergy(qm_xyz_ang, 0, no_point_charges, energy, ierr)
      call stop_on_error(ierr)
      call clearQuickMBXSystem(ierr)
      call stop_on_error(ierr)
      call deleteQuickJob(ierr)
      call stop_on_error(ierr)
   end function run_qm_mbx_point

   subroutine make_monomer_b_at_distance(roo, monomer_a_xyz, monomer_b_ref_xyz, monomer_b_xyz)
      implicit none

      double precision, intent(in) :: roo
      double precision, intent(in) :: monomer_a_xyz(3,natom_water)
      double precision, intent(in) :: monomer_b_ref_xyz(3,natom_water)
      double precision, intent(out) :: monomer_b_xyz(3,natom_water)
      double precision :: oo_vec(3), unit_vec(3), shift(3), target_o(3), norm
      integer :: i

      oo_vec = monomer_b_ref_xyz(:,1) - monomer_a_xyz(:,1)
      norm = sqrt(sum(oo_vec*oo_vec))
      unit_vec = oo_vec/norm
      target_o = monomer_a_xyz(:,1) + roo*unit_vec
      shift = target_o - monomer_b_ref_xyz(:,1)

      do i = 1, natom_water
         monomer_b_xyz(:,i) = monomer_b_ref_xyz(:,i) + shift
      enddo
   end subroutine make_monomer_b_at_distance

   subroutine stop_on_error(ierr_in)
      implicit none

      integer, intent(in) :: ierr_in

      if (ierr_in /= 0) then
         write(*,'(A,1X,I0)') 'QUICK_MBX_DIMER_SCAN_ERROR', ierr_in
         stop 1
      endif
   end subroutine stop_on_error

   function get_scan_limit(default_nscan) result(nlimit)
      implicit none

      integer, intent(in) :: default_nscan
      integer :: nlimit, env_status, parsed
      character(len=32) :: env_value

      nlimit = default_nscan
      call get_environment_variable('QUICK_MBX_DIMER_SCAN_NPOINTS', env_value, status=env_status)
      if (env_status == 0) then
         read(env_value, *, iostat=env_status) parsed
         if (env_status == 0) nlimit = max(1, min(default_nscan, parsed))
      endif
   end function get_scan_limit

   function scan_index(irun, nlimit, default_nscan) result(idx)
      implicit none

      integer, intent(in) :: irun, nlimit, default_nscan
      integer :: idx

      idx = irun
      if (nlimit == 3 .and. default_nscan >= 20) then
         select case (irun)
         case (1)
            idx = 1
         case (2)
            idx = 7
         case default
            idx = default_nscan
         end select
      endif
   end function scan_index

end program quick_mbx_dimer_scan
