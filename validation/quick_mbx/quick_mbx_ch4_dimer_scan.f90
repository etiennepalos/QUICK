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
! Small PBE0-D3/MB-nrg methane-dimer scan using the generic QUICK-MBX  !
! API hook. Monomer A is treated with QUICK; monomer B is treated with !
! the MBX ch4 model.  The scan translates monomer B along the C--C axis!
! from the MBX 040_ch4-ch4_mb-nrg_2bnb example.                       !
!_____________________________________________________________________!

program quick_mbx_ch4_dimer_scan

   use quick_api_module, only: setQuickJob, getQuickEnergy, deleteQuickJob
   use quick_api_module, only: setQuickMBXSystem, clearQuickMBXSystem

   implicit none

   integer, parameter :: natom_ch4 = 5
   integer, parameter :: natom_dimer = 10
   integer, parameter :: nscan = 20
   double precision, parameter :: distances(nscan) = (/ &
      3.20d0, 3.30d0, 3.40d0, 3.50d0, 3.60d0, &
      3.70d0, 3.80d0, 3.90d0, 4.00d0, 4.10d0, &
      4.25d0, 4.40d0, 4.60d0, 4.80d0, 5.00d0, &
      5.25d0, 5.50d0, 5.75d0, 6.00d0, 6.50d0/)

   integer :: ierr, i, idx, nrun
   integer :: ch4_atomic_numbers(natom_ch4)
   integer :: dimer_atomic_numbers(natom_dimer)
   integer :: nat_monomers(1)
   double precision :: monomer_a_xyz(3,natom_ch4)
   double precision :: monomer_b_ref_xyz(3,natom_ch4)
   double precision :: monomer_b_xyz(3,natom_ch4)
   double precision :: dimer_xyz(3,natom_dimer)
   double precision :: qm_a_au, qm_b_au, qm_dimer_au, hybrid_ab_au
   double precision, allocatable :: no_point_charges(:,:)
   character(len=256) :: keywd_qm, keywd_mbx
   character(len=1) :: atom_names(natom_ch4)
   character(len=3) :: monomer_names(1)

   ierr = 0
   ch4_atomic_numbers = (/6,1,1,1,1/)
   dimer_atomic_numbers = (/6,1,1,1,1,6,1,1,1,1/)
   nat_monomers = (/natom_ch4/)
   atom_names = (/'C','H','H','H','H'/)
   monomer_names = (/'ch4'/)

   monomer_a_xyz(:,1) = (/ 0.1713338779d0,  0.0402580564d0,  0.2343860323d0/)
   monomer_a_xyz(:,2) = (/-0.0070796542d0, -0.8125863746d0,  0.8833225480d0/)
   monomer_a_xyz(:,3) = (/ 0.5295486445d0,  0.8773662734d0,  0.8265597026d0/)
   monomer_a_xyz(:,4) = (/ 0.9203628978d0, -0.2207373935d0, -0.5083610174d0/)
   monomer_a_xyz(:,5) = (/-0.7529439533d0,  0.3158362319d0, -0.2656777218d0/)

   monomer_b_ref_xyz(:,1) = (/ 3.0280437961d0,  2.6077617055d0,  0.8685578966d0/)
   monomer_b_ref_xyz(:,2) = (/ 2.6352959969d0,  2.4318876714d0,  1.8662202179d0/)
   monomer_b_ref_xyz(:,3) = (/ 3.2755779869d0,  1.6557409443d0,  0.4078436157d0/)
   monomer_b_ref_xyz(:,4) = (/ 2.2795425973d0,  3.1170415698d0,  0.2678406518d0/)
   monomer_b_ref_xyz(:,5) = (/ 3.9204494427d0,  3.2239426955d0,  0.9341346223d0/)

   keywd_qm = 'DFT PBE0 D3 BASIS=AUG-CC-PVTZ CUTOFF=1.0D-9 XCCUTOFF=1.0D-8 BASISCUTOFF=1.0D-8 DENSERMS=1.0D-6'
   keywd_mbx = trim(keywd_qm)//' MBX_QMMM'
   nrun = get_scan_limit(nscan)

   allocate(no_point_charges(4,0), stat=ierr)
   call stop_on_error(ierr)

   qm_a_au = run_qm_ch4('quick_mbx_ch4_scan_qm_monomer_a', monomer_a_xyz, keywd_qm, no_point_charges)
   qm_b_au = run_qm_ch4('quick_mbx_ch4_scan_qm_monomer_b', monomer_b_ref_xyz, keywd_qm, no_point_charges)

   write(*,'(A)') '# QUICK-MBX methane-dimer scan'
   write(*,'(A)') '# method=PBE0-D3/AUG-CC-PVTZ for QUICK, MB-nrg ch4 model for MBX'
   write(*,'(A)') '# monomer A is QUICK in the hybrid; monomer B is MBX/ch4 in the hybrid'
   write(*,'(A,I0,A,I0)') '# running ', nrun, ' of ', nscan, ' configured scan points'
   write(*,'(A)') 'distance_ang,qm_dimer_au,qm_a_iso_au,qm_b_iso_au,hybrid_ab_au'

   do i = 1, nrun
      idx = scan_index(i, nrun, nscan)
      call make_monomer_b_at_distance(distances(idx), monomer_a_xyz, monomer_b_ref_xyz, monomer_b_xyz)
      dimer_xyz(:,1:natom_ch4) = monomer_a_xyz(:,1:natom_ch4)
      dimer_xyz(:,natom_ch4+1:natom_dimer) = monomer_b_xyz(:,1:natom_ch4)

      qm_dimer_au = run_qm_dimer('quick_mbx_ch4_scan_qm_dimer', idx, dimer_xyz, keywd_qm, no_point_charges)
      hybrid_ab_au = run_qm_mbx_point('quick_mbx_ch4_scan_hybrid_ab', idx, monomer_a_xyz, monomer_b_xyz, &
         keywd_mbx, no_point_charges)

      write(*,'(F8.4,",",F24.16,",",F24.16,",",F24.16,",",F24.16)') distances(idx), qm_dimer_au, qm_a_au, &
         qm_b_au, hybrid_ab_au
   enddo

   if (allocated(no_point_charges)) deallocate(no_point_charges)

contains

   function run_qm_ch4(fname, qm_xyz_ang, keywd, no_point_charges) result(energy)
      implicit none

      character(len=*), intent(in) :: fname, keywd
      double precision, intent(in) :: qm_xyz_ang(3,natom_ch4)
      double precision, intent(in) :: no_point_charges(:,:)
      double precision :: energy

      call setQuickJob(fname, keywd, natom_ch4, ch4_atomic_numbers, .false., ierr)
      call stop_on_error(ierr)
      call getQuickEnergy(qm_xyz_ang, 0, no_point_charges, energy, ierr)
      call stop_on_error(ierr)
      call deleteQuickJob(ierr)
      call stop_on_error(ierr)
   end function run_qm_ch4

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
      double precision, intent(in) :: qm_xyz_ang(3,natom_ch4)
      double precision, intent(in) :: mbx_xyz_ang(3,natom_ch4)
      double precision, intent(in) :: no_point_charges(:,:)
      double precision :: energy
      character(len=80) :: fname

      write(fname,'(A,"_",I0)') trim(prefix), idx
      call setQuickJob(fname, keywd, natom_ch4, ch4_atomic_numbers, .false., ierr)
      call stop_on_error(ierr)
      call setQuickMBXSystem(natom_ch4, natom_ch4, 1, nat_monomers, mbx_xyz_ang, atom_names, monomer_names, &
         'mbx.json', ierr)
      call stop_on_error(ierr)
      call getQuickEnergy(qm_xyz_ang, 0, no_point_charges, energy, ierr)
      call stop_on_error(ierr)
      call clearQuickMBXSystem(ierr)
      call stop_on_error(ierr)
      call deleteQuickJob(ierr)
      call stop_on_error(ierr)
   end function run_qm_mbx_point

   subroutine make_monomer_b_at_distance(rcc, monomer_a_xyz, monomer_b_ref_xyz, monomer_b_xyz)
      implicit none

      double precision, intent(in) :: rcc
      double precision, intent(in) :: monomer_a_xyz(3,natom_ch4)
      double precision, intent(in) :: monomer_b_ref_xyz(3,natom_ch4)
      double precision, intent(out) :: monomer_b_xyz(3,natom_ch4)
      double precision :: cc_vec(3), unit_vec(3), shift(3), target_c(3), norm
      integer :: i

      cc_vec = monomer_b_ref_xyz(:,1) - monomer_a_xyz(:,1)
      norm = sqrt(sum(cc_vec*cc_vec))
      unit_vec = cc_vec/norm
      target_c = monomer_a_xyz(:,1) + rcc*unit_vec
      shift = target_c - monomer_b_ref_xyz(:,1)

      do i = 1, natom_ch4
         monomer_b_xyz(:,i) = monomer_b_ref_xyz(:,i) + shift
      enddo
   end subroutine make_monomer_b_at_distance

   subroutine stop_on_error(ierr_in)
      implicit none

      integer, intent(in) :: ierr_in

      if (ierr_in /= 0) then
         write(*,'(A,1X,I0)') 'QUICK_MBX_CH4_DIMER_SCAN_ERROR', ierr_in
         stop 1
      endif
   end subroutine stop_on_error

   function get_scan_limit(default_nscan) result(nlimit)
      implicit none

      integer, intent(in) :: default_nscan
      integer :: nlimit, env_status, parsed
      character(len=32) :: env_value

      nlimit = default_nscan
      call get_environment_variable('QUICK_MBX_CH4_DIMER_SCAN_NPOINTS', env_value, status=env_status)
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
            idx = 3
         case (2)
            idx = 9
         case default
            idx = 19
         end select
      endif
   end function scan_index

end program quick_mbx_ch4_dimer_scan
