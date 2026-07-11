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
! Generic small-molecule QUICK-MBX pair scan driver. The case is       !
! selected with QUICK_MBX_PAIR_CASE. The driver prints full-QM pair    !
! energies and one QM/MBX hybrid leg on matched geometries.            !
!_____________________________________________________________________!

program quick_mbx_pair_scan

   use quick_api_module, only: setQuickJob, getQuickEnergy, deleteQuickJob
   use quick_api_module, only: setQuickMBXSystem, setQuickMBXWaterSystem, clearQuickMBXSystem

   implicit none

   integer, parameter :: max_atoms = 10
   integer, parameter :: max_scan = 20

   integer :: ierr, i, idx, nrun
   integer :: nqm, nmbx, ndimer, nscan, mbx_nsites
   integer :: qm_atomic_numbers(max_atoms), mbx_atomic_numbers(max_atoms), dimer_atomic_numbers(2*max_atoms)
   integer :: nat_monomers(1)
   double precision :: qm_xyz_ref(3,max_atoms), mbx_xyz_ref(3,max_atoms), mbx_xyz(3,max_atoms)
   double precision :: dimer_xyz(3,2*max_atoms), distances(max_scan)
   double precision :: qm_a_au, qm_b_au, qm_dimer_au, hybrid_ab_au
   double precision, allocatable :: no_point_charges(:,:)
   character(len=256) :: keywd_qm, keywd_mbx, keywd_qm_b, keywd_dimer
   character(len=64) :: case_name, case_label
   character(len=2) :: mbx_atom_names(max_atoms)
   character(len=4) :: mbx_monomer_names(1)
   logical :: mbx_is_water

   ierr = 0
   call setup_case(case_name, case_label, nqm, nmbx, ndimer, nscan, distances, qm_atomic_numbers, &
      mbx_atomic_numbers, dimer_atomic_numbers, qm_xyz_ref, mbx_xyz_ref, mbx_atom_names, &
      mbx_monomer_names, mbx_nsites, mbx_is_water, keywd_qm, keywd_qm_b, keywd_dimer)
   keywd_mbx = trim(keywd_qm)//' MBX_QMMM'
   nrun = get_scan_limit(nscan)

   allocate(no_point_charges(4,0), stat=ierr)
   call stop_on_error(ierr)

   qm_a_au = run_qm_monomer('quick_mbx_pair_scan_qm_a', nqm, qm_atomic_numbers, qm_xyz_ref, keywd_qm, &
      no_point_charges)
   qm_b_au = run_qm_monomer('quick_mbx_pair_scan_qm_b', nmbx, mbx_atomic_numbers, mbx_xyz_ref, keywd_qm_b, &
      no_point_charges)

   write(*,'(A)') '# QUICK-MBX mixed pair scan'
   write(*,'(A,A)') '# case=', trim(case_name)
   write(*,'(A,A)') '# label=', trim(case_label)
   write(*,'(A)') '# method=PBE0-D3/AUG-CC-PVTZ for QUICK; MBX model for the MBX monomer'
   write(*,'(A,I0,A,I0)') '# running ', nrun, ' of ', nscan, ' configured scan points'
   write(*,'(A)') 'distance_ang,qm_dimer_au,qm_a_iso_au,qm_b_iso_au,hybrid_ab_au'

   do i = 1, nrun
      idx = scan_index(i, nrun, nscan)
      call make_mbx_at_distance(distances(idx), nqm, nmbx, qm_xyz_ref, mbx_xyz_ref, mbx_xyz)
      dimer_xyz(:,1:nqm) = qm_xyz_ref(:,1:nqm)
      dimer_xyz(:,nqm+1:ndimer) = mbx_xyz(:,1:nmbx)

      qm_dimer_au = run_qm_dimer('quick_mbx_pair_scan_qm_dimer', idx, ndimer, dimer_atomic_numbers, &
         dimer_xyz, keywd_dimer, no_point_charges)
      hybrid_ab_au = run_qm_mbx_point('quick_mbx_pair_scan_hybrid_ab', idx, nqm, nmbx, qm_atomic_numbers, &
         qm_xyz_ref, mbx_xyz, keywd_mbx, no_point_charges)

      write(*,'(F8.4,",",F24.16,",",F24.16,",",F24.16,",",F24.16)') distances(idx), qm_dimer_au, &
         qm_a_au, qm_b_au, hybrid_ab_au
   enddo

   if (allocated(no_point_charges)) deallocate(no_point_charges)

contains

   subroutine setup_case(case_name, case_label, nqm, nmbx, ndimer, nscan, distances, qm_atomic_numbers, &
         mbx_atomic_numbers, dimer_atomic_numbers, qm_xyz_ref, mbx_xyz_ref, mbx_atom_names, &
         mbx_monomer_names, mbx_nsites, mbx_is_water, keywd_qm, keywd_qm_b, keywd_dimer)
      implicit none

      character(len=64), intent(out) :: case_name, case_label
      integer, intent(out) :: nqm, nmbx, ndimer, nscan, mbx_nsites
      integer, intent(out) :: qm_atomic_numbers(max_atoms), mbx_atomic_numbers(max_atoms), dimer_atomic_numbers(2*max_atoms)
      double precision, intent(out) :: distances(max_scan), qm_xyz_ref(3,max_atoms), mbx_xyz_ref(3,max_atoms)
      character(len=2), intent(out) :: mbx_atom_names(max_atoms)
      character(len=4), intent(out) :: mbx_monomer_names(1)
      character(len=256), intent(out) :: keywd_qm, keywd_qm_b, keywd_dimer
      logical, intent(out) :: mbx_is_water

      integer :: env_status

      case_name = 'h2o_ch4_h2o_qm'
      call get_environment_variable('QUICK_MBX_PAIR_CASE', case_name, status=env_status)
      case_name = adjustl(case_name)

      qm_atomic_numbers(:) = 0
      mbx_atomic_numbers(:) = 0
      dimer_atomic_numbers(:) = 0
      qm_xyz_ref(:,:) = 0.0d0
      mbx_xyz_ref(:,:) = 0.0d0
      distances(:) = 0.0d0
      mbx_atom_names(:) = '  '
      mbx_monomer_names(1) = '    '
      mbx_is_water = .false.

      keywd_qm = 'DFT PBE0 D3 BASIS=AUG-CC-PVTZ CUTOFF=1.0D-9 XCCUTOFF=1.0D-8 BASISCUTOFF=1.0D-8 DENSERMS=1.0D-6'
      keywd_qm_b = keywd_qm
      keywd_dimer = keywd_qm

      select case (trim(case_name))
      case ('h2o_ch4_h2o_qm')
         case_label = 'H2O(QM)/CH4(MBX)'
         nqm = 3
         nmbx = 5
         ndimer = nqm + nmbx
         nscan = 10
         distances(1:nscan) = (/3.00d0,3.20d0,3.40d0,3.60d0,3.80d0,4.00d0,4.30d0,4.70d0,5.20d0,6.00d0/)
         qm_atomic_numbers(1:nqm) = (/8,1,1/)
         mbx_atomic_numbers(1:nmbx) = (/6,1,1,1,1/)
         dimer_atomic_numbers(1:ndimer) = (/8,1,1,6,1,1,1,1/)
         call set_h2o_ch4_geometry(qm_xyz_ref, mbx_xyz_ref, .true.)
         mbx_atom_names(1:nmbx) = (/'C ','H ','H ','H ','H '/)
         mbx_monomer_names(1) = 'ch4 '
         mbx_nsites = 5

      case ('h2o_ch4_ch4_qm')
         case_label = 'CH4(QM)/H2O(MBX)'
         nqm = 5
         nmbx = 3
         ndimer = nqm + nmbx
         nscan = 10
         distances(1:nscan) = (/3.00d0,3.20d0,3.40d0,3.60d0,3.80d0,4.00d0,4.30d0,4.70d0,5.20d0,6.00d0/)
         qm_atomic_numbers(1:nqm) = (/6,1,1,1,1/)
         mbx_atomic_numbers(1:nmbx) = (/8,1,1/)
         dimer_atomic_numbers(1:ndimer) = (/6,1,1,1,1,8,1,1/)
         call set_h2o_ch4_geometry(qm_xyz_ref, mbx_xyz_ref, .false.)
         mbx_atom_names(1:nmbx) = (/'O ','H ','H '/)
         mbx_monomer_names(1) = 'h2o '
         mbx_nsites = 4
         mbx_is_water = .true.

      case ('cl_h2o_cl_qm')
         case_label = 'Cl-(QM)/H2O(MBX)'
         nqm = 1
         nmbx = 3
         ndimer = nqm + nmbx
         nscan = 10
         distances(1:nscan) = (/2.40d0,2.60d0,2.80d0,3.00d0,3.20d0,3.50d0,3.90d0,4.40d0,5.00d0,6.00d0/)
         qm_atomic_numbers(1:nqm) = (/17/)
         mbx_atomic_numbers(1:nmbx) = (/8,1,1/)
         dimer_atomic_numbers(1:ndimer) = (/17,8,1,1/)
         qm_xyz_ref(:,1) = (/-2.2371172880d0, -0.0867486952d0,  1.6637199587d0/)
         mbx_xyz_ref(:,1) = (/-0.0749612883d0, -1.9439663623d0, -0.5660661146d0/)
         mbx_xyz_ref(:,2) = (/-0.4061648717d0, -1.2773088447d0, -1.1890851631d0/)
         mbx_xyz_ref(:,3) = (/-0.6788183774d0, -1.7987964639d0,  0.1834998125d0/)
         mbx_atom_names(1:nmbx) = (/'O ','H ','H '/)
         mbx_monomer_names(1) = 'h2o '
         mbx_nsites = 4
         mbx_is_water = .true.
         keywd_qm = trim(keywd_qm)//' CHARGE=-1'
         keywd_dimer = trim(keywd_dimer)//' CHARGE=-1'

      case ('o2_h2o_o2_qm')
         case_label = 'triplet O2(QM)/H2O(MBX)'
         nqm = 2
         nmbx = 3
         ndimer = nqm + nmbx
         nscan = 10
         distances(1:nscan) = (/2.60d0,2.80d0,3.00d0,3.20d0,3.50d0,3.80d0,4.20d0,4.70d0,5.30d0,6.00d0/)
         qm_atomic_numbers(1:nqm) = (/8,8/)
         mbx_atomic_numbers(1:nmbx) = (/8,1,1/)
         dimer_atomic_numbers(1:ndimer) = (/8,8,8,1,1/)
         qm_xyz_ref(:,1) = (/0.0d0, 0.0d0, -0.60375d0/)
         qm_xyz_ref(:,2) = (/0.0d0, 0.0d0,  0.60375d0/)
         mbx_xyz_ref(:,1) = (/3.20d0, 0.0d0, 0.0d0/)
         mbx_xyz_ref(:,2) = (/3.958602d0, 0.0d0, 0.587079d0/)
         mbx_xyz_ref(:,3) = (/2.441398d0, 0.0d0, 0.587079d0/)
         mbx_atom_names(1:nmbx) = (/'O ','H ','H '/)
         mbx_monomer_names(1) = 'h2o '
         mbx_nsites = 4
         mbx_is_water = .true.
         keywd_qm = 'UDFT PBE0 D3 BASIS=AUG-CC-PVTZ CUTOFF=1.0D-9 XCCUTOFF=1.0D-8 BASISCUTOFF=1.0D-8 DENSERMS=1.0D-6 MULT=3'
         keywd_qm_b = 'DFT PBE0 D3 BASIS=AUG-CC-PVTZ CUTOFF=1.0D-9 XCCUTOFF=1.0D-8 BASISCUTOFF=1.0D-8 DENSERMS=1.0D-6'
         keywd_dimer = keywd_qm

      case default
         write(*,'(A,A)') 'QUICK_MBX_PAIR_SCAN_UNKNOWN_CASE ', trim(case_name)
         stop 1
      end select
   end subroutine setup_case

   subroutine set_h2o_ch4_geometry(first_xyz, second_xyz, h2o_first)
      implicit none

      double precision, intent(out) :: first_xyz(3,max_atoms), second_xyz(3,max_atoms)
      logical, intent(in) :: h2o_first
      double precision :: h2o_xyz(3,3), ch4_xyz(3,5)

      ch4_xyz(:,1) = (/ 0.1780116536d0,  0.1439591651d0,  0.0767959473d0/)
      ch4_xyz(:,2) = (/-0.4732221193d0, -0.4346691605d0, -0.5725415936d0/)
      ch4_xyz(:,3) = (/ 1.1935894004d0,  0.1140036913d0, -0.3086389627d0/)
      ch4_xyz(:,4) = (/-0.1786432286d0,  1.1713010682d0,  0.1063843766d0/)
      ch4_xyz(:,5) = (/ 0.1644153918d0, -0.2800416213d0,  1.0771061590d0/)

      h2o_xyz(:,1) = (/2.0786004022d0, -0.8324584110d0, 3.0706206061d0/)
      h2o_xyz(:,2) = (/2.2502730350d0, -1.0317483584d0, 3.9948528330d0/)
      h2o_xyz(:,3) = (/2.0608921579d0,  0.1418457568d0, 3.0121656318d0/)

      first_xyz(:,:) = 0.0d0
      second_xyz(:,:) = 0.0d0
      if (h2o_first) then
         first_xyz(:,1:3) = h2o_xyz(:,1:3)
         second_xyz(:,1:5) = ch4_xyz(:,1:5)
      else
         first_xyz(:,1:5) = ch4_xyz(:,1:5)
         second_xyz(:,1:3) = h2o_xyz(:,1:3)
      endif
   end subroutine set_h2o_ch4_geometry

   function run_qm_monomer(fname, natom, atomic_numbers, qm_xyz_ang, keywd, no_point_charges) result(energy)
      implicit none

      character(len=*), intent(in) :: fname, keywd
      integer, intent(in) :: natom
      integer, intent(in) :: atomic_numbers(max_atoms)
      double precision, intent(in) :: qm_xyz_ang(3,max_atoms)
      double precision, intent(in) :: no_point_charges(:,:)
      double precision :: energy

      call setQuickJob(fname, keywd, natom, atomic_numbers(1:natom), .false., ierr)
      call stop_on_error(ierr)
      call getQuickEnergy(qm_xyz_ang(:,1:natom), 0, no_point_charges, energy, ierr)
      call stop_on_error(ierr)
      call deleteQuickJob(ierr)
      call stop_on_error(ierr)
   end function run_qm_monomer

   function run_qm_dimer(prefix, idx, natom, atomic_numbers, qm_xyz_ang, keywd, no_point_charges) result(energy)
      implicit none

      character(len=*), intent(in) :: prefix, keywd
      integer, intent(in) :: idx, natom
      integer, intent(in) :: atomic_numbers(2*max_atoms)
      double precision, intent(in) :: qm_xyz_ang(3,2*max_atoms)
      double precision, intent(in) :: no_point_charges(:,:)
      double precision :: energy
      character(len=96) :: fname

      write(fname,'(A,"_",I0)') trim(prefix), idx
      call setQuickJob(fname, keywd, natom, atomic_numbers(1:natom), .false., ierr)
      call stop_on_error(ierr)
      call getQuickEnergy(qm_xyz_ang(:,1:natom), 0, no_point_charges, energy, ierr)
      call stop_on_error(ierr)
      call deleteQuickJob(ierr)
      call stop_on_error(ierr)
   end function run_qm_dimer

   function run_qm_mbx_point(prefix, idx, nqm, nmbx, qm_atomic_numbers, qm_xyz_ang, mbx_xyz_ang, keywd, &
         no_point_charges) result(energy)
      implicit none

      character(len=*), intent(in) :: prefix, keywd
      integer, intent(in) :: idx, nqm, nmbx
      integer, intent(in) :: qm_atomic_numbers(max_atoms)
      double precision, intent(in) :: qm_xyz_ang(3,max_atoms), mbx_xyz_ang(3,max_atoms)
      double precision, intent(in) :: no_point_charges(:,:)
      double precision :: energy
      character(len=96) :: fname

      write(fname,'(A,"_",I0)') trim(prefix), idx
      call setQuickJob(fname, keywd, nqm, qm_atomic_numbers(1:nqm), .false., ierr)
      call stop_on_error(ierr)
      if (mbx_is_water) then
         call setQuickMBXWaterSystem(1, mbx_xyz_ang(:,1:nmbx), 'mbx.json', ierr)
      else
         nat_monomers(1) = nmbx
         call setQuickMBXSystem(nmbx, mbx_nsites, 1, nat_monomers, mbx_xyz_ang(:,1:nmbx), &
            mbx_atom_names(1:nmbx), mbx_monomer_names, 'mbx.json', ierr)
      endif
      call stop_on_error(ierr)
      call getQuickEnergy(qm_xyz_ang(:,1:nqm), 0, no_point_charges, energy, ierr)
      call stop_on_error(ierr)
      call clearQuickMBXSystem(ierr)
      call stop_on_error(ierr)
      call deleteQuickJob(ierr)
      call stop_on_error(ierr)
   end function run_qm_mbx_point

   subroutine make_mbx_at_distance(rab, nqm, nmbx, qm_xyz_ref, mbx_xyz_ref, mbx_xyz)
      implicit none

      double precision, intent(in) :: rab
      integer, intent(in) :: nqm, nmbx
      double precision, intent(in) :: qm_xyz_ref(3,max_atoms), mbx_xyz_ref(3,max_atoms)
      double precision, intent(out) :: mbx_xyz(3,max_atoms)
      double precision :: ab_vec(3), unit_vec(3), shift(3), target_b0(3), norm
      integer :: i

      ab_vec = mbx_xyz_ref(:,1) - qm_xyz_ref(:,1)
      norm = sqrt(sum(ab_vec*ab_vec))
      unit_vec = ab_vec/norm
      target_b0 = qm_xyz_ref(:,1) + rab*unit_vec
      shift = target_b0 - mbx_xyz_ref(:,1)
      mbx_xyz(:,:) = 0.0d0
      do i = 1, nmbx
         mbx_xyz(:,i) = mbx_xyz_ref(:,i) + shift
      enddo
   end subroutine make_mbx_at_distance

   subroutine stop_on_error(ierr_in)
      implicit none

      integer, intent(in) :: ierr_in

      if (ierr_in /= 0) then
         write(*,'(A,1X,I0)') 'QUICK_MBX_PAIR_SCAN_ERROR', ierr_in
         stop 1
      endif
   end subroutine stop_on_error

   function get_scan_limit(default_nscan) result(nlimit)
      implicit none

      integer, intent(in) :: default_nscan
      integer :: nlimit, env_status, parsed
      character(len=32) :: env_value

      nlimit = default_nscan
      call get_environment_variable('QUICK_MBX_PAIR_SCAN_NPOINTS', env_value, status=env_status)
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
      if (nlimit == 3 .and. default_nscan >= 10) then
         select case (irun)
         case (1)
            idx = 1
         case (2)
            idx = default_nscan/2
         case default
            idx = default_nscan
         end select
      endif
   end function scan_index

end program quick_mbx_pair_scan
