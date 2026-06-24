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
! Fixed-geometry water-dimer scan driver for publication validation.   !
! The geometry stack is read from QUICK_MBX_WATER_XYZ as repeated XYZ  !
! frames in the order O H H O H H. Monomer A is QUICK/PBE-D3 and       !
! monomer B is MBX, evaluated once as MB-pol (h2o) and once as MB-PBE  !
! (mbpbe).                                                            !
!_____________________________________________________________________!

program quick_mbx_water_xyz_scan

   use quick_api_module, only: setQuickJob, getQuickEnergy, deleteQuickJob
   use quick_api_module, only: setQuickMBXSystem, clearQuickMBXSystem

   implicit none

   integer, parameter :: natom_water = 3
   integer, parameter :: natom_dimer = 6
   integer, parameter :: max_geoms = 64

   integer :: ierr, igeom, ngeom, env_status
   integer :: water_atomic_numbers(natom_water)
   integer :: dimer_atomic_numbers(natom_dimer)
   double precision :: xyz_stack(3,natom_dimer,max_geoms), distances(max_geoms)
   double precision :: qm_a_au, qm_b_au, qm_dimer_au, hybrid_mbpol_au, hybrid_mbpbe_au
   double precision, allocatable :: no_point_charges(:,:)
   character(len=512) :: xyz_path
   character(len=256) :: keywd_qm, keywd_mbx

   ierr = 0
   water_atomic_numbers = (/8,1,1/)
   dimer_atomic_numbers = (/8,1,1,8,1,1/)
   keywd_qm = 'DFT PBE D3 BASIS=AUG-CC-PVTZ CUTOFF=1.0D-9 XCCUTOFF=1.0D-8 BASISCUTOFF=1.0D-8 DENSERMS=1.0D-6'
   keywd_mbx = trim(keywd_qm)//' MBX_QMMM'

   xyz_path = 'water_dimer_optimized.xyz'
   call get_environment_variable('QUICK_MBX_WATER_XYZ', xyz_path, status=env_status)
   xyz_path = adjustl(xyz_path)

   call read_xyz_stack(trim(xyz_path), ngeom, distances, xyz_stack)

   allocate(no_point_charges(4,0), stat=ierr)
   call stop_on_error(ierr)

   write(*,'(A)') '# QUICK-MBX fixed-geometry water-dimer scan'
   write(*,'(A,A)') '# xyz_stack=', trim(xyz_path)
   write(*,'(A)') '# method=PBE-D3/AUG-CC-PVTZ for QUICK; MBX h2o and mbpbe for the MBX monomer'
   write(*,'(A)') '# monomer A is QUICK in the hybrid; monomer B is MBX in the hybrid'
   write(*,'(A,I0)') '# frames=', ngeom
   write(*,'(A)') 'point,distance_ang,qm_dimer_au,qm_a_iso_au,qm_b_iso_au,hybrid_mbpol_au,hybrid_mbpbe_au'

   do igeom = 1, ngeom
      qm_a_au = run_qm_water('quick_mbx_water_xyz_qm_a', igeom, xyz_stack(:,1:3,igeom), keywd_qm, no_point_charges)
      qm_b_au = run_qm_water('quick_mbx_water_xyz_qm_b', igeom, xyz_stack(:,4:6,igeom), keywd_qm, no_point_charges)
      qm_dimer_au = run_qm_dimer('quick_mbx_water_xyz_qm_dimer', igeom, xyz_stack(:,:,igeom), keywd_qm, no_point_charges)
      hybrid_mbpol_au = run_qm_mbx_point('quick_mbx_water_xyz_hybrid_mbpol', igeom, xyz_stack(:,1:3,igeom), &
         xyz_stack(:,4:6,igeom), 'h2o', keywd_mbx, no_point_charges)
      hybrid_mbpbe_au = run_qm_mbx_point('quick_mbx_water_xyz_hybrid_mbpbe', igeom, xyz_stack(:,1:3,igeom), &
         xyz_stack(:,4:6,igeom), 'mbpbe', keywd_mbx, no_point_charges)

      write(*,'(I0,",",F10.5,",",F24.16,",",F24.16,",",F24.16,",",F24.16,",",F24.16)') &
         igeom, distances(igeom), qm_dimer_au, qm_a_au, qm_b_au, hybrid_mbpol_au, hybrid_mbpbe_au
   enddo

   if (allocated(no_point_charges)) deallocate(no_point_charges)

contains

   subroutine read_xyz_stack(path, ngeom, distances, xyz_stack)
      implicit none

      character(len=*), intent(in) :: path
      integer, intent(out) :: ngeom
      double precision, intent(out) :: distances(max_geoms)
      double precision, intent(out) :: xyz_stack(3,natom_dimer,max_geoms)

      integer :: unit, iatom, nat, iostat
      character(len=256) :: comment
      character(len=8) :: sym

      unit = 77
      ngeom = 0
      distances(:) = 0.0d0
      xyz_stack(:,:,:) = 0.0d0

      open(unit=unit, file=trim(path), status='old', action='read', iostat=iostat)
      if (iostat /= 0) then
         write(*,'(A,A)') 'QUICK_MBX_WATER_XYZ_SCAN_COULD_NOT_OPEN ', trim(path)
         stop 1
      endif

      do
         read(unit,*,iostat=iostat) nat
         if (iostat < 0) exit
         if (iostat /= 0) then
            write(*,'(A)') 'QUICK_MBX_WATER_XYZ_SCAN_BAD_XYZ_NATOM_LINE'
            stop 1
         endif
         if (nat /= natom_dimer) then
            write(*,'(A,I0)') 'QUICK_MBX_WATER_XYZ_SCAN_EXPECTED_6_ATOMS_GOT ', nat
            stop 1
         endif
         if (ngeom >= max_geoms) then
            write(*,'(A,I0)') 'QUICK_MBX_WATER_XYZ_SCAN_TOO_MANY_FRAMES ', max_geoms
            stop 1
         endif

         read(unit,'(A)',iostat=iostat) comment
         if (iostat /= 0) then
            write(*,'(A)') 'QUICK_MBX_WATER_XYZ_SCAN_BAD_COMMENT_LINE'
            stop 1
         endif
         ngeom = ngeom + 1
         do iatom = 1, natom_dimer
            read(unit,*,iostat=iostat) sym, xyz_stack(1,iatom,ngeom), xyz_stack(2,iatom,ngeom), xyz_stack(3,iatom,ngeom)
            if (iostat /= 0) then
               write(*,'(A,I0)') 'QUICK_MBX_WATER_XYZ_SCAN_BAD_ATOM_LINE_FRAME ', ngeom
               stop 1
            endif
         enddo
         distances(ngeom) = sqrt(sum((xyz_stack(:,4,ngeom)-xyz_stack(:,1,ngeom))**2))
      enddo

      close(unit)
      if (ngeom <= 0) then
         write(*,'(A,A)') 'QUICK_MBX_WATER_XYZ_SCAN_NO_FRAMES ', trim(path)
         stop 1
      endif
   end subroutine read_xyz_stack

   function run_qm_water(prefix, idx, qm_xyz_ang, keywd, no_point_charges) result(energy)
      implicit none

      character(len=*), intent(in) :: prefix, keywd
      integer, intent(in) :: idx
      double precision, intent(in) :: qm_xyz_ang(3,natom_water)
      double precision, intent(in) :: no_point_charges(:,:)
      double precision :: energy
      character(len=96) :: fname

      write(fname,'(A,"_",I0)') trim(prefix), idx
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
      character(len=96) :: fname

      write(fname,'(A,"_",I0)') trim(prefix), idx
      call setQuickJob(fname, keywd, natom_dimer, dimer_atomic_numbers, .false., ierr)
      call stop_on_error(ierr)
      call getQuickEnergy(qm_xyz_ang, 0, no_point_charges, energy, ierr)
      call stop_on_error(ierr)
      call deleteQuickJob(ierr)
      call stop_on_error(ierr)
   end function run_qm_dimer

   function run_qm_mbx_point(prefix, idx, qm_xyz_ang, mbx_xyz_ang, mbx_monomer_name, keywd, &
         no_point_charges) result(energy)
      implicit none

      character(len=*), intent(in) :: prefix, mbx_monomer_name, keywd
      integer, intent(in) :: idx
      double precision, intent(in) :: qm_xyz_ang(3,natom_water), mbx_xyz_ang(3,natom_water)
      double precision, intent(in) :: no_point_charges(:,:)
      double precision :: energy
      integer :: nat_monomers(1)
      character(len=2) :: atom_names(natom_water)
      character(len=8) :: monomer_names(1)
      character(len=96) :: fname

      write(fname,'(A,"_",I0)') trim(prefix), idx
      atom_names = (/'O ','H ','H '/)
      monomer_names(1) = trim(mbx_monomer_name)
      nat_monomers(1) = natom_water

      call setQuickJob(fname, keywd, natom_water, water_atomic_numbers, .false., ierr)
      call stop_on_error(ierr)
      call setQuickMBXSystem(natom_water, 4, 1, nat_monomers, mbx_xyz_ang, atom_names, monomer_names, &
         'mbx.json', ierr)
      call stop_on_error(ierr)
      call getQuickEnergy(qm_xyz_ang, 0, no_point_charges, energy, ierr)
      call stop_on_error(ierr)
      call clearQuickMBXSystem(ierr)
      call stop_on_error(ierr)
      call deleteQuickJob(ierr)
      call stop_on_error(ierr)
   end function run_qm_mbx_point

   subroutine stop_on_error(ierr_in)
      implicit none

      integer, intent(in) :: ierr_in

      if (ierr_in /= 0) then
         write(*,'(A,1X,I0)') 'QUICK_MBX_WATER_XYZ_SCAN_ERROR', ierr_in
         stop 1
      endif
   end subroutine stop_on_error

end program quick_mbx_water_xyz_scan
