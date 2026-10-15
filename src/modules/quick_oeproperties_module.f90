#include "util.fh"
!---------------------------------------------------------------------!
! Created by Etienne Palos on  01/20/2024                             !
! Contributor: Vikrant Tripathy                                       !
!                                                                     !
! Purpose:  " Compute electrostatic properties on grid points "       !
!                                                                     !
! Capabilities:                                                       !
!              - ESP        Serial and MPI                            !
!              - EFIELD     Serial and MPI                            !
!              - EFG        Serial and MPI                            !
!                                                                     ! 
! Copyright (C) 2024-2025                                             !
!                                                                     !
! This Source Code Form is subject to the terms of the Mozilla Public !
! License, v. 2.0. If a copy of the MPL was not distributed with this !
! file, You can obtain one at http://mozilla.org/MPL/2.0/.            !
!---------------------------------------------------------------------!
module quick_oeproperties_module
 private
 public :: compute_oeprop
 public :: compute_oeprop_values, compute_esp_values
 public :: compute_efield_values, compute_efg_values_analytic, compute_efg_values_numerical

 contains

!--------------------------------------------------------------------!
!  The following subroutine "compute_oeprop" is the only routine     !
!  called from outside. This routine performs calculations as per    !
!  user provided keywords.                                           !
!--------------------------------------------------------------------!

 Subroutine compute_oeprop()
   use quick_method_module, only: quick_method
   use quick_files_module, only : ioutfile
   use quick_molsurface_module, only: generate_MKS_surfaces, generate_density_surfaces
   use quick_molspec_module, only: quick_molspec
   use quick_calculated_module, only: quick_qm_struct
#if defined(MPIV)
   use quick_mpi_module, only: master, quick_comm, quick_mpi_error
   use mpi
#endif
   implicit none

   integer :: alloc_status

   if (quick_method%ext_grid) then
      call compute_oeprop_grid(quick_molspec%nextpoint,quick_molspec%extpointxyz)
   else if (quick_method%density_surface) then

#ifdef MPIV
      if(master)then
#endif
        call generate_density_surfaces()
#ifdef MPIV
      endif
      call MPI_BCAST(quick_molspec%nvdwpoint,1,mpi_integer,0,quick_comm,quick_mpi_error)
      if(.not.master)then
        allocate(quick_molspec%vdwpointxyz(3,quick_molspec%nvdwpoint), stat=alloc_status)

        if(alloc_status /= 0) then
          call PrtErr(OUTFILEHANDLE, '!!quick_molspec%vdwpointxyz array reallocation failed in compute_oeprop!!')
          call quick_exit(OUTFILEHANDLE,1)
        endif

      endif
      call MPI_BCAST(quick_molspec%vdwpointxyz,quick_molspec%nvdwpoint*3,mpi_double_precision,0,quick_comm,quick_mpi_error)
#endif

      call compute_oeprop_grid(quick_molspec%nvdwpoint,quick_molspec%vdwpointxyz)

      deallocate(quick_molspec%vdwpointxyz)
   else if (quick_method%esp_charge) then

#ifdef MPIV
      if(master)then
#endif
        call generate_MKS_surfaces()
#ifdef MPIV
      endif
      call MPI_BCAST(quick_molspec%nvdwpoint,1,mpi_integer,0,quick_comm,quick_mpi_error)
      if(.not.master)then
        allocate(quick_molspec%vdwpointxyz(3,quick_molspec%nvdwpoint), stat=alloc_status)

        if(alloc_status /= 0) then
          call PrtErr(OUTFILEHANDLE, '!!quick_molspec%vdwpointxyz array reallocation failed in compute_oeprop!!')
          call quick_exit(OUTFILEHANDLE,1)
        endif
      
      endif
      call MPI_BCAST(quick_molspec%vdwpointxyz,quick_molspec%nvdwpoint*3,mpi_double_precision,0,quick_comm,quick_mpi_error)
#endif

      call compute_oeprop_grid(quick_molspec%nvdwpoint,quick_molspec%vdwpointxyz)

      deallocate(quick_molspec%vdwpointxyz)

   else
#ifdef MPIV
      if(master) then
#endif
      write (ioutfile,'("  Skipping one-electron property calculation.")')
#ifdef MPIV
      endif
#endif
   end if

 end Subroutine

 Subroutine compute_oeprop_grid(npoints,xyz_points)
   use quick_exception_module
   use quick_files_module, only: iESPFile, espFileName, iVdwSurfFile, VdwSurfFileName
   use quick_method_module, only: quick_method
   use quick_timer_module, only : timer_begin, timer_end, timer_cumer
#if defined(MPIV)
   use quick_mpi_module, only: master
#endif

   implicit none
   integer :: ierr, npoints, alloc_status
   double precision, allocatable :: esp_on_points(:)
   double precision, intent(in) :: xyz_points(:,:)

   if (quick_method%esp_grid .or. quick_method%esp_charge) then
     allocate(esp_on_points(npoints), stat=alloc_status)

     if(alloc_status /= 0) then
       call PrtErr(OUTFILEHANDLE, '!!esp_on_points array reallocation failed in compute_oeprop_grid!!')
       call quick_exit(OUTFILEHANDLE,1)
     endif
   end if
      
   ierr = 0

   if (quick_method%esp_grid .or. quick_method%esp_charge) then
     call compute_esp(npoints,xyz_points,esp_on_points)
   endif

   ! Electrostatic Potential
   if (quick_method%esp_grid) then
     ! Print ESP at external points
#ifdef MPIV
     if (master) then
#endif
       SAFE_CALL(quick_open(iESPFile,espFileName,'U','F','R',.false.,ierr))
       call print_esp(esp_on_points,npoints,xyz_points,iESPFile,espFileName)
       close(iESPFile)
#ifdef MPIV
     endif
#endif
   end if

   ! Compute ESP charge using the MKS grid
   if (quick_method%esp_charge) then
#ifdef MPIV
     if (master) then
#endif
       SAFE_CALL(quick_open(iVdwSurfFile,VdwSurfFileName,'U','F','R',.false.,ierr))
       call print_esp(esp_on_points,npoints,xyz_points,iVdwSurfFile,VdwSurfFileName)
       close(iVdwSurfFile)
#ifdef MPIV
     endif
#endif

     RECORD_TIME(timer_begin%TESPCharge)

#ifdef MPIV
     if (master) then
#endif
       call compute_ESP_charge(npoints,xyz_points,esp_on_points)
#ifdef MPIV
     end if
#endif

     RECORD_TIME(timer_end%TESPCharge)
     timer_cumer%TESPCharge=timer_cumer%TESPCharge+timer_end%TESPCharge-timer_begin%TESPCharge

   end if

   ! Electric field
   if (quick_method%efield_grid) then
     call compute_efield(npoints,xyz_points)
   end if

   ! Electric field gradient
   if (quick_method%efg_grid) then
     call compute_efg(npoints,xyz_points)
   end if

   if (allocated(esp_on_points)) deallocate(esp_on_points)

 end Subroutine

!--------------------------------------------------------------------!
!  No-I/O OEPROP evaluator for external embedding interfaces.        !
!  Coordinates are in bohr. Returned ESP, EFIELD, and EFG values use !
!  QUICK atomic-unit conventions. EFG follows G_ij=dE_i/dC_j.        !
!--------------------------------------------------------------------!

 subroutine compute_oeprop_values(npoints,xyz_points,esp,efield,efg)
   implicit none

   integer, intent(in) :: npoints
   double precision, intent(in) :: xyz_points(:,:)
   double precision, intent(out), optional :: esp(:)
   double precision, intent(out), optional :: efield(:,:)
   double precision, intent(out), optional :: efg(:,:,:)

   if (present(esp)) call compute_esp(npoints,xyz_points,esp)
   if (present(efield)) call compute_efield_values(npoints,xyz_points,efield)
   if (present(efg)) call compute_efg_values_analytic(npoints,xyz_points,efg)

 end subroutine compute_oeprop_values

 subroutine compute_esp_values(npoints,xyz_points,esp)
   implicit none

   integer, intent(in) :: npoints
   double precision, intent(in) :: xyz_points(:,:)
   double precision, intent(out) :: esp(:)

   call compute_esp(npoints,xyz_points,esp)

 end subroutine compute_esp_values

!--------------------------------------------------------------------!
!   Shell-pair and density-contraction kernels are included from     !
!   ./include/attrashell.fh and ./include/nuclearattra.fh.            !
!                                                                    !
!   The header files are called with OEPROP being defined.           !
!--------------------------------------------------------------------!

#define OEPROP
#include "./include/attrashell.fh"
#include "./include/nuclearattra.fh"
#undef OEPROP

 !----------------------------------------------------------------------------!
 ! This is the subroutine that "computes" the Electrostatic Potential (ESP)   !
 ! at a given point , V(r) = V_nuc(r) + V_elec(r), and prints it to file.prop !
 !                                                                            !
 ! This subroutine combines shell-pair electronic terms with direct           !
 ! classical contributions from nuclei and external point charges.            !
 !----------------------------------------------------------------------------!
 subroutine compute_esp(npoints,xyz_points,esp)
   use quick_basis_module, only: jshell
   use quick_calculated_module, only: quick_qm_struct
   use quick_timer_module, only : timer_begin, timer_end, timer_cumer
#if defined(MPIV)
    use mpi
    use quick_basis_module, only: mpi_jshelln, mpi_jshell
    use quick_mpi_module, only: master, quick_comm, quick_comm_rank, quick_mpi_error
#endif
#if defined(GPU) || defined(MPIV_GPU)
    use quick_method_module, only: quick_method
#endif

   implicit none

   integer :: ierr, alloc_status
   integer :: IIsh, JJsh
   integer :: npoints

   double precision, intent(in)  :: xyz_points(:,:)
   double precision, intent(out) :: esp(:)
#ifdef MPIV
   double precision, allocatable :: esp_aggregate(:)
#endif
   integer :: Ish

   ierr = 0
   
#ifdef MPIV
   allocate(esp_aggregate(npoints), stat=alloc_status)
   if(alloc_status /= 0) then
     call PrtErr(OUTFILEHANDLE, '!! esp_aggregate array allocation failed in compute_esp!!')
     call quick_exit(OUTFILEHANDLE,1)
   endif
#endif

   ! The caller-owned output is also the electronic accumulation workspace.
   esp(:) = 0.0d0

   RECORD_TIME(timer_begin%TESPGrid)

   ! Computes ESP_ELEC
#if defined(GPU) || defined(MPIV_GPU)
   call gpu_upload_oeprop(npoints, xyz_points, esp, ierr)
   call gpu_upload_density_matrix(quick_qm_struct%dense)
   if (quick_method%UNRST) call gpu_upload_beta_density_matrix(quick_qm_struct%denseb)
   call gpu_get_oeprop(esp)
#if defined MPIV
   call MPI_REDUCE(esp, esp_aggregate, npoints, &
     MPI_double_precision, MPI_SUM, 0, quick_comm, quick_mpi_error)
#endif
   ! Sum over contributions from different shell pairs
#elif defined MPIV
   ! MPI parallellization is performed over shell-pairs
   ! Different processes consider different shell-pairs
   do Ish=1,mpi_jshelln(quick_comm_rank)
      IIsh=mpi_jshell(quick_comm_rank,Ish)
      do JJsh=IIsh,jshell
         call esp_shell_pair(IIsh, JJsh, npoints, xyz_points, esp)
      enddo
   enddo
   call MPI_REDUCE(esp, esp_aggregate, npoints, &
     MPI_double_precision, MPI_SUM, 0, quick_comm, quick_mpi_error)
#else
   do IIsh = 1, jshell
      do JJsh = IIsh, jshell
        call esp_shell_pair(IIsh, JJsh, npoints, xyz_points, esp)
      end do
   end do
#endif

   ! Add nuclei and external point charges once, after the MPI reduction.
#ifdef MPIV
   if (master) then
     esp(:) = esp_aggregate(:)
     call add_classical_esp(npoints,xyz_points,esp)
   else
     esp(:) = 0.0d0
   endif
#else
   call add_classical_esp(npoints,xyz_points,esp)
#endif

   RECORD_TIME(timer_end%TESPGrid)
   timer_cumer%TESPGrid=timer_cumer%TESPGrid+timer_end%TESPGrid-timer_begin%TESPGrid

#ifdef MPIV
   deallocate(esp_aggregate)
#endif

 end subroutine compute_esp

!----------------------------------------------------------!
!  Obtain ESP charge by solving:                           !
!             Aq=B                                         !
!  B is a column vector of dimension (natom+1)             !
!  A is a symmetric matrix of dimension (natom+1,natom+1)  !
!  q is a column vector of charges. Dimension: (natom+1)   !
!                                                          !
!  Only upper triangle of A is stored.                     !
!----------------------------------------------------------!

 subroutine compute_ESP_charge(npoints,xyz_points,esp)
   use quick_molspec_module, only: quick_molspec, natom, xyz
   use quick_files_module, only: ioutfile
   use quick_exception_module, only: RaiseException
   use quick_constants_module, only : symbol
#if defined(MPIV)
   use quick_mpi_module, only: master
#endif

   implicit none

   integer, external :: ILAENV
   double precision, external :: rootSquare

   integer, allocatable :: IPIV(:)
   integer :: iatom, jatom, igridpoint, npoints, ierr, NB, LWORK, LDA, alloc_status
   double precision, intent(in) :: esp(:), xyz_points(:,:)
   double precision, allocatable :: WORK(:)
   double precision :: A(natom+1,natom+1), B(natom+1), q(natom+1)
   double precision :: distance, distanceb, invdistance, Net_charge

   double precision, allocatable :: invdist_arr(:,:)

   double precision, parameter :: One = 1.0d0, Zero = 0.0d0

!  A, B and q are initialized.

   q = Zero

   B = Zero
   B(natom+1) = quick_molspec%molchg

   A = Zero
   A(1:natom,natom+1) = One

   allocate(invdist_arr(natom,npoints), stat=alloc_status)

   if (alloc_status /= 0) then
     ! The matrix A and vector B is formed.
     do iatom = 1, natom  
       do igridpoint = 1, npoints
         distance = rootSquare(xyz(1:3,iatom), xyz_points(1:3,igridpoint), 3)
         invdistance = 1/distance
         B(iatom) = B(iatom) + esp(igridpoint) * invdistance
         do jatom = 1, iatom
           distanceb = rootSquare(xyz(1:3,jatom), xyz_points(1:3,igridpoint), 3)
           A(jatom,iatom) = A(jatom,iatom) + invdistance/distanceb
         end do
       end do
     end do
   else
     ! First the inverse distance matrix is formed
     do iatom = 1, natom
       do igridpoint = 1, npoints
         invdist_arr(iatom,igridpoint) = 1/rootSquare(xyz(1:3,iatom), xyz_points(1:3,igridpoint), 3)
       end do
     end do

     ! Using the inverse distance matrix to form the matrix A and vector B.
     call DGEMV('N', natom, npoints, One, invdist_arr, natom, esp, 1, Zero, B, 1)
     call MAT_DGEMM('N', 'T', natom, natom, npoints, One, invdist_arr, natom, &
             invdist_arr, natom, Zero, A(1:natom,1:natom), natom)

     deallocate(invdist_arr)

   end if

   call symmetrize('U',A,natom+1)

!  A is inverted.

   NB = ILAENV(1,'DGETRI',' ',natom+1,-1,-1,-1)

   LWORK = (natom+1)*NB

   allocate(WORK(LWORK))
   allocate(IPIV(natom+1))

   LDA = natom+1
   CALL DGETRF(natom+1,natom+1,A,LDA,IPIV,ierr)
   call DGETRI(natom+1,A,LDA,IPIV,WORK,LWORK,ierr)

   if (ierr /= 0) then
     ierr = 40
     call RaiseException(ierr)
   end if

   deallocate(IPIV)
   deallocate(WORK)

!  q = A-1*B

   call DGEMV('N', natom+1, natom+1, One, A, LDA, B, 1, Zero, q, 1)

!  B is copied to charge array.

   Net_charge = Zero

#ifdef MPIV
   if(master) then
#endif
     write (ioutfile,'("  ESP charges:")')
     write (ioutfile,'("  ----------------")')
     do iatom = 1, natom
       Net_charge = Net_charge + q(iatom)
       write (ioutfile,'(3x,I3,3x,A2,3x,F10.6)') iatom, symbol(quick_molspec%iattype(iatom)), q(iatom)
     end do
     write (ioutfile,'("  ----------------")')
     write (ioutfile,'("  Net charge = ",F10.6)')Net_charge
     write (ioutfile,'("  ")')
#ifdef MPIV
   endif
#endif

 end subroutine compute_ESP_charge

 !---------------------------------------------------------------------------------------------!
 ! This subroutine formats and prints the ESP data to "file.esp"                               !
 !---------------------------------------------------------------------------------------------!
 subroutine print_esp(esp, npoints, xyz_points, iESPFile, espFileName)
   use quick_method_module, only: quick_method
   use quick_files_module, only: ioutfile, iVdwSurfFile
   use quick_constants_module, only: BOHRS_TO_A

   implicit none
   integer, intent(in) :: npoints

   integer, intent(in) :: iESPFile
   character :: espFileName*(*)

   double precision, intent(in) :: xyz_points(:,:), esp(:)

   integer :: igridpoint
   double precision :: Cx, Cy, Cz

   if (iESPFile.eq.iVdwSurfFile)then
     quick_method%extgrid_angstrom = .True.
   endif

   ! If ESP_GRID is true, print to table X, Y, Z, V(r)
   if (quick_method%extgrid_angstrom)  then
     if (iESPFile.eq.iVdwSurfFile)then
       write (ioutfile,'(" *** Printing Electrostatic Potential (ESP) &
               &at points on vdw surface to ",A,x,"with coordinates &
               &in angstroms***")') &
           trim(espFileName)
       write (iESPFile,'(/," ELECTROSTATIC POTENTIAL CALCULATION (ESP) &
               &with coordinates of the points on vdw surface in angstroms")')
     else if (quick_method%density_surface) then
       write (ioutfile,'(" *** Printing Electrostatic Potential (ESP) &
               &at points on electron-density surface to ",A,x,"with &
               &coordinates in angstroms***")') &
           trim(espFileName)
       write (iESPFile,'(/," ELECTROSTATIC POTENTIAL CALCULATION (ESP) &
               &with coordinates of the points on electron-density &
               &surface in angstroms")')
     else
       write (ioutfile,'(" *** Printing Electrostatic Potential (ESP) &
               at external points to ",A,x,"with coordinates &
               in angstroms***")') &
           trim(espFileName)
       write (iESPFile,'(/," ELECTROSTATIC POTENTIAL CALCULATION (ESP) &
               &with coordinates in angstroms")')
     endif
     write (iESPFile,'(100("-"))')
     write (iESPFile,'(6x,"X[A]",10x ,"Y[A]",9x,"Z[A]",13x, "ESP_TOTAL [a.u.] ")')
   else
     write (ioutfile,'(" *** Printing Electrostatic Potential (ESP) &
             &[a.u.] at external points to ",A,x,"***")') &
           trim(espFileName)
     write (iESPFile,'(/," ELECTROSTATIC POTENTIAL CALCULATION (ESP) &
             &[atomic units] ")')
     write (iESPFile,'(100("-"))')
     ! Default is X, Y, and V_total in a.u.
     write (iESPFile,'(9x,"X",13x,"Y",12x,"Z",16x,"ESP")')
   endif

   ! Collect ESP and print
   do igridpoint = 1, npoints
     if (quick_method%extgrid_angstrom)  then
       Cx = (xyz_points(1, igridpoint)*BOHRS_TO_A)
       Cy = (xyz_points(2, igridpoint)*BOHRS_TO_A)
       Cz = (xyz_points(3, igridpoint)*BOHRS_TO_A)
     else
       Cx = xyz_points(1, igridpoint)
       Cy = xyz_points(2, igridpoint)
       Cz = xyz_points(3, igridpoint)
     endif
     write(iESPFile, '(2x,3(F14.10, 1x), 3F14.10)') Cx, Cy, Cz, esp(igridpoint)
   end do

 end subroutine print_esp

 !-----------------------------------------------------------------------!
 ! Add the classical point-source potential from nuclei and EXTCHARGES. !
 !-----------------------------------------------------------------------!
 subroutine add_classical_esp(npoints,xyz_points,esp)
   use quick_molspec_module, only: natom, quick_molspec, xyz

   implicit none
   integer, intent(in) :: npoints
   double precision, intent(in)  :: xyz_points(:,:)
   double precision, intent(inout) :: esp(:)

   double precision :: charge, classical, inv_distance
   double precision :: rvec(3), source_xyz(3)
   integer :: isource, igridpoint

   do igridpoint = 1, npoints
     classical = 0.0d0
     do isource=1,natom+quick_molspec%nextatom
       if(isource<=natom)then
         charge = quick_molspec%chg(isource)
         source_xyz(:) = xyz(:,isource)
       else
         charge = quick_molspec%extchg(isource-natom)
         source_xyz(:) = quick_molspec%extxyz(:,isource-natom)
       endif
       rvec(:) = xyz_points(:,igridpoint) - source_xyz(:)
       inv_distance = 1.0d0/sqrt(dot_product(rvec,rvec))
       classical = classical + charge*inv_distance
     enddo
     esp(igridpoint) = esp(igridpoint) + classical
   enddo

 end subroutine add_classical_esp


 !----------------------------------------------------------------------------------!
 ! This is the subroutine that "computes" the Electric Field (EFIELD)               !
 ! at a given point , E(x,y,z) = E_nuc(x,y,z) + E_elec(x,y,z), printing the         !
 ! result to file.efield                                                            !
 !                                                                                  !
 !----------------------------------------------------------------------------------!
 subroutine compute_efield(npoints,xyz_points)
  use quick_exception_module
  use quick_files_module, only: iEFIELDFile, efieldFileName
  use quick_timer_module, only: timer_begin, timer_end, timer_cumer
#ifdef MPIV
   use quick_mpi_module, only: master
#endif

   implicit none
   integer :: ierr, npoints, alloc_status
   double precision, intent(in) :: xyz_points(:,:)
   double precision, allocatable :: efield(:,:)

   ierr = 0
   allocate(efield(3,npoints), stat=alloc_status)
   if(alloc_status /= 0) then
     call PrtErr(OUTFILEHANDLE, '!! efield array allocation failed in compute_efield!!')
     call quick_exit(OUTFILEHANDLE,1)
   endif

   RECORD_TIME(timer_begin%TEFIELDGrid)

   call compute_efield_values(npoints,xyz_points,efield)

   RECORD_TIME(timer_end%TEFIELDGrid)
   timer_cumer%TEFIELDGrid=timer_cumer%TEFIELDGrid+timer_end%TEFIELDGrid-timer_begin%TEFIELDGrid

#ifdef MPIV
   if (master) then
#endif
     SAFE_CALL(quick_open(iEFIELDFile,efieldFileName,'U','F','R',.false.,ierr))
     call print_efield(efield,npoints,xyz_points)
     close(iEFIELDFile)
#ifdef MPIV
   endif
#endif

   deallocate(efield)

 end subroutine compute_efield

!----------------------------------------------------------------------------------!
! This subroutine computes EFIELD values on the supplied grid. It is used by both   !
! EFIELD_GRID output and finite-difference EFG_GRID.                               !
!----------------------------------------------------------------------------------!
 subroutine compute_efield_values(npoints,xyz_points,efield)
  use quick_basis_module, only: jshell
  use quick_calculated_module, only: quick_qm_struct
  use quick_exception_module
#ifdef MPIV
   use mpi
   use quick_basis_module, only: mpi_jshelln, mpi_jshell
   use quick_mpi_module, only: master, quick_comm, quick_comm_rank, quick_mpi_error
#endif
#if defined(GPU) || defined(MPIV_GPU)
   use quick_method_module, only: quick_method
#endif

   implicit none
   integer :: ierr, IIsh, JJsh
   integer :: npoints, alloc_status
   double precision, intent(in) :: xyz_points(:,:)
   double precision, intent(out) :: efield(:,:)
#ifdef MPIV
   double precision, allocatable :: efield_aggregate(:,:)
#endif
   integer :: Ish

#ifdef MPIV
   allocate(efield_aggregate(3,npoints), stat=alloc_status)
   if(alloc_status /= 0) then
     call PrtErr(OUTFILEHANDLE, '!! efield_aggregate array allocation failed in compute_efield_values!!')
     call quick_exit(OUTFILEHANDLE,1)
   endif
#endif

   ! EFIELD(3,npoints) is component-first: (Ex,Ey,Ez) for each point.
   efield(:,:) = 0.0d0

   ! Computes EFIELD_ELEC by summing over contributions from shell-pairs.
#if defined(GPU) || defined(MPIV_GPU)
   ierr = 0
   call gpu_upload_oeprop_efield(npoints, xyz_points, efield, ierr)
   call gpu_upload_density_matrix(quick_qm_struct%dense)
   if (quick_method%UNRST) call gpu_upload_beta_density_matrix(quick_qm_struct%denseb)
   call gpu_get_oeprop_efield(efield)
#if defined MPIV
   call MPI_REDUCE(efield, efield_aggregate, 3*npoints, &
     MPI_double_precision, MPI_SUM, 0, quick_comm, quick_mpi_error)
#endif
#elif defined MPIV
   do Ish=1,mpi_jshelln(quick_comm_rank)
      IIsh=mpi_jshell(quick_comm_rank,Ish)
      do JJsh=IIsh,jshell
         call efield_shell_pair(IIsh,JJsh,npoints,xyz_points,efield)
      enddo
   enddo
   call MPI_REDUCE(efield, efield_aggregate, 3*npoints, &
     MPI_double_precision, MPI_SUM, 0, quick_comm, quick_mpi_error)
#else
   do IIsh = 1, jshell
      do JJsh = IIsh, jshell
        call efield_shell_pair(IIsh,JJsh,npoints,xyz_points,efield)
      end do
   end do
#endif

   ! Add nuclei and external point charges once, after the MPI reduction.
#ifdef MPIV
   if (master) then
     efield(:,:) = efield_aggregate(:,:)
     call add_classical_efield(npoints,xyz_points,efield)
   else
     efield(:,:) = 0.0d0
   endif
#else
   call add_classical_efield(npoints,xyz_points,efield)
#endif

#ifdef MPIV
   deallocate(efield_aggregate)
#endif

 end subroutine compute_efield_values

 !----------------------------------------------------------------------------------!
 ! This subroutine computes the Electric Field Gradient (EFG) on the supplied grid. !
 ! The default algorithm evaluates analytic field-gradient integrals. The           !
 ! EFG_GRID_NUMERICAL keyword requests the central finite-difference reference.     !
 !----------------------------------------------------------------------------------!
 subroutine compute_efg(npoints,xyz_points)
  use quick_exception_module
  use quick_files_module, only: iEFGFile, efgFileName
  use quick_method_module, only: quick_method
  use quick_timer_module, only: timer_begin, timer_end, timer_cumer
#ifdef MPIV
   use quick_mpi_module, only: master
#endif

   implicit none
   integer :: ierr, npoints, alloc_status
   double precision, intent(in) :: xyz_points(:,:)
   double precision, allocatable :: efg(:,:,:)

   ierr = 0

   allocate(efg(3,3,npoints), stat=alloc_status)
   if(alloc_status /= 0) then
     call PrtErr(OUTFILEHANDLE, '!! EFG array allocation failed in compute_efg!!')
     call quick_exit(OUTFILEHANDLE,1)
   endif

   RECORD_TIME(timer_begin%TEFGGrid)

   if (quick_method%efg_grid_numerical) then
     call compute_efg_values_numerical(npoints,xyz_points,efg)
   else
     call compute_efg_values_analytic(npoints,xyz_points,efg)
   endif

   RECORD_TIME(timer_end%TEFGGrid)
   timer_cumer%TEFGGrid=timer_cumer%TEFGGrid+timer_end%TEFGGrid-timer_begin%TEFGGrid

#ifdef MPIV
   if (master) then
#endif
     SAFE_CALL(quick_open(iEFGFile,efgFileName,'U','F','R',.false.,ierr))
     call print_efg(efg,npoints,xyz_points)
     close(iEFGFile)
#ifdef MPIV
   endif
#endif

   deallocate(efg)

 end subroutine compute_efg

!----------------------------------------------------------------------------------!
! This subroutine computes analytic EFG values on the supplied grid.                !
!----------------------------------------------------------------------------------!
 subroutine compute_efg_values_analytic(npoints,xyz_points,efg)
  use quick_basis_module, only: jshell
  use quick_calculated_module, only: quick_qm_struct
  use quick_exception_module
#ifdef MPIV
   use mpi
   use quick_basis_module, only: mpi_jshelln, mpi_jshell
   use quick_mpi_module, only: master, quick_comm, quick_comm_rank, quick_mpi_error
#endif
#if defined(GPU) || defined(MPIV_GPU)
   use quick_method_module, only: quick_method
#endif

   implicit none
   integer :: ierr, IIsh, JJsh
   integer :: npoints, alloc_status
   double precision, intent(in) :: xyz_points(:,:)
   double precision, intent(out) :: efg(:,:,:)
#ifdef MPIV
   double precision, allocatable :: efg_aggregate(:,:,:)
#endif
   integer :: Ish

#ifdef MPIV
   allocate(efg_aggregate(3,3,npoints), stat=alloc_status)
   if(alloc_status /= 0) then
     call PrtErr(OUTFILEHANDLE, '!! efg_aggregate array allocation failed in compute_efg_values_analytic!!')
     call quick_exit(OUTFILEHANDLE,1)
   endif
#endif

   ! EFG(i,j,p)=dE_i/dC_j. Flattened Fortran storage is
   ! (xx,yx,zx,xy,yy,zy,xz,yz,zz) for each point.
   efg(:,:,:) = 0.0d0

   ! Computes EFG_ELEC by summing over contributions from shell-pairs.
#if defined(GPU) || defined(MPIV_GPU)
   ierr = 0
   call gpu_upload_oeprop_efg(npoints, xyz_points, efg, ierr)
   call gpu_upload_density_matrix(quick_qm_struct%dense)
   if (quick_method%UNRST) call gpu_upload_beta_density_matrix(quick_qm_struct%denseb)
   call gpu_get_oeprop_efg(efg)
#if defined MPIV
   call MPI_REDUCE(efg, efg_aggregate, 9*npoints, &
     MPI_double_precision, MPI_SUM, 0, quick_comm, quick_mpi_error)
#endif
#elif defined MPIV
   do Ish=1,mpi_jshelln(quick_comm_rank)
      IIsh=mpi_jshell(quick_comm_rank,Ish)
      do JJsh=IIsh,jshell
         call efg_shell_pair(IIsh,JJsh,npoints,xyz_points,efg)
      enddo
   enddo
   call MPI_REDUCE(efg, efg_aggregate, 9*npoints, &
     MPI_double_precision, MPI_SUM, 0, quick_comm, quick_mpi_error)
#else
   do IIsh = 1, jshell
      do JJsh = IIsh, jshell
        call efg_shell_pair(IIsh,JJsh,npoints,xyz_points,efg)
      end do
   end do
#endif

   ! Add nuclei and external point charges once, after the MPI reduction.
#ifdef MPIV
   if (master) then
     efg(:,:,:) = efg_aggregate(:,:,:)
     call add_classical_efg(npoints,xyz_points,efg)
   else
     efg(:,:,:) = 0.0d0
   endif
#else
   call add_classical_efg(npoints,xyz_points,efg)
#endif

#ifdef MPIV
   deallocate(efg_aggregate)
#endif

 end subroutine compute_efg_values_analytic

!----------------------------------------------------------------------------------!
! This subroutine computes the Electric Field Gradient (EFG) by central finite     !
! difference of EFIELD on the supplied grid.                                      !
!----------------------------------------------------------------------------------!
 subroutine compute_efg_values_numerical(npoints,xyz_points,efg)
#ifdef MPIV
   use quick_mpi_module, only: master
#endif

   implicit none
   integer :: npoints, alloc_status
   integer :: idir, ifield, igridpoint
   double precision, intent(in) :: xyz_points(:,:)
   double precision, intent(out) :: efg(:,:,:)
   double precision, allocatable :: efield_plus(:,:), efield_minus(:,:)
   double precision, allocatable :: xyz_displaced(:,:)
   double precision, parameter :: efg_fd_step = 1.0d-4

   allocate(efield_plus(3,npoints), efield_minus(3,npoints), &
            xyz_displaced(3,npoints), stat=alloc_status)
   if(alloc_status /= 0) then
     call PrtErr(OUTFILEHANDLE, '!! EFG finite-difference arrays allocation failed in compute_efg_values_numerical!!')
     call quick_exit(OUTFILEHANDLE,1)
   endif

   efg(:,:,:) = 0.0d0

   do idir=1,3
     xyz_displaced(:,:) = xyz_points(:,:)
     xyz_displaced(idir,:) = xyz_displaced(idir,:) + efg_fd_step
     call compute_efield_values(npoints,xyz_displaced,efield_plus)

     xyz_displaced(idir,:) = xyz_points(idir,:) - efg_fd_step
     call compute_efield_values(npoints,xyz_displaced,efield_minus)

#ifdef MPIV
     if (master) then
#endif
       do igridpoint=1,npoints
         do ifield=1,3
           efg(ifield,idir,igridpoint) = &
             (efield_plus(ifield,igridpoint)-efield_minus(ifield,igridpoint))/(2.0d0*efg_fd_step)
         end do
       end do
#ifdef MPIV
     endif
#endif
   end do

   deallocate(efield_plus, efield_minus, xyz_displaced)

 end subroutine compute_efg_values_numerical

!------------------------------------------------------------------------!
! Add the classical field from nuclei and EXTCHARGES.                     !
!------------------------------------------------------------------------!
 subroutine add_classical_efield(npoints,xyz_points,efield)
  use quick_molspec_module, only: natom, quick_molspec, xyz
  implicit none

  integer, intent(in) :: npoints
  double precision, intent(in) :: xyz_points(:,:)
  double precision, intent(inout) :: efield(:,:)

  double precision :: charge, dist_square, inv_dist, inv_dist_cube
  double precision :: classical(3), rvec(3), source_xyz(3)
  integer :: igridpoint, isource

#if defined(GPU) || defined(MPIV_GPU)
  ! Keep small grids on the original path; block dense grids for host SIMD.
  if (npoints >= 128) then
    call add_classical_efield_blocked(npoints,xyz_points,efield)
    return
  endif
#endif

  do igridpoint=1,npoints
    classical(:) = 0.0d0
    do isource=1,natom+quick_molspec%nextatom
      if(isource<=natom)then
        charge = quick_molspec%chg(isource)
        source_xyz(:) = xyz(:,isource)
      else
        charge = quick_molspec%extchg(isource-natom)
        source_xyz(:) = quick_molspec%extxyz(:,isource-natom)
      endif
      rvec(:) = xyz_points(:,igridpoint) - source_xyz(:)
      dist_square = dot_product(rvec,rvec)
      inv_dist = 1.0d0/sqrt(dist_square)
      inv_dist_cube = inv_dist/dist_square
      classical(:) = classical(:) + charge*rvec(:)*inv_dist_cube
    enddo
    efield(:,igridpoint) = efield(:,igridpoint) + classical(:)
  end do

 end subroutine add_classical_efield

!--------------------------------------------------------------------------------!
! Add G_ij=dE_i/dC_j from nuclei and EXTCHARGES.                          !
!--------------------------------------------------------------------------------!
 subroutine add_classical_efg(npoints,xyz_points,efg)
  use quick_molspec_module, only: natom, quick_molspec, xyz
  implicit none

  integer, intent(in) :: npoints
  double precision, intent(in) :: xyz_points(:,:)
  double precision, intent(inout) :: efg(:,:,:)

  double precision :: charge, dist_square, inv_dist, inv_dist_cube, inv_dist_fifth
  double precision :: classical(3,3), rvec(3), source_xyz(3)
  integer :: igridpoint, isource, i, j

#if defined(GPU) || defined(MPIV_GPU)
  ! Keep small grids on the original path; block dense grids for host SIMD.
  if (npoints >= 128) then
    call add_classical_efg_blocked(npoints,xyz_points,efg)
    return
  endif
#endif

  do igridpoint=1,npoints
    classical(:,:) = 0.0d0
    do isource=1,natom+quick_molspec%nextatom
      if(isource<=natom)then
        charge = quick_molspec%chg(isource)
        source_xyz(:) = xyz(:,isource)
      else
        charge = quick_molspec%extchg(isource-natom)
        source_xyz(:) = quick_molspec%extxyz(:,isource-natom)
      endif
      rvec(:) = xyz_points(:,igridpoint) - source_xyz(:)
      dist_square = dot_product(rvec,rvec)
      inv_dist = 1.0d0/sqrt(dist_square)
      inv_dist_cube = inv_dist/dist_square
      inv_dist_fifth = inv_dist_cube/dist_square
      do i=1,3
        do j=1,3
          classical(i,j) = classical(i,j) - 3.0d0*charge*rvec(i)*rvec(j)*inv_dist_fifth
        enddo
        classical(i,i) = classical(i,i) + charge*inv_dist_cube
      enddo
    enddo
    efg(:,:,igridpoint) = efg(:,:,igridpoint) + classical(:,:)
  end do

 end subroutine add_classical_efg

#if defined(GPU) || defined(MPIV_GPU)
!------------------------------------------------------------------------!
! Dense GPU grids: accumulate independent points in contiguous blocks.   !
! Each point retains the original source order and arithmetic, including !
! all nine tensor components and the final add to the electronic result. !
! CPU reference routines above remain unchanged.                         !
!------------------------------------------------------------------------!
 subroutine add_classical_efield_blocked(npoints,xyz_points,efield)
  use quick_molspec_module, only: natom, quick_molspec, xyz
  implicit none
  integer, intent(in) :: npoints
  double precision, intent(in) :: xyz_points(:,:)
  double precision, intent(inout) :: efield(:,:)
  integer, parameter :: POINT_BLOCK = 128
  double precision :: classical(POINT_BLOCK,3), point_xyz(POINT_BLOCK,3)
  double precision :: charge, dist_square, inv_dist, inv_dist_cube
  double precision :: rvec(3), source_xyz(3)
  integer :: ibegin, nblock, ipoint, igridpoint, isource
  do ibegin=1,npoints,POINT_BLOCK
    nblock = min(POINT_BLOCK,npoints-ibegin+1)
    classical(:,:) = 0.0d0
    do ipoint=1,nblock
      point_xyz(ipoint,:) = xyz_points(:,ibegin+ipoint-1)
    enddo
    do isource=1,natom+quick_molspec%nextatom
      if(isource<=natom)then
        charge = quick_molspec%chg(isource)
        source_xyz(:) = xyz(:,isource)
      else
        charge = quick_molspec%extchg(isource-natom)
        source_xyz(:) = quick_molspec%extxyz(:,isource-natom)
      endif
      do ipoint=1,nblock
        rvec(:) = point_xyz(ipoint,:) - source_xyz(:)
        dist_square = dot_product(rvec,rvec)
        inv_dist = 1.0d0/sqrt(dist_square)
        inv_dist_cube = inv_dist/dist_square
        classical(ipoint,1) = classical(ipoint,1) + charge*rvec(1)*inv_dist_cube
        classical(ipoint,2) = classical(ipoint,2) + charge*rvec(2)*inv_dist_cube
        classical(ipoint,3) = classical(ipoint,3) + charge*rvec(3)*inv_dist_cube
      enddo
    enddo
    do ipoint=1,nblock
      igridpoint = ibegin+ipoint-1
      efield(:,igridpoint) = efield(:,igridpoint) + classical(ipoint,:)
    enddo
  enddo
 end subroutine add_classical_efield_blocked
 subroutine add_classical_efg_blocked(npoints,xyz_points,efg)
  use quick_molspec_module, only: natom, quick_molspec, xyz
  implicit none
  integer, intent(in) :: npoints
  double precision, intent(in) :: xyz_points(:,:)
  double precision, intent(inout) :: efg(:,:,:)
  integer, parameter :: POINT_BLOCK = 128
  double precision :: classical(POINT_BLOCK,3,3), point_xyz(POINT_BLOCK,3)
  double precision :: charge, dist_square, inv_dist, inv_dist_cube, inv_dist_fifth
  double precision :: rvec(3), source_xyz(3)
  integer :: ibegin, nblock, ipoint, igridpoint, isource
  do ibegin=1,npoints,POINT_BLOCK
    nblock = min(POINT_BLOCK,npoints-ibegin+1)
    classical(:,:,:) = 0.0d0
    do ipoint=1,nblock
      point_xyz(ipoint,:) = xyz_points(:,ibegin+ipoint-1)
    enddo
    do isource=1,natom+quick_molspec%nextatom
      if(isource<=natom)then
        charge = quick_molspec%chg(isource)
        source_xyz(:) = xyz(:,isource)
      else
        charge = quick_molspec%extchg(isource-natom)
        source_xyz(:) = quick_molspec%extxyz(:,isource-natom)
      endif
      do ipoint=1,nblock
        rvec(:) = point_xyz(ipoint,:) - source_xyz(:)
        dist_square = dot_product(rvec,rvec)
        inv_dist = 1.0d0/sqrt(dist_square)
        inv_dist_cube = inv_dist/dist_square
        inv_dist_fifth = inv_dist_cube/dist_square
        classical(ipoint,1,1) = classical(ipoint,1,1) - 3.0d0*charge*rvec(1)*rvec(1)*inv_dist_fifth
        classical(ipoint,1,2) = classical(ipoint,1,2) - 3.0d0*charge*rvec(1)*rvec(2)*inv_dist_fifth
        classical(ipoint,1,3) = classical(ipoint,1,3) - 3.0d0*charge*rvec(1)*rvec(3)*inv_dist_fifth
        classical(ipoint,1,1) = classical(ipoint,1,1) + charge*inv_dist_cube
        classical(ipoint,2,1) = classical(ipoint,2,1) - 3.0d0*charge*rvec(2)*rvec(1)*inv_dist_fifth
        classical(ipoint,2,2) = classical(ipoint,2,2) - 3.0d0*charge*rvec(2)*rvec(2)*inv_dist_fifth
        classical(ipoint,2,3) = classical(ipoint,2,3) - 3.0d0*charge*rvec(2)*rvec(3)*inv_dist_fifth
        classical(ipoint,2,2) = classical(ipoint,2,2) + charge*inv_dist_cube
        classical(ipoint,3,1) = classical(ipoint,3,1) - 3.0d0*charge*rvec(3)*rvec(1)*inv_dist_fifth
        classical(ipoint,3,2) = classical(ipoint,3,2) - 3.0d0*charge*rvec(3)*rvec(2)*inv_dist_fifth
        classical(ipoint,3,3) = classical(ipoint,3,3) - 3.0d0*charge*rvec(3)*rvec(3)*inv_dist_fifth
        classical(ipoint,3,3) = classical(ipoint,3,3) + charge*inv_dist_cube
      enddo
    enddo
    do ipoint=1,nblock
      igridpoint = ibegin+ipoint-1
      efg(:,:,igridpoint) = efg(:,:,igridpoint) + classical(ipoint,:,:)
    enddo
  enddo
 end subroutine add_classical_efg_blocked
#endif

 !---------------------------------------------------------------------------------------------!
 ! This subroutine formats and prints the EFIELD data to file.efield                           !
 !---------------------------------------------------------------------------------------------!
 subroutine print_efield(efield,npoints,xyz_points)
  use quick_method_module, only: quick_method
  use quick_files_module, only: ioutfile, iEFIELDFile, efieldFileName
  use quick_constants_module, only: BOHRS_TO_A

  implicit none
  integer, intent(in) :: npoints
  double precision, intent(in) :: efield(:,:), xyz_points(:,:)

  integer :: igridpoint
  double precision :: Cx, Cy, Cz

  if (quick_method%density_surface) then
    write (ioutfile,'(" *** Printing Electric Field (EFIELD) &
            &[a.u.] on electron-density surface ",A,x,"***")') &
      trim(efieldFileName)
  else
    write (ioutfile,'(" *** Printing Electric Field (EFIELD) &
            &[a.u.] on grid ",A,x,"***")') trim(efieldFileName)
  endif
  if (quick_method%extgrid_angstrom)  then
    if (quick_method%density_surface) then
      write (iEFIELDFile,'(/," ELECTRIC FIELD CALCULATION (EFIELD) &
              &[atomic units] with coordinates of the points on &
              &electron-density surface in angstroms")')
    else
      write (iEFIELDFile,'(/," ELECTRIC FIELD CALCULATION (EFIELD) &
              &[atomic units] with coordinates in angstroms")')
    endif
    write (iEFIELDFile,'(100("-"))')
    write (iEFIELDFile,'(6x,"X[A]",10x,"Y[A]",9x,"Z[A]",16x, &
            &"EFIELD_X",12x, "EFIELD_Y",8x,"EFIELD_Z")')
  else
    write (iEFIELDFile,'(/," ELECTRIC FIELD CALCULATION (EFIELD) &
            &[atomic units] ")')
    write (iEFIELDFile,'(100("-"))')
    write (iEFIELDFile,'(9x,"X",13x,"Y",12x,"Z",16x, &
            &"EFIELD_X",12x, "EFIELD_Y",8x,"EFIELD_Z")')
  endif

  do igridpoint = 1, npoints
    if (quick_method%extgrid_angstrom)  then
      Cx = xyz_points(1,igridpoint)*BOHRS_TO_A
      Cy = xyz_points(2,igridpoint)*BOHRS_TO_A
      Cz = xyz_points(3,igridpoint)*BOHRS_TO_A
    else
      Cx = xyz_points(1,igridpoint)
      Cy = xyz_points(2,igridpoint)
      Cz = xyz_points(3,igridpoint)
    endif

    write(iEFIELDFile,'(2x,6(ES14.6,1x))') Cx, Cy, Cz, efield(1,igridpoint), &
      efield(2,igridpoint), efield(3,igridpoint)
  end do

 end subroutine print_efield

 !---------------------------------------------------------------------------------------------!
 ! This subroutine formats and prints the EFG data to file.efg                                 !
 !---------------------------------------------------------------------------------------------!
 subroutine print_efg(efg,npoints,xyz_points)
  use quick_method_module, only: quick_method
  use quick_files_module, only: ioutfile, iEFGFile, efgFileName
  use quick_constants_module, only: BOHRS_TO_A

  implicit none
  integer, intent(in) :: npoints
  double precision, intent(in) :: efg(:,:,:), xyz_points(:,:)

  integer :: igridpoint
  double precision :: Cx, Cy, Cz

  if (quick_method%density_surface) then
    write (ioutfile,'(" *** Printing Electric Field Gradient (EFG) &
            &[a.u.] on electron-density surface ",A,x,"***")') &
      trim(efgFileName)
  else
    write (ioutfile,'(" *** Printing Electric Field Gradient (EFG) &
            &[a.u.] on grid ",A,x,"***")') trim(efgFileName)
  endif
  if (quick_method%extgrid_angstrom)  then
    if (quick_method%density_surface) then
      write (iEFGFile,'(/," ELECTRIC FIELD GRADIENT CALCULATION (EFG) &
              &[atomic units] with coordinates of the points on &
              &electron-density surface in angstroms")')
    else
      write (iEFGFile,'(/," ELECTRIC FIELD GRADIENT CALCULATION (EFG) &
              &[atomic units] with coordinates in angstroms")')
    endif
    write (iEFGFile,'(140("-"))')
    write (iEFGFile,'(6x,"X[A]",10x,"Y[A]",9x,"Z[A]",16x, &
            &"EFG_XX",10x,"EFG_XY",10x,"EFG_XZ",10x, &
            &"EFG_YX",10x,"EFG_YY",10x,"EFG_YZ",10x, &
            &"EFG_ZX",10x,"EFG_ZY",10x,"EFG_ZZ")')
  else
    write (iEFGFile,'(/," ELECTRIC FIELD GRADIENT CALCULATION (EFG) &
            &[atomic units] ")')
    write (iEFGFile,'(140("-"))')
    write (iEFGFile,'(9x,"X",13x,"Y",12x,"Z",16x, &
            &"EFG_XX",10x,"EFG_XY",10x,"EFG_XZ",10x, &
            &"EFG_YX",10x,"EFG_YY",10x,"EFG_YZ",10x, &
            &"EFG_ZX",10x,"EFG_ZY",10x,"EFG_ZZ")')
  endif

  do igridpoint = 1, npoints
    if (quick_method%extgrid_angstrom)  then
      Cx = xyz_points(1,igridpoint)*BOHRS_TO_A
      Cy = xyz_points(2,igridpoint)*BOHRS_TO_A
      Cz = xyz_points(3,igridpoint)*BOHRS_TO_A
    else
      Cx = xyz_points(1,igridpoint)
      Cy = xyz_points(2,igridpoint)
      Cz = xyz_points(3,igridpoint)
    endif

    write(iEFGFile,'(2x,12(ES14.6,1x))') Cx, Cy, Cz, &
      efg(1,1,igridpoint), efg(1,2,igridpoint), efg(1,3,igridpoint), &
      efg(2,1,igridpoint), efg(2,2,igridpoint), efg(2,3,igridpoint), &
      efg(3,1,igridpoint), efg(3,2,igridpoint), efg(3,3,igridpoint)
  end do

 end subroutine print_efg
end module quick_oeproperties_module
