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
#ifdef MPIV
   use mpi
   use quick_mpi_module, only: master, mpierror
#endif
   implicit none

   logical fail
   integer ierr, nbasis, alloc_status

   if (quick_method%ext_grid) then
      call compute_oeprop_grid(quick_molspec%nextpoint,quick_molspec%extpointxyz)
   else if (quick_method%density_surface) then

#ifdef MPIV
      if(master)then
#endif
        call generate_density_surfaces()
#ifdef MPIV
      endif
      call MPI_BCAST(quick_molspec%nvdwpoint,1,mpi_integer,0,MPI_COMM_WORLD,mpierror)
      if(.not.master)then
        allocate(quick_molspec%vdwpointxyz(3,quick_molspec%nvdwpoint), stat=alloc_status)

        if(alloc_status /= 0) then
          call PrtErr(OUTFILEHANDLE, '!!quick_molspec%vdwpointxyz array reallocation failed in compute_oeprop!!')
          call quick_exit(OUTFILEHANDLE,1)
        endif

      endif
      call MPI_BCAST(quick_molspec%vdwpointxyz,quick_molspec%nvdwpoint*3,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
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
      call MPI_BCAST(quick_molspec%nvdwpoint,1,mpi_integer,0,MPI_COMM_WORLD,mpierror)
      if(.not.master)then
        allocate(quick_molspec%vdwpointxyz(3,quick_molspec%nvdwpoint), stat=alloc_status)

        if(alloc_status /= 0) then
          call PrtErr(OUTFILEHANDLE, '!!quick_molspec%vdwpointxyz array reallocation failed in compute_oeprop!!')
          call quick_exit(OUTFILEHANDLE,1)
        endif
      
      endif
      call MPI_BCAST(quick_molspec%vdwpointxyz,quick_molspec%nvdwpoint*3,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
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
#ifdef MPIV
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

   ! Electrostatic Potential
   if (quick_method%esp_grid) then
     call compute_esp(npoints,xyz_points,esp_on_points)
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
     call compute_esp(npoints,xyz_points,esp_on_points)
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
!   The subroutines esp_shell_pair, efield_shell_pair and            !
!   esp_1pdm, efield_1pdm are present in ./include/attrashell.fh     !
!   and ./include/nuclearattra.fh header files respectively.         !
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
 ! This subroutine is called from the main program.                           !
 ! It calls the following subroutines:                                        !
 !     1. esp_nuc: Computes the nuclear contribution to the ESP               !
 !     2. esp_shell_pair: Computes the electronic contribution to the ESP     !
 !----------------------------------------------------------------------------!
 subroutine compute_esp(npoints,xyz_points,esp)
   use quick_timer_module, only : timer_begin, timer_end, timer_cumer
   use quick_basis_module, only: jshell
   use quick_calculated_module, only: quick_qm_struct
#ifdef MPIV
    use mpi
    use quick_basis_module, only: mpi_jshelln, mpi_jshell
    use quick_mpi_module, only: master, mpirank, mpierror
#endif
#if defined(GPU) || defined(MPIV_GPU)
    use quick_method_module, only: quick_method
#endif


   implicit none
   integer :: ierr, alloc_status
   integer :: IIsh, JJsh
   integer :: igridpoint, npoints

   double precision, allocatable :: esp_electronic(:),esp_nuclear(:)
   double precision, intent(in)  :: xyz_points(:,:)
   double precision, intent(out) :: esp(:)
#ifdef MPIV
   double precision, allocatable :: esp_electronic_aggregate(:)
#endif
   integer :: Ish

   ierr = 0
   
   ! Allocates ESP_NUC and ESP_ELEC arrays
   allocate(esp_nuclear(npoints), stat=alloc_status)

   if(alloc_status /= 0) then
     call PrtErr(OUTFILEHANDLE, '!! esp_nuclear array reallocation failed in compute_esp!!')
     call quick_exit(OUTFILEHANDLE,1)
   endif
      
   allocate(esp_electronic(npoints), stat=alloc_status)

   if(alloc_status /= 0) then
     call PrtErr(OUTFILEHANDLE, '!! esp_electronic array reallocation failed in compute_esp!!')
     call quick_exit(OUTFILEHANDLE,1)
   endif
      
#ifdef MPIV
   allocate(esp_electronic_aggregate(npoints))

   if(alloc_status /= 0) then
     call PrtErr(OUTFILEHANDLE, '!! esp_electronic_aggregate array reallocation failed in compute_esp!!')
     call quick_exit(OUTFILEHANDLE,1)
   endif
      
#endif

   ! ESP_ELEC array need initialization as we will be iterating
   ! over shells and updating ESP_ELEC.
   esp_electronic(:) = 0.0d0

   RECORD_TIME(timer_begin%TESPGrid)

   ! Computes ESP_NUC 
   call esp_nuc(npoints, xyz_points, esp_nuclear)

   ! Computes ESP_ELEC
#if defined(GPU) || defined(MPIV_GPU)
   call gpu_upload_oeprop(npoints, xyz_points, esp_electronic, ierr)
   call gpu_upload_density_matrix(quick_qm_struct%dense)
   if (quick_method%UNRST) call gpu_upload_beta_density_matrix(quick_qm_struct%denseb)
   call gpu_get_oeprop(esp_electronic)
#if defined MPIV
   call MPI_REDUCE(esp_electronic, esp_electronic_aggregate, npoints, &
     MPI_double_precision, MPI_SUM, 0, MPI_COMM_WORLD, mpierror)
#endif
   ! Sum over contributions from different shell pairs
#elif defined MPIV
   ! MPI parallellization is performed over shell-pairs
   ! Different processes consider different shell-pairs
   do Ish=1,mpi_jshelln(mpirank)
      IIsh=mpi_jshell(mpirank,Ish)
      do JJsh=IIsh,jshell
         call esp_shell_pair(IIsh, JJsh, npoints, xyz_points, esp_electronic)
      enddo
   enddo
   ! MPI_REDUCE is called to sum over esp_electronic obtained from all the processes
   call MPI_REDUCE(esp_electronic, esp_electronic_aggregate, npoints, &
     MPI_double_precision, MPI_SUM, 0, MPI_COMM_WORLD, mpierror)
#else
   do IIsh = 1, jshell
      do JJsh = IIsh, jshell
        call esp_shell_pair(IIsh, JJsh, npoints, xyz_points, esp_electronic)
      end do
   end do
#endif

   ! Sum the nuclear and electronic part of ESP
#ifdef MPIV
   if (master) then
#endif
   do igridpoint=1,npoints
#ifdef MPIV
     esp(igridpoint) = esp_nuclear(igridpoint)+esp_electronic_aggregate(igridpoint)
#else
     esp(igridpoint) = esp_nuclear(igridpoint)+esp_electronic(igridpoint)
#endif
   end do
#ifdef MPIV
   else
     esp(:) = 0.0d0
   endif
#endif

   RECORD_TIME(timer_end%TESPGrid)
   timer_cumer%TESPGrid=timer_cumer%TESPGrid+timer_end%TESPGrid-timer_begin%TESPGrid

   deallocate(esp_electronic)
   deallocate(esp_nuclear)
#ifdef MPIV
   deallocate(esp_electronic_aggregate)
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
#ifdef MPIV
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
 ! This subroutine calculates V_nuc(r) = sum Z_k/|r-Rk|                  !
 !-----------------------------------------------------------------------!
 subroutine esp_nuc(npoints, xyz_points, esp_nuclear)
   use quick_molspec_module, only: natom, quick_molspec, xyz

   implicit none
   integer, intent(in) :: npoints
   double precision, intent(in)  :: xyz_points(:,:)
   double precision, intent(out) :: esp_nuclear(:)

   double precision :: distance
   double precision, external :: rootSquare
   integer :: inucleus, igridpoint

   do igridpoint = 1, npoints
     esp_nuclear(igridpoint) = 0.d0
     do inucleus=1,natom+quick_molspec%nextatom
       if(inucleus<=natom)then
         distance = rootSquare(xyz(1:3,inucleus), xyz_points(1:3,igridpoint), 3)
         esp_nuclear(igridpoint) = esp_nuclear(igridpoint) + quick_molspec%chg(inucleus) / distance
       else
         distance = rootSquare(quick_molspec%extxyz(1:3,inucleus-natom), xyz_points(1:3,igridpoint), 3)
         esp_nuclear(igridpoint) = esp_nuclear(igridpoint) + quick_molspec%extchg(inucleus-natom) / distance
       endif
     enddo
   enddo

 end subroutine esp_nuc


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
   use quick_mpi_module, only: master, mpirank, mpierror
#endif
#if defined(GPU) || defined(MPIV_GPU)
   use quick_method_module, only: quick_method
#endif

   implicit none
   integer :: ierr, IIsh, JJsh
   integer :: igridpoint, npoints, alloc_status
   double precision, intent(in) :: xyz_points(:,:)
   double precision, intent(out) :: efield(:,:)
   double precision, allocatable :: efield_electronic(:,:)
   double precision, allocatable :: efield_nuclear(:,:)
#ifdef MPIV
   double precision, allocatable :: efield_electronic_aggregate(:,:)
#endif
   integer :: Ish

   allocate(efield_electronic(3,npoints), stat=alloc_status)
   if(alloc_status /= 0) then
     call PrtErr(OUTFILEHANDLE, '!! efield_electronic array allocation failed in compute_efield_values!!')
     call quick_exit(OUTFILEHANDLE,1)
   endif

   allocate(efield_nuclear(3,npoints), stat=alloc_status)
   if(alloc_status /= 0) then
     call PrtErr(OUTFILEHANDLE, '!! efield_nuclear array allocation failed in compute_efield_values!!')
     call quick_exit(OUTFILEHANDLE,1)
   endif

#ifdef MPIV
   allocate(efield_electronic_aggregate(3,npoints), stat=alloc_status)
   if(alloc_status /= 0) then
     call PrtErr(OUTFILEHANDLE, '!! efield_electronic_aggregate array allocation failed in compute_efield_values!!')
     call quick_exit(OUTFILEHANDLE,1)
   endif
#endif

   ! Initializes efield_electronic as it will be updated to account
   ! for contributions from different shell-pairs.
   efield_electronic(:,:) = 0.0d0

   ! Computes efield_nuclear.
   do igridpoint=1,npoints
     call efield_nuc(igridpoint,xyz_points,efield_nuclear(1,igridpoint))
   end do

   ! Computes EFIELD_ELEC by summing over contributions from shell-pairs.
#if defined(GPU) || defined(MPIV_GPU)
   ierr = 0
   call gpu_upload_oeprop_efield(npoints, xyz_points, efield_electronic, ierr)
   call gpu_upload_density_matrix(quick_qm_struct%dense)
   if (quick_method%UNRST) call gpu_upload_beta_density_matrix(quick_qm_struct%denseb)
   call gpu_get_oeprop_efield(efield_electronic)
#if defined MPIV
   call MPI_REDUCE(efield_electronic, efield_electronic_aggregate, 3*npoints, &
     MPI_double_precision, MPI_SUM, 0, MPI_COMM_WORLD, mpierror)
#endif
#elif defined MPIV
   do Ish=1,mpi_jshelln(mpirank)
      IIsh=mpi_jshell(mpirank,Ish)
      do JJsh=IIsh,jshell
         call efield_shell_pair(IIsh,JJsh,npoints,xyz_points,efield_electronic)
      enddo
   enddo
   call MPI_REDUCE(efield_electronic, efield_electronic_aggregate, 3*npoints, &
     MPI_double_precision, MPI_SUM, 0, MPI_COMM_WORLD, mpierror)
#else
   do IIsh = 1, jshell
      do JJsh = IIsh, jshell
        call efield_shell_pair(IIsh,JJsh,npoints,xyz_points,efield_electronic)
      end do
   end do
#endif

   ! Sum the nuclear and electronic parts of EFIELD.
#ifdef MPIV
   if (master) then
     efield(:,:) = efield_nuclear(:,:) + efield_electronic_aggregate(:,:)
   else
     efield(:,:) = 0.0d0
   endif
#else
   efield(:,:) = efield_nuclear(:,:) + efield_electronic(:,:)
#endif

   deallocate(efield_electronic)
   deallocate(efield_nuclear)
#ifdef MPIV
   deallocate(efield_electronic_aggregate)
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
   use quick_mpi_module, only: master, mpirank, mpierror
#endif
#if defined(GPU) || defined(MPIV_GPU)
   use quick_method_module, only: quick_method
#endif

   implicit none
   integer :: ierr, IIsh, JJsh
   integer :: igridpoint, npoints, alloc_status
   double precision, intent(in) :: xyz_points(:,:)
   double precision, intent(out) :: efg(:,:,:)
   double precision, allocatable :: efg_electronic(:,:,:)
   double precision, allocatable :: efg_nuclear(:,:,:)
#ifdef MPIV
   double precision, allocatable :: efg_electronic_aggregate(:,:,:)
#endif
   integer :: Ish

   allocate(efg_electronic(3,3,npoints), stat=alloc_status)
   if(alloc_status /= 0) then
     call PrtErr(OUTFILEHANDLE, '!! efg_electronic array allocation failed in compute_efg_values_analytic!!')
     call quick_exit(OUTFILEHANDLE,1)
   endif

   allocate(efg_nuclear(3,3,npoints), stat=alloc_status)
   if(alloc_status /= 0) then
     call PrtErr(OUTFILEHANDLE, '!! efg_nuclear array allocation failed in compute_efg_values_analytic!!')
     call quick_exit(OUTFILEHANDLE,1)
   endif

#ifdef MPIV
   allocate(efg_electronic_aggregate(3,3,npoints), stat=alloc_status)
   if(alloc_status /= 0) then
     call PrtErr(OUTFILEHANDLE, '!! efg_electronic_aggregate array allocation failed in compute_efg_values_analytic!!')
     call quick_exit(OUTFILEHANDLE,1)
   endif
#endif

   ! Initializes efg_electronic as it will be updated to account
   ! for contributions from different shell-pairs.
   efg_electronic(:,:,:) = 0.0d0

   ! Computes efg_nuclear.
   do igridpoint=1,npoints
     call efg_nuc(igridpoint,xyz_points,efg_nuclear(1,1,igridpoint))
   end do

   ! Computes EFG_ELEC by summing over contributions from shell-pairs.
#if defined(GPU) || defined(MPIV_GPU)
   ierr = 0
   call gpu_upload_oeprop_efg(npoints, xyz_points, efg_electronic, ierr)
   call gpu_upload_density_matrix(quick_qm_struct%dense)
   if (quick_method%UNRST) call gpu_upload_beta_density_matrix(quick_qm_struct%denseb)
   call gpu_get_oeprop_efg(efg_electronic)
#if defined MPIV
   call MPI_REDUCE(efg_electronic, efg_electronic_aggregate, 9*npoints, &
     MPI_double_precision, MPI_SUM, 0, MPI_COMM_WORLD, mpierror)
#endif
#elif defined MPIV
   do Ish=1,mpi_jshelln(mpirank)
      IIsh=mpi_jshell(mpirank,Ish)
      do JJsh=IIsh,jshell
         call efg_shell_pair(IIsh,JJsh,npoints,xyz_points,efg_electronic)
      enddo
   enddo
   call MPI_REDUCE(efg_electronic, efg_electronic_aggregate, 9*npoints, &
     MPI_double_precision, MPI_SUM, 0, MPI_COMM_WORLD, mpierror)
#else
   do IIsh = 1, jshell
      do JJsh = IIsh, jshell
        call efg_shell_pair(IIsh,JJsh,npoints,xyz_points,efg_electronic)
      end do
   end do
#endif

   ! Sum the nuclear and electronic parts of EFG.
#ifdef MPIV
   if (master) then
     efg(:,:,:) = efg_nuclear(:,:,:) + efg_electronic_aggregate(:,:,:)
   else
     efg(:,:,:) = 0.0d0
   endif
#else
   efg(:,:,:) = efg_nuclear(:,:,:) + efg_electronic(:,:,:)
#endif

   deallocate(efg_electronic)
   deallocate(efg_nuclear)
#ifdef MPIV
   deallocate(efg_electronic_aggregate)
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
! This subroutine calculates EFIELD_nuc(r) = sum Z_k*(r-Rk)/(|r-Rk|^3)   !
!------------------------------------------------------------------------!
 subroutine efield_nuc(igridpoint,xyz_points,efield_nuclear_term)
  use quick_molspec_module, only: natom, quick_molspec, xyz
  implicit none

  integer, intent(in) :: igridpoint
  double precision, intent(in) :: xyz_points(:,:)
  double precision, intent(out) :: efield_nuclear_term(3)

  double precision :: dist_square, inv_dist, inv_dist_cube
  double precision :: rx_nuc_gridpoint, ry_nuc_gridpoint, rz_nuc_gridpoint
  integer :: inucleus

  efield_nuclear_term = 0.0d0

  do inucleus = 1, natom+quick_molspec%nextatom
    if(inucleus<=natom)then
      rx_nuc_gridpoint = xyz_points(1,igridpoint) - xyz(1,inucleus)
      ry_nuc_gridpoint = xyz_points(2,igridpoint) - xyz(2,inucleus)
      rz_nuc_gridpoint = xyz_points(3,igridpoint) - xyz(3,inucleus)
      dist_square = rx_nuc_gridpoint*rx_nuc_gridpoint + &
        ry_nuc_gridpoint*ry_nuc_gridpoint + rz_nuc_gridpoint*rz_nuc_gridpoint
      inv_dist = 1.0d0/dsqrt(dist_square)
      inv_dist_cube = inv_dist*inv_dist*inv_dist

      ! Compute nuclear components to EFIELD_NUCLEAR.
      efield_nuclear_term(1) = efield_nuclear_term(1) + quick_molspec%chg(inucleus)*(rx_nuc_gridpoint*inv_dist_cube)
      efield_nuclear_term(2) = efield_nuclear_term(2) + quick_molspec%chg(inucleus)*(ry_nuc_gridpoint*inv_dist_cube)
      efield_nuclear_term(3) = efield_nuclear_term(3) + quick_molspec%chg(inucleus)*(rz_nuc_gridpoint*inv_dist_cube)
    else
      rx_nuc_gridpoint = xyz_points(1,igridpoint) - quick_molspec%extxyz(1,inucleus-natom)
      ry_nuc_gridpoint = xyz_points(2,igridpoint) - quick_molspec%extxyz(2,inucleus-natom)
      rz_nuc_gridpoint = xyz_points(3,igridpoint) - quick_molspec%extxyz(3,inucleus-natom)
      dist_square = rx_nuc_gridpoint*rx_nuc_gridpoint + &
        ry_nuc_gridpoint*ry_nuc_gridpoint + rz_nuc_gridpoint*rz_nuc_gridpoint
      inv_dist = 1.0d0/dsqrt(dist_square)
      inv_dist_cube = inv_dist*inv_dist*inv_dist

      ! Compute external-charge components to EFIELD_NUCLEAR.
      efield_nuclear_term(1) = efield_nuclear_term(1) + quick_molspec%extchg(inucleus-natom)*(rx_nuc_gridpoint*inv_dist_cube)
      efield_nuclear_term(2) = efield_nuclear_term(2) + quick_molspec%extchg(inucleus-natom)*(ry_nuc_gridpoint*inv_dist_cube)
      efield_nuclear_term(3) = efield_nuclear_term(3) + quick_molspec%extchg(inucleus-natom)*(rz_nuc_gridpoint*inv_dist_cube)
    endif
  end do

 end subroutine efield_nuc

!--------------------------------------------------------------------------------!
! This subroutine calculates EFG_nuc(r) = d EFIELD_nuc(r) / dr on each grid point.!
!--------------------------------------------------------------------------------!
 subroutine efg_nuc(igridpoint,xyz_points,efg_nuclear_term)
  use quick_molspec_module, only: natom, quick_molspec, xyz
  implicit none

  integer, intent(in) :: igridpoint
  double precision, intent(in) :: xyz_points(:,:)
  double precision, intent(out) :: efg_nuclear_term(3,3)

  double precision :: charge, dist_square, inv_dist, inv_dist_cube, inv_dist_fifth
  double precision :: rx_nuc_gridpoint, ry_nuc_gridpoint, rz_nuc_gridpoint
  double precision :: rvec(3)
  integer :: inucleus, i, j

  efg_nuclear_term(:,:) = 0.0d0

  do inucleus = 1, natom+quick_molspec%nextatom
    if(inucleus<=natom)then
      charge = quick_molspec%chg(inucleus)

      rx_nuc_gridpoint = xyz_points(1,igridpoint) - xyz(1,inucleus)
      ry_nuc_gridpoint = xyz_points(2,igridpoint) - xyz(2,inucleus)
      rz_nuc_gridpoint = xyz_points(3,igridpoint) - xyz(3,inucleus)
    else
      charge = quick_molspec%extchg(inucleus-natom)

      rx_nuc_gridpoint = xyz_points(1,igridpoint) - quick_molspec%extxyz(1,inucleus-natom)
      ry_nuc_gridpoint = xyz_points(2,igridpoint) - quick_molspec%extxyz(2,inucleus-natom)
      rz_nuc_gridpoint = xyz_points(3,igridpoint) - quick_molspec%extxyz(3,inucleus-natom)
    endif

    dist_square = rx_nuc_gridpoint*rx_nuc_gridpoint + &
      ry_nuc_gridpoint*ry_nuc_gridpoint + rz_nuc_gridpoint*rz_nuc_gridpoint
    inv_dist = 1.0d0/dsqrt(dist_square)
    inv_dist_cube = inv_dist*inv_dist*inv_dist
    inv_dist_fifth = inv_dist_cube/dist_square
    rvec(1) = rx_nuc_gridpoint
    rvec(2) = ry_nuc_gridpoint
    rvec(3) = rz_nuc_gridpoint

    ! Compute nuclear and external-charge components to EFG_NUCLEAR.
    do i=1,3
      do j=1,3
        efg_nuclear_term(i,j) = efg_nuclear_term(i,j) - &
          3.0d0*charge*rvec(i)*rvec(j)*inv_dist_fifth
      end do
      efg_nuclear_term(i,i) = efg_nuclear_term(i,i) + charge*inv_dist_cube
    end do
  end do

 end subroutine efg_nuc

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
