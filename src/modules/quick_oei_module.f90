#include "util.fh"
!---------------------------------------------------------------------!
! Created by Madu Manathunga on 03/24/2021                            !
!                                                                     !
! Previous contributors: Yipu Miao, Xio He, Alessandro Genoni,        !
!                         Ken Ayers & Ed Brothers                     !
!                                                                     ! 
! Copyright (C) 2021-2022 Merz lab                                    !
! Copyright (C) 2021-2022 Götz lab                                    !
!                                                                     !
! This Source Code Form is subject to the terms of the Mozilla Public !
! License, v. 2.0. If a copy of the MPL was not distributed with this !
! file, You can obtain one at http://mozilla.org/MPL/2.0/.            !
!_____________________________________________________________________!

!---------------------------------------------------------------------!
! This module contains all one electron integral (oei) & oei gradient ! 
! code.                                                               !
!---------------------------------------------------------------------!

module quick_oei_module

  implicit double precision(a-h,o-z)
  private

  public :: get1eEnergy, get1e, attrashellopt, ekinetic, kineticO, attrashell
  public :: get_external_efield_gradient
  public :: add_external_field_operator, get_external_field_gpu_counts
  public :: check_external_field_finite
  public :: bCalc1e

  logical :: bCalc1e = .false.

#if defined(CUDA) || defined(CUDA_MPIV)
  interface
     subroutine gpu_external_field_matrix(nb,na,maxprim,nc,ang,centers,expo,coeff,atoms, &
        field,origin,owned,cutoff,matrix,ierr)
        implicit none
        integer, intent(in) :: nb, na, maxprim, nc(nb), ang(3,nb), centers(nb), owned(nb)
        double precision, intent(in) :: expo(maxprim,nb), coeff(maxprim,nb), atoms(3,na)
        double precision, intent(in) :: field(3), origin(3), cutoff
        double precision, intent(inout) :: matrix(nb,nb)
        integer, intent(out) :: ierr
     end subroutine gpu_external_field_matrix
     subroutine gpu_external_field_gradient(nb,na,maxprim,nc,ang,centers,expo,coeff,atoms, &
        field,origin,owned,cutoff,density,gradient,ierr)
        implicit none
        integer, intent(in) :: nb, na, maxprim, nc(nb), ang(3,nb), centers(nb), owned(nb)
        double precision, intent(in) :: expo(maxprim,nb), coeff(maxprim,nb), atoms(3,na)
        double precision, intent(in) :: field(3), origin(3), cutoff, density(nb,nb)
        double precision, intent(inout) :: gradient(3,na)
        integer, intent(out) :: ierr
     end subroutine gpu_external_field_gradient
     subroutine gpu_external_field_tensor_matrix(nb,na,maxprim,nc,ang,centers,expo,coeff,atoms, &
        field,origin,tensor,owned,cutoff,matrix,ierr)
        implicit none
        integer, intent(in) :: nb, na, maxprim, nc(nb), ang(3,nb), centers(nb), owned(nb)
        double precision, intent(in) :: expo(maxprim,nb), coeff(maxprim,nb), atoms(3,na)
        double precision, intent(in) :: field(3), origin(3), tensor(3,3), cutoff
        double precision, intent(inout) :: matrix(nb,nb)
        integer, intent(out) :: ierr
     end subroutine gpu_external_field_tensor_matrix
     subroutine gpu_external_field_tensor_gradient(nb,na,maxprim,nc,ang,centers,expo,coeff,atoms, &
        field,origin,tensor,owned,cutoff,density,gradient,ierr)
        implicit none
        integer, intent(in) :: nb, na, maxprim, nc(nb), ang(3,nb), centers(nb), owned(nb)
        double precision, intent(in) :: expo(maxprim,nb), coeff(maxprim,nb), atoms(3,na)
        double precision, intent(in) :: field(3), origin(3), tensor(3,3), cutoff, density(nb,nb)
        double precision, intent(inout) :: gradient(3,na)
        integer, intent(out) :: ierr
     end subroutine gpu_external_field_tensor_gradient
     subroutine gpu_external_field_launch_counts(matrix,gradient)
        implicit none
        integer, intent(out) :: matrix, gradient
     end subroutine gpu_external_field_launch_counts
  end interface
#endif

contains

#define OEI
#include "./include/attrashell.fh"
#undef OEI

  !------------------------------------------------
  ! get1eEnergy
  !------------------------------------------------
  subroutine get1eEnergy(deltaO)
     !------------------------------------------------
     ! This subroutine is to get 1e integral
     !------------------------------------------------
     use quick_calculated_module, only: quick_qm_struct
     use quick_method_module,only: quick_method
     use quick_timer_module, only:timer_begin, timer_end, timer_cumer
     use quick_basis_module

     implicit double precision(a-h,o-z)

     logical, intent(in) :: deltaO

     RECORD_TIME(timer_begin%tE)
  
     if(.not. deltaO) quick_qm_struct%E1e=0.0d0
     quick_qm_struct%E1e=quick_qm_struct%E1e+sum2mat(quick_qm_struct%dense,quick_qm_struct%oneElecO,nbasis)

     if (quick_method%unrst) then
       quick_qm_struct%E1e = quick_qm_struct%E1e+sum2mat(quick_qm_struct%denseb,quick_qm_struct%oneElecO,nbasis)
     endif

     quick_qm_struct%Eel=quick_qm_struct%E1e

     RECORD_TIME(timer_end%tE)
     timer_cumer%TE=timer_cumer%TE+timer_end%TE-timer_begin%TE
  
  end subroutine get1eEnergy

subroutine get1e(deltaO)
   use allmod
#ifdef CEW
   use quick_cew_module, only : quick_cew, quick_cew_prescf
#endif
   use quick_mpi_module, only: master
#if defined(MPIV)
   use quick_mpi_module, only: bMPI, quick_comm_rank
   use mpi
#endif
   
   implicit double precision(a-h,o-z)
   double precision :: temp2d(nbasis,nbasis)
   logical, intent(in) :: deltaO

   !------------------------------------------------
   ! This subroutine is to obtain Hcore, and store it
   ! to oneElecO so we don't need to calculate it repeatly for
   ! every scf cycle
   !------------------------------------------------


#ifdef MPIV
   if ((.not.bMPI).or.(nbasis.le.MIN_1E_MPI_BASIS)) then
#endif

     if (master) then
       RECORD_TIME(timer_begin%T1e)
       if(bCalc1e) then

         !=================================================================
         ! Step 1. evaluate 1e integrals
         !-----------------------------------------------------------------
         ! The first part is kinetic part
         ! O(I,J) =  F(I,J) = "KE(I,J)" + IJ
         !-----------------------------------------------------------------
         RECORD_TIME(timer_begin%T1eT)
         do Ibas=1,nbasis
            call kineticO(Ibas)
         enddo
         RECORD_TIME(timer_end%T1eT)


         !-----------------------------------------------------------------
         ! The second part is attraction part
         !-----------------------------------------------------------------
         RECORD_TIME(timer_begin%T1eV)

#if defined(GPU)
         if(.not. quick_method%hasF) then
           call gpu_get_oei(quick_qm_struct%o)
         else

           do IIsh=1,jshell
              do JJsh=IIsh,jshell
                 call attrashell(IIsh,JJsh)
              enddo
           enddo
         endif
#else
         do IIsh=1,jshell
            do JJsh=IIsh,jshell
               call attrashell(IIsh,JJsh)
            enddo
         enddo
#endif

         RECORD_TIME(timer_end%T1eV)

         ! The external electric field is a one-electron perturbation,
         ! h^F = F.s + s^T G s/2, s = r-r0, added after the usual T+V terms.
         if (quick_method%external_efield) then
            call add_external_field_operator(.false.)
         endif

         timer_cumer%T1eT=timer_cumer%T1eT+timer_end%T1eT-timer_begin%T1eT
         timer_cumer%T1eV=timer_cumer%T1eV+timer_end%T1eV-timer_begin%T1eV

#ifdef CEW
         if ( quick_cew%use_cew ) then
            
            RECORD_TIME(timer_begin%Tcew)

            call quick_cew_prescf()

            RECORD_TIME(timer_end%Tcew)

            timer_cumer%Tcew=timer_cumer%Tcew+timer_end%Tcew-timer_begin%Tcew

         end if
#endif
         
         call copySym(quick_qm_struct%o,nbasis)

         quick_qm_struct%oneElecO(:,:) = quick_qm_struct%o(:,:)

         if (quick_method%debug) then
                write(iOutFile,*) "ONE ELECTRON MATRIX"
                call PriSym(iOutFile,nbasis,quick_qm_struct%oneElecO,'f14.8')
         endif
         bCalc1e=.false.

       else
         if (.not. deltaO) quick_qm_struct%o(:,:)=quick_qm_struct%oneElecO(:,:)
       endif
       RECORD_TIME(timer_end%t1e)

       timer_cumer%T1e=timer_cumer%T1e+timer_end%T1e-timer_begin%T1e
       timer_cumer%TOp = timer_cumer%TOp+timer_end%T1e-timer_begin%T1e
       timer_cumer%TSCF = timer_cumer%TSCF+timer_end%T1e-timer_begin%T1e

     endif
#ifdef MPIV
   else
    RECORD_TIME(timer_begin%t1e)
    if(bCalc1e) then

      !------- MPI/ ALL NODES -------------------

      !=================================================================
      ! Step 1. evaluate 1e integrals
      ! This job is only done on master node since it won't cost much resource
      ! and parallel will even waste more than it saves
      !-----------------------------------------------------------------
      ! The first part is kinetic part
      ! O(I,J) =  F(I,J) = "KE(I,J)" + IJ
      !-----------------------------------------------------------------
      RECORD_TIME(timer_begin%T1eT)

      do i=1,mpi_nbasisn(quick_comm_rank)
         Ibas=mpi_nbasis(quick_comm_rank,i)
         call kineticO(Ibas)
      enddo
      RECORD_TIME(timer_end%T1eT)

      !-----------------------------------------------------------------
      ! The second part is attraction part
      !-----------------------------------------------------------------
      RECORD_TIME(timer_begin%T1eV)

#if defined(MPIV_GPU)
      if(.not. quick_method%hasF) then
        call gpu_get_oei(quick_qm_struct%o)
      else
        do i=1,mpi_jshelln(quick_comm_rank)
           IIsh=mpi_jshell(quick_comm_rank,i)
           do JJsh=IIsh,jshell
              call attrashell(IIsh,JJsh)
           enddo
        enddo
      endif
#else
      do i=1,mpi_jshelln(quick_comm_rank)
         IIsh=mpi_jshell(quick_comm_rank,i)
         do JJsh=IIsh,jshell
            call attrashell(IIsh,JJsh)
         enddo
      enddo
#endif
      RECORD_TIME(timer_end%T1eV)

      ! Each MPI rank contributes only its owned basis-function rows to the
      ! finite-field one-electron operator; the normal reductions then apply.
      if (quick_method%external_efield) then
         call add_external_field_operator(.true.)
      endif

#ifdef CEW

         if ( quick_cew%use_cew ) then

            RECORD_TIME(timer_begin%Tcew)

            call quick_cew_prescf()

            RECORD_TIME(timer_end%Tcew)

            timer_cumer%Tcew=timer_cumer%Tcew+timer_end%Tcew-timer_begin%Tcew

         endif
#endif

      call copySym(quick_qm_struct%o,nbasis)

      quick_qm_struct%oneElecO(:,:) = quick_qm_struct%o(:,:)

      bCalc1e=.false.
      !------- END MPI/ALL NODES ------------
     else
       if (.not. deltaO) quick_qm_struct%o(:,:)=quick_qm_struct%oneElecO(:,:)
     endif


     RECORD_TIME(timer_end%t1e)
     timer_cumer%T1e=timer_cumer%T1e+timer_end%T1e-timer_begin%T1e
     timer_cumer%T1eT=timer_cumer%T1eT+timer_end%T1eT-timer_begin%T1eT
     timer_cumer%T1eV=timer_cumer%T1eV+timer_end%T1eV-timer_begin%T1eV

   endif
#endif
end subroutine get1e


subroutine kineticO(IBAS)

   !------------------------------------------------
   ! This subroutine is to get 1e integral Operator
   !------------------------------------------------
   use allmod
   use quick_overlap_module, only: gpt, opf
   implicit double precision(a-h,o-z)
   integer Ibas 
   integer g_count
   double precision g_table(200)
   double precision :: valopf

   ix = itype(1,Ibas)
   iy = itype(2,Ibas)
   iz = itype(3,Ibas)
   xyzxi = xyz(1,quick_basis%ncenter(Ibas))
   xyzyi = xyz(2,quick_basis%ncenter(Ibas))
   xyzzi = xyz(3,quick_basis%ncenter(Ibas))

   do Jbas=Ibas,nbasis

      jx = itype(1,Jbas)
      jy = itype(2,Jbas)
      jz = itype(3,Jbas)
      xyzxj = xyz(1,quick_basis%ncenter(Jbas))
      xyzyj = xyz(2,quick_basis%ncenter(Jbas))
      xyzzj = xyz(3,quick_basis%ncenter(Jbas))

      g_count = ix+iy+iz+jx+jy+jz+2

      OJI = 0.d0 
      do Icon=1,ncontract(ibas)
         ai = aexp(Icon,Ibas)

         do Jcon=1,ncontract(jbas)
            F = dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas)
           aj = aexp(Jcon,Jbas)

           valopf = opf(ai, aj, dcoeff(Jcon,Jbas), dcoeff(Icon,Ibas), xyzxi, xyzyi, xyzzi, xyzxj, xyzyj, xyzzj)

           if(abs(valopf) .gt. quick_method%coreIntegralCutoff) then 

             ! The first part is the kinetic energy.
             call gpt(aj,ai,xyzxj,xyzyj,xyzzj,xyzxi,xyzyi,xyzzi,Px,Py,Pz,g_count,g_table)

              OJI = OJI + F*ekinetic(aj,   ai, &
                  jx,   jy,   jz,&
                  ix,   iy,   iz, &
                  xyzxj,xyzyj,xyzzj,&
                  xyzxi,xyzyi,xyzzi,Px,Py,Pz,g_table)
           endif
         enddo
      enddo
      quick_qm_struct%o(Jbas,Ibas) = OJI
   enddo

end subroutine kineticO

subroutine externalEFieldO(IBAS)

   !---------------------------------------------------------------------!
   ! Uniform applied field: electronic Hamiltonian (CPU reference).      !
   ! h_F = +F . (r-r0); H_mu,nu += <chi_mu|h_F|chi_nu>.                  !
   ! F is in atomic units; coordinates and r0 are in bohr.               !
   ! Use normalized Cartesian contractions and the Hcore cutoff.         !
   ! Update lower-triangular AO pairs indexed by Ibas.                   !
   ! Nuclear/point-charge energy is added separately in getEnergy;       !
   ! get_external_efield_gradient supplies the matching derivatives.     !
   !---------------------------------------------------------------------!
   use allmod
   use quick_overlap_module, only: opf
   implicit double precision(a-h,o-z)
   integer Ibas
   double precision, external :: xmoment
   double precision :: coef, valopf, moment_x, moment_y, moment_z
   double precision :: field_x, field_y, field_z
   double precision :: origin_x, origin_y, origin_z

   ! Keep every arithmetic operation in the uniform-only reference path.
   if (any(quick_method%external_efield_tensor /= 0.0d0)) then
      call external_efield_tensor_o(ibas)
      return
   endif

   field_x = quick_method%external_efield_vector(1)
   field_y = quick_method%external_efield_vector(2)
   field_z = quick_method%external_efield_vector(3)
   if (abs(field_x)+abs(field_y)+abs(field_z) .eq. 0.0d0) return

   origin_x = quick_method%external_efield_origin(1)
   origin_y = quick_method%external_efield_origin(2)
   origin_z = quick_method%external_efield_origin(3)

   ix = itype(1,Ibas)
   iy = itype(2,Ibas)
   iz = itype(3,Ibas)
   xyzxi = xyz(1,quick_basis%ncenter(Ibas))
   xyzyi = xyz(2,quick_basis%ncenter(Ibas))
   xyzzi = xyz(3,quick_basis%ncenter(Ibas))

   do Jbas=Ibas,nbasis

      jx = itype(1,Jbas)
      jy = itype(2,Jbas)
      jz = itype(3,Jbas)
      xyzxj = xyz(1,quick_basis%ncenter(Jbas))
      xyzyj = xyz(2,quick_basis%ncenter(Jbas))
      xyzzj = xyz(3,quick_basis%ncenter(Jbas))

      OJI = 0.0d0
      do Icon=1,ncontract(ibas)
         ai = aexp(Icon,Ibas)

         do Jcon=1,ncontract(jbas)
            aj = aexp(Jcon,Jbas)
            coef = dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas)

            valopf = opf(ai, aj, dcoeff(Jcon,Jbas), dcoeff(Icon,Ibas), &
               xyzxi, xyzyi, xyzzi, xyzxj, xyzyj, xyzzj)

            if(abs(valopf) .gt. quick_method%coreIntegralCutoff) then
               moment_x = xmoment(aj,ai,jx,jy,jz,ix,iy,iz,1,0,0, &
                  xyzxj,xyzyj,xyzzj,xyzxi,xyzyi,xyzzi, &
                  origin_x,origin_y,origin_z)
               moment_y = xmoment(aj,ai,jx,jy,jz,ix,iy,iz,0,1,0, &
                  xyzxj,xyzyj,xyzzj,xyzxi,xyzyi,xyzzi, &
                  origin_x,origin_y,origin_z)
               moment_z = xmoment(aj,ai,jx,jy,jz,ix,iy,iz,0,0,1, &
                  xyzxj,xyzyj,xyzzj,xyzxi,xyzyi,xyzzi, &
                  origin_x,origin_y,origin_z)

               OJI = OJI + coef*(field_x*moment_x + field_y*moment_y + field_z*moment_z)
            endif
         enddo
      enddo

      quick_qm_struct%o(Jbas,Ibas) = quick_qm_struct%o(Jbas,Ibas) + OJI
   enddo

end subroutine externalEFieldO

subroutine external_efield_tensor_o(ibas)
   !---------------------------------------------------------------------!
   ! Static affine applied field: electronic Hamiltonian, CPU reference. !
   ! s = r-r0; E(s) = F+G*s; phi(s) = -F.s - (s^T G s)/2.              !
   ! For an electron, h = -phi = F.s + (s^T G s)/2.                    !
   ! G_ij = dE_i/dr_j is symmetric, in a.u. of field/bohr; F is in a.u. !
   ! Coordinates and the fixed laboratory origin r0 are in bohr.         !
   ! Contract first and second Cartesian moments with the same original !
   ! normalized primitive coefficients and Hcore screening as F alone.   !
   ! For symmetric G, each off-diagonal moment has coefficient G_ij;     !
   ! each diagonal second moment has coefficient G_ii/2.                !
   !---------------------------------------------------------------------!
   use quick_basis_module, only: quick_basis, nbasis, ncontract, itype, aexp, dcoeff
   use quick_calculated_module, only: quick_qm_struct
   use quick_method_module, only: quick_method
   use quick_molspec_module, only: xyz
   use quick_overlap_module, only: opf
   implicit none
   integer, intent(in) :: ibas
   integer :: jbas, icon, jcon, angular_i(3), angular_j(3)
   double precision :: ai, aj, coef, prefactor, value
   double precision :: center_i(3), center_j(3)

   angular_i = itype(:,ibas)
   center_i = xyz(:,quick_basis%ncenter(ibas))
   do jbas=ibas,nbasis
      angular_j = itype(:,jbas)
      center_j = xyz(:,quick_basis%ncenter(jbas))
      value = 0.0d0
      do icon=1,ncontract(ibas)
         ai = aexp(icon,ibas)
         do jcon=1,ncontract(jbas)
            aj = aexp(jcon,jbas)
            coef = dcoeff(jcon,jbas)*dcoeff(icon,ibas)
            prefactor = opf(ai,aj,dcoeff(jcon,jbas),dcoeff(icon,ibas), &
               center_i(1),center_i(2),center_i(3),center_j(1),center_j(2),center_j(3))
            if (abs(prefactor) <= quick_method%coreIntegralCutoff) cycle
            value = value + coef*external_efield_primitive_moment(aj,ai,angular_j,angular_i, &
               center_j,center_i,quick_method%external_efield_vector, &
               quick_method%external_efield_origin,quick_method%external_efield_tensor)
         enddo
      enddo
      call check_external_field_finite([value],'affine-field contracted AO moment')
      quick_qm_struct%o(jbas,ibas) = quick_qm_struct%o(jbas,ibas) + value
      call check_external_field_finite([quick_qm_struct%o(jbas,ibas)],'affine-field AO matrix')
   enddo
end subroutine external_efield_tensor_o

subroutine add_external_field_operator(distributed)
   !---------------------------------------------------------------------!
   ! Static applied field: CPU/CUDA Hamiltonian dispatch.                !
   ! Add electronic h_F to the current lower-triangular o matrix.        !
   ! With MPI active, distributed selects locally owned Ibas indices;   !
   ! otherwise process all Ibas indices. CUDA uses the same ownership    !
   ! mask, normalized Cartesian basis and primitive cutoff as CPU.       !
   ! Classical charge-field energy is handled in getEnergy.              !
   !---------------------------------------------------------------------!
   use quick_basis_module, only: quick_basis, nbasis, ncontract, itype, aexp, dcoeff
   use quick_calculated_module, only: quick_qm_struct
   use quick_method_module, only: quick_method
   use quick_molspec_module, only: natom, xyz
#ifdef MPIV
   use quick_basis_module, only: mpi_nbasisn, mpi_nbasis
   use quick_mpi_module, only: bMPI, quick_comm_rank
#endif
   implicit none
   logical, intent(in) :: distributed
   integer :: ibas
#ifdef MPIV
   integer :: irow
#endif
#if defined(CUDA) || defined(CUDA_MPIV)
   integer, allocatable :: owned(:)
   integer :: ierr
#endif

   if (.not.quick_method%external_efield) return
#if defined(CUDA) || defined(CUDA_MPIV)
   if (quick_method%bGPU) then
      ! Small MPI Hcore jobs call this on the master with every AO row;
      ! the distributed branch owns exactly the same Ibas rows as the CPU.
      allocate(owned(nbasis),stat=ierr)
      call external_field_gpu_failure(ierr,'allocate Hcore row ownership')
      owned = 1
#ifdef MPIV
      if (distributed .and. bMPI) then
         owned = 0
         do irow=1,mpi_nbasisn(quick_comm_rank)
            owned(mpi_nbasis(quick_comm_rank,irow)) = 1
         enddo
      endif
#endif
      if (any(quick_method%external_efield_tensor /= 0.0d0)) then
         call gpu_external_field_tensor_matrix(nbasis,natom,size(aexp,1),ncontract,itype,quick_basis%ncenter, &
            aexp,dcoeff,xyz,quick_method%external_efield_vector,quick_method%external_efield_origin, &
            quick_method%external_efield_tensor,owned,quick_method%coreIntegralCutoff,quick_qm_struct%o,ierr)
      else
         call gpu_external_field_matrix(nbasis,natom,size(aexp,1),ncontract,itype,quick_basis%ncenter, &
            aexp,dcoeff,xyz,quick_method%external_efield_vector,quick_method%external_efield_origin, &
            owned,quick_method%coreIntegralCutoff,quick_qm_struct%o,ierr)
      endif
      call external_field_gpu_failure(ierr,'applied-field AO matrix')
      return
   endif
#endif
   ! Preserve the established CPU integral evaluator and accumulation order.
#ifdef MPIV
   if (distributed .and. bMPI) then
      do irow=1,mpi_nbasisn(quick_comm_rank)
         ibas = mpi_nbasis(quick_comm_rank,irow)
         call externalEFieldO(ibas)
      enddo
      return
   endif
#endif
   do ibas=1,nbasis
      call externalEFieldO(ibas)
   enddo
end subroutine add_external_field_operator

subroutine get_external_efield_gradient()

   !---------------------------------------------------------------------!
   ! Static affine applied field: analytic nuclear/point-charge gradients.!
   ! At fixed F, symmetric G and r0, add +dE/dR; forces = -dE/dR.        !
   ! h = +F.s + (s^T G s)/2, s = r-r0, uses contracted first/second      !
   ! moments and the Hamiltonian primitive screening. F is in a.u.;      !
   ! G = dE/dr is in a.u. of field/bohr; coordinates/r0 are in bohr.     !
   ! Classical energy -q*(F.s+s^T G s/2) gives -q*(F+G*s) in Eh/bohr.   !
   ! Classical terms occur once on master; MPI ranks distribute          !
   ! AO pairs before the caller's shared gradient reduction.             !
   ! CUDA evaluates the electronic contribution; classical terms         !
   ! remain on the host. The ordinary Pulay contribution from            !
   ! field-polarized SCF orbitals is included by the caller.             !
   !---------------------------------------------------------------------!
   use quick_basis_module, only: quick_basis, nbasis, itype, ncontract, aexp, dcoeff
   use quick_calculated_module, only: quick_qm_struct
   use quick_method_module, only: quick_method
   use quick_molspec_module, only: quick_molspec, natom, xyz
   use quick_overlap_module, only: opf
   use quick_mpi_module, only: master
#ifdef MPIV
   use quick_basis_module, only: mpi_nbasisn, mpi_nbasis
   use quick_mpi_module, only: bMPI, quick_comm_rank
#endif
   implicit none

   integer :: ibas, jbas, icon, jcon, irow, nrows, ia, ja, axis
   integer :: angular_i(3), angular_j(3), shifted(3)
   double precision :: ai, aj, coef, density, prefactor
   double precision :: field(3), origin(3), tensor(3,3), center_i(3), center_j(3)
   double precision :: derivative_i(3), derivative_j(3), value_i, value_j, classical_field(3)
   logical :: has_tensor

   if (.not. quick_method%external_efield) return
   field = quick_method%external_efield_vector
   tensor = quick_method%external_efield_tensor
   has_tensor = any(tensor /= 0.0d0)
   if (all(field == 0.0d0) .and. .not.has_tensor) return
   origin = quick_method%external_efield_origin

   ! Classical terms occur once before the caller's MPI sum. Point charges
   ! have the same field coupling as the corresponding term in Ecore.
   if (master) then
      if (has_tensor) then
         do ia=1,natom
            classical_field = field + matmul(tensor,xyz(:,ia)-origin)
            call check_external_field_finite(classical_field,'affine field at a nucleus')
            quick_qm_struct%gradient(3*ia-2:3*ia) = quick_qm_struct%gradient(3*ia-2:3*ia) &
               - quick_molspec%chg(ia)*classical_field
            call check_external_field_finite(quick_qm_struct%gradient(3*ia-2:3*ia),'affine-field nuclear gradient')
         enddo
         do ia=1,quick_molspec%nextatom
            classical_field = field + matmul(tensor,quick_molspec%extxyz(:,ia)-origin)
            call check_external_field_finite(classical_field,'affine field at an external point charge')
            quick_qm_struct%ptchg_gradient(3*ia-2:3*ia) = quick_qm_struct%ptchg_gradient(3*ia-2:3*ia) &
               - quick_molspec%extchg(ia)*classical_field
            call check_external_field_finite(quick_qm_struct%ptchg_gradient(3*ia-2:3*ia), &
               'affine-field external-charge gradient')
         enddo
      else
         do ia=1,natom
            quick_qm_struct%gradient(3*ia-2:3*ia) = quick_qm_struct%gradient(3*ia-2:3*ia) &
               - quick_molspec%chg(ia)*field
         enddo
         do ia=1,quick_molspec%nextatom
            quick_qm_struct%ptchg_gradient(3*ia-2:3*ia) = quick_qm_struct%ptchg_gradient(3*ia-2:3*ia) &
               - quick_molspec%extchg(ia)*field
         enddo
      endif
   endif

#if defined(CUDA) || defined(CUDA_MPIV)
   if (quick_method%bGPU) then
      call add_external_field_gpu_gradient()
      return
   endif
#endif

   nrows = nbasis
#ifdef MPIV
   if (bMPI) nrows = mpi_nbasisn(quick_comm_rank)
#endif
   do irow=1,nrows
      ibas = irow
#ifdef MPIV
      if (bMPI) ibas = mpi_nbasis(quick_comm_rank,irow)
#endif
      ia = quick_basis%ncenter(ibas)
      angular_i = itype(:,ibas)
      center_i = xyz(:,ia)
      do jbas=ibas,nbasis
         ja = quick_basis%ncenter(jbas)
         angular_j = itype(:,jbas)
         center_j = xyz(:,ja)
         density = quick_qm_struct%dense(jbas,ibas)
         if (quick_method%unrst) density = density + quick_qm_struct%denseb(jbas,ibas)
         if (ibas /= jbas) density = 2.0d0*density
         if (density == 0.0d0) cycle
         derivative_i = 0.0d0
         derivative_j = 0.0d0
         do icon=1,ncontract(ibas)
            ai = aexp(icon,ibas)
            do jcon=1,ncontract(jbas)
               aj = aexp(jcon,jbas)
               coef = dcoeff(jcon,jbas)*dcoeff(icon,ibas)
               prefactor = opf(ai,aj,dcoeff(jcon,jbas),dcoeff(icon,ibas), &
                  center_i(1),center_i(2),center_i(3),center_j(1),center_j(2),center_j(3))
               if (abs(prefactor) <= quick_method%coreIntegralCutoff) cycle

               ! d chi_l / d A_k = 2 alpha chi_(l+e_k) - l_k chi_(l-e_k).
               ! Keep original normalization coefficients for both shifted
               ! polynomials. Same-center pairs also contribute in a field.
               do axis=1,3
                  shifted = angular_i
                  shifted(axis) = shifted(axis) + 1
                  value_i = 2.0d0*ai*external_efield_primitive_moment(aj,ai,angular_j,shifted, &
                     center_j,center_i,field,origin,tensor)
                  if (angular_i(axis) > 0) then
                     shifted(axis) = shifted(axis) - 2
                     value_i = value_i - dble(angular_i(axis))*external_efield_primitive_moment(aj,ai, &
                        angular_j,shifted,center_j,center_i,field,origin,tensor)
                  endif

                  shifted = angular_j
                  shifted(axis) = shifted(axis) + 1
                  value_j = 2.0d0*aj*external_efield_primitive_moment(aj,ai,shifted,angular_i, &
                     center_j,center_i,field,origin,tensor)
                  if (angular_j(axis) > 0) then
                     shifted(axis) = shifted(axis) - 2
                     value_j = value_j - dble(angular_j(axis))*external_efield_primitive_moment(aj,ai, &
                        shifted,angular_i,center_j,center_i,field,origin,tensor)
                  endif
                  derivative_i(axis) = derivative_i(axis) + coef*value_i
                  derivative_j(axis) = derivative_j(axis) + coef*value_j
               enddo
            enddo
         enddo
         if (has_tensor) then
            call check_external_field_finite(derivative_i,'affine-field first AO-center derivative')
            call check_external_field_finite(derivative_j,'affine-field second AO-center derivative')
         endif
         quick_qm_struct%gradient(3*ia-2:3*ia) = quick_qm_struct%gradient(3*ia-2:3*ia) + density*derivative_i
         quick_qm_struct%gradient(3*ja-2:3*ja) = quick_qm_struct%gradient(3*ja-2:3*ja) + density*derivative_j
         if (has_tensor) then
            call check_external_field_finite(quick_qm_struct%gradient(3*ia-2:3*ia),'affine-field electronic gradient')
            call check_external_field_finite(quick_qm_struct%gradient(3*ja-2:3*ja),'affine-field electronic gradient')
         endif
      enddo
   enddo

end subroutine get_external_efield_gradient

subroutine check_external_field_finite(values,operation)
   ! A finite affine-field input can still overflow after multiplication by
   ! coordinates or moments. Abort the job communicator before a rank-local
   ! failure can leave other ranks waiting in an energy/gradient collective.
   use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
   use quick_files_module, only: ioutfile
#ifdef MPIV
   use mpi
   use quick_mpi_module, only: bMPI, quick_comm, quick_mpi_error
#endif
   implicit none
   double precision, intent(in) :: values(:)
   character(len=*), intent(in) :: operation
   integer :: status

   if (all(ieee_is_finite(values))) return
   status = 53
   write(ioutfile,'(A,1X,A)') 'Nonfinite applied-field result:',trim(operation)
#ifdef MPIV
   if (bMPI) call MPI_ABORT(quick_comm,status,quick_mpi_error)
#endif
   call quick_exit(ioutfile,status)
end subroutine check_external_field_finite

#if defined(CUDA) || defined(CUDA_MPIV)
subroutine add_external_field_gpu_gradient()
   use quick_basis_module, only: quick_basis, nbasis, ncontract, itype, aexp, dcoeff
   use quick_calculated_module, only: quick_qm_struct
   use quick_method_module, only: quick_method
   use quick_molspec_module, only: natom, xyz
#ifdef MPIV
   use quick_basis_module, only: mpi_nbasisn, mpi_nbasis
   use quick_mpi_module, only: bMPI, quick_comm_rank
#endif
   implicit none
   integer, allocatable :: owned(:)
   double precision, allocatable :: density(:,:), gradient(:,:)
   integer :: ierr
#ifdef MPIV
   integer :: irow
#endif

   allocate(owned(nbasis),density(nbasis,nbasis),gradient(3,natom),stat=ierr)
   call external_field_gpu_failure(ierr,'allocate electronic field-gradient workspace')
   owned = 1
#ifdef MPIV
   if (bMPI) then
      owned = 0
      do irow=1,mpi_nbasisn(quick_comm_rank)
         owned(mpi_nbasis(quick_comm_rank,irow)) = 1
      enddo
   endif
#endif
   density = quick_qm_struct%dense
   if (quick_method%unrst) density = density + quick_qm_struct%denseb
   ! Keep the cheap classical terms on the host, exactly once on master.
   ! The kernel adds only the electronic contribution before MPI reduction.
   gradient = reshape(quick_qm_struct%gradient,[3,natom])
   if (any(quick_method%external_efield_tensor /= 0.0d0)) then
      call gpu_external_field_tensor_gradient(nbasis,natom,size(aexp,1),ncontract,itype,quick_basis%ncenter, &
         aexp,dcoeff,xyz,quick_method%external_efield_vector,quick_method%external_efield_origin, &
         quick_method%external_efield_tensor,owned,quick_method%coreIntegralCutoff,density,gradient,ierr)
   else
      call gpu_external_field_gradient(nbasis,natom,size(aexp,1),ncontract,itype,quick_basis%ncenter, &
         aexp,dcoeff,xyz,quick_method%external_efield_vector,quick_method%external_efield_origin, &
         owned,quick_method%coreIntegralCutoff,density,gradient,ierr)
   endif
   call external_field_gpu_failure(ierr,'applied-field electronic nuclear gradient')
   quick_qm_struct%gradient = reshape(gradient,[3*natom])
end subroutine add_external_field_gpu_gradient

subroutine external_field_gpu_failure(status,operation)
   use quick_files_module, only: ioutfile
   use quick_mpi_module, only: master
#ifdef MPIV
   use mpi
   use quick_mpi_module, only: bMPI, quick_comm, quick_mpi_error
#endif
   implicit none
   integer, intent(inout) :: status
   character(len=*), intent(in) :: operation
   if (status == 0) return
   if (master) write(ioutfile,'(A,1X,A)') 'External-field CUDA failure:',trim(operation)
#ifdef MPIV
   if (bMPI) call MPI_ABORT(quick_comm,status,quick_mpi_error)
#endif
   call quick_exit(ioutfile,status)
end subroutine external_field_gpu_failure
#endif

subroutine get_external_field_gpu_counts(matrix,gradient)
   implicit none
   integer, intent(out) :: matrix, gradient
   matrix = 0
   gradient = 0
#if defined(CUDA) || defined(CUDA_MPIV)
   call gpu_external_field_launch_counts(matrix,gradient)
#endif
end subroutine get_external_field_gpu_counts

function external_efield_primitive_moment(a,b,angular_a,angular_b,center_a,center_b,field,origin,tensor) result(moment)
   implicit none

   double precision, intent(in) :: a, b, center_a(3), center_b(3), field(3), origin(3)
   double precision, optional, intent(in) :: tensor(3,3)
   integer, intent(in) :: angular_a(3), angular_b(3)
   double precision, external :: xmoment
   double precision :: moment, coefficient
   integer :: component, other, power(3)

   moment = 0.0d0
   do component=1,3
      if (field(component) == 0.0d0) cycle
      power = 0
      power(component) = 1
      moment = moment + field(component)*xmoment(a,b,angular_a(1),angular_a(2),angular_a(3), &
         angular_b(1),angular_b(2),angular_b(3),power(1),power(2),power(3), &
         center_a(1),center_a(2),center_a(3),center_b(1),center_b(2),center_b(3),origin(1),origin(2),origin(3))
   enddo

   ! Six independent Cartesian second moments. The symmetric tensor is
   ! defined in the laboratory frame, with no trace-free constraint.
   ! Return directly for uniform fields so their arithmetic is unchanged.
   if (.not.present(tensor)) return
   if (all(tensor == 0.0d0)) return
   do component=1,3
      do other=component,3
         coefficient = tensor(component,other)
         if (component == other) coefficient = 0.5d0*coefficient
         if (coefficient == 0.0d0) cycle
         power = 0
         power(component) = power(component) + 1
         power(other) = power(other) + 1
         moment = moment + coefficient*xmoment(a,b,angular_a(1),angular_a(2),angular_a(3), &
            angular_b(1),angular_b(2),angular_b(3),power(1),power(2),power(3), &
            center_a(1),center_a(2),center_a(3),center_b(1),center_b(2),center_b(3),origin(1),origin(2),origin(3))
      enddo
   enddo

end function external_efield_primitive_moment

double precision function ekinetic(a,b,i,j,k,ii,jj,kk,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table)
   use quick_overlap_module, only: overlap_core
   implicit none
   double precision :: kinetic
   double precision :: a,b
   integer :: i,j,k,ii,jj,kk,g,g_count
   double precision :: Ax,Ay,Az,Bx,By,Bz
   double precision :: Px,Py,Pz

   double precision :: xi,xj,xk,g_table(200)

   ! The purpose of this subroutine is to calculate the kinetic energy
   ! of an electron  distributed between gtfs with orbital exponents a
   ! and b on A and B with angular momentums defined by i,j,k (a's x, y
   ! and z exponents, respectively) and ii,jj,and kk on B.

   ! The first step is to see if this function is zero due to symmetry.
   ! If it is not, reset kinetic to 0.

   kinetic = (1+(-1)**(i+ii))*(1+(-1)**(j+jj))*(1+(-1)**(k+kk)) &
         +(Ax-Bx)**2 + (Ay-By)**2 + (Az-Bz)**2
   if (kinetic .ne. 0.d0) then
      kinetic=0.d0

      ! Kinetic energy is the integral of an orbital times the second derivative
      ! over space of the other orbital.  For GTFs, this means that it is just a
      ! sum of various overlap integrals with the powers adjusted.

      xi = dble(i)
      xj = dble(j)
      xk = dble(k)

      kinetic = kinetic &
            +        (-1.d0+     xi)*xi  *overlap_core(a,b,i-2,j,k,ii,jj,kk,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table) &
            - 2.d0*a*( 1.d0+2.d0*xi)     *overlap_core(a,b,i  ,j,k,ii,jj,kk,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table) &
            + 4.d0*(a**2.d0)             *overlap_core(a,b,i+2,j,k,ii,jj,kk,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table)
      kinetic = kinetic &
            +         (-1.d0+     xj)*xj *overlap_core(a,b,i,j-2,k,ii,jj,kk,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table) &
            - 2.d0*a* ( 1.d0+2.d0*xj)    *overlap_core(a,b,i,j  ,k,ii,jj,kk,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table) &
            + 4.d0*(a**2.d0)             *overlap_core(a,b,i,j+2,k,ii,jj,kk,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table)
      kinetic = kinetic &
            +         (-1.d0+     xk)*xk *overlap_core(a,b,i,j,k-2,ii,jj,kk,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table) &
            - 2.d0*a* ( 1.d0+2.d0*xk)    *overlap_core(a,b,i,j,k  ,ii,jj,kk,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table) &
            + 4.d0*(a**2.d0)             *overlap_core(a,b,i,j,k+2,ii,jj,kk,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table)
   endif
   ekinetic = kinetic*(-0.5d0)  *exp(-((a*b*((Ax-Bx)**2.d0 + (Ay-By)**2.d0+(Az-Bz)**2.d0))/(a+b)))

   return
end function ekinetic

end module quick_oei_module
