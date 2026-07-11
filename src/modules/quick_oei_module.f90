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
  public :: add_point_charge_operator, add_point_dipole_field_operator
  public :: add_point_multipole_qm_gradient
  public :: bCalc1e

  logical :: bCalc1e = .false.

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
     if (deltaO) then
       quick_qm_struct%E1e=quick_qm_struct%E1e+sum2mat(quick_qm_struct%dense,quick_qm_struct%oneElecO,nbasis)
       if (quick_method%unrst) then
         quick_qm_struct%E1e = quick_qm_struct%E1e+sum2mat(quick_qm_struct%denseb,quick_qm_struct%oneElecO,nbasis)
       endif
     else
       quick_qm_struct%E1e=quick_qm_struct%E1e+sum2mat(quick_qm_struct%dense,quick_qm_struct%o,nbasis)
       if (quick_method%unrst) then
         quick_qm_struct%E1e = quick_qm_struct%E1e+sum2mat(quick_qm_struct%denseb,quick_qm_struct%o,nbasis)
       endif
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

#ifdef MPIV
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
         ! h^F = F . (r-r0), added to Hcore after the usual T+V terms.
         if (quick_method%external_efield) then
            do Ibas=1,nbasis
               call externalEFieldO(Ibas)
            enddo
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

      do i=1,mpi_nbasisn(mpirank)
         Ibas=mpi_nbasis(mpirank,i)
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
        do i=1,mpi_jshelln(mpirank)
           IIsh=mpi_jshell(mpirank,i)
           do JJsh=IIsh,jshell
              call attrashell(IIsh,JJsh)
           enddo
        enddo
      endif
#else
      do i=1,mpi_jshelln(mpirank)
         IIsh=mpi_jshell(mpirank,i)
         do JJsh=IIsh,jshell
            call attrashell(IIsh,JJsh)
         enddo
      enddo
#endif
      RECORD_TIME(timer_end%T1eV)

      ! Each MPI rank contributes only its owned basis-function rows to the
      ! finite-field one-electron operator; the normal reductions then apply.
      if (quick_method%external_efield) then
         do i=1,mpi_nbasisn(mpirank)
            Ibas=mpi_nbasis(mpirank,i)
            call externalEFieldO(Ibas)
         enddo
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

   !------------------------------------------------
   ! This subroutine adds the one-electron operator
   ! for a uniform external electric field,
   !       h^F = F_x (x-x0) + F_y (y-y0) + F_z (z-z0)
   ! to the lower triangle of the one-electron matrix.
   !------------------------------------------------
   use allmod
   use quick_overlap_module, only: opf
   implicit double precision(a-h,o-z)
   integer Ibas
   double precision, external :: xmoment
   double precision :: coef, valopf, moment_x, moment_y, moment_z
   double precision :: field_x, field_y, field_z
   double precision :: origin_x, origin_y, origin_z

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

subroutine add_point_charge_operator(nsites,site_xyz_bohr,site_charge_e)

   !------------------------------------------------
   ! Add the one-electron operator from classical
   ! point charges to the lower triangle of O.
   !------------------------------------------------
   use allmod
#ifdef MPIV
   use mpi
#endif

   implicit double precision(a-h,o-z)

   integer, intent(in) :: nsites
   double precision, intent(in) :: site_xyz_bohr(3,nsites)
   double precision, intent(in) :: site_charge_e(nsites)

   integer :: isite, ish, IIsh, JJsh

   if (nsites <= 0) return

#ifdef MPIV
   if (bMPI) then
      do ish=1,mpi_jshelln(mpirank)
         IIsh=mpi_jshell(mpirank,ish)
         do JJsh=IIsh,jshell
            do isite=1,nsites
               if (abs(site_charge_e(isite)) > 0.0d0) then
                  call point_charge_operator_shell_pair(IIsh,JJsh,site_xyz_bohr(:,isite),site_charge_e(isite))
               endif
            enddo
         enddo
      enddo
   else
#endif
      do IIsh=1,jshell
         do JJsh=IIsh,jshell
            do isite=1,nsites
               if (abs(site_charge_e(isite)) > 0.0d0) then
                  call point_charge_operator_shell_pair(IIsh,JJsh,site_xyz_bohr(:,isite),site_charge_e(isite))
               endif
            enddo
         enddo
      enddo
#ifdef MPIV
   endif
#endif

end subroutine add_point_charge_operator

subroutine add_point_dipole_field_operator(nsites,site_xyz_bohr,site_mu_au)

   !------------------------------------------------
   ! Add h_mu = -mu . E_mu_nu(C) for classical point
   ! dipoles to the lower triangle of the AO operator.
   !------------------------------------------------
   use allmod
#ifdef MPIV
   use mpi
#endif

   implicit none

   integer, intent(in) :: nsites
   double precision, intent(in) :: site_xyz_bohr(3,nsites)
   double precision, intent(in) :: site_mu_au(3,nsites)

   integer :: isite, ish, IIsh, JJsh

   if (nsites <= 0) return

#ifdef MPIV
   if (bMPI) then
      do ish=1,mpi_jshelln(mpirank)
         IIsh=mpi_jshell(mpirank,ish)
         do JJsh=IIsh,jshell
            do isite=1,nsites
               if (any(site_mu_au(:,isite) /= 0.0d0)) &
                  call point_dipole_operator_shell_pair(IIsh,JJsh,site_xyz_bohr(:,isite),site_mu_au(:,isite))
            enddo
         enddo
      enddo
   else
#endif
      do IIsh=1,jshell
         do JJsh=IIsh,jshell
            do isite=1,nsites
               if (any(site_mu_au(:,isite) /= 0.0d0)) &
                  call point_dipole_operator_shell_pair(IIsh,JJsh,site_xyz_bohr(:,isite),site_mu_au(:,isite))
            enddo
         enddo
      enddo
#ifdef MPIV
   endif
#endif

end subroutine add_point_dipole_field_operator

subroutine point_dipole_operator_shell_pair(IIsh,JJsh,dipole_xyz_bohr,mu)
   use quick_overlap_module, only: opf
   use quick_molspec_module, only: xyz
   use quick_basis_module, only: quick_basis
   use quick_method_module, only: quick_method
   use quick_os_attraction_module, only: os_attraction_derivatives
   use quick_calculated_module, only: quick_qm_struct
   use quick_constants_module, only: Pi

   implicit none

   integer, intent(in) :: IIsh, JJsh
   double precision, intent(in) :: dipole_xyz_bohr(3), mu(3)

   integer :: ips, jps, L, Maxm, NII2, NJJ2
   integer :: Iang, Jang, III, III1, III2, JJJ, JJJ1, JJJ2
   integer :: NBI1, NBI2, NBJ1, NBJ2, lx1, ly1, lz1, lx2, ly2, lz2
   double precision :: a, b, Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, g, U
   double precision :: constant, PCsquare, Px, Py, Pz, inv_g, rABsquare, valopf
   double precision :: contract_i, contract_ij, operator_value
   double precision :: aux(0:20), deriv(10)

   Ax=xyz(1,quick_basis%katom(IIsh))
   Ay=xyz(2,quick_basis%katom(IIsh))
   Az=xyz(3,quick_basis%katom(IIsh))
   Bx=xyz(1,quick_basis%katom(JJsh))
   By=xyz(2,quick_basis%katom(JJsh))
   Bz=xyz(3,quick_basis%katom(JJsh))
   Cx=dipole_xyz_bohr(1)
   Cy=dipole_xyz_bohr(2)
   Cz=dipole_xyz_bohr(3)

   NII2=quick_basis%Qfinal(IIsh)
   NJJ2=quick_basis%Qfinal(JJsh)
   Maxm=NII2+NJJ2+2
   rABsquare=(Ax-Bx)**2 + (Ay-By)**2 + (Az-Bz)**2

   do ips=1,quick_basis%kprim(IIsh)
      a=quick_basis%gcexpo(ips,quick_basis%ksumtype(IIsh))
      do jps=1,quick_basis%kprim(JJsh)
         b=quick_basis%gcexpo(jps,quick_basis%ksumtype(JJsh))
         valopf=opf(a,b,quick_basis%gccoeff(ips,quick_basis%ksumtype(IIsh)), &
            quick_basis%gccoeff(jps,quick_basis%ksumtype(JJsh)),Ax,Ay,Az,Bx,By,Bz)
         if (abs(valopf) <= quick_method%coreIntegralCutoff) cycle

         g=a+b
         inv_g=1.0d0/g
         Px=(a*Ax+b*Bx)*inv_g
         Py=(a*Ay+b*By)*inv_g
         Pz=(a*Az+b*Bz)*inv_g
         constant=dexp(-a*b*rABsquare*inv_g)*2.0d0*Pi*inv_g
         PCsquare=(Px-Cx)**2+(Py-Cy)**2+(Pz-Cz)**2
         U=g*PCsquare
         call FmT(Maxm,U,aux)
         aux(0:Maxm)=constant*aux(0:Maxm)

         do Iang=quick_basis%Qstart(IIsh),quick_basis%Qfinal(IIsh)
            contract_i=quick_basis%gccoeff(ips,quick_basis%ksumtype(IIsh)+Iang)
            NBI1=quick_basis%Qsbasis(IIsh,Iang)
            NBI2=quick_basis%Qfbasis(IIsh,Iang)
            III1=quick_basis%ksumtype(IIsh)+NBI1
            III2=quick_basis%ksumtype(IIsh)+NBI2
            do Jang=quick_basis%Qstart(JJsh),quick_basis%Qfinal(JJsh)
               contract_ij=contract_i*quick_basis%gccoeff(jps,quick_basis%ksumtype(JJsh)+Jang)
               NBJ1=quick_basis%Qsbasis(JJsh,Jang)
               NBJ2=quick_basis%Qfbasis(JJsh,Jang)
               JJJ1=quick_basis%ksumtype(JJsh)+NBJ1
               JJJ2=quick_basis%ksumtype(JJsh)+NBJ2
               do III=III1,III2
                  lx1=quick_basis%KLMN(1,III)
                  ly1=quick_basis%KLMN(2,III)
                  lz1=quick_basis%KLMN(3,III)
                  do JJJ=max(III,JJJ1),JJJ2
                     lx2=quick_basis%KLMN(1,JJJ)
                     ly2=quick_basis%KLMN(2,JJJ)
                     lz2=quick_basis%KLMN(3,JJJ)
                     deriv=os_attraction_derivatives(lx1,ly1,lz1,lx2,ly2,lz2,0,aux, &
                        Ax,Ay,Az,Bx,By,Bz,Cx,Cy,Cz,Px,Py,Pz,g)
                     operator_value=-dot_product(mu,deriv(2:4))
                     quick_qm_struct%o(JJJ,III)=quick_qm_struct%o(JJJ,III)+contract_ij* &
                        quick_basis%cons(III)*quick_basis%cons(JJJ)*operator_value
                  enddo
               enddo
            enddo
         enddo
      enddo
   enddo
end subroutine point_dipole_operator_shell_pair

subroutine add_point_multipole_qm_gradient(nsites,site_xyz_bohr,site_charge_e,site_mu_au)
   use allmod
#ifdef MPIV
   use mpi
#endif
   implicit none

   integer, intent(in) :: nsites
   double precision, intent(in) :: site_xyz_bohr(3,nsites), site_charge_e(nsites)
   double precision, intent(in) :: site_mu_au(3,nsites)

   integer :: isite, ish, IIsh, JJsh

   if (nsites <= 0) return
#ifdef MPIV
   if (bMPI) then
      do ish=1,mpi_jshelln(mpirank)
         IIsh=mpi_jshell(mpirank,ish)
         do JJsh=IIsh,jshell
            do isite=1,nsites
               call point_multipole_gradient_shell_pair(IIsh,JJsh,site_xyz_bohr(:,isite), &
                  site_charge_e(isite),site_mu_au(:,isite))
            enddo
         enddo
      enddo
      if (master) call add_point_multipole_nuclear_gradient(nsites,site_xyz_bohr,site_charge_e,site_mu_au)
   else
#endif
      do IIsh=1,jshell
         do JJsh=IIsh,jshell
            do isite=1,nsites
               call point_multipole_gradient_shell_pair(IIsh,JJsh,site_xyz_bohr(:,isite), &
                  site_charge_e(isite),site_mu_au(:,isite))
            enddo
         enddo
      enddo
      call add_point_multipole_nuclear_gradient(nsites,site_xyz_bohr,site_charge_e,site_mu_au)
#ifdef MPIV
   endif
#endif
end subroutine add_point_multipole_qm_gradient

subroutine point_multipole_gradient_shell_pair(IIsh,JJsh,site_xyz_bohr,charge,mu)
   use quick_overlap_module, only: opf
   use quick_molspec_module, only: xyz
   use quick_basis_module, only: quick_basis
   use quick_method_module, only: quick_method
   use quick_os_attraction_module, only: os_attraction_derivatives
   use quick_calculated_module, only: quick_qm_struct
   use quick_constants_module, only: Pi
   implicit none

   integer, intent(in) :: IIsh, JJsh
   double precision, intent(in) :: site_xyz_bohr(3), charge, mu(3)

   integer :: ips, jps, L, Maxm, NII2, NJJ2
   integer :: Iang, Jang, III, III1, III2, JJJ, JJJ1, JJJ2
   integer :: NBI1, NBI2, NBJ1, NBJ2, l1(3), l2(3), lwork(3)
   integer :: iatom_a, iatom_b, idir
   double precision :: a, b, Axyz(3), Bxyz(3), Cxyz(3), Pxyz(3), g, U
   double precision :: constant, PCsquare, inv_g, rABsquare, valopf
   double precision :: contract_i, contract_ij, denseji, weight
   double precision :: aux(0:20), deriv(10), dA(3), dB(3)

   if (charge == 0.0d0 .and. .not. any(mu /= 0.0d0)) return

   iatom_a=quick_basis%katom(IIsh)
   iatom_b=quick_basis%katom(JJsh)
   Axyz(:)=xyz(:,iatom_a)
   Bxyz(:)=xyz(:,iatom_b)
   Cxyz(:)=site_xyz_bohr(:)
   NII2=quick_basis%Qfinal(IIsh)
   NJJ2=quick_basis%Qfinal(JJsh)
   Maxm=NII2+NJJ2+3
   rABsquare=sum((Axyz-Bxyz)**2)

   do ips=1,quick_basis%kprim(IIsh)
      a=quick_basis%gcexpo(ips,quick_basis%ksumtype(IIsh))
      do jps=1,quick_basis%kprim(JJsh)
         b=quick_basis%gcexpo(jps,quick_basis%ksumtype(JJsh))
         valopf=opf(a,b,quick_basis%gccoeff(ips,quick_basis%ksumtype(IIsh)), &
            quick_basis%gccoeff(jps,quick_basis%ksumtype(JJsh)),Axyz(1),Axyz(2),Axyz(3), &
            Bxyz(1),Bxyz(2),Bxyz(3))
         if (abs(valopf) <= quick_method%coreIntegralCutoff) cycle

         g=a+b
         inv_g=1.0d0/g
         Pxyz(:)=(a*Axyz(:)+b*Bxyz(:))*inv_g
         constant=dexp(-a*b*rABsquare*inv_g)*2.0d0*Pi*inv_g
         PCsquare=sum((Pxyz-Cxyz)**2)
         U=g*PCsquare
         call FmT(Maxm,U,aux)
         aux(0:Maxm)=constant*aux(0:Maxm)

         do Iang=quick_basis%Qstart(IIsh),quick_basis%Qfinal(IIsh)
            contract_i=quick_basis%gccoeff(ips,quick_basis%ksumtype(IIsh)+Iang)
            NBI1=quick_basis%Qsbasis(IIsh,Iang)
            NBI2=quick_basis%Qfbasis(IIsh,Iang)
            III1=quick_basis%ksumtype(IIsh)+NBI1
            III2=quick_basis%ksumtype(IIsh)+NBI2
            do Jang=quick_basis%Qstart(JJsh),quick_basis%Qfinal(JJsh)
               contract_ij=contract_i*quick_basis%gccoeff(jps,quick_basis%ksumtype(JJsh)+Jang)
               NBJ1=quick_basis%Qsbasis(JJsh,Jang)
               NBJ2=quick_basis%Qfbasis(JJsh,Jang)
               JJJ1=quick_basis%ksumtype(JJsh)+NBJ1
               JJJ2=quick_basis%ksumtype(JJsh)+NBJ2
               do III=III1,III2
                  l1(:)=quick_basis%KLMN(:,III)
                  do JJJ=max(III,JJJ1),JJJ2
                     l2(:)=quick_basis%KLMN(:,JJJ)
                     if (quick_method%UNRST) then
                        denseji=quick_qm_struct%dense(JJJ,III)+quick_qm_struct%denseb(JJJ,III)
                     else
                        denseji=quick_qm_struct%dense(JJJ,III)
                     endif
                     if (III /= JJJ) denseji=2.0d0*denseji
                     weight=contract_ij*quick_basis%cons(III)*quick_basis%cons(JJJ)*denseji

                     do idir=1,3
                        lwork(:)=l1(:)
                        lwork(idir)=lwork(idir)+1
                        deriv=os_attraction_derivatives(lwork(1),lwork(2),lwork(3), &
                           l2(1),l2(2),l2(3),0,aux,Axyz(1),Axyz(2),Axyz(3), &
                           Bxyz(1),Bxyz(2),Bxyz(3),Cxyz(1),Cxyz(2),Cxyz(3), &
                           Pxyz(1),Pxyz(2),Pxyz(3),g)
                        dA(idir)=2.0d0*a*multipole_integral(deriv,charge,mu)
                        if (l1(idir) > 0) then
                           lwork(:)=l1(:)
                           lwork(idir)=lwork(idir)-1
                           deriv=os_attraction_derivatives(lwork(1),lwork(2),lwork(3), &
                              l2(1),l2(2),l2(3),0,aux,Axyz(1),Axyz(2),Axyz(3), &
                              Bxyz(1),Bxyz(2),Bxyz(3),Cxyz(1),Cxyz(2),Cxyz(3), &
                              Pxyz(1),Pxyz(2),Pxyz(3),g)
                           dA(idir)=dA(idir)-dble(l1(idir))*multipole_integral(deriv,charge,mu)
                        endif

                        lwork(:)=l2(:)
                        lwork(idir)=lwork(idir)+1
                        deriv=os_attraction_derivatives(l1(1),l1(2),l1(3), &
                           lwork(1),lwork(2),lwork(3),0,aux,Axyz(1),Axyz(2),Axyz(3), &
                           Bxyz(1),Bxyz(2),Bxyz(3),Cxyz(1),Cxyz(2),Cxyz(3), &
                           Pxyz(1),Pxyz(2),Pxyz(3),g)
                        dB(idir)=2.0d0*b*multipole_integral(deriv,charge,mu)
                        if (l2(idir) > 0) then
                           lwork(:)=l2(:)
                           lwork(idir)=lwork(idir)-1
                           deriv=os_attraction_derivatives(l1(1),l1(2),l1(3), &
                              lwork(1),lwork(2),lwork(3),0,aux,Axyz(1),Axyz(2),Axyz(3), &
                              Bxyz(1),Bxyz(2),Bxyz(3),Cxyz(1),Cxyz(2),Cxyz(3), &
                              Pxyz(1),Pxyz(2),Pxyz(3),g)
                           dB(idir)=dB(idir)-dble(l2(idir))*multipole_integral(deriv,charge,mu)
                        endif
                     enddo

                     quick_qm_struct%gradient(3*(iatom_a-1)+1:3*iatom_a)= &
                        quick_qm_struct%gradient(3*(iatom_a-1)+1:3*iatom_a)+weight*dA(:)
                     quick_qm_struct%gradient(3*(iatom_b-1)+1:3*iatom_b)= &
                        quick_qm_struct%gradient(3*(iatom_b-1)+1:3*iatom_b)+weight*dB(:)
                  enddo
               enddo
            enddo
         enddo
      enddo
   enddo
end subroutine point_multipole_gradient_shell_pair

double precision function multipole_integral(deriv,charge,mu)
   implicit none
   double precision, intent(in) :: deriv(10), charge, mu(3)

   multipole_integral=-charge*deriv(1)-dot_product(mu,deriv(2:4))
end function multipole_integral

subroutine add_point_multipole_nuclear_gradient(nsites,site_xyz_bohr,site_charge_e,site_mu_au)
   use quick_molspec_module, only: natom, xyz, quick_molspec
   use quick_calculated_module, only: quick_qm_struct
   implicit none

   integer, intent(in) :: nsites
   double precision, intent(in) :: site_xyz_bohr(3,nsites), site_charge_e(nsites)
   double precision, intent(in) :: site_mu_au(3,nsites)

   integer :: iatom, isite, offset
   double precision :: r(3), r2, rinv, rinv3, rinv5, mudotr, z

   do iatom=1,natom
      z=quick_molspec%chg(iatom)
      offset=3*(iatom-1)
      do isite=1,nsites
         r(:)=xyz(:,iatom)-site_xyz_bohr(:,isite)
         r2=dot_product(r,r)
         if (r2 <= 1.0d-24) cycle
         rinv=1.0d0/dsqrt(r2)
         rinv3=rinv*rinv*rinv
         rinv5=rinv3*rinv*rinv
         mudotr=dot_product(site_mu_au(:,isite),r)
         quick_qm_struct%gradient(offset+1:offset+3)=quick_qm_struct%gradient(offset+1:offset+3) &
            -z*site_charge_e(isite)*r(:)*rinv3 &
            +z*(site_mu_au(:,isite)*rinv3-3.0d0*mudotr*r(:)*rinv5)
      enddo
   enddo
end subroutine add_point_multipole_nuclear_gradient

subroutine point_charge_operator_shell_pair(IIsh,JJsh,charge_xyz_bohr,charge)

   !------------------------------------------------
   ! Add the one-electron operator from one classical
   ! point charge to the lower triangle of O.
   !------------------------------------------------
   use quick_overlap_module, only: opf
   use quick_molspec_module, only: xyz
   use quick_basis_module, only: quick_basis, attraxiao
   use quick_method_module, only: quick_method
   use quick_constants_module, only: Pi

   implicit none

   integer, intent(in) :: IIsh, JJsh
   double precision, intent(in) :: charge_xyz_bohr(3), charge

   integer :: ips, jps, L, Maxm, NII2, NIJ1, NJJ2
   double precision :: a, b, Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, g, U
   double precision :: constant, PCsquare, Px, Py, Pz
   double precision :: inv_g, rABsquare, valopf, Z
   double precision, dimension(0:20) :: aux

   double precision :: attra, AA(3), BB(3), CC(3), PP(3)
   common /xiaoattra/attra,aux,AA,BB,CC,PP,g

   if (charge == 0.0d0) return

   Ax=xyz(1,quick_basis%katom(IIsh))
   Ay=xyz(2,quick_basis%katom(IIsh))
   Az=xyz(3,quick_basis%katom(IIsh))

   Bx=xyz(1,quick_basis%katom(JJsh))
   By=xyz(2,quick_basis%katom(JJsh))
   Bz=xyz(3,quick_basis%katom(JJsh))

   Cx=charge_xyz_bohr(1)
   Cy=charge_xyz_bohr(2)
   Cz=charge_xyz_bohr(3)
   Z=-charge

   NII2=quick_basis%Qfinal(IIsh)
   NJJ2=quick_basis%Qfinal(JJsh)
   Maxm=NII2+NJJ2+1+1
   NIJ1=10*NII2+NJJ2
   rABsquare=(Ax-Bx)**2.d0 + (Ay-By)**2.d0 + (Az-Bz)**2.d0

   do ips=1,quick_basis%kprim(IIsh)
      a=quick_basis%gcexpo(ips,quick_basis%ksumtype(IIsh))
      do jps=1,quick_basis%kprim(JJsh)
         b=quick_basis%gcexpo(jps,quick_basis%ksumtype(JJsh))
         valopf = opf(a, b, quick_basis%gccoeff(ips,quick_basis%ksumtype(IIsh)),&
            quick_basis%gccoeff(jps,quick_basis%ksumtype(JJsh)), Ax, Ay, Az, Bx, By, Bz)

         if(abs(valopf) .gt. quick_method%coreIntegralCutoff) then
            g = a+b
            inv_g = 1.0d0/g
            Px = (a*Ax + b*Bx)*inv_g
            Py = (a*Ay + b*By)*inv_g
            Pz = (a*Az + b*Bz)*inv_g

            constant = dexp(-a*b*rABsquare*inv_g) * 2.d0 * Pi * inv_g
            PCsquare = (Px-Cx)**2 + (Py-Cy)**2 + (Pz-Cz)**2
            U = g*PCsquare

            call FmT(Maxm,U,aux)
            do L = 0,Maxm
               aux(L) = aux(L)*constant*Z
               attraxiao(1,1,L)=aux(L)
            enddo

            call nuclearattra(ips,jps,IIsh,JJsh,NIJ1,Ax,Ay,Az,Bx,By,Bz, &
               Cx,Cy,Cz,Px,Py,Pz)
         endif
      enddo
   enddo

end subroutine point_charge_operator_shell_pair

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
