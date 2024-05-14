#include "util.fh"

#ifdef MPIV
     include "mpif.h"
#endif

!---------------------------------------------------------------------!
! Created by Etienne Palos on   xx/xx/2024                            !
!                                                                     ! 
! Copyright (C) 2024-2025 Götz lab                                    !
!                                                                     !
! This Source Code Form is subject to the terms of the Mozilla Public !
! License, v. 2.0. If a copy of the MPL was not distributed with this !
! file, You can obtain one at http://mozilla.org/MPL/2.0/.            !
!_____________________________________________________________________!

module quick_oeproperties_module
 implicit double precision(a-h,o-z)
 private 
 public :: compute_esp

 contains

 !----------------------------------------------------------------------------!
 ! This is the main subroutine that computes the Electrostatic Potential (ESP)!
 ! at a given point , V(r) = V_nuc(r) + V_elec(r), and prints it to file.prop !        
 !----------------------------------------------------------------------------!
 subroutine compute_esp(ierr)
   use allmod
   
#ifdef MPIV
   use mpi
#endif
 
   implicit none
   integer, intent(out) :: ierr
   logical :: debug = .true.
   integer :: IIsh, JJsh
   integer :: igridpoint
   
   double precision, allocatable :: esp_electronic(:)   
   double precision, allocatable :: esp_nuclear(:)   

   ierr = 0
   
   ! Allocates & initiates ESP_NUC and ESP_ELEC arrays
   allocate(esp_nuclear(quick_molspec%nextpoint))
   allocate(esp_electronic(quick_molspec%nextpoint))
   
   esp_nuclear(:) = 0.0d0
   esp_electronic(:) = 0.0d0

   RECORD_TIME(timer_begin%TESPGrid)

   call quick_open(iPropFile,propFileName,'U','F','R',.false.,ierr)

   ! Computes ESP_NUC 
   do igridpoint=1,quick_molspec%nextpoint
     call esp_nuc(ierr, igridpoint, esp_nuclear(igridpoint))
   end do
   
  ! Computes ESP_ELEC
   do IIsh = 1, jshell
      do JJsh = IIsh, jshell
        call esp_shell_pair(IIsh, JJsh, esp_electronic)
      end do
   end do

   RECORD_TIME(timer_end%TESPGrid)
   timer_cumer%TESPGrid=timer_cumer%TESPGrid+timer_end%TESPGrid-timer_begin%TESPGrid
   
   ! Calls print ESP
   call print_esp(esp_nuclear,esp_electronic, ierr)

   deallocate(esp_electronic)
   deallocate(esp_nuclear)

 close(iPropFile)

 end subroutine compute_esp

 !---------------------------------------------------------------------------------------------!
 ! This subroutine formats and prints the Electrostatic Potential (ESP) to the output file.prop!
 !---------------------------------------------------------------------------------------------!
 subroutine print_esp(esp_nuclear, esp_electronic, ierr)
   use quick_molspec_module
   use quick_method_module
   use quick_files_module, only: ioutfile, iPropFile
   use quick_mpi_module, only: master
   use quick_constants_module, only: BOHRS_TO_A

   implicit none
   integer, intent(out) :: ierr
   logical :: debug = .true.

   double precision, allocatable :: esp_nuclear(:)
   double precision, allocatable :: esp_electronic(:)

   integer :: igridpoint
   double precision :: Cx, Cy, Cz

  if (.not. allocated(esp_electronic)) then
   allocate(esp_electronic(igridpoint))
 endif

   if (.not. allocated(esp_nuclear)) then
   allocate(esp_nuclear(igridpoint))
 endif

 ! If ESP_GRID is true, print to table X, Y, Z, V(r)
 if ((quick_method%esp_grid) .or. (quick_method%esp_print_terms)) then
      write (ioutfile,'(/," QUICK PROPERTIES MODULE: ELECTROSTATIC POTENTIAL CALCULATION (ESP) [atomic units] ")')
      write (ioutfile,'(84("-"))')
      ! Do you want V_nuc and V_elec?  
    if ((quick_method%esp_print_terms) .and. (quick_method%esp_grid)) then
      write (ioutfile,'(9x,"X",13x,"Y",12x,"Z",16x, "ESP_NUC",12x, "ESP_ELEC",8x,"ESP_TOTAL")')
    else
      ! Default is V_total  
      write (ioutfile,'(9x,"X",13x,"Y",12x,"Z",16x, "ESP_TOTAL")')
      endif
      write (ioutfile,'(84("-"))')
      write (ioutfile,'(/," ESP is printed to PROPERTIES file ")')
      write (ioutfile,'(84("-"))')
      write (iPropFile,'(/," ELECTROSTATIC POTENTIAL CALCULATION (ESP) [atomic units] ")')
      write (iPropFile,'(100("-"))')
    ! Do you want V_nuc and V_elec?
      if ((quick_method%esp_print_terms) .and. (quick_method%esp_grid)) then
          write (iPropFile,'(9x,"X",13x,"Y",12x,"Z",16x, "ESP_NUC",12x, "ESP_ELEC",8x,"ESP_TOTAL")')
    ! Do you want  X, Y, and Z in Angstrom?
      else if ((quick_method%extgrid_angstrom) .and. (quick_method%esp_grid))  then
          write (iPropFile,'(6x,"X[A]",10x ,"Y[A]",9x,"Z[A]",13x, "ESP_TOTAL [a.u.] ")')
       else
      ! Default is X, Y, and V_total in a.u.
          write (iPropFile,'(9x,"X",13x,"Y",12x,"Z",16x,"ESP")')
      endif

      ! Collect ESP and print
      do igridpoint = 1, quick_molspec%nextpoint 
       if ((quick_method%extgrid_angstrom) .and. (quick_method%esp_grid))  then
        Cx = (quick_molspec%extxyz(1, igridpoint)*BOHRS_TO_A)
        Cy = (quick_molspec%extxyz(2, igridpoint)*BOHRS_TO_A)
        Cz = (quick_molspec%extxyz(3, igridpoint)*BOHRS_TO_A)
       else
        Cx = quick_molspec%extxyz(1, igridpoint)
        Cy = quick_molspec%extxyz(2, igridpoint)
        Cz = quick_molspec%extxyz(3, igridpoint)
       endif

        if (allocated(esp_electronic) .and. igridpoint <= size(esp_electronic)) then
            ! Additional option 1 : PRINT ESP_NUC, ESP_ELEC, and ESP_TOTAL
            if ((quick_method%esp_print_terms) .and. (quick_method%esp_grid)) then
                write(iPropFile, '(2x,3(F14.10, 1x), 3x,F14.10,3x,F14.10,3x,3F14.10)') Cx, Cy, Cz,  &
                esp_nuclear(igridpoint), esp_electronic(igridpoint), (esp_nuclear(igridpoint)+esp_electronic(igridpoint))
            else
                write(iPropFile, '(2x,3(F14.10, 1x), 3F14.10)') Cx, Cy, Cz,  &
                 (esp_nuclear(igridpoint)+esp_electronic(igridpoint))
            endif
        else
           write(iPropFile, '(3F14.10,3x,A)') Cx, Cy, Cz, 'N/A', 'N/A'
        endif

      end do
  endif
 end subroutine print_esp

 !-----------------------------------------------------------------------!
 ! This subroutine calculates V_nuc(r) = sum Z_k/|r-Rk|                  !
 !-----------------------------------------------------------------------!
 subroutine esp_nuc(ierr, igridpoint, esp_nuclear_term)
    use quick_molspec_module
    !, only: extxyz, chg
    !use allmod
   implicit none

   integer, intent(inout) :: ierr

   double precision :: distance
   double precision, external :: rootSquare
   integer i,j

   double precision, intent(inout) :: esp_nuclear_term
   integer ,intent(in) :: igridpoint
   
   logical :: debug = .true.

   esp_nuclear_term = 0.d0

     do i=1,natom
        distance = rootSquare(xyz(1:3,i), quick_molspec%extxyz(1:3,igridpoint), 3)
        esp_nuclear_term = esp_nuclear_term + quick_molspec%chg(i) / distance
     enddo
 end subroutine esp_nuc 

 !-----------------------------------------------------------------------------------------
 ! This subroutine computes the 1 particle contribution to the V_elec(r)
 ! This is \sum_{mu nu} P_{mu nu} * V_{mu nu}                                 
 ! See Eq. A14 of Oibara & Saika [J. Chem. Phys. 84, 3963 (1986)]                                     
 ! First, calculates 〈 phi_mu | phi_nu 〉 for all mu and nu                               
 ! Then, P_{mu nu} * 〈 phi_mu | 1/|r-C| | phi_nu 〉                                        
 !-----------------------------------------------------------------------------------------
 subroutine esp_1pdm(Ips,Jps,IIsh,JJsh,NIJ1,Ax,Ay,Az,Bx,By,Bz,Cx,Cy,Cz,Px,Py,Pz,esp)
   use allmod
   use quick_files_module, only: ioutfile

   implicit double precision(a-h,o-z)

   double precision attra,aux(0:20)
   integer a(3),b(3)
   double precision Ax,Ay,Az,Bx,By,Bz,Cx,Cy,Cz,Px,Py,Pz,g
   double precision AA(3),BB(3),CC(3),PP(3)

   double precision, intent(inout) :: esp

   common /xiaoattra/attra,aux,AA,BB,CC,PP,g

   logical :: debug = .true.

   AA(1)=Ax
   AA(2)=Ay
   AA(3)=Az
   BB(1)=Bx
   BB(2)=By
   BB(3)=Bz
   CC(1)=Cx
   CC(2)=Cy
   CC(3)=Cz
   PP(1)=Px
   PP(2)=Py
   PP(3)=Pz

   select case (NIJ1)

   case(0)
   case(10)
      call PSattra(0)
   case(1)
      call SPattra(0)
   case(11)
      call SPattra(0)
      call PSattra(0)
      call PSattra(1)
      call PPattra(0)
   case(20)
      call PSattra(0)
      call PSattra(1)
      call DSattra(0)
   case(2)
      call SPattra(0)
      call SPattra(1)
      call SDattra(0)
   case(21)
      call PSattra(0)
      call PSattra(1)
      call PSattra(2)
      call DSattra(0)
      call DSattra(1)
      call DPattra(0)
   case(12)
      call SPattra(0)
      call SPattra(1)
      call SPattra(2)
      call SDattra(0)
      call SDattra(1)
      call PDattra(0)
   case(22)
      do itempt=0,3
         call PSattra(itempt)
      enddo
      do itempt=0,1
         call PPattra(itempt)
      enddo
      do itempt=0,2
         call DSattra(itempt)
      enddo
      do itempt=0,1
         call DPattra(itempt)
      enddo
      call DDattra(0)
   case(30)
      do itemp=0,2
         call PSattra(itemp)
      enddo
      do itemp=0,1
         call DSattra(itemp)
      enddo
      call FSattra(0)
   case(3)
      do itemp=0,2
         call SPattra(itemp)
      enddo
      do itemp=0,1
         call SDattra(itemp)
      enddo
      call SFattra(0)
   case(31)
      do itemp=0,3
         call PSattra(itemp)
      enddo
      do itemp=0,2
         call PPattra(itemp)
      enddo
      do itemp=0,2
         call DSattra(itemp)
      enddo
      do itemp=0,1
         call DPattra(itemp)
      enddo
      do itemp=0,1
         call FSattra(itemp)
      enddo
      call FPattra(0)
   case(13)
      do itemp=0,3
         call SPattra(itemp)
         call PSattra(itemp)
      enddo
      do itemp=0,2
         call PPattra(itemp)
      enddo
      do itemp=0,2
         call SDattra(itemp)
      enddo
      do itemp=0,1
         call PDattra(itemp)
      enddo
      do itemp=0,1
         call SFattra(itemp)
      enddo
      call PFattra(0)
   case(32)
      do itemp=0,4
         call PSattra(itemp)
      enddo
      do itemp=0,3
         call PPattra(itemp)
      enddo
      do itemp=0,3
         call DSattra(itemp)
      enddo
      do itemp=0,2
         call DPattra(itemp)
      enddo
      do itemp=0,2
         call FSattra(itemp)
      enddo
      do itemp=0,1
         call FPattra(itemp)
      enddo
      call FDattra(0)
   case(23)
      do itemp=0,4
         call SPattra(itemp)
         call PSattra(itemp)
      enddo
      do itemp=0,3
         call PPattra(itemp)
      enddo
      do itemp=0,3
         call SDattra(itemp)
      enddo
      do itemp=0,2
         call PDattra(itemp)
      enddo
      do itemp=0,2
         call SFattra(itemp)
      enddo
      do itemp=0,1
         call PFattra(itemp)
      enddo
      call DFattra(0)
   case(33)
      do itemp=0,5
         call PSattra(itemp)
      enddo
      do itemp=0,4
         call PPattra(itemp)
      enddo
      do itemp=0,4
         call DSattra(itemp)
      enddo
      do itemp=0,3
         call DPattra(itemp)
      enddo
      do itemp=0,2
         call DDattra(itemp)
      enddo
      do itemp=0,3
         call FSattra(itemp)
      enddo
      do itemp=0,2
         call FPattra(itemp)
      enddo
      do itemp=0,1
         call FDattra(itemp)
      enddo
      call FFattra(0)
   end select

   do Iang=quick_basis%Qstart(IIsh),quick_basis%Qfinal(IIsh)
      X1temp=quick_basis%gccoeff(ips,quick_basis%ksumtype(IIsh)+Iang)
      do Jang=quick_basis%Qstart(JJsh),quick_basis%Qfinal(JJsh)
         NBI1=quick_basis%Qsbasis(IIsh,Iang)
         NBI2=quick_basis%Qfbasis(IIsh,Iang)
         NBJ1=quick_basis%Qsbasis(JJsh,Jang)
         NBJ2=quick_basis%Qfbasis(JJsh,Jang)

         III1=quick_basis%ksumtype(IIsh)+NBI1
         III2=quick_basis%ksumtype(IIsh)+NBI2
         JJJ1=quick_basis%ksumtype(JJsh)+NBJ1
         JJJ2=quick_basis%ksumtype(JJsh)+NBJ2

         Xconstant=X1temp*quick_basis%gccoeff(jps,quick_basis%ksumtype(JJsh)+Jang)
         do III=III1,III2

            itemp1=trans(quick_basis%KLMN(1,III),quick_basis%KLMN(2,III),quick_basis%KLMN(3,III))
            do JJJ=max(III,JJJ1),JJJ2
               itemp2=trans(quick_basis%KLMN(1,JJJ),quick_basis%KLMN(2,JJJ),quick_basis%KLMN(3,JJJ))

              dense_sym_factor = 1.0d0
              if (III /= JJJ) dense_sym_factor = 2.0d0

              if (quick_method%UNRST) then
               esp = esp + dense_sym_factor*(quick_qm_struct%dense(JJJ,III)+quick_qm_struct%denseb(JJJ,III))&
                       *Xconstant*quick_basis%cons(III)*quick_basis%cons(JJJ)*attraxiao(itemp1,itemp2,0)
              else
              esp = esp + dense_sym_factor*quick_qm_struct%denseSave(JJJ,III)*Xconstant &
                       *quick_basis%cons(III)*quick_basis%cons(JJJ)*attraxiao(itemp1,itemp2,0)
              endif 

            
            enddo
         enddo

      enddo
   enddo
   201 return

 End subroutine esp_1pdm

 !-----------------------------------------------------------------------------------------
 ! This subroutine computes loops over 
 ! This is \sum_{mu nu} P_{mu nu} * V_{mu nu}                                 
 ! See Eqn. A14 of Obara-Saika [J. Chem. Phys. 84, 3963 (1986)]                                     
 ! First, calculates 〈 phi_mu | phi_nu 〉 for all mu and nu                               
 ! Then, P_{mu nu} * 〈 phi_mu | 1/|r-C| | phi_nu 〉                                        
 !-----------------------------------------------------------------------------------------
 subroutine esp_shell_pair(IIsh, JJsh, esp_electronic)
   use allmod ! revisit, avoid allmod
   use quick_molspec_module
   use quick_overlap_module, only: gpt, opf, overlap_core
   use quick_files_module, only: ioutfile, iPropFile
   use quick_mpi_module, only: master
   implicit double precision(a-h,o-z)
   
   dimension aux(0:20)
   double precision AA(3),BB(3),CC(3),PP(3)
   common /xiaoattra/attra,aux,AA,BB,CC,PP,g 
   double precision RA(3),RB(3),RP(3),inv_g,g_table(200), valopf

   integer :: igridpoint
   logical :: debug = .true.
 
   double precision, dimension(:), intent(inout) :: esp_electronic
  
   ! Related to positions of "QM" atoms
   Ax=xyz(1,quick_basis%katom(IIsh))
   Ay=xyz(2,quick_basis%katom(IIsh))
   Az=xyz(3,quick_basis%katom(IIsh))
 
   Bx=xyz(1,quick_basis%katom(JJsh))
   By=xyz(2,quick_basis%katom(JJsh))
   Bz=xyz(3,quick_basis%katom(JJsh))
 
   NII2=quick_basis%Qfinal(IIsh)
   NJJ2=quick_basis%Qfinal(JJsh)
   Maxm=NII2+NJJ2
            
 
   ! Calculation of V_elec starts here
   do ips=1,quick_basis%kprim(IIsh)
      a=quick_basis%gcexpo(ips,quick_basis%ksumtype(IIsh))
      do jps=1,quick_basis%kprim(JJsh)
         b=quick_basis%gcexpo(jps,quick_basis%ksumtype(JJsh))
 
         valopf = opf(a, b, quick_basis%gccoeff(ips,quick_basis%ksumtype(IIsh)), &
         quick_basis%gccoeff(jps,quick_basis%ksumtype(JJsh)), Ax, Ay, Az, Bx, By, Bz)
 
         if(abs(valopf) .gt. quick_method%coreIntegralCutoff) then
 
           !Eqn 14 Obara-Saika 86
           call gpt(a,b,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,0,g_table)
           g = a+b
           !Eqn 15 Obara-Saika 86
           inv_g = 1.0d0 / dble(g)
 
           ! Calculate first two terms of Obara & Saika Eqn A20
           constanttemp=dexp(-((a*b*((Ax - Bx)**2.d0 + (Ay - By)**2.d0 + (Az - Bz)**2.d0))*inv_g))
           constant = overlap_core(a,b,0,0,0,0,0,0,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table) * 2.d0 * sqrt(g/Pi)*constanttemp
           
           ! Loops over external grid points/MM atoms
            do igridpoint=1,quick_molspec%nextpoint
               Cx=quick_molspec%extxyz(1,igridpoint)
               Cy=quick_molspec%extxyz(2,igridpoint)
               Cz=quick_molspec%extxyz(3,igridpoint)

                ! Calculate the last term of Obara--Saika Eqn A21
                PCsquare = (Px-Cx)**2 + (Py -Cy)**2 + (Pz -Cz)**2
 
                ! Compute Obara--Saika Eqn A21
                U = g* PCsquare
 
               ! Calculate the last term of Obara--Saika Eqn A20
               call FmT(Maxm,U,aux)
 
               ! Calculate all the auxilary integrals and store in attraxiao array
               do L = 0,maxm
                  ! sign (-1.0d0) is used to ensure the auxilary integrals are negative
                  aux(L) = -1.0d0*aux(L)*constant
                  attraxiao(1,1,L)=aux(L)
               enddo
 
               NIJ1=10*NII2+NJJ2
              
               call esp_1pdm(ips,jps,IIsh,JJsh,NIJ1,Ax,Ay,Az,Bx,By,Bz, &
                     Cx,Cy,Cz,Px,Py,Pz, esp_electronic(igridpoint))

            enddo

         endif
      enddo
   enddo
 end subroutine esp_shell_pair


! EFIELD

 ! EFIELD_GRID = -  (dx, dy, dz) ESP_GRID


 ! E_nuc = - grad. V_nuc 
 !       = - (d/drx, d/dry, d/drz) V_nuc(r)

 subroutine esp_nuc_gradient(ierr, igridpoint, esp_nuclear_gradient)
   use quick_molspec_module
   implicit none

   integer, intent(inout) :: ierr
   integer, intent(in) :: igridpoint
   double precision, external :: rootSquare
   double precision, intent(out) :: esp_nuclear_gradient(3)

   double precision :: distance
   integer :: i

   esp_nuclear_gradient = 0.0d0

   ! Compute the distance between the grid point and each atom
   do i = 1, natom
       distance = rootSquare(xyz(1:3, i), quick_molspec%extxyz(1:3, igridpoint), 3)

      ! Compute components of gradient (partial derivatives) using the chain rule
       esp_nuclear_gradient(1) = esp_nuclear_gradient(1) - quick_molspec%chg(i) * (xyz(1, i) - quick_molspec%extxyz(1, igridpoint)) / (distance**3)
       esp_nuclear_gradient(2) = esp_nuclear_gradient(2) - quick_molspec%chg(i) * (xyz(2, i) - quick_molspec%extxyz(2, igridpoint)) / (distance**3)
       esp_nuclear_gradient(3) = esp_nuclear_gradient(3) - quick_molspec%chg(i) * (xyz(3, i) - quick_molspec%extxyz(3, igridpoint)) / (distance**3)

      ! esp_nuclear_gradient(1) = esp_nuclear_gradient(1) + quick_molspec%chg(i) * (xyz(1, i) - quick_molspec%extxyz(1, igridpoint)) / (distance**3)
      ! esp_nuclear_gradient(2) = esp_nuclear_gradient(2) + quick_molspec%chg(i) * (xyz(2, i) - quick_molspec%extxyz(2, igridpoint)) / (distance**3)
      ! esp_nuclear_gradient(3) = esp_nuclear_gradient(3) + quick_molspec%chg(i) * (xyz(3, i) - quick_molspec%extxyz(3, igridpoint)) / (distance**3)
   end do

end subroutine esp_nuc_gradient

  subroutine logger(name, status)
    use quick_files_module
    implicit none
    character (len=*), intent(in) :: name
    character (len=*), intent(in) :: status
    write(ioutfile, '(3(a,x))') '>>> DEBUG', name, status
    call flush(ioutfile)
 end subroutine logger

end module quick_oeproperties_module
