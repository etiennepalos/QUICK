#include "util.fh"
!---------------------------------------------------------------------!
! MBX energy-gradient parity check through QUICK's thin MBX adapter.  !
!_____________________________________________________________________!

program quick_mbx_gradient_parity

   use quick_constants_module, only: BOHRS_TO_A
   use quick_mbx_module, only: quick_mbx_initialize_system, quick_mbx_finalize, &
      quick_mbx_set_real_coordinates, quick_mbx_get_real_atom_gradient

   implicit none

   integer, parameter :: natoms = 6
   integer, parameter :: nsites = 8
   integer, parameter :: nwaters = 2
   double precision, parameter :: fd_step_bohr = 1.0d-3

   integer :: ierr, argc
   double precision :: coords(3,natoms), grad(3,natoms), energy
   double precision :: max_abs, rms
   character(len=256) :: csv_file, json_file, json_mbx

   ierr = 0
   csv_file = 'mbx_gradient_parity.csv'
   json_file = 'mbx_gradient_parity_summary.json'
   json_mbx = 'mbx.json'
   argc = command_argument_count()
   if (argc >= 1) call get_command_argument(1, csv_file)
   if (argc >= 2) call get_command_argument(2, json_file)
   if (argc >= 3) call get_command_argument(3, json_mbx)

   call setup_water_dimer(coords)
   call initialize_mbx(coords,json_mbx,ierr)
   call stop_on_error(ierr,'initialize MBX')

   call quick_mbx_get_real_atom_gradient(energy,grad,ierr)
   call stop_on_error(ierr,'MBX analytic gradient')

   call finite_difference(csv_file,coords,grad,max_abs,rms,ierr)
   call stop_on_error(ierr,'finite difference')

   call write_summary(json_file,energy,max_abs,rms,ierr)
   call stop_on_error(ierr,'write summary')

   call quick_mbx_finalize()

   write(*,'("MBX_GRAD_PARITY_ENERGY_AU ",ES24.16)') energy
   write(*,'("MBX_GRAD_PARITY_MAX_ABS ",ES12.4)') max_abs
   write(*,'("MBX_GRAD_PARITY_RMS ",ES12.4)') rms

contains

   subroutine setup_water_dimer(coords)
      implicit none

      double precision, intent(out) :: coords(3,natoms)

      coords(:,1) = (/-1.58972425d0, 1.04337922d0, -0.08780840d0/)
      coords(:,2) = (/-0.63591971d0, 0.97898520d0,  0.00000000d0/)
      coords(:,3) = (/-1.90066280d0, 1.74501050d0, -0.66454990d0/)
      coords(:,4) = (/ 1.64924507d0, 1.08594656d0,  0.00000000d0/)
      coords(:,5) = (/ 2.60878026d0, 1.09587704d0, -0.02817115d0/)
      coords(:,6) = (/ 1.33830653d0, 1.78757784d0,  0.57674150d0/)
   end subroutine setup_water_dimer

   subroutine initialize_mbx(coords,json_mbx,ierr)
      implicit none

      double precision, intent(in) :: coords(3,natoms)
      character(len=*), intent(in) :: json_mbx
      integer, intent(inout) :: ierr

      integer :: nat_monomers(nwaters)
      character(len=1) :: atom_names(natoms)
      character(len=3) :: monomer_names(nwaters)

      nat_monomers(:) = 3
      atom_names = (/'O','H','H','O','H','H'/)
      monomer_names(:) = 'h2o'
      call quick_mbx_initialize_system(natoms,nsites,nwaters,nat_monomers,coords, &
         atom_names,monomer_names,json_mbx,ierr)
   end subroutine initialize_mbx

   subroutine evaluate(coords,energy,ierr)
      implicit none

      double precision, intent(in) :: coords(3,natoms)
      double precision, intent(out) :: energy
      integer, intent(inout) :: ierr

      double precision :: grad_tmp(3,natoms)

      call quick_mbx_set_real_coordinates(natoms,coords,ierr)
      if (ierr /= 0) return
      call quick_mbx_get_real_atom_gradient(energy,grad_tmp,ierr)
   end subroutine evaluate

   subroutine finite_difference(csv_file,coords,grad,max_abs,rms,ierr)
      implicit none

      character(len=*), intent(in) :: csv_file
      double precision, intent(in) :: coords(3,natoms), grad(3,natoms)
      double precision, intent(out) :: max_abs, rms
      integer, intent(inout) :: ierr

      integer :: unit, iatom, ixyz, ncomp
      double precision :: coords_p(3,natoms), coords_m(3,natoms)
      double precision :: eplus, eminus, fd, diff, accum, disp_ang

      max_abs = 0.0d0
      accum = 0.0d0
      ncomp = 0
      disp_ang = fd_step_bohr*BOHRS_TO_A

      open(newunit=unit,file=csv_file,status='replace',action='write',iostat=ierr)
      if (ierr /= 0) return
      write(unit,'(A)') 'atom,axis,e_minus_au,e_plus_au,fd_grad_au_per_bohr,analytic_grad_au_per_bohr,abs_diff'

      do iatom=1,natoms
         do ixyz=1,3
            coords_p(:,:) = coords(:,:)
            coords_m(:,:) = coords(:,:)
            coords_p(ixyz,iatom) = coords_p(ixyz,iatom) + disp_ang
            coords_m(ixyz,iatom) = coords_m(ixyz,iatom) - disp_ang
            call evaluate(coords_p,eplus,ierr)
            if (ierr /= 0) return
            call evaluate(coords_m,eminus,ierr)
            if (ierr /= 0) return
            fd = (eplus - eminus)/(2.0d0*fd_step_bohr)
            diff = fd - grad(ixyz,iatom)
            max_abs = max(max_abs,abs(diff))
            accum = accum + diff*diff
            ncomp = ncomp + 1
            write(unit,'(I0,",",A,",",ES24.16,",",ES24.16,",",ES24.16,",",ES24.16,",",ES12.4)') &
               iatom, axis_name(ixyz), eminus, eplus, fd, grad(ixyz,iatom), abs(diff)
         enddo
      enddo

      close(unit)
      rms = dsqrt(accum/dble(max(1,ncomp)))
      call quick_mbx_set_real_coordinates(natoms,coords,ierr)
   end subroutine finite_difference

   subroutine write_summary(json_file,energy,max_abs,rms,ierr)
      implicit none

      character(len=*), intent(in) :: json_file
      double precision, intent(in) :: energy, max_abs, rms
      integer, intent(inout) :: ierr

      integer :: unit

      open(newunit=unit,file=json_file,status='replace',action='write',iostat=ierr)
      if (ierr /= 0) return
      write(unit,'(A)') '{'
      write(unit,'(A,ES24.16,A)') '  "energy_au": ', energy, ','
      write(unit,'(A,ES12.4,A)') '  "fd_step_bohr": ', fd_step_bohr, ','
      write(unit,'(A,ES12.4,A)') '  "max_abs_au_per_bohr": ', max_abs, ','
      write(unit,'(A,ES12.4)') '  "rms_au_per_bohr": ', rms
      write(unit,'(A)') '}'
      close(unit)
   end subroutine write_summary

   character(len=1) function axis_name(ixyz)
      implicit none

      integer, intent(in) :: ixyz

      select case(ixyz)
      case(1)
         axis_name = 'x'
      case(2)
         axis_name = 'y'
      case default
         axis_name = 'z'
      end select
   end function axis_name

   subroutine stop_on_error(ierr_in,label)
      implicit none

      integer, intent(in) :: ierr_in
      character(len=*), intent(in) :: label

      if (ierr_in /= 0) then
         write(*,'("MBX_GRAD_PARITY_ERROR ",A,1X,I0)') trim(label), ierr_in
         stop 1
      endif
   end subroutine stop_on_error

end program quick_mbx_gradient_parity
