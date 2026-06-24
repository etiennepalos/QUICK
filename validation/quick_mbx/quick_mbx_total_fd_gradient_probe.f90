#include "util.fh"
!---------------------------------------------------------------------!
! Total QUICK-MBX finite-difference gradient probe for a tiny system. !
!                                                                     !
! This is a reference path for validating signs, units, and stability  !
! before analytic QUICK-MBX gradients are claimed.                     !
!_____________________________________________________________________!

program quick_mbx_total_fd_gradient_probe

   use quick_api_module, only: setQuickJob, getQuickMBXEnergyGradientsFD, deleteQuickJob, clearQuickMBXSystem

   implicit none

   integer, parameter :: natom_qm = 3
   integer, parameter :: nwaters = 1
   integer, parameter :: natom_mbx = 3*nwaters
   double precision, parameter :: fd_step1_bohr = 1.0d-3
   double precision, parameter :: fd_step2_bohr = 2.0d-3

   integer :: ierr, argc
   integer :: atomic_numbers(natom_qm)
   double precision :: qm_xyz(3,natom_qm), mbx_xyz(3,natom_mbx)
   double precision :: energy1, energy2
   double precision :: qm_grad1(3,natom_qm), qm_grad2(3,natom_qm)
   double precision :: mbx_grad1(3,natom_mbx), mbx_grad2(3,natom_mbx)
   double precision :: qm_step_max, mbx_step_max, total_step_max
   double precision :: net_grad_norm, max_grad_abs
   character(len=256) :: csv_file, json_file, json_mbx
   character(len=80) :: job_name
   character(len=256) :: keywd

   ierr = 0
   csv_file = 'quick_mbx_total_fd_gradient_probe.csv'
   json_file = 'quick_mbx_total_fd_gradient_probe_summary.json'
   json_mbx = 'mbx.json'
   argc = command_argument_count()
   if (argc >= 1) call get_command_argument(1, csv_file)
   if (argc >= 2) call get_command_argument(2, json_file)
   if (argc >= 3) call get_command_argument(3, json_mbx)

   atomic_numbers = (/8,1,1/)
   call setup_water_pair(qm_xyz,mbx_xyz)

   job_name = 'quick_mbx_total_fd_gradient_probe'
   keywd = 'HF BASIS=STO-3G CUTOFF=1.0D-10 DENSERMS=1.0D-6 MBX_QMMM'
   call setQuickJob(job_name,keywd,natom_qm,atomic_numbers,.true.,ierr)
   call stop_on_error(ierr,'setQuickJob')

   call getQuickMBXEnergyGradientsFD(qm_xyz,nwaters,mbx_xyz,json_mbx,fd_step1_bohr, &
      energy1,qm_grad1,mbx_grad1,ierr)
   call stop_on_error(ierr,'fd gradient step 1')

   call getQuickMBXEnergyGradientsFD(qm_xyz,nwaters,mbx_xyz,json_mbx,fd_step2_bohr, &
      energy2,qm_grad2,mbx_grad2,ierr)
   call stop_on_error(ierr,'fd gradient step 2')

   call compare_gradients(qm_grad1,qm_grad2,mbx_grad1,mbx_grad2, &
      qm_step_max,mbx_step_max,total_step_max,net_grad_norm,max_grad_abs)
   call write_csv(csv_file,qm_grad1,qm_grad2,mbx_grad1,mbx_grad2,ierr)
   call stop_on_error(ierr,'write csv')
   call write_summary(json_file,energy1,energy2,qm_step_max,mbx_step_max, &
      total_step_max,net_grad_norm,max_grad_abs,ierr)
   call stop_on_error(ierr,'write summary')

   call clearQuickMBXSystem(ierr)
   call deleteQuickJob(ierr)
   call stop_on_error(ierr,'deleteQuickJob')

   write(*,'("QUICK_MBX_TOTAL_FD_ENERGY_AU ",ES24.16)') energy1
   write(*,'("QUICK_MBX_TOTAL_FD_STEP_MAX_ABS ",ES12.4)') total_step_max
   write(*,'("QUICK_MBX_TOTAL_FD_NET_GRAD_NORM ",ES12.4)') net_grad_norm
   write(*,'("QUICK_MBX_TOTAL_FD_MAX_GRAD_ABS ",ES12.4)') max_grad_abs

contains

   subroutine setup_water_pair(qm_xyz,mbx_xyz)
      implicit none

      double precision, intent(out) :: qm_xyz(3,natom_qm), mbx_xyz(3,natom_mbx)

      qm_xyz(:,1) = (/-1.58972425d0, 1.04337922d0, -0.08780840d0/)
      qm_xyz(:,2) = (/-0.63591971d0, 0.97898520d0,  0.00000000d0/)
      qm_xyz(:,3) = (/-1.90066280d0, 1.74501050d0, -0.66454990d0/)
      mbx_xyz(:,1) = (/ 1.64924507d0, 1.08594656d0,  0.00000000d0/)
      mbx_xyz(:,2) = (/ 2.60878026d0, 1.09587704d0, -0.02817115d0/)
      mbx_xyz(:,3) = (/ 1.33830653d0, 1.78757784d0,  0.57674150d0/)
   end subroutine setup_water_pair

   subroutine compare_gradients(qm_grad1,qm_grad2,mbx_grad1,mbx_grad2, &
         qm_step_max,mbx_step_max,total_step_max,net_grad_norm,max_grad_abs)
      implicit none

      double precision, intent(in) :: qm_grad1(3,natom_qm), qm_grad2(3,natom_qm)
      double precision, intent(in) :: mbx_grad1(3,natom_mbx), mbx_grad2(3,natom_mbx)
      double precision, intent(out) :: qm_step_max, mbx_step_max, total_step_max
      double precision, intent(out) :: net_grad_norm, max_grad_abs

      integer :: iatom, ixyz
      double precision :: net(3)

      qm_step_max = 0.0d0
      mbx_step_max = 0.0d0
      max_grad_abs = 0.0d0
      net(:) = 0.0d0

      do iatom=1,natom_qm
         do ixyz=1,3
            qm_step_max = max(qm_step_max,abs(qm_grad1(ixyz,iatom)-qm_grad2(ixyz,iatom)))
            max_grad_abs = max(max_grad_abs,abs(qm_grad1(ixyz,iatom)))
            net(ixyz) = net(ixyz) + qm_grad1(ixyz,iatom)
         enddo
      enddo
      do iatom=1,natom_mbx
         do ixyz=1,3
            mbx_step_max = max(mbx_step_max,abs(mbx_grad1(ixyz,iatom)-mbx_grad2(ixyz,iatom)))
            max_grad_abs = max(max_grad_abs,abs(mbx_grad1(ixyz,iatom)))
            net(ixyz) = net(ixyz) + mbx_grad1(ixyz,iatom)
         enddo
      enddo

      total_step_max = max(qm_step_max,mbx_step_max)
      net_grad_norm = dsqrt(sum(net*net))
   end subroutine compare_gradients

   subroutine write_csv(csv_file,qm_grad1,qm_grad2,mbx_grad1,mbx_grad2,ierr)
      implicit none

      character(len=*), intent(in) :: csv_file
      double precision, intent(in) :: qm_grad1(3,natom_qm), qm_grad2(3,natom_qm)
      double precision, intent(in) :: mbx_grad1(3,natom_mbx), mbx_grad2(3,natom_mbx)
      integer, intent(inout) :: ierr

      integer :: unit, iatom, ixyz

      open(newunit=unit,file=csv_file,status='replace',action='write',iostat=ierr)
      if (ierr /= 0) return
      write(unit,'(A)') 'group,index,axis,grad_1e3_au_per_bohr,grad_2e3_au_per_bohr,abs_diff'
      do iatom=1,natom_qm
         do ixyz=1,3
            write(unit,'(A,",",I0,",",A,",",ES24.16,",",ES24.16,",",ES12.4)') &
               'qm_atom', iatom, axis_name(ixyz), qm_grad1(ixyz,iatom), qm_grad2(ixyz,iatom), &
               abs(qm_grad1(ixyz,iatom)-qm_grad2(ixyz,iatom))
         enddo
      enddo
      do iatom=1,natom_mbx
         do ixyz=1,3
            write(unit,'(A,",",I0,",",A,",",ES24.16,",",ES24.16,",",ES12.4)') &
               'mbx_atom', iatom, axis_name(ixyz), mbx_grad1(ixyz,iatom), mbx_grad2(ixyz,iatom), &
               abs(mbx_grad1(ixyz,iatom)-mbx_grad2(ixyz,iatom))
         enddo
      enddo
      close(unit)
   end subroutine write_csv

   subroutine write_summary(json_file,energy1,energy2,qm_step_max,mbx_step_max, &
         total_step_max,net_grad_norm,max_grad_abs,ierr)
      implicit none

      character(len=*), intent(in) :: json_file
      double precision, intent(in) :: energy1, energy2, qm_step_max, mbx_step_max
      double precision, intent(in) :: total_step_max, net_grad_norm, max_grad_abs
      integer, intent(inout) :: ierr

      integer :: unit

      open(newunit=unit,file=json_file,status='replace',action='write',iostat=ierr)
      if (ierr /= 0) return
      write(unit,'(A)') '{'
      write(unit,'(A,ES24.16,A)') '  "energy_step_1e3_au": ', energy1, ','
      write(unit,'(A,ES24.16,A)') '  "energy_step_2e3_au": ', energy2, ','
      write(unit,'(A,ES12.4,A)') '  "qm_step_sensitivity_max_au_per_bohr": ', qm_step_max, ','
      write(unit,'(A,ES12.4,A)') '  "mbx_step_sensitivity_max_au_per_bohr": ', mbx_step_max, ','
      write(unit,'(A,ES12.4,A)') '  "total_step_sensitivity_max_au_per_bohr": ', total_step_max, ','
      write(unit,'(A,ES12.4,A)') '  "net_gradient_norm_au_per_bohr": ', net_grad_norm, ','
      write(unit,'(A,ES12.4)') '  "max_gradient_abs_au_per_bohr": ', max_grad_abs
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
         write(*,'("QUICK_MBX_TOTAL_FD_ERROR ",A,1X,I0)') trim(label), ierr_in
         stop 1
      endif
   end subroutine stop_on_error

end program quick_mbx_total_fd_gradient_probe
