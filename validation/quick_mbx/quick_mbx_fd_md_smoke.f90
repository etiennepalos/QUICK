#include "util.fh"
!---------------------------------------------------------------------!
! Tiny finite-difference QUICK-MBX MD smoke driver.                   !
!                                                                     !
! This validates repeated total-energy/total-gradient API calls and a !
! conservative velocity-Verlet loop for CPU-only QM/MB-pol testing.   !
! The gradients are finite differences, not production analytic MD.   !
!_____________________________________________________________________!

program quick_mbx_fd_md_smoke

   use quick_api_module, only: setQuickJob, getQuickMBXEnergyGradientsFD, deleteQuickJob, clearQuickMBXSystem
   use quick_constants_module, only: BOHRS_TO_A

   implicit none

   integer, parameter :: natom_qm = 3
   integer, parameter :: nwaters = 1
   integer, parameter :: natom_mbx = 3*nwaters
   double precision, parameter :: fd_step_bohr = 2.0d-3
   double precision, parameter :: fs_to_au = 1.0d0/2.4188843265857d-2
   double precision, parameter :: amu_to_me = 1822.888486209d0

   integer :: ierr, argc, nsteps, istep
   integer :: atomic_numbers(natom_qm)
   double precision :: qm_xyz(3,natom_qm), mbx_xyz(3,natom_mbx)
   double precision :: qm_vel(3,natom_qm), mbx_vel(3,natom_mbx)
   double precision :: qm_acc(3,natom_qm), mbx_acc(3,natom_mbx)
   double precision :: qm_grad(3,natom_qm), mbx_grad(3,natom_mbx)
   double precision :: qm_mass(natom_qm), mbx_mass(natom_mbx)
   double precision :: energy, kinetic, total0, total, dt_fs, dt_au
   double precision :: max_drift, max_grad_abs, net_grad_norm
   character(len=256) :: csv_file, json_file, json_mbx, arg
   character(len=80) :: job_name
   character(len=256) :: keywd

   ierr = 0
   nsteps = 2
   dt_fs = 0.10d0
   csv_file = 'quick_mbx_fd_md_smoke.csv'
   json_file = 'quick_mbx_fd_md_smoke_summary.json'
   json_mbx = 'mbx.json'
   argc = command_argument_count()
   if (argc >= 1) call get_command_argument(1, csv_file)
   if (argc >= 2) call get_command_argument(2, json_file)
   if (argc >= 3) call get_command_argument(3, json_mbx)
   if (argc >= 4) then
      call get_command_argument(4, arg)
      read(arg,*,iostat=ierr) nsteps
      call stop_on_error(ierr,'read nsteps')
   endif
   if (argc >= 5) then
      call get_command_argument(5, arg)
      read(arg,*,iostat=ierr) dt_fs
      call stop_on_error(ierr,'read timestep')
   endif
   if (nsteps < 1 .or. dt_fs <= 0.0d0) then
      write(*,'("QUICK_MBX_FD_MD_ERROR invalid controls")')
      stop 1
   endif
   dt_au = dt_fs*fs_to_au

   atomic_numbers = (/8,1,1/)
   qm_mass = (/15.999d0, 1.008d0, 1.008d0/)*amu_to_me
   mbx_mass = qm_mass
   call setup_water_pair(qm_xyz,mbx_xyz)
   qm_vel(:,:) = 0.0d0
   mbx_vel(:,:) = 0.0d0

   job_name = 'quick_mbx_fd_md_smoke'
   keywd = 'HF BASIS=STO-3G CUTOFF=1.0D-10 DENSERMS=1.0D-6 MBX_QMMM'
   call setQuickJob(job_name,keywd,natom_qm,atomic_numbers,.true.,ierr)
   call stop_on_error(ierr,'setQuickJob')

   call getQuickMBXEnergyGradientsFD(qm_xyz,nwaters,mbx_xyz,json_mbx,fd_step_bohr, &
      energy,qm_grad,mbx_grad,ierr)
   call stop_on_error(ierr,'initial gradient')
   call gradients_to_accelerations(qm_grad,mbx_grad,qm_mass,mbx_mass,qm_acc,mbx_acc)
   kinetic = kinetic_energy(qm_vel,mbx_vel,qm_mass,mbx_mass)
   total0 = energy + kinetic
   total = total0
   max_drift = 0.0d0

   call write_header(csv_file,ierr)
   call stop_on_error(ierr,'write csv header')
   call summarize_gradient(qm_grad,mbx_grad,max_grad_abs,net_grad_norm)
   call append_step(csv_file,0,0.0d0,energy,kinetic,total,total-total0, &
      max_grad_abs,net_grad_norm,ierr)
   call stop_on_error(ierr,'write step 0')

   do istep=1,nsteps
      call half_kick(qm_vel,mbx_vel,qm_acc,mbx_acc,dt_au)
      call drift(qm_xyz,mbx_xyz,qm_vel,mbx_vel,dt_au)

      call getQuickMBXEnergyGradientsFD(qm_xyz,nwaters,mbx_xyz,json_mbx,fd_step_bohr, &
         energy,qm_grad,mbx_grad,ierr)
      call stop_on_error(ierr,'md gradient')
      call gradients_to_accelerations(qm_grad,mbx_grad,qm_mass,mbx_mass,qm_acc,mbx_acc)
      call half_kick(qm_vel,mbx_vel,qm_acc,mbx_acc,dt_au)

      kinetic = kinetic_energy(qm_vel,mbx_vel,qm_mass,mbx_mass)
      total = energy + kinetic
      max_drift = max(max_drift,abs(total-total0))
      call summarize_gradient(qm_grad,mbx_grad,max_grad_abs,net_grad_norm)
      call append_step(csv_file,istep,dble(istep)*dt_fs,energy,kinetic,total,total-total0, &
         max_grad_abs,net_grad_norm,ierr)
      call stop_on_error(ierr,'write md step')
   enddo

   call write_summary(json_file,nsteps,dt_fs,fd_step_bohr,total0,total,max_drift, &
      max_grad_abs,net_grad_norm,ierr)
   call stop_on_error(ierr,'write summary')

   call clearQuickMBXSystem(ierr)
   call deleteQuickJob(ierr)
   call stop_on_error(ierr,'deleteQuickJob')

   write(*,'("QUICK_MBX_FD_MD_STEPS ",I0)') nsteps
   write(*,'("QUICK_MBX_FD_MD_FINAL_TOTAL_AU ",ES24.16)') total
   write(*,'("QUICK_MBX_FD_MD_MAX_DRIFT_AU ",ES12.4)') max_drift
   write(*,'("QUICK_MBX_FD_MD_FINAL_MAX_GRAD ",ES12.4)') max_grad_abs

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

   subroutine gradients_to_accelerations(qm_grad,mbx_grad,qm_mass,mbx_mass,qm_acc,mbx_acc)
      implicit none

      double precision, intent(in) :: qm_grad(3,natom_qm), mbx_grad(3,natom_mbx)
      double precision, intent(in) :: qm_mass(natom_qm), mbx_mass(natom_mbx)
      double precision, intent(out) :: qm_acc(3,natom_qm), mbx_acc(3,natom_mbx)

      integer :: iatom

      do iatom=1,natom_qm
         qm_acc(:,iatom) = -qm_grad(:,iatom)/qm_mass(iatom)
      enddo
      do iatom=1,natom_mbx
         mbx_acc(:,iatom) = -mbx_grad(:,iatom)/mbx_mass(iatom)
      enddo
   end subroutine gradients_to_accelerations

   subroutine half_kick(qm_vel,mbx_vel,qm_acc,mbx_acc,dt_au)
      implicit none

      double precision, intent(inout) :: qm_vel(3,natom_qm), mbx_vel(3,natom_mbx)
      double precision, intent(in) :: qm_acc(3,natom_qm), mbx_acc(3,natom_mbx), dt_au

      qm_vel(:,:) = qm_vel(:,:) + 0.5d0*dt_au*qm_acc(:,:)
      mbx_vel(:,:) = mbx_vel(:,:) + 0.5d0*dt_au*mbx_acc(:,:)
   end subroutine half_kick

   subroutine drift(qm_xyz,mbx_xyz,qm_vel,mbx_vel,dt_au)
      implicit none

      double precision, intent(inout) :: qm_xyz(3,natom_qm), mbx_xyz(3,natom_mbx)
      double precision, intent(in) :: qm_vel(3,natom_qm), mbx_vel(3,natom_mbx), dt_au

      qm_xyz(:,:) = qm_xyz(:,:) + dt_au*qm_vel(:,:)*BOHRS_TO_A
      mbx_xyz(:,:) = mbx_xyz(:,:) + dt_au*mbx_vel(:,:)*BOHRS_TO_A
   end subroutine drift

   double precision function kinetic_energy(qm_vel,mbx_vel,qm_mass,mbx_mass)
      implicit none

      double precision, intent(in) :: qm_vel(3,natom_qm), mbx_vel(3,natom_mbx)
      double precision, intent(in) :: qm_mass(natom_qm), mbx_mass(natom_mbx)

      integer :: iatom

      kinetic_energy = 0.0d0
      do iatom=1,natom_qm
         kinetic_energy = kinetic_energy + 0.5d0*qm_mass(iatom)*sum(qm_vel(:,iatom)**2)
      enddo
      do iatom=1,natom_mbx
         kinetic_energy = kinetic_energy + 0.5d0*mbx_mass(iatom)*sum(mbx_vel(:,iatom)**2)
      enddo
   end function kinetic_energy

   subroutine summarize_gradient(qm_grad,mbx_grad,max_grad_abs,net_grad_norm)
      implicit none

      double precision, intent(in) :: qm_grad(3,natom_qm), mbx_grad(3,natom_mbx)
      double precision, intent(out) :: max_grad_abs, net_grad_norm

      integer :: iatom
      double precision :: net(3)

      max_grad_abs = 0.0d0
      net(:) = 0.0d0
      do iatom=1,natom_qm
         max_grad_abs = max(max_grad_abs,maxval(abs(qm_grad(:,iatom))))
         net(:) = net(:) + qm_grad(:,iatom)
      enddo
      do iatom=1,natom_mbx
         max_grad_abs = max(max_grad_abs,maxval(abs(mbx_grad(:,iatom))))
         net(:) = net(:) + mbx_grad(:,iatom)
      enddo
      net_grad_norm = dsqrt(sum(net*net))
   end subroutine summarize_gradient

   subroutine write_header(csv_file,ierr)
      implicit none

      character(len=*), intent(in) :: csv_file
      integer, intent(inout) :: ierr

      integer :: unit

      open(newunit=unit,file=csv_file,status='replace',action='write',iostat=ierr)
      if (ierr /= 0) return
      write(unit,'(A)') 'step,time_fs,potential_au,kinetic_au,total_au,delta_total_au,max_grad_au_per_bohr,net_grad_norm_au_per_bohr'
      close(unit)
   end subroutine write_header

   subroutine append_step(csv_file,istep,time_fs,energy,kinetic,total,drift, &
         max_grad_abs,net_grad_norm,ierr)
      implicit none

      character(len=*), intent(in) :: csv_file
      integer, intent(in) :: istep
      double precision, intent(in) :: time_fs, energy, kinetic, total, drift
      double precision, intent(in) :: max_grad_abs, net_grad_norm
      integer, intent(inout) :: ierr

      integer :: unit

      open(newunit=unit,file=csv_file,status='old',position='append',action='write',iostat=ierr)
      if (ierr /= 0) return
      write(unit,'(I0,",",F10.5,",",ES24.16,",",ES24.16,",",ES24.16,",",ES12.4,",",ES12.4,",",ES12.4)') &
         istep, time_fs, energy, kinetic, total, drift, max_grad_abs, net_grad_norm
      close(unit)
   end subroutine append_step

   subroutine write_summary(json_file,nsteps,dt_fs,fd_step_bohr,total0,total,max_drift, &
         max_grad_abs,net_grad_norm,ierr)
      implicit none

      character(len=*), intent(in) :: json_file
      integer, intent(in) :: nsteps
      double precision, intent(in) :: dt_fs, fd_step_bohr, total0, total
      double precision, intent(in) :: max_drift, max_grad_abs, net_grad_norm
      integer, intent(inout) :: ierr

      integer :: unit

      open(newunit=unit,file=json_file,status='replace',action='write',iostat=ierr)
      if (ierr /= 0) return
      write(unit,'(A)') '{'
      write(unit,'(A,I0,A)') '  "nsteps": ', nsteps, ','
      write(unit,'(A,ES12.4,A)') '  "dt_fs": ', dt_fs, ','
      write(unit,'(A,ES12.4,A)') '  "fd_step_bohr": ', fd_step_bohr, ','
      write(unit,'(A,ES24.16,A)') '  "initial_total_au": ', total0, ','
      write(unit,'(A,ES24.16,A)') '  "final_total_au": ', total, ','
      write(unit,'(A,ES12.4,A)') '  "max_abs_drift_au": ', max_drift, ','
      write(unit,'(A,ES12.4,A)') '  "final_max_gradient_abs_au_per_bohr": ', max_grad_abs, ','
      write(unit,'(A,ES12.4)') '  "final_net_gradient_norm_au_per_bohr": ', net_grad_norm
      write(unit,'(A)') '}'
      close(unit)
   end subroutine write_summary

   subroutine stop_on_error(ierr_in,label)
      implicit none

      integer, intent(in) :: ierr_in
      character(len=*), intent(in) :: label

      if (ierr_in /= 0) then
         write(*,'("QUICK_MBX_FD_MD_ERROR ",A,1X,I0)') trim(label), ierr_in
         stop 1
      endif
   end subroutine stop_on_error

end program quick_mbx_fd_md_smoke
