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
! Deterministic analytic-force QUICK-MBX NVE smoke trajectory.       !
! Coordinates passed to QUICK are Angstrom; integration uses atomic  !
! units, and returned gradients are hartree/bohr.                     !
!_____________________________________________________________________!

program quick_mbx_analytic_md_smoke

   use ieee_arithmetic, only: ieee_is_finite
   use quick_api_module, only: setQuickJob, setQuickMBXWaterSystem, setQuickMBXWaterCrossResidual, &
      getQuickMBXEnergyGradients, getQuickOEPROP, clearQuickMBXSystem, deleteQuickJob
   use quick_api_module, only: getQuickMBXEnergyTerms
   use quick_constants_module, only: BOHRS_TO_A
   implicit none

   integer, parameter :: natom_qm=3, natom_mbx=3, natom_total=6
   integer, parameter :: dof=3*natom_total-3
   double precision, parameter :: fs_to_au=1.0d0/2.4188843265857d-2
   double precision, parameter :: amu_to_me=1822.888486209d0
   double precision, parameter :: kb_au=3.166811563d-6
   double precision, parameter :: initial_temperature=50.0d0
   double precision, parameter :: drift_tolerance=1.0d-4
   double precision, parameter :: net_gradient_tolerance=1.0d-4

   integer :: ierr, argc, istep, nsteps, csv_unit, xyz_unit, oeprop_unit
   integer :: atomic_numbers(natom_qm)
   double precision :: qm_xyz(3,natom_qm), mbx_xyz(3,natom_mbx)
   double precision :: qm_vel(3,natom_qm), mbx_vel(3,natom_mbx)
   double precision :: qm_acc(3,natom_qm), mbx_acc(3,natom_mbx)
   double precision :: qm_grad(3,natom_qm), mbx_grad(3,natom_mbx)
   double precision :: qm_grad_old(3,natom_qm), mbx_grad_old(3,natom_mbx)
   double precision :: mbx_base(3,natom_mbx), induced_efg(3,natom_mbx)
   double precision :: qm_cross(3,natom_qm), mbx_cross(3,natom_mbx)
   double precision :: qm_mass(natom_qm), mbx_mass(natom_mbx)
   double precision :: energy, kinetic, total0, total, dt_fs, dt_au
   double precision :: drift, max_drift, force_jump, max_force_jump
   double precision :: max_grad, net_grad, max_net_grad, temperature
   double precision :: max_mbx_base, max_induced_efg, max_cross_residual
   double precision :: mbx_energy, mbx_external, mbx_additive, permanent_terms(3), induced_terms(3)
   double precision :: cross_terms(3), external_half_residual
   double precision :: esp(natom_mbx), efield(3,natom_mbx), efg(3,3,natom_mbx)
   character(len=256) :: csv_file, xyz_file, oeprop_file, summary_file, json_mbx, arg
   character(len=80) :: job_name
   character(len=256) :: keywd
   logical :: validation_failed

   ierr=0
   nsteps=20
   dt_fs=0.05d0
   csv_file='quick_mbx_analytic_md.csv'
   xyz_file='quick_mbx_analytic_md.xyz'
   oeprop_file='quick_mbx_analytic_md_oeprop.csv'
   summary_file='quick_mbx_analytic_md_summary.json'
   json_mbx='mbx.json'
   argc=command_argument_count()
   if (argc >= 1) call get_command_argument(1,csv_file)
   if (argc >= 2) call get_command_argument(2,xyz_file)
   if (argc >= 3) call get_command_argument(3,summary_file)
   if (argc >= 4) call get_command_argument(4,json_mbx)
   if (argc >= 5) then
      call get_command_argument(5,arg)
      read(arg,*,iostat=ierr) nsteps
      call stop_on_error(ierr,'read nsteps')
   endif
   if (argc >= 6) then
      call get_command_argument(6,arg)
      read(arg,*,iostat=ierr) dt_fs
      call stop_on_error(ierr,'read timestep')
   endif
   if (argc >= 7) call get_command_argument(7,oeprop_file)
   if (nsteps < 1 .or. dt_fs <= 0.0d0) call stop_on_error(45,'invalid controls')
   dt_au=dt_fs*fs_to_au

   atomic_numbers=(/8,1,1/)
   qm_mass=(/15.999d0,1.008d0,1.008d0/)*amu_to_me
   mbx_mass=qm_mass
   call setup_water_pair(qm_xyz,mbx_xyz)
   call initialize_velocities(qm_mass,mbx_mass,qm_vel,mbx_vel)

   keywd='HF BASIS=STO-3G CUTOFF=1.0D-12 GRADCUTOFF=1.0D-12 '// &
      'DENSERMS=1.0D-10 GRADIENT MBX_QMMM'
   job_name='quick_mbx_analytic_md'
   call setQuickJob(job_name,keywd,natom_qm,atomic_numbers,.true.,ierr)
   call stop_on_error(ierr,'setQuickJob')
   call setQuickMBXWaterSystem(1,mbx_xyz,json_mbx,ierr)
   call stop_on_error(ierr,'setQuickMBXWaterSystem')
   call setQuickMBXWaterCrossResidual(1,qm_xyz,1,mbx_xyz,json_mbx,ierr)
   call stop_on_error(ierr,'setQuickMBXWaterCrossResidual')
   call getQuickMBXEnergyGradients(qm_xyz,mbx_xyz,energy,qm_grad,mbx_grad,ierr,mbx_base,induced_efg, &
      mbx_cross,qm_cross)
   call stop_on_error(ierr,'initial energy and gradient')
   call getQuickMBXEnergyTerms(mbx_energy,mbx_external,mbx_additive,permanent_terms,induced_terms, &
      cross_terms,external_half_residual,ierr)
   call stop_on_error(ierr,'initial energy decomposition')
   call getQuickOEPROP(natom_mbx,mbx_xyz/BOHRS_TO_A,ierr,esp=esp,efield=efield,efg=efg)
   call stop_on_error(ierr,'initial OEPROP')

   call gradients_to_accelerations(qm_grad,mbx_grad,qm_mass,mbx_mass,qm_acc,mbx_acc)
   kinetic=kinetic_energy(qm_vel,mbx_vel,qm_mass,mbx_mass)
   total0=energy+kinetic
   total=total0
   max_drift=0.0d0
   max_force_jump=0.0d0
   max_net_grad=0.0d0
   force_jump=0.0d0

   open(newunit=csv_unit,file=csv_file,status='replace',action='write',iostat=ierr)
   call stop_on_error(ierr,'open csv')
   open(newunit=xyz_unit,file=xyz_file,status='replace',action='write',iostat=ierr)
   call stop_on_error(ierr,'open xyz')
   open(newunit=oeprop_unit,file=oeprop_file,status='replace',action='write',iostat=ierr)
   call stop_on_error(ierr,'open OEPROP csv')
   write(csv_unit,'(A)') 'step,time_fs,potential_au,kinetic_au,total_au,delta_total_au,'// &
      'temperature_K,max_gradient_au_per_bohr,net_gradient_au_per_bohr,'// &
      'max_gradient_change_au_per_bohr,max_mbx_base_gradient,max_induced_efg_gradient,'// &
      'max_cross_residual_gradient,cross_2b_au,cross_3b_au,cross_dispersion_au'
   write(oeprop_unit,'(A)') 'step,time_fs,probe,x_ang,y_ang,z_ang,esp,efield_x,efield_y,efield_z,'// &
      'efield_magnitude,efg_xx,efg_xy,efg_xz,efg_yx,efg_yy,efg_yz,efg_zx,efg_zy,efg_zz,'// &
      'efg_trace,efg_frobenius'

   call summarize(qm_grad,mbx_grad,mbx_base,induced_efg,qm_cross,mbx_cross,max_grad,net_grad, &
      max_mbx_base,max_induced_efg,max_cross_residual)
   max_net_grad=max(max_net_grad,net_grad)
   temperature=2.0d0*kinetic/(dble(dof)*kb_au)
   call write_step(csv_unit,xyz_unit,0,0.0d0,qm_xyz,mbx_xyz,energy,kinetic,total,0.0d0, &
      temperature,max_grad,net_grad,force_jump,max_mbx_base,max_induced_efg,max_cross_residual,cross_terms)
   call write_oeprop(oeprop_unit,0,0.0d0,mbx_xyz,esp,efield,efg)

   do istep=1,nsteps
      qm_grad_old=qm_grad
      mbx_grad_old=mbx_grad
      call half_kick(qm_vel,mbx_vel,qm_acc,mbx_acc,dt_au)
      call drift_coordinates(qm_xyz,mbx_xyz,qm_vel,mbx_vel,dt_au)

      call getQuickMBXEnergyGradients(qm_xyz,mbx_xyz,energy,qm_grad,mbx_grad,ierr,mbx_base,induced_efg, &
         mbx_cross,qm_cross)
      call stop_on_error(ierr,'MD energy and gradient')
      call getQuickMBXEnergyTerms(mbx_energy,mbx_external,mbx_additive,permanent_terms,induced_terms, &
         cross_terms,external_half_residual,ierr)
      call stop_on_error(ierr,'MD energy decomposition')
      call getQuickOEPROP(natom_mbx,mbx_xyz/BOHRS_TO_A,ierr,esp=esp,efield=efield,efg=efg)
      call stop_on_error(ierr,'MD OEPROP')
      call gradients_to_accelerations(qm_grad,mbx_grad,qm_mass,mbx_mass,qm_acc,mbx_acc)
      call half_kick(qm_vel,mbx_vel,qm_acc,mbx_acc,dt_au)

      kinetic=kinetic_energy(qm_vel,mbx_vel,qm_mass,mbx_mass)
      total=energy+kinetic
      drift=total-total0
      force_jump=max(maxval(abs(qm_grad-qm_grad_old)),maxval(abs(mbx_grad-mbx_grad_old)))
      max_drift=max(max_drift,abs(drift))
      max_force_jump=max(max_force_jump,force_jump)
      call summarize(qm_grad,mbx_grad,mbx_base,induced_efg,qm_cross,mbx_cross,max_grad,net_grad, &
         max_mbx_base,max_induced_efg,max_cross_residual)
      max_net_grad=max(max_net_grad,net_grad)
      temperature=2.0d0*kinetic/(dble(dof)*kb_au)
      call write_step(csv_unit,xyz_unit,istep,dble(istep)*dt_fs,qm_xyz,mbx_xyz,energy,kinetic,total,drift, &
         temperature,max_grad,net_grad,force_jump,max_mbx_base,max_induced_efg,max_cross_residual,cross_terms)
      call write_oeprop(oeprop_unit,istep,dble(istep)*dt_fs,mbx_xyz,esp,efield,efg)
   enddo
   close(csv_unit)
   close(xyz_unit)
   close(oeprop_unit)

   call write_summary(summary_file,nsteps,dt_fs,total0,total,max_drift,max_force_jump,max_net_grad,ierr)
   call stop_on_error(ierr,'write summary')
   validation_failed=.not. ieee_is_finite(total) .or. max_drift > drift_tolerance .or. &
      max_net_grad > net_gradient_tolerance

   call clearQuickMBXSystem(ierr)
   call stop_on_error(ierr,'clearQuickMBXSystem')
   call deleteQuickJob(ierr)
   call stop_on_error(ierr,'deleteQuickJob')

   write(*,'("QUICK_MBX_ANALYTIC_MD_STEPS ",I0)') nsteps
   write(*,'("QUICK_MBX_ANALYTIC_MD_MAX_DRIFT_AU ",ES12.4)') max_drift
   write(*,'("QUICK_MBX_ANALYTIC_MD_MAX_GRADIENT_CHANGE ",ES12.4)') max_force_jump
   write(*,'("QUICK_MBX_ANALYTIC_MD_MAX_NET_GRADIENT ",ES12.4)') max_net_grad
   if (validation_failed) stop 2

contains

   subroutine setup_water_pair(qm_xyz,mbx_xyz)
      implicit none
      double precision, intent(out) :: qm_xyz(3,natom_qm), mbx_xyz(3,natom_mbx)

      qm_xyz(:,1)=(/-1.58972425d0,1.04337922d0,-0.08780840d0/)
      qm_xyz(:,2)=(/-0.63591971d0,0.97898520d0, 0.00000000d0/)
      qm_xyz(:,3)=(/-1.90066280d0,1.74501050d0,-0.66454990d0/)
      mbx_xyz(:,1)=(/1.64924507d0,1.08594656d0, 0.00000000d0/)
      mbx_xyz(:,2)=(/2.60878026d0,1.09587704d0,-0.02817115d0/)
      mbx_xyz(:,3)=(/1.33830653d0,1.78757784d0, 0.57674150d0/)
   end subroutine setup_water_pair

   subroutine initialize_velocities(qm_mass,mbx_mass,qm_vel,mbx_vel)
      implicit none
      double precision, intent(in) :: qm_mass(natom_qm), mbx_mass(natom_mbx)
      double precision, intent(out) :: qm_vel(3,natom_qm), mbx_vel(3,natom_mbx)
      integer :: i, iatom
      double precision :: momentum(3), total_mass, scale, target_kinetic

      do iatom=1,natom_qm
         do i=1,3
            qm_vel(i,iatom)=1.0d-4*dble(mod(7*i+11*iatom,9)-4)
            mbx_vel(i,iatom)=1.0d-4*dble(mod(5*i+13*(iatom+natom_qm),9)-4)
         enddo
      enddo
      momentum=sum(qm_vel*spread(qm_mass,1,3),dim=2)+sum(mbx_vel*spread(mbx_mass,1,3),dim=2)
      total_mass=sum(qm_mass)+sum(mbx_mass)
      qm_vel=qm_vel-spread(momentum/total_mass,2,natom_qm)
      mbx_vel=mbx_vel-spread(momentum/total_mass,2,natom_mbx)
      target_kinetic=0.5d0*dble(dof)*kb_au*initial_temperature
      scale=dsqrt(target_kinetic/kinetic_energy(qm_vel,mbx_vel,qm_mass,mbx_mass))
      qm_vel=scale*qm_vel
      mbx_vel=scale*mbx_vel
   end subroutine initialize_velocities

   subroutine gradients_to_accelerations(qm_grad,mbx_grad,qm_mass,mbx_mass,qm_acc,mbx_acc)
      implicit none
      double precision, intent(in) :: qm_grad(3,natom_qm), mbx_grad(3,natom_mbx)
      double precision, intent(in) :: qm_mass(natom_qm), mbx_mass(natom_mbx)
      double precision, intent(out) :: qm_acc(3,natom_qm), mbx_acc(3,natom_mbx)

      qm_acc=-qm_grad/spread(qm_mass,1,3)
      mbx_acc=-mbx_grad/spread(mbx_mass,1,3)
   end subroutine gradients_to_accelerations

   subroutine half_kick(qm_vel,mbx_vel,qm_acc,mbx_acc,dt)
      implicit none
      double precision, intent(inout) :: qm_vel(3,natom_qm), mbx_vel(3,natom_mbx)
      double precision, intent(in) :: qm_acc(3,natom_qm), mbx_acc(3,natom_mbx), dt

      qm_vel=qm_vel+0.5d0*dt*qm_acc
      mbx_vel=mbx_vel+0.5d0*dt*mbx_acc
   end subroutine half_kick

   subroutine drift_coordinates(qm_xyz,mbx_xyz,qm_vel,mbx_vel,dt)
      implicit none
      double precision, intent(inout) :: qm_xyz(3,natom_qm), mbx_xyz(3,natom_mbx)
      double precision, intent(in) :: qm_vel(3,natom_qm), mbx_vel(3,natom_mbx), dt

      qm_xyz=qm_xyz+dt*qm_vel*BOHRS_TO_A
      mbx_xyz=mbx_xyz+dt*mbx_vel*BOHRS_TO_A
   end subroutine drift_coordinates

   double precision function kinetic_energy(qm_vel,mbx_vel,qm_mass,mbx_mass)
      implicit none
      double precision, intent(in) :: qm_vel(3,natom_qm), mbx_vel(3,natom_mbx)
      double precision, intent(in) :: qm_mass(natom_qm), mbx_mass(natom_mbx)

      kinetic_energy=0.5d0*sum(qm_vel*qm_vel*spread(qm_mass,1,3))+ &
         0.5d0*sum(mbx_vel*mbx_vel*spread(mbx_mass,1,3))
   end function kinetic_energy

   subroutine summarize(qm_grad,mbx_grad,mbx_base,induced,qm_cross,mbx_cross,max_grad,net_grad, &
         max_base,max_induced,max_cross)
      implicit none
      double precision, intent(in) :: qm_grad(3,natom_qm), mbx_grad(3,natom_mbx)
      double precision, intent(in) :: mbx_base(3,natom_mbx), induced(3,natom_mbx)
      double precision, intent(in) :: qm_cross(3,natom_qm), mbx_cross(3,natom_mbx)
      double precision, intent(out) :: max_grad, net_grad, max_base, max_induced, max_cross
      double precision :: net(3)

      max_grad=max(maxval(abs(qm_grad)),maxval(abs(mbx_grad)))
      net=sum(qm_grad,dim=2)+sum(mbx_grad,dim=2)
      net_grad=dsqrt(dot_product(net,net))
      max_base=maxval(abs(mbx_base))
      max_induced=maxval(abs(induced))
      max_cross=max(maxval(abs(qm_cross)),maxval(abs(mbx_cross)))
   end subroutine summarize

   subroutine write_step(csv_unit,xyz_unit,step,time_fs,qm_xyz,mbx_xyz,potential,kinetic,total,drift, &
         temperature,max_grad,net_grad,force_jump,max_base,max_induced,max_cross,cross_terms)
      implicit none
      integer, intent(in) :: csv_unit, xyz_unit, step
      double precision, intent(in) :: time_fs, qm_xyz(3,natom_qm), mbx_xyz(3,natom_mbx)
      double precision, intent(in) :: potential, kinetic, total, drift, temperature
      double precision, intent(in) :: max_grad, net_grad, force_jump, max_base, max_induced, max_cross
      double precision, intent(in) :: cross_terms(3)
      integer :: iatom

      write(csv_unit,'(I0,15(",",ES24.16))') step,time_fs,potential,kinetic,total,drift,temperature, &
         max_grad,net_grad,force_jump,max_base,max_induced,max_cross,cross_terms
      write(xyz_unit,'(I0)') natom_total
      write(xyz_unit,'("step=",I0,1X,"time_fs=",F10.5,1X,"total_au=",ES24.16)') step,time_fs,total
      do iatom=1,natom_qm
         write(xyz_unit,'(A,3(1X,F18.10))') atom_symbol(iatom),qm_xyz(:,iatom)
      enddo
      do iatom=1,natom_mbx
         write(xyz_unit,'(A,3(1X,F18.10))') atom_symbol(iatom),mbx_xyz(:,iatom)
      enddo
   end subroutine write_step

   subroutine write_oeprop(unit,step,time_fs,probes_ang,esp,efield,efg)
      implicit none
      integer, intent(in) :: unit,step
      double precision, intent(in) :: time_fs,probes_ang(3,natom_mbx)
      double precision, intent(in) :: esp(natom_mbx),efield(3,natom_mbx),efg(3,3,natom_mbx)
      integer :: iprobe
      double precision :: field_magnitude,trace,frobenius

      do iprobe=1,natom_mbx
         field_magnitude=sqrt(sum(efield(:,iprobe)**2))
         trace=efg(1,1,iprobe)+efg(2,2,iprobe)+efg(3,3,iprobe)
         frobenius=sqrt(sum(efg(:,:,iprobe)**2))
         write(unit,'(I0,",",ES24.16,",",I0,19(",",ES24.16))') step,time_fs,iprobe, &
            probes_ang(:,iprobe),esp(iprobe),efield(:,iprobe),field_magnitude, &
            efg(1,1,iprobe),efg(1,2,iprobe),efg(1,3,iprobe), &
            efg(2,1,iprobe),efg(2,2,iprobe),efg(2,3,iprobe), &
            efg(3,1,iprobe),efg(3,2,iprobe),efg(3,3,iprobe),trace,frobenius
      enddo
   end subroutine write_oeprop

   character(len=1) function atom_symbol(iatom)
      implicit none
      integer, intent(in) :: iatom
      if (iatom == 1) then
         atom_symbol='O'
      else
         atom_symbol='H'
      endif
   end function atom_symbol

   subroutine write_summary(file,nsteps,dt_fs,total0,total,max_drift,max_force_jump,max_net_grad,ierr)
      implicit none
      character(len=*), intent(in) :: file
      integer, intent(in) :: nsteps
      double precision, intent(in) :: dt_fs,total0,total,max_drift,max_force_jump,max_net_grad
      integer, intent(out) :: ierr
      integer :: unit

      open(newunit=unit,file=file,status='replace',action='write',iostat=ierr)
      if (ierr /= 0) return
      write(unit,'(A)') '{'
      write(unit,'(A)') '  "method": "HF/STO-3G + MB-pol",'
      write(unit,'(A,I0,A)') '  "nsteps": ',nsteps,','
      write(unit,'(A,ES24.16,A)') '  "dt_fs": ',dt_fs,','
      write(unit,'(A,ES24.16,A)') '  "initial_total_au": ',total0,','
      write(unit,'(A,ES24.16,A)') '  "final_total_au": ',total,','
      write(unit,'(A,ES24.16,A)') '  "max_abs_drift_au": ',max_drift,','
      write(unit,'(A,ES24.16,A)') '  "max_gradient_change_au_per_bohr": ',max_force_jump,','
      write(unit,'(A,ES24.16)') '  "max_net_gradient_au_per_bohr": ',max_net_grad
      write(unit,'(A)') '}'
      close(unit)
   end subroutine write_summary

   subroutine stop_on_error(ierr_in,label)
      implicit none
      integer, intent(in) :: ierr_in
      character(len=*), intent(in) :: label

      if (ierr_in /= 0) then
         write(*,'("QUICK_MBX_ANALYTIC_MD_ERROR ",A,1X,I0)') trim(label),ierr_in
         stop 1
      endif
   end subroutine stop_on_error

end program quick_mbx_analytic_md_smoke
