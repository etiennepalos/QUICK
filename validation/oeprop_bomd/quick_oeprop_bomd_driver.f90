#include "util.fh"
!---------------------------------------------------------------------!
! Copyright (C) 2026 QUICK contributors                               !
! All rights reserved.                                                !
!                                                                     !
! This Source Code Form is subject to the terms of the Mozilla Public !
! License, v. 2.0. If a copy of the MPL was not distributed with this !
! file, You can obtain one at http://mozilla.org/MPL/2.0/.            !
!_____________________________________________________________________!

program quick_oeprop_bomd_driver

  use iso_fortran_env, only: int64
  use ieee_arithmetic, only: ieee_is_finite
  use quick_api_module, only: setQuickJob, getQuickEnergyGradients, &
    getQuickOEPROP, getQuickFieldHessian, getQuickDensitySurface, deleteQuickJob, quick_api
  use quick_constants_module, only: BOHRS_TO_A
  use quick_input_parser_module, only: found_keyword
  use quick_oei_module, only: get_external_field_gpu_counts
  use quick_field_hessian_module, only: get_field_hessian_gpu_count
#ifdef MPIV
  use mpi
  use quick_api_module, only: setQuickMPI
  use quick_mpi_module, only: quick_comm, quick_comm_rank, quick_comm_size, quick_mpi_error
#endif
  implicit none

  integer, parameter :: max_control_atoms=512
  double precision, parameter :: fs_to_au=1.0d0/2.4188843265857d-2
  double precision, parameter :: amu_to_me=1822.888486209d0
  double precision, parameter :: kb_au=3.166811563d-6
  double precision, parameter :: pi=acos(-1.0d0)

  character(len=256) :: control_file, geometry_file, probe_file, output_prefix
  character(len=256) :: checkpoint_file, restart_file, keywd
  character(len=32) :: probe_mode
  integer :: nsteps, oeprop_stride, nfrozen, frozen_atoms(max_control_atoms)
  integer :: anchor_atoms(3), coordinate_atoms(2)
  double precision :: dt_fs, temperature_k
  double precision :: density_isovalue, density_spacing_ang, density_max_radius_ang
  logical :: field_hessian
  namelist /quick_bomd/ geometry_file,probe_file,output_prefix,checkpoint_file, &
    restart_file,keywd,probe_mode,nsteps,oeprop_stride,dt_fs,temperature_k,nfrozen,frozen_atoms, &
    anchor_atoms,coordinate_atoms,density_isovalue,density_spacing_ang,density_max_radius_ang,field_hessian

  integer :: ierr, argc, unit, ios, natom, npoints, start_step, step, local_step
  integer :: energy_unit, xyz_unit, oeprop_unit, active_dof
  integer :: field_hessian_unit,field_hessian_calls,hessian_launches_before,hessian_launches_after
  integer, allocatable :: atomic_numbers(:), probe_atoms(:)
  logical, allocatable :: frozen(:)
  double precision, allocatable :: coords(:,:), velocities(:,:), accelerations(:,:)
  double precision, allocatable :: gradients(:,:), previous_gradients(:,:), masses(:)
  double precision, allocatable :: probe_reference(:,:), probes_ang(:,:), probes_bohr(:,:)
  double precision, allocatable :: esp(:), efield(:,:), efg(:,:,:)
  double precision, allocatable :: hessian_values(:,:,:,:)
  double precision :: ptchg(4,0), ptchg_grad(3,0)
  double precision :: potential, kinetic, total, total0, drift, dt_au
  double precision :: max_drift, force_change, max_force_change, max_gradient
  double precision :: net_gradient, max_net_gradient, current_temperature, reaction_coordinate
  character(len=80) :: job_name
  character(len=256) :: energy_file, xyz_file, oeprop_file, summary_file, kernels_file
  character(len=16) :: backend
  logical :: restarting, restart_hamiltonian_verified
  integer :: rank_id, rank_count, force_calls, api_energy_gradient_calls
  integer :: property_calls, probe_generation_calls, properties_extra_api_calls
  integer :: launches_before(3), launches_after(3), launch_deltas(3)
  integer :: field_before(2), field_after(2), field_deltas(2)
  double precision :: force_wall_seconds, probe_wall_seconds, oeprop_wall_seconds
  logical :: is_master

  geometry_file='geometry.xyz'
  probe_file='probes.dat'
  output_prefix='quick_oeprop_bomd'
  checkpoint_file=''
  restart_file=''
  keywd='HF BASIS=STO-3G CUTOFF=1.0D-12 GRADCUTOFF=1.0D-12 DENSERMS=1.0D-10 GRADIENT CHARGE=0'
  probe_mode='fixed'
  nsteps=10
  oeprop_stride=1
  field_hessian=.false.
  field_hessian_unit=-1
  field_hessian_calls=0
  dt_fs=0.05d0
  temperature_k=50.0d0
  nfrozen=0
  frozen_atoms=0
  anchor_atoms=(/1,2,3/)
  coordinate_atoms=(/1,2/)
  density_isovalue=1.0d-3
  density_spacing_ang=0.75d0
  density_max_radius_ang=8.0d0
  ierr=0
  is_master=.true.
  rank_id=0
  rank_count=1
  backend='cpu'
#if defined(CUDA) || defined(CUDA_MPIV)
  backend='cuda'
#elif defined(HIP) || defined(HIP_MPIV)
  backend='hip'
#elif defined(GPU) || defined(MPIV_GPU)
  backend='gpu'
#endif
  force_calls=0
  api_energy_gradient_calls=0
  property_calls=0
  probe_generation_calls=0
  properties_extra_api_calls=0
  force_wall_seconds=0.0d0
  probe_wall_seconds=0.0d0
  oeprop_wall_seconds=0.0d0
  launches_before=0
  launches_after=0
  restart_hamiltonian_verified=.true.
#ifdef MPIV
  call MPI_INIT(quick_mpi_error)
  call setQuickMPI(MPI_COMM_WORLD,ierr)
  is_master=quick_comm_rank == 0
  rank_id=quick_comm_rank
  rank_count=quick_comm_size
  call stop_on_error(ierr,'setQuickMPI')
#endif

  argc=command_argument_count()
  if (argc /= 1) then
    if (is_master) write(*,'("Usage: test-api-oeprop-bomd <control.nml>")')
    call stop_on_error(45,'command line')
  endif
  call get_command_argument(1,control_file)
  open(newunit=unit,file=control_file,status='old',action='read',iostat=ierr)
  call stop_on_error(ierr,'open control')
  read(unit,nml=quick_bomd,iostat=ios)
  close(unit)
  call stop_on_error(merge(0,45,ios == 0),'read control')
  if (.not. ieee_is_finite(dt_fs) .or. .not. ieee_is_finite(temperature_k)) &
    call stop_on_error(45,'nonfinite controls')
  if (nsteps < 0 .or. oeprop_stride < 0 .or. dt_fs <= 0.0d0 .or. temperature_k < 0.0d0) &
    call stop_on_error(45,'invalid controls')
  keywd=uppercase(keywd)
  if (found_keyword(keywd,'OPT') .or. found_keyword(keywd,'OPTIMIZE')) &
    call stop_on_error(45,'optimizer keyword in BOMD')
  if (found_keyword(keywd,'MP2')) call stop_on_error(45,'MP2 unavailable in energy-gradient API')
  if (found_keyword(keywd,'ESP') .or. found_keyword(keywd,'RESP') .or. &
      found_keyword(keywd,'ESP_CHARGE') .or. found_keyword(keywd,'ESP_GRID') .or. &
      found_keyword(keywd,'ESP_SURFACE') .or. found_keyword(keywd,'EFIELD') .or. &
      found_keyword(keywd,'EFIELD_GRID') .or. found_keyword(keywd,'EFIELD_SURFACE') .or. &
      found_keyword(keywd,'EFG') .or. found_keyword(keywd,'EFG_GRID') .or. &
      found_keyword(keywd,'EFG_GRID_NUMERICAL') .or. found_keyword(keywd,'EFG_SURFACE')) &
    call stop_on_error(45,'property keywords belong in BOMD probe controls')
  if (.not. found_keyword(keywd,'GRADIENT')) then
    if (len_trim(keywd)+9 > len(keywd)) call stop_on_error(45,'keyword length')
    keywd=trim(keywd)//' GRADIENT'
  endif
  keywd=uppercase(keywd)
  if (nfrozen < 0 .or. nfrozen > max_control_atoms) call stop_on_error(45,'invalid frozen atoms')
  if (len_trim(checkpoint_file) == 0) checkpoint_file=trim(output_prefix)//'.chk'
  restarting=len_trim(restart_file) > 0

  if (restarting) then
    call read_checkpoint(restart_file,natom,start_step,dt_fs,total0,atomic_numbers,coords,velocities,frozen,ierr)
    call stop_on_error(ierr,'read checkpoint')
  else
    call read_xyz(geometry_file,natom,atomic_numbers,coords,ierr)
    call stop_on_error(ierr,'read geometry')
    allocate(velocities(3,natom),frozen(natom),stat=ierr)
    call stop_on_error(ierr,'allocate dynamics state')
    frozen=.false.
    do unit=1,nfrozen
      if (frozen_atoms(unit) < 1 .or. frozen_atoms(unit) > natom) call stop_on_error(45,'frozen atom index')
      frozen(frozen_atoms(unit))=.true.
    enddo
    start_step=0
  endif
  call validate_atom_indices(natom,anchor_atoms,coordinate_atoms,ierr)
  call stop_on_error(ierr,'atom index controls')

  allocate(masses(natom),accelerations(3,natom),gradients(3,natom),previous_gradients(3,natom),stat=ierr)
  call stop_on_error(ierr,'allocate dynamics arrays')
  call assign_masses(atomic_numbers,masses,ierr)
  call stop_on_error(ierr,'atomic masses')
  if (.not. restarting) call initialize_velocities(atomic_numbers,masses,frozen,temperature_k,velocities)
  where(spread(frozen,1,3)) velocities=0.0d0
  active_dof=max(1,3*count(.not. frozen)-3)
  dt_au=dt_fs*fs_to_au

  npoints=0
  if (oeprop_stride > 0) then
    call read_probe_definition(probe_mode,probe_file,npoints,probe_reference,probe_atoms,ierr)
    call stop_on_error(ierr,'probe definition')
  endif
  job_name='quick_oeprop_bomd'
  call setQuickJob(job_name,keywd,natom,atomic_numbers,.true.,ierr)
  call stop_on_error(ierr,'setQuickJob')
  call get_external_field_gpu_counts(field_before(1),field_before(2))
  call get_field_hessian_gpu_count(hessian_launches_before)
#if defined(GPU) || defined(MPIV_GPU)
  call gpu_oeprop_launch_counts(launches_before(1),launches_before(2),launches_before(3))
#endif
  call evaluate_forces()
  call gradients_to_accelerations(gradients,masses,frozen,accelerations)
  kinetic=kinetic_energy(velocities,masses)
  if (.not. restarting) total0=potential+kinetic
  total=potential+kinetic

  energy_file=trim(output_prefix)//'_energy.csv'
  xyz_file=trim(output_prefix)//'_trajectory.xyz'
  oeprop_file=trim(output_prefix)//'_oeprop.csv'
  summary_file=trim(output_prefix)//'_summary.json'
  kernels_file=trim(output_prefix)//'_kernels.csv'
  call open_outputs(energy_file,xyz_file,oeprop_file,energy_unit,xyz_unit,oeprop_unit,ierr)
  call stop_on_error(ierr,'open outputs')
  if (field_hessian) then
    if (is_master) then
      open(newunit=field_hessian_unit,file=trim(output_prefix)//'_field_hessian.csv', &
        status='replace',action='write',iostat=ierr)
      if (ierr == 0) write(field_hessian_unit,'(A)') &
        'step,time_fs,probe,x_ang,y_ang,z_ang,h_xxx,h_xxy,h_xxz,h_xyy,h_xyz,h_xzz,h_yyy,h_yyz,h_yzz,h_zzz'
    endif
#ifdef MPIV
    call MPI_BCAST(ierr,1,MPI_INTEGER,0,quick_comm,quick_mpi_error)
#endif
    call stop_on_error(ierr,'open field Hessian output')
  endif

  max_drift=abs(total-total0)
  max_force_change=0.0d0
  call summarize_gradient(gradients,max_gradient,net_gradient)
  max_net_gradient=net_gradient
  current_temperature=2.0d0*kinetic/(dble(active_dof)*kb_au)
  reaction_coordinate=distance(coords(:,coordinate_atoms(1)),coords(:,coordinate_atoms(2)))
  call write_dynamics_frame(energy_unit,xyz_unit,start_step,dt_fs,atomic_numbers,coords, &
    potential,kinetic,total,total-total0,current_temperature,max_gradient,net_gradient,0.0d0)
  call sample_properties(start_step)
  call write_checkpoint(checkpoint_file,natom,start_step,dt_fs,total0,atomic_numbers,coords,velocities,frozen,ierr)
  call stop_on_error(ierr,'initial checkpoint')

  do local_step=1,nsteps
    step=start_step+local_step
    previous_gradients=gradients
    velocities=velocities+0.5d0*dt_au*accelerations
    coords=coords+dt_au*velocities*BOHRS_TO_A

    call evaluate_forces()
    call gradients_to_accelerations(gradients,masses,frozen,accelerations)
    velocities=velocities+0.5d0*dt_au*accelerations
    where(spread(frozen,1,3)) velocities=0.0d0

    kinetic=kinetic_energy(velocities,masses)
    total=potential+kinetic
    drift=total-total0
    force_change=maxval(abs(gradients-previous_gradients))
    max_drift=max(max_drift,abs(drift))
    max_force_change=max(max_force_change,force_change)
    call summarize_gradient(gradients,max_gradient,net_gradient)
    max_net_gradient=max(max_net_gradient,net_gradient)

    current_temperature=2.0d0*kinetic/(dble(active_dof)*kb_au)
    reaction_coordinate=distance(coords(:,coordinate_atoms(1)),coords(:,coordinate_atoms(2)))
    call write_dynamics_frame(energy_unit,xyz_unit,step,dt_fs,atomic_numbers,coords, &
      potential,kinetic,total,drift,current_temperature,max_gradient,net_gradient,force_change)
    call sample_properties(step)
    call write_checkpoint(checkpoint_file,natom,step,dt_fs,total0,atomic_numbers,coords,velocities,frozen,ierr)
    call stop_on_error(ierr,'checkpoint')
  enddo

  if (is_master) then
    close(energy_unit)
    close(xyz_unit)
    close(oeprop_unit)
    if (field_hessian) close(field_hessian_unit)
  endif
#if defined(GPU) || defined(MPIV_GPU)
  call gpu_oeprop_launch_counts(launches_after(1),launches_after(2),launches_after(3))
#endif
  launch_deltas=launches_after-launches_before
  call get_external_field_gpu_counts(field_after(1),field_after(2))
  call get_field_hessian_gpu_count(hessian_launches_after)
  field_deltas=field_after-field_before
  call write_kernel_counts(kernels_file,ierr)
  call stop_on_error(ierr,'write launch counts')
  call write_summary(summary_file,start_step+nsteps,nsteps,dt_fs,total0,total,max_drift, &
    max_force_change,max_net_gradient,probe_mode,ierr)
  call stop_on_error(ierr,'write summary')
  call deleteQuickJob(ierr)
  call stop_on_error(ierr,'deleteQuickJob')

  if (is_master) then
    write(*,'("QUICK_OEPROP_BOMD_FINAL_STEP ",I0)') start_step+nsteps
    write(*,'("QUICK_OEPROP_BOMD_MAX_DRIFT_AU ",ES12.4)') max_drift
    write(*,'("QUICK_OEPROP_BOMD_MAX_FORCE_CHANGE ",ES12.4)') max_force_change
    write(*,'("QUICK_OEPROP_BOMD_MAX_NET_GRADIENT ",ES12.4)') max_net_gradient
  endif
#ifdef MPIV
  call MPI_FINALIZE(quick_mpi_error)
#endif

contains

  double precision function wall_time_seconds()
    implicit none
#ifndef MPIV
    integer(int64) :: ticks, ticks_per_second
#endif
#ifdef MPIV
    wall_time_seconds=MPI_WTIME()
#else
    call system_clock(ticks,ticks_per_second)
    if (ticks_per_second <= 0_int64) call stop_on_error(45,'wall clock unavailable')
    wall_time_seconds=dble(ticks)/dble(ticks_per_second)
#endif
  end function wall_time_seconds

  subroutine evaluate_forces()
    use quick_method_module, only: quick_method
    implicit none
    integer :: before_step,delta,convergence_status
#ifdef MPIV
    integer :: global_convergence_status
#endif
    double precision :: begin_time

    before_step=quick_api%step
    begin_time=wall_time_seconds()
    call getQuickEnergyGradients(coords,0,ptchg,potential,gradients,ptchg_grad,ierr)
    force_wall_seconds=force_wall_seconds+wall_time_seconds()-begin_time
    call stop_on_error(ierr,'energy and gradient')
    convergence_status=0
    if (quick_method%unrst) then
      if (.not.quick_method%uscf_conv) convergence_status=53
    else
      if (.not.quick_method%scf_conv) convergence_status=53
    endif
#ifdef MPIV
    call MPI_ALLREDUCE(convergence_status,global_convergence_status,1,MPI_INTEGER,MPI_MAX,quick_comm,quick_mpi_error)
    call stop_on_error(quick_mpi_error,'check SCF convergence across job ranks')
    convergence_status=global_convergence_status
#endif
    call stop_on_error(convergence_status,'unconverged SCF in force evaluation')
    delta=quick_api%step-before_step
    force_calls=force_calls+1
    api_energy_gradient_calls=api_energy_gradient_calls+delta
    call stop_on_error(merge(0,53,delta == 1),'one API evaluation per force call')
    if (.not. ieee_is_finite(potential) .or. .not. all(ieee_is_finite(gradients))) &
      call stop_on_error(45,'nonfinite energy or gradient')
  end subroutine evaluate_forces

  subroutine sample_properties(frame_step)
    implicit none
    integer, intent(in) :: frame_step
    integer :: before_step,delta,i,j
    double precision :: begin_time

    if (oeprop_stride == 0) return
    if (mod(frame_step,oeprop_stride) /= 0) return
    before_step=quick_api%step
    begin_time=wall_time_seconds()
    call prepare_probes(probe_mode,coords,anchor_atoms,density_isovalue,density_spacing_ang, &
      density_max_radius_ang,probe_reference,probe_atoms,npoints,probes_ang,probes_bohr,ierr)
    call stop_on_error(ierr,'prepare probes')
    if (npoints < 1 .or. .not. all(ieee_is_finite(probes_ang))) &
      call stop_on_error(45,'invalid probes')
    do i=1,npoints
      do j=1,natom
        if (distance(probes_ang(:,i),coords(:,j)) <= 1.0d-12) &
          call stop_on_error(45,'probe at nuclear singularity')
      enddo
    enddo
    probe_wall_seconds=probe_wall_seconds+wall_time_seconds()-begin_time
    probe_generation_calls=probe_generation_calls+1
    delta=quick_api%step-before_step
    properties_extra_api_calls=properties_extra_api_calls+delta
    call stop_on_error(merge(0,53,delta == 0),'probe generation changed API step')

    before_step=quick_api%step
    begin_time=wall_time_seconds()
    call evaluate_properties(npoints,probes_bohr,esp,efield,efg,ierr)
    call stop_on_error(ierr,'on-the-fly OEPROP')
    if (field_hessian) then
      if (allocated(hessian_values)) deallocate(hessian_values)
      allocate(hessian_values(3,3,3,npoints),stat=ierr)
      call stop_on_error(ierr,'allocate field Hessian output')
      call getQuickFieldHessian(npoints,probes_bohr,hessian_values,ierr)
      call stop_on_error(ierr,'on-the-fly field Hessian')
      field_hessian_calls=field_hessian_calls+1
      if (.not. all(ieee_is_finite(hessian_values))) call stop_on_error(45,'nonfinite field Hessian')
    endif
    oeprop_wall_seconds=oeprop_wall_seconds+wall_time_seconds()-begin_time
    call stop_on_error(ierr,'on-the-fly OEPROP')
    property_calls=property_calls+1
    delta=quick_api%step-before_step
    properties_extra_api_calls=properties_extra_api_calls+delta
    call stop_on_error(merge(0,53,delta == 0),'OEPROP changed API step')
    if (.not. all(ieee_is_finite(esp)) .or. .not. all(ieee_is_finite(efield)) .or. &
        .not. all(ieee_is_finite(efg))) call stop_on_error(45,'nonfinite OEPROP')
    call write_property_frame(oeprop_unit,frame_step,dt_fs,probe_mode,probes_ang, &
      reaction_coordinate,esp,efield,efg)
    if (field_hessian) call write_field_hessian_frame(frame_step)
  end subroutine sample_properties

  ! Ten independent entries of H_ijk=d2 E_i/dC_j dC_k=-d3 V/dC_i dC_j dC_k.
  ! Components stay in laboratory Cartesian axes, including moving probes.
  subroutine write_field_hessian_frame(frame_step)
    implicit none
    integer, intent(in) :: frame_step
    integer :: p
    if (.not. is_master) return
    do p=1,npoints
      write(field_hessian_unit,'(I0,",",ES24.16,",",I0,13(",",ES24.16))') &
        frame_step,dble(frame_step)*dt_fs,p,probes_ang(:,p), &
        hessian_values(1,1,1,p),hessian_values(1,1,2,p),hessian_values(1,1,3,p), &
        hessian_values(1,2,2,p),hessian_values(1,2,3,p),hessian_values(1,3,3,p), &
        hessian_values(2,2,2,p),hessian_values(2,2,3,p),hessian_values(2,3,3,p),hessian_values(3,3,3,p)
    enddo
  end subroutine write_field_hessian_frame

  subroutine write_kernel_counts(file,ierr)
    implicit none
    character(len=*), intent(in) :: file
    integer, intent(out) :: ierr
    integer :: unit,i,counts(8)
    integer, allocatable :: rank_counts(:,:)
    double precision :: times(3)
    double precision, allocatable :: rank_times(:,:)

    counts=(/force_calls,api_energy_gradient_calls,property_calls,probe_generation_calls, &
      properties_extra_api_calls,launch_deltas/)
    times=(/force_wall_seconds,probe_wall_seconds,oeprop_wall_seconds/)
    allocate(rank_counts(8,rank_count),rank_times(3,rank_count),stat=ierr)
    call stop_on_error(ierr,'allocate rank evidence')
#ifdef MPIV
    call MPI_GATHER(counts,8,MPI_INTEGER,rank_counts,8,MPI_INTEGER,0,quick_comm,quick_mpi_error)
    call stop_on_error(quick_mpi_error,'gather API and launch counts')
    call MPI_GATHER(times,3,MPI_DOUBLE_PRECISION,rank_times,3,MPI_DOUBLE_PRECISION,0,quick_comm,quick_mpi_error)
    call stop_on_error(quick_mpi_error,'gather wall times')
#else
    rank_counts(:,1)=counts
    rank_times(:,1)=times
#endif
    ierr=0
    if (is_master) then
      open(newunit=unit,file=file,status='replace',action='write',iostat=ierr)
      if (ierr == 0) then
        write(unit,'(A)') 'rank,force_calls,api_energy_gradient_calls,property_calls,probe_generation_calls,'// &
          'properties_extra_api_calls,esp_launches,efield_launches,efg_launches,'// &
          'force_wall_seconds,probe_wall_seconds,oeprop_wall_seconds'
        do i=1,rank_count
          write(unit,'(I0,8(",",I0),3(",",ES24.16))') i-1,rank_counts(:,i),rank_times(:,i)
        enddo
        close(unit)
      endif
    endif
#ifdef MPIV
    call MPI_BCAST(ierr,1,MPI_INTEGER,0,quick_comm,quick_mpi_error)
#endif
  end subroutine write_kernel_counts

  subroutine read_xyz(file,natom,z,coords,ierr)
    implicit none
    character(len=*), intent(in) :: file
    integer, intent(out) :: natom, ierr
    integer, allocatable, intent(out) :: z(:)
    double precision, allocatable, intent(out) :: coords(:,:)
    integer :: unit, i, ios
    character(len=256) :: line
    character(len=16) :: symbol

    ierr=0
    open(newunit=unit,file=file,status='old',action='read',iostat=ierr)
    if (ierr /= 0) return
    read(unit,*,iostat=ios) natom
    if (ios /= 0 .or. natom < 1) then
      ierr=49
      close(unit)
      return
    endif
    read(unit,'(A)',iostat=ios) line
    allocate(z(natom),coords(3,natom),stat=ierr)
    if (ierr /= 0) then
      close(unit)
      return
    endif
    do i=1,natom
      read(unit,*,iostat=ios) symbol,coords(:,i)
      if (ios /= 0) then
        ierr=49
        exit
      endif
      z(i)=symbol_to_z(symbol)
      if (z(i) < 1) then
        ierr=52
        exit
      endif
    enddo
    close(unit)
  end subroutine read_xyz

  subroutine read_probe_definition(mode,file,npoints,reference,probe_atoms,ierr)
    implicit none
    character(len=*), intent(in) :: mode,file
    integer, intent(out) :: npoints,ierr
    double precision, allocatable, intent(out) :: reference(:,:)
    integer, allocatable, intent(out) :: probe_atoms(:)
    integer :: unit, i, ios

    ierr=0
    npoints=0
    if (trim(mode) == 'density') return
    if (trim(mode) /= 'fixed' .and. trim(mode) /= 'molecule' .and. trim(mode) /= 'atom') then
      ierr=45
      return
    endif
    open(newunit=unit,file=file,status='old',action='read',iostat=ierr)
    if (ierr /= 0) return
    read(unit,*,iostat=ios) npoints
    if (ios /= 0 .or. npoints < 1) then
      ierr=49
      close(unit)
      return
    endif
    allocate(reference(3,npoints),probe_atoms(npoints),stat=ierr)
    if (ierr /= 0) then
      close(unit)
      return
    endif
    probe_atoms=0
    do i=1,npoints
      if (trim(mode) == 'atom') then
        read(unit,*,iostat=ios) probe_atoms(i),reference(:,i)
      else
        read(unit,*,iostat=ios) reference(:,i)
      endif
      if (ios /= 0) then
        ierr=49
        exit
      endif
    enddo
    close(unit)
  end subroutine read_probe_definition

  subroutine prepare_probes(mode,coords,anchors,isovalue,spacing,max_radius,reference,probe_atoms, &
      npoints,probes_ang,probes_bohr,ierr)
    implicit none
    character(len=*), intent(in) :: mode
    double precision, intent(in) :: coords(:,:),isovalue,spacing,max_radius
    integer, intent(in) :: anchors(3)
    double precision, allocatable, intent(in) :: reference(:,:)
    integer, allocatable, intent(in) :: probe_atoms(:)
    integer, intent(inout) :: npoints
    double precision, allocatable, intent(inout) :: probes_ang(:,:),probes_bohr(:,:)
    integer, intent(out) :: ierr
    integer :: i
    double precision :: e1(3),e2(3),e3(3),trial(3),norm_value

    ierr=0
    if (allocated(probes_ang)) deallocate(probes_ang)
    if (allocated(probes_bohr)) deallocate(probes_bohr)
    if (trim(mode) == 'density') then
      call getQuickDensitySurface(isovalue,spacing,max_radius,npoints,probes_bohr,ierr)
      if (ierr /= 0) return
      allocate(probes_ang(3,npoints),stat=ierr)
      if (ierr == 0) probes_ang=probes_bohr*BOHRS_TO_A
      return
    endif

    allocate(probes_ang(3,npoints),probes_bohr(3,npoints),stat=ierr)
    if (ierr /= 0) return
    select case(trim(mode))
    case('fixed')
      probes_ang=reference
    case('atom')
      do i=1,npoints
        if (probe_atoms(i) < 1 .or. probe_atoms(i) > size(coords,2)) then
          ierr=45
          return
        endif
        probes_ang(:,i)=coords(:,probe_atoms(i))+reference(:,i)
      enddo
    case('molecule')
      e1=coords(:,anchors(2))-coords(:,anchors(1))
      norm_value=sqrt(dot_product(e1,e1))
      if (norm_value < 1.0d-12) then
        ierr=45
        return
      endif
      e1=e1/norm_value
      trial=coords(:,anchors(3))-coords(:,anchors(1))
      e3=cross_product(e1,trial)
      norm_value=sqrt(dot_product(e3,e3))
      if (norm_value < 1.0d-12) then
        ierr=45
        return
      endif
      e3=e3/norm_value
      e2=cross_product(e3,e1)
      do i=1,npoints
        probes_ang(:,i)=coords(:,anchors(1))+reference(1,i)*e1+reference(2,i)*e2+reference(3,i)*e3
      enddo
    end select
    probes_bohr=probes_ang/BOHRS_TO_A
  end subroutine prepare_probes

  subroutine evaluate_properties(npoints,probes,esp,efield,efg,ierr)
    implicit none
    integer, intent(in) :: npoints
    double precision, intent(in) :: probes(3,npoints)
    double precision, allocatable, intent(inout) :: esp(:),efield(:,:),efg(:,:,:)
    integer, intent(out) :: ierr

    if (allocated(esp)) deallocate(esp,efield,efg)
    allocate(esp(npoints),efield(3,npoints),efg(3,3,npoints),stat=ierr)
    if (ierr /= 0) return
    call getQuickOEPROP(npoints,probes,ierr,esp=esp,efield=efield,efg=efg)
  end subroutine evaluate_properties

  subroutine assign_masses(z,masses,ierr)
    implicit none
    integer, intent(in) :: z(:)
    double precision, intent(out) :: masses(size(z))
    integer, intent(out) :: ierr
    integer :: i

    ierr=0
    do i=1,size(z)
      select case(z(i))
      case(1);  masses(i)=1.008d0
      case(6);  masses(i)=12.011d0
      case(7);  masses(i)=14.007d0
      case(8);  masses(i)=15.999d0
      case(9);  masses(i)=18.998403163d0
      case(17); masses(i)=35.45d0
      case default
        ierr=45
        return
      end select
    enddo
    masses=masses*amu_to_me
  end subroutine assign_masses

  subroutine initialize_velocities(z,masses,frozen,temperature,velocities)
    implicit none
    integer, intent(in) :: z(:)
    double precision, intent(in) :: masses(:),temperature
    logical, intent(in) :: frozen(:)
    double precision, intent(out) :: velocities(3,size(z))
    integer :: i,j,nmobile,dof_local
    double precision :: momentum(3),mobile_mass,current_kinetic,target_kinetic,scale

    velocities=0.0d0
    if (temperature == 0.0d0) return
    do j=1,size(z)
      if (frozen(j)) cycle
      do i=1,3
        velocities(i,j)=1.0d-4*dble(mod(7*i+11*j+3*z(j),17)-8)
      enddo
    enddo
    momentum=0.0d0
    mobile_mass=0.0d0
    do j=1,size(z)
      if (frozen(j)) cycle
      momentum=momentum+masses(j)*velocities(:,j)
      mobile_mass=mobile_mass+masses(j)
    enddo
    if (mobile_mass > 0.0d0) then
      do j=1,size(z)
        if (.not. frozen(j)) velocities(:,j)=velocities(:,j)-momentum/mobile_mass
      enddo
    endif
    nmobile=count(.not. frozen)
    dof_local=max(1,3*nmobile-3)
    current_kinetic=kinetic_energy(velocities,masses)
    target_kinetic=0.5d0*dble(dof_local)*kb_au*temperature
    if (current_kinetic > 0.0d0) then
      scale=sqrt(target_kinetic/current_kinetic)
      velocities=scale*velocities
    endif
  end subroutine initialize_velocities

  subroutine gradients_to_accelerations(gradients,masses,frozen,accelerations)
    implicit none
    double precision, intent(in) :: gradients(:,:),masses(:)
    logical, intent(in) :: frozen(:)
    double precision, intent(out) :: accelerations(size(gradients,1),size(gradients,2))

    accelerations=-gradients/spread(masses,1,3)
    where(spread(frozen,1,3)) accelerations=0.0d0
  end subroutine gradients_to_accelerations

  double precision function kinetic_energy(velocities,masses)
    implicit none
    double precision, intent(in) :: velocities(:,:),masses(:)
    kinetic_energy=0.5d0*sum(velocities*velocities*spread(masses,1,3))
  end function kinetic_energy

  subroutine summarize_gradient(gradients,max_gradient,net_gradient)
    implicit none
    double precision, intent(in) :: gradients(:,:)
    double precision, intent(out) :: max_gradient,net_gradient
    double precision :: net(3)

    max_gradient=maxval(abs(gradients))
    net=sum(gradients,dim=2)
    net_gradient=sqrt(dot_product(net,net))
  end subroutine summarize_gradient

  subroutine open_outputs(energy_file,xyz_file,oeprop_file,energy_unit,xyz_unit,oeprop_unit,ierr)
    implicit none
    character(len=*), intent(in) :: energy_file,xyz_file,oeprop_file
    integer, intent(out) :: energy_unit,xyz_unit,oeprop_unit,ierr

    ierr=0
    energy_unit=-1
    xyz_unit=-1
    oeprop_unit=-1
    if (is_master) then
      open(newunit=energy_unit,file=energy_file,status='replace',action='write',iostat=ierr)
      if (ierr == 0) open(newunit=xyz_unit,file=xyz_file,status='replace',action='write',iostat=ierr)
      if (ierr == 0) open(newunit=oeprop_unit,file=oeprop_file,status='replace',action='write',iostat=ierr)
      if (ierr == 0) then
        write(energy_unit,'(A)') 'step,time_fs,potential_au,kinetic_au,total_au,delta_total_au,'// &
          'temperature_K,max_gradient_au_per_bohr,net_gradient_au_per_bohr,max_gradient_change_au_per_bohr'
        write(oeprop_unit,'(A)') 'step,time_fs,probe,probe_mode,reaction_coordinate_ang,x_ang,y_ang,z_ang,esp,'// &
          'efield_x,efield_y,efield_z,efield_mag,efg_xx,efg_xy,efg_xz,efg_yx,efg_yy,efg_yz,'// &
          'efg_zx,efg_zy,efg_zz,efg_trace,efg_frobenius,efg_eigenvalue_1,efg_eigenvalue_2,'// &
          'efg_eigenvalue_3,efg_anisotropy'
      endif
    endif
#ifdef MPIV
    call MPI_BCAST(ierr,1,MPI_INTEGER,0,quick_comm,quick_mpi_error)
#endif
  end subroutine open_outputs

  subroutine write_dynamics_frame(energy_unit,xyz_unit,step,dt_fs,z,coords,potential, &
      kinetic,total,drift,temperature,max_gradient,net_gradient,force_change)
    implicit none
    integer, intent(in) :: energy_unit,xyz_unit,step,z(:)
    double precision, intent(in) :: dt_fs,coords(:,:),potential,kinetic,total,drift
    double precision, intent(in) :: temperature,max_gradient,net_gradient,force_change
    integer :: i
    double precision :: time_fs

    if (.not. is_master) return

    time_fs=dble(step)*dt_fs
    write(energy_unit,'(I0,9(",",ES24.16))') step,time_fs,potential,kinetic,total,drift,temperature, &
      max_gradient,net_gradient,force_change
    write(xyz_unit,'(I0)') size(z)
    write(xyz_unit,'("step=",I0,1X,"time_fs=",F12.6,1X,"total_au=",ES24.16)') step,time_fs,total
    do i=1,size(z)
      write(xyz_unit,'(A,3(1X,F20.12))') z_to_symbol(z(i)),coords(:,i)
    enddo
  end subroutine write_dynamics_frame

  subroutine write_property_frame(oeprop_unit,step,dt_fs,mode,probes,reaction_coordinate,esp,efield,efg)
    implicit none
    integer, intent(in) :: oeprop_unit,step
    character(len=*), intent(in) :: mode
    double precision, intent(in) :: dt_fs,probes(:,:),reaction_coordinate
    double precision, intent(in) :: esp(:),efield(:,:),efg(:,:,:)
    integer, parameter :: OUTPUT_BLOCK=128
    integer :: i,j,first,last,prefix_length
    double precision :: time_fs,field_mag,trace,frobenius,eigenvalues(3),anisotropy
    double precision :: rows(23,OUTPUT_BLOCK)
    character(len=64) :: frame_prefix
    character(len=len_trim(mode)+26) :: mode_coordinate

    if (.not. is_master) return
    time_fs=dble(step)*dt_fs
    write(frame_prefix,'(I0,",",ES24.16,",")') step,time_fs
    prefix_length=len_trim(frame_prefix)
    write(mode_coordinate,'(",",A,",",ES24.16)') trim(mode),reaction_coordinate

    ! Cache frame constants and amortize WRITE setup with bounded row blocks.
    ! The colon suppresses the final slash, leaving exactly one record per probe.
    do first=1,size(esp),OUTPUT_BLOCK
      last=min(size(esp),first+OUTPUT_BLOCK-1)
      do i=first,last
        j=i-first+1
        field_mag=sqrt(sum(efield(:,i)**2))
        trace=efg(1,1,i)+efg(2,2,i)+efg(3,3,i)
        frobenius=sqrt(sum(efg(:,:,i)**2))
        call symmetric_eigenvalues(efg(:,:,i),eigenvalues)
        anisotropy=sqrt(0.5d0*((eigenvalues(1)-eigenvalues(2))**2+ &
          (eigenvalues(2)-eigenvalues(3))**2+(eigenvalues(3)-eigenvalues(1))**2))
        rows(:,j)=(/probes(:,i),esp(i),efield(:,i),field_mag, &
          efg(1,1,i),efg(1,2,i),efg(1,3,i),efg(2,1,i),efg(2,2,i),efg(2,3,i), &
          efg(3,1,i),efg(3,2,i),efg(3,3,i),trace,frobenius,eigenvalues,anisotropy/)
      enddo
      write(oeprop_unit,'(*(A,I0,A,23(",",ES24.16),:/))') &
        (frame_prefix(:prefix_length),i,mode_coordinate,rows(:,i-first+1),i=first,last)
    enddo
  end subroutine write_property_frame

  subroutine symmetric_eigenvalues(matrix,eigenvalues)
    implicit none
    double precision, intent(in) :: matrix(3,3)
    double precision, intent(out) :: eigenvalues(3)
    double precision :: a(3,3),b(3,3),q,p2,p,r,phi,temp
    integer :: i,j

    a=0.5d0*(matrix+transpose(matrix))
    q=(a(1,1)+a(2,2)+a(3,3))/3.0d0
    p2=(a(1,1)-q)**2+(a(2,2)-q)**2+(a(3,3)-q)**2+ &
      2.0d0*(a(1,2)**2+a(1,3)**2+a(2,3)**2)
    p=sqrt(p2/6.0d0)
    if (p < 1.0d-15) then
      eigenvalues=q
    else
      b=a
      do i=1,3
        b(i,i)=b(i,i)-q
      enddo
      b=b/p
      r=0.5d0*(b(1,1)*(b(2,2)*b(3,3)-b(2,3)*b(3,2))- &
        b(1,2)*(b(2,1)*b(3,3)-b(2,3)*b(3,1))+b(1,3)*(b(2,1)*b(3,2)-b(2,2)*b(3,1)))
      phi=acos(max(-1.0d0,min(1.0d0,r)))/3.0d0
      eigenvalues(3)=q+2.0d0*p*cos(phi)
      eigenvalues(1)=q+2.0d0*p*cos(phi+2.0d0*pi/3.0d0)
      eigenvalues(2)=3.0d0*q-eigenvalues(1)-eigenvalues(3)
    endif
    do i=1,2
      do j=i+1,3
        if (eigenvalues(i) > eigenvalues(j)) then
          temp=eigenvalues(i)
          eigenvalues(i)=eigenvalues(j)
          eigenvalues(j)=temp
        endif
      enddo
    enddo
  end subroutine symmetric_eigenvalues

  subroutine write_checkpoint(file,natom,step,dt_fs,total0,z,coords,velocities,frozen,ierr)
    implicit none
    character(len=*), intent(in) :: file
    integer, intent(in) :: natom,step,z(natom)
    double precision, intent(in) :: dt_fs,total0,coords(3,natom),velocities(3,natom)
    logical, intent(in) :: frozen(natom)
    integer, intent(out) :: ierr
    integer :: unit,i

    ierr=0
    if (is_master) then
      open(newunit=unit,file=file,status='replace',action='write',iostat=ierr)
      if (ierr == 0) then
        write(unit,'(A)') 'QUICK_OEPROP_BOMD_CHECKPOINT_V2'
        write(unit,'(I0,1X,I0)') natom,step
        write(unit,'(ES24.16,1X,ES24.16)') dt_fs,total0
        write(unit,'(A)') trim(keywd)
        do i=1,natom
          write(unit,'(I0,1X,I0,6(1X,ES24.16))') z(i),merge(1,0,frozen(i)),coords(:,i),velocities(:,i)
        enddo
        close(unit)
      endif
    endif
#ifdef MPIV
    call MPI_BCAST(ierr,1,MPI_INTEGER,0,quick_comm,quick_mpi_error)
#endif
  end subroutine write_checkpoint

  subroutine read_checkpoint(file,natom,step,dt_fs,total0,z,coords,velocities,frozen,ierr)
    implicit none
    character(len=*), intent(in) :: file
    integer, intent(out) :: natom,step,ierr
    double precision, intent(out) :: dt_fs,total0
    integer, allocatable, intent(out) :: z(:)
    double precision, allocatable, intent(out) :: coords(:,:),velocities(:,:)
    logical, allocatable, intent(out) :: frozen(:)
    integer :: unit,i,ios,flag
    character(len=64) :: header
    character(len=256) :: checkpoint_keywd

    ierr=0
    open(newunit=unit,file=file,status='old',action='read',iostat=ierr)
    if (ierr /= 0) return
    read(unit,'(A)',iostat=ios) header
    if (ios /= 0 .or. (trim(header) /= 'QUICK_OEPROP_BOMD_CHECKPOINT_V1' .and. &
        trim(header) /= 'QUICK_OEPROP_BOMD_CHECKPOINT_V2')) then
      ierr=49
      close(unit)
      return
    endif
    read(unit,*,iostat=ios) natom,step
    if (ios /= 0 .or. natom < 1 .or. step < 0) then
      ierr=49
      close(unit)
      return
    endif
    read(unit,*,iostat=ios) dt_fs,total0
    if (ios /= 0) then
      ierr=49
      close(unit)
      return
    endif
    if (.not. ieee_is_finite(dt_fs) .or. .not. ieee_is_finite(total0) .or. dt_fs <= 0.0d0) then
      ierr=49
      close(unit)
      return
    endif
    restart_hamiltonian_verified=trim(header) == 'QUICK_OEPROP_BOMD_CHECKPOINT_V2'
    if (restart_hamiltonian_verified) then
      read(unit,'(A)',iostat=ios) checkpoint_keywd
      if (ios /= 0 .or. trim(checkpoint_keywd) /= trim(keywd)) then
        ierr=49
        close(unit)
        return
      endif
    else
      if (is_master) write(*,'(A)') 'QUICK_OEPROP_BOMD_RESTART_V1_UNVERIFIED_HAMILTONIAN'
    endif
    allocate(z(natom),coords(3,natom),velocities(3,natom),frozen(natom),stat=ierr)
    if (ierr /= 0) then
      close(unit)
      return
    endif
    do i=1,natom
      read(unit,*,iostat=ios) z(i),flag,coords(:,i),velocities(:,i)
      if (ios /= 0 .or. (flag /= 0 .and. flag /= 1)) then
        ierr=49
        exit
      endif
      if (z(i) < 1 .or. .not. all(ieee_is_finite(coords(:,i))) .or. &
          .not. all(ieee_is_finite(velocities(:,i)))) then
        ierr=49
        exit
      endif
      frozen(i)=flag == 1
    enddo
    close(unit)
  end subroutine read_checkpoint

  subroutine write_summary(file,final_step,steps_run,dt_fs,total0,total,max_drift,max_force_change, &
      max_net_gradient,mode,ierr)
    use iso_c_binding, only: c_int64_t
    use quick_method_module, only: quick_method
    use quick_solvation_module, only: solvation_updates, solvation_wall_seconds, solvation_timing_labels, &
      solvation_gpu_volume_density, solvation_gpu_volume_matrix, solvation_gpu_volume_gradient
    use quick_solvation_coulomb_module, only: solvation_coulomb_cache_info
    implicit none
    character(len=*), intent(in) :: file,mode
    integer, intent(in) :: final_step,steps_run
    double precision, intent(in) :: dt_fs,total0,total,max_drift,max_force_change,max_net_gradient
    integer, intent(out) :: ierr
    integer :: unit, phase
    integer(c_int64_t) :: cache_data(7)

    ierr=0
    if (is_master) then
      open(newunit=unit,file=file,status='replace',action='write',iostat=ierr)
      if (ierr == 0) then
        write(unit,'(A)') '{'
        write(unit,'(A,I0,A)') '  "final_step": ',final_step,','
        write(unit,'(A,I0,A)') '  "steps_run": ',steps_run,','
        write(unit,'(A,I0,A)') '  "start_step": ',start_step,','
        write(unit,'(A,I0,A)') '  "oeprop_stride": ',oeprop_stride,','
        write(unit,'(A,A,A)') '  "backend": "',trim(backend),'",'
        write(unit,'(A,I0,A)') '  "mpi_ranks": ',rank_count,','
        write(unit,'(A,I0,A)') '  "force_calls": ',force_calls,','
        write(unit,'(A,I0,A)') '  "api_energy_gradient_calls": ',api_energy_gradient_calls,','
        write(unit,'(A,I0,A)') '  "property_calls": ',property_calls,','
        write(unit,'(A,I0,A)') '  "probe_generation_calls": ',probe_generation_calls,','
        write(unit,'(A,I0,A)') '  "properties_extra_api_calls": ',properties_extra_api_calls,','
        if (field_hessian) then
          write(unit,'(A,I0,A)') '  "field_hessian_calls": ',field_hessian_calls,','
          write(unit,'(A,I0,A)') '  "field_hessian_master_gpu_launches": ', &
            hessian_launches_after-hessian_launches_before,','
          write(unit,'(A)') '  "field_hessian_convention": "H_ijk=d2 E_i/dC_j dC_k=-d3 V/dC_i dC_j dC_k",'
          write(unit,'(A)') '  "field_hessian_units": "hartree/(e bohr^3); laboratory Cartesian axes",'
        endif
        write(unit,'(A,ES24.16,A)') '  "force_wall_seconds": ',force_wall_seconds,','
        write(unit,'(A,ES24.16,A)') '  "probe_wall_seconds": ',probe_wall_seconds,','
        write(unit,'(A,ES24.16,A)') '  "oeprop_wall_seconds": ',oeprop_wall_seconds,','
        write(unit,'(A,A,A)') '  "restart_hamiltonian_verified": ', &
          trim(merge('true ','false',restart_hamiltonian_verified)),','
        write(unit,'(A)') '  "property_units": "atomic units; probe coordinates in Angstrom",'
        write(unit,'(A)') '  "efield_axes": "laboratory Cartesian",'
        write(unit,'(A)') '  "efg_axes": "laboratory Cartesian; G_ij = dE_i/dC_j = -d2V/dC_i dC_j",'
        write(unit,'(A)') '  "density_probe_ids": "local to each sampled frame; no temporal correspondence",'
        write(unit,'(A)') '  "timing_scope": "per rank wall time; force includes SCF plus gradient and API internal '// &
          'work/output; excludes driver trajectory/property IO",'
        write(unit,'(A,A,A)') '  "probe_mode": "',trim(mode),'",'
        write(unit,'(A,ES24.16,A)') '  "dt_fs": ',dt_fs,','
        write(unit,'(A,ES24.16,A)') '  "initial_total_au": ',total0,','
        write(unit,'(A,ES24.16,A)') '  "final_total_au": ',total,','
        write(unit,'(A,ES24.16,A)') '  "max_abs_drift_au": ',max_drift,','
        write(unit,'(A,ES24.16,A)') '  "max_gradient_change_au_per_bohr": ',max_force_change,','
        if (quick_method%external_efield) then
          write(unit,'(A,I0,A,I0,A)') '  "applied_field_master_gpu_calls": [',field_deltas(1),',',field_deltas(2),'],'
        endif
        if (quick_method%ddx) then
          call solvation_coulomb_cache_info(cache_data(1),cache_data(2),cache_data(3), &
            cache_data(4),cache_data(5),cache_data(6),cache_data(7))
          write(unit,'(A,I0,6(A,I0),A)') '  "solvation_gpu_coulomb_cache": [', &
            cache_data(1),',',cache_data(2),',',cache_data(3),',',cache_data(4),',',cache_data(5), &
            ',',cache_data(6),',',cache_data(7),'],'
          write(unit,'(A,I0,A)') '  "solvation_updates": ',solvation_updates,','
          write(unit,'(A,I0,A,I0,A,I0,A)') '  "solvation_gpu_volume_calls": [', &
            solvation_gpu_volume_density,',',solvation_gpu_volume_matrix,',',solvation_gpu_volume_gradient,'],'
          write(unit,'(A)') '  "solvation_master_phase_seconds": {'
          do phase=1,size(solvation_wall_seconds)
            write(unit,'(3A,ES24.16,A)') '    "',trim(solvation_timing_labels(phase)),'": ', &
              solvation_wall_seconds(phase),merge(',', ' ',phase < size(solvation_wall_seconds))
          enddo
          write(unit,'(A)') '  },'
        endif
        write(unit,'(A,ES24.16)') '  "max_net_gradient_au_per_bohr": ',max_net_gradient
        write(unit,'(A)') '}'
        close(unit)
      endif
    endif
#ifdef MPIV
    call MPI_BCAST(ierr,1,MPI_INTEGER,0,quick_comm,quick_mpi_error)
#endif
  end subroutine write_summary

  subroutine validate_atom_indices(natom,anchors,coordinate_atoms,ierr)
    implicit none
    integer, intent(in) :: natom,anchors(3),coordinate_atoms(2)
    integer, intent(out) :: ierr
    ierr=0
    if (any(anchors < 1) .or. any(anchors > natom) .or. any(coordinate_atoms < 1) .or. &
        any(coordinate_atoms > natom)) ierr=45
  end subroutine validate_atom_indices

  pure function cross_product(a,b) result(c)
    implicit none
    double precision, intent(in) :: a(3),b(3)
    double precision :: c(3)
    c=(/a(2)*b(3)-a(3)*b(2),a(3)*b(1)-a(1)*b(3),a(1)*b(2)-a(2)*b(1)/)
  end function cross_product

  pure double precision function distance(a,b)
    implicit none
    double precision, intent(in) :: a(3),b(3)
    distance=sqrt(sum((a-b)**2))
  end function distance

  integer function symbol_to_z(symbol)
    implicit none
    character(len=*), intent(in) :: symbol
    select case(trim(adjustl(symbol)))
    case('H','h'); symbol_to_z=1
    case('C','c'); symbol_to_z=6
    case('N','n'); symbol_to_z=7
    case('O','o'); symbol_to_z=8
    case('F','f'); symbol_to_z=9
    case('Cl','CL','cl'); symbol_to_z=17
    case default; symbol_to_z=-1
    end select
  end function symbol_to_z

  character(len=2) function z_to_symbol(z)
    implicit none
    integer, intent(in) :: z
    select case(z)
    case(1); z_to_symbol='H '
    case(6); z_to_symbol='C '
    case(7); z_to_symbol='N '
    case(8); z_to_symbol='O '
    case(9); z_to_symbol='F '
    case(17); z_to_symbol='Cl'
    case default; z_to_symbol='X '
    end select
  end function z_to_symbol

  function uppercase(input) result(output)
    implicit none
    character(len=*), intent(in) :: input
    character(len=len(input)) :: output
    integer :: i,code

    output=input
    do i=1,len(input)
      code=iachar(output(i:i))
      if (code >= iachar('a') .and. code <= iachar('z')) output(i:i)=achar(code-32)
    enddo
  end function uppercase

  subroutine stop_on_error(ierr_in,label)
    implicit none
    integer, intent(in) :: ierr_in
    character(len=*), intent(in) :: label
    if (ierr_in /= 0) then
      if (is_master) write(*,'("QUICK_OEPROP_BOMD_ERROR ",A,1X,I0)') trim(label),ierr_in
#ifdef MPIV
      call MPI_ABORT(quick_comm,ierr_in,quick_mpi_error)
#endif
      stop 1
    endif
  end subroutine stop_on_error

end program quick_oeprop_bomd_driver
