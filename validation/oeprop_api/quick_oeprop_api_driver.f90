#include "util.fh"
!---------------------------------------------------------------------!
! QUICK no-I/O OEPROP API validation and trajectory driver.            !
!                                                                     !
! This Source Code Form is subject to the terms of the Mozilla Public !
! License, v. 2.0.                                                    !
!---------------------------------------------------------------------!

program quick_oeprop_api_driver

  use quick_api_module, only: setQuickJob, getQuickEnergy, getQuickEnergyGradients, getQuickOEPROP, deleteQuickJob
  use quick_constants_module, only: BOHRS_TO_A
#ifdef MPIV
  use mpi
  use quick_api_module, only: setQuickMPI
  use quick_mpi_module, only: quick_comm, quick_comm_rank, quick_comm_size, quick_mpi_error
#endif

  implicit none

  character(len=32) :: mode
  character(len=256) :: arg1, arg2, arg3, arg4, arg5, arg6
  integer :: argc, ierr, test_group
  logical :: is_master
#ifdef MPIV
  integer :: input_comm, setup_world_rank, setup_world_size
#endif

  is_master = .true.
  ierr = 0
  test_group = 0
  argc = command_argument_count()
  mode = 'validate'
  if (argc >= 1) call get_command_argument(1, mode)
#ifdef MPIV
  call MPI_INIT(quick_mpi_error)
  input_comm = MPI_COMM_WORLD
  if (trim(mode) == 'mpi-subcommunicator') then
    call MPI_COMM_RANK(MPI_COMM_WORLD,setup_world_rank,quick_mpi_error)
    call MPI_COMM_SIZE(MPI_COMM_WORLD,setup_world_size,quick_mpi_error)
    if (setup_world_size /= 4) then
      if (setup_world_rank == 0) write(*,'(A)') 'mpi-subcommunicator requires exactly four ranks'
      call MPI_ABORT(MPI_COMM_WORLD,49,quick_mpi_error)
    endif
    test_group = mod(setup_world_rank,2)
    call MPI_COMM_SPLIT(MPI_COMM_WORLD,test_group,setup_world_rank,input_comm,quick_mpi_error)
  endif
  call setQuickMPI(input_comm,ierr)
  if (trim(mode) == 'mpi-subcommunicator') call MPI_COMM_FREE(input_comm,quick_mpi_error)
  is_master = (quick_comm_rank == 0)
#endif

  select case(trim(mode))
  case('parser-contract')
    arg1 = 'parser_contract.csv'
    if (argc >= 2) call get_command_argument(2,arg1)
    call run_parser_contract(trim(arg1),ierr)
  case('mpi-subcommunicator','subcommunicator-reference')
    arg1 = 'subcommunicator'
    if (argc >= 2) call get_command_argument(2,arg1)
    if (trim(mode) == 'subcommunicator-reference') then
      if (argc /= 3) then
        ierr = 49
      else
        call get_command_argument(3,arg2)
        read(arg2,*,iostat=ierr) test_group
      endif
    else
#ifndef MPIV
      ierr = 49
#endif
    endif
    if (ierr == 0) call run_subcommunicator_case(trim(arg1),test_group,ierr)
  case('validate')
    arg1 = 'oeprop_api_validation.csv'
    if (argc >= 2) call get_command_argument(2, arg1)
    call run_validation(trim(arg1), ierr)
  case('trajectory')
    if (argc < 4) then
      if (is_master) write(*,'("Usage: test-api-oeprop trajectory <trajectory.xyz> <probe.xyz> <output.csv> [keyword] [basename]")')
      ierr = 49
    else
      call get_command_argument(2, arg1)
      call get_command_argument(3, arg2)
      call get_command_argument(4, arg3)
      arg4 = 'HF BASIS=STO-3G CUTOFF=1.0D-10 DENSERMS=1.0D-8 ENERGY CHARGE=0'
      arg5 = 'oeprop_api_traj'
      if (argc >= 5) call get_command_argument(5, arg4)
      if (argc >= 6) call get_command_argument(6, arg5)
      call run_trajectory(trim(arg1), trim(arg2), trim(arg3), trim(arg4), trim(arg5), ierr)
    endif
  case('acceptance', 'acceptance-gradient', 'lifecycle-gradient', 'benchmark-case')
    if (argc < 5) then
      if (is_master) write(*,'(A)') &
        'Usage: test-api-oeprop acceptance|acceptance-gradient|lifecycle-gradient|benchmark-case <molecule.xyz> <probes.dat> <prefix> <keyword> [density.dat|-] [repeats]'
      ierr = 49
    else
      call get_command_argument(2, arg1)
      call get_command_argument(3, arg2)
      call get_command_argument(4, arg3)
      call get_command_argument(5, arg4)
      arg5 = '-'
      arg6 = '3'
      if (argc >= 6) call get_command_argument(6, arg5)
      if (argc >= 7) call get_command_argument(7, arg6)
      call run_acceptance(trim(mode), trim(arg1), trim(arg2), trim(arg3), trim(arg4), trim(arg5), trim(arg6), ierr)
    endif
  case('benchmark')
    arg1 = 'oeprop_grid_scaling.csv'
    if (argc >= 2) call get_command_argument(2, arg1)
    call run_benchmark(trim(arg1), ierr)
  case('optimizer-endpoint')
    arg1 = 'oeprop_optimizer_endpoint'
    arg2 = 'lopt'
    if (argc >= 2) call get_command_argument(2, arg1)
    if (argc >= 3) call get_command_argument(3, arg2)
    call run_optimizer_endpoint(trim(arg1), trim(arg2), ierr)
  case default
    if (is_master) write(*,'("Unknown mode: ",A)') trim(mode)
    ierr = 49
  end select

  if (ierr /= 0) then
    if (is_master) write(*,'("QUICK OEPROP API driver failed with ierr=",I0)') ierr
#ifdef MPIV
    call MPI_FINALIZE(quick_mpi_error)
#endif
    stop 1
  endif

#ifdef MPIV
  call MPI_FINALIZE(quick_mpi_error)
#endif

contains

  subroutine run_parser_contract(filename,ierr)
    use quick_method_module, only: quick_method_type, init, read
    implicit none
    character(len=*), intent(in) :: filename
    integer, intent(out) :: ierr
    integer, parameter :: ncases=18
    character(len=48), parameter :: labels(ncases)=[character(len=48) :: &
      'analytic_grid','numerical_implies_grid','reject_property_substrings', &
      'esp_surface','efield_surface','efg_surface','density_value_implies_mode', &
      'density_spacing_implies_mode','density_radius_implies_mode', &
      'reject_density_substring','external_field_origin','finite_field_alias', &
      'finite_alias_precedence','reject_field_substrings','cutoff_boundaries', &
      'rejected_then_valid_token','origin_only_enables_field','tab_boundaries']
    character(len=300), parameter :: cards(ncases)=[character(len=300) :: &
      'HF BASIS=STO-3G EFG_GRID', &
      'HF BASIS=STO-3G EFG_GRID_NUMERICAL', &
      'HF BASIS=STO-3G XEFG_GRID EFG_GRID_NUMERICAL_SUFFIX ESP_SURFACE_SUFFIX', &
      'HF BASIS=STO-3G ESP_SURFACE', &
      'HF BASIS=STO-3G EFIELD_SURFACE', &
      'HF BASIS=STO-3G EFG_SURFACE', &
      'HF BASIS=STO-3G ESP_GRID DENSITY_SURFACE_VALUE=2.0D-3', &
      'HF BASIS=STO-3G EFIELD_GRID DENSITY_SURFACE_SPACING=0.9', &
      'HF BASIS=STO-3G EFG_GRID DENSITY_SURFACE_MAX_RADIUS=7.0', &
      'HF BASIS=STO-3G ESP_GRID XDENSITY_SURFACE_VALUE=2.0D-3', &
      'HF BASIS=STO-3G EXTERNAL_EFIELD_X=1.0D-4 EXTERNAL_EFIELD_ORIGIN_Y=2.0', &
      'HF BASIS=STO-3G FINITE_FIELD_Z=-2.0D-4 FINITE_FIELD_ORIGIN_X=-1.0', &
      'HF BASIS=STO-3G EXTERNAL_EFIELD_X=1.0D-4 FINITE_FIELD_X=3.0D-4', &
      'HF BASIS=STO-3G XFINITE_FIELD_X=1.0D-4 EXTERNAL_EFIELD_X_SUFFIX=2.0', &
      'HF BASIS=STO-3G GRADCUTOFF=1.0D-12', &
      'HF BASIS=STO-3G XEFG_GRID EFG_GRID_NUMERICAL', &
      'HF BASIS=STO-3G EXTERNAL_EFIELD_ORIGIN_Z=0.5', &
      'HF'//achar(9)//'BASIS=STO-3G'//achar(9)//'EFG_GRID_NUMERICAL']
    type(quick_method_type) :: method
    character(len=300) :: keyword
    logical :: flags(8),expected(8),passed
    double precision :: field(3),origin(3),surface(3),expected_gradcutoff
    integer :: i,unit,failures,local_error

    ierr=0
    failures=0
    unit=-1
    if (is_master) then
      open(newunit=unit,file=filename,status='replace',iostat=ierr)
      if (ierr == 0) write(unit,'(A)') 'case,passed,esp,efield,efg,numerical,density,external,extgrid,angstrom,'// &
        'field_x,field_y,field_z,origin_x,origin_y,origin_z,isovalue,spacing,max_radius,cutoff,gradcutoff'
    endif
    call sync_status(ierr)
    if (ierr /= 0) return
    do i=1,ncases
      local_error=0
      call init(method,local_error)
      keyword=cards(i)
      call read(method,keyword,local_error)
      expected=.false.
      field=0.0d0
      origin=0.0d0
      surface=(/1.0d-3,0.50d0,10.0d0/)
      expected_gradcutoff=1.0d-7
      select case(i)
      case(1)
        expected(3)=.true.; expected(7)=.true.
      case(2,16,18)
        expected(3:4)=.true.; expected(7)=.true.
      case(4:6)
        expected(i-3)=.true.; expected(5)=.true.; expected(8)=.true.
      case(7:9)
        expected(i-6)=.true.; expected(5)=.true.; expected(8)=.true.
        if (i == 7) surface(1)=2.0d-3
        if (i == 8) surface(2)=0.9d0
        if (i == 9) surface(3)=7.0d0
      case(10)
        expected(1)=.true.; expected(7)=.true.
      case(11)
        expected(6)=.true.; field(1)=1.0d-4; origin(2)=2.0d0
      case(12)
        expected(6)=.true.; field(3)=-2.0d-4; origin(1)=-1.0d0
      case(13)
        expected(6)=.true.; field(1)=3.0d-4
      case(15)
        expected_gradcutoff=1.0d-12
      case(17)
        expected(6)=.true.; origin(3)=0.5d0
      end select
      flags=(/method%esp_grid,method%efield_grid,method%efg_grid,method%efg_grid_numerical, &
        method%density_surface,method%external_efield,method%ext_grid,method%extgrid_angstrom/)
      passed=local_error == 0 .and. all(flags .eqv. expected) &
        .and. maxval(abs(method%external_efield_vector-field)) <= 1.0d-15 &
        .and. maxval(abs(method%external_efield_origin-origin)) <= 1.0d-15 &
        .and. maxval(abs((/method%density_surface_value,method%density_surface_spacing, &
          method%density_surface_max_radius/)-surface)) <= 1.0d-15 &
        .and. abs(method%integralCutoff-1.0d-10) <= 1.0d-20 &
        .and. abs(method%gradCutoff-expected_gradcutoff) <= 1.0d-20
      if (.not. passed) failures=failures+1
      if (is_master) write(unit,'(A,",",L1,8(",",L1),11(",",ES24.16))') trim(labels(i)),passed,flags, &
        method%external_efield_vector,method%external_efield_origin,method%density_surface_value, &
        method%density_surface_spacing,method%density_surface_max_radius,method%integralCutoff,method%gradCutoff
    enddo
    if (is_master) close(unit)
    if (failures /= 0) ierr=53
    call sync_status(ierr)
  end subroutine run_parser_contract


  subroutine run_subcommunicator_case(prefix,group,ierr)
    use ieee_arithmetic, only: ieee_is_finite
    use quick_api_module, only: quick_api, getQuickDensitySurface
    use quick_method_module, only: quick_method
    implicit none
    character(len=*), intent(in) :: prefix
    integer, intent(in) :: group
    integer, intent(out) :: ierr
    integer :: natoms,npoints,api_step,after_step,unit,j,nsurface,cleanup,repeat
    integer, allocatable :: atomic_numbers(:)
    character(len=80) :: job
    character(len=256) :: case_prefix,keyword
    double precision, allocatable :: coords(:,:),points(:,:),v(:),e(:,:),g(:,:,:)
    double precision, allocatable :: vr(:),er(:,:),gr(:,:,:),surface_points(:,:),sv(:),se(:,:),sg(:,:,:)
    double precision :: energy,ptchg(4,0)
    double precision, allocatable :: values(:),smallest(:),largest(:)
    logical :: converged

    ierr=0
    if (group < 0 .or. group > 1) then
      ierr=49
      return
    endif
    write(case_prefix,'(A,".group",I0)') trim(prefix),group
    write(job,'("oeprop_split_group",I0)') group
    natoms=merge(3,1,group == 0)
    npoints=merge(4,3,group == 0)
    allocate(atomic_numbers(natoms),coords(3,natoms),points(3,npoints),v(npoints),e(3,npoints),g(3,3,npoints))
    allocate(vr(npoints),er(3,npoints),gr(3,3,npoints),values(13*npoints),smallest(13*npoints),largest(13*npoints))
    if (group == 0) then
      atomic_numbers=(/8,1,1/)
      coords(:,1)=(/0.0d0,0.0d0,0.0d0/)
      coords(:,2)=(/0.0d0,0.0d0,0.96d0/)
      coords(:,3)=(/0.90d0,0.0d0,-0.24d0/)
      keyword='HF BASIS=STO-3G CUTOFF=1.0D-12 DENSERMS=1.0D-9 ENERGY CHARGE=0'
    else
      atomic_numbers=2
      coords(:,1)=(/0.2d0,-0.1d0,0.3d0/)
      keyword='HF BASIS=STO-3G CUTOFF=1.0D-12 DENSERMS=1.0D-9 ENERGY CHARGE=0 FINITE_FIELD_X=1.0D-4'
    endif
    do j=1,npoints
      points(:,j)=(/2.0d0+0.3d0*j,-1.2d0+0.2d0*j,1.6d0+0.1d0*j/)/BOHRS_TO_A
    enddo
    call setQuickJob(job,keyword,natoms,atomic_numbers,.true.,ierr)
    call sync_status(ierr)
    if (ierr /= 0) return
    call getQuickOEPROP(npoints,points,ierr,esp=v,efield=e,efg=g)
    if (ierr == 47) then
      ierr=0
    else
      ierr=53
    endif
    call sync_status(ierr)
    if (ierr /= 0) return
    call getQuickEnergy(coords,0,ptchg,energy,ierr)
    converged=quick_method%scf_conv
    if (.not. converged .or. .not. ieee_is_finite(energy)) ierr=53
    call sync_status(ierr)
    if (ierr /= 0) return
    api_step=quick_api%step
    call getQuickOEPROP(npoints,points,ierr,esp=v,efield=e,efg=g)
    call sync_status(ierr)
    if (ierr /= 0) return
    if (.not. all(ieee_is_finite(v)) .or. .not. all(ieee_is_finite(e)) .or. &
        .not. all(ieee_is_finite(g))) ierr=53
    call sync_status(ierr)
    if (ierr /= 0) return
    do repeat=1,group+1
      call getQuickOEPROP(npoints,points,ierr,efield=er)
      call sync_status(ierr)
      if (ierr /= 0) return
      if (.not. all(ieee_is_finite(er))) ierr=53
      if (maxval(abs(er-e)) > 1.0d-12) ierr=53
      call sync_status(ierr)
      if (ierr /= 0) return
    enddo
    values=(/v,reshape(e,(/3*npoints/)),reshape(g,(/9*npoints/))/)
#ifdef MPIV
    call MPI_ALLREDUCE(values,smallest,size(values),MPI_DOUBLE_PRECISION,MPI_MIN,quick_comm,quick_mpi_error)
    call MPI_ALLREDUCE(values,largest,size(values),MPI_DOUBLE_PRECISION,MPI_MAX,quick_comm,quick_mpi_error)
#else
    smallest=values
    largest=values
#endif
    if (maxval(abs(largest-smallest)) /= 0.0d0) ierr=53
    call sync_status(ierr)
    if (ierr /= 0) return
    if (is_master) call write_property_csv(trim(case_prefix)//'.properties.csv',1,npoints,energy, &
      points*BOHRS_TO_A,v,e,g,ierr)
    call sync_status(ierr)
    if (ierr /= 0) return
    call getQuickDensitySurface(1.0d-3*(group+1),1.0d0,5.0d0,nsurface,surface_points,ierr)
    call sync_status(ierr)
    if (ierr /= 0) return
    if (nsurface < 1 .or. .not. all(ieee_is_finite(surface_points))) ierr=53
    call sync_status(ierr)
    if (ierr /= 0) return
    allocate(sv(nsurface),se(3,nsurface),sg(3,3,nsurface))
    call getQuickOEPROP(nsurface,surface_points,ierr,esp=sv,efield=se,efg=sg)
    call sync_status(ierr)
    if (ierr /= 0) return
    if (.not. all(ieee_is_finite(sv)) .or. .not. all(ieee_is_finite(se)) .or. &
        .not. all(ieee_is_finite(sg))) ierr=53
    after_step=quick_api%step
    if (after_step /= api_step) ierr=53
    call sync_status(ierr)
    if (ierr /= 0) return
    if (is_master) call write_property_csv(trim(case_prefix)//'.surface.csv',1,nsurface,energy, &
      surface_points*BOHRS_TO_A,sv,se,sg,ierr)
    call sync_status(ierr)
    if (ierr /= 0) return
    unit=-1
    if (is_master) then
      open(newunit=unit,file=trim(case_prefix)//'.state.csv',status='replace',iostat=ierr)
      if (ierr == 0) then
        write(unit,'(A)') 'group,ranks,api_step_before,api_step_after,scf_converged,surface_points'
#ifdef MPIV
        write(unit,'(4(I0,","),L1,",",I0)') group,quick_comm_size,api_step,after_step,converged,nsurface
#else
        write(unit,'(4(I0,","),L1,",",I0)') group,1,api_step,after_step,converged,nsurface
#endif
        close(unit)
      endif
    endif
    call sync_status(ierr)
    if (ierr /= 0) return
    call deleteQuickJob(ierr)
    call sync_status(ierr)
    if (ierr /= 0) return
    call getQuickOEPROP(npoints,points,cleanup,esp=vr,efield=er,efg=gr)
    if (cleanup /= 47) ierr=53
    call sync_status(ierr)
  end subroutine run_subcommunicator_case


  subroutine run_validation(csv_file, ierr)
    implicit none
    character(len=*), intent(in) :: csv_file
    integer, intent(out) :: ierr

    integer, parameter :: natoms = 3
    integer, parameter :: npoints = 4
    integer :: atomic_numbers(natoms)
    integer :: i
    double precision :: coords_ang(3,natoms)
    double precision :: probes_ang(3,npoints)
    double precision :: probes_bohr(3,npoints)
    double precision :: ptchg(4,0)
    double precision :: energy
    double precision :: esp(npoints), efield(3,npoints), efg(3,3,npoints)
    character(len=256) :: keywd
    character(len=80) :: job_name

    ierr = 0
    atomic_numbers = (/8, 1, 1/)
    coords_ang(:,1) = (/-0.33840d0,  0.00380d0,  0.23923d0/)
    coords_ang(:,2) = (/-0.33510d0, -0.00190d0, -0.83277d0/)
    coords_ang(:,3) = (/ 0.67350d0, -0.00190d0,  0.59353d0/)

    probes_ang(:,1) = (/ 0.00000d0,  0.00000d0,  2.00000d0/)
    probes_ang(:,2) = (/ 1.50000d0,  0.00000d0,  0.00000d0/)
    probes_ang(:,3) = (/ 0.00000d0,  1.50000d0,  0.00000d0/)
    probes_ang(:,4) = (/-1.00000d0, -1.00000d0,  0.50000d0/)
    probes_bohr = probes_ang / BOHRS_TO_A

    keywd = 'HF BASIS=STO-3G CUTOFF=1.0D-10 DENSERMS=1.0D-8 ENERGY CHARGE=0'
    job_name = 'oeprop_api_validation'
    call setQuickJob(job_name, keywd, natoms, atomic_numbers, .true., ierr)
    if (ierr /= 0) return

    call getQuickOEPROP(npoints, probes_bohr, ierr, esp=esp, efield=efield, efg=efg)
    if (ierr /= 47) then
      write(*,'("Expected pre-density getQuickOEPROP ierr=47, got ",I0)') ierr
      if (ierr == 0) ierr = 47
      call deleteQuickJob(i)
      return
    endif
    ierr = 0

    call getQuickEnergy(coords_ang, 0, ptchg, energy, ierr)
    if (ierr /= 0) then
      call deleteQuickJob(i)
      return
    endif

    call getQuickOEPROP(npoints, probes_bohr, ierr, esp=esp, efield=efield, efg=efg)
    if (ierr /= 0) then
      call deleteQuickJob(i)
      return
    endif

    if (is_master) call write_property_csv(csv_file, 1, npoints, energy, probes_ang, esp, efield, efg, ierr)
#ifdef MPIV
    call MPI_BCAST(ierr,1,MPI_INTEGER,0,quick_comm,quick_mpi_error)
#endif
    call deleteQuickJob(i)
    if (ierr == 0 .and. i /= 0) ierr = i
  end subroutine run_validation

  subroutine run_optimizer_endpoint(prefix, optimizer, ierr)
    implicit none
    character(len=*), intent(in) :: prefix, optimizer
    integer, intent(out) :: ierr
    integer :: unit, icase, ncases, imethod, nmethods
    character(len=16), parameter :: labels(3) = (/'maxiter         ', 'converged       ', 'maxiter-one     '/)
    character(len=6) :: methods(2)
    character(len=256) :: method_prefix

    ierr = 0
    unit = -1
    if (optimizer /= 'lopt' .and. optimizer /= 'dlfind' .and. optimizer /= 'both') then
      if (is_master) write(*,'(A)') 'Usage: test-api-oeprop optimizer-endpoint [prefix] [lopt|dlfind|both]'
      ierr = 49
      return
    endif
    methods = (/'lopt  ', 'dlfind'/)
    nmethods = 1
    if (optimizer == 'dlfind') methods(1) = 'dlfind'
    if (optimizer == 'both') nmethods = 2
    if (is_master) then
      open(newunit=unit,file=prefix // '.optimizer.csv',status='replace',iostat=ierr)
      if (ierr == 0) write(unit,'(A)') &
        'optimizer,exit,optimization_cycles,geometry_input_max_abs_ang,energy_abs,density_max_abs,esp_max_abs,efield_max_abs,efg_max_abs,repeat_max_abs,integral_cutoff,primitive_cutoff,initial_integral_cutoff,initial_primitive_cutoff,api_steps_opt,api_steps_fresh,properties_preserve_step,post_delete_ierr'
    endif
    call sync_status(ierr)
    if (ierr /= 0) return
    do imethod=1,nmethods
      ncases = 2
      if (methods(imethod) == 'dlfind') ncases = 3
      method_prefix = prefix
      if (optimizer == 'both') method_prefix = prefix // '.' // trim(methods(imethod))
      do icase=1,ncases
        call check_optimizer_endpoint(trim(labels(icase)),trim(methods(imethod)),trim(method_prefix),unit,ierr)
        if (ierr /= 0) exit
      enddo
      if (ierr /= 0) exit
    enddo
    if (is_master) close(unit)
  end subroutine run_optimizer_endpoint

  subroutine check_optimizer_endpoint(label, optimizer, prefix, unit, ierr)
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use quick_api_module, only: quick_api
    use quick_calculated_module, only: quick_qm_struct
    use quick_method_module, only: quick_method
    use quick_molspec_module, only: quick_molspec
    use dlf_constants, only: dlf_constants_get
    implicit none
    character(len=*), intent(in) :: label, optimizer, prefix
    integer, intent(in) :: unit
    integer, intent(out) :: ierr
    integer, parameter :: natoms=3, npoints=4
    integer :: numbers(natoms), cleanup, before_step, opt_steps, fresh_steps, cycles, i, coord_unit
    integer :: launches_before(3), launches_after(3), gpu_ready_before
    double precision :: coords(3,natoms), returned(3,natoms), logged(3,natoms), points(3,npoints), ptchg(4,0)
    double precision :: opt_energy, fresh_energy, v(npoints), vr(npoints), e(3,npoints), er(3,npoints)
    double precision :: g(3,3,npoints), gr(3,3,npoints), errors(6), repeat_error, screening(3), initial_screening(3), print_factor
    double precision, allocatable :: density(:,:)
    logical :: job_active, log_converged
    character(len=256) :: keyword, case_prefix
    character(len=80) :: job

    ierr = 0
    job_active = .false.
    numbers = (/8,1,1/)
    coords(:,1) = (/-0.741530d0, 1.752130d0, 2.896280d0/)
    coords(:,2) = (/-1.111151d0, 0.979769d0, 3.352290d0/)
    coords(:,3) = (/-0.920500d0, 2.036450d0, 1.984040d0/)
    points(:,1) = (/-0.7d0, 1.8d0, 5.0d0/) / BOHRS_TO_A
    points(:,2) = (/ 2.0d0, 1.8d0, 2.9d0/) / BOHRS_TO_A
    points(:,3) = (/-0.7d0,-1.0d0, 2.9d0/) / BOHRS_TO_A
    points(:,4) = (/-0.7d0, 1.8d0, 0.0d0/) / BOHRS_TO_A
    keyword = 'HF BASIS=cc-pVDZ CUTOFF=1.0D-9 DENSERMS=1.0D-10 GRADCUTOFF=1.0D-9 OPTIMIZE=5 GTOL=1.0D-12 ETOL=1.0D-12'
    if (optimizer == 'lopt') keyword = trim(keyword) // ' LOPT'
    job = 'oeprop_endpoint_' // optimizer // '_' // label
    case_prefix = prefix // '.' // label
    call setQuickJob(job,keyword,natoms,numbers,.true.,ierr)
    job_active = (ierr == 0)
    call sync_status(ierr)
    if (ierr /= 0) goto 900
    if (quick_method%usedlfind .neqv. (optimizer == 'dlfind')) ierr = 53
    call sync_status(ierr)
    if (ierr /= 0) goto 900
    if (label == 'maxiter-one') quick_method%iopt = 1
    if (label == 'converged' .and. optimizer == 'lopt') then
      ! Keep the first gradient test tight, then force a later done branch.
      ! This checks the exit contract, rather than locating a physical minimum.
      quick_method%geoMaxCrt = 1.d2
      quick_method%EChange = 1.d2
    endif
    if (label == 'converged' .and. optimizer == 'dlfind') then
      ! Loose thresholds exercise the DL-FIND converged initial evaluation.
      quick_method%gradMaxCrt = 1.d2
      quick_method%EChange = 1.d2
    endif
    initial_screening = (/quick_method%integralCutoff, quick_method%primLimit, quick_method%maxIntegralCutoff/)
    before_step = quick_api%step
    call getQuickEnergy(coords,0,ptchg,opt_energy,ierr)
    opt_steps = quick_api%step - before_step
    call sync_status(ierr)
    if (ierr /= 0) goto 900
    if (.not. quick_api%density_ready .or. opt_steps /= 1) ierr = 53
    if (.not. ieee_is_finite(opt_energy) .or. .not. all(ieee_is_finite(quick_qm_struct%dense))) ierr = 53
    returned = quick_molspec%xyz
    if (.not. all(ieee_is_finite(returned))) ierr = 53
    density = quick_qm_struct%dense
    screening = (/quick_method%integralCutoff, quick_method%primLimit, quick_method%maxIntegralCutoff/)
    print_factor = 0.529177249d0
    if (optimizer == 'dlfind') call dlf_constants_get('ANG_AU',print_factor)
    launches_before = 0
    launches_after = 0
    gpu_ready_before = -1
#if defined(GPU) || defined(MPIV_GPU)
    gpu_ready_before = merge(1,0,quick_api%gpu_molspecs_ready)
    if (gpu_ready_before /= 0) ierr = 53
    call gpu_oeprop_launch_counts(launches_before(1),launches_before(2),launches_before(3))
#endif
    before_step = quick_api%step
    call sync_status(ierr)
    if (ierr /= 0) goto 900
    call getQuickOEPROP(npoints,points,ierr,esp=v,efield=e,efg=g)
    call sync_status(ierr)
    if (ierr /= 0) goto 900
#if defined(GPU) || defined(MPIV_GPU)
    if (.not. quick_api%gpu_molspecs_ready) ierr = 53
#endif
    if (quick_api%step /= before_step) ierr = 53
    if (.not. all(ieee_is_finite(v)) .or. .not. all(ieee_is_finite(e)) .or. .not. all(ieee_is_finite(g))) ierr = 53
    call sync_status(ierr)
    if (ierr /= 0) goto 900
    call getQuickOEPROP(npoints,points,ierr,esp=vr,efield=er,efg=gr)
    call sync_status(ierr)
    if (ierr /= 0) goto 900
    if (.not. all(ieee_is_finite(vr)) .or. .not. all(ieee_is_finite(er)) .or. .not. all(ieee_is_finite(gr))) ierr = 53
    repeat_error = max(maxval(abs(v-vr)),maxval(abs(e-er)),maxval(abs(g-gr)))
    if (.not. ieee_is_finite(repeat_error) .or. repeat_error > 1.d-12) ierr = 53
    if (quick_api%step /= before_step) ierr = 53
    if (is_master .and. ierr == 0) then
      call write_property_csv(trim(case_prefix)//'.opt.properties.csv',1,npoints,opt_energy,points*BOHRS_TO_A,v,e,g,ierr)
      open(newunit=coord_unit,file=trim(case_prefix)//'.coordinates.csv',status='replace',iostat=cleanup)
      if (cleanup /= 0) then
        ierr = 49
      else
        write(coord_unit,'(A)') 'atom,x_bohr,y_bohr,z_bohr'
        do i=1,natoms
          write(coord_unit,'(I0,3(",",ES24.16))') i,returned(:,i)
        enddo
        close(coord_unit)
      endif
    endif
    call sync_status(ierr)
    if (ierr /= 0) goto 900
    call deleteQuickJob(cleanup)
    job_active = .false.
    if (cleanup /= 0) ierr = cleanup
    call getQuickOEPROP(npoints,points,cleanup,esp=vr,efield=er,efg=gr)
    if (cleanup /= 47) ierr = 53
    if (is_master) call read_last_opt_geometry(trim(job)//'.out',natoms,logged,cycles,log_converged,ierr)
    call sync_status(ierr)
    if (ierr /= 0) goto 900
#ifdef MPIV
    call MPI_BCAST(logged,3*natoms,MPI_DOUBLE_PRECISION,0,quick_comm,quick_mpi_error)
    call MPI_BCAST(cycles,1,MPI_INTEGER,0,quick_comm,quick_mpi_error)
    call MPI_BCAST(log_converged,1,MPI_LOGICAL,0,quick_comm,quick_mpi_error)
#endif
    ! Match the optimizer Angstrom conversion and six-decimal log precision.
    if (.not. all(ieee_is_finite(logged))) ierr = 53
    errors(1) = maxval(abs(returned*print_factor-logged))
    if (errors(1) > 5.1d-7) ierr = 53
    if (label == 'maxiter' .and. (cycles /= 5 .or. log_converged)) ierr = 53
    if (label == 'maxiter-one' .and. (cycles /= 1 .or. log_converged)) ierr = 53
    if (label == 'converged') then
      if (.not. log_converged) ierr = 53
      if (optimizer == 'lopt' .and. cycles /= 2) ierr = 53
      if (optimizer == 'dlfind' .and. cycles /= 1) ierr = 53
    endif
    call sync_status(ierr)
    if (ierr /= 0) goto 900
    job = 'oeprop_endpoint_fresh_' // optimizer // '_' // label
    keyword = 'HF BASIS=cc-pVDZ CUTOFF=1.0D-9 DENSERMS=1.0D-10 GRADCUTOFF=1.0D-9 ENERGY'
    ! Switch from an inline OPT keyword to a template ENERGY job. This also
    ! checks that creating a job resets keyword selection and first-step state.
    if (is_master) then
      open(newunit=coord_unit,file=trim(job)//'.in',status='replace',iostat=ierr)
      if (ierr == 0) then
        write(coord_unit,'(A)') trim(keyword)
        write(coord_unit,'(A)') ''
        close(coord_unit)
      endif
    endif
    call sync_status(ierr)
    if (ierr /= 0) goto 900
    keyword = ''
    call setQuickJob(job,keyword,natoms,numbers,.true.,ierr)
    job_active = (ierr == 0)
    call sync_status(ierr)
    if (ierr /= 0) goto 900
    if (quick_method%opt .or. .not. quick_api%firstStep .or. quick_api%step /= 1) ierr = 53
    call sync_status(ierr)
    if (ierr /= 0) goto 900
    ! Match initial SCF screening as well as its final adaptive thresholds;
    ! incremental Fock construction can retain terms from early SCF cycles.
    quick_method%integralCutoff = initial_screening(1)
    quick_method%primLimit = initial_screening(2)
    quick_method%maxIntegralCutoff = initial_screening(3)
    before_step = quick_api%step
    call getQuickEnergy(returned*BOHRS_TO_A,0,ptchg,fresh_energy,ierr)
    fresh_steps = quick_api%step - before_step
    call sync_status(ierr)
    if (ierr /= 0) goto 900
    if (quick_method%integralCutoff /= screening(1) .or. quick_method%primLimit /= screening(2) &
        .or. quick_method%maxIntegralCutoff /= screening(3)) ierr = 53
    if (fresh_steps /= 1 .or. maxval(abs(quick_molspec%xyz-returned)) > 1.d-12) ierr = 53
    if (.not. ieee_is_finite(fresh_energy) .or. .not. all(ieee_is_finite(quick_qm_struct%dense))) ierr = 53
    errors(2) = abs(opt_energy-fresh_energy)
    errors(3) = maxval(abs(density-quick_qm_struct%dense))
    before_step = quick_api%step
    call sync_status(ierr)
    if (ierr /= 0) goto 900
    call getQuickOEPROP(npoints,points,ierr,esp=vr,efield=er,efg=gr)
    call sync_status(ierr)
    if (ierr /= 0) goto 900
    if (quick_api%step /= before_step) ierr = 53
    if (.not. all(ieee_is_finite(vr)) .or. .not. all(ieee_is_finite(er)) .or. .not. all(ieee_is_finite(gr))) ierr = 53
    errors(4:6) = (/maxval(abs(v-vr)),maxval(abs(e-er)),maxval(abs(g-gr))/)
    if (any(errors(2:6) > 2.d-7)) then
      if (is_master) write(*,'(A,1X,A,1X,A,5(1X,ES14.6))') &
        'Endpoint energy/density/ESP/EFIELD/EFG errors:',optimizer,label,errors(2:6)
      ierr = 53
    endif
#if defined(GPU) || defined(MPIV_GPU)
    call gpu_oeprop_launch_counts(launches_after(1),launches_after(2),launches_after(3))
    if (any(launches_after-launches_before /= 3)) ierr = 53
#endif
    call sync_status(ierr)
    if (ierr /= 0) goto 900
    if (is_master) call write_property_csv(trim(case_prefix)//'.fresh.properties.csv',1,npoints,fresh_energy,points*BOHRS_TO_A,vr,er,gr,ierr)
    call write_kernel_counts(trim(case_prefix)//'.kernels.csv',launches_after-launches_before,ierr)
    call sync_status(ierr)
900 if (job_active) then
      call deleteQuickJob(cleanup)
      if (ierr == 0) ierr = cleanup
    endif
    call sync_status(ierr)
    if (ierr /= 0) return
    call getQuickOEPROP(npoints,points,cleanup,esp=vr,efield=er,efg=gr)
    if (cleanup /= 47) ierr = 53
    if (is_master .and. ierr == 0) write(unit,'(A,",",A,",",I0,11(",",ES24.16),4(",",I0))') &
      optimizer,label,cycles,errors,repeat_error,screening(1:2),initial_screening(1:2),opt_steps,fresh_steps,1,cleanup
    call sync_status(ierr)
  end subroutine check_optimizer_endpoint

  subroutine read_last_opt_geometry(filename, natoms, coords_ang, cycles, converged, ierr)
    implicit none
    character(len=*), intent(in) :: filename
    integer, intent(in) :: natoms
    double precision, intent(out) :: coords_ang(3,natoms)
    integer, intent(out) :: cycles
    logical, intent(out) :: converged
    integer, intent(inout) :: ierr
    integer :: unit, status, i
    character(len=512) :: line
    character(len=8) :: element

    if (ierr /= 0) return
    cycles = 0
    converged = .false.
    open(newunit=unit,file=filename,status='old',iostat=ierr)
    if (ierr /= 0) return
    do
      read(unit,'(A)',iostat=status) line
      if (status < 0) exit
      if (status /= 0) then
        ierr = 49
        exit
      endif
      if (index(line,'GEOMETRY OPTIMIZED AFTER') > 0) converged = .true.
      if (trim(adjustl(line)) /= 'GEOMETRY INPUT') cycle
      read(unit,'(A)',iostat=status) line
      do i=1,natoms
        if (status == 0) read(unit,*,iostat=status) element,coords_ang(:,i)
      enddo
      if (status /= 0) then
        ierr = 49
        exit
      endif
      cycles = cycles+1
    enddo
    close(unit)
    if (cycles == 0) ierr = 49
  end subroutine read_last_opt_geometry

  subroutine run_benchmark(csv_file, ierr)
    implicit none
    character(len=*), intent(in) :: csv_file
    integer, intent(out) :: ierr

    integer, parameter :: natoms = 3, nsizes = 6, maxpoints = 65536
    integer, parameter :: sizes(nsizes) = (/64,256,1024,4096,16384,65536/)
    integer :: atomic_numbers(natoms), isize, irep, icall, npoints, ncalls
    integer :: count_start, count_end, count_rate, i, ounit
    double precision :: coords_ang(3,natoms), energy, elapsed, best
    double precision :: ptchg(4,0)
    double precision, allocatable :: probes(:,:), esp(:), efield(:,:), efg(:,:,:)
    character(len=256) :: keywd
    character(len=80) :: job_name

    ierr = 0
    atomic_numbers = (/8,1,1/)
    coords_ang(:,1) = (/-0.33840d0, 0.00380d0, 0.23923d0/)
    coords_ang(:,2) = (/-0.33510d0,-0.00190d0,-0.83277d0/)
    coords_ang(:,3) = (/ 0.67350d0,-0.00190d0, 0.59353d0/)
    keywd = 'HF BASIS=STO-3G CUTOFF=1.0D-10 DENSERMS=1.0D-8 ENERGY CHARGE=0'
    job_name = 'oeprop_grid_scaling'

    allocate(probes(3,maxpoints),esp(maxpoints),efield(3,maxpoints), &
      efg(3,3,maxpoints),stat=ierr)
    if (ierr /= 0) return

    call setQuickJob(job_name,keywd,natoms,atomic_numbers,.true.,ierr)
    if (ierr /= 0) return
    call getQuickEnergy(coords_ang,0,ptchg,energy,ierr)
    if (ierr /= 0) then
      call deleteQuickJob(i)
      return
    endif

    ounit = -1
    if (is_master) then
      open(newunit=ounit,file=csv_file,status='replace',action='write',iostat=ierr)
      if (ierr == 0) write(ounit,'(A)') &
        'property,npoints,calls_per_trial,best_seconds_per_call,points_per_second,energy_hartree'
    endif
#ifdef MPIV
    call MPI_BCAST(ierr,1,MPI_INTEGER,0,quick_comm,quick_mpi_error)
#endif
    if (ierr /= 0) then
      call deleteQuickJob(i)
      return
    endif

    call system_clock(count_rate=count_rate)
    do isize=1,nsizes
      npoints = sizes(isize)
      call fill_probe_grid(npoints,probes(:,1:npoints))
      ! Each timing trial processes at least 65536 points so that small-grid
      ! measurements remain above the millisecond system-clock resolution.
      ncalls = max(1,65536/npoints)

      best = huge(best)
      do irep=1,3
        call system_clock(count_start)
        do icall=1,ncalls
          call getQuickOEPROP(npoints,probes(:,1:npoints),ierr,esp=esp(1:npoints))
          if (ierr /= 0) exit
        enddo
        call system_clock(count_end)
        if (ierr /= 0) exit
        elapsed = dble(count_end-count_start)/(dble(count_rate)*dble(ncalls))
        best = min(best,elapsed)
      enddo
      if (ierr /= 0) exit
      if (is_master) write(ounit,'(A,",",I0,",",I0,",",ES16.8,",",ES16.8,",",ES24.16)') &
        'ESP',npoints,ncalls,best,dble(npoints)/best,energy

      best = huge(best)
      do irep=1,3
        call system_clock(count_start)
        do icall=1,ncalls
          call getQuickOEPROP(npoints,probes(:,1:npoints),ierr,efield=efield(:,1:npoints))
          if (ierr /= 0) exit
        enddo
        call system_clock(count_end)
        if (ierr /= 0) exit
        elapsed = dble(count_end-count_start)/(dble(count_rate)*dble(ncalls))
        best = min(best,elapsed)
      enddo
      if (ierr /= 0) exit
      if (is_master) write(ounit,'(A,",",I0,",",I0,",",ES16.8,",",ES16.8,",",ES24.16)') &
        'EFIELD',npoints,ncalls,best,dble(npoints)/best,energy

      best = huge(best)
      do irep=1,3
        call system_clock(count_start)
        do icall=1,ncalls
          call getQuickOEPROP(npoints,probes(:,1:npoints),ierr,efg=efg(:,:,1:npoints))
          if (ierr /= 0) exit
        enddo
        call system_clock(count_end)
        if (ierr /= 0) exit
        elapsed = dble(count_end-count_start)/(dble(count_rate)*dble(ncalls))
        best = min(best,elapsed)
      enddo
      if (ierr /= 0) exit
      if (is_master) write(ounit,'(A,",",I0,",",I0,",",ES16.8,",",ES16.8,",",ES24.16)') &
        'EFG',npoints,ncalls,best,dble(npoints)/best,energy
    enddo

    if (is_master) close(ounit)
    call deleteQuickJob(i)
    if (ierr == 0 .and. i /= 0) ierr = i
  end subroutine run_benchmark

  subroutine fill_probe_grid(npoints,probes)
    implicit none
    integer, intent(in) :: npoints
    double precision, intent(out) :: probes(3,npoints)
    integer :: ip
    double precision :: phi, radius, rxy, zdir
    double precision, parameter :: pi = acos(-1.0d0)

    do ip=1,npoints
      zdir = 1.0d0-2.0d0*(dble(ip)-0.5d0)/dble(npoints)
      rxy = sqrt(max(0.0d0,1.0d0-zdir*zdir))
      phi = pi*(3.0d0-sqrt(5.0d0))*dble(ip-1)
      radius = 4.5d0+0.5d0*sin(0.37d0*dble(ip))
      probes(:,ip) = radius*(/rxy*cos(phi),rxy*sin(phi),zdir/)
    enddo
  end subroutine fill_probe_grid

  subroutine run_trajectory(xyz_file, probe_file, csv_file, keywd, basename, ierr)
    implicit none
    character(len=*), intent(in) :: xyz_file, probe_file, csv_file, keywd, basename
    integer, intent(out) :: ierr

    integer :: natoms, npoints, frame, i, ios, iunit, ounit
    integer, allocatable :: atomic_numbers(:)
    integer, allocatable :: frame_numbers(:)
    double precision, allocatable :: coords_ang(:,:), probes_ang(:,:), probes_bohr(:,:)
    double precision, allocatable :: esp(:), efield(:,:), efg(:,:,:)
    double precision :: energy
    double precision :: ptchg(4,0)
    character(len=256) :: comment
    character(len=80) :: job_name
    character(len=256) :: keywd_local

    ierr = 0
    call read_probes(probe_file, npoints, probes_ang, ierr)
    if (ierr /= 0) return
    allocate(probes_bohr(3,npoints), esp(npoints), efield(3,npoints), efg(3,3,npoints), stat=ierr)
    if (ierr /= 0) return
    probes_bohr = probes_ang / BOHRS_TO_A

    call read_first_xyz_frame(xyz_file, natoms, atomic_numbers, coords_ang, ierr)
    if (ierr /= 0) return

    job_name = basename
    keywd_local = keywd
    call setQuickJob(job_name, keywd_local, natoms, atomic_numbers, .true., ierr)
    if (ierr /= 0) return

    ounit = -1
    if (is_master) then
      open(newunit=ounit, file=csv_file, status='replace', action='write', iostat=ierr)
      if (ierr == 0) call write_csv_header(ounit)
    endif
#ifdef MPIV
    call MPI_BCAST(ierr,1,MPI_INTEGER,0,quick_comm,quick_mpi_error)
#endif
    if (ierr /= 0) return

    open(newunit=iunit, file=xyz_file, status='old', action='read', iostat=ierr)
    if (ierr /= 0) return

    frame = 0
    do
      read(iunit,*,iostat=ios) i
      if (ios < 0) exit
      if (ios /= 0) then
        ierr = 49
        exit
      endif
      if (i /= natoms) then
        ierr = 50
        exit
      endif
      read(iunit,'(A)',iostat=ios) comment
      if (ios /= 0) then
        ierr = 49
        exit
      endif
      allocate(frame_numbers(natoms), stat=ierr)
      if (ierr /= 0) exit
      call read_xyz_atoms(iunit, natoms, frame_numbers, coords_ang, ierr)
      if (ierr /= 0) exit
      if (any(frame_numbers /= atomic_numbers)) then
        ierr = 51
        exit
      endif
      deallocate(frame_numbers)

      frame = frame + 1
      call getQuickEnergy(coords_ang, 0, ptchg, energy, ierr)
      if (ierr /= 0) exit
      call getQuickOEPROP(npoints, probes_bohr, ierr, esp=esp, efield=efield, efg=efg)
      if (ierr /= 0) exit
      if (is_master) call append_property_csv(ounit, frame, npoints, energy, probes_ang, esp, efield, efg)
    enddo

    close(iunit)
    if (is_master) close(ounit)
    call deleteQuickJob(i)
    if (allocated(frame_numbers)) deallocate(frame_numbers)
    if (ierr == 0 .and. i /= 0) ierr = i
  end subroutine run_trajectory

  subroutine read_first_xyz_frame(xyz_file, natoms, atomic_numbers, coords_ang, ierr)
    implicit none
    character(len=*), intent(in) :: xyz_file
    integer, intent(out) :: natoms
    integer, allocatable, intent(out) :: atomic_numbers(:)
    double precision, allocatable, intent(out) :: coords_ang(:,:)
    integer, intent(out) :: ierr
    integer :: iunit, ios
    character(len=256) :: comment

    ierr = 0
    open(newunit=iunit, file=xyz_file, status='old', action='read', iostat=ierr)
    if (ierr /= 0) return
    read(iunit,*,iostat=ios) natoms
    if (ios /= 0 .or. natoms <= 0) then
      ierr = 49
      close(iunit)
      return
    endif
    read(iunit,'(A)',iostat=ios) comment
    if (ios /= 0) then
      ierr = 49
      close(iunit)
      return
    endif
    allocate(atomic_numbers(natoms), coords_ang(3,natoms), stat=ierr)
    if (ierr /= 0) then
      close(iunit)
      return
    endif
    call read_xyz_atoms(iunit, natoms, atomic_numbers, coords_ang, ierr)
    close(iunit)
  end subroutine read_first_xyz_frame

  subroutine read_xyz_atoms(iunit, natoms, atomic_numbers, coords_ang, ierr)
    implicit none
    integer, intent(in) :: iunit, natoms
    integer, intent(out) :: atomic_numbers(natoms)
    double precision, intent(out) :: coords_ang(3,natoms)
    integer, intent(out) :: ierr
    integer :: i, ios
    character(len=16) :: sym

    ierr = 0
    do i=1,natoms
      read(iunit,*,iostat=ios) sym, coords_ang(1,i), coords_ang(2,i), coords_ang(3,i)
      if (ios /= 0) then
        ierr = 49
        return
      endif
      atomic_numbers(i) = symbol_to_z(sym)
      if (atomic_numbers(i) <= 0) then
        ierr = 52
        return
      endif
    enddo
  end subroutine read_xyz_atoms

  subroutine read_probes(probe_file, npoints, probes_ang, ierr)
    implicit none
    character(len=*), intent(in) :: probe_file
    integer, intent(out) :: npoints
    double precision, allocatable, intent(out) :: probes_ang(:,:)
    integer, intent(out) :: ierr
    integer :: iunit, ios, i

    ierr = 0
    open(newunit=iunit, file=probe_file, status='old', action='read', iostat=ierr)
    if (ierr /= 0) return
    read(iunit,*,iostat=ios) npoints
    if (ios /= 0 .or. npoints <= 0) then
      ierr = 46
      close(iunit)
      return
    endif
    allocate(probes_ang(3,npoints), stat=ierr)
    if (ierr /= 0) then
      close(iunit)
      return
    endif
    do i=1,npoints
      read(iunit,*,iostat=ios) probes_ang(1,i), probes_ang(2,i), probes_ang(3,i)
      if (ios /= 0) then
        ierr = 49
        exit
      endif
    enddo
    close(iunit)
  end subroutine read_probes

  subroutine write_property_csv(csv_file, frame, npoints, energy, probes_ang, esp, efield, efg, ierr)
    implicit none
    character(len=*), intent(in) :: csv_file
    integer, intent(in) :: frame, npoints
    double precision, intent(in) :: energy, probes_ang(3,npoints), esp(npoints), efield(3,npoints), efg(3,3,npoints)
    integer, intent(out) :: ierr
    integer :: ounit

    open(newunit=ounit, file=csv_file, status='replace', action='write', iostat=ierr)
    if (ierr /= 0) return
    call write_csv_header(ounit)
    call append_property_csv(ounit, frame, npoints, energy, probes_ang, esp, efield, efg)
    close(ounit)
  end subroutine write_property_csv

  subroutine write_csv_header(ounit)
    implicit none
    integer, intent(in) :: ounit

    write(ounit,'(A)') 'frame,probe,energy_hartree,x_ang,y_ang,z_ang,esp,efield_x,efield_y,efield_z,' // &
      'efield_mag,efg_xx,efg_xy,efg_xz,efg_yx,efg_yy,efg_yz,efg_zx,efg_zy,efg_zz,' // &
      'efg_trace,efg_frobenius,efg_anisotropy'
  end subroutine write_csv_header

  subroutine append_property_csv(ounit, frame, npoints, energy, probes_ang, esp, efield, efg)
    implicit none
    integer, intent(in) :: ounit, frame, npoints
    double precision, intent(in) :: energy, probes_ang(3,npoints), esp(npoints), efield(3,npoints), efg(3,3,npoints)
    integer :: ip
    double precision :: efield_mag, trace, frob, anis

    do ip=1,npoints
      efield_mag = sqrt(sum(efield(:,ip)**2))
      trace = efg(1,1,ip) + efg(2,2,ip) + efg(3,3,ip)
      frob = sqrt(sum(efg(:,:,ip)**2))
      anis = efg_anisotropy(efg(:,:,ip), trace)
      write(ounit,'(I0,",",I0,",",ES24.16,",",19(ES24.16,","),ES24.16)') &
        frame, ip, energy, probes_ang(1,ip), probes_ang(2,ip), probes_ang(3,ip), esp(ip), &
        efield(1,ip), efield(2,ip), efield(3,ip), efield_mag, &
        efg(1,1,ip), efg(1,2,ip), efg(1,3,ip), &
        efg(2,1,ip), efg(2,2,ip), efg(2,3,ip), &
        efg(3,1,ip), efg(3,2,ip), efg(3,3,ip), trace, frob, anis
    enddo
  end subroutine append_property_csv

  function efg_anisotropy(g, trace) result(anis)
    implicit none
    double precision, intent(in) :: g(3,3), trace
    double precision :: anis, iso, dev2
    integer :: i, j

    iso = trace / 3.0d0
    dev2 = 0.0d0
    do i=1,3
      do j=1,3
        if (i == j) then
          dev2 = dev2 + (g(i,j)-iso)**2
        else
          dev2 = dev2 + g(i,j)**2
        endif
      enddo
    enddo
    anis = sqrt(1.5d0 * dev2)
  end function efg_anisotropy

  function symbol_to_z(sym_in) result(z)
    implicit none
    character(len=*), intent(in) :: sym_in
    integer :: z
    character(len=16) :: sym

    sym = adjustl(sym_in)
    select case(trim(sym))
    case('H','h')
      z = 1
    case('C','c')
      z = 6
    case('N','n')
      z = 7
    case('O','o')
      z = 8
    case('F','f')
      z = 9
    case('Cl','CL','cl')
      z = 17
    case default
      z = -1
    end select
  end function symbol_to_z

  ! Acceptance and timing share exactly one initialized SCF density. The
  ! density snapshot is test-only, portable text, in Fortran AO column order.
  subroutine run_acceptance(mode, xyz_file, probe_file, prefix, keyword, density_file, repeats_arg, ierr)
    use quick_api_module, only: quick_api
    use quick_calculated_module, only: quick_qm_struct
    use quick_method_module, only: quick_method
    use quick_oeproperties_module, only: compute_efg_values_numerical
    implicit none
    character(len=*), intent(in) :: mode, xyz_file, probe_file, prefix, keyword, density_file, repeats_arg
    integer, intent(out) :: ierr
    integer :: natoms, npoints, cleanup, unit, j, k, np, rep, repeats, before_step, scf_calls, keyword_pos
    integer :: launches_before(3), launches_after(3)
    integer, allocatable :: numbers(:)
    integer, parameter :: sizes(6) = (/64,256,1024,4096,16384,65536/)
    double precision, allocatable :: coords(:,:), points_ang(:,:), points(:,:), displaced(:,:)
    double precision, allocatable :: v(:), vp(:), vm(:), e(:,:), ep(:,:), em(:,:), g(:,:,:), gn(:,:,:)
    double precision, allocatable :: gradients(:,:), first_gradients(:,:)
    double precision :: energy, first_energy, ptchg(4,0), ptchg_grad(3,0), h, elapsed, symmetry
    logical :: gradient_mode
    character(len=256) :: keywd
    character(len=80) :: job
    character(len=8), parameter :: props(4) = (/'esp     ','efield  ','efg     ','all     '/)

    ierr = 0
    unit = -1
    call read_first_xyz_frame(xyz_file, natoms, numbers, coords, ierr)
    if (ierr /= 0) return
    call read_probes(probe_file, npoints, points_ang, ierr)
    if (ierr /= 0) return
    allocate(points(3,npoints), displaced(3,npoints), v(npoints), vp(npoints), vm(npoints), &
      e(3,npoints), ep(3,npoints), em(3,npoints), g(3,3,npoints), gn(3,3,npoints), &
      gradients(3,natoms), first_gradients(3,natoms), stat=ierr)
    if (ierr /= 0) return
    points = points_ang / BOHRS_TO_A
    keywd = keyword
    gradient_mode = index(mode,'gradient') > 0
    if (gradient_mode) then
      keyword_pos = index(keywd,'ENERGY')
      if (keyword_pos > 0) keywd = keywd(:keyword_pos-1) // 'GRADIENT' // trim(keywd(keyword_pos+6:))
      if (index(keywd,'GRADIENT') == 0) keywd = trim(keywd) // ' GRADIENT'
      if (index(keywd,'GRADCUTOFF') == 0) keywd = trim(keywd) // ' GRADCUTOFF=1.0D-12'
    endif
    job = 'oeprop_acceptance'
    call setQuickJob(job, keywd, natoms, numbers, .true., ierr)
    if (ierr /= 0) return
    call getQuickOEPROP(npoints, points, ierr, esp=vp, efield=ep, efg=gn)
    if (ierr /= 47) then
      ierr = 49
      goto 900
    endif
    before_step = quick_api%step
    if (gradient_mode) then
      call getQuickEnergyGradients(coords,0,ptchg,energy,gradients,ptchg_grad,ierr)
      first_gradients = gradients
    else
      call getQuickEnergy(coords,0,ptchg,energy,ierr)
    endif
    if (ierr /= 0) goto 900
    if (quick_api%step /= before_step+1) then
      ierr = 53
      goto 900
    endif
    scf_calls = 1
    first_energy = energy
    if (density_file /= '-') call read_density(density_file, ierr)
    if (ierr /= 0) goto 900
    call write_density(prefix // '.density.dat', ierr)
    if (ierr /= 0) goto 900
    before_step = quick_api%step
    launches_before = 0
#if defined(GPU) || defined(MPIV_GPU)
    call gpu_oeprop_launch_counts(launches_before(1),launches_before(2),launches_before(3))
#endif

    if (mode == 'benchmark-case') then
      read(repeats_arg,*,iostat=ierr) repeats
      if (ierr /= 0 .or. repeats < 1) then
        ierr = 49
        goto 900
      endif
      if (is_master) then
        open(newunit=unit, file=prefix // '.timing.csv', status='replace', iostat=ierr)
        if (ierr == 0) write(unit,'(A)') 'npoints,property,repeat,seconds,checksum,scf_calls'
      endif
      call sync_status(ierr)
      if (ierr /= 0) goto 900
      do k=1,size(sizes)
        np = sizes(k)
        if (np > npoints) cycle
        do j=1,size(props)
          ! One warm call at each size/property, outside all recorded trials.
          call timed_property(trim(props(j)), np, points(:,:np), v(:np), e(:,:np), g(:,:,:np), elapsed, ierr)
          if (ierr /= 0) goto 900
          do rep=1,repeats
            call timed_property(trim(props(j)), np, points(:,:np), v(:np), e(:,:np), g(:,:,:np), elapsed, ierr)
            if (ierr /= 0) goto 900
            if (is_master) write(unit,'(I0,",",A,",",I0,",",ES24.16,",",ES24.16,",",I0)') &
              np, trim(props(j)), rep, elapsed, property_checksum(trim(props(j)), v(:np), e(:,:np), g(:,:,:np)), 1
          enddo
        enddo
      enddo
      if (is_master) close(unit)
    else
      call getQuickOEPROP(npoints, points, ierr, esp=v, efield=e, efg=g)
      if (ierr /= 0) goto 900
      if (is_master) call write_property_csv(prefix // '.api.csv', 1, npoints, energy, points_ang, v, e, g, ierr)
      call sync_status(ierr)
      if (ierr /= 0) goto 900
      if (is_master) then
        open(newunit=unit, file=prefix // '.checks.csv', status='replace', iostat=ierr)
        if (ierr == 0) write(unit,'(A)') 'check,max_abs'
      endif
      call sync_status(ierr)
      if (ierr /= 0) goto 900

      ! Reverse point order, change the buffer size, and request outputs singly.
      ! These calls detect stale device grids/results and optional-output coupling.
      displaced = points(:,npoints:1:-1)
      call getQuickOEPROP(npoints, displaced, ierr, esp=vp, efield=ep, efg=gn)
      if (ierr /= 0) goto 900
      call write_check(unit, 'reverse_esp', maxval(abs(v-vp(npoints:1:-1))))
      call write_check(unit, 'reverse_efield', maxval(abs(e-ep(:,npoints:1:-1))))
      call write_check(unit, 'reverse_efg', maxval(abs(g-gn(:,:,npoints:1:-1))))
      np = max(1,npoints/2)
      call getQuickOEPROP(np, points(:,:np), ierr, esp=vp(:np), efield=ep(:,:np), efg=gn(:,:,:np))
      if (ierr /= 0) goto 900
      call write_check(unit, 'smaller_esp', maxval(abs(v(:np)-vp(:np))))
      call write_check(unit, 'smaller_efield', maxval(abs(e(:,:np)-ep(:,:np))))
      call write_check(unit, 'smaller_efg', maxval(abs(g(:,:,:np)-gn(:,:,:np))))
      call getQuickOEPROP(npoints, points, ierr, esp=vp)
      if (ierr /= 0) goto 900
      call write_check(unit, 'repeat_esp', maxval(abs(v-vp)))
      call getQuickOEPROP(npoints, points, ierr, efield=ep)
      if (ierr /= 0) goto 900
      call write_check(unit, 'repeat_efield', maxval(abs(e-ep)))
      call getQuickOEPROP(npoints, points, ierr, efg=gn)
      if (ierr /= 0) goto 900
      call write_check(unit, 'repeat_efg', maxval(abs(g-gn)))
      call getQuickOEPROP(0, points(:,:0), ierr, esp=vp(:0))
      if (ierr /= 46) then
        ierr = 49
        goto 900
      endif
      call write_check(unit, 'invalid_npoints', 0.0d0)
      call getQuickOEPROP(npoints, points, ierr)
      if (ierr /= 48) then
        ierr = 49
        goto 900
      endif
      call write_check(unit, 'missing_outputs', 0.0d0)
      ierr = 0

      ! Central differences in bohr on a fixed density, with two step sizes.
      ! E=-grad(V), G_ij=dE_i/dC_j; h and h/2 expose truncation error.
      do k=1,2
        h = 1.0d-4 / dble(k)
        do j=1,3
          displaced = points
          displaced(j,:) = points(j,:) + h
          call getQuickOEPROP(npoints, displaced, ierr, esp=vp, efield=ep)
          if (ierr /= 0) goto 900
          displaced(j,:) = points(j,:) - h
          call getQuickOEPROP(npoints, displaced, ierr, esp=vm, efield=em)
          if (ierr /= 0) goto 900
          call write_check(unit, 'efield_fd_' // achar(ichar('x')+j-1) // '_' // achar(ichar('0')+k), &
            maxval(abs(e(j,:)+(vp-vm)/(2.0d0*h))))
          call write_check(unit, 'efg_fd_' // achar(ichar('x')+j-1) // '_' // achar(ichar('0')+k), &
            maxval(abs(g(:,j,:)-(ep-em)/(2.0d0*h))))
        enddo
      enddo
      call compute_efg_values_numerical(npoints, points, gn)
      call write_check(unit, 'analytic_vs_numerical_efg', maxval(abs(g-gn)))
      symmetry = 0.0d0
      do j=1,3
        do k=1,3
          symmetry = max(symmetry,maxval(abs(g(j,k,:)-g(k,j,:))))
        enddo
      enddo
      call write_check(unit, 'efg_symmetry', symmetry)
      call check_dispatch(prefix, npoints, points, energy, ierr)
      if (ierr /= 0) goto 900
      call getQuickOEPROP(npoints, points, ierr, esp=vp, efield=ep, efg=gn)
      if (ierr /= 0) goto 900
      call write_check(unit, 'after_surface_esp', maxval(abs(v-vp)))
      call write_check(unit, 'after_surface_efield', maxval(abs(e-ep)))
      call write_check(unit, 'after_surface_efg', maxval(abs(g-gn)))
      if (mode == 'lifecycle-gradient') then
        if (quick_api%step /= before_step) then
          ierr = 53
          goto 900
        endif
        ! A second gradient SCF at identical geometry must refresh GPU state
        ! after the preceding optional-output and generated-surface calls.
        quick_method%density_surface = .false.
        quick_method%esp_grid = .false.
        quick_method%efield_grid = .false.
        quick_method%efg_grid = .false.
        call getQuickEnergyGradients(coords,0,ptchg,energy,gradients,ptchg_grad,ierr)
        if (ierr /= 0) goto 900
        if (quick_api%step /= before_step+1) then
          ierr = 53
          goto 900
        endif
        before_step = quick_api%step
        scf_calls = 2
        call getQuickOEPROP(npoints,points,ierr,esp=vp,efield=ep,efg=gn)
        if (ierr /= 0) goto 900
        call write_check(unit, 'repeated_gradient_energy', abs(first_energy-energy))
        call write_check(unit, 'repeated_gradient_components', maxval(abs(first_gradients-gradients)))
        call write_check(unit, 'repeated_gradient_esp', maxval(abs(v-vp)))
        call write_check(unit, 'repeated_gradient_efield', maxval(abs(e-ep)))
        call write_check(unit, 'repeated_gradient_efg', maxval(abs(g-gn)))
      endif
      if (is_master) close(unit)
    endif
    ! This check fails if any property/timing call inadvertently invokes SCF.
    if (quick_api%step /= before_step) ierr = 53
    launches_after = 0
#if defined(GPU) || defined(MPIV_GPU)
    call gpu_oeprop_launch_counts(launches_after(1),launches_after(2),launches_after(3))
    if (any(launches_after-launches_before <= 0)) ierr = 53
#endif
    call write_kernel_counts(prefix // '.kernels.csv',launches_after-launches_before,ierr)
    call sync_status(ierr)
900 call deleteQuickJob(cleanup)
    if (ierr == 0) ierr = cleanup
    if (ierr /= 0) return
    ! Caller buffers remain allocated; a deleted density must be rejected
    ! before attempting to touch deallocated host/device molecular state.
    call getQuickOEPROP(npoints,points,cleanup,esp=vp,efield=ep,efg=gn)
    if (cleanup /= 47) then
      ierr = 49
      return
    endif
    if (is_master) then
      open(newunit=unit,file=prefix // '.lifecycle.csv',status='replace',iostat=ierr)
      if (ierr == 0) then
        write(unit,'(A)') 'scf_calls,property_calls_preserve_scf,post_delete_ierr'
        write(unit,'(I0,",",I0,",",I0)') scf_calls,1,cleanup
        close(unit)
      endif
    endif
    call sync_status(ierr)
  end subroutine run_acceptance

  subroutine write_check(unit, label, error)
    implicit none
    integer, intent(in) :: unit
    character(len=*), intent(in) :: label
    double precision, intent(in) :: error
    if (is_master) write(unit,'(A,",",ES24.16)') label, error
  end subroutine write_check

  subroutine timed_property(property, npoints, points, esp, efield, efg, elapsed, ierr)
    use iso_fortran_env, only: int64
    implicit none
    character(len=*), intent(in) :: property
    integer, intent(in) :: npoints
    double precision, intent(in) :: points(3,npoints)
    double precision, intent(out) :: esp(npoints), efield(3,npoints), efg(3,3,npoints), elapsed
    integer, intent(out) :: ierr
    integer(int64) :: start, finish, rate
#ifdef MPIV
    integer :: mpierr
    double precision :: max_elapsed
    call MPI_BARRIER(quick_comm, mpierr)
#endif
    esp = 0.0d0
    efield = 0.0d0
    efg = 0.0d0
    call system_clock(start, rate)
    select case(property)
    case('esp')
      call getQuickOEPROP(npoints, points, ierr, esp=esp)
    case('efield')
      call getQuickOEPROP(npoints, points, ierr, efield=efield)
    case('efg')
      call getQuickOEPROP(npoints, points, ierr, efg=efg)
    case('all')
      call getQuickOEPROP(npoints, points, ierr, esp=esp, efield=efield, efg=efg)
    end select
    call system_clock(finish)
    elapsed = dble(finish-start)/dble(rate)
#ifdef MPIV
    call MPI_ALLREDUCE(elapsed, max_elapsed, 1, MPI_DOUBLE_PRECISION, MPI_MAX, quick_comm, mpierr)
    elapsed = max_elapsed
#endif
  end subroutine timed_property

  function property_checksum(property, esp, efield, efg) result(value)
    implicit none
    character(len=*), intent(in) :: property
    double precision, intent(in) :: esp(:), efield(:,:), efg(:,:,:)
    double precision :: value
    value = 0.0d0
    if (property == 'esp' .or. property == 'all') value = value + sum(esp)
    if (property == 'efield' .or. property == 'all') value = value + sum(efield)
    if (property == 'efg' .or. property == 'all') value = value + sum(efg)
  end function property_checksum

  subroutine write_density(filename, ierr)
    use quick_calculated_module, only: quick_qm_struct
    use quick_method_module, only: quick_method
    implicit none
    character(len=*), intent(in) :: filename
    integer, intent(out) :: ierr
    integer :: unit
    ierr = 0
    if (is_master) then
      open(newunit=unit,file=filename,status='replace',iostat=ierr)
      if (ierr == 0) then
        write(unit,*) size(quick_qm_struct%dense,1), quick_method%UNRST
        write(unit,'(ES26.17)') quick_qm_struct%dense
        if (quick_method%UNRST) write(unit,'(ES26.17)') quick_qm_struct%denseb
        close(unit)
      endif
    endif
    call sync_status(ierr)
  end subroutine write_density

  subroutine read_density(filename, ierr)
    use quick_calculated_module, only: quick_qm_struct
    use quick_method_module, only: quick_method
    implicit none
    character(len=*), intent(in) :: filename
    integer, intent(out) :: ierr
    integer :: unit, n
    logical :: unrestricted
    open(newunit=unit,file=filename,status='old',iostat=ierr)
    if (ierr /= 0) return
    read(unit,*,iostat=ierr) n, unrestricted
    if (ierr /= 0) then
      close(unit)
      return
    endif
    if (n /= size(quick_qm_struct%dense,1) .or. (unrestricted .neqv. quick_method%UNRST)) then
      ierr = 49
      close(unit)
      return
    endif
    read(unit,*,iostat=ierr) quick_qm_struct%dense
    if (ierr == 0 .and. unrestricted) read(unit,*,iostat=ierr) quick_qm_struct%denseb
    close(unit)
    if (ierr == 0 .and. unrestricted) quick_qm_struct%denseab = quick_qm_struct%dense + quick_qm_struct%denseb
  end subroutine read_density

  subroutine check_dispatch(prefix, npoints, points, energy, ierr)
    use quick_method_module, only: quick_method
    use quick_molspec_module, only: quick_molspec
    use quick_files_module, only: espFileName, efieldFileName, efgFileName
    use quick_molsurface_module, only: generate_density_surfaces
    use quick_oeproperties_module, only: compute_oeprop
    implicit none
    character(len=*), intent(in) :: prefix
    integer, intent(in) :: npoints
    double precision, intent(in) :: points(3,npoints), energy
    integer, intent(out) :: ierr
    integer :: nsurface
    double precision, allocatable :: v(:), e(:,:), g(:,:,:)
#ifdef MPIV
    integer :: mpierr
#endif
    ierr = 0
    quick_method%ext_grid = .true.
    quick_method%esp_grid = .true.
    quick_method%efield_grid = .true.
    quick_method%efg_grid = .true.
    quick_method%efg_grid_numerical = .false.
    quick_method%extgrid_angstrom = .true.
    quick_molspec%nextpoint = npoints
    allocate(quick_molspec%extpointxyz(3,npoints),stat=ierr)
    if (ierr /= 0) return
    quick_molspec%extpointxyz = points
    espFileName = prefix // '.esp'
    efieldFileName = prefix // '.efield'
    efgFileName = prefix // '.efg'
    call compute_oeprop()
    deallocate(quick_molspec%extpointxyz)

    quick_method%ext_grid = .false.
    quick_method%density_surface = .true.
    quick_method%density_surface_value = 1.0d-3
    quick_method%density_surface_spacing = 1.25d0
    quick_method%density_surface_max_radius = 6.0d0
    espFileName = prefix // '.surface.esp'
    efieldFileName = prefix // '.surface.efield'
    efgFileName = prefix // '.surface.efg'
    call compute_oeprop()
    ! Regenerate the deterministic points from the same density, then compare
    ! the no-I/O API with the genuine generated-surface file dispatch above.
    if (is_master) call generate_density_surfaces()
    nsurface = quick_molspec%nvdwpoint
#ifdef MPIV
    call MPI_BCAST(nsurface,1,MPI_INTEGER,0,quick_comm,mpierr)
    if (.not.is_master) allocate(quick_molspec%vdwpointxyz(3,nsurface),stat=ierr)
    call MPI_BCAST(quick_molspec%vdwpointxyz,3*nsurface,MPI_DOUBLE_PRECISION,0,quick_comm,mpierr)
#endif
    allocate(v(nsurface),e(3,nsurface),g(3,3,nsurface),stat=ierr)
    if (ierr /= 0) return
    call getQuickOEPROP(nsurface,quick_molspec%vdwpointxyz,ierr,esp=v,efield=e,efg=g)
    if (ierr /= 0) return
    if (is_master) call write_property_csv(prefix // '.surface.api.csv',1,nsurface,energy, &
      quick_molspec%vdwpointxyz*BOHRS_TO_A,v,e,g,ierr)
    call sync_status(ierr)
    deallocate(quick_molspec%vdwpointxyz)
  end subroutine check_dispatch


  subroutine sync_status(ierr)
    implicit none
    integer, intent(inout) :: ierr
#ifdef MPIV
    integer :: collective_status, mpierr
    ierr = abs(ierr)
    call MPI_ALLREDUCE(ierr,collective_status,1,MPI_INTEGER,MPI_MAX,quick_comm,mpierr)
    ierr = collective_status
#endif
  end subroutine sync_status

  subroutine write_kernel_counts(filename, counts, ierr)
    implicit none
    character(len=*), intent(in) :: filename
    integer, intent(in) :: counts(3)
    integer, intent(inout) :: ierr
    integer :: unit, i, ranks, io_status
    integer, allocatable :: all_counts(:,:)
#ifdef MPIV
    integer :: mpierr
    ranks = quick_comm_size
#else
    ranks = 1
#endif
    allocate(all_counts(3,ranks))
#ifdef MPIV
    call MPI_GATHER(counts,3,MPI_INTEGER,all_counts,3,MPI_INTEGER,0,quick_comm,mpierr)
#else
    all_counts(:,1) = counts
#endif
    if (is_master) then
      open(newunit=unit,file=filename,status='replace',iostat=io_status)
      if (io_status /= 0) then
        ierr = 49
      else
        write(unit,'(A)') 'rank,esp_launches,efield_launches,efg_launches'
        do i=1,ranks
          write(unit,'(I0,3(",",I0))') i-1,all_counts(:,i)
        enddo
        close(unit)
      endif
    endif
  end subroutine write_kernel_counts

end program quick_oeprop_api_driver
