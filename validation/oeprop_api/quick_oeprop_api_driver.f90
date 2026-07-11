#include "util.fh"
!---------------------------------------------------------------------!
! QUICK no-I/O OEPROP API validation and trajectory driver.            !
!                                                                     !
! This Source Code Form is subject to the terms of the Mozilla Public !
! License, v. 2.0.                                                    !
!---------------------------------------------------------------------!

program quick_oeprop_api_driver

  use quick_api_module, only: setQuickJob, getQuickEnergy, getQuickOEPROP, deleteQuickJob
  use quick_constants_module, only: BOHRS_TO_A
#ifdef MPIV
  use mpi
  use quick_api_module, only: setQuickMPI
#endif

  implicit none

  character(len=32) :: mode
  character(len=256) :: arg1, arg2, arg3, arg4, arg5
  integer :: argc, ierr
  logical :: is_master
#ifdef MPIV
  integer :: mpierror, mpirank, mpisize
#endif

  is_master = .true.
  ierr = 0
#ifdef MPIV
  call MPI_INIT(mpierror)
  call MPI_COMM_RANK(MPI_COMM_WORLD,mpirank,mpierror)
  call MPI_COMM_SIZE(MPI_COMM_WORLD,mpisize,mpierror)
  is_master = (mpirank == 0)
  call setQuickMPI(mpirank,mpisize,ierr)
#endif

  argc = command_argument_count()
  mode = 'validate'
  if (argc >= 1) call get_command_argument(1, mode)

  select case(trim(mode))
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
  case('benchmark')
    arg1 = 'oeprop_grid_scaling.csv'
    if (argc >= 2) call get_command_argument(2, arg1)
    call run_benchmark(trim(arg1), ierr)
  case default
    if (is_master) write(*,'("Unknown mode: ",A)') trim(mode)
    ierr = 49
  end select

  if (ierr /= 0) then
    if (is_master) write(*,'("QUICK OEPROP API driver failed with ierr=",I0)') ierr
#ifdef MPIV
    call MPI_FINALIZE(mpierror)
#endif
    stop 1
  endif

#ifdef MPIV
  call MPI_FINALIZE(mpierror)
#endif

contains

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
    call MPI_BCAST(ierr,1,MPI_INTEGER,0,MPI_COMM_WORLD,mpierror)
#endif
    call deleteQuickJob(i)
    if (ierr == 0 .and. i /= 0) ierr = i
  end subroutine run_validation

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
    call MPI_BCAST(ierr,1,MPI_INTEGER,0,MPI_COMM_WORLD,mpierror)
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
    call MPI_BCAST(ierr,1,MPI_INTEGER,0,MPI_COMM_WORLD,mpierror)
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

end program quick_oeprop_api_driver
