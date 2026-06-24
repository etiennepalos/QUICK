!---------------------------------------------------------------------!
! Created by Madu Manathunga on 04/16/2020                            !
!                                                                     !
! Copyright (C) 2020-2021 Merz lab                                    !
! Copyright (C) 2020-2021 Götz lab                                    !
!                                                                     !
! This Source Code Form is subject to the terms of the Mozilla Public !
! License, v. 2.0. If a copy of the MPL was not distributed with this !
! file, You can obtain one at http://mozilla.org/MPL/2.0/.            !
!_____________________________________________________________________!

#include "util.fh"

! Interface for quick libarary
module quick_api_module

  implicit none
  private

  public :: quick_api
  public :: setQuickJob, getQuickEnergy, getQuickEnergyGradients, deleteQuickJob
  public :: getQuickOEPROP
  public :: setQuickMBXSystem, setQuickMBXWaterSystem, clearQuickMBXSystem
  public :: getQuickMBXEnergyGradientsFD

#ifdef MPIV
  public :: setQuickMPI
#endif

  type quick_api_type
    ! indicates if quick should run in library mode. This will help
    ! setting up files for quick run.
    logical :: apiMode = .false.

    ! used to determine if memory should be allocated and SAD guess should
    ! be performed
    logical :: firstStep = .true.

    ! current md step, should come from MM code
    integer :: mdstep = 1

    ! keeps track of how many times quick is called by MM code
    integer :: step = 1

    ! number of atoms
    integer :: natoms = 0

    ! number of atom types
    integer :: natm_type = 0

    ! number of external point charges
    integer :: nxt_ptchg = 0

    ! atom type/atomic number of each atom
    integer, allocatable, dimension(:) :: atm_type_id

    ! atomic numbers of atoms
    integer, allocatable, dimension(:) :: atomic_numbers

    ! xyz coordinates of atoms, size is 3*natoms
    double precision, allocatable, dimension(:,:) :: coords

    ! charge and the coordinates of external point charges
    ! size is 4*nxt_ptchg, where last element of a row holds the charge
    double precision, allocatable, dimension(:,:) :: ptchg_crd

    ! job card for quick job, essentially the first line of regular quick input file
    ! default length is 300 characters
    character(len=300) :: keywd

    ! Is the job card provided by passing a string? default is false
    logical :: hasKeywd = .false.

    ! template file name with job card
    character(len=80) :: fqin

    ! if density matrix of the previous step should be used for the current
    ! md step
    logical :: reuse_dmx = .true.

    ! total energy in hartree
    double precision :: tot_ene = 0.0d0

    ! true after a successful energy or gradient evaluation has produced
    ! a density matrix that can be used by no-I/O OEPROP evaluators.
    logical :: density_ready = .false.

    ! if gradients and point charge gradients are requested
    logical :: isForce = .false.

    ! gradients
    double precision, allocatable, dimension(:,:) :: gradient

    ! point charge gradients
    double precision, allocatable, dimension(:,:) :: ptchg_grad

    ! DL-Find opt
    logical :: usedlfind                     = .true.   ! DL-Find used as default optimizer  
    integer :: dlfind_iopt                   = 3        ! type of optimisation algorithm
    integer :: dlfind_icoord                 = 3        ! type of internal coordinates


  end type quick_api_type

! save a quick_api_type varible that othe quick modules can access
  type (quick_api_type), save :: quick_api

#ifdef MPIV
  interface setQuickMPI
    module procedure set_quick_mpi
  end interface
#endif

  interface setQuickJob
    module procedure set_quick_job
  end interface

  interface getQuickEnergy
    module procedure get_quick_energy
  end interface

  interface getQuickEnergyGradients
    module procedure get_quick_energy_gradients
  end interface

  interface getQuickOEPROP
    module procedure get_quick_oeprop
  end interface

  interface getQuickMBXEnergyGradientsFD
    module procedure get_quick_mbx_energy_gradients_fd_water
  end interface

  interface deleteQuickJob
    module procedure delete_quick_job
  end interface

  interface setQuickMBXSystem
    module procedure set_quick_mbx_system
  end interface

  interface setQuickMBXWaterSystem
    module procedure set_quick_mbx_water_system
  end interface

  interface clearQuickMBXSystem
    module procedure clear_quick_mbx_system
  end interface

contains


! allocates memory for a new quick_api_type variable
subroutine new_quick_api_type(self, natoms, atomic_numbers, ierr)

  implicit none

  type(quick_api_type), intent(inout) :: self
  integer, intent(in)   :: natoms
  integer, intent(in)   :: atomic_numbers(natoms)
  integer, intent(inout)  :: ierr
  integer :: atm_type_id(natoms)
  integer :: i, natm_type

  ! get atom types and number of types
  call get_atom_types(natoms, atomic_numbers, natm_type, atm_type_id, ierr)

  if ( .not. allocated(self%atm_type_id))    allocate(self%atm_type_id(natm_type), stat=ierr)
  if ( .not. allocated(self%atomic_numbers)) allocate(self%atomic_numbers(natoms), stat=ierr)
  if ( .not. allocated(self%coords))         allocate(self%coords(3,natoms), stat=ierr)
  if ( .not. allocated(self%gradient))          allocate(self%gradient(3,natoms), stat=ierr)

 ! save values in the quick_api struct
  self%natoms         = natoms
  self%natm_type      = natm_type
  self%atomic_numbers = atomic_numbers

  do i=1, natm_type
    self%atm_type_id(i) = atm_type_id(i)
  enddo

  ! set result vectors and matrices to zero
  self%gradient  = 0.0d0
  self%density_ready = .false.

end subroutine new_quick_api_type

! this subroutine checks if the string passed through api is a file name
! or a job card
subroutine check_fqin(fqin, keywd, ierr)

  implicit none

  character(len=80), intent(in)  :: fqin
  character(len=256), intent(in) :: keywd
  integer, intent(inout) :: ierr

  call upcase(keywd, 256)

  if (index(keywd, 'BASIS=') .ne. 0 .and. &
      (index(keywd, 'HF') .ne. 0 .or. &
       index(keywd, 'DFT') .ne. 0 .or. &
       index(keywd, 'PBE0') .ne. 0 .or. &
       index(keywd, 'B3LYP') .ne. 0 .or. &
       index(keywd, 'LIBXC=') .ne. 0)) then
    quick_api%hasKeywd = .true.
    quick_api%Keywd = keywd
  endif

  quick_api%fqin    = trim(fqin) // '.in'

end subroutine check_fqin

! reads the job card from template file with .qin extension and initialize quick
! also allocate memory for quick_api internal arrays
subroutine set_quick_job(fqin, keywd, natoms, atomic_numbers, reusedmx, ierr)

  use quick_files_module
  use quick_molspec_module, only : quick_molspec, alloc
  use quick_exception_module
  use quick_method_module
#ifdef MPIV
  use quick_mpi_module
#endif

  implicit none

  character(len=80), intent(in)  :: fqin
  character(len=256), intent(in) :: keywd
  integer, intent(in) :: natoms
  integer, intent(in) :: atomic_numbers(natoms)
  logical, intent(in) :: reusedmx
  integer, intent(out) :: ierr
  integer :: flen
  ierr=0
  

  ! allocate memory for quick_api_type
  call new_quick_api_type(quick_api, natoms, atomic_numbers, ierr)

  quick_api%reuse_dmx=reusedmx

  ! check if fqin string is a input file name or job card
  flen = LEN_TRIM(fqin)

  if(flen .gt. 1) then

    quick_api%apiMode = .true.

    call check_fqin(fqin, keywd, ierr)

  endif

  ! Quick calling flow is extremely horrible. Modules are
  ! disorganized and uses stupid tricks to avoid cyclic module
  ! dependency. This must be fixed in future! Being a sheep
  ! for now..

  ! Initialize quick
  call initialize1(ierr)

  ! set the file name and template mode in quick_files_module
  inFileName = quick_api%fqin
  isTemplate = quick_api%apiMode

#ifdef MPIV
  if(master) then
#endif

    ! set quick files
    call set_quick_files(.true.,ierr)
    CHECK_ERROR(ierr)

    ! open output file
    SAFE_CALL(quick_open(iOutFile,outFileName,'U','F','R',.false.,ierr))

    ! print copyright information
    SAFE_CALL(outputCopyright(iOutFile,ierr))

    ! write job information into output file
    SAFE_CALL(PrtDate(iOutFile,'TASK STARTS ON:',ierr))
    call print_quick_io_file(iOutFile,ierr)

#ifdef MPIV
    ! check the mpisize and turn on mpi mode
    !call check_quick_mpi(iOutFile,ierr)

    ! print mpi info into output
    if(bMPI) call print_quick_mpi(iOutFile,ierr)

  endif
#endif

#if defined(GPU) || defined(MPIV_GPU)
#if defined(GPU)
  SAFE_CALL(gpu_new(ierr))
  SAFE_CALL(gpu_init_device(ierr))
  SAFE_CALL(gpu_write_info(iOutFile, ierr))
#elif defined(MPIV_GPU)
  SAFE_CALL(gpu_new(mpirank, ierr))
  SAFE_CALL(mgpu_query(mpisize, mpirank, mgpu_id, ierr))
  SAFE_CALL(mgpu_setup(ierr))
  if (master) SAFE_CALL(mgpu_write_info(iOutFile, mpisize, mgpu_ids, ierr))
  SAFE_CALL(mgpu_init_device(mpirank, mpisize, mgpu_id, ierr))
#endif
  call gpu_allocate_scratch(.true.)
#endif

  ! read job specifications
  SAFE_CALL(read_Job_and_Atom(ierr))

#if defined(GPU) || defined(MPIV_GPU)
  call upload(quick_method, ierr)
#endif

  ! save atom number, number of atom types and number of point charges
  ! into quick_molspec
  quick_molspec%natom     = quick_api%natoms
  quick_molspec%iAtomType = quick_api%natm_type

  ! allocate memory for coordinates and charges in molspec
  SAFE_CALL(alloc(quick_molspec, quick_method%readxyz, ierr))

end subroutine set_quick_job


! initialize an optional MBX environment for QM/MB-pol library-mode jobs
subroutine set_quick_mbx_system(natoms, nsites, nmon, nat_monomers, coords_ang, &
           atom_names, monomer_names, json_file, ierr)

  use quick_mbx_module, only: quick_mbx_initialize_system

  implicit none

  integer, intent(in) :: natoms, nsites, nmon
  integer, intent(in) :: nat_monomers(nmon)
  double precision, intent(in) :: coords_ang(3,natoms)
  character(len=*), intent(in) :: atom_names(natoms)
  character(len=*), intent(in) :: monomer_names(nmon)
  character(len=*), intent(in) :: json_file
  integer, intent(inout) :: ierr

  call quick_mbx_initialize_system(natoms, nsites, nmon, nat_monomers, coords_ang, &
       atom_names, monomer_names, json_file, ierr)

end subroutine set_quick_mbx_system


! initialize an MB-pol water environment; MBX uses four electrostatic sites per water
subroutine set_quick_mbx_water_system(nwaters, coords_ang, json_file, ierr)

  use quick_mbx_module, only: quick_mbx_initialize_system

  implicit none

  integer, intent(in) :: nwaters
  double precision, intent(in) :: coords_ang(:,:)
  character(len=*), intent(in) :: json_file
  integer, intent(inout) :: ierr

  integer :: iwater, iatom, natoms, nsites
  integer, allocatable :: nat_monomers(:)
  double precision, allocatable :: coords_local(:,:)
  character(len=1), allocatable :: atom_names(:)
  character(len=3), allocatable :: monomer_names(:)

  if (ierr /= 0) return
  if (nwaters <= 0) then
     ierr = 45
     return
  endif

  natoms = 3*nwaters
  nsites = 4*nwaters
  if (size(coords_ang,1) < 3 .or. size(coords_ang,2) < natoms) then
     ierr = 45
     return
  endif

  allocate(nat_monomers(nwaters), stat=ierr)
  if (ierr /= 0) return
  allocate(coords_local(3,natoms), stat=ierr)
  if (ierr /= 0) return
  allocate(atom_names(natoms), stat=ierr)
  if (ierr /= 0) return
  allocate(monomer_names(nwaters), stat=ierr)
  if (ierr /= 0) return

  nat_monomers(:) = 3
  monomer_names(:) = 'h2o'
  coords_local(:,:) = coords_ang(1:3,1:natoms)
  do iwater=1,nwaters
     iatom = 3*(iwater-1)
     atom_names(iatom+1) = 'O'
     atom_names(iatom+2) = 'H'
     atom_names(iatom+3) = 'H'
  enddo

  call quick_mbx_initialize_system(natoms, nsites, nwaters, nat_monomers, coords_local, &
       atom_names, monomer_names, json_file, ierr)

  if (allocated(nat_monomers)) deallocate(nat_monomers)
  if (allocated(coords_local)) deallocate(coords_local)
  if (allocated(atom_names)) deallocate(atom_names)
  if (allocated(monomer_names)) deallocate(monomer_names)

end subroutine set_quick_mbx_water_system


! finalize and clear the optional MBX environment
subroutine clear_quick_mbx_system(ierr)

  use quick_mbx_module, only: quick_mbx_finalize

  implicit none

  integer, intent(out) :: ierr

  ierr = 0
  call quick_mbx_finalize()

end subroutine clear_quick_mbx_system


! computes atom types
subroutine get_atom_types(natoms, atomic_numbers, natm_type, atm_type_id, ierr)

  implicit none

  integer, intent(in)  :: natoms
  integer, intent(in)  :: atomic_numbers(natoms)
  integer, intent(out) :: natm_type
  integer, intent(out) :: atm_type_id(natoms)
  integer, intent(inout) :: ierr
  integer :: i, j, iatm
  logical :: new_atm_type

  ! set atm_type_id to zero
  call zeroiVec(atm_type_id, natoms)

  ! go through the atomic numbers, find out atom types and save them in
  ! atm_type_id vector
  natm_type = 1
  do i=1, natoms

    new_atm_type = .true.
    iatm = atomic_numbers(i)

    do j=1, natm_type
      if(atm_type_id(j) .eq. iatm) new_atm_type = .false.
    enddo

    if(new_atm_type) then
      atm_type_id(natm_type) = iatm
      natm_type = natm_type+1
    endif

  enddo

  natm_type = natm_type-1

end subroutine get_atom_types

! allocate memory for point charges and gradients
subroutine allocate_point_charge(isgrad,ierr)

  use quick_molspec_module, only: quick_molspec, realloc
  use quick_calculated_module, only: quick_qm_struct, realloc
  implicit none
  logical, intent(in) :: isgrad
  integer, intent(inout) :: ierr
  
  ! allocate memory only if external charges exist
  if ( .not. allocated(quick_api%ptchg_crd)) allocate(quick_api%ptchg_crd(4,quick_api%nxt_ptchg), stat=ierr)

  call realloc(quick_molspec,ierr)

  if(isgrad) then
    if ( .not. allocated(quick_api%ptchg_grad)) allocate(quick_api%ptchg_grad(3,quick_api%nxt_ptchg), stat=ierr)
    quick_api%ptchg_grad =0.0d0
    call realloc(quick_qm_struct,ierr)
  endif
  

end subroutine allocate_point_charge

! allocate memory for point charges and gradients
subroutine deallocate_point_charge(isgrad,ierr)

  use quick_calculated_module
  implicit none
  logical, intent(in) :: isgrad
  integer, intent(inout) :: ierr

  if ( allocated(quick_api%ptchg_crd))     deallocate(quick_api%ptchg_crd, stat=ierr)

  if(isgrad) then
    if ( allocated(quick_api%ptchg_grad))     deallocate(quick_api%ptchg_grad, stat=ierr)  
  endif

end subroutine deallocate_point_charge

! returns quick qm energy
subroutine get_quick_energy(coords, nxt_ptchg, ptchg_crd, energy, ierr)

  use quick_molspec_module, only: quick_molspec
  implicit none
  
  integer, intent(in)           :: nxt_ptchg
  double precision, intent(in)  :: coords(3,quick_api%natoms)
  double precision, intent(in)  :: ptchg_crd(4,nxt_ptchg)
  double precision, intent(out) :: energy
  integer, intent(out) :: ierr
  ierr=0

  ! assign passed parameter values into quick_api struct
  quick_api%nxt_ptchg = nxt_ptchg
  quick_api%coords        = coords
  quick_api%density_ready = .false.

  ! set number of external atoms in quick_molspec
  quick_molspec%nextatom  = quick_api%nxt_ptchg

  if(quick_api%nxt_ptchg .gt. 0) then
    call allocate_point_charge(.false., ierr)
    quick_api%ptchg_crd     = ptchg_crd
  endif

  call run_quick(quick_api,ierr)
  if (ierr == 0) quick_api%density_ready = .true.

  ! send back total energy and charges
  energy = quick_api%tot_ene

  if(quick_api%nxt_ptchg .gt. 0) call deallocate_point_charge(.false., ierr)

end subroutine get_quick_energy


! calculates and returns energy, gradients and point charge gradients
subroutine get_quick_energy_gradients(coords, nxt_ptchg, ptchg_crd, &
           energy, gradients, ptchg_grad, ierr)

  use quick_molspec_module, only: quick_molspec
  implicit none

  integer, intent(in)             :: nxt_ptchg 
  double precision, intent(in)    :: coords(3,quick_api%natoms)
  double precision, intent(in)    :: ptchg_crd(4,nxt_ptchg)
  double precision, intent(out)   :: energy
  double precision, intent(out)   :: gradients(3,quick_api%natoms)
  double precision, intent(out) :: ptchg_grad(3,nxt_ptchg)
  integer, intent(out) :: ierr
  ierr=0

  ! assign passed parameter values into quick_api struct
  quick_api%coords         = coords
  quick_api%nxt_ptchg = nxt_ptchg
  quick_api%density_ready = .false.

  ! set number of external atoms in quick_molspec
  quick_molspec%nextatom  = quick_api%nxt_ptchg

  if(quick_api%nxt_ptchg .gt. 0) then
    call allocate_point_charge(.true., ierr)
    quick_api%ptchg_crd = ptchg_crd
  endif

  call run_quick(quick_api,ierr)
  if (ierr == 0) quick_api%density_ready = .true.

  ! send back total energy, gradients and point charge gradients
  energy     = quick_api%tot_ene
  gradients     = quick_api%gradient

  if(quick_api%nxt_ptchg .gt. 0) then
    ptchg_grad = quick_api%ptchg_grad
    call deallocate_point_charge(.true., ierr)
  endif

end subroutine get_quick_energy_gradients


! evaluates no-I/O electrostatic properties at caller-supplied probe points.
! Probe coordinates are in bohr. Returned properties are in QUICK atomic units:
! ESP V(C), E_i(C)=-dV/dC_i, and G_ij(C)=dE_i/dC_j.
subroutine get_quick_oeprop(npoints, probe_xyz_bohr, ierr, esp, efield, efg)

  use quick_oeproperties_module, only: compute_oeprop_values

  implicit none

  integer, intent(in) :: npoints
  double precision, intent(in) :: probe_xyz_bohr(3,npoints)
  integer, intent(out) :: ierr
  double precision, intent(out), optional :: esp(npoints)
  double precision, intent(out), optional :: efield(3,npoints)
  double precision, intent(out), optional :: efg(3,3,npoints)

  ierr = 0

  if (npoints <= 0) then
    ierr = 46
    return
  endif

  if (.not. quick_api%density_ready) then
    ierr = 47
    return
  endif

  if (.not. (present(esp) .or. present(efield) .or. present(efg))) then
    ierr = 48
    return
  endif

  call compute_oeprop_values(npoints, probe_xyz_bohr, esp, efield, efg)

end subroutine get_quick_oeprop


! Reference finite-difference total QUICK-MBX gradients for tiny water
! environments.  Coordinates are Angstrom, displacements are bohr, and
! gradients are dE/dR in hartree/bohr.  This routine is intentionally for
! validation and CPU smoke dynamics, not production analytic MD.
subroutine get_quick_mbx_energy_gradients_fd_water(qm_coords, nwaters, mbx_coords, &
           json_file, fd_step_bohr, energy, qm_grad, mbx_grad, ierr)

  use quick_constants_module, only: BOHRS_TO_A

  implicit none

  integer, intent(in) :: nwaters
  double precision, intent(in) :: qm_coords(3,quick_api%natoms)
  double precision, intent(in) :: mbx_coords(3,3*nwaters)
  character(len=*), intent(in) :: json_file
  double precision, intent(in) :: fd_step_bohr
  double precision, intent(out) :: energy
  double precision, intent(out) :: qm_grad(3,quick_api%natoms)
  double precision, intent(out) :: mbx_grad(3,3*nwaters)
  integer, intent(out) :: ierr

  integer :: iatom, ixyz, nmbx_atoms
  double precision :: disp_ang, eplus, eminus
  double precision :: qm_p(3,quick_api%natoms), qm_m(3,quick_api%natoms)
  double precision :: mbx_p(3,3*nwaters), mbx_m(3,3*nwaters)

  ierr = 0
  energy = 0.0d0
  qm_grad(:,:) = 0.0d0
  mbx_grad(:,:) = 0.0d0

  if (nwaters <= 0 .or. fd_step_bohr <= 0.0d0) then
    ierr = 45
    return
  endif

  nmbx_atoms = 3*nwaters
  disp_ang = fd_step_bohr*BOHRS_TO_A

  call evaluate_quick_mbx_water_energy(qm_coords,nwaters,mbx_coords,json_file,energy,ierr)
  if (ierr /= 0) return

  do iatom=1,quick_api%natoms
    do ixyz=1,3
      qm_p(:,:) = qm_coords(:,:)
      qm_m(:,:) = qm_coords(:,:)
      qm_p(ixyz,iatom) = qm_p(ixyz,iatom) + disp_ang
      qm_m(ixyz,iatom) = qm_m(ixyz,iatom) - disp_ang
      call evaluate_quick_mbx_water_energy(qm_p,nwaters,mbx_coords,json_file,eplus,ierr)
      if (ierr /= 0) return
      call evaluate_quick_mbx_water_energy(qm_m,nwaters,mbx_coords,json_file,eminus,ierr)
      if (ierr /= 0) return
      qm_grad(ixyz,iatom) = (eplus - eminus)/(2.0d0*fd_step_bohr)
    enddo
  enddo

  do iatom=1,nmbx_atoms
    do ixyz=1,3
      mbx_p(:,:) = mbx_coords(:,:)
      mbx_m(:,:) = mbx_coords(:,:)
      mbx_p(ixyz,iatom) = mbx_p(ixyz,iatom) + disp_ang
      mbx_m(ixyz,iatom) = mbx_m(ixyz,iatom) - disp_ang
      call evaluate_quick_mbx_water_energy(qm_coords,nwaters,mbx_p,json_file,eplus,ierr)
      if (ierr /= 0) return
      call evaluate_quick_mbx_water_energy(qm_coords,nwaters,mbx_m,json_file,eminus,ierr)
      if (ierr /= 0) return
      mbx_grad(ixyz,iatom) = (eplus - eminus)/(2.0d0*fd_step_bohr)
    enddo
  enddo

  ! Leave MBX in the undisplaced geometry for callers that reuse the job.
  call set_quick_mbx_water_system(nwaters,mbx_coords,json_file,ierr)

end subroutine get_quick_mbx_energy_gradients_fd_water


subroutine evaluate_quick_mbx_water_energy(qm_coords, nwaters, mbx_coords, json_file, energy, ierr)

  implicit none

  integer, intent(in) :: nwaters
  double precision, intent(in) :: qm_coords(3,quick_api%natoms)
  double precision, intent(in) :: mbx_coords(3,3*nwaters)
  character(len=*), intent(in) :: json_file
  double precision, intent(out) :: energy
  integer, intent(out) :: ierr

  double precision, allocatable :: no_point_charges(:,:)

  ierr = 0
  energy = 0.0d0

  call clear_quick_mbx_system(ierr)
  if (ierr /= 0) return
  call set_quick_mbx_water_system(nwaters,mbx_coords,json_file,ierr)
  if (ierr /= 0) return

  allocate(no_point_charges(4,0), stat=ierr)
  if (ierr /= 0) return
  call get_quick_energy(qm_coords,0,no_point_charges,energy,ierr)
  if (allocated(no_point_charges)) deallocate(no_point_charges)

end subroutine evaluate_quick_mbx_water_energy


! runs quick, partially resembles quick main program
subroutine run_quick(self,ierr)

  use quick_timer_module
  use quick_method_module, only: quick_method
  use quick_files_module
  use quick_calculated_module, only: quick_qm_struct
  use quick_gridpoints_module, only: quick_dft_grid, deform_dft_grid
  use quick_cutoff_module, only: schwarzoff
  use quick_exception_module
  use quick_eri_cshell_module, only: getEriPrecomputables
  use quick_grad_cshell_module, only: cshell_gradient
  use quick_grad_oshell_module, only: oshell_gradient
  use quick_optimizer_module
  use quick_sad_guess_module, only: getSadGuess
  use quick_molden_module, only: quick_molden, initializeExport, exportCoordinates, exportBasis, &
      exportMO, exportSCF, exportOPT


#ifdef CEW 
  use quick_cew_module
#endif

#ifdef MPIV
  use quick_mpi_module
#endif

  implicit none

  type(quick_api_type), intent(inout) :: self
  integer, intent(out) :: ierr
  integer :: i, j, k
  logical :: failed = .false.
  ierr=0

  ! trun off extcharges in quick_method is external charges become zero
  if(quick_api%nxt_ptchg .eq. 0) then
    quick_method%extCharges = .false.
  else
    quick_method%extCharges = .true.
  endif

  ! print step into quick output file
  call print_step(self,ierr)

  ! if dft is requested, make sure to delete dft grid variables from previous
  ! the md step before proceeding
  if(( self%step .gt. 1 ) .and. (quick_method%DFT &
#ifdef CEW
  .or. quick_cew%use_cew &
#endif
   )) then
    call deform_dft_grid(quick_dft_grid)
  endif

  ! set molecular information into quick_molspec
  SAFE_CALL(set_quick_molspecs(quick_api,ierr))

  ! start the timer for initial guess
  RECORD_TIME(timer_begin%TIniGuess)

  ! we will reuse density matrix for steps above 1. For the 1st step, we should
  ! read basis file and run SAD guess.
  if(self%firstStep .or. (.not. self%reuse_dmx)) then

    ! perform the initial guess
    if (quick_method%SAD) SAFE_CALL(getSadGuess(ierr))

    ! assign basis functions
    SAFE_CALL(getMol(ierr))

    self%firstStep = .false.

  endif

  ! pre-calculate 2 index coefficients and schwarz cutoff criteria
  if(.not.quick_method%opt) then
    call getEriPrecomputables
    call schwarzoff
  endif

#if defined(GPU) || defined(MPIV_GPU)
  ! upload molecular and basis information to gpu
  if(.not.quick_method%opt) call gpu_upload_molspecs(ierr)

#ifdef CEW
  call upload(quick_cew, ierr)
#endif
#endif

  if(write_molden) then
#ifdef MPIV
     if(master) then
#endif
     write(moldenFileName, '(A, ".molden.", I0)') trim(baseinFileName), quick_api%step
     call initializeExport(quick_molden, ierr)
#ifdef MPIV
     endif
#endif
  endif

  ! stop the timer for initial guess
  RECORD_TIME(timer_end%TIniGuess)

#if defined(MPIV_GPU)
    timer_begin%T2elb = timer_end%T2elb
    call mgpu_get_2elb_time(timer_end%T2elb)
    timer_cumer%T2elb = timer_cumer%T2elb+timer_end%T2elb-timer_begin%T2elb
#endif

  timer_cumer%TIniGuess = timer_cumer%TIniGuess+timer_end%TIniGuess-timer_begin%TIniGuess &
                           - (timer_end%T2elb-timer_begin%T2elb)

  ! compute energy
  if ( .not. quick_method%opt .and. .not. quick_method%grad) then
    SAFE_CALL(getEnergy(.false.,ierr))
  endif

  ! compute gradients
  if (.not.quick_method%opt .and. quick_method%grad) then
      if (quick_method%UNRST) then
          SAFE_CALL(oshell_gradient(ierr))
      else
          SAFE_CALL(cshell_gradient(ierr))
      endif
  endif

  ! run optimization
  if (quick_method%opt) then
      if (quick_method%usedlfind) then

#ifdef MPIV
          SAFE_CALL(dl_find(ierr, master))   ! DLC
#else 
          SAFE_CALL(dl_find(ierr, .true.))   ! DLC
#endif
      else
          SAFE_CALL(lopt(ierr))         ! Cartesian
      endif
  endif

  if(write_molden) then
#ifdef MPIV
     if(master) then
#endif
     call exportCoordinates(quick_molden, ierr)
     call exportBasis(quick_molden, ierr)
     call exportMO(quick_molden, ierr)
     if (quick_method%opt) then
        call exportSCF(quick_molden, ierr)
        call exportOPT(quick_molden, ierr)
     end if
#ifdef MPIV
     endif
#endif
  endif

#if defined(GPU) || defined(MPIV_GPU)
      if (quick_method%bGPU) then
        call gpu_cleanup()
      endif
#endif

#ifdef MPIV
  if(master) then
#endif

  ! calculate charges
  if (quick_method%dipole) call dipole

! save the results in quick_api struct
  self%tot_ene = quick_qm_struct%Etot

! save gradients and point charge gradients in quick_api struct.
! Note that quick_qm_struct saves both gradients in vector formats and
! we should organize them back into matrix format.
  if (quick_method%grad) then
    k=1
    do i=1,self%natoms
      do j=1,3
        self%gradient(j,i) = quick_qm_struct%gradient(k)
        k=k+1
      enddo
    enddo

    if (quick_method%extCharges .and. self%nxt_ptchg .gt. 0) then
      k=1
      do i=1,self%nxt_ptchg
        do j=1,3
          self%ptchg_grad(j,i) = quick_qm_struct%ptchg_gradient(k)
          k=k+1
        enddo
      enddo
    endif
  endif

#ifdef MPIV
  endif

  ! broadcast results from master to slaves
  call broadcast_quick_mpi_results(self,ierr)
#endif

  ! increase internal quick step by one
  quick_api%step = quick_api%step + 1

end subroutine run_quick


! this subroutine will print the step into quick output file
subroutine print_step(self,ierr)

  use quick_files_module
#ifdef MPIV
  use quick_mpi_module
#endif

  implicit none
  type (quick_api_type) :: self
  integer, intent(inout) :: ierr

  ! print step into quick output file
#ifdef MPIV
  if(master) then
#endif

  write(iOutFile, '(A1)') ' '
  write(iOutFile, '(1x,A16,1x,I12)') '@ Running Step :',self%step
  write(iOutFile, '(A1)') ' '

#ifdef MPIV
  endif
#endif

end subroutine print_step


! this rubroutine will set atom number, types and number of external atoms
! based on the information provided through library api
subroutine set_quick_molspecs(self,ierr)

  use quick_files_module
  use quick_constants_module, only: BOHRS_TO_A, BOHRS_TO_A_AMBER, symbol
  use quick_molspec_module, only : quick_molspec, xyz

#ifdef CEW
  use quick_cew_module, only: quick_cew
#endif

  implicit none
  type (quick_api_type) :: self
  integer :: i, j
  integer, intent(inout) :: ierr
  double precision :: A_TO_BOHRS

  A_TO_BOHRS = 1.0D0 / BOHRS_TO_A

#ifdef CEW
  if (quick_cew%use_cew) then
    ! Amber-consistent conversion factors for use with CEw
    ! Parts of the Ewald are performed by sander, cew, and quick;
    ! they need to use consistent conversions in order for real
    ! and reciprocal space interactions to properly cancel.
    A_TO_BOHRS = 1.0D0 / BOHRS_TO_A_AMBER
  endif
#endif

  ! pass the step id to quick_files_module
  wrtStep = self%step

  ! save the atom types
  do i=1, self%natm_type
    quick_molspec%atom_type_sym(i) = symbol(self%atm_type_id(i))
  enddo

  ! save the coordinates and atomic numbers
  do i=1, self%natoms
    quick_molspec%iattype(i) = self%atomic_numbers(i)
    do j=1, 3
       xyz(j,i) = self%coords(j,i) * A_TO_BOHRS
    enddo
  enddo

  quick_molspec%xyz => xyz

  ! save the external point charges and coordinates
  if(self%nxt_ptchg .gt. 0) then
    do i=1, self%nxt_ptchg
      do j=1,3
        quick_molspec%extxyz(j,i) = self%ptchg_crd(j,i) * A_TO_BOHRS
      enddo
      quick_molspec%extchg(i)     = self%ptchg_crd(4,i)
    enddo
  endif

end subroutine set_quick_molspecs

#if defined(GPU) || defined(MPIV_GPU)

! uploads molecular information into gpu
subroutine gpu_upload_molspecs(ierr)

  use quick_molspec_module, only : quick_molspec
  use quick_basis_module

  implicit none
  integer, intent(inout) :: ierr

  call gpu_setup(quick_molspec%natom,nbasis, quick_molspec%nElec, quick_molspec%imult, &
       quick_molspec%molchg, quick_molspec%iAtomType)
  call gpu_upload_xyz(quick_molspec%xyz)
  call gpu_upload_atom_and_chg(quick_molspec%iattype, quick_molspec%chg)

  call gpu_upload_basis(nshell, nprim, jshell, jbasis, maxcontract, &
  ncontract, itype, aexp, dcoeff, &
  quick_basis%first_basis_function, quick_basis%last_basis_function, &
  quick_basis%first_shell_basis_function, quick_basis%last_shell_basis_function, &
  quick_basis%ncenter, quick_basis%kstart, quick_basis%katom, &
  quick_basis%ktype, quick_basis%kprim, quick_basis%kshell,quick_basis%Ksumtype, &
  quick_basis%Qnumber, quick_basis%Qstart, quick_basis%Qfinal, quick_basis%Qsbasis, quick_basis%Qfbasis, &
  quick_basis%gccoeff, quick_basis%cons, quick_basis%gcexpo, quick_basis%KLMN)

  call gpu_upload_cutoff_matrix(Ycutoff, cutPrim)

  call gpu_upload_oei(quick_molspec%nextatom, quick_molspec%extxyz, quick_molspec%extchg, ierr)

end subroutine gpu_upload_molspecs

#endif


#ifdef MPIV

! sets mpi variables in quick api
subroutine set_quick_mpi(mpi_rank, mpi_size, ierr)

  use quick_mpi_module

  implicit none

  integer, intent(in) :: mpi_rank, mpi_size
  integer, intent(out) :: ierr

  ! save information in quick_mpi module
  mpirank    = mpi_rank
  mpisize    = mpi_size
  libMPIMode = .true.
  ierr = 0
  

end subroutine set_quick_mpi

! broadcasts results from master to slaves

subroutine broadcast_quick_mpi_results(self,ierr)
  use mpi

  implicit none

  type(quick_api_type), intent(inout) :: self
  integer :: mpierror
  integer, intent(inout) :: ierr

  call MPI_BCAST(self%tot_ene,1,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
  call MPI_BCAST(self%gradient,3*self%natoms,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
  call MPI_BCAST(self%ptchg_grad,3*self%nxt_ptchg,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)

end subroutine

#endif

! fialize quick and deallocate memory of quick_api internal arrays
subroutine delete_quick_job(ierr)

  use quick_files_module
  use quick_mpi_module
  use quick_exception_module
  use quick_method_module
  use quick_mbx_module, only: quick_mbx_finalize

  implicit none
  integer, intent(out) :: ierr
  ierr=0

#if defined(GPU) || defined(MPIV_GPU)
  call delete(quick_method, ierr)
  call gpu_deallocate_scratch(.true.)
#if defined(MPIV_GPU)
  SAFE_CALL(delete_mgpu_setup(ierr))
#endif
  SAFE_CALL(gpu_delete(ierr))
#endif

  call quick_mbx_finalize()

  ! finalize quick
  call finalize(iOutFile,ierr,1)

  ! deallocate memory
  call delete_quick_api_type(quick_api,ierr)

end subroutine delete_quick_job


! deallocates memory for quick_api_type variable
subroutine delete_quick_api_type(self,ierr)

  implicit none
  type(quick_api_type), intent(inout) :: self
  integer, intent(inout) :: ierr

  if ( allocated(self%atm_type_id))    deallocate(self%atm_type_id, stat=ierr)
  if ( allocated(self%atomic_numbers)) deallocate(self%atomic_numbers, stat=ierr)
  if ( allocated(self%coords))         deallocate(self%coords, stat=ierr)
  if ( allocated(self%gradient))          deallocate(self%gradient, stat=ierr)

end subroutine delete_quick_api_type

end module quick_api_module
