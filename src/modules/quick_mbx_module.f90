#include "util.fh"
!---------------------------------------------------------------------!
! Optional QUICK-MBX polarizable embedding adapter.                    !
!                                                                     !
! QUICK owns QM density response and one-electron operators; MBX owns  !
! MB-pol/MB-nrg site electrostatics, induced dipoles, and many-body    !
! force-field terms. Coordinates stored in this module are in bohr.    !
!_____________________________________________________________________!

module quick_mbx_module
   use iso_c_binding, only: c_char, c_double, c_int, c_null_char, c_ptr, c_loc
   use quick_constants_module, only: BOHRS_TO_A, A_TO_BOHRS, KCAL_TO_AU

   implicit none
   private

   public :: quick_mbx_available, quick_mbx_active, quick_mbx_has_sites
   public :: quick_mbx_set_sites, quick_mbx_clear, quick_mbx_send_qm_field
   public :: quick_mbx_initialize_system, quick_mbx_finalize
   public :: quick_mbx_refresh_energy, quick_mbx_get_energy_terms
   public :: quick_mbx_update_scf_operator
   public :: quick_mbx_get_site_data, quick_mbx_get_induced_dipoles_au
   public :: quick_mbx_set_real_coordinates, quick_mbx_get_real_atom_gradient
   public :: quick_mbx_redistribute_site_gradient
   public :: quick_mbx_add_qm_gradient, quick_mbx_get_coupled_real_atom_gradient
   public :: quick_mbx_get_cross_qm_gradient
   public :: quick_mbx_initialize_cross_residual, quick_mbx_cross_residual_active

   logical :: mbx_active = .false.
   logical :: mbx_initialized = .false.
   logical :: mbx_qm_field_ready = .false.
   integer :: mbx_nsites = 0
   integer :: mbx_natoms_real = 0
   double precision, allocatable :: mbx_site_xyz_bohr(:,:)
   double precision, allocatable :: mbx_real_xyz_ang(:,:)
   double precision, allocatable :: mbx_phi_au(:)
   double precision, allocatable :: mbx_efield_au(:,:)
   double precision, allocatable :: mbx_charge_e(:)
   double precision, allocatable :: mbx_mu_ind_ea(:,:)
   double precision :: mbx_energy_kcalmol = 0.0d0
   double precision :: mbx_external_energy_kcalmol = 0.0d0
   double precision :: mbx_energy_add_au = 0.0d0
   double precision :: mbx_perm_total_au = 0.0d0
   double precision :: mbx_perm_classical_au = 0.0d0
   double precision :: mbx_perm_electronic_au = 0.0d0
   double precision :: mbx_ind_total_au = 0.0d0
   double precision :: mbx_ind_classical_au = 0.0d0
   double precision :: mbx_ind_electronic_au = 0.0d0
   double precision :: mbx_external_perm_half_residual_au = 0.0d0
   logical :: mbx_cross_active = .false.
   integer :: mbx_cross_natoms_qm = 0
   integer :: mbx_cross_natoms_total = 0
   double precision :: mbx_cross_terms_au(3) = 0.0d0
   double precision, allocatable :: mbx_cross_qm_grad_au(:,:,:)
   double precision, allocatable :: mbx_cross_mb_grad_au(:,:,:)
   double precision :: mbx_perm_operator_scale = 1.0d0
   double precision :: mbx_ind_operator_scale = 1.0d0

   double precision, parameter :: ESP_AU_TO_MBX = 1.0d0 / BOHRS_TO_A
   double precision, parameter :: EFIELD_AU_TO_MBX = 1.0d0 / (BOHRS_TO_A*BOHRS_TO_A)
   double precision, parameter :: DIPOLE_MBX_TO_AU = A_TO_BOHRS

#ifdef MBX
   interface
      subroutine mbx_initialize_system(coords, nat_monomers, at_names, monomers, nmon, json_file) &
            bind(C,name="initialize_system_py_")
         import :: c_char, c_double, c_int, c_ptr
         real(c_double), intent(in) :: coords(*)
         integer(c_int), intent(in) :: nat_monomers(*)
         type(c_ptr), intent(in) :: at_names(*)
         type(c_ptr), intent(in) :: monomers(*)
         integer(c_int), intent(in) :: nmon
         character(kind=c_char), intent(in) :: json_file(*)
      end subroutine mbx_initialize_system

      subroutine mbx_get_energy(coords, natoms, energy) bind(C,name="get_energy_")
         import :: c_double, c_int
         real(c_double), intent(in) :: coords(*)
         integer(c_int), intent(in) :: natoms
         real(c_double), intent(out) :: energy
      end subroutine mbx_get_energy

      subroutine mbx_get_energy_g(coords, natoms, energy, grads) bind(C,name="get_energy_g_")
         import :: c_double, c_int
         real(c_double), intent(in) :: coords(*)
         integer(c_int), intent(in) :: natoms
         real(c_double), intent(out) :: energy
         real(c_double), intent(out) :: grads(*)
      end subroutine mbx_get_energy_g

      subroutine mbx_get_external_field_contribution_to_energy(energy) &
            bind(C,name="get_external_field_contribution_to_energy_")
         import :: c_double
         real(c_double), intent(out) :: energy
      end subroutine mbx_get_external_field_contribution_to_energy

      subroutine mbx_get_charges(charges) bind(C,name="get_charges_")
         import :: c_double
         real(c_double), intent(out) :: charges(*)
      end subroutine mbx_get_charges

      subroutine mbx_get_xyz(coords) bind(C,name="get_xyz_")
         import :: c_double
         real(c_double), intent(out) :: coords(*)
      end subroutine mbx_get_xyz

      subroutine mbx_set_real_xyz(coords, natoms) bind(C,name="set_real_xyz_")
         import :: c_double, c_int
         real(c_double), intent(in) :: coords(*)
         integer(c_int), intent(in) :: natoms
      end subroutine mbx_set_real_xyz

      subroutine mbx_set_potential_and_electric_field_on_sites(phi, ef, nsites) &
            bind(C,name="set_potential_and_electric_field_on_sites_")
         import :: c_double, c_int
         real(c_double), intent(in) :: phi(*)
         real(c_double), intent(in) :: ef(*)
         integer(c_int), intent(in) :: nsites
      end subroutine mbx_set_potential_and_electric_field_on_sites

      subroutine mbx_get_induced_dipoles(mu_ind) bind(C,name="get_induced_dipoles_")
         import :: c_double
         real(c_double), intent(out) :: mu_ind(*)
      end subroutine mbx_get_induced_dipoles

      subroutine mbx_redistribute_gradients(grads, nsites) bind(C,name="redistribute_gradients_")
         import :: c_double, c_int
         real(c_double), intent(inout) :: grads(*)
         integer(c_int), intent(in) :: nsites
      end subroutine mbx_redistribute_gradients

      subroutine mbx_finalize_system() bind(C,name="finalize_system_")
      end subroutine mbx_finalize_system

      subroutine mbx_cross_initialize(coords,nat_monomers,at_names,monomers,regions,nmon,json_file,ierr) &
            bind(C,name="quick_mbx_cross_residual_initialize_")
         import :: c_char, c_double, c_int, c_ptr
         real(c_double), intent(in) :: coords(*)
         integer(c_int), intent(in) :: nat_monomers(*)
         type(c_ptr), intent(in) :: at_names(*)
         type(c_ptr), intent(in) :: monomers(*)
         integer(c_int), intent(in) :: regions(*)
         integer(c_int), intent(in) :: nmon
         character(kind=c_char), intent(in) :: json_file(*)
         integer(c_int), intent(out) :: ierr
      end subroutine mbx_cross_initialize

      subroutine mbx_cross_evaluate(coords,natoms,energies,gradients,ierr) &
            bind(C,name="quick_mbx_cross_residual_evaluate_")
         import :: c_double, c_int
         real(c_double), intent(in) :: coords(*)
         integer(c_int), intent(in) :: natoms
         real(c_double), intent(out) :: energies(*)
         real(c_double), intent(out) :: gradients(*)
         integer(c_int), intent(out) :: ierr
      end subroutine mbx_cross_evaluate

      subroutine mbx_cross_finalize() bind(C,name="quick_mbx_cross_residual_finalize_")
      end subroutine mbx_cross_finalize
   end interface
#endif

contains

   logical function quick_mbx_available()
#ifdef MBX
      quick_mbx_available = .true.
#else
      quick_mbx_available = .false.
#endif
   end function quick_mbx_available

   logical function quick_mbx_active()
      quick_mbx_active = mbx_active
   end function quick_mbx_active

   logical function quick_mbx_cross_residual_active()
      quick_mbx_cross_residual_active = mbx_cross_active
   end function quick_mbx_cross_residual_active

   logical function quick_mbx_has_sites()
      quick_mbx_has_sites = allocated(mbx_site_xyz_bohr) .and. mbx_nsites > 0
   end function quick_mbx_has_sites

   subroutine quick_mbx_set_sites(nsites,xyz_sites_bohr)
      integer, intent(in) :: nsites
      double precision, intent(in) :: xyz_sites_bohr(:,:)

      if (mbx_initialized) then
         call quick_mbx_finalize()
      else
         call quick_mbx_clear()
      endif
      mbx_nsites = nsites
      allocate(mbx_site_xyz_bohr(3,mbx_nsites))
      allocate(mbx_phi_au(mbx_nsites))
      allocate(mbx_efield_au(3,mbx_nsites))
      allocate(mbx_mu_ind_ea(3,mbx_nsites))
      mbx_site_xyz_bohr(:,:) = xyz_sites_bohr(:,1:mbx_nsites)
      mbx_phi_au(:) = 0.0d0
      mbx_efield_au(:,:) = 0.0d0
      mbx_mu_ind_ea(:,:) = 0.0d0
      mbx_active = .true.
   end subroutine quick_mbx_set_sites

   subroutine quick_mbx_clear()
      if (allocated(mbx_site_xyz_bohr)) deallocate(mbx_site_xyz_bohr)
      if (allocated(mbx_real_xyz_ang)) deallocate(mbx_real_xyz_ang)
      if (allocated(mbx_phi_au)) deallocate(mbx_phi_au)
      if (allocated(mbx_efield_au)) deallocate(mbx_efield_au)
      if (allocated(mbx_charge_e)) deallocate(mbx_charge_e)
      if (allocated(mbx_mu_ind_ea)) deallocate(mbx_mu_ind_ea)
      if (allocated(mbx_cross_qm_grad_au)) deallocate(mbx_cross_qm_grad_au)
      if (allocated(mbx_cross_mb_grad_au)) deallocate(mbx_cross_mb_grad_au)
      mbx_nsites = 0
      mbx_natoms_real = 0
      mbx_energy_kcalmol = 0.0d0
      mbx_external_energy_kcalmol = 0.0d0
      mbx_energy_add_au = 0.0d0
      mbx_perm_total_au = 0.0d0
      mbx_perm_classical_au = 0.0d0
      mbx_perm_electronic_au = 0.0d0
      mbx_ind_total_au = 0.0d0
      mbx_ind_classical_au = 0.0d0
      mbx_ind_electronic_au = 0.0d0
      mbx_external_perm_half_residual_au = 0.0d0
      mbx_cross_active = .false.
      mbx_cross_natoms_qm = 0
      mbx_cross_natoms_total = 0
      mbx_cross_terms_au(:) = 0.0d0
      mbx_initialized = .false.
      mbx_qm_field_ready = .false.
      mbx_active = .false.
   end subroutine quick_mbx_clear

   subroutine quick_mbx_finalize()
#ifdef MPIV
      use quick_mpi_module, only: master
#endif
      implicit none

#ifdef MBX
#ifdef MPIV
      if (master .and. mbx_cross_active) call mbx_cross_finalize()
      if (master .and. mbx_initialized) call mbx_finalize_system()
#else
      if (mbx_cross_active) call mbx_cross_finalize()
      if (mbx_initialized) call mbx_finalize_system()
#endif
#endif
      call quick_mbx_clear()
   end subroutine quick_mbx_finalize

   subroutine quick_mbx_initialize_system(natoms,nsites,nmon,nat_monomers,coords_ang, &
         atom_names,monomer_names,json_file,ierr)
#ifdef MPIV
      use mpi
      use quick_mpi_module, only: master, mpierror
#endif
      implicit none

      integer, intent(in) :: natoms, nsites, nmon
      integer, intent(in) :: nat_monomers(nmon)
      double precision, intent(in) :: coords_ang(3,natoms)
      character(len=*), intent(in) :: atom_names(natoms)
      character(len=*), intent(in) :: monomer_names(nmon)
      character(len=*), intent(in) :: json_file
      integer, intent(inout) :: ierr

#ifdef MBX
      integer :: iatom, imon, ixyz
      integer :: total_atoms
      integer, parameter :: c_name_len = 16
      integer, parameter :: c_path_len = 256
      integer(c_int) :: c_natoms, c_nmon
      integer(c_int), allocatable :: c_nat_monomers(:)
      character(kind=c_char), allocatable, target :: c_atom_names(:,:), c_monomer_names(:,:)
      type(c_ptr), allocatable :: c_atom_name_ptrs(:), c_monomer_name_ptrs(:)
      character(kind=c_char) :: c_json_file(c_path_len)
      double precision, allocatable :: coords_flat(:)
      logical :: json_exists
#endif

      if (ierr /= 0) return

#ifdef MBX
      if (natoms <= 0 .or. nsites <= 0 .or. nmon <= 0) then
         ierr = 45
         return
      endif
      total_atoms = sum(nat_monomers)
      if (total_atoms /= natoms) then
         ierr = 45
         return
      endif
      if (len_trim(json_file) > c_path_len-1) then
         ierr = 45
         return
      endif
      json_exists=.false.
#ifdef MPIV
      if (master) inquire(file=trim(json_file),exist=json_exists)
      call MPI_BCAST(json_exists,1,mpi_logical,0,MPI_COMM_WORLD,mpierror)
#else
      inquire(file=trim(json_file),exist=json_exists)
#endif
      if (.not. json_exists) then
         ierr=55
         return
      endif

      if (mbx_initialized) call quick_mbx_finalize()
      call quick_mbx_clear()

      allocate(coords_flat(3*natoms))
      allocate(c_nat_monomers(nmon))
      allocate(c_atom_names(c_name_len,natoms))
      allocate(c_monomer_names(c_name_len,nmon))
      allocate(c_atom_name_ptrs(natoms))
      allocate(c_monomer_name_ptrs(nmon))

      do iatom=1,natoms
         do ixyz=1,3
            coords_flat(3*(iatom-1)+ixyz) = coords_ang(ixyz,iatom)
         enddo
         call set_c_string(atom_names(iatom),c_atom_names(:,iatom))
         c_atom_name_ptrs(iatom) = c_loc(c_atom_names(1,iatom))
      enddo
      do imon=1,nmon
         c_nat_monomers(imon) = int(nat_monomers(imon),c_int)
         call set_c_string(monomer_names(imon),c_monomer_names(:,imon))
         c_monomer_name_ptrs(imon) = c_loc(c_monomer_names(1,imon))
      enddo
      call set_c_string(json_file,c_json_file)

      c_natoms = int(natoms,c_int)
      c_nmon = int(nmon,c_int)
#ifdef MPIV
      if (master) then
#endif
         call mbx_initialize_system(coords_flat,c_nat_monomers,c_atom_name_ptrs,c_monomer_name_ptrs,c_nmon,c_json_file)
#ifdef MPIV
      endif
#endif

      mbx_natoms_real = natoms
      mbx_nsites = nsites
      allocate(mbx_real_xyz_ang(3,mbx_natoms_real))
      allocate(mbx_site_xyz_bohr(3,mbx_nsites))
      allocate(mbx_phi_au(mbx_nsites))
      allocate(mbx_efield_au(3,mbx_nsites))
      allocate(mbx_charge_e(mbx_nsites))
      allocate(mbx_mu_ind_ea(3,mbx_nsites))
      mbx_real_xyz_ang(:,:) = coords_ang(:,:)
      mbx_qm_field_ready = .false.
      mbx_phi_au(:) = 0.0d0
      mbx_efield_au(:,:) = 0.0d0
      mbx_charge_e(:) = 0.0d0
      mbx_mu_ind_ea(:,:) = 0.0d0

      mbx_initialized = .true.
      mbx_active = .true.
      call quick_mbx_refresh_site_data(ierr)

      deallocate(coords_flat)
      deallocate(c_nat_monomers)
      deallocate(c_atom_names)
      deallocate(c_monomer_names)
      deallocate(c_atom_name_ptrs)
      deallocate(c_monomer_name_ptrs)
#else
      ierr = 44
#endif
   end subroutine quick_mbx_initialize_system

   subroutine quick_mbx_initialize_cross_residual(natoms,nmon,nat_monomers,coords_ang, &
         atom_names,monomer_names,regions,json_file,ierr)
#ifdef MPIV
      use mpi
      use quick_mpi_module, only: master, mpierror
#endif
      implicit none

      integer, intent(in) :: natoms, nmon
      integer, intent(in) :: nat_monomers(nmon), regions(nmon)
      double precision, intent(in) :: coords_ang(3,natoms)
      character(len=*), intent(in) :: atom_names(natoms), monomer_names(nmon)
      character(len=*), intent(in) :: json_file
      integer, intent(inout) :: ierr

#ifdef MBX
      integer, parameter :: c_name_len=16, c_path_len=256
      integer :: iatom, imon, ixyz, nqm, nmb, total_atoms
      integer(c_int) :: c_nmon, c_ierr
      integer(c_int), allocatable :: c_nat_monomers(:), c_regions(:)
      character(kind=c_char), allocatable, target :: c_atom_names(:,:), c_monomer_names(:,:)
      character(kind=c_char) :: c_json_file(c_path_len)
      type(c_ptr), allocatable :: c_atom_name_ptrs(:), c_monomer_name_ptrs(:)
      double precision, allocatable :: coords_flat(:)
      logical :: seen_mb, json_exists
#endif

      if (ierr /= 0) return
#ifdef MBX
      if (.not. mbx_initialized .or. natoms <= 0 .or. nmon < 2) then
         ierr=45
         return
      endif
      if (len_trim(json_file) > c_path_len-1) then
         ierr=45
         return
      endif
      json_exists=.false.
#ifdef MPIV
      if (master) inquire(file=trim(json_file),exist=json_exists)
      call MPI_BCAST(json_exists,1,mpi_logical,0,MPI_COMM_WORLD,mpierror)
#else
      inquire(file=trim(json_file),exist=json_exists)
#endif
      if (.not. json_exists) then
         ierr=55
         return
      endif

      total_atoms=sum(nat_monomers)
      nqm=0
      nmb=0
      seen_mb=.false.
      do imon=1,nmon
         if (nat_monomers(imon) <= 0 .or. (regions(imon) /= 0 .and. regions(imon) /= 1)) then
            ierr=45
            return
         endif
         if (regions(imon) == 0) then
            if (seen_mb) then
               ierr=45
               return
            endif
            nqm=nqm+nat_monomers(imon)
         else
            seen_mb=.true.
            nmb=nmb+nat_monomers(imon)
         endif
      enddo
      if (total_atoms /= natoms .or. nqm <= 0 .or. nmb /= mbx_natoms_real) then
         ierr=45
         return
      endif

#ifdef MPIV
      if (master .and. mbx_cross_active) call mbx_cross_finalize()
#else
      if (mbx_cross_active) call mbx_cross_finalize()
#endif
      if (allocated(mbx_cross_qm_grad_au)) deallocate(mbx_cross_qm_grad_au)
      if (allocated(mbx_cross_mb_grad_au)) deallocate(mbx_cross_mb_grad_au)
      mbx_cross_active=.false.
      mbx_cross_terms_au(:)=0.0d0

      allocate(coords_flat(3*natoms),c_nat_monomers(nmon),c_regions(nmon))
      allocate(c_atom_names(c_name_len,natoms),c_monomer_names(c_name_len,nmon))
      allocate(c_atom_name_ptrs(natoms),c_monomer_name_ptrs(nmon))
      do iatom=1,natoms
         do ixyz=1,3
            coords_flat(3*(iatom-1)+ixyz)=coords_ang(ixyz,iatom)
         enddo
         call set_c_string(atom_names(iatom),c_atom_names(:,iatom))
         c_atom_name_ptrs(iatom)=c_loc(c_atom_names(1,iatom))
      enddo
      do imon=1,nmon
         c_nat_monomers(imon)=int(nat_monomers(imon),c_int)
         c_regions(imon)=int(regions(imon),c_int)
         call set_c_string(monomer_names(imon),c_monomer_names(:,imon))
         c_monomer_name_ptrs(imon)=c_loc(c_monomer_names(1,imon))
      enddo
      call set_c_string(json_file,c_json_file)
      c_nmon=int(nmon,c_int)
      c_ierr=0_c_int
#ifdef MPIV
      if (master) call mbx_cross_initialize(coords_flat,c_nat_monomers,c_atom_name_ptrs, &
         c_monomer_name_ptrs,c_regions,c_nmon,c_json_file,c_ierr)
      call MPI_BCAST(c_ierr,1,mpi_integer,0,MPI_COMM_WORLD,mpierror)
#else
      call mbx_cross_initialize(coords_flat,c_nat_monomers,c_atom_name_ptrs, &
         c_monomer_name_ptrs,c_regions,c_nmon,c_json_file,c_ierr)
#endif
      ierr=int(c_ierr)
      if (ierr == 0) then
         mbx_cross_natoms_qm=nqm
         mbx_cross_natoms_total=natoms
         allocate(mbx_cross_qm_grad_au(3,nqm,3),mbx_cross_mb_grad_au(3,nmb,3))
         mbx_cross_qm_grad_au(:,:,:)=0.0d0
         mbx_cross_mb_grad_au(:,:,:)=0.0d0
         mbx_cross_active=.true.
      endif

      deallocate(coords_flat,c_nat_monomers,c_regions,c_atom_names,c_monomer_names)
      deallocate(c_atom_name_ptrs,c_monomer_name_ptrs)
#else
      ierr=44
#endif
   end subroutine quick_mbx_initialize_cross_residual

   subroutine quick_mbx_send_qm_field(ierr)
      use quick_oeproperties_module, only: compute_oeprop_values
#ifdef MPIV
      use mpi
      use quick_mpi_module, only: master
#endif
      implicit none

      integer, intent(inout) :: ierr
#ifdef MBX
      integer :: isite, ixyz
      integer(c_int) :: c_nsites
      double precision, allocatable :: phi_mbx(:), efield_mbx(:)
#endif

      if (.not. mbx_active) return
      if (.not. quick_mbx_has_sites()) then
         ierr = 43
         return
      endif

#ifdef MBX
      call compute_oeprop_values(mbx_nsites,mbx_site_xyz_bohr,mbx_phi_au,mbx_efield_au)

#ifdef MPIV
      if (master) then
#endif
         allocate(phi_mbx(mbx_nsites))
         allocate(efield_mbx(3*mbx_nsites))
         phi_mbx(:) = ESP_AU_TO_MBX*mbx_phi_au(:)
         do isite=1,mbx_nsites
            do ixyz=1,3
               efield_mbx(3*(isite-1)+ixyz) = EFIELD_AU_TO_MBX*mbx_efield_au(ixyz,isite)
            enddo
         enddo

         c_nsites = int(mbx_nsites,c_int)
         call mbx_set_potential_and_electric_field_on_sites(phi_mbx,efield_mbx,c_nsites)
         deallocate(phi_mbx)
         deallocate(efield_mbx)
#ifdef MPIV
      endif
#endif
      mbx_qm_field_ready = .true.
#else
      ierr = 44
#endif
   end subroutine quick_mbx_send_qm_field

   subroutine quick_mbx_refresh_energy(ierr)
      implicit none

      integer, intent(inout) :: ierr

      if (.not. mbx_active) return
      call quick_mbx_send_qm_field(ierr)
      if (ierr /= 0) return
      call quick_mbx_update_energy(ierr)
      if (ierr /= 0) return
      call quick_mbx_refresh_cross_residual(ierr)
   end subroutine quick_mbx_refresh_energy

   subroutine quick_mbx_refresh_cross_residual(ierr)
      use quick_molspec_module, only: natom, xyz
#ifdef MPIV
      use mpi
      use quick_mpi_module, only: master, mpierror
#endif
      implicit none

      integer, intent(inout) :: ierr
#ifdef MBX
      integer :: iatom, ixyz, icomponent, nmb
      integer(c_int) :: c_natoms, c_ierr
      double precision, allocatable :: coords_flat(:), grad_kcal(:,:,:)
      double precision :: terms_kcal(3), gradient_scale
#endif

      if (ierr /= 0 .or. .not. mbx_cross_active) return
#ifdef MBX
      nmb=mbx_cross_natoms_total-mbx_cross_natoms_qm
      if (natom /= mbx_cross_natoms_qm .or. nmb /= mbx_natoms_real) then
         ierr=45
         return
      endif

      allocate(coords_flat(3*mbx_cross_natoms_total))
      allocate(grad_kcal(3,mbx_cross_natoms_total,3))
      coords_flat(:)=0.0d0
      terms_kcal(:)=0.0d0
      grad_kcal(:,:,:)=0.0d0
      do iatom=1,natom
         do ixyz=1,3
            coords_flat(3*(iatom-1)+ixyz)=BOHRS_TO_A*xyz(ixyz,iatom)
         enddo
      enddo
      do iatom=1,nmb
         do ixyz=1,3
            coords_flat(3*(natom+iatom-1)+ixyz)=mbx_real_xyz_ang(ixyz,iatom)
         enddo
      enddo

      c_natoms=int(mbx_cross_natoms_total,c_int)
      c_ierr=0_c_int
#ifdef MPIV
      if (master) call mbx_cross_evaluate(coords_flat,c_natoms,terms_kcal,grad_kcal,c_ierr)
      call MPI_BCAST(c_ierr,1,mpi_integer,0,MPI_COMM_WORLD,mpierror)
      if (c_ierr == 0_c_int) then
         call MPI_BCAST(terms_kcal,3,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
         call MPI_BCAST(grad_kcal,9*mbx_cross_natoms_total,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
      endif
#else
      call mbx_cross_evaluate(coords_flat,c_natoms,terms_kcal,grad_kcal,c_ierr)
#endif
      ierr=int(c_ierr)
      if (ierr == 0) then
         gradient_scale=KCAL_TO_AU*BOHRS_TO_A
         mbx_cross_terms_au(:)=KCAL_TO_AU*terms_kcal(:)
         do icomponent=1,3
            mbx_cross_qm_grad_au(:,:,icomponent)=gradient_scale* &
               grad_kcal(:,1:mbx_cross_natoms_qm,icomponent)
            mbx_cross_mb_grad_au(:,:,icomponent)=gradient_scale* &
               grad_kcal(:,mbx_cross_natoms_qm+1:mbx_cross_natoms_total,icomponent)
         enddo
         mbx_energy_add_au=mbx_energy_add_au+sum(mbx_cross_terms_au)
      endif
      deallocate(coords_flat,grad_kcal)
#else
      ierr=44
#endif
   end subroutine quick_mbx_refresh_cross_residual

   subroutine quick_mbx_get_site_data(nsites,xyz_sites_bohr,charges_e,ierr)
      implicit none

      integer, intent(in) :: nsites
      double precision, intent(out) :: xyz_sites_bohr(3,nsites)
      double precision, intent(out) :: charges_e(nsites)
      integer, intent(inout) :: ierr

      xyz_sites_bohr(:,:) = 0.0d0
      charges_e(:) = 0.0d0
      if (ierr /= 0) return
      if (.not. mbx_initialized) then
         ierr = 43
         return
      endif
      if (nsites /= mbx_nsites) then
         ierr = 45
         return
      endif
      if (.not. allocated(mbx_site_xyz_bohr) .or. .not. allocated(mbx_charge_e)) then
         ierr = 43
         return
      endif

      xyz_sites_bohr(:,:) = mbx_site_xyz_bohr(:,:)
      charges_e(:) = mbx_charge_e(:)
   end subroutine quick_mbx_get_site_data

   subroutine quick_mbx_get_induced_dipoles_au(nsites,mu_ind_au,ierr)
      implicit none

      integer, intent(in) :: nsites
      double precision, intent(out) :: mu_ind_au(3,nsites)
      integer, intent(inout) :: ierr

      mu_ind_au(:,:) = 0.0d0
      if (ierr /= 0) return
      if (.not. mbx_initialized) then
         ierr = 43
         return
      endif
      if (nsites /= mbx_nsites) then
         ierr = 45
         return
      endif
      if (.not. allocated(mbx_mu_ind_ea)) then
         ierr = 43
         return
      endif

      mu_ind_au(:,:) = DIPOLE_MBX_TO_AU*mbx_mu_ind_ea(:,:)
   end subroutine quick_mbx_get_induced_dipoles_au

   subroutine quick_mbx_set_real_coordinates(natoms,coords_ang,ierr)
#ifdef MPIV
      use quick_mpi_module, only: master
#endif
      implicit none

      integer, intent(in) :: natoms
      double precision, intent(in) :: coords_ang(3,natoms)
      integer, intent(inout) :: ierr
#ifdef MBX
      integer :: iatom, ixyz
      integer(c_int) :: c_natoms
      double precision, allocatable :: coords_flat(:)
#endif

      if (ierr /= 0) return
      if (.not. mbx_initialized) then
         ierr = 43
         return
      endif
      if (natoms /= mbx_natoms_real) then
         ierr = 45
         return
      endif

      mbx_real_xyz_ang(:,:) = coords_ang(:,:)
      mbx_qm_field_ready = .false.
#ifdef MBX
      allocate(coords_flat(3*natoms))
      do iatom=1,natoms
         do ixyz=1,3
            coords_flat(3*(iatom-1)+ixyz)=coords_ang(ixyz,iatom)
         enddo
      enddo
      c_natoms=int(natoms,c_int)
#ifdef MPIV
      if (master) call mbx_set_real_xyz(coords_flat,c_natoms)
#else
      call mbx_set_real_xyz(coords_flat,c_natoms)
#endif
      deallocate(coords_flat)
      call quick_mbx_refresh_site_data(ierr)
#else
      ierr = 44
#endif
   end subroutine quick_mbx_set_real_coordinates

   subroutine quick_mbx_refresh_site_data(ierr)
#ifdef MPIV
      use mpi
      use quick_mpi_module, only: master, mpierror
#endif
      implicit none

      integer, intent(inout) :: ierr
#ifdef MBX
      integer :: isite, ixyz
      double precision, allocatable :: sites_flat(:), charges(:)
#endif

      if (ierr /= 0) return
      if (.not. allocated(mbx_site_xyz_bohr) .or. .not. allocated(mbx_charge_e)) then
         ierr = 43
         return
      endif
#ifdef MBX
      allocate(sites_flat(3*mbx_nsites),charges(mbx_nsites))
      sites_flat(:)=0.0d0
      charges(:)=0.0d0
#ifdef MPIV
      if (master) then
#endif
         call mbx_get_xyz(sites_flat)
         call mbx_get_charges(charges)
#ifdef MPIV
      endif
      call MPI_BCAST(sites_flat,3*mbx_nsites,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(charges,mbx_nsites,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
#endif
      do isite=1,mbx_nsites
         do ixyz=1,3
            mbx_site_xyz_bohr(ixyz,isite)=A_TO_BOHRS*sites_flat(3*(isite-1)+ixyz)
         enddo
         mbx_charge_e(isite)=charges(isite)
      enddo
      deallocate(sites_flat,charges)
#else
      ierr = 44
#endif
   end subroutine quick_mbx_refresh_site_data

   subroutine quick_mbx_get_real_atom_gradient(energy_au,grad_au_per_bohr,ierr)
#ifdef MPIV
      use mpi
      use quick_mpi_module, only: master, mpierror
#endif
      implicit none

      double precision, intent(out) :: energy_au
      double precision, intent(out) :: grad_au_per_bohr(:,:)
      integer, intent(inout) :: ierr
#ifdef MBX
      integer :: iatom, ixyz
      integer(c_int) :: c_natoms
      double precision, allocatable :: coords_flat(:), grad_flat(:)
#endif

      energy_au = 0.0d0
      grad_au_per_bohr(:,:) = 0.0d0
      if (ierr /= 0) return
      if (.not. mbx_initialized) then
         ierr = 43
         return
      endif
      if (size(grad_au_per_bohr,1) < 3 .or. size(grad_au_per_bohr,2) < mbx_natoms_real) then
         ierr = 45
         return
      endif

#ifdef MBX
      allocate(coords_flat(3*mbx_natoms_real))
      allocate(grad_flat(3*mbx_natoms_real))
      coords_flat(:) = 0.0d0
      grad_flat(:) = 0.0d0

      do iatom=1,mbx_natoms_real
         do ixyz=1,3
            coords_flat(3*(iatom-1)+ixyz) = mbx_real_xyz_ang(ixyz,iatom)
         enddo
      enddo

      c_natoms = int(mbx_natoms_real,c_int)
#ifdef MPIV
      if (master) then
#endif
         call mbx_get_energy_g(coords_flat,c_natoms,mbx_energy_kcalmol,grad_flat)
#ifdef MPIV
      endif
      call MPI_BCAST(mbx_energy_kcalmol,1,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(grad_flat,3*mbx_natoms_real,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
#endif

      energy_au = KCAL_TO_AU*mbx_energy_kcalmol
      do iatom=1,mbx_natoms_real
         do ixyz=1,3
            grad_au_per_bohr(ixyz,iatom) = KCAL_TO_AU*BOHRS_TO_A*grad_flat(3*(iatom-1)+ixyz)
         enddo
      enddo

      deallocate(coords_flat)
      deallocate(grad_flat)
#else
      ierr = 44
#endif
   end subroutine quick_mbx_get_real_atom_gradient

   subroutine quick_mbx_redistribute_site_gradient(nsites,site_grad_au_per_bohr,redistributed_grad_au_per_bohr,ierr)
#ifdef MPIV
      use mpi
      use quick_mpi_module, only: master, mpierror
#endif
      implicit none

      integer, intent(in) :: nsites
      double precision, intent(in) :: site_grad_au_per_bohr(3,nsites)
      double precision, intent(out) :: redistributed_grad_au_per_bohr(3,nsites)
      integer, intent(inout) :: ierr
#ifdef MBX
      integer :: isite, ixyz
      integer(c_int) :: c_nsites
      double precision, allocatable :: grad_flat(:)
      double precision :: au_bohr_to_kcal_ang, sum_in(3), sum_out(3), scale
#endif

      redistributed_grad_au_per_bohr(:,:) = 0.0d0
      if (ierr /= 0) return
      if (.not. mbx_initialized) then
         ierr = 43
         return
      endif
      if (nsites /= mbx_nsites) then
         ierr = 45
         return
      endif

#ifdef MBX
      allocate(grad_flat(3*nsites))
      au_bohr_to_kcal_ang = 1.0d0/(KCAL_TO_AU*BOHRS_TO_A)
      do isite=1,nsites
         do ixyz=1,3
            grad_flat(3*(isite-1)+ixyz) = au_bohr_to_kcal_ang*site_grad_au_per_bohr(ixyz,isite)
         enddo
      enddo

      c_nsites = int(nsites,c_int)
#ifdef MPIV
      if (master) then
#endif
         call mbx_redistribute_gradients(grad_flat,c_nsites)
#ifdef MPIV
      endif
      call MPI_BCAST(grad_flat,3*nsites,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
#endif

      do isite=1,nsites
         do ixyz=1,3
            redistributed_grad_au_per_bohr(ixyz,isite) = KCAL_TO_AU*BOHRS_TO_A*grad_flat(3*(isite-1)+ixyz)
         enddo
      enddo
#ifdef MPIV
      if (master) then
#endif
         sum_in=sum(site_grad_au_per_bohr,dim=2)
         sum_out=sum(redistributed_grad_au_per_bohr(:,1:mbx_natoms_real),dim=2)
         scale=max(1.0d0,maxval(abs(sum_in)))
         if (maxval(abs(sum_out-sum_in)) > 1.0d-10*scale) ierr=53
#ifdef MPIV
      endif
      call MPI_BCAST(ierr,1,mpi_integer,0,MPI_COMM_WORLD,mpierror)
#endif
      deallocate(grad_flat)
#else
      ierr = 44
#endif
   end subroutine quick_mbx_redistribute_site_gradient

   subroutine quick_mbx_add_qm_gradient(ierr)
      use quick_oei_module, only: add_point_multipole_qm_gradient
      use quick_calculated_module, only: quick_qm_struct
      use quick_molspec_module, only: natom
#ifdef MPIV
      use quick_mpi_module, only: master
#endif
      implicit none

      integer, intent(inout) :: ierr
      integer :: iatom, ixyz
      double precision, allocatable :: mu_ind_au(:,:)

      if (ierr /= 0 .or. .not. mbx_active) return
      if (.not. mbx_initialized .or. .not. mbx_qm_field_ready) then
         ierr = 43
         return
      endif

      call quick_mbx_refresh_operator_scales()
      if (abs(mbx_perm_operator_scale-1.0d0) > 1.0d-14 .or. &
          abs(mbx_ind_operator_scale-1.0d0) > 1.0d-14) then
         ierr = 45
         return
      endif

      allocate(mu_ind_au(3,mbx_nsites))
      mu_ind_au(:,:)=DIPOLE_MBX_TO_AU*mbx_mu_ind_ea(:,:)
      call add_point_multipole_qm_gradient(mbx_nsites,mbx_site_xyz_bohr,mbx_charge_e,mu_ind_au)
      deallocate(mu_ind_au)

      if (mbx_cross_active) then
         if (natom /= mbx_cross_natoms_qm .or. .not. allocated(mbx_cross_qm_grad_au)) then
            ierr=45
            return
         endif
#ifdef MPIV
         if (master) then
#endif
            do iatom=1,natom
               do ixyz=1,3
                  quick_qm_struct%gradient(3*(iatom-1)+ixyz)= &
                     quick_qm_struct%gradient(3*(iatom-1)+ixyz)+ &
                     sum(mbx_cross_qm_grad_au(ixyz,iatom,:))
               enddo
            enddo
#ifdef MPIV
         endif
#endif
      endif
   end subroutine quick_mbx_add_qm_gradient

   subroutine quick_mbx_get_cross_qm_gradient(grad_au_per_bohr,ierr)
      implicit none

      double precision, intent(out) :: grad_au_per_bohr(:,:)
      integer, intent(inout) :: ierr

      grad_au_per_bohr(:,:)=0.0d0
      if (ierr /= 0 .or. .not. mbx_cross_active) return
      if (.not. allocated(mbx_cross_qm_grad_au) .or. size(grad_au_per_bohr,1) < 3 .or. &
          size(grad_au_per_bohr,2) < mbx_cross_natoms_qm) then
         ierr=45
         return
      endif
      grad_au_per_bohr(:,1:mbx_cross_natoms_qm)=sum(mbx_cross_qm_grad_au,dim=3)
   end subroutine quick_mbx_get_cross_qm_gradient

   subroutine quick_mbx_get_coupled_real_atom_gradient(grad_au_per_bohr,ierr,mbx_energy_au, &
         mbx_base_grad,induced_efg_grad,cross_residual_grad)
      use quick_oeproperties_module, only: compute_oeprop_values
      implicit none

      double precision, intent(out) :: grad_au_per_bohr(:,:)
      integer, intent(inout) :: ierr
      double precision, intent(out), optional :: mbx_energy_au
      double precision, intent(out), optional :: mbx_base_grad(:,:), induced_efg_grad(:,:)
      double precision, intent(out), optional :: cross_residual_grad(:,:)

      integer :: isite, i, j
      double precision :: energy_au
      double precision, allocatable :: internal_grad(:,:), efg(:,:,:)
      double precision, allocatable :: mu_ind_au(:,:), site_grad(:,:), redistributed(:,:)

      grad_au_per_bohr(:,:)=0.0d0
      if (present(mbx_energy_au)) mbx_energy_au=0.0d0
      if (present(mbx_base_grad)) mbx_base_grad(:,:)=0.0d0
      if (present(induced_efg_grad)) induced_efg_grad(:,:)=0.0d0
      if (present(cross_residual_grad)) cross_residual_grad(:,:)=0.0d0
      if (ierr /= 0) return
      if (.not. mbx_initialized .or. .not. mbx_qm_field_ready) then
         ierr = 43
         return
      endif
      if (size(grad_au_per_bohr,1) < 3 .or. size(grad_au_per_bohr,2) < mbx_natoms_real) then
         ierr = 45
         return
      endif
      if (present(mbx_base_grad)) then
         if (size(mbx_base_grad,1) < 3 .or. size(mbx_base_grad,2) < mbx_natoms_real) then
            ierr=45
            return
         endif
      endif
      if (present(induced_efg_grad)) then
         if (size(induced_efg_grad,1) < 3 .or. size(induced_efg_grad,2) < mbx_natoms_real) then
            ierr=45
            return
         endif
      endif
      if (present(cross_residual_grad)) then
         if (size(cross_residual_grad,1) < 3 .or. size(cross_residual_grad,2) < mbx_natoms_real) then
            ierr=45
            return
         endif
      endif

      allocate(internal_grad(3,mbx_natoms_real),efg(3,3,mbx_nsites))
      allocate(mu_ind_au(3,mbx_nsites),site_grad(3,mbx_nsites),redistributed(3,mbx_nsites))

      ! MBX differentiates all internal many-body terms while the sampled
      ! external QM potential and field are held fixed. Its analytic gradient
      ! already includes the permanent -q E force from the supplied field.
      call quick_mbx_get_real_atom_gradient(energy_au,internal_grad,ierr)
      if (ierr /= 0) goto 900
      call quick_mbx_fetch_induced_dipoles(ierr)
      if (ierr /= 0) goto 900

      ! MBX has not been given dE_i/dR_j, so QUICK supplies the remaining
      ! stationary induced-dipole force explicitly from its analytic EFG.
      call compute_oeprop_values(mbx_nsites,mbx_site_xyz_bohr,efg=efg)
      mu_ind_au(:,:)=DIPOLE_MBX_TO_AU*mbx_mu_ind_ea(:,:)
      site_grad(:,:)=0.0d0
      do isite=1,mbx_nsites
         do j=1,3
            do i=1,3
               site_grad(j,isite)=site_grad(j,isite)-mu_ind_au(i,isite)*efg(i,j,isite)
            enddo
         enddo
      enddo
      call quick_mbx_redistribute_site_gradient(mbx_nsites,site_grad,redistributed,ierr)
      if (ierr /= 0) goto 900

      grad_au_per_bohr(:,1:mbx_natoms_real)=internal_grad(:,:)+redistributed(:,1:mbx_natoms_real)
      if (mbx_cross_active) then
         if (.not. allocated(mbx_cross_mb_grad_au) .or. &
             size(mbx_cross_mb_grad_au,2) /= mbx_natoms_real) then
            ierr=45
            goto 900
         endif
         grad_au_per_bohr(:,1:mbx_natoms_real)=grad_au_per_bohr(:,1:mbx_natoms_real)+ &
            sum(mbx_cross_mb_grad_au,dim=3)
      endif
      if (present(mbx_energy_au)) mbx_energy_au=energy_au
      if (present(mbx_base_grad)) mbx_base_grad(:,1:mbx_natoms_real)=internal_grad(:,:)
      if (present(induced_efg_grad)) induced_efg_grad(:,1:mbx_natoms_real)= &
         redistributed(:,1:mbx_natoms_real)
      if (present(cross_residual_grad) .and. mbx_cross_active) &
         cross_residual_grad(:,1:mbx_natoms_real)=sum(mbx_cross_mb_grad_au,dim=3)

900   continue
      deallocate(internal_grad,efg,mu_ind_au,site_grad,redistributed)
   end subroutine quick_mbx_get_coupled_real_atom_gradient

   subroutine quick_mbx_update_energy(ierr)
#ifdef MPIV
      use mpi
      use quick_mpi_module, only: master, mpierror
#endif
      implicit none

      integer, intent(inout) :: ierr
#ifdef MBX
      integer :: iatom, ixyz
      integer(c_int) :: c_natoms
      double precision, allocatable :: coords_flat(:)
#endif

      if (.not. mbx_initialized) return

#ifdef MBX
      allocate(coords_flat(3*mbx_natoms_real))
      do iatom=1,mbx_natoms_real
         do ixyz=1,3
            coords_flat(3*(iatom-1)+ixyz) = mbx_real_xyz_ang(ixyz,iatom)
         enddo
      enddo

      c_natoms = int(mbx_natoms_real,c_int)
#ifdef MPIV
      if (master) then
#endif
         call mbx_get_energy(coords_flat,c_natoms,mbx_energy_kcalmol)
         call mbx_get_external_field_contribution_to_energy(mbx_external_energy_kcalmol)
#ifdef MPIV
      endif
      call MPI_BCAST(mbx_energy_kcalmol,1,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(mbx_external_energy_kcalmol,1,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
#endif
      deallocate(coords_flat)
      call quick_mbx_fetch_induced_dipoles(ierr)
      if (ierr /= 0) return
      call quick_mbx_compute_energy_decomposition()
#else
      ierr = 44
#endif
   end subroutine quick_mbx_update_energy

   subroutine quick_mbx_fetch_induced_dipoles(ierr)
#ifdef MPIV
      use mpi
      use quick_mpi_module, only: master, mpierror
#endif
      implicit none

      integer, intent(inout) :: ierr
#ifdef MBX
      integer :: isite, ixyz
      double precision, allocatable :: mu_flat(:)
#endif

      if (.not. mbx_initialized) return
      if (.not. allocated(mbx_mu_ind_ea)) return

#ifdef MBX
      mbx_mu_ind_ea(:,:) = 0.0d0
#ifdef MPIV
      if (master) then
#endif
         allocate(mu_flat(3*mbx_nsites))
         call mbx_get_induced_dipoles(mu_flat)
         do isite=1,mbx_nsites
            do ixyz=1,3
               mbx_mu_ind_ea(ixyz,isite) = mu_flat(3*(isite-1)+ixyz)
            enddo
         enddo
         deallocate(mu_flat)
#ifdef MPIV
      endif
      call MPI_BCAST(mbx_mu_ind_ea,3*mbx_nsites,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
#endif
#else
      ierr = 44
#endif
   end subroutine quick_mbx_fetch_induced_dipoles

   subroutine quick_mbx_compute_energy_decomposition()
#ifdef MPIV
      use mpi
      use quick_mpi_module, only: master, mpierror
#endif
      use quick_molspec_module, only: natom, quick_molspec, xyz
      implicit none

      integer :: isite, iatom, ixyz
      double precision :: rvec(3), dist2, inv_dist, inv_dist3
      double precision :: phi_classical, efield_classical(3)
      double precision :: charge, mu_au(3), efield_electronic(3)
      double precision :: terms(8)

#ifdef MPIV
      if (master) then
#endif
         mbx_perm_total_au = 0.0d0
         mbx_perm_classical_au = 0.0d0
         mbx_perm_electronic_au = 0.0d0
         mbx_ind_total_au = 0.0d0
         mbx_ind_classical_au = 0.0d0
         mbx_ind_electronic_au = 0.0d0

         do isite=1,mbx_nsites
            phi_classical = 0.0d0
            efield_classical(:) = 0.0d0

            do iatom=1,natom+quick_molspec%nextatom
               if (iatom <= natom) then
                  charge = quick_molspec%chg(iatom)
                  rvec(:) = mbx_site_xyz_bohr(:,isite) - xyz(1:3,iatom)
               else
                  charge = quick_molspec%extchg(iatom-natom)
                  rvec(:) = mbx_site_xyz_bohr(:,isite) - quick_molspec%extxyz(1:3,iatom-natom)
               endif

               dist2 = rvec(1)*rvec(1) + rvec(2)*rvec(2) + rvec(3)*rvec(3)
               if (dist2 > 1.0d-24) then
                  inv_dist = 1.0d0/dsqrt(dist2)
                  inv_dist3 = inv_dist*inv_dist*inv_dist
                  phi_classical = phi_classical + charge*inv_dist
                  do ixyz=1,3
                     efield_classical(ixyz) = efield_classical(ixyz) + charge*rvec(ixyz)*inv_dist3
                  enddo
               endif
            enddo

            charge = mbx_charge_e(isite)
            mbx_perm_total_au = mbx_perm_total_au + charge*mbx_phi_au(isite)
            mbx_perm_classical_au = mbx_perm_classical_au + charge*phi_classical
            mbx_perm_electronic_au = mbx_perm_electronic_au + charge*(mbx_phi_au(isite)-phi_classical)

            do ixyz=1,3
               mu_au(ixyz) = DIPOLE_MBX_TO_AU*mbx_mu_ind_ea(ixyz,isite)
               efield_electronic(ixyz) = mbx_efield_au(ixyz,isite) - efield_classical(ixyz)
            enddo
            mbx_ind_total_au = mbx_ind_total_au - dot_product(mu_au,mbx_efield_au(:,isite))
            mbx_ind_classical_au = mbx_ind_classical_au - dot_product(mu_au,efield_classical)
            mbx_ind_electronic_au = mbx_ind_electronic_au - dot_product(mu_au,efield_electronic)
         enddo

         ! QUICK's SCF energy already contains the electronic expectation value
         ! of the MBX permanent and induced AO operators.  Replace those
         ! electronic operator terms with MBX's energy expression.  MBX reports
         ! one half of the permanent external electrostatic energy, so restore
         ! the missing half of the total permanent QM--MBX interaction here.
         call quick_mbx_refresh_operator_scales()
         mbx_energy_add_au = KCAL_TO_AU*mbx_energy_kcalmol &
            - mbx_perm_operator_scale*mbx_perm_electronic_au &
            - mbx_ind_operator_scale*mbx_ind_electronic_au &
            + 0.5d0*mbx_perm_total_au
         mbx_external_perm_half_residual_au = KCAL_TO_AU*mbx_external_energy_kcalmol &
            - 0.5d0*mbx_perm_total_au
#ifdef MPIV
      endif

      terms(1) = mbx_energy_add_au
      terms(2) = mbx_perm_total_au
      terms(3) = mbx_perm_classical_au
      terms(4) = mbx_perm_electronic_au
      terms(5) = mbx_ind_total_au
      terms(6) = mbx_ind_classical_au
      terms(7) = mbx_ind_electronic_au
      terms(8) = mbx_external_perm_half_residual_au
      call MPI_BCAST(terms,8,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
      mbx_energy_add_au = terms(1)
      mbx_perm_total_au = terms(2)
      mbx_perm_classical_au = terms(3)
      mbx_perm_electronic_au = terms(4)
      mbx_ind_total_au = terms(5)
      mbx_ind_classical_au = terms(6)
      mbx_ind_electronic_au = terms(7)
      mbx_external_perm_half_residual_au = terms(8)
#endif
   end subroutine quick_mbx_compute_energy_decomposition

   subroutine quick_mbx_get_energy_terms(energy_au,external_energy_au,additive_energy_au, &
         perm_total_au,perm_classical_au,perm_electronic_au,ind_total_au, &
         ind_classical_au,ind_electronic_au,external_half_residual_au, &
         cross_2b_au,cross_3b_au,cross_disp_au)
      implicit none

      double precision, intent(out) :: energy_au
      double precision, intent(out) :: external_energy_au
      double precision, intent(out), optional :: additive_energy_au
      double precision, intent(out), optional :: perm_total_au
      double precision, intent(out), optional :: perm_classical_au
      double precision, intent(out), optional :: perm_electronic_au
      double precision, intent(out), optional :: ind_total_au
      double precision, intent(out), optional :: ind_classical_au
      double precision, intent(out), optional :: ind_electronic_au
      double precision, intent(out), optional :: external_half_residual_au
      double precision, intent(out), optional :: cross_2b_au, cross_3b_au, cross_disp_au

      energy_au = 0.0d0
      external_energy_au = 0.0d0
      if (present(additive_energy_au)) additive_energy_au = 0.0d0
      if (present(perm_total_au)) perm_total_au = 0.0d0
      if (present(perm_classical_au)) perm_classical_au = 0.0d0
      if (present(perm_electronic_au)) perm_electronic_au = 0.0d0
      if (present(ind_total_au)) ind_total_au = 0.0d0
      if (present(ind_classical_au)) ind_classical_au = 0.0d0
      if (present(ind_electronic_au)) ind_electronic_au = 0.0d0
      if (present(external_half_residual_au)) external_half_residual_au = 0.0d0
      if (present(cross_2b_au)) cross_2b_au=0.0d0
      if (present(cross_3b_au)) cross_3b_au=0.0d0
      if (present(cross_disp_au)) cross_disp_au=0.0d0
      if (.not. mbx_initialized) return
      energy_au = KCAL_TO_AU*mbx_energy_kcalmol
      external_energy_au = KCAL_TO_AU*mbx_external_energy_kcalmol
      if (present(additive_energy_au)) additive_energy_au = mbx_energy_add_au
      if (present(perm_total_au)) perm_total_au = mbx_perm_total_au
      if (present(perm_classical_au)) perm_classical_au = mbx_perm_classical_au
      if (present(perm_electronic_au)) perm_electronic_au = mbx_perm_electronic_au
      if (present(ind_total_au)) ind_total_au = mbx_ind_total_au
      if (present(ind_classical_au)) ind_classical_au = mbx_ind_classical_au
      if (present(ind_electronic_au)) ind_electronic_au = mbx_ind_electronic_au
      if (present(external_half_residual_au)) external_half_residual_au = mbx_external_perm_half_residual_au
      if (present(cross_2b_au)) cross_2b_au=mbx_cross_terms_au(1)
      if (present(cross_3b_au)) cross_3b_au=mbx_cross_terms_au(2)
      if (present(cross_disp_au)) cross_disp_au=mbx_cross_terms_au(3)
   end subroutine quick_mbx_get_energy_terms

   subroutine quick_mbx_update_scf_operator(ierr)
      use quick_oei_module, only: add_point_charge_operator, add_point_dipole_field_operator
      implicit none

      integer, intent(inout) :: ierr
#ifdef MBX
      integer :: isite, ixyz
      double precision, allocatable :: mu_ind_au(:,:)
#endif

      if (.not. mbx_active) return
      call quick_mbx_send_qm_field(ierr)
      if (ierr /= 0) return
      call quick_mbx_update_energy(ierr)
      if (ierr /= 0) return

#ifdef MBX
      if (allocated(mbx_charge_e)) then
         call quick_mbx_refresh_operator_scales()
         if (mbx_perm_operator_scale /= 0.0d0) then
            call add_point_charge_operator(mbx_nsites,mbx_site_xyz_bohr,mbx_perm_operator_scale*mbx_charge_e)
         endif
      endif

      allocate(mu_ind_au(3,mbx_nsites))
      mu_ind_au(:,:) = 0.0d0

      do isite=1,mbx_nsites
         do ixyz=1,3
            mu_ind_au(ixyz,isite) = mbx_ind_operator_scale*DIPOLE_MBX_TO_AU*mbx_mu_ind_ea(ixyz,isite)
         enddo
      enddo

      call add_point_dipole_field_operator(mbx_nsites,mbx_site_xyz_bohr,mu_ind_au)
      deallocate(mu_ind_au)
#else
      ierr = 44
#endif
   end subroutine quick_mbx_update_scf_operator

   subroutine quick_mbx_refresh_operator_scales()
      implicit none

      integer :: ierr_env
      character(len=64) :: env_value

      mbx_perm_operator_scale = 1.0d0
      mbx_ind_operator_scale = 1.0d0

      call get_environment_variable('QUICK_MBX_PERM_FOCK_SCALE', env_value, status=ierr_env)
      if (ierr_env == 0) then
         read(env_value, *, iostat=ierr_env) mbx_perm_operator_scale
         if (ierr_env /= 0) mbx_perm_operator_scale = 1.0d0
      endif

      call get_environment_variable('QUICK_MBX_IND_FOCK_SCALE', env_value, status=ierr_env)
      if (ierr_env == 0) then
         read(env_value, *, iostat=ierr_env) mbx_ind_operator_scale
         if (ierr_env /= 0) mbx_ind_operator_scale = 1.0d0
      endif
   end subroutine quick_mbx_refresh_operator_scales

   subroutine set_c_string(src,dst)
      implicit none

      character(len=*), intent(in) :: src
      character(kind=c_char), intent(out) :: dst(:)

      integer :: i, ncopy

      dst(:) = c_null_char
      ncopy = min(len_trim(src),size(dst)-1)
      do i=1,ncopy
         dst(i) = src(i:i)
      enddo
   end subroutine set_c_string

end module quick_mbx_module
