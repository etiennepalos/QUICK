#include "util.fh"
!---------------------------------------------------------------------!
! QUICK-MBX gradient decomposition diagnostic.                         !
!                                                                     !
! Compares total finite-difference QUICK-MBX gradients against the     !
! MBX-side analytic pieces currently available without a production    !
! MBX external-derivative ABI.                                         !
!_____________________________________________________________________!

program quick_mbx_gradient_decomp_probe

   use quick_api_module, only: setQuickJob, getQuickEnergy, getQuickMBXEnergyGradientsFD, &
      getQuickOEPROP, setQuickMBXWaterSystem, clearQuickMBXSystem, deleteQuickJob
   use quick_mbx_module, only: quick_mbx_get_site_data, quick_mbx_get_induced_dipoles_au, &
      quick_mbx_get_real_atom_gradient, quick_mbx_redistribute_site_gradient

   implicit none

   integer, parameter :: natom_qm = 3
   integer, parameter :: nwaters = 1
   integer, parameter :: natom_mbx = 3*nwaters
   integer, parameter :: nsites = 4*nwaters
   double precision, parameter :: fd_step_bohr = 2.0d-3

   integer :: ierr, argc
   integer :: atomic_numbers(natom_qm)
   double precision :: qm_xyz(3,natom_qm), mbx_xyz(3,natom_mbx)
   double precision :: fd_energy, base_energy, mbx_internal_energy
   double precision :: fd_qm_grad(3,natom_qm), fd_mbx_grad(3,natom_mbx)
   double precision :: site_xyz_bohr(3,nsites), site_charge(nsites)
   double precision :: efield(3,nsites), efg(3,3,nsites), mu_ind_au(3,nsites)
   double precision :: site_perm_grad(3,nsites), site_ind_grad(3,nsites)
   double precision :: red_perm_grad(3,nsites), red_ind_grad(3,nsites)
   double precision :: mbx_internal_grad(3,natom_mbx)
   double precision :: mbx_perm_grad(3,natom_mbx), mbx_ind_grad(3,natom_mbx)
   double precision :: mbx_internal_perm(3,natom_mbx), mbx_internal_perm_ind(3,natom_mbx)
   double precision :: max_internal, rms_internal, max_perm, rms_perm, max_perm_ind, rms_perm_ind
   double precision :: missing_norm, max_fd
   character(len=256) :: csv_file, json_file, json_mbx
   character(len=80) :: job_name
   character(len=256) :: keywd

   ierr = 0
   csv_file = 'quick_mbx_gradient_decomp_probe.csv'
   json_file = 'quick_mbx_gradient_decomp_probe_summary.json'
   json_mbx = 'mbx.json'
   argc = command_argument_count()
   if (argc >= 1) call get_command_argument(1, csv_file)
   if (argc >= 2) call get_command_argument(2, json_file)
   if (argc >= 3) call get_command_argument(3, json_mbx)

   atomic_numbers = (/8,1,1/)
   call setup_water_pair(qm_xyz,mbx_xyz)

   job_name = 'quick_mbx_gradient_decomp_probe'
   keywd = 'HF BASIS=STO-3G CUTOFF=1.0D-10 DENSERMS=1.0D-6 MBX_QMMM'
   call setQuickJob(job_name,keywd,natom_qm,atomic_numbers,.true.,ierr)
   call stop_on_error(ierr,'setQuickJob')

   call getQuickMBXEnergyGradientsFD(qm_xyz,nwaters,mbx_xyz,json_mbx,fd_step_bohr, &
      fd_energy,fd_qm_grad,fd_mbx_grad,ierr)
   call stop_on_error(ierr,'total finite-difference gradient')

   call evaluate_base_qmmm(qm_xyz,mbx_xyz,json_mbx,base_energy,ierr)
   call stop_on_error(ierr,'base QUICK-MBX density')

   call quick_mbx_get_site_data(nsites,site_xyz_bohr,site_charge,ierr)
   call stop_on_error(ierr,'get MBX site data')
   call getQuickOEPROP(nsites,site_xyz_bohr,ierr,efield=efield,efg=efg)
   call stop_on_error(ierr,'get QUICK field and EFG')
   call quick_mbx_get_induced_dipoles_au(nsites,mu_ind_au,ierr)
   call stop_on_error(ierr,'get MBX induced dipoles')

   call build_site_gradients(site_charge,efield,efg,mu_ind_au,site_perm_grad,site_ind_grad)
   call quick_mbx_redistribute_site_gradient(nsites,site_perm_grad,red_perm_grad,ierr)
   call stop_on_error(ierr,'redistribute permanent gradient')
   call quick_mbx_redistribute_site_gradient(nsites,site_ind_grad,red_ind_grad,ierr)
   call stop_on_error(ierr,'redistribute frozen-induced gradient')
   mbx_perm_grad(:,:) = red_perm_grad(:,1:natom_mbx)
   mbx_ind_grad(:,:) = red_ind_grad(:,1:natom_mbx)

   call evaluate_mbx_internal_gradient(mbx_xyz,json_mbx,mbx_internal_energy,mbx_internal_grad,ierr)
   call stop_on_error(ierr,'MBX internal gradient')

   mbx_internal_perm(:,:) = mbx_internal_grad(:,:) + mbx_perm_grad(:,:)
   mbx_internal_perm_ind(:,:) = mbx_internal_perm(:,:) + mbx_ind_grad(:,:)

   call compare_component(fd_mbx_grad,mbx_internal_grad,max_internal,rms_internal,missing_norm,max_fd)
   call compare_component(fd_mbx_grad,mbx_internal_perm,max_perm,rms_perm,missing_norm,max_fd)
   call compare_component(fd_mbx_grad,mbx_internal_perm_ind,max_perm_ind,rms_perm_ind,missing_norm,max_fd)

   call write_csv(csv_file,fd_qm_grad,fd_mbx_grad,mbx_internal_grad,mbx_perm_grad, &
      mbx_ind_grad,mbx_internal_perm_ind,ierr)
   call stop_on_error(ierr,'write csv')
   call write_summary(json_file,fd_energy,base_energy,mbx_internal_energy,max_fd, &
      max_internal,rms_internal,max_perm,rms_perm,max_perm_ind,rms_perm_ind,missing_norm,ierr)
   call stop_on_error(ierr,'write summary')

   call clearQuickMBXSystem(ierr)
   call deleteQuickJob(ierr)
   call stop_on_error(ierr,'deleteQuickJob')

   write(*,'("QUICK_MBX_GRAD_DECOMP_FD_ENERGY_AU ",ES24.16)') fd_energy
   write(*,'("QUICK_MBX_GRAD_DECOMP_MBX_FD_MAX ",ES12.4)') max_fd
   write(*,'("QUICK_MBX_GRAD_DECOMP_INTERNAL_MAX_DIFF ",ES12.4)') max_internal
   write(*,'("QUICK_MBX_GRAD_DECOMP_INTERNAL_PERM_MAX_DIFF ",ES12.4)') max_perm
   write(*,'("QUICK_MBX_GRAD_DECOMP_INTERNAL_PERM_IND_MAX_DIFF ",ES12.4)') max_perm_ind
   write(*,'("QUICK_MBX_GRAD_DECOMP_MISSING_NORM ",ES12.4)') missing_norm

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

   subroutine evaluate_base_qmmm(qm_xyz,mbx_xyz,json_mbx,energy,ierr)
      implicit none

      double precision, intent(in) :: qm_xyz(3,natom_qm), mbx_xyz(3,natom_mbx)
      character(len=*), intent(in) :: json_mbx
      double precision, intent(out) :: energy
      integer, intent(inout) :: ierr

      double precision, allocatable :: no_point_charges(:,:)

      call clearQuickMBXSystem(ierr)
      if (ierr /= 0) return
      call setQuickMBXWaterSystem(nwaters,mbx_xyz,json_mbx,ierr)
      if (ierr /= 0) return
      allocate(no_point_charges(4,0),stat=ierr)
      if (ierr /= 0) return
      call getQuickEnergy(qm_xyz,0,no_point_charges,energy,ierr)
      if (allocated(no_point_charges)) deallocate(no_point_charges)
   end subroutine evaluate_base_qmmm

   subroutine evaluate_mbx_internal_gradient(mbx_xyz,json_mbx,energy,grad,ierr)
      implicit none

      double precision, intent(in) :: mbx_xyz(3,natom_mbx)
      character(len=*), intent(in) :: json_mbx
      double precision, intent(out) :: energy, grad(3,natom_mbx)
      integer, intent(inout) :: ierr

      call clearQuickMBXSystem(ierr)
      if (ierr /= 0) return
      call setQuickMBXWaterSystem(nwaters,mbx_xyz,json_mbx,ierr)
      if (ierr /= 0) return
      call quick_mbx_get_real_atom_gradient(energy,grad,ierr)
   end subroutine evaluate_mbx_internal_gradient

   subroutine build_site_gradients(site_charge,efield,efg,mu_ind_au,site_perm_grad,site_ind_grad)
      implicit none

      double precision, intent(in) :: site_charge(nsites), efield(3,nsites), efg(3,3,nsites)
      double precision, intent(in) :: mu_ind_au(3,nsites)
      double precision, intent(out) :: site_perm_grad(3,nsites), site_ind_grad(3,nsites)

      integer :: isite, i, j

      site_perm_grad(:,:) = 0.0d0
      site_ind_grad(:,:) = 0.0d0
      do isite=1,nsites
         do j=1,3
            site_perm_grad(j,isite) = -site_charge(isite)*efield(j,isite)
            do i=1,3
               site_ind_grad(j,isite) = site_ind_grad(j,isite) - mu_ind_au(i,isite)*efg(i,j,isite)
            enddo
         enddo
      enddo
   end subroutine build_site_gradients

   subroutine compare_component(fd_grad,model_grad,max_abs,rms,norm_diff,max_fd)
      implicit none

      double precision, intent(in) :: fd_grad(3,natom_mbx), model_grad(3,natom_mbx)
      double precision, intent(out) :: max_abs, rms, norm_diff, max_fd

      integer :: iatom, ixyz, ncomp
      double precision :: diff, accum

      max_abs = 0.0d0
      max_fd = 0.0d0
      accum = 0.0d0
      ncomp = 0
      do iatom=1,natom_mbx
         do ixyz=1,3
            diff = fd_grad(ixyz,iatom) - model_grad(ixyz,iatom)
            max_abs = max(max_abs,abs(diff))
            max_fd = max(max_fd,abs(fd_grad(ixyz,iatom)))
            accum = accum + diff*diff
            ncomp = ncomp + 1
         enddo
      enddo
      rms = dsqrt(accum/dble(max(1,ncomp)))
      norm_diff = dsqrt(accum)
   end subroutine compare_component

   subroutine write_csv(csv_file,fd_qm_grad,fd_mbx_grad,mbx_internal_grad,mbx_perm_grad, &
         mbx_ind_grad,semi_grad,ierr)
      implicit none

      character(len=*), intent(in) :: csv_file
      double precision, intent(in) :: fd_qm_grad(3,natom_qm), fd_mbx_grad(3,natom_mbx)
      double precision, intent(in) :: mbx_internal_grad(3,natom_mbx), mbx_perm_grad(3,natom_mbx)
      double precision, intent(in) :: mbx_ind_grad(3,natom_mbx), semi_grad(3,natom_mbx)
      integer, intent(inout) :: ierr

      integer :: unit, iatom, ixyz

      open(newunit=unit,file=csv_file,status='replace',action='write',iostat=ierr)
      if (ierr /= 0) return
      write(unit,'(A)') 'group,index,axis,fd_total,mbx_internal,perm_site,frozen_induced,semi_total,missing'
      do iatom=1,natom_qm
         do ixyz=1,3
            write(unit,'(A,",",I0,",",A,",",ES24.16,",",A,",",A,",",A,",",A,",",A)') &
               'qm_atom', iatom, axis_name(ixyz), fd_qm_grad(ixyz,iatom), '', '', '', '', ''
         enddo
      enddo
      do iatom=1,natom_mbx
         do ixyz=1,3
            write(unit,'(A,",",I0,",",A,",",ES24.16,",",ES24.16,",",ES24.16,",",ES24.16,",",ES24.16,",",ES24.16)') &
               'mbx_atom', iatom, axis_name(ixyz), fd_mbx_grad(ixyz,iatom), &
               mbx_internal_grad(ixyz,iatom), mbx_perm_grad(ixyz,iatom), &
               mbx_ind_grad(ixyz,iatom), semi_grad(ixyz,iatom), &
               fd_mbx_grad(ixyz,iatom) - semi_grad(ixyz,iatom)
         enddo
      enddo
      close(unit)
   end subroutine write_csv

   subroutine write_summary(json_file,fd_energy,base_energy,mbx_internal_energy,max_fd, &
         max_internal,rms_internal,max_perm,rms_perm,max_perm_ind,rms_perm_ind,missing_norm,ierr)
      implicit none

      character(len=*), intent(in) :: json_file
      double precision, intent(in) :: fd_energy, base_energy, mbx_internal_energy, max_fd
      double precision, intent(in) :: max_internal, rms_internal, max_perm, rms_perm
      double precision, intent(in) :: max_perm_ind, rms_perm_ind, missing_norm
      integer, intent(inout) :: ierr

      integer :: unit

      open(newunit=unit,file=json_file,status='replace',action='write',iostat=ierr)
      if (ierr /= 0) return
      write(unit,'(A)') '{'
      write(unit,'(A,ES24.16,A)') '  "fd_total_energy_au": ', fd_energy, ','
      write(unit,'(A,ES24.16,A)') '  "base_qmmm_energy_au": ', base_energy, ','
      write(unit,'(A,ES24.16,A)') '  "mbx_internal_energy_au": ', mbx_internal_energy, ','
      write(unit,'(A,ES12.4,A)') '  "fd_mbx_max_abs_au_per_bohr": ', max_fd, ','
      write(unit,'(A,ES12.4,A)') '  "internal_only_max_abs_diff_au_per_bohr": ', max_internal, ','
      write(unit,'(A,ES12.4,A)') '  "internal_only_rms_diff_au_per_bohr": ', rms_internal, ','
      write(unit,'(A,ES12.4,A)') '  "internal_plus_perm_max_abs_diff_au_per_bohr": ', max_perm, ','
      write(unit,'(A,ES12.4,A)') '  "internal_plus_perm_rms_diff_au_per_bohr": ', rms_perm, ','
      write(unit,'(A,ES12.4,A)') '  "internal_plus_perm_frozen_ind_max_abs_diff_au_per_bohr": ', max_perm_ind, ','
      write(unit,'(A,ES12.4,A)') '  "internal_plus_perm_frozen_ind_rms_diff_au_per_bohr": ', rms_perm_ind, ','
      write(unit,'(A,ES12.4)') '  "internal_plus_perm_frozen_ind_missing_norm_au_per_bohr": ', missing_norm
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
         write(*,'("QUICK_MBX_GRAD_DECOMP_ERROR ",A,1X,I0)') trim(label), ierr_in
         stop 1
      endif
   end subroutine stop_on_error

end program quick_mbx_gradient_decomp_probe
