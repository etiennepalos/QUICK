#include "util.fh"
!---------------------------------------------------------------------!
! CPU permanent-electrostatic QM/MM gradient validation probe.         !
!                                                                     !
! This driver intentionally avoids MBX-specific force APIs. It uses    !
! QUICK's public API with MB-pol-like permanent sites represented as   !
! fixed point charges, then validates the available analytic gradient  !
! slice against central finite differences.                            !
!_____________________________________________________________________!

program quick_qmmm_perm_grad_probe

   use quick_api_module, only: setQuickJob, getQuickEnergyGradients, getQuickOEPROP, deleteQuickJob
   use quick_constants_module, only: BOHRS_TO_A
#ifdef MPIV
   use mpi
   use quick_api_module, only: setQuickMPI
#endif

   implicit none

   integer, parameter :: natom_qm = 3
   integer, parameter :: nsites = 3
   integer, parameter :: ndim = 3
   double precision, parameter :: fd_step_bohr = 1.0d-3
   double precision, parameter :: m_site_offset_ang = 0.150d0
   double precision, parameter :: q_h = 0.520d0
   double precision, parameter :: q_m = -2.0d0*q_h

   integer :: ierr, argc, i
   integer :: atomic_numbers(natom_qm)
   double precision :: qm_xyz(3,natom_qm)
   double precision :: pc(4,nsites)
   double precision :: energy0
   double precision :: grad_qm(3,natom_qm), grad_pc(3,nsites)
   double precision :: efield(3,nsites), efg(3,3,nsites)
   double precision :: efg_max_abs, efg_rms
   double precision :: qm_max_abs, qm_rms, pc_max_abs, pc_rms
   character(len=256) :: csv_file, json_file
   character(len=80) :: job_name
   character(len=256) :: keywd
#ifdef MPIV
   integer :: mpierror, mpirank, mpisize
   logical :: master
#else
   logical :: master
#endif

   ierr = 0
   master = .true.

#ifdef MPIV
   call MPI_INIT(mpierror)
   call MPI_COMM_RANK(MPI_COMM_WORLD,mpirank,mpierror)
   call MPI_COMM_SIZE(MPI_COMM_WORLD,mpisize,mpierror)
   master = (mpirank == 0)
   call setQuickMPI(mpirank,mpisize,ierr)
   call stop_on_error(ierr, 'setQuickMPI')
#endif

   argc = command_argument_count()
   csv_file = 'qmmm_perm_grad_probe.csv'
   json_file = 'qmmm_perm_grad_probe_summary.json'
   if (argc >= 1) call get_command_argument(1, csv_file)
   if (argc >= 2) call get_command_argument(2, json_file)

   atomic_numbers = (/8, 1, 1/)
   call setup_water_dimer_probe(qm_xyz, pc)

   job_name = 'qmmm_perm_grad_probe'
   keywd = 'HF BASIS=STO-3G CUTOFF=1.0D-10 DENSERMS=1.0D-8 GRADIENT CHARGE=0'
   call setQuickJob(job_name, keywd, natom_qm, atomic_numbers, .true., ierr)
   call stop_on_error(ierr, 'setQuickJob')

   call evaluate(qm_xyz, pc, energy0, grad_qm, grad_pc, ierr)
   call stop_on_error(ierr, 'base gradient')

   call validate_efg(pc, efield, efg, efg_max_abs, efg_rms, ierr)
   call stop_on_error(ierr, 'OEPROP derivative checks')

   call finite_difference_gradients(csv_file, qm_xyz, pc, grad_qm, grad_pc, qm_max_abs, qm_rms, pc_max_abs, pc_rms, ierr)
   call stop_on_error(ierr, 'finite-difference gradients')

   if (master) then
      call write_summary(json_file, energy0, qm_max_abs, qm_rms, pc_max_abs, pc_rms, efg_max_abs, efg_rms, ierr)
      call stop_on_error(ierr, 'write summary')
   endif

   call deleteQuickJob(i)
   call stop_on_error(i, 'deleteQuickJob')

   if (master) then
      write(*,'("QMMM_PERM_GRAD_PROBE_BASE_ENERGY_AU ",ES24.16)') energy0
      write(*,'("QMMM_PERM_GRAD_PROBE_QM_MAX_ABS ",ES12.4)') qm_max_abs
      write(*,'("QMMM_PERM_GRAD_PROBE_PC_MAX_ABS ",ES12.4)') pc_max_abs
      write(*,'("QMMM_PERM_GRAD_PROBE_EFG_MAX_ABS ",ES12.4)') efg_max_abs
   endif

#ifdef MPIV
   call MPI_FINALIZE(mpierror)
#endif

contains

   subroutine setup_water_dimer_probe(qm_xyz, pc)
      implicit none

      double precision, intent(out) :: qm_xyz(3,natom_qm)
      double precision, intent(out) :: pc(4,nsites)

      double precision :: o(3), h1(3), h2(3), mid(3), bis(3), norm

      qm_xyz(:,1) = (/-1.58972425d0, 1.04337922d0, -0.08780840d0/)
      qm_xyz(:,2) = (/-0.63591971d0, 0.97898520d0,  0.00000000d0/)
      qm_xyz(:,3) = (/-1.90066280d0, 1.74501050d0, -0.66454990d0/)

      o  = (/ 1.64924507d0, 1.08594656d0,  0.00000000d0/)
      h1 = (/ 2.60878026d0, 1.09587704d0, -0.02817115d0/)
      h2 = (/ 1.33830653d0, 1.78757784d0,  0.57674150d0/)
      mid = 0.5d0*(h1 + h2)
      bis = mid - o
      norm = dsqrt(sum(bis*bis))
      if (norm > 0.0d0) bis = bis/norm

      pc(1:3,1) = h1
      pc(4,1) = q_h
      pc(1:3,2) = h2
      pc(4,2) = q_h
      pc(1:3,3) = o + m_site_offset_ang*bis
      pc(4,3) = q_m
   end subroutine setup_water_dimer_probe

   subroutine evaluate(qm_xyz, pc, energy, grad_qm, grad_pc, ierr)
      implicit none

      double precision, intent(in) :: qm_xyz(3,natom_qm)
      double precision, intent(in) :: pc(4,nsites)
      double precision, intent(out) :: energy
      double precision, intent(out) :: grad_qm(3,natom_qm)
      double precision, intent(out) :: grad_pc(3,nsites)
      integer, intent(out) :: ierr

      call getQuickEnergyGradients(qm_xyz, nsites, pc, energy, grad_qm, grad_pc, ierr)
   end subroutine evaluate

   subroutine validate_efg(pc, efield, efg, efg_max_abs, efg_rms, ierr)
      implicit none

      double precision, intent(in) :: pc(4,nsites)
      double precision, intent(out) :: efield(3,nsites)
      double precision, intent(out) :: efg(3,3,nsites)
      double precision, intent(out) :: efg_max_abs, efg_rms
      integer, intent(out) :: ierr

      integer :: isite, idir, icomp, n_efg
      double precision :: probes(3,nsites), probes_p(3,nsites), probes_m(3,nsites)
      double precision :: efield_p(3,nsites), efield_m(3,nsites)
      double precision :: fd, diff, accum_efg

      ierr = 0
      ! These probes are intentionally independent of the external charge
      ! sites. The production MBX path evaluates QM fields at MBX sites without
      ! also registering those sites as QUICK external charges. This validation
      ! probe does register fixed charges, so coincident OEPROP probes would
      ! include singular self-field terms from the validation charges.
      probes(:,1) = (/ 0.00000d0, 0.00000d0, 2.00000d0/) / BOHRS_TO_A
      probes(:,2) = (/ 1.50000d0, 0.00000d0, 0.00000d0/) / BOHRS_TO_A
      probes(:,3) = (/ 0.00000d0, 1.50000d0, 0.00000d0/) / BOHRS_TO_A
      call getQuickOEPROP(nsites, probes, ierr, efield=efield, efg=efg)
      if (ierr /= 0) return

      efg_max_abs = 0.0d0
      accum_efg = 0.0d0
      n_efg = 0

      do isite=1,nsites
         do idir=1,3
            probes_p = probes
            probes_m = probes
            probes_p(idir,isite) = probes_p(idir,isite) + fd_step_bohr
            probes_m(idir,isite) = probes_m(idir,isite) - fd_step_bohr
            call getQuickOEPROP(nsites, probes_p, ierr, efield=efield_p)
            if (ierr /= 0) return
            call getQuickOEPROP(nsites, probes_m, ierr, efield=efield_m)
            if (ierr /= 0) return
            do icomp=1,3
               fd = (efield_p(icomp,isite) - efield_m(icomp,isite))/(2.0d0*fd_step_bohr)
               diff = fd - efg(icomp,idir,isite)
               efg_max_abs = max(efg_max_abs, abs(diff))
               accum_efg = accum_efg + diff*diff
               n_efg = n_efg + 1
            enddo
         enddo
      enddo

      efg_rms = dsqrt(accum_efg/dble(max(1,n_efg)))
   end subroutine validate_efg

   subroutine finite_difference_gradients(csv_file, qm_xyz, pc, grad_qm, grad_pc, &
         qm_max_abs, qm_rms, pc_max_abs, pc_rms, ierr)
      implicit none

      character(len=*), intent(in) :: csv_file
      double precision, intent(in) :: qm_xyz(3,natom_qm)
      double precision, intent(in) :: pc(4,nsites)
      double precision, intent(in) :: grad_qm(3,natom_qm)
      double precision, intent(in) :: grad_pc(3,nsites)
      double precision, intent(out) :: qm_max_abs, qm_rms, pc_max_abs, pc_rms
      integer, intent(out) :: ierr

      integer :: unit, iatom, isite, idir, n_qm, n_pc
      double precision :: qm_p(3,natom_qm), qm_m(3,natom_qm)
      double precision :: pc_p(4,nsites), pc_m(4,nsites)
      double precision :: energy_p, energy_m, fd, diff
      double precision :: gtmp_qm(3,natom_qm), gtmp_pc(3,nsites)
      double precision :: accum_qm, accum_pc

      ierr = 0
      qm_max_abs = 0.0d0
      pc_max_abs = 0.0d0
      accum_qm = 0.0d0
      accum_pc = 0.0d0
      n_qm = 0
      n_pc = 0

      if (master) then
         open(newunit=unit, file=csv_file, status='replace', action='write', iostat=ierr)
         if (ierr /= 0) return
         write(unit,'(A)') 'group,index,axis,e_minus_au,e_plus_au,fd_grad_au_per_bohr,analytic_grad_au_per_bohr,abs_diff'
      endif

      do iatom=1,natom_qm
         do idir=1,3
            qm_p = qm_xyz
            qm_m = qm_xyz
            qm_p(idir,iatom) = qm_p(idir,iatom) + fd_step_bohr*BOHRS_TO_A
            qm_m(idir,iatom) = qm_m(idir,iatom) - fd_step_bohr*BOHRS_TO_A
            call evaluate(qm_p, pc, energy_p, gtmp_qm, gtmp_pc, ierr)
            if (ierr /= 0) return
            call evaluate(qm_m, pc, energy_m, gtmp_qm, gtmp_pc, ierr)
            if (ierr /= 0) return
            fd = (energy_p - energy_m)/(2.0d0*fd_step_bohr)
            diff = fd - grad_qm(idir,iatom)
            qm_max_abs = max(qm_max_abs, abs(diff))
            accum_qm = accum_qm + diff*diff
            n_qm = n_qm + 1
            if (master) then
               write(unit,'(A,",",I0,",",A,",",ES24.16,",",ES24.16,",",ES24.16,",",ES24.16,",",ES12.4)') &
                  'qm_atom', iatom, axis_name(idir), energy_m, energy_p, fd, grad_qm(idir,iatom), abs(diff)
            endif
         enddo
      enddo

      do isite=1,nsites
         do idir=1,3
            pc_p = pc
            pc_m = pc
            pc_p(idir,isite) = pc_p(idir,isite) + fd_step_bohr*BOHRS_TO_A
            pc_m(idir,isite) = pc_m(idir,isite) - fd_step_bohr*BOHRS_TO_A
            call evaluate(qm_xyz, pc_p, energy_p, gtmp_qm, gtmp_pc, ierr)
            if (ierr /= 0) return
            call evaluate(qm_xyz, pc_m, energy_m, gtmp_qm, gtmp_pc, ierr)
            if (ierr /= 0) return
            fd = (energy_p - energy_m)/(2.0d0*fd_step_bohr)
            diff = fd - grad_pc(idir,isite)
            pc_max_abs = max(pc_max_abs, abs(diff))
            accum_pc = accum_pc + diff*diff
            n_pc = n_pc + 1
            if (master) then
               write(unit,'(A,",",I0,",",A,",",ES24.16,",",ES24.16,",",ES24.16,",",ES24.16,",",ES12.4)') &
                  'mm_site', isite, axis_name(idir), energy_m, energy_p, fd, grad_pc(idir,isite), abs(diff)
            endif
         enddo
      enddo

      if (master) close(unit)
      qm_rms = dsqrt(accum_qm/dble(max(1,n_qm)))
      pc_rms = dsqrt(accum_pc/dble(max(1,n_pc)))
   end subroutine finite_difference_gradients

   subroutine write_summary(json_file, energy0, qm_max_abs, qm_rms, pc_max_abs, pc_rms, efg_max_abs, efg_rms, ierr)
      implicit none

      character(len=*), intent(in) :: json_file
      double precision, intent(in) :: energy0, qm_max_abs, qm_rms, pc_max_abs, pc_rms
      double precision, intent(in) :: efg_max_abs, efg_rms
      integer, intent(out) :: ierr

      integer :: unit

      ierr = 0
      open(newunit=unit, file=json_file, status='replace', action='write', iostat=ierr)
      if (ierr /= 0) return
      write(unit,'(A)') '{'
      write(unit,'(A,ES24.16,A)') '  "base_energy_au": ', energy0, ','
      write(unit,'(A,ES12.4,A)') '  "fd_step_bohr": ', fd_step_bohr, ','
      write(unit,'(A)') '  "permanent_gradient": {'
      write(unit,'(A,ES12.4,A)') '    "qm_max_abs_au_per_bohr": ', qm_max_abs, ','
      write(unit,'(A,ES12.4,A)') '    "qm_rms_au_per_bohr": ', qm_rms, ','
      write(unit,'(A,ES12.4,A)') '    "mm_site_max_abs_au_per_bohr": ', pc_max_abs, ','
      write(unit,'(A,ES12.4,A)') '    "mm_site_rms_au_per_bohr": ', pc_rms
      write(unit,'(A)') '  },'
      write(unit,'(A)') '  "oeprop_derivatives": {'
      write(unit,'(A,ES12.4,A)') '    "efg_vs_fd_efield_max_abs": ', efg_max_abs, ','
      write(unit,'(A,ES12.4,A)') '    "efg_vs_fd_efield_rms": ', efg_rms, ','
      write(unit,'(A)') '    "probe_note": "EFG probes are offset from validation point charges to avoid self-field singularities."'
      write(unit,'(A)') '  }'
      write(unit,'(A)') '}'
      close(unit)
   end subroutine write_summary

   character(len=1) function axis_name(idir)
      implicit none

      integer, intent(in) :: idir

      select case(idir)
      case(1)
         axis_name = 'x'
      case(2)
         axis_name = 'y'
      case default
         axis_name = 'z'
      end select
   end function axis_name

   subroutine stop_on_error(ierr_in, label)
      implicit none

      integer, intent(in) :: ierr_in
      character(len=*), intent(in) :: label

      if (ierr_in /= 0) then
         write(*,'("QMMM_PERM_GRAD_PROBE_ERROR ",A,1X,I0)') trim(label), ierr_in
#ifdef MPIV
         call MPI_ABORT(MPI_COMM_WORLD,ierr_in,mpierror)
#else
         stop 1
#endif
      endif
   end subroutine stop_on_error

end program quick_qmmm_perm_grad_probe
