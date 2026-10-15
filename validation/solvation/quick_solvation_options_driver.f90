#include "util.fh"
! (C) Copyright 2026 QUICK contributors. Mozilla Public License, v. 2.0.
program quick_solvation_options_driver
   use quick_method_module, only: quick_method_type, init, read, check
   use quick_ddx_backend_module, only: quick_ddx_available
   implicit none
   type(quick_method_type) :: method
   integer :: ierr, unit, failures, checks

   failures = 0
   checks = 0
   open(newunit=unit,status='scratch',action='readwrite')
   call options('HF DDX=PCM',quick_ddx_available)
   call assert(method%ddx .and. method%ddx_model == 2,'PCM model selection')
   call options('HF DDX_MODEL=DDCOSMO',quick_ddx_available)
   call assert(method%ddx .and. method%ddx_model == 1,'COSMO model alias')
   call options('HF NOT_DDX=PCM DDX_EPSILON=3.0 MY_DDX_LMAX=1',.true.)
   call assert(.not.method%ddx .and. method%ddx_lmax == 6 .and. &
      method%ddx_epsilon == 78.3553d0,'boundary-aware solvent tokens')
   call options('HF DDX=PCM DDX_EPS=1',quick_ddx_available)
   call options('HF DDX=COSMO EXTERNAL_EFIELD_X=0.001',.false.)
   call options('HF DDX=COSMO EXTERNAL_EFIELD_X=0.001 DDX_FIELD=SOLUTE',quick_ddx_available)
   call assert(method%consistent_optimizer_state,'full SCF state with field and solvent')
   call options('HF DDX=PCM DDX_FIELD=MACROSCOPIC',.false.)
   call options('HF DDX=LPB',.false.)
   call options('HF DDX=PCM DDX_EPS=0.99',.false.)
   call options('HF DDX=PCM DDX_EPS=NaN',.false.)
   call options('HF DDX=PCM DDX_LMAX=-1',.false.)
   call options('HF DDX=PCM DDX_LMAX=21',.false.)
   call options('HF DDX=PCM DDX_NRAD=0',.false.)
   call options('HF DDX=PCM DDX_RADIUS_SCALE=-1',.false.)
   call options('HF DDX=PCM DDX_TOL=0',.false.)
   call options('HF DDX=PCM DDX_TOL=1',.false.)
   call options('HF DDX=PCM ECP=LANL2DZ',.false.)
   call options('HF DDX=PCM PBSOL',.false.)
   call options('MP2 DDX=PCM',.false.)
   call options('HF EXTERNAL_EFIELD_X=NaN',.false.)
   call options('HF FINITE_FIELD_ORIGIN_Y=NaN',.false.)
   call options('HF FINITE_FIELD_X=0.001 FINITE_FIELD_ORIGIN_Z=2',.true.)
   call assert(method%external_efield .and. method%external_efield_vector(1) == 0.001d0 .and. &
      method%external_efield_origin(3) == 2.d0,'finite-field aliases')
   call options('HF',.true.)
   call assert(.not.method%ddx .and. .not.method%ddx_field_solute .and. &
      .not.method%external_efield .and. all(method%external_efield_vector == 0.d0) .and. &
      all(method%external_efield_origin == 0.d0) .and. method%ddx_model == 1 .and. &
      method%ddx_nrad == 75 .and. method%ddx_nang == 302 .and. method%ddx_nlebedev == 302 .and. &
      method%ddx_radius_scale == 1.1d0 .and. method%ddx_tolerance == 1.d-10, &
      'new-job defaults restored after prior solvent and field')
   close(unit)
   write(*,'(A,I0,A,I0)') 'option_checks ',checks,' failures ',failures
   if (failures /= 0) stop 1
contains
   subroutine options(card,expected_success)
      implicit none
      character(len=*), intent(in) :: card
      logical, intent(in) :: expected_success
      character(len=300) :: keyword
      keyword = card
      ierr = 0
      call init(method,ierr)
      call read(method,keyword,ierr)
      if (ierr == 0) call check(method,unit,ierr)
      call assert((ierr == 0) .eqv. expected_success,trim(card))
   end subroutine options
   subroutine assert(passed,label)
      implicit none
      logical, intent(in) :: passed
      character(len=*), intent(in) :: label
      checks = checks + 1
      if (.not.passed) failures = failures + 1
      write(*,'(L1,1X,A)') passed,label
   end subroutine assert
end program quick_solvation_options_driver
