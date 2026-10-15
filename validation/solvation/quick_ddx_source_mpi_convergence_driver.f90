#include "util.fh"
!
! (C) Copyright 2026 QUICK contributors. Mozilla Public License, v. 2.0.
! Two-rank source broadcast and collective molecular-gradient smoke test.
!
program quick_ddx_source_mpi_convergence_driver
   use mpi
   use quick_api_module, only: setQuickMPI, setQuickJob, getQuickEnergyGradients, deleteQuickJob, quick_api
   use quick_method_module, only: quick_method
   use quick_mpi_module, only: quick_comm, quick_comm_rank
   use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
   implicit none
   integer :: ierr, comm_error, cleanup, low_source, high_source, unit, a
   integer :: numbers(3)=[8,1,1]
   double precision :: xyz(3,3), energy, grad(3,3), q(4,0), qgrad(3,0), low_energy, high_energy
   double precision :: low_grad(3,3), high_grad(3,3)
   character(len=80) :: jobname
   character(len=256) :: keyword
   character(len=16) :: model
   call MPI_INIT(comm_error)
   ierr=0
   call setQuickMPI(MPI_COMM_WORLD,ierr)
   if(ierr/=0) call MPI_ABORT(MPI_COMM_WORLD,1,comm_error)
   if(command_argument_count()/=1) call MPI_ABORT(quick_comm,2,comm_error)
   call get_command_argument(1,model)
   if(trim(model)/='PCM'.and.trim(model)/='COSMO') call MPI_ABORT(quick_comm,3,comm_error)
   xyz(:,1)=[-0.33840d0,0.00380d0,0.23923d0]
   xyz(:,2)=[-0.33510d0,-0.00190d0,-0.83277d0]
   xyz(:,3)=[0.67350d0,-0.00190d0,0.59353d0]
   keyword='HF BASIS=STO-3G GRADIENT SCF=250 DENSERMS=1e-10 CUTOFF=1e-12 GRADCUTOFF=1e-12 DDX='// &
      trim(model)//' DDX_EPS=78.3553 DDX_SOURCE=COULOMB DDX_LMAX=6 DDX_NLEBEDEV=302 DDX_NRAD=50 DDX_NANG=110 DDX_TOL=1e-10'
   jobname='source_mpi'
   call setQuickJob(jobname,keyword,3,numbers,.false.,ierr)
   if(ierr/=0) call MPI_ABORT(quick_comm,4,comm_error)
   call MPI_ALLREDUCE(quick_method%ddx_source,low_source,1,MPI_INTEGER,MPI_MIN,quick_comm,comm_error)
   call MPI_ALLREDUCE(quick_method%ddx_source,high_source,1,MPI_INTEGER,MPI_MAX,quick_comm,comm_error)
   if(low_source/=2.or.high_source/=2) call MPI_ABORT(quick_comm,5,comm_error)
   call getQuickEnergyGradients(xyz,0,q,energy,grad,qgrad,ierr)
   if(ierr/=0.or..not.quick_method%scf_conv) call MPI_ABORT(quick_comm,6,comm_error)
   if(.not.ieee_is_finite(energy).or..not.all(ieee_is_finite(grad))) call MPI_ABORT(quick_comm,7,comm_error)
   call MPI_ALLREDUCE(energy,low_energy,1,MPI_DOUBLE_PRECISION,MPI_MIN,quick_comm,comm_error)
   call MPI_ALLREDUCE(energy,high_energy,1,MPI_DOUBLE_PRECISION,MPI_MAX,quick_comm,comm_error)
   call MPI_ALLREDUCE(grad,low_grad,9,MPI_DOUBLE_PRECISION,MPI_MIN,quick_comm,comm_error)
   call MPI_ALLREDUCE(grad,high_grad,9,MPI_DOUBLE_PRECISION,MPI_MAX,quick_comm,comm_error)
   if(energy/=low_energy.or.energy/=high_energy.or.any(low_grad/=high_grad)) call MPI_ABORT(quick_comm,8,comm_error)
   if(quick_api%step/=2) call MPI_ABORT(quick_comm,9,comm_error)
   if(quick_comm_rank==0) then
      open(newunit=unit,file='source_mpi.dat',status='new')
      write(unit,'(A,ES26.17E3)') 'energy ',energy
      write(unit,'(A,2I4)') 'source_minmax ',low_source,high_source
      do a=1,3
         write(unit,'(A,I0,3(1X,ES26.17E3))') 'gradient ',a,grad(:,a)
      enddo
      close(unit)
      write(*,'(A)') 'source_mpi_complete consistent_source_and_identical_collective_results_pass'
   endif
   cleanup=0
   call deleteQuickJob(cleanup)
   if(cleanup/=0) call MPI_ABORT(MPI_COMM_WORLD,10,comm_error)
   call MPI_FINALIZE(comm_error)
end program quick_ddx_source_mpi_convergence_driver
