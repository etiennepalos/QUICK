#include "util.fh"
!
! (C) Copyright 2026 QUICK contributors. Mozilla Public License, v. 2.0.
! Retained-density mode changes must invalidate the geometry/source cache.
!
program quick_ddx_source_cache_convergence_driver
   use quick_api_module, only: setQuickJob, getQuickEnergy, deleteQuickJob, quick_api
   use quick_method_module, only: quick_method
   use quick_calculated_module, only: quick_qm_struct
   use quick_basis_module, only: nbasis
   use quick_solvation_module, only: get_solvation_contributions
   implicit none
   integer :: ierr, cleanup, unit, natoms, atom, saved_step
   integer, allocatable :: numbers(:)
   double precision, allocatable :: coords(:,:), density(:,:), fock(:,:), regular(:,:), coulomb(:,:), restored(:,:)
   double precision :: energy, e1, e2, e3, saved_eel, saved_etot, point_charges(4,0)
   double precision :: restore_energy_error, restore_matrix_error, changed_matrix
   character(len=80) :: jobname
   character(len=256) :: keyword
   character(len=512) :: filename
   if(command_argument_count()/=1) stop 1
   call get_command_argument(1,filename)
   open(newunit=unit,file=trim(filename),status='old')
   read(unit,'(A)') keyword
   read(unit,*) natoms
   allocate(numbers(natoms),coords(3,natoms))
   do atom=1,natoms
      read(unit,*) numbers(atom),coords(:,atom)
   enddo
   close(unit)
   ierr=0
   jobname='source_cache'
   call setQuickJob(jobname,keyword,natoms,numbers,.false.,ierr)
   if(ierr/=0) stop 2
   if(quick_method%ddx_source/=1) stop 3
   call getQuickEnergy(coords,0,point_charges,energy,ierr)
   if(ierr/=0.or..not.quick_method%scf_conv) stop 4
   density=quick_qm_struct%dense
   fock=quick_qm_struct%o
   saved_step=quick_api%step
   saved_eel=quick_qm_struct%Eel
   saved_etot=quick_qm_struct%Etot
   allocate(regular(nbasis,nbasis),coulomb(nbasis,nbasis),restored(nbasis,nbasis))
   call get_solvation_contributions(regular,e1)
   quick_method%ddx_source=2
   call get_solvation_contributions(coulomb,e2)
   quick_method%ddx_source=1
   call get_solvation_contributions(restored,e3)
   restore_energy_error=abs(e1-e3)
   restore_matrix_error=maxval(abs(regular-restored))
   changed_matrix=maxval(abs(regular-coulomb))
   open(newunit=unit,file='source_cache.dat',status='new')
   write(unit,'(A,ES26.17E3)') 'restored_energy_error ',restore_energy_error
   write(unit,'(A,ES26.17E3)') 'restored_matrix_error ',restore_matrix_error
   write(unit,'(A,ES26.17E3)') 'changed_matrix ',changed_matrix
   write(unit,'(A,ES26.17E3)') 'changed_energy ',abs(e1-e2)
   write(unit,'(A,ES26.17E3)') 'density_change ',maxval(abs(density-quick_qm_struct%dense))
   write(unit,'(A,ES26.17E3)') 'fock_change ',maxval(abs(fock-quick_qm_struct%o))
   write(unit,'(A,ES26.17E3)') 'stored_energy_change ',max(abs(saved_eel-quick_qm_struct%Eel),abs(saved_etot-quick_qm_struct%Etot))
   write(unit,'(A,I0)') 'api_step_change ',quick_api%step-saved_step
   close(unit)
   if(restore_energy_error>1.d-9.or.restore_matrix_error>1.d-8.or.changed_matrix<=1.d-10) stop 5
   if(any(density/=quick_qm_struct%dense).or.any(fock/=quick_qm_struct%o)) stop 6
   if(saved_eel/=quick_qm_struct%Eel.or.saved_etot/=quick_qm_struct%Etot.or.quick_api%step/=saved_step) stop 7
   cleanup=0
   call deleteQuickJob(cleanup)
   if(cleanup/=0) stop 8
   write(*,'(A)') 'source_cache_complete regular_to_coulomb_to_regular_pass'
end program quick_ddx_source_cache_convergence_driver
