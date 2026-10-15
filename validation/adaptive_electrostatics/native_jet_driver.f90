! Raw independent jet/majorant validation client; no SCF or GPU execution.
program native_jet_driver
   use quick_electrostatic_jet_module
   use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
   implicit none
   integer :: nb,na,mp,np,ncell,mode,unit_in,unit_out,iv,ib,i
   integer, allocatable :: nc(:),ang(:,:),centers(:),owned(:)
   double precision, allocatable :: expo(:,:),coeff(:,:),atoms(:,:),density(:,:),points(:,:)
   double precision, allocatable :: lower(:,:),upper(:,:),v(:),e(:,:),g(:,:,:),h(:,:,:,:),b(:,:)
   type(electrostatic_jet_status) :: sv,sb
   character(len=2048) :: input_path,output_path

   call get_command_argument(1,input_path)
   call get_command_argument(2,output_path)
   open(newunit=unit_in,file=trim(input_path),status='old',action='read')
   read(unit_in,*) nb,na,mp,np,ncell,mode
   allocate(nc(nb),ang(3,nb),centers(nb),owned(nb),expo(mp,nb),coeff(mp,nb),atoms(3,na), &
      density(nb,nb),points(3,np),lower(3,ncell),upper(3,ncell),v(np),e(3,np),g(3,3,np), &
      h(3,3,3,np),b(35,ncell))
   read(unit_in,*) nc
   read(unit_in,*) ang
   read(unit_in,*) centers
   read(unit_in,*) owned
   read(unit_in,*) expo
   read(unit_in,*) coeff
   read(unit_in,*) atoms
   read(unit_in,*) density
   read(unit_in,*) points
   read(unit_in,*) lower
   read(unit_in,*) upper
   close(unit_in)
   if (mode == 1) expo(1,1) = -1.0d0
   if (mode == 2) points(1,1) = ieee_value(0.0d0,ieee_quiet_nan)
   if (mode == 3) lower(1,1) = upper(1,1)+1.0d0
   if (mode == 4) density(1,1) = ieee_value(0.0d0,ieee_quiet_nan)
   if (mode == 5) points(1,1) = huge(1.0d0)
   v=0.125d0
   e=-0.25d0
   g=0.5d0
   h=-0.75d0
   b=0.0625d0
   call electronic_jet_cpu(nb,na,mp,nc,ang,centers,expo,coeff,atoms,density,np,points,owned,v,e,g,h,sv,iv)
   call electronic_cell_majorants_cpu(nb,na,mp,nc,ang,centers,expo,coeff,atoms,density,ncell,lower,upper, &
      owned,b,sb,ib)
   open(newunit=unit_out,file=trim(output_path),status='replace',action='write')
   write(unit_out,*) iv,ib
   write(unit_out,*) sv%no_intentional_screen,sv%numerical_budget_known,sv%certified,sv%underflow_detected, &
      sv%primitive_pairs,sv%prefactor_underflow_count,sv%boys_underflow_count
   write(unit_out,*) sb%no_intentional_screen,sb%numerical_budget_known,sb%certified,sb%underflow_detected, &
      sb%primitive_pairs,sb%prefactor_underflow_count,sb%boys_underflow_count
   do i=1,np
      write(unit_out,'(40(es26.17,1x))') v(i),e(:,i),g(:,:,i),h(:,:,:,i)
   enddo
   do i=1,ncell
      write(unit_out,'(37(es26.17,1x))') b(:,i),jet_rank_norm(b(:,i)-0.0625d0,3), &
         jet_rank_norm(b(:,i)-0.0625d0,4)
   enddo
   close(unit_out)
end program native_jet_driver
