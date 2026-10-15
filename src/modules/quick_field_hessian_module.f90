#include "util.fh"
!---------------------------------------------------------------------!
! Analytic spatial Hessian of the electric field on observation points.!
! H_ijk(C) = d^2 E_i(C)/(dC_j dC_k) = -d^3 V(C)/(dC_i dC_j dC_k).     !
! This independent implementation leaves ESP/EFIELD/EFG routes intact. !
! All coordinates are bohr; H is in Eh/(e bohr^3).                    !
! (C) Copyright 2026 QUICK contributors. Mozilla Public License, v2.0. !
!---------------------------------------------------------------------!
module quick_field_hessian_module
   use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
   implicit none
   private
   public :: compute_field_hessian_values, field_hessian_cpu
   public :: get_field_hessian_gpu_count

   integer, parameter :: MAX_ANGULAR = 3, MAX_HERMITE = 2*MAX_ANGULAR + 3
   integer, parameter :: DIMENSION_ERROR = 46, NUMERIC_ERROR = 55, BACKEND_ERROR = 56
   integer, parameter :: COMPONENT_I(10) = [1,1,1,1,1,1,2,2,2,3]
   integer, parameter :: COMPONENT_J(10) = [1,1,1,2,2,3,2,2,3,3]
   integer, parameter :: COMPONENT_K(10) = [1,2,3,2,3,3,2,3,3,3]

#if defined(CUDA) || defined(CUDA_MPIV)
   interface
      subroutine gpu_field_hessian(nb,na,maxprim,nc,ang,centers,expo,coeff,atoms,density, &
            npoints,points,owned,cutoff,result,ierr)
         implicit none
         integer, intent(in) :: nb, na, maxprim, npoints
         integer, intent(in) :: nc(nb), ang(3,nb), centers(nb), owned(nb)
         double precision, intent(in) :: expo(maxprim,nb), coeff(maxprim,nb), atoms(3,na)
         double precision, intent(in) :: density(nb,nb), points(3,npoints), cutoff
         double precision, intent(inout) :: result(3,3,3,npoints)
         integer, intent(out) :: ierr
      end subroutine gpu_field_hessian
      subroutine gpu_field_hessian_launch_count(count)
         implicit none
         integer, intent(out) :: count
      end subroutine gpu_field_hessian_launch_count
   end interface
#endif

contains

   subroutine compute_field_hessian_values(npoints,xyz_points,field_hessian,ierr)
      !---------------------------------------------------------------------!
      ! Source-only field Hessian from the converged, spin-summed density,  !
      ! nuclei and supplied external point charges. Direct applied/solvent !
      ! backgrounds are excluded, as in the existing OEPROP grid routines. !
      ! The raw CPU/CUDA evaluator adds only the electronic contribution. !
      ! MPI owns the higher index of each triangular AO pair; classical   !
      ! sources occur once on master after reduction on quick_comm.        !
      ! Errors are collective; caller output is unchanged on any failure. !
      ! On success, nonmaster output is zero for the public API broadcast. !
      !---------------------------------------------------------------------!
      use quick_basis_module, only: quick_basis, nbasis, ncontract, itype, aexp, dcoeff
      use quick_calculated_module, only: quick_qm_struct
      use quick_method_module, only: quick_method
      use quick_molspec_module, only: natom, xyz
      use quick_mpi_module, only: master
#ifdef MPIV
      use mpi
      use quick_basis_module, only: mpi_nbasisn, mpi_nbasis
      use quick_mpi_module, only: bMPI, quick_comm, quick_comm_rank, quick_mpi_error
#endif
      implicit none
      integer, intent(in) :: npoints
      double precision, intent(in) :: xyz_points(:,:)
      double precision, intent(inout) :: field_hessian(:,:,:,:)
      integer, intent(out) :: ierr
      integer, allocatable :: owned(:)
      double precision, allocatable :: density(:,:), work(:,:,:,:), aggregate(:,:,:,:)
      integer :: allocation_status
#ifdef MPIV
      integer :: irow
#endif

      ierr = 0
      if (npoints < 1 .or. npoints > huge(npoints)/27) ierr = DIMENSION_ERROR
      if (size(xyz_points,1) /= 3 .or. size(xyz_points,2) /= npoints) ierr = DIMENSION_ERROR
      if (any(shape(field_hessian) /= [3,3,3,npoints])) ierr = DIMENSION_ERROR
      if (ierr == 0) then
         if (.not.all(ieee_is_finite(xyz_points))) ierr = NUMERIC_ERROR
      endif
      if (ierr == 0) call validate_host_state(ierr)
      call synchronize_error(ierr)
      if (ierr /= 0) return

      allocate(owned(nbasis),density(nbasis,nbasis),work(3,3,3,npoints), &
         aggregate(3,3,3,npoints),stat=allocation_status)
      if (allocation_status /= 0) ierr = BACKEND_ERROR
      call synchronize_error(ierr)
      if (ierr /= 0) return
      density = quick_qm_struct%dense
      if (quick_method%unrst) density = density + quick_qm_struct%denseb
      owned = 1
#ifdef MPIV
      if (bMPI) then
         if (.not.allocated(mpi_nbasisn) .or. .not.allocated(mpi_nbasis)) then
            ierr = NUMERIC_ERROR
         else if (quick_comm_rank < lbound(mpi_nbasisn,1) .or. &
                  quick_comm_rank > ubound(mpi_nbasisn,1)) then
            ierr = NUMERIC_ERROR
         else if (quick_comm_rank < lbound(mpi_nbasis,1) .or. &
                  quick_comm_rank > ubound(mpi_nbasis,1)) then
            ierr = NUMERIC_ERROR
         else if (mpi_nbasisn(quick_comm_rank) < 0 .or. &
                  mpi_nbasisn(quick_comm_rank) > size(mpi_nbasis,2)) then
            ierr = NUMERIC_ERROR
         else
            owned = 0
            do irow=1,mpi_nbasisn(quick_comm_rank)
               if (mpi_nbasis(quick_comm_rank,irow) < 1 .or. &
                   mpi_nbasis(quick_comm_rank,irow) > nbasis) then
                  ierr = NUMERIC_ERROR
                  exit
               endif
               owned(mpi_nbasis(quick_comm_rank,irow)) = 1
            enddo
         endif
      endif
#endif
      call synchronize_error(ierr)
      if (ierr /= 0) return
      work = 0.0d0
#if defined(CUDA) || defined(CUDA_MPIV)
      if (quick_method%bGPU) then
         call gpu_field_hessian(nbasis,natom,size(aexp,1),ncontract,itype,quick_basis%ncenter, &
            aexp,dcoeff,xyz,density,npoints,xyz_points,owned,quick_method%coreIntegralCutoff,work,ierr)
      else
#endif
         call field_hessian_cpu(nbasis,natom,size(aexp,1),ncontract,itype,quick_basis%ncenter, &
            aexp,dcoeff,xyz,density,npoints,xyz_points,owned,quick_method%coreIntegralCutoff,work,ierr)
#if defined(CUDA) || defined(CUDA_MPIV)
      endif
#endif
      call synchronize_error(ierr)
      if (ierr /= 0) return
#ifdef MPIV
      if (bMPI) then
         call MPI_REDUCE(work,aggregate,27*npoints,MPI_DOUBLE_PRECISION,MPI_SUM,0,quick_comm,quick_mpi_error)
         if (quick_mpi_error /= MPI_SUCCESS) ierr = BACKEND_ERROR
      else
         aggregate = work
      endif
#else
      aggregate = work
#endif
      call synchronize_error(ierr)
      if (ierr /= 0) return
      if (master) call add_classical_hessian(npoints,xyz_points,aggregate,ierr)
      call synchronize_error(ierr)
      if (ierr /= 0) return
      field_hessian = 0.0d0
      if (master) field_hessian = aggregate
   end subroutine compute_field_hessian_values

   subroutine validate_host_state(ierr)
      use quick_basis_module, only: quick_basis, nbasis, ncontract, itype, aexp, dcoeff
      use quick_calculated_module, only: quick_qm_struct
      use quick_method_module, only: quick_method
      use quick_molspec_module, only: quick_molspec, natom, xyz
      implicit none
      integer, intent(out) :: ierr

      ierr = DIMENSION_ERROR
      if (nbasis < 1 .or. natom < 1) return
      if (nbasis > huge(nbasis)/nbasis) return
      ierr = NUMERIC_ERROR
      if (.not.allocated(ncontract) .or. .not.allocated(itype)) return
      if (.not.allocated(aexp) .or. .not.allocated(dcoeff)) return
      if (.not.allocated(quick_basis%ncenter) .or. .not.allocated(xyz)) return
      if (.not.allocated(quick_qm_struct%dense) .or. .not.allocated(quick_molspec%chg)) return
      if (size(ncontract) /= nbasis .or. size(quick_basis%ncenter) /= nbasis) return
      if (any(shape(itype) /= [3,nbasis])) return
      if (size(aexp,1) < 1 .or. size(aexp,2) /= nbasis) return
      if (any(shape(aexp) /= shape(dcoeff))) return
      if (any(shape(quick_qm_struct%dense) /= [nbasis,nbasis])) return
      if (any(shape(xyz) /= [3,natom]) .or. size(quick_molspec%chg) /= natom) return
      if (quick_method%unrst) then
         if (.not.allocated(quick_qm_struct%denseb)) return
         if (any(shape(quick_qm_struct%denseb) /= [nbasis,nbasis])) return
      endif
      if (.not.all(ieee_is_finite(quick_molspec%chg))) return
      if (quick_molspec%nextatom < 0) return
      if (quick_molspec%nextatom > 0) then
         if (.not.allocated(quick_molspec%extxyz) .or. .not.allocated(quick_molspec%extchg)) return
         if (any(shape(quick_molspec%extxyz) /= [3,quick_molspec%nextatom])) return
         if (size(quick_molspec%extchg) /= quick_molspec%nextatom) return
         if (.not.all(ieee_is_finite(quick_molspec%extxyz))) return
         if (.not.all(ieee_is_finite(quick_molspec%extchg))) return
      endif
      ierr = 0
   end subroutine validate_host_state

   subroutine synchronize_error(ierr)
#ifdef MPIV
      use mpi
      use quick_mpi_module, only: bMPI, quick_comm, quick_mpi_error
#endif
      implicit none
      integer, intent(inout) :: ierr
#ifdef MPIV
      integer :: aggregate_error
      if (bMPI) then
         call MPI_ALLREDUCE(ierr,aggregate_error,1,MPI_INTEGER,MPI_MAX,quick_comm,quick_mpi_error)
         if (quick_mpi_error == MPI_SUCCESS) then
            ierr = aggregate_error
         else
            ierr = BACKEND_ERROR
         endif
      endif
#endif
   end subroutine synchronize_error

   subroutine get_field_hessian_gpu_count(count)
      implicit none
      integer, intent(out) :: count
      count = 0
#if defined(CUDA) || defined(CUDA_MPIV)
      call gpu_field_hessian_launch_count(count)
#endif
   end subroutine get_field_hessian_gpu_count

   subroutine field_hessian_cpu(nb,na,maxprim,nc,ang,centers,expo,coeff,atoms,density, &
         npoints,points,owned,cutoff,result,ierr)
      !---------------------------------------------------------------------!
      ! Raw electronic reference, with the exact native-CUDA argument list. !
      ! Normalized Cartesian contraction coefficients are supplied intact. !
      ! Ownership refers to mu in mu>=nu; off-diagonal density is doubled. !
      ! Screen the original primitive overlap prefactor, without density.  !
      ! V_e=-sum(D*I), so H_e=+sum(D*d_C^3 I). Hermite R derivatives use    !
      ! P-C: the three observation derivatives supply the minus sign below.!
      ! Add to result only after every calculation succeeds.                !
      !---------------------------------------------------------------------!
      use quick_constants_module, only: pi
      implicit none
      integer, intent(in) :: nb, na, maxprim, npoints
      integer, intent(in) :: nc(nb), ang(3,nb), centers(nb), owned(nb)
      double precision, intent(in) :: expo(maxprim,nb), coeff(maxprim,nb), atoms(3,na)
      double precision, intent(in) :: density(nb,nb), points(3,npoints), cutoff
      double precision, intent(inout) :: result(3,3,3,npoints)
      integer, intent(out) :: ierr
      double precision, allocatable :: values(:,:), total(:,:,:,:)
      double precision :: hermite(0:2*MAX_ANGULAR,3)
      double precision :: rtable(0:MAX_HERMITE,0:MAX_HERMITE,0:MAX_HERMITE,0:MAX_HERMITE)
      double precision :: alpha, beta, g, inverse_g, reduced, distance2, gaussian
      double precision :: prefactor, overlap_prefactor, weight, contribution, coefficient
      double precision :: center_a(3), center_b(3), product_center(3), pa(3), pb(3), pc(3), boys_t
      integer :: mu, nu, icon, jcon, axis, point, component, i, j, k, t, u, v
      integer :: maxima(3), derivative(3), order, allocation_status

      call validate_raw_request(nb,na,maxprim,nc,ang,centers,expo,coeff,atoms,density, &
         npoints,points,owned,cutoff,result,ierr)
      if (ierr /= 0) return
      if (.not.any(owned == 1)) return
      allocate(values(10,npoints),total(3,3,3,npoints),stat=allocation_status)
      if (allocation_status /= 0) then
         ierr = BACKEND_ERROR
         return
      endif
      values = 0.0d0
      do mu=1,nb
         if (owned(mu) == 0) cycle
         center_a = atoms(:,centers(mu))
         do nu=1,mu
            weight = density(nu,mu)
            if (mu /= nu) weight = 2.0d0*weight
            if (weight == 0.0d0) cycle
            center_b = atoms(:,centers(nu))
            distance2 = sum((center_a-center_b)**2)
            maxima = ang(:,mu)+ang(:,nu)
            order = sum(maxima)+3
            do icon=1,nc(mu)
               alpha = expo(icon,mu)
               do jcon=1,nc(nu)
                  beta = expo(jcon,nu)
                  g = alpha+beta
                  if (.not.ieee_is_finite(g) .or. g <= 0.0d0) then
                     ierr = NUMERIC_ERROR
                     return
                  endif
                  inverse_g = 1.0d0/g
                  reduced = alpha*(beta*inverse_g)
                  gaussian = exp(-reduced*distance2)
                  overlap_prefactor = gaussian*sqrt(pi*inverse_g)*pi*inverse_g &
                     *coeff(jcon,nu)*coeff(icon,mu)
                  if (.not.ieee_is_finite(overlap_prefactor)) then
                     ierr = NUMERIC_ERROR
                     return
                  endif
                  if (abs(overlap_prefactor) <= cutoff) cycle
                  prefactor = gaussian*2.0d0*pi*inverse_g*coeff(jcon,nu)*coeff(icon,mu)*weight
                  pa = beta*inverse_g*(center_b-center_a)
                  pb = -alpha*inverse_g*(center_b-center_a)
                  product_center = center_a+pa
                  do axis=1,3
                     call hermite_coefficients(ang(axis,mu),ang(axis,nu),pa(axis),pb(axis), &
                        0.5d0*inverse_g,hermite(:,axis))
                  enddo
                  do point=1,npoints
                     pc = product_center-points(:,point)
                     boys_t = g*sum(pc*pc)
                     if (.not.ieee_is_finite(boys_t) .or. boys_t < 0.0d0) then
                        ierr = NUMERIC_ERROR
                        return
                     endif
                     call coulomb_derivatives(order,g,boys_t,pc,rtable,ierr)
                     if (ierr /= 0) return
                     do component=1,10
                        derivative = 0
                        derivative(COMPONENT_I(component)) = derivative(COMPONENT_I(component))+1
                        derivative(COMPONENT_J(component)) = derivative(COMPONENT_J(component))+1
                        derivative(COMPONENT_K(component)) = derivative(COMPONENT_K(component))+1
                        contribution = 0.0d0
                        do t=0,maxima(1)
                           do u=0,maxima(2)
                              coefficient = hermite(t,1)*hermite(u,2)
                              do v=0,maxima(3)
                                 contribution = contribution + coefficient*hermite(v,3) &
                                    *rtable(t+derivative(1),u+derivative(2),v+derivative(3),0)
                              enddo
                           enddo
                        enddo
                        values(component,point) = values(component,point)-prefactor*contribution
                     enddo
                  enddo
               enddo
            enddo
         enddo
      enddo
      if (.not.all(ieee_is_finite(values))) then
         ierr = NUMERIC_ERROR
         return
      endif
      total = result
      do point=1,npoints
         do component=1,10
            i = COMPONENT_I(component)
            j = COMPONENT_J(component)
            k = COMPONENT_K(component)
            ! Add the same number to every permutation, with no duplicate
            ! addition when two or three Cartesian indices coincide.
            do axis=1,3
               do t=1,3
                  do u=1,3
                     if (same_component(axis,t,u,i,j,k)) total(axis,t,u,point) &
                        = total(axis,t,u,point)+values(component,point)
                  enddo
               enddo
            enddo
         enddo
      enddo
      if (.not.all(ieee_is_finite(total))) then
         ierr = NUMERIC_ERROR
         return
      endif
      result = total
   end subroutine field_hessian_cpu

   subroutine validate_raw_request(nb,na,maxprim,nc,ang,centers,expo,coeff,atoms,density, &
         npoints,points,owned,cutoff,result,ierr)
      implicit none
      integer, intent(in) :: nb, na, maxprim, npoints
      integer, intent(in) :: nc(nb), ang(3,nb), centers(nb), owned(nb)
      double precision, intent(in) :: expo(maxprim,nb), coeff(maxprim,nb), atoms(3,na)
      double precision, intent(in) :: density(nb,nb), points(3,npoints), cutoff, result(3,3,3,npoints)
      integer, intent(out) :: ierr
      integer :: mu

      ierr = DIMENSION_ERROR
      if (min(nb,na,maxprim,npoints) < 1 .or. npoints > huge(npoints)/27) return
      if (nb > huge(nb)/nb .or. maxprim > huge(maxprim)/nb) return
      ierr = NUMERIC_ERROR
      if (.not.ieee_is_finite(cutoff) .or. cutoff < 0.0d0) return
      if (.not.all(ieee_is_finite(atoms)) .or. .not.all(ieee_is_finite(points))) return
      if (.not.all(ieee_is_finite(density)) .or. .not.all(ieee_is_finite(result))) return
      if (any(abs(density-transpose(density)) > 1.0d-10)) return
      if (any(owned /= 0 .and. owned /= 1)) return
      if (any(centers < 1 .or. centers > na)) return
      if (any(ang < 0 .or. ang > MAX_ANGULAR)) return
      do mu=1,nb
         if (sum(ang(:,mu)) > MAX_ANGULAR) return
         if (nc(mu) < 1 .or. nc(mu) > maxprim) return
         if (.not.all(ieee_is_finite(expo(1:nc(mu),mu)))) return
         if (.not.all(ieee_is_finite(coeff(1:nc(mu),mu)))) return
         if (any(expo(1:nc(mu),mu) <= 0.0d0)) return
      enddo
      ierr = 0
   end subroutine validate_raw_request

   subroutine hermite_coefficients(la,lb,pa,pb,inverse_two_g,coefficients)
      implicit none
      integer, intent(in) :: la, lb
      double precision, intent(in) :: pa, pb, inverse_two_g
      double precision, intent(out) :: coefficients(0:2*MAX_ANGULAR)
      double precision :: work(0:2*MAX_ANGULAR), displacement
      integer :: raising, degree, t

      coefficients = 0.0d0
      coefficients(0) = 1.0d0
      do raising=1,la+lb
         displacement = pa
         if (raising > la) displacement = pb
         degree = raising-1
         work = 0.0d0
         do t=0,degree+1
            if (t > 0) work(t) = inverse_two_g*coefficients(t-1)
            if (t <= degree) work(t) = work(t)+displacement*coefficients(t)
            if (t < degree) work(t) = work(t)+dble(t+1)*coefficients(t+1)
         enddo
         coefficients = work
      enddo
   end subroutine hermite_coefficients

   subroutine coulomb_derivatives(order,g,boys_t,pc,rtable,ierr)
      implicit none
      integer, intent(in) :: order
      double precision, intent(in) :: g, boys_t, pc(3)
      double precision, intent(out) :: rtable(0:MAX_HERMITE,0:MAX_HERMITE,0:MAX_HERMITE,0:MAX_HERMITE)
      integer, intent(out) :: ierr
      double precision :: boys(0:MAX_HERMITE), factor
      integer :: n, degree, t, u, v

      call boys_values(order,boys_t,boys,ierr)
      if (ierr /= 0) return
      rtable = 0.0d0
      factor = 1.0d0
      do n=0,order
         rtable(0,0,0,n) = factor*boys(n)
         if (n < order) factor = factor*(-2.0d0*g)
      enddo
      do n=order-1,0,-1
         do degree=1,order-n
            do t=0,degree
               do u=0,degree-t
                  v = degree-t-u
                  if (t > 0) then
                     rtable(t,u,v,n) = pc(1)*rtable(t-1,u,v,n+1)
                     if (t > 1) rtable(t,u,v,n) = rtable(t,u,v,n) &
                        +dble(t-1)*rtable(t-2,u,v,n+1)
                  else if (u > 0) then
                     rtable(t,u,v,n) = pc(2)*rtable(t,u-1,v,n+1)
                     if (u > 1) rtable(t,u,v,n) = rtable(t,u,v,n) &
                        +dble(u-1)*rtable(t,u-2,v,n+1)
                  else
                     rtable(t,u,v,n) = pc(3)*rtable(t,u,v-1,n+1)
                     if (v > 1) rtable(t,u,v,n) = rtable(t,u,v,n) &
                        +dble(v-1)*rtable(t,u,v-2,n+1)
                  endif
               enddo
            enddo
         enddo
      enddo
      if (.not.all(ieee_is_finite(rtable))) ierr = NUMERIC_ERROR
   end subroutine coulomb_derivatives

   subroutine boys_values(order,boys_t,values,ierr)
      use quick_constants_module, only: pi
      implicit none
      integer, intent(in) :: order
      double precision, intent(in) :: boys_t
      double precision, intent(out) :: values(0:MAX_HERMITE)
      integer, intent(out) :: ierr
      double precision :: term, series, exponential
      integer :: k, n
      logical :: converged

      ierr = 0
      values = 0.0d0
      exponential = exp(-boys_t)
      if (boys_t <= 32.0d0) then
         ! Positive-term expansion of exp(T)*F_order(T), followed by stable
         ! downward recurrence; no cancellation at a Gaussian product center.
         term = 1.0d0/dble(2*order+1)
         series = term
         converged = .false.
         do k=1,256
            term = term*boys_t/(dble(order+k)+0.5d0)
            series = series+term
            if (abs(term) <= epsilon(series)*abs(series)) then
               converged = .true.
               exit
            endif
         enddo
         if (.not.converged) then
            ierr = NUMERIC_ERROR
            return
         endif
         values(order) = exponential*series
         do n=order-1,0,-1
            values(n) = (2.0d0*boys_t*values(n+1)+exponential)/dble(2*n+1)
         enddo
      else
         values(0) = 0.5d0*sqrt(pi/boys_t)*erf(sqrt(boys_t))
         do n=1,order
            values(n) = (dble(2*n-1)*values(n-1)-exponential)/(2.0d0*boys_t)
         enddo
      endif
      if (.not.all(ieee_is_finite(values))) ierr = NUMERIC_ERROR
   end subroutine boys_values

   logical function same_component(a,b,c,i,j,k)
      implicit none
      integer, intent(in) :: a, b, c, i, j, k
      integer :: axis
      same_component = .true.
      do axis=1,3
         if (count([a,b,c] == axis) /= count([i,j,k] == axis)) then
            same_component = .false.
            return
         endif
      enddo
   end function same_component

   subroutine add_classical_hessian(npoints,points,result,ierr)
      use quick_molspec_module, only: quick_molspec, natom, xyz
      implicit none
      integer, intent(in) :: npoints
      double precision, intent(in) :: points(3,npoints)
      double precision, intent(inout) :: result(3,3,3,npoints)
      integer, intent(out) :: ierr
      double precision :: r(3), source(3), charge, distance2, inverse_r, inverse_r5, inverse_r7
      double precision :: component_value, linear
      integer :: point, isource, component, i, j, k, a, b, c

      ierr = 0
      do point=1,npoints
         do isource=1,natom+quick_molspec%nextatom
            if (isource <= natom) then
               source = xyz(:,isource)
               charge = quick_molspec%chg(isource)
            else
               source = quick_molspec%extxyz(:,isource-natom)
               charge = quick_molspec%extchg(isource-natom)
            endif
            if (charge == 0.0d0) cycle
            r = points(:,point)-source
            distance2 = sum(r*r)
            if (.not.ieee_is_finite(distance2) .or. distance2 <= 0.0d0) then
               ierr = NUMERIC_ERROR
               return
            endif
            inverse_r = 1.0d0/sqrt(distance2)
            inverse_r5 = inverse_r/distance2/distance2
            inverse_r7 = inverse_r5/distance2
            do component=1,10
               i = COMPONENT_I(component)
               j = COMPONENT_J(component)
               k = COMPONENT_K(component)
               linear = 0.0d0
               if (i == j) linear = linear+r(k)
               if (i == k) linear = linear+r(j)
               if (j == k) linear = linear+r(i)
               component_value = charge*(15.0d0*r(i)*r(j)*r(k)*inverse_r7-3.0d0*linear*inverse_r5)
               do a=1,3
                  do b=1,3
                     do c=1,3
                        if (same_component(a,b,c,i,j,k)) result(a,b,c,point) &
                           = result(a,b,c,point)+component_value
                     enddo
                  enddo
               enddo
            enddo
         enddo
      enddo
      if (.not.all(ieee_is_finite(result))) ierr = NUMERIC_ERROR
   end subroutine add_classical_hessian

end module quick_field_hessian_module
