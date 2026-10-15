#include "util.fh"
!---------------------------------------------------------------------!
! Independent, unscreened electronic electrostatic jet and cell bounds.!
! V_e=-sum(D*I), E=-grad(V_e), G=grad(E), H=grad(G). Coordinates are     !
! bohr; derivatives are with respect to the observation point, with    !
! geometry and the supplied spin-summed AO density held fixed.         !
! No classical, applied-field or continuum-reaction terms occur here.  !
! Ordinary FP64 evaluation of a real-arithmetic majorant is NOT a      !
! certified bound on all numerical error. Status states this explicitly.!
! (C) Copyright 2026 QUICK contributors. Mozilla Public License, v2.0. !
!---------------------------------------------------------------------!
module quick_electrostatic_jet_module
   use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
   implicit none
   private
   public :: electronic_jet_cpu, electronic_cell_majorants_cpu
   public :: electrostatic_jet_status, JET_COMPONENT_POWERS, JET_COMPONENT_RANK
   public :: jet_rank_norm

   integer, parameter :: MAX_ANGULAR=3, MAX_ORDER=10, NSLOTS=286, NCOMPONENTS=35
   integer, parameter :: DIMENSION_ERROR=46, NUMERIC_ERROR=55, BACKEND_ERROR=56
   double precision, parameter :: PI_JET=3.141592653589793238462643383279502884d0
   integer, parameter :: JET_COMPONENT_RANK(35) = [0,1,1,1,2,2,2,2,2,2, &
      3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4]
   integer, parameter :: JET_COMPONENT_POWERS(3,35) = reshape([ &
      0,0,0, 1,0,0, 0,1,0, 0,0,1, &
      2,0,0, 1,1,0, 1,0,1, 0,2,0, 0,1,1, 0,0,2, &
      3,0,0, 2,1,0, 2,0,1, 1,2,0, 1,1,1, 1,0,2, &
      0,3,0, 0,2,1, 0,1,2, 0,0,3, &
      4,0,0, 3,1,0, 3,0,1, 2,2,0, 2,1,1, 2,0,2, &
      1,3,0, 1,2,1, 1,1,2, 1,0,3, 0,4,0, 0,3,1, &
      0,2,2, 0,1,3, 0,0,4], [3,35])

   type :: electrostatic_jet_status
      ! Counters describe detected range events, not a rigorous omitted-tail
      ! budget. Recurrence/accumulation roundoff and other underflow may be
      ! unquantified; neither a zero count nor no screening certifies FP64.
      logical :: no_intentional_screen = .true.
      logical :: numerical_budget_known = .false.
      logical :: certified = .false.
      logical :: underflow_detected = .false.
      integer(kind=8) :: primitive_pairs = 0_8
      integer(kind=8) :: prefactor_underflow_count = 0_8
      integer(kind=8) :: boys_underflow_count = 0_8
   end type electrostatic_jet_status

contains

   subroutine electronic_jet_cpu(nb,na,maxprim,nc,ang,centers,expo,coeff,atoms,density, &
         npoints,points,owned,potential,efield,efg,field_hessian,status,ierr)
      !---------------------------------------------------------------------!
      ! Full supplied density, with native normalized Cartesian coefficients.!
      ! Higher AO index mu owns the pair mu>=nu; off-diagonal D is doubled. !
      ! Exact zero coefficients/density may be skipped; there is no cutoff. !
      ! All four outputs are transactional ADDITIVE, with identical density,!
      ! Hermite coefficients and Boys recurrence in one coherent evaluation.!
      ! Inputs D must already contain D_alpha+D_beta for unrestricted jobs. !
      ! This raw routine performs no MPI collectives and no GPU dispatch.   !
      !---------------------------------------------------------------------!
      implicit none
      integer, intent(in) :: nb, na, maxprim, npoints
      integer, intent(in) :: nc(nb), ang(3,nb), centers(nb), owned(nb)
      double precision, intent(in) :: expo(maxprim,nb), coeff(maxprim,nb), atoms(3,na)
      double precision, intent(in) :: density(nb,nb), points(3,npoints)
      double precision, intent(inout) :: potential(npoints), efield(3,npoints)
      double precision, intent(inout) :: efg(3,3,npoints), field_hessian(3,3,3,npoints)
      type(electrostatic_jet_status), intent(out) :: status
      integer, intent(out) :: ierr
      double precision, allocatable :: packed(:,:), vwork(:), ework(:,:), gwork(:,:,:), hwork(:,:,:,:)
      double precision :: unused(3,1)
      integer :: allocation_status, i, j, k, component, powers(3)

      status = electrostatic_jet_status()
      call validate_basis(nb,na,maxprim,nc,ang,centers,expo,coeff,atoms,density,owned,ierr)
      if (ierr /= 0) return
      ierr = DIMENSION_ERROR
      if (npoints < 1 .or. npoints > huge(npoints)/40) return
      ierr = NUMERIC_ERROR
      if (.not.all(ieee_is_finite(points))) return
      if (.not.all(ieee_is_finite(potential)) .or. .not.all(ieee_is_finite(efield))) return
      if (.not.all(ieee_is_finite(efg)) .or. .not.all(ieee_is_finite(field_hessian))) return
      allocate(packed(35,npoints),vwork(npoints),ework(3,npoints),gwork(3,3,npoints), &
         hwork(3,3,3,npoints),stat=allocation_status)
      if (allocation_status /= 0) then
         ierr = BACKEND_ERROR
         return
      endif
      unused = 0.0d0
      call contract_electronic(nb,na,maxprim,nc,ang,centers,expo,coeff,atoms,density, &
         npoints,points,unused,owned,.false.,packed,status,ierr)
      if (ierr /= 0) return
      vwork = potential+packed(1,:)
      ework = efield+packed(2:4,:)
      gwork = efg
      hwork = field_hessian
      do i=1,3
         do j=1,3
            powers = 0
            powers(i) = powers(i)+1
            powers(j) = powers(j)+1
            component = packed_component(powers,5,10)
            gwork(i,j,:) = gwork(i,j,:)+packed(component,:)
            do k=1,3
               powers(k) = powers(k)+1
               component = packed_component(powers,11,20)
               hwork(i,j,k,:) = hwork(i,j,k,:)+packed(component,:)
               powers(k) = powers(k)-1
            enddo
         enddo
      enddo
      ierr = NUMERIC_ERROR
      if (.not.all(ieee_is_finite(vwork)) .or. .not.all(ieee_is_finite(ework))) return
      if (.not.all(ieee_is_finite(gwork)) .or. .not.all(ieee_is_finite(hwork))) return
      potential = vwork
      efield = ework
      efg = gwork
      field_hessian = hwork
      ierr = 0
   end subroutine electronic_jet_cpu

   subroutine electronic_cell_majorants_cpu(nb,na,maxprim,nc,ang,centers,expo,coeff,atoms,density, &
         ncells,lower,upper,owned,component_bounds,status,ierr)
      !---------------------------------------------------------------------!
      ! Nonnegative component majorants for ranks zero through four over   !
      ! each ENTIRE closed axis-aligned support box, not just its center. !
      ! T_min=p*dist(P,box)^2 bounds Boys functions by monotonicity; q_max  !
      ! bounds |P-C|. Positive Hermite and Coulomb recurrences remove every !
      ! cancellation before contraction with absolute density prefactors. !
      ! Their real-arithmetic inequalities do not include FP64 rounding or !
      ! underflow errors. Output is transactional ADDITIVE and nonnegative.!
      !---------------------------------------------------------------------!
      implicit none
      integer, intent(in) :: nb, na, maxprim, ncells
      integer, intent(in) :: nc(nb), ang(3,nb), centers(nb), owned(nb)
      double precision, intent(in) :: expo(maxprim,nb), coeff(maxprim,nb), atoms(3,na)
      double precision, intent(in) :: density(nb,nb), lower(3,ncells), upper(3,ncells)
      double precision, intent(inout) :: component_bounds(35,ncells)
      type(electrostatic_jet_status), intent(out) :: status
      integer, intent(out) :: ierr
      double precision, allocatable :: work(:,:)
      integer :: allocation_status

      status = electrostatic_jet_status()
      call validate_basis(nb,na,maxprim,nc,ang,centers,expo,coeff,atoms,density,owned,ierr)
      if (ierr /= 0) return
      ierr = DIMENSION_ERROR
      if (ncells < 1 .or. ncells > huge(ncells)/35) return
      ierr = NUMERIC_ERROR
      if (.not.all(ieee_is_finite(lower)) .or. .not.all(ieee_is_finite(upper))) return
      if (any(lower > upper)) return
      if (.not.all(ieee_is_finite(component_bounds)) .or. any(component_bounds < 0.0d0)) return
      allocate(work(35,ncells),stat=allocation_status)
      if (allocation_status /= 0) then
         ierr = BACKEND_ERROR
         return
      endif
      call contract_electronic(nb,na,maxprim,nc,ang,centers,expo,coeff,atoms,density, &
         ncells,lower,upper,owned,.true.,work,status,ierr)
      if (ierr /= 0) return
      work = work+component_bounds
      if (.not.all(ieee_is_finite(work))) then
         ierr = NUMERIC_ERROR
         return
      endif
      component_bounds = work
   end subroutine electronic_cell_majorants_cpu

   subroutine validate_basis(nb,na,maxprim,nc,ang,centers,expo,coeff,atoms,density,owned,ierr)
      implicit none
      integer, intent(in) :: nb, na, maxprim
      integer, intent(in) :: nc(nb), ang(3,nb), centers(nb), owned(nb)
      double precision, intent(in) :: expo(maxprim,nb), coeff(maxprim,nb), atoms(3,na), density(nb,nb)
      integer, intent(out) :: ierr
      integer :: mu

      ierr = DIMENSION_ERROR
      if (min(nb,na,maxprim) < 1) return
      if (nb > huge(nb)/nb .or. maxprim > huge(maxprim)/nb) return
      ierr = NUMERIC_ERROR
      if (.not.all(ieee_is_finite(atoms)) .or. .not.all(ieee_is_finite(density))) return
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
   end subroutine validate_basis

   subroutine contract_electronic(nb,na,maxprim,nc,ang,centers,expo,coeff,atoms,density, &
         nlocations,locations,upper,owned,majorant,values,status,ierr)
      implicit none
      integer, intent(in) :: nb, na, maxprim, nlocations
      integer, intent(in) :: nc(nb), ang(3,nb), centers(nb), owned(nb)
      double precision, intent(in) :: expo(maxprim,nb), coeff(maxprim,nb), atoms(3,na)
      double precision, intent(in) :: density(nb,nb), locations(3,nlocations), upper(3,*)
      logical, intent(in) :: majorant
      double precision, intent(out) :: values(35,nlocations)
      type(electrostatic_jet_status), intent(inout) :: status
      integer, intent(out) :: ierr
      double precision :: hcoeff(0:6,3), term_coeff(343), rzero(NSLOTS)
      double precision :: alpha, beta, p, inverse_p, reduced, distance2, gaussian, exponent_argument
      double precision :: prefactor, weight, factor, term, contribution, inverse_two_p
      double precision :: displacement(3), pa(3), pb(3), product_center(3), q(3), qlo(3), qhi(3), qmin(3), boys_t
      integer :: slot(0:MAX_ORDER,0:MAX_ORDER,0:MAX_ORDER), powers(3,NSLOTS), degree(NSLOTS)
      integer :: previous1(NSLOTS), previous2(NSLOTS), direction(NSLOTS), term_slot(343,35)
      integer :: mu, nu, icon, jcon, axis, location, component, ncomponents, order, maxima(3)
      integer :: t, u, v, count_terms, n, rank

      ierr = 0
      values = 0.0d0
      ncomponents = 20
      if (majorant) ncomponents = 35
      call prepare_slots(slot,powers,degree,previous1,previous2,direction)
      do mu=1,nb
         if (owned(mu) == 0) cycle
         do nu=1,mu
            weight = density(nu,mu)
            if (mu /= nu) weight = 2.0d0*weight
            if (weight == 0.0d0) cycle
            displacement = atoms(:,centers(nu))-atoms(:,centers(mu))
            distance2 = sum(displacement*displacement)
            if (.not.ieee_is_finite(weight) .or. .not.ieee_is_finite(distance2)) then
               ierr = NUMERIC_ERROR
               return
            endif
            maxima = ang(:,mu)+ang(:,nu)
            order = sum(maxima)+3
            if (majorant) order = order+1
            do icon=1,nc(mu)
               if (coeff(icon,mu) == 0.0d0) cycle
               alpha = expo(icon,mu)
               do jcon=1,nc(nu)
                  if (coeff(jcon,nu) == 0.0d0) cycle
                  status%primitive_pairs = status%primitive_pairs+1_8
                  beta = expo(jcon,nu)
                  p = alpha+beta
                  if (.not.ieee_is_finite(p) .or. p <= 0.0d0) then
                     ierr = NUMERIC_ERROR
                     return
                  endif
                  inverse_p = 1.0d0/p
                  reduced = alpha*(beta*inverse_p)
                  exponent_argument = reduced*distance2
                  if (.not.ieee_is_finite(inverse_p) .or. .not.ieee_is_finite(exponent_argument)) then
                     ierr = NUMERIC_ERROR
                     return
                  endif
                  gaussian = exp(-exponent_argument)
                  prefactor = gaussian*2.0d0*PI_JET*inverse_p*coeff(jcon,nu)*coeff(icon,mu)*weight
                  if (.not.ieee_is_finite(prefactor)) then
                     ierr = NUMERIC_ERROR
                     return
                  endif
                  if (gaussian < tiny(gaussian) .or. abs(prefactor) < tiny(prefactor)) then
                     status%underflow_detected = .true.
                     status%prefactor_underflow_count = status%prefactor_underflow_count+1_8
                  endif
                  if (prefactor == 0.0d0) cycle
                  pa = beta*inverse_p*displacement
                  pb = -alpha*inverse_p*displacement
                  product_center = atoms(:,centers(mu))+pa
                  inverse_two_p = 0.5d0*inverse_p
                  do axis=1,3
                     call hermite_coefficients(ang(axis,mu),ang(axis,nu),pa(axis),pb(axis), &
                        inverse_two_p,majorant,hcoeff(:,axis))
                  enddo
                  if (.not.all(ieee_is_finite(hcoeff)) .or. &
                      .not.all(ieee_is_finite(product_center))) then
                     ierr = NUMERIC_ERROR
                     return
                  endif
                  count_terms = 0
                  do t=0,maxima(1)
                     do u=0,maxima(2)
                        do v=0,maxima(3)
                           count_terms = count_terms+1
                           term_coeff(count_terms) = hcoeff(t,1)*hcoeff(u,2)*hcoeff(v,3)
                           do component=1,ncomponents
                              term_slot(count_terms,component) = slot(t+JET_COMPONENT_POWERS(1,component), &
                                 u+JET_COMPONENT_POWERS(2,component),v+JET_COMPONENT_POWERS(3,component))
                           enddo
                        enddo
                     enddo
                  enddo
                  if (.not.all(ieee_is_finite(term_coeff(1:count_terms)))) then
                     ierr = NUMERIC_ERROR
                     return
                  endif
                  if (majorant) prefactor = abs(prefactor)
                  do location=1,nlocations
                     if (majorant) then
                        qlo = product_center-upper(:,location)
                        qhi = product_center-locations(:,location)
                        q = max(abs(qlo),abs(qhi))
                        qmin = min(abs(qlo),abs(qhi))
                        where (qlo <= 0.0d0 .and. qhi >= 0.0d0) qmin = 0.0d0
                        boys_t = p*sum(qmin*qmin)
                     else
                        q = product_center-locations(:,location)
                        boys_t = p*sum(q*q)
                     endif
                     if (.not.all(ieee_is_finite(q)) .or. .not.ieee_is_finite(boys_t) .or. boys_t < 0.0d0) then
                        ierr = NUMERIC_ERROR
                        return
                     endif
                     call coulomb_layer(order,p,boys_t,q,majorant,powers,degree,previous1,previous2, &
                        direction,rzero,status,ierr)
                     if (ierr /= 0) return
                     do component=1,ncomponents
                        contribution = 0.0d0
                        do n=1,count_terms
                           term = term_coeff(n)*rzero(term_slot(n,component))
                           contribution = contribution+term
                        enddo
                        factor = prefactor
                        if (.not.majorant) then
                           rank = JET_COMPONENT_RANK(component)
                           if (rank == 0 .or. mod(rank,2) == 1) factor = -factor
                        endif
                        values(component,location) = values(component,location)+factor*contribution
                     enddo
                  enddo
               enddo
            enddo
         enddo
      enddo
      if (.not.all(ieee_is_finite(values))) ierr = NUMERIC_ERROR
   end subroutine contract_electronic

   subroutine prepare_slots(slot,powers,degree,previous1,previous2,direction)
      implicit none
      integer, intent(out) :: slot(0:MAX_ORDER,0:MAX_ORDER,0:MAX_ORDER)
      integer, intent(out) :: powers(3,NSLOTS), degree(NSLOTS), previous1(NSLOTS), previous2(NSLOTS)
      integer, intent(out) :: direction(NSLOTS)
      integer :: d, t, u, v, n, lower(3), axis

      slot = 0
      n = 0
      do d=0,MAX_ORDER
         do t=d,0,-1
            do u=d-t,0,-1
               v = d-t-u
               n = n+1
               slot(t,u,v) = n
               powers(:,n) = [t,u,v]
               degree(n) = d
            enddo
         enddo
      enddo
      previous1 = 0
      previous2 = 0
      direction = 0
      do n=2,NSLOTS
         do axis=1,3
            if (powers(axis,n) > 0) exit
         enddo
         direction(n) = axis
         lower = powers(:,n)
         lower(axis) = lower(axis)-1
         previous1(n) = slot(lower(1),lower(2),lower(3))
         if (powers(axis,n) > 1) then
            lower(axis) = lower(axis)-1
            previous2(n) = slot(lower(1),lower(2),lower(3))
         endif
      enddo
   end subroutine prepare_slots

   subroutine hermite_coefficients(la,lb,pa,pb,inverse_two_p,positive,coefficients)
      implicit none
      integer, intent(in) :: la, lb
      double precision, intent(in) :: pa, pb, inverse_two_p
      logical, intent(in) :: positive
      double precision, intent(out) :: coefficients(0:6)
      double precision :: work(0:6), displacement
      integer :: raising, d, t

      coefficients = 0.0d0
      coefficients(0) = 1.0d0
      do raising=1,la+lb
         displacement = pa
         if (raising > la) displacement = pb
         if (positive) displacement = abs(displacement)
         d = raising-1
         work = 0.0d0
         do t=0,d+1
            if (t > 0) work(t) = inverse_two_p*coefficients(t-1)
            if (t <= d) work(t) = work(t)+displacement*coefficients(t)
            if (t < d) work(t) = work(t)+dble(t+1)*coefficients(t+1)
         enddo
         coefficients = work
      enddo
   end subroutine hermite_coefficients

   subroutine coulomb_layer(order,p,boys_t,q,positive,powers,degree,previous1,previous2, &
         direction,rzero,status,ierr)
      ! Two degree-10 tetrahedral Cartesian layers: 2*286 doubles, F_10.
      ! Signed seeds (-2p)^m differentiate in P-C; positive seeds (2p)^m
      ! and q_max bound the same recurrence over a continuous support box.
      implicit none
      integer, intent(in) :: order, powers(3,NSLOTS), degree(NSLOTS), previous1(NSLOTS), previous2(NSLOTS)
      integer, intent(in) :: direction(NSLOTS)
      double precision, intent(in) :: p, boys_t, q(3)
      logical, intent(in) :: positive
      double precision, intent(out) :: rzero(NSLOTS)
      type(electrostatic_jet_status), intent(inout) :: status
      integer, intent(out) :: ierr
      double precision :: boys(0:MAX_ORDER), seed(0:MAX_ORDER), previous(NSLOTS), current(NSLOTS), factor
      integer :: m, n, axis

      call boys_values(order,boys_t,boys,status,ierr)
      if (ierr /= 0) return
      factor = 1.0d0
      do m=0,order
         seed(m) = factor*boys(m)
         if (m < order) then
            if (positive) then
               factor = factor*(2.0d0*p)
            else
               factor = factor*(-2.0d0*p)
            endif
         endif
      enddo
      if (.not.all(ieee_is_finite(seed(0:order)))) then
         ierr = NUMERIC_ERROR
         return
      endif
      previous = 0.0d0
      do m=order,0,-1
         current = 0.0d0
         current(1) = seed(m)
         do n=2,NSLOTS
            if (degree(n) > order-m) exit
            axis = direction(n)
            current(n) = q(axis)*previous(previous1(n))
            if (previous2(n) > 0) current(n) = current(n) &
               +dble(powers(axis,n)-1)*previous(previous2(n))
         enddo
         if (.not.all(ieee_is_finite(current))) then
            ierr = NUMERIC_ERROR
            return
         endif
         previous = current
      enddo
      rzero = previous
   end subroutine coulomb_layer

   subroutine boys_values(order,boys_t,values,status,ierr)
      implicit none
      integer, intent(in) :: order
      double precision, intent(in) :: boys_t
      double precision, intent(out) :: values(0:MAX_ORDER)
      type(electrostatic_jet_status), intent(inout) :: status
      integer, intent(out) :: ierr
      double precision :: term, series, exponential
      integer :: k, m
      logical :: converged

      ierr = 0
      values = 0.0d0
      exponential = exp(-boys_t)
      if (boys_t <= 32.0d0) then
         ! Positive-term highest-order seed, then stable downward recurrence.
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
         do m=order-1,0,-1
            values(m) = (2.0d0*boys_t*values(m+1)+exponential)/dble(2*m+1)
         enddo
      else
         values(0) = 0.5d0*sqrt(PI_JET/boys_t)*erf(sqrt(boys_t))
         do m=1,order
            values(m) = (dble(2*m-1)*values(m-1)-exponential)/(2.0d0*boys_t)
         enddo
      endif
      if (.not.all(ieee_is_finite(values))) then
         ierr = NUMERIC_ERROR
         return
      endif
      do m=0,order
         if (values(m) < tiny(values(m))) then
            status%underflow_detected = .true.
            status%boys_underflow_count = status%boys_underflow_count+1_8
         endif
      enddo
   end subroutine boys_values

   integer function packed_component(powers,first,last)
      implicit none
      integer, intent(in) :: powers(3), first, last
      integer :: component
      packed_component = 0
      do component=first,last
         if (all(JET_COMPONENT_POWERS(:,component) == powers)) then
            packed_component = component
            return
         endif
      enddo
   end function packed_component

   double precision function jet_rank_norm(components,rank)
      ! Frobenius norm of the symmetric rank-r tensor, including all
      ! permutation multiplicities. Stable scaling avoids squaring overflow.
      implicit none
      double precision, intent(in) :: components(35)
      integer, intent(in) :: rank
      integer, parameter :: FACTORIAL(0:4)=[1,1,2,6,24]
      integer :: component, multiplicity, axis
      double precision :: scale, total, value

      jet_rank_norm = -1.0d0
      if (rank < 0 .or. rank > 4) return
      if (.not.all(ieee_is_finite(components))) return
      scale = 0.0d0
      total = 0.0d0
      do component=1,35
         if (JET_COMPONENT_RANK(component) /= rank) cycle
         multiplicity = FACTORIAL(rank)
         do axis=1,3
            multiplicity = multiplicity/FACTORIAL(JET_COMPONENT_POWERS(axis,component))
         enddo
         value = abs(components(component))*sqrt(dble(multiplicity))
         if (value > scale) then
            total = 1.0d0+total*(scale/value)**2
            scale = value
         else if (value > 0.0d0) then
            total = total+(value/scale)**2
         endif
      enddo
      jet_rank_norm = scale*sqrt(total)
   end function jet_rank_norm

end module quick_electrostatic_jet_module
