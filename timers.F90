! CWA
! This module can be compiled to produce only timings, or 
! timings and hardware counter info. To enable h/w counter
! info, user -DHAVE_PAPI when compiling.
! 
! If h/w counting the counters have to be initialised at 
! the start of the program with a call to init_counters.
! In this case, 'use timers' in the main program and 
! call init_counters after mpi_init.
!
! To time a region of code, 'use timers' in the subroutine
! the region of interest lives in and define a region with
! start_timer and end_timer, passing an integer between 1
! and NUMREGIONS (defined in the preprocessing directive 
! below). Not every process has to enter the region.
!
! At the end of the program (before mpi_finalize) call 
! print_timers to print the timings and h/w stats (if used).
!
! The output shows the max, min and avg stats for each
! region over the processes that actually entered that region.
!
! PAPI info: http://icl.cs.utk.edu/

      module timers

#define NUMREGIONS 25

! Define the message to be printed for each hardware counter
#define PRINT1 " FLOPS = "
#define PRINT1max " max FLOPS = "
#define PRINT1min " min FLOPS = "
#define PRINT1avg " avg FLOPS = "
#define PRINT2 " L1DCM = "
#define PRINT2max " max L1DCM = "
#define PRINT2min " min L1DCM = "
#define PRINT2avg " avg L1DCM = "
#define PRINT3 " L2DCM = "
#define PRINT3max " max L2DCM = "
#define PRINT3min " min L2DCM = "
#define PRINT3avg " avg L2DCM = "
#define PRINT4 " TLBM = "
#define PRINT4max " max TLBM = "
#define PRINT4min " min TLBM = "
#define PRINT4avg " avg TLBM = "

#define PRTU 6

      use mpi
      implicit none
#ifdef HAVE_PAPI
      include "f90papi.h"
#endif

      private

      public init_counters,start_timer,end_timer,print_timers

      integer, dimension(NUMREGIONS) :: counts = 0
      double precision, dimension(NUMREGIONS) :: times = 0.d0
      double precision, dimension(NUMREGIONS) :: avg_times = 0.d0
      double precision, dimension(NUMREGIONS) :: max_times = 0.d0
      double precision, dimension(NUMREGIONS) :: min_times = 0.d0
      double precision, dimension(NUMREGIONS) :: cuml_times = 0.d0
      integer, dimension(NUMREGIONS) :: init = 0
#ifdef HAVE_PAPI
      integer :: eventSet = PAPI_NULL
      integer, parameter :: wpi = selected_int_kind(2*range(1))
      integer(kind=wpi), dimension(NUMREGIONS) :: hw1,hw2,hw3,hw4 = 0
#endif 


      contains


      subroutine init_counters()

#ifdef HAVE_PAPI
      integer :: eventCode
      integer :: check = PAPI_VER_CURRENT
#endif

      integer :: ierr
      logical :: isinit
      call mpi_initialized(isinit,ierr)
      if(.not.isinit) call mpi_init(ierr)

#ifdef HAVE_PAPI
      call PAPIF_library_init(check)
       if(check.ne.PAPI_VER_CURRENT)then
         write(0,*) "variables: ",PAPI_EINVAL,PAPI_ENOMEM,PAPI_ESBSTR,PAPI_ESYS
         write(0,*) "check: ",check
         stop "library_init failed"
       end if
       call PAPIF_create_eventset(eventSet,check)
       if(check.ne.PAPI_OK) stop "create_eventset failed"

! Register which hardware counters to use (pre-defined PAPI events)
       call PAPIF_add_event(eventSet,PAPI_FP_OPS,check)
       if(check.ne.0) write(0,*) "warning: add_event 1 failed"
       call PAPIF_add_event(eventSet,PAPI_L1_DCM,check)
       if(check.ne.0) write(0,*) "warning: add_event 2 failed"
       call PAPIF_add_event(eventSet,PAPI_L2_DCM,check)
       if(check.ne.0) write(0,*) "warning: add_event 3 failed"
       call PAPIF_add_event(eventSet,PAPI_TLB_DM,check)
       if(check.ne.0) write(0,*) "warning: add_event 4 failed"

!
! This is how to use native events (check listings with papi_native_avail utility)
!
!      CALL PAPIF_event_name_to_code("DISPATCHED_FPU_OPS:OPS_PIPE0", eventCode,check)
!      if(check.ne.0) stop "event_name_to_code failed 1"
!      call PAPIF_add_event(eventSet,eventCode,check)
!      CALL PAPIF_event_name_to_code("DISPATCHED_FPU_OPS:OPS_PIPE1", eventCode,check)
!      if(check.ne.0) stop "event_name_to_code failed 2"
!      call PAPIF_add_event(eventSet,eventCode,check)
!      CALL PAPIF_event_name_to_code("DISPATCHED_FPU_OPS:OPS_PIPE2", eventCode,check)
!      if(check.ne.0) stop "event_name_to_code failed 3"
!      call PAPIF_add_event(eventSet,eventCode,check)
!      CALL PAPIF_event_name_to_code("DISPATCHED_FPU_OPS:OPS_PIPE3", eventCode,check)
!      if(check.ne.0) stop "event_name_to_code failed 4"
!      call PAPIF_add_event(eventSet,eventCode,check)
!      CALL PAPIF_event_name_to_code("L3_CACHE_MISSES:ALL", eventCode,check)
!      if(check.ne.0) stop "event_name_to_code failed, L3_CACHE_MISSES"
!      call PAPIF_add_event(eventSet,eventCode,check)
!
      call PAPIF_start(eventSet,check)
#endif 

      return
      end subroutine init_counters


      subroutine start_timer(event)
      integer :: event
#ifdef HAVE_PAPI
      integer :: check
#endif
      if(init(event).eq.0)then
       init(event)=1
      end if
#ifdef HAVE_PAPI
      call PAPIF_reset(eventSet,check)
      if(check.ne.0) stop "start failed"
#endif
      times(event) = mpi_wtime()
      end subroutine start_timer


      subroutine end_timer(event)
      integer :: event
#ifdef HAVE_PAPI
      integer(kind=wpi) counter(4)
      integer :: check
#endif
      times(event) = mpi_wtime() - times(event)
      counts(event) = counts(event) + 1
      if(times(event).gt.max_times(event)) max_times(event) = times(event)
      if(counts(event).eq.1 .or. times(event).lt.min_times(event)) min_times(event) = times(event)
      cuml_times(event) = cuml_times(event) + times(event)
      avg_times(event) = cuml_times(event)/counts(event)
#ifdef HAVE_PAPI
      call PAPIF_read(eventSet,counter,check)
      if(check.ne.0) stop "read failed"
      hw1(event) = hw1(event) + counter(1)
      hw2(event) = hw2(event) + counter(2)
      hw3(event) = hw3(event) + counter(3)
      hw4(event) = hw4(event) + counter(4)
#endif
      end subroutine end_timer


      subroutine print_timers()
      integer :: i,j,rank,psize,ierr
      integer, dimension(mpi_status_size) :: istat
      double precision :: maxt,mint,rtime,totalt
      integer, dimension(:,:), allocatable :: allinits
      integer :: maxproct,minproct,numcallers,maxcountt,mincountt,rcounts
#ifdef HAVE_PAPI
      integer(kind=wpi) :: rhw1,rhw2,rhw3,rhw4
      integer(kind=wpi) :: maxhw1,maxhw2,maxhw3,maxhw4
      integer(kind=wpi) :: minhw1,minhw2,minhw3,minhw4
      integer(kind=wpi) :: totalhw1,totalhw2,totalhw3,totalhw4
      integer(kind=wpi) :: maxprochw1,maxprochw2,maxprochw3,maxprochw4
      integer(kind=wpi) :: minprochw1,minprochw2,minprochw3,minprochw4
      integer(kind=wpi) :: maxcounthw1,maxcounthw2,maxcounthw3,maxcounthw4
      integer(kind=wpi) :: mincounthw1,mincounthw2,mincounthw3,mincounthw4
#endif
      integer :: int8type

      call mpi_comm_rank(mpi_comm_world,rank,ierr)
      call mpi_comm_size(mpi_comm_world,psize,ierr)
!      call mpi_type_create_f90_integer(range(1_8),int8type,ierr)
      call mpi_type_match_size(mpi_typeclass_integer,8,int8type,ierr)

      allocate(allinits(NUMREGIONS,psize))

      if(rank.eq.0) write(PRTU,*) "--------------"
      
      ! Gather the init flags for every region on every proc
      call mpi_gather(init,NUMREGIONS,MPI_INTEGER,allinits,NUMREGIONS,MPI_INTEGER,0,mpi_comm_world,ierr)

      do j=1,NUMREGIONS

       if(rank.ne.0)then
        ! Non root procs: send timing for current region if used
        if(init(j).eq.1)then
         call mpi_ssend(cuml_times(j),1,MPI_DOUBLE_PRECISION,0,rank*NUMREGIONS+j,mpi_comm_world,ierr)
         call mpi_ssend(counts(j),1,MPI_INTEGER,0,rank*NUMREGIONS+j,mpi_comm_world,ierr)
#ifdef HAVE_PAPI
         call mpi_ssend(hw1(j),1,int8type,0,rank*NUMREGIONS+j,mpi_comm_world,ierr)
         call mpi_ssend(hw2(j),1,int8type,0,rank*NUMREGIONS+j,mpi_comm_world,ierr)
         call mpi_ssend(hw3(j),1,int8type,0,rank*NUMREGIONS+j,mpi_comm_world,ierr)
         call mpi_ssend(hw4(j),1,int8type,0,rank*NUMREGIONS+j,mpi_comm_world,ierr)
#endif
        end if
       end if

       if(rank.eq.0)then
        ! number of processes that called this region
        numcallers = 0
        ! number of times region entered
        maxcountt = 0
        mincountt = 0
        !time
        maxt = 0.d0
        mint = 0.d0
        totalt = 0.d0
        maxproct = -1
        minproct = -1
#ifdef HAVE_PAPI
        !hw1
        maxhw1 = 0
        minhw1 = 0
        totalhw1 = 0
        maxprochw1 = -1
        !hw2
        maxhw2 = 0
        minhw2 = 0
        totalhw2 = 0
        maxprochw2 = -1
        !hw3
        maxhw3 = 0
        minhw3 = 0
        totalhw3 = 0
        maxprochw3 = -1
        !hw4
        maxhw4 = 0
        minhw4 = 0
        totalhw4 = 0
        maxprochw4 = -1
#endif

        ! If the current region has been used on root, set min and max
        if(init(j).eq.1)then
         write(PRTU,'(A,I3)') "REGION ",j
         maxt=cuml_times(j)
         maxproct=0
         mint=cuml_times(j)
         totalt=cuml_times(j)
         minproct=0
         numcallers = 1
         maxcountt = counts(j)
         mincountt = counts(j)
#ifdef HAVE_PAPI
         maxhw1 = hw1(j)
         maxprochw1 = 0
         maxcounthw1 = counts(j)
         minhw1 = hw1(j)
         minprochw1 = 0
         mincounthw1 = counts(j)
         totalhw1 = hw1(j)

         maxhw2 = hw2(j)
         maxprochw2 = 0
         maxcounthw2 = counts(j)
         minhw2 = hw2(j)
         minprochw2 = 0
         mincounthw2 = counts(j)
         totalhw2 = hw2(j)

         maxhw3 = hw3(j)
         maxprochw3 = 0
         maxcounthw3 = counts(j)
         minhw3 = hw3(j)
         minprochw3 = 0
         mincounthw3 = counts(j)
         totalhw3 = hw3(j)

         maxhw4 = hw4(j)
         maxprochw4 = 0
         maxcounthw4 = counts(j)
         minhw4 = hw4(j)
         minprochw4 = 0
         mincounthw4 = counts(j)
         totalhw4 = hw4(j)

#endif
        end if
!
        do i=2,psize
         ! Root proc: receive timing for current region form each proc on which it was used
         if(allinits(j,i).eq.1)then
          numcallers = numcallers + 1
          call mpi_recv(rtime,1,MPI_DOUBLE_PRECISION,i-1,(i-1)*NUMREGIONS+j,mpi_comm_world,istat,ierr)
          call mpi_recv(rcounts,1,MPI_INTEGER,i-1,(i-1)*NUMREGIONS+j,mpi_comm_world,istat,ierr)
!
#ifdef HAVE_PAPI
          call mpi_recv(rhw1,1,int8type,i-1,(i-1)*NUMREGIONS+j,mpi_comm_world,istat,ierr)
          call mpi_recv(rhw2,1,int8type,i-1,(i-1)*NUMREGIONS+j,mpi_comm_world,istat,ierr)
          call mpi_recv(rhw3,1,int8type,i-1,(i-1)*NUMREGIONS+j,mpi_comm_world,istat,ierr)
          call mpi_recv(rhw4,1,int8type,i-1,(i-1)*NUMREGIONS+j,mpi_comm_world,istat,ierr)
#endif
!
          ! Set the max and min times if they are unset or if values received are new max/min
          if(maxproct.eq.-1 .or. rtime.gt.maxt)then
           maxt = rtime
           maxproct = i-1
           maxcountt = rcounts
          end if
          if(minproct.eq.-1 .or. rtime.lt.mint)then
           mint = rtime
           minproct = i-1
           mincountt = rcounts
          end if
          totalt = totalt + rtime

#ifdef HAVE_PAPI
          if(maxprochw1.eq.-1 .or. rhw1.gt.maxhw1)then
           maxhw1 = rhw1
           maxprochw1 = i-1
           maxcounthw1 = rcounts
          end if
          if(minprochw1.eq.-1 .or. rhw1.lt.minhw1)then
           minhw1 = rhw1
           minprochw1 = i-1
           mincounthw1 = rcounts
          end if
          totalhw1 = totalhw1 + rhw1
 
          if(maxprochw2.eq.-1 .or. rhw2.gt.maxhw2)then
           maxhw2 = rhw2
           maxprochw2 = i-1
           maxcounthw2 = rcounts
          end if
          if(minprochw2.eq.-1 .or. rhw2.lt.minhw2)then
           minhw2 = rhw2
           minprochw2 = i-1
           mincounthw2 = rcounts
          end if
          totalhw2 = totalhw2 + rhw2
 
          if(maxprochw3.eq.-1 .or. rhw3.gt.maxhw3)then
           maxhw3 = rhw3
           maxprochw3 = i-1
           maxcounthw3 = rcounts
          end if
          if(minprochw3.eq.-1 .or. rhw3.lt.minhw3)then
           minhw3 = rhw3
           minprochw3 = i-1
           mincounthw3 = rcounts
          end if
          totalhw3 = totalhw3 + rhw3
 
          if(maxprochw4.eq.-1 .or. rhw4.gt.maxhw4)then
           maxhw4 = rhw4
           maxprochw4 = i-1
           maxcounthw4 = rcounts
          end if
          if(minprochw4.eq.-1 .or. rhw4.lt.minhw4)then
           minhw4 = rhw4
           minprochw4 = i-1
           mincounthw4 = rcounts
          end if
          totalhw4 = totalhw4 + rhw4
#endif

         end if

        end do
        
        if(maxproct.ne.-1)then
         write(PRTU,'(A,I3)') " processes count = ",numcallers
         write(PRTU,'(A,F14.8,A,I3,I8,A)') " max time        = ",maxt," (proc ",maxproct,maxcountt," entries)"
         write(PRTU,'(A,F14.8,A,I3,I8,A)') " min time        = ",mint," (proc ",minproct,mincountt," entries)"
         write(PRTU,'(A,F14.8)') " avg time        = ",totalt/numcallers
#ifdef HAVE_PAPI
         write(PRTU,'(A,I16,A,I3,I8,A)') PRINT1max,maxhw1," (proc",maxprochw1,maxcounthw1," entries)"
         write(PRTU,'(A,I16,A,I3,I8,A)') PRINT1min,minhw1," (proc",minprochw1,mincounthw1," entries)"
         write(PRTU,'(A,F18.0)') PRINT1avg,dble(totalhw1)/dble(numcallers)
         write(PRTU,'(A,I16,A,I3,I8,A)') PRINT2max,maxhw2," (proc",maxprochw2,maxcounthw2," entries)"
         write(PRTU,'(A,I16,A,I3,I8,A)') PRINT2min,minhw2," (proc",minprochw2,mincounthw2," entries)"
         write(PRTU,'(A,F18.0)') PRINT2avg,dble(totalhw2)/dble(numcallers)
         write(PRTU,'(A,I16,A,I3,I8,A)') PRINT3max,maxhw3," (proc",maxprochw3,maxcounthw3," entries)"
         write(PRTU,'(A,I16,A,I3,I8,A)') PRINT3min,minhw3," (proc",minprochw3,mincounthw3," entries)"
         write(PRTU,'(A,F18.0)') PRINT3avg,dble(totalhw3)/dble(numcallers)
         write(PRTU,'(A,I16,A,I3,I8,A)') PRINT4max,maxhw4," (proc",maxprochw4,maxcounthw4," entries)"
         write(PRTU,'(A,I16,A,I3,I8,A)') PRINT4min,minhw4," (proc",minprochw4,mincounthw4," entries)"
         write(PRTU,'(A,F18.0)') PRINT4avg,dble(totalhw4)/dble(numcallers)
#endif
         write(PRTU,*) "--------------"
        end if
       end if

      end do

      end subroutine print_timers

      end module timers
