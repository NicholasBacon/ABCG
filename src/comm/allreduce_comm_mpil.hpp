#ifndef ALLREDUCE_COMM_MPIL_HPP
#define ALLREDUCE_COMM_MPIL_HPP

#include "allreduce_comm.hpp"
#include "mpil_compat.hpp"

class AllreduceCommMPIL : public AllreduceComm
{
    public:
        AllreduceCommMPIL() : AllreduceComm()
        {
        }

        ~AllreduceCommMPIL()
        {
        }
        
        void init(double* loc, double* glob)
        {
            local_sum = loc;
            global_sum = glob;
        }

        void start()
        {
#ifdef MPIL_OLD_API
            // No MPI Advance allreduce in the old API; ALLREDUCE_PMPI is PMPI_Allreduce.
            MPI_Allreduce(local_sum, global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#else
            MPIL_Allreduce(local_sum, global_sum, 1, MPI_DOUBLE, MPI_SUM, mpil_world());
#endif
        }

        void wait()
        {
        }
      
};

#endif

