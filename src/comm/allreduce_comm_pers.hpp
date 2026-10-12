#ifndef ALLREDUCE_COMM_PERS_HPP
#define ALLREDUCE_COMM_PERS_HPP

#include "allreduce_comm.hpp"

// MPI_Allreduce_init is MPI 4.0; Open MPI 4.x (MPI 3.1) offers the same call as
// MPIX_Allreduce_init in its pcollreq extension.
#if MPI_VERSION < 4 && defined(OPEN_MPI)
#include <mpi-ext.h>
#define MPI_Allreduce_init MPIX_Allreduce_init
#endif

class AllreduceCommPers : public AllreduceComm
{
    public:
        AllreduceCommPers() : AllreduceComm()
        {
            req = MPI_REQUEST_NULL;
        }

        ~AllreduceCommPers()
        {
            MPI_Request_free(&req);
        }
        
        void init(double* loc, double* glob)
        {
            local_sum = loc;
            global_sum = glob;

            MPI_Allreduce_init(local_sum, global_sum, 1, MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD, MPI_INFO_NULL, &req);
        }

        void start()
        {
             MPI_Start(&req);
        }

        void wait()
        {
            MPI_Wait(&req, MPI_STATUS_IGNORE);
        }
      

    private:
        MPI_Request req;
};

#endif

