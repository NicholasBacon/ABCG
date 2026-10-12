#ifndef SPMV_COMM_MPIL_HPP
#define SPMV_COMM_MPIL_HPP

#include "spmv_comm.hpp"
#include "mpil_compat.hpp"

class SpMVCommMPIL : public SpMVComm
{
    public:
        SpMVCommMPIL() : SpMVComm()
        {
            spmv_req = NULL;
        }

        ~SpMVCommMPIL()
        {
            MPIL_Topo_free(&mpil_topo);
        }

        void init(ParMat& A_in, double* sbuf, double* rbuf)
        {
            A = &A_in;
            sendbuf = sbuf;
            recvbuf = rbuf;
            n = A->recv_comm.n_msgs + A->send_comm.n_msgs;

            MPIL_Info* mpil_info;
            MPIL_Info_init(&mpil_info);

            MPIL_Topo_init(A->recv_comm.n_msgs,
                           A->recv_comm.procs.data(),
                           MPI_UNWEIGHTED,
                           A->send_comm.n_msgs,
                           A->send_comm.procs.data(),
                           MPI_UNWEIGHTED,
                           mpil_info,
                           &mpil_topo);

            MPIL_Info_free(&mpil_info);

        }

        void start()
        {
            int proc, start, end;
            int tag = 0;

            
            MPIL_Neighbor_alltoallv_topo(sendbuf,
                                         A->send_comm.counts.data(),
                                         A->send_comm.ptr.data(),
                                         MPI_DOUBLE,
                                         recvbuf,
                                         A->recv_comm.counts.data(),
                                         A->recv_comm.ptr.data(),
                                         MPI_DOUBLE,
                                         mpil_topo,
                                         mpil_world());

        }

        void wait()
        {
            
        }

    private:
        MPIL_Request* spmv_req;
        MPIL_Topo* mpil_topo;

};

#endif
