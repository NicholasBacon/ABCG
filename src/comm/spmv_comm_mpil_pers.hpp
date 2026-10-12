#ifndef SPMV_COMM_MPIL_PERS_HPP
#define SPMV_COMM_MPIL_PERS_HPP

#include "spmv_comm.hpp"
#include "mpil_compat.hpp"

class SpMVCommMPILPers : public SpMVComm
{
    public:
        SpMVCommMPILPers() : SpMVComm()
        {
            spmv_req = NULL;
        }

        ~SpMVCommMPILPers()
        {
            MPIL_Request_free(&spmv_req);
        }

        void init(ParMat& A_in, double* sbuf, double* rbuf)
        {
            A = &A_in;
            sendbuf = sbuf;
            recvbuf = rbuf;
            n = A->recv_comm.n_msgs + A->send_comm.n_msgs;

            MPIL_Info* mpil_info;
            MPIL_Info_init(&mpil_info);
            MPIL_Topo* mpil_topo;

            MPIL_Topo_init(A->recv_comm.n_msgs,
                               A->recv_comm.procs.data(),
                               MPI_UNWEIGHTED,
                               A->send_comm.n_msgs,
                               A->send_comm.procs.data(),
                               MPI_UNWEIGHTED,
                               mpil_info,
                               &mpil_topo);
            std::vector<long> global_send_idx(A->send_comm.size_msgs);
            for (int i = 0; i < A->send_comm.size_msgs; i++)
            {
                global_send_idx[i] = A->send_comm.idx[i] + A->first_row;
            }
            MPIL_Neighbor_alltoallv_init_ext_topo(sendbuf,
                                                  A->send_comm.counts.data(),
                                                  A->send_comm.ptr.data(),
                                                  global_send_idx.data(),
                                                  MPI_DOUBLE,
                                                  recvbuf,
                                                  A->recv_comm.counts.data(),
                                                  A->recv_comm.ptr.data(),
                                                  A->off_proc_columns.data(),
                                                  MPI_DOUBLE,
                                                  mpil_topo,
                                                  mpil_world(),
                                                  mpil_info,
                                                  &spmv_req);

            MPIL_Info_free(&mpil_info);
            MPIL_Topo_free(&mpil_topo);   
        }

        void start()
        {
            MPIL_Start(spmv_req);
        }

        void wait()
        {
            MPIL_Wait(spmv_req, MPI_STATUS_IGNORE);
        }

    private:
        MPIL_Request* spmv_req;

};

#endif
