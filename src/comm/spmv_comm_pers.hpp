#ifndef SPMV_COMM_PERS_HPP
#define SPMV_COMM_PERS_HPP

#include "spmv_comm.hpp"

class SpMVCommPers : public SpMVComm
{
    public:
        SpMVCommPers() : SpMVComm()
        {
            spmv_req = NULL;
        }

        ~SpMVCommPers()
        {
            for (int i = 0; i < n; i++)
            {
                MPI_Request_free(&(spmv_req[i]));
            }

            if (spmv_req)
            {
                delete[] spmv_req;
            }
        }

        void init(ParMat& A_in, double* sbuf, double* rbuf)
        {
            A = &A_in;
            sendbuf = sbuf;
            recvbuf = rbuf;
            n = A->recv_comm.n_msgs + A->send_comm.n_msgs;
            spmv_req = new MPI_Request[n];

            int proc, start, end;
            int tag = 0;

            for (int i = 0; i < A->recv_comm.n_msgs; i++)
            {
                proc  = A->recv_comm.procs[i];
                start = A->recv_comm.ptr[i];
                end   = A->recv_comm.ptr[i + 1];
                MPI_Recv_init(&(recvbuf[start]),
                          (int)(end - start),
                          MPI_DOUBLE,
                          proc,
                          tag,
                          MPI_COMM_WORLD,
                          &(spmv_req[i]));
            }

            for (int i = 0; i < A->send_comm.n_msgs; i++)
            {
                proc  = A->send_comm.procs[i];
                start = A->send_comm.ptr[i];
                end   = A->send_comm.ptr[i + 1];
                MPI_Send_init(&(sendbuf[start]),
                          (int)(end - start),
                          MPI_DOUBLE,
                          proc,
                          tag,
                          MPI_COMM_WORLD,
                          &(spmv_req[A->recv_comm.n_msgs + i]));
            }
        }

        void start()
        {
            MPI_Startall(n, spmv_req);            
        }

        void wait()
        {
            MPI_Waitall(n, spmv_req, MPI_STATUSES_IGNORE);
        }

    private:
        MPI_Request* spmv_req;

};

#endif
