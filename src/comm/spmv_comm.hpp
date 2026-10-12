#ifndef SPMV_COMM_HPP
#define SPMV_COMM_HPP

#include "mpi.h"

class SpMVComm
{
    public:
        SpMVComm()
        {
            A = NULL;
            sendbuf = NULL;
            recvbuf = NULL;
            spmv_req = NULL;
            n = 0;
        }

        // virtual: CG(), spmv() and the pilots hold SpMVComm*, so without it every call
        // runs this class's Isend/Irecv exchange even for the MPI Advance subclasses.
        virtual ~SpMVComm()
        {
            if (spmv_req)
            {
                delete[] spmv_req;
            }
        }

        virtual void init(ParMat& A_in, double* sbuf, double* rbuf)
        {
            A = &A_in;
            sendbuf = sbuf;
            recvbuf = rbuf;
            n = A->recv_comm.n_msgs + A->send_comm.n_msgs;
            spmv_req = new MPI_Request[n];
        }

        void pack(std::vector<double>& x)
        {
            for (int i = 0; i < A->send_comm.size_msgs; i++)
            {
                sendbuf[i] = x[A->send_comm.idx[i]];
            }
        }

        virtual void start()
        {
            int proc, start, end;
            int tag = 0;

            for (int i = 0; i < A->recv_comm.n_msgs; i++)
            {
                proc  = A->recv_comm.procs[i];
                start = A->recv_comm.ptr[i];
                end   = A->recv_comm.ptr[i + 1];
                MPI_Irecv(&(recvbuf[start]),
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
                MPI_Isend(&(sendbuf[start]),
                          (int)(end - start),
                          MPI_DOUBLE,
                          proc,
                          tag,
                          MPI_COMM_WORLD,
                          &(spmv_req[A->recv_comm.n_msgs + i]));
            }

        }

        virtual void wait()
        {
            if (n)
            {
                MPI_Waitall(n, 
                        spmv_req,
                        MPI_STATUSES_IGNORE);
            }
        }

        double* get_recvbuf()
        {
            return recvbuf;
        }


    protected:
        double* sendbuf;
        double* recvbuf;
        int n;
        ParMat* A;

    private:
        MPI_Request* spmv_req;
};

#endif
