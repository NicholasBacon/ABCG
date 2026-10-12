// SpMV halo exchange only: each iteration packs the entries of x other ranks need,
// exchanges the halo, and waits at a barrier. No multiply, allreduce, or CG vector
// math. Timed like irregular-benchmarking under -b: barrier to barrier, the first k
// iterations totalled as startupWindow and the next n as timedTotal.
#include "pilot_common.hpp"

#include <cmath>
#ifdef ABCG_VERNIER
#include "vernier.h"
#endif

static const char* usage =
    "usage: benchmark_spmv MATRIX.pm [-c MPI|STANDARD|LOCALITY] [-k K] [-n N] [--calc SECONDS]\n"
    "       [-P NAME] [-W none|nearest|all] [--persistent on|off]\n"
    "       [-s SOCKET|NUMA|NODE]\n"
    "  --calc prints the n that makes startup + q*(k + n) take SECONDS (pilot mode)\n"
    "  -P names the output rows (default: the matrix file name); -W is irregular-benchmarking's\n"
    "  untimed warmup; --persistent (STANDARD/LOCALITY, default on) picks MPI Advance's\n"
    "  persistent neighbor alltoallv or one call per exchange; -s calls MPIL_Set_split (old\n"
    "  locality_aware API builds only)\n";

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
#ifdef MPIL
    mpil_init();
#endif
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    PilotOptions opt = parse_pilot_options(argc, argv, usage);
    run_warmup(opt.warmup);

    ParMat A;
    readParMatrix(opt.file.c_str(), A);
    form_comm(A);

    // Each x entry holds its global column, so after an exchange every received
    // value must equal the global column the halo asked for.
    std::vector<double> x(A.local_cols);
    for (int i = 0; i < A.local_cols; i++) x[i] = A.first_col + i;
    std::vector<double> sendbuf(A.send_comm.size_msgs);
    std::vector<double> recvbuf(A.recv_comm.size_msgs);

    MPI_Barrier(MPI_COMM_WORLD);
    double t = MPI_Wtime();
    SpMVComm* comm = make_spmv_comm(opt);
    comm->init(A, sendbuf.data(), recvbuf.data());
    double setup_ms = (MPI_Wtime() - t) * 1000.0;

    double window_ms = 0.0, timed_ms = 0.0;
    MPI_Barrier(MPI_COMM_WORLD);
    for (long i = 0; i < opt.k + opt.n; i++)
    {
        t = MPI_Wtime();
        comm->pack(x);
#ifdef ABCG_VERNIER
        // Only the exchange's own messages: the barrier, setup and warmup stay outside.
        // The region carries the pattern name (-P, default the matrix), since Vernier's
        // files are otherwise named only by rank.
        begin_pattern(opt.name.c_str());
#endif
        comm->start();
        comm->wait();
#ifdef ABCG_VERNIER
        end_pattern();
#endif
        MPI_Barrier(MPI_COMM_WORLD);
        double elapsed = (MPI_Wtime() - t) * 1000.0;
        if (i < opt.k) window_ms += elapsed;
        else timed_ms += elapsed;
    }

    double bad = 0;
    for (int j = 0; j < A.recv_comm.size_msgs; j++)
    {
        if (recvbuf[j] != (double)A.off_proc_columns[j]) bad++;
    }

    print_settings(opt, opt.n);
    const char* labels[] = {"setupTime", "startupWindow", "timedTotal", "numberOfmessages",
                            "data_sent", "verifyErrors"};
    double local[] = {setup_ms, window_ms, timed_ms, (double)A.send_comm.n_msgs,
                      A.send_comm.size_msgs * (double)sizeof(double), bad};
    print_rows(opt.name, 6, labels, local);
    if (opt.calc_s > 0) print_calc(opt, window_ms, timed_ms, opt.n);

    // A wrong halo value fails the run, so a campaign's manifest records it as an error.
    double bad_total;
    MPI_Allreduce(&bad, &bad_total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0 && bad_total > 0)
        printf("(%s)  check FAILED: %.0f received halo values were wrong\n", opt.name.c_str(), bad_total);

    delete comm;
#ifdef MPIL
    mpil_finalize();
#endif
    MPI_Finalize();
    return bad_total > 0 ? 1 : 0;
}
