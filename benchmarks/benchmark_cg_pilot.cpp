// Times CG iterations: k window iterations then n timed ones, counted across as many
// solves as that takes (each solve restarts from x = 0), and with --calc prints the
// iteration count that fills a time budget. Each iteration is one pass of CG()'s main
// loop (src/cg.hpp): an SpMV, two allreduce inner products, and the vector updates.
// The arm (-c) sets the SpMV communication and the allreduce: MPI_Allreduce for MPI,
// MPI Advance's allreduce with ALLREDUCE_PMPI for STANDARD and LOCALITY.
#include "pilot_common.hpp"

#include <algorithm>
#include <cmath>

static const char* usage =
    "usage: benchmark_cg_pilot MATRIX.pm [-c MPI|STANDARD|LOCALITY] [-k K] [-n N] [--calc SECONDS]\n"
    "       [-P NAME] [-W none|nearest|all] [--persistent on|off]\n"
    "       [-s SOCKET|NUMA|NODE]\n"
    "  --calc prints the CG iteration count n that makes startup + q*(k + n) take SECONDS\n"
    "  -P names the output rows (default: the matrix file name); -W is irregular-benchmarking's\n"
    "  untimed warmup; --persistent (STANDARD/LOCALITY, default on) picks MPI Advance's\n"
    "  persistent neighbor alltoallv or one call per exchange; -s calls MPIL_Set_split (old\n"
    "  locality_aware API builds only)\n";

struct IterTimes
{
    int k;
    long done = 0;          // iterations run so far, across solves
    double window_ms = 0.0;
    double timed_ms = 0.0;
    int full_solves = 0;    // solves that converged (or hit max_iter) before the budget ran out
    long full_solve_iters = 0;
    double last_time = 0.0;

    // Each iteration is timed from the end of the previous one, so a solve's restart
    // (r = b - A*x and its inner product) is charged to that solve's first iteration.
    void tick()
    {
        double now = MPI_Wtime();
        double elapsed = (now - last_time) * 1000.0;
        last_time = now;
        if (done < k) window_ms += elapsed;
        else timed_ms += elapsed;
        done++;
    }
};

// CG() from src/cg.hpp, run from x = 0 until it converges, reaches max_iter, or the
// iteration budget `total` runs out, timing each pass of the main loop.
static void timed_solve(ParMat& A, std::vector<double>& x, std::vector<double>& b,
                        SpMVComm* spmv_comm, AllreduceComm* allreduce_comm, long total,
                        IterTimes& times, double tol = 1e-06)
{
    std::vector<double> r(A.local_rows);
    std::vector<double> p(A.local_rows);
    std::vector<double> Ap(A.local_rows);
    int max_iter = 500;
    int recompute_r = 8;

    std::fill(x.begin(), x.end(), 0);
    r = b;
    spmv(-1.0, A, x, 1.0, r, spmv_comm);
    p = r;
    double rr_inner = inner_product(r, r, allreduce_comm);
    double norm_r = sqrt(rr_inner);
    if (norm_r != 0.0) tol = tol * norm_r;

    int iter = 0;
    while (norm_r > tol && iter < max_iter && times.done < total)
    {
        spmv(1.0, A, p, 0.0, Ap, spmv_comm);
        double App_inner = inner_product(Ap, p, allreduce_comm);
        if (App_inner < 0.0)
        {
            printf("Indefinite matrix detected in CG! Aborting...\n");
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
        double alpha = rr_inner / App_inner;
        axpy(alpha, x, p);
        if ((iter % recompute_r) && iter > 0)
        {
            axpy(-1.0 * alpha, r, Ap);
        }
        else
        {
            r = b;
            spmv(-1.0, A, x, 1.0, r, spmv_comm);
        }
        double next_inner = inner_product(r, r, allreduce_comm);
        double beta = next_inner / rr_inner;
        scale(beta, p);
        axpy(1.0, p, r);
        rr_inner = next_inner;
        norm_r = sqrt(rr_inner);
        iter++;
        times.tick();
    }
    if (!(norm_r > tol && iter < max_iter))
    {
        times.full_solves++;
        times.full_solve_iters += iter;
    }
}

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

    // b = A * x for a random x, as the ABCG drivers set up the system; seeded by rank
    // instead of the clock so a run can be repeated.
    std::vector<double> x(A.local_cols);
    std::vector<double> b(A.local_rows);
    std::vector<double> sendbuf(A.send_comm.size_msgs);
    std::vector<double> recvbuf(A.recv_comm.size_msgs);
    double local_sum, global_sum;
    srand(rank + 1);
    std::generate(x.begin(), x.end(), [&]() { return (double)(rand()) / RAND_MAX; });

    MPI_Barrier(MPI_COMM_WORLD);
    double t = MPI_Wtime();
    SpMVComm* spmv_comm = make_spmv_comm(opt);
    spmv_comm->init(A, sendbuf.data(), recvbuf.data());
    AllreduceComm* allreduce_comm = make_allreduce_comm(opt);
    allreduce_comm->init(&local_sum, &global_sum);
    double setup_ms = (MPI_Wtime() - t) * 1000.0;
    spmv(1.0, A, x, 0.0, b, spmv_comm);

    IterTimes times;
    times.k = opt.k;
    long total = opt.k + opt.n;
    MPI_Barrier(MPI_COMM_WORLD);
    times.last_time = MPI_Wtime();
    int solves = 0;
    while (times.done < total)
    {
        timed_solve(A, x, b, spmv_comm, allreduce_comm, total, times);
        solves++;
    }

    // The reference: ABCG's own CG() on the same system, from x = 0.
    std::fill(x.begin(), x.end(), 0);
    int reference_iters = CG<SpMVComm, AllreduceComm>(A, x, b);

    print_settings(opt, opt.n);
    double per_solve = times.full_solves ? (double)times.full_solve_iters / times.full_solves : 0.0;
    const char* labels[] = {"setupTime", "startupWindow", "timedTotal", "cgSolves",
                            "cgItersPerSolve", "cgReferenceIters", "numberOfmessages", "data_sent"};
    double local[] = {setup_ms, times.window_ms, times.timed_ms, (double)solves, per_solve,
                      (double)reference_iters, (double)A.send_comm.n_msgs,
                      A.send_comm.size_msgs * (double)sizeof(double)};
    print_rows(opt.name, 8, labels, local);
    if (opt.calc_s > 0) print_calc(opt, times.window_ms, times.timed_ms, opt.n);

    // CG takes the same iterations whatever carries the halo, so a solve that differs from
    // CG()'s count means wrong exchanged values; it fails the run. Every rank holds the
    // same counts (they follow from allreduced inner products).
    bool failed = times.full_solves > 0 && per_solve != reference_iters;
    if (rank == 0 && failed)
        printf("(%s)  check FAILED: %.2f iterations per solve, CG() took %d\n", opt.name.c_str(),
               per_solve, reference_iters);
    if (rank == 0 && times.full_solves == 0)
        printf("(%s)  check skipped: no solve finished within k + n iterations\n", opt.name.c_str());

    delete spmv_comm;
    delete allreduce_comm;
#ifdef MPIL
    mpil_finalize();
#endif
    MPI_Finalize();
    return failed ? 1 : 0;
}
