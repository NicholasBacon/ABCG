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
    "       [-s SOCKET|NUMA|NODE] [--iter-prefix PREFIX [--iter-raw] [--iter-compress]]\n"
    "  --calc prints the CG iteration count n that makes startup + q*(k + n) take SECONDS\n"
    "  -P names the output rows (default: the matrix file name); -W is irregular-benchmarking's\n"
    "  untimed warmup; --persistent (STANDARD/LOCALITY, default on) picks MPI Advance's\n"
    "  persistent neighbor alltoallv or one call per exchange; -s calls MPIL_Set_split (old\n"
    "  locality_aware API builds only)\n"
    "  --iter-prefix writes PREFIX_NAME_s0.iter.npy, one row per CG iteration reduced over\n"
    "  ranks; --iter-raw adds PREFIX_NAME_s0.raw.npy with every rank's rows (and the clock\n"
    "  offsets); --iter-compress gzips them (.npy.gz)\n"
    "  --max-iter N caps each solve at N iterations (default 500); --ignore-convergence runs\n"
    "  every solve to the cap whatever its residual\n";

struct IterTimes
{
    int k;
    long done = 0;          // iterations run so far, across solves
    double window_ms = 0.0;
    double timed_ms = 0.0;
    int full_solves = 0;    // solves that converged (or hit max_iter) before the budget ran out
    long full_solve_iters = 0;
    int underflow_restarts = 0; // solves ended because rr_inner fell to 1e-250 or below
    double last_time = 0.0;
    IterLog log;            // --iter-prefix: each iteration's time, as well as the totals
    double last_wall = 0.0; // wall-clock start of the current iteration, when log.on

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
        if (log.on)
        {
            log.record(last_wall, 0.0, elapsed);
            last_wall = wall_seconds();
        }
    }
};

// CG() from src/cg.hpp, run from x = 0 until it converges (unless ignore_convergence),
// reaches max_iter, or the iteration budget `total` runs out, timing each pass of the main
// loop. A solve also ends when rr_inner is no longer above 1e-250 (zero, NaN, or near the
// denormals, whose arithmetic is slow and would show as a timing change). With residuals
// set, norm_r after every iteration is appended to it.
static void timed_solve(ParMat& A, std::vector<double>& x, std::vector<double>& b,
                        SpMVComm* spmv_comm, AllreduceComm* allreduce_comm, long total,
                        IterTimes& times, int max_iter, bool ignore_convergence,
                        std::vector<double>* residuals, double tol = 1e-06)
{
    std::vector<double> r(A.local_rows);
    std::vector<double> p(A.local_rows);
    std::vector<double> Ap(A.local_rows);
    int recompute_r = 8;
    const double underflow = 1e-250;

    std::fill(x.begin(), x.end(), 0);
    r = b;
    spmv(-1.0, A, x, 1.0, r, spmv_comm);
    p = r;
    double rr_inner = inner_product(r, r, allreduce_comm);
    double norm_r = sqrt(rr_inner);
    if (norm_r != 0.0) tol = tol * norm_r;

    int iter = 0;
    while ((ignore_convergence || norm_r > tol) && iter < max_iter && rr_inner > underflow &&
           times.done < total)
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
        if (residuals) residuals->push_back(norm_r);
    }
    if (!(rr_inner > underflow)) times.underflow_restarts++;
    if (!((ignore_convergence || norm_r > tol) && iter < max_iter && rr_inner > underflow))
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
    times.log.on = !opt.iter_prefix.empty();
    if (times.log.on) times.log.reserve(total);
    if (times.log.on && opt.iter_raw) times.log.offsets_before = measure_clock_offsets();
    MPI_Barrier(MPI_COMM_WORLD);
    times.last_time = MPI_Wtime();
    if (times.log.on) times.last_wall = wall_seconds();
    int solves = 0;
    std::vector<double> residuals; // the first solve's norm_r per iteration
    residuals.reserve(opt.max_iter);
    while (times.done < total)
    {
        timed_solve(A, x, b, spmv_comm, allreduce_comm, total, times, opt.max_iter,
                    opt.ignore_convergence, solves == 0 ? &residuals : nullptr);
        solves++;
    }
    if (times.log.on && opt.iter_raw) times.log.offsets_after = measure_clock_offsets();

    // Untimed reference for the halo check: the first solve again, with ABCG's own
    // Isend/Irecv exchange and MPI_Allreduce. Correct halo values give the same SpMV
    // results, so the residual histories should agree; a wrong value moves them at order
    // one within a few iterations. Every rank holds the same residuals (allreduced).
    std::vector<double> reference_residuals;
    {
        std::vector<double> ref_sendbuf(A.send_comm.size_msgs), ref_recvbuf(A.recv_comm.size_msgs);
        double ref_local, ref_global;
        SpMVComm ref_spmv;
        ref_spmv.init(A, ref_sendbuf.data(), ref_recvbuf.data());
        AllreduceComm ref_allreduce;
        ref_allreduce.init(&ref_local, &ref_global);
        IterTimes ref_times;
        ref_times.k = 0;
        timed_solve(A, x, b, &ref_spmv, &ref_allreduce, (long)residuals.size(), ref_times,
                    opt.max_iter, opt.ignore_convergence, &reference_residuals);
    }
    double residual_diff = 0.0;
    size_t compared = std::min(residuals.size(), reference_residuals.size());
    for (size_t i = 0; i < compared; i++)
    {
        double ref = reference_residuals[i];
        double d = std::fabs(residuals[i] - ref);
        residual_diff = std::max(residual_diff, ref != 0.0 ? d / std::fabs(ref) : d);
        if (std::isnan(residuals[i]) != std::isnan(ref)) residual_diff = INFINITY;
    }
    // A wrong halo can also change where the first solve converges.
    bool length_differs = residuals.size() != reference_residuals.size();

    // ABCG's own CG() on the same system, from x = 0. CG() caps every solve at 500
    // whatever its max_iters argument (src/cg.hpp), so its count is comparable only at the
    // default cap with convergence on.
    std::fill(x.begin(), x.end(), 0);
    int reference_iters = CG<SpMVComm, AllreduceComm>(A, x, b);
    bool count_comparable = opt.max_iter == 500 && !opt.ignore_convergence;

    print_settings(opt, opt.n, true);
    double per_solve = times.full_solves ? (double)times.full_solve_iters / times.full_solves : 0.0;
    const char* labels[] = {"setupTime", "startupWindow", "timedTotal", "cgSolves",
                            "cgItersPerSolve", "cgReferenceIters", "numberOfmessages", "data_sent",
                            "residualMaxRelDiff", "residualItersCompared", "underflowRestarts"};
    double local[] = {setup_ms, times.window_ms, times.timed_ms, (double)solves, per_solve,
                      (double)reference_iters, (double)A.send_comm.n_msgs,
                      A.send_comm.size_msgs * (double)sizeof(double), residual_diff,
                      (double)compared, (double)times.underflow_restarts};
    print_rows(opt.name, 11, labels, local);
    if (opt.calc_s > 0) print_calc(opt, times.window_ms, times.timed_ms, opt.n);
    if (times.log.on)
    {
        // Rank 0 writes every file, so its elapsed time is the whole cost; printed so a run
        // shows what recording costs between runs.
        double write_start = MPI_Wtime();
        write_iter_files(opt.iter_prefix, opt.name, 0, times.log, opt.iter_raw, opt.iter_compressed);
        if (rank == 0) printf("-iter write (s): %.3f\n", MPI_Wtime() - write_start);
    }

    // Both checks fail the run. The residual check compares the first solve with the
    // reference above; 1e-8 is far below the order-one change a wrong halo value makes.
    // The count check: CG takes the same iterations whatever carries the halo, so a solve
    // that differs from CG()'s count means wrong exchanged values. Every rank holds the
    // same values (they follow from allreduced inner products).
    bool residual_failed = !(residual_diff <= 1e-8) || length_differs;
    if (rank == 0 && residual_failed)
        printf("(%s)  check FAILED: residuals differ from the plain-MPI reference by up to %g "
               "(relative) over %zu iterations; first solve %zu iterations, reference %zu\n",
               opt.name.c_str(), residual_diff, compared, residuals.size(), reference_residuals.size());
    bool count_failed = count_comparable && times.full_solves > 0 && per_solve != reference_iters;
    if (rank == 0 && count_failed)
        printf("(%s)  check FAILED: %.2f iterations per solve, CG() took %d\n", opt.name.c_str(),
               per_solve, reference_iters);
    if (rank == 0 && !count_comparable)
        printf("(%s)  count check skipped: --max-iter or --ignore-convergence differs from CG()\n",
               opt.name.c_str());
    else if (rank == 0 && times.full_solves == 0)
        printf("(%s)  count check skipped: no solve finished within k + n iterations\n", opt.name.c_str());
    bool failed = residual_failed || count_failed;

    delete spmv_comm;
    delete allreduce_comm;
#ifdef MPIL
    mpil_finalize();
#endif
    MPI_Finalize();
    return failed ? 1 : 0;
}
