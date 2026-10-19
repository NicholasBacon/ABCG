# ABCG
<i>A</i>manda <i>B</i>ienz's _CG_ solver. 

Currently offers two variants:
1. Baseline MPI version
2. MPI Advance version

By default, the code will only build the MPI version. To enable the MPI Advance version, make sure that CMake can find your installation of MPI Advance and set the `USE_LOCALITY` flag. For an older locality_aware (e.g. 752a19b, whose API spells the neighbor setters `alogorithm` and has no `MPIL_COMM_WORLD` or MPI Advance allreduce) also set `-DMPIL_OLD_API=ON`; `benchmark_cg_mpil` is not built then, and the MPI Advance allreduce falls back to `MPI_Allreduce`, which is what `ALLREDUCE_PMPI` calls. With an MPI that has no persistent allreduce (MPI 3.1 MPICH
derivatives such as MVAPICH2 2.3; Open MPI 4.x provides `MPIX_Allreduce_init`),
`benchmark_cg_mpi` is not built; the pilots below are unaffected. Both modes are fairly simple CMake builds:
```bash
mkdir build && cd build
# MPI Only build
cmake ..
# MPI + MPI Advance build
cmake -DUSE_LOCALITY=ON -DCMAKE_PREFIX_PATH=<path to MPI Advance> ..
make
```

To run, point the executable to the path of the `.pm` matrix file to use. An optional second parameter that controls the number of tests to do can be added after the matrix path like so:
```bash
# Assuming you are in the build directory:
<mpi running command> ./benchmarks/benchmark_cg_mpi ../test_data/Dubcova2.pm
# To run the MPI advance version
<mpi running command> ./benchmarks/benchmark_cg_mpil ../test_data/Dubcova2.pm
```

The clang-formatting for this project uses a column width of 100 characters per file to help some of the longer C++ tuples remain in a readable fashion.

## Pilots and timing runs

Two further programs time pieces of the solver the way irregular-benchmarking times its
gathers: the first `k` iterations are totalled as `startupWindow`, the next `n` as
`timedTotal`, and every time is reduced over ranks (min, max, average). `-c` picks the arm:
`MPI` (ABCG's Isend/Irecv exchange and `MPI_Allreduce`) or, in a `USE_LOCALITY` build, MPI
Advance's neighbor alltoallv with `STANDARD` or `LOCALITY` and MPI Advance's allreduce with
`ALLREDUCE_PMPI`. `--persistent on|off` (default `on`) picks MPI Advance's persistent
neighbor alltoallv (init once, then start/wait) or one call per exchange. `-P NAME` labels
the output rows and `-W none|nearest|all` runs irregular-benchmarking's untimed warmup first.
`-s SOCKET|NUMA|NODE` calls `MPIL_Set_split` for the STANDARD and LOCALITY arms (old-API builds
only), like irregular-benchmarking's `-s`; the MPI arm ignores it.
```bash
<mpi running command> ./benchmarks/benchmark_spmv MATRIX.pm [-c MPI|STANDARD|LOCALITY] [-k 25] [-n 100] [--calc 10] [--persistent on|off] [-P NAME] [-W nearest] [--iter-prefix PREFIX [--iter-raw] [--iter-compress]]
<mpi running command> ./benchmarks/benchmark_cg_pilot MATRIX.pm [-c MPI|STANDARD|LOCALITY] [-k 25] [-n 100] [--calc 10] [--persistent on|off] [-P NAME] [-W nearest] [--iter-prefix PREFIX [--iter-raw] [--iter-compress]]
```
- `benchmark_spmv` runs only the SpMV halo exchange (pack, exchange, barrier), with no
  multiply, allreduce, or vector math, and checks that every received value is the entry
  the halo asked for (`verifyErrors`).
- `benchmark_cg_pilot` times iterations of `CG()`'s main loop (an SpMV, two allreduce inner
  products, and the vector updates), restarting from `x = 0` whenever a solve converges, and
  reports `cgItersPerSolve` beside `cgReferenceIters`, the count from `CG()` itself
  (`cgItersPerSolve` is 0 when no solve finished within `k + n` iterations).
- Both programs' checks fail the run (exit 1, "check FAILED"). `benchmark_spmv` checks every
  received value. `benchmark_cg_pilot` runs its first solve again after timing (untimed)
  with ABCG's own Isend/Irecv exchange and `MPI_Allreduce`, and compares the residual after
  every iteration (`residualMaxRelDiff`, `residualItersCompared`): it fails above 1e-8
  relative or if the two solves differ in length. A wrong halo value moves the residuals at
  order one within a few iterations; a correct one matched bitwise in local tests. It also
  compares iterations per finished solve with `CG()`'s count, but only at the default
  `--max-iter 500` with convergence on, since `CG()` caps every solve at 500 whatever its
  `max_iters` argument (`src/cg.hpp`); otherwise, or when no solve finished within
  `k + n`, it prints "count check skipped". None of the six campaign matrices converges
  within 500 iterations at the default tolerance, so there only the residual check
  applies.
- `benchmark_cg_pilot --max-iter N` caps each solve at N iterations (default 500), and
  `--ignore-convergence` runs every solve to the cap whatever its residual. A solve also
  ends, and restarts from `x = 0`, when its squared residual is no longer above 1e-250
  (zero, NaN, or near the denormals, whose slow arithmetic would show as a timing change);
  `underflowRestarts` counts those. In local tests the residual stalled near rounding
  level rather than falling that far, so the guard fired only when `b` was scaled to
  1e-120.
- `--calc SECONDS` (pilot mode) prints the `n` for which startup + q·(k + n) fills
  `SECONDS`, from the max-over-ranks totals, in the line format irregular-benchmarking's
  `pilot.sbatch` reads into `targets`.
- `--iter-prefix PREFIX` keeps every iteration's time in memory and, after the run, writes
  `PREFIX_NAME_s0.iter.npy`, one row per iteration (window first) reduced over ranks, in
  irregular-benchmarking's format (its README, "Per-iteration times"): `start_s` (rank 0's
  wall-clock start, seconds since the epoch), and for `benchmark_spmv` `apply_max_ms`,
  `apply_min_ms`, `apply_mean_ms` (pack and exchange, stopped before the barrier),
  `slowest_rank` (largest apply), `iter_max_ms`, `iter_min_ms` (through the barrier), and
  `iter_min_rank` (smallest barrier-inclusive time: the rank the others waited for when it
  was delayed outside the exchange). `benchmark_cg_pilot` has no barrier, and its only
  synchronizations are CG's allreduces, so it writes `iter_max_ms`, `iter_min_ms`,
  `iter_mean_ms`, `slowest_rank`, and `iter_min_rank` over whole iterations; a solve's
  restart is charged to its first iteration, as in the totals. `--iter-raw` adds
  `PREFIX_NAME_s0.raw.npy` with every rank's rows, shape (ranks, iterations);
  `--iter-compress` gzips the files (`.npy.gz`). Read with `np.load(path)`, or
  `np.load(gzip.open(path))` when compressed.
  Each run prints `-iter write (s):`, the time rank 0 took to write these files after
  timing.
  `--iter-raw` also measures each rank's clock offset from rank 0 before and after the
  loop and writes `PREFIX_NAME_s0.offsets.npy`, for irregular-benchmarking's
  `scripts/barrier-skew.py` (barrier exit skew; for the CG program, the spread of
  iteration starts after its allreduces).
  Needs zlib.
- Both programs print `-MPI_Wtick:` and `-CLOCK_MONOTONIC resolution (ns):` (the clock
  `MPI_Wtime` reads in Open MPI 4.1 on Linux) with their settings.
- Both programs print `-source commit:` and `-locality_aware commit:` with their settings,
  and carry them as one `PROVENANCE: source=<..> locality_aware=<..>` string that
  `strings` finds. CMake takes them from `-DSOURCE_COMMIT` and `-DLOCALITY_AWARE_COMMIT`;
  without `-DSOURCE_COMMIT` it reads `git rev-parse HEAD` (`-dirty` with uncommitted
  changes) at configure time when the source is the top of its own checkout, and records
  `unknown (not a git checkout)` otherwise. `LOCALITY_AWARE_COMMIT` defaults to `none`
  without `USE_LOCALITY` and `unknown` with it.

### Capturing the SpMV pattern with Vernier

`-DABCG_VERNIER=ON` (with `-Dvernier_DIR=<Vernier install>/share/vernier`) links
`benchmark_spmv` against [Vernier](https://github.com/JDTruj2018/Vernier) and wraps each
halo exchange (start and wait only; setup, warmup, and the barrier stay outside) in a Vernier
pattern named by `-P`, by default the matrix's name, so Vernier's files, which are named only
by rank, still say which matrix they hold. Only the MPI arm can be captured: Vernier records
every MPI call in the process, so CMake refuses `ABCG_VERNIER` together with `USE_LOCALITY`.
One exchange is a capture of the whole pattern:
```bash
VERNIER_NO_REQUESTS=1 \
    <mpi running command> ./benchmarks/benchmark_spmv MATRIX.pm -c MPI -k 0 -n 1 -W none
```
Vernier writes into the working directory: `vernier-0-0.hist-aggregate.json` (all ranks
aggregated, on rank 0), and per rank `vernier-<rank>-0.hist-hierarchical.json` (per call
type and partner, the message sizes; what irregular-benchmarking's fine patterns replay),
`vernier-<rank>-0.hist-combined.json`, and the trace `vernier-<rank>-0.txt`. `VERNIER_NO_REQUESTS` is needed with Vernier 09d650b, whose
aggregate output otherwise aborts every rank but 0 in `MPI_Finalize`; it only drops request
ids from the trace. irregular-benchmarking's `pilot.sbatch` runs this capture for each
matrix when `VERNIER_SPMV` is set.
