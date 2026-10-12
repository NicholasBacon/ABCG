# ABCG
<i>A</i>manda <i>B</i>ienz's _CG_ solver. 

Currently offers two variants:
1. Baseline MPI version
2. MPI Advance version

By default, the code will only build the MPI version. To enable the MPI Advance version, make sure that CMake can find your installation of MPI Advance and set the `USE_LOCALITY` flag. For an older locality_aware (e.g. 752a19b, whose API spells the neighbor setters `alogorithm` and has no `MPIL_COMM_WORLD` or MPI Advance allreduce) also set `-DMPIL_OLD_API=ON`; `benchmark_cg_mpil` is not built then, and the MPI Advance allreduce falls back to `MPI_Allreduce`, which is what `ALLREDUCE_PMPI` calls. Both modes are fairly simple CMake builds:
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
<mpi running command> ./benchmarks/benchmark_spmv MATRIX.pm [-c MPI|STANDARD|LOCALITY] [-k 25] [-n 100] [--calc 10] [--persistent on|off] [-P NAME] [-W nearest]
<mpi running command> ./benchmarks/benchmark_cg_pilot MATRIX.pm [-c MPI|STANDARD|LOCALITY] [-k 25] [-n 100] [--calc 10] [--persistent on|off] [-P NAME] [-W nearest]
```
- `benchmark_spmv` runs only the SpMV halo exchange (pack, exchange, barrier), with no
  multiply, allreduce, or vector math, and checks that every received value is the entry
  the halo asked for (`verifyErrors`).
- `benchmark_cg_pilot` times iterations of `CG()`'s main loop (an SpMV, two allreduce inner
  products, and the vector updates), restarting from `x = 0` whenever a solve converges, and
  reports `cgItersPerSolve` beside `cgReferenceIters`, the count from `CG()` itself
  (`cgItersPerSolve` is 0 when no solve finished within `k + n` iterations).
- Both checks fail the run: `benchmark_spmv` exits 1 when any received value is wrong, and
  `benchmark_cg_pilot` exits 1 when a finished solve took a different number of iterations
  than `CG()` (it prints "check skipped" when no solve finished within `k + n`).
- `--calc SECONDS` (pilot mode) prints the `n` for which startup + q·(k + n) fills
  `SECONDS`, from the max-over-ranks totals, in the line format irregular-benchmarking's
  `pilot.sbatch` reads into `targets`.

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
