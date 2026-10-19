#ifndef PILOT_COMMON_HPP
#define PILOT_COMMON_HPP

// Shared by benchmark_spmv and benchmark_cg_pilot: the command line, the choice of
// SpMV communication and allreduce (the arm), and output in irregular-benchmarking's format, so
// that benchmark's analysis scripts and pilot.sbatch's targets parsing read these
// outputs unchanged.

#include "cg.hpp"

#include <zlib.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>

#include <time.h>

// irregular-benchmarking's scripts/provenance.sh finds this with strings(1), so it must
// stay one literal; the commits come from CMake (the top-level CMakeLists.txt).
__attribute__((used)) static const char provenance_line[] =
    "PROVENANCE: source=" ABCG_SOURCE_COMMIT " locality_aware=" LOCALITY_AWARE_COMMIT;

struct PilotOptions
{
    std::string file;
    std::string name;      // -P, or the matrix file name without directory or .pm
    std::string arm = "MPI";
    int k = 25;            // startup window: the first k iterations, totalled apart
    long n = 100;          // timed iterations after the window
    double calc_s = 0.0;   // > 0: print the n that makes the run take this many seconds
    std::string warmup = "none";
    bool persistent = true; // STANDARD/LOCALITY: persistent init/start/wait, or one call per exchange
    std::string split;      // -s SOCKET|NUMA|NODE: MPIL_Set_split, old API only; empty leaves the default
    std::string iter_prefix;     // --iter-prefix: record per-iteration times (see IterLog)
    bool iter_raw = false;       // --iter-raw: also every rank's rows
    bool iter_compressed = false; // --iter-compress: gzip the files
    int max_iter = 500;            // --max-iter: cap on each CG solve (benchmark_cg_pilot only)
    bool ignore_convergence = false; // --ignore-convergence: run every solve to the cap
    bool cg_options = false;       // either of the two was given
};

inline PilotOptions parse_pilot_options(int argc, char* argv[], const char* usage)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    PilotOptions opt;
    for (int i = 1; i < argc; i++)
    {
        std::string a = argv[i];
        bool has_value = i + 1 < argc;
        if (a == "-c" && has_value) opt.arm = argv[++i];
        else if (a == "-k" && has_value) opt.k = atoi(argv[++i]);
        else if (a == "-n" && has_value) opt.n = atol(argv[++i]);
        else if (a == "--calc" && has_value) opt.calc_s = atof(argv[++i]);
        else if (a == "-P" && has_value) opt.name = argv[++i];
        else if (a == "-W" && has_value) opt.warmup = argv[++i];
        else if (a == "-s" && has_value) opt.split = argv[++i];
        else if (a == "--iter-prefix" && has_value) opt.iter_prefix = argv[++i];
        else if (a == "--iter-raw") opt.iter_raw = true;
        else if (a == "--iter-compress") opt.iter_compressed = true;
        else if (a == "--max-iter" && has_value)
        {
            opt.max_iter = atoi(argv[++i]);
            opt.cg_options = true;
        }
        else if (a == "--ignore-convergence")
        {
            opt.ignore_convergence = true;
            opt.cg_options = true;
        }
        else if (a == "--persistent" && has_value)
        {
            std::string v = argv[++i];
            if (v != "on" && v != "off")
            {
                if (rank == 0) printf("--persistent takes on or off\n%s", usage);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            opt.persistent = v == "on";
        }
        else if (a[0] != '-' && opt.file.empty()) opt.file = a;
        else
        {
            if (rank == 0) printf("unknown or incomplete argument %s\n%s", a.c_str(), usage);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    bool arm_ok = opt.arm == "MPI" || opt.arm == "STANDARD" || opt.arm == "LOCALITY";
    bool warmup_ok = opt.warmup == "none" || opt.warmup == "nearest" || opt.warmup == "all";
    bool iter_ok = !opt.iter_prefix.empty() || (!opt.iter_raw && !opt.iter_compressed);
    if (opt.file.empty() || !arm_ok || !warmup_ok || !iter_ok || opt.k < 0 || opt.n < 1 ||
        opt.max_iter < 1)
    {
        if (rank == 0) printf("%s", usage);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
#ifndef MPIL
    if (opt.arm != "MPI")
    {
        if (rank == 0) printf("-c %s needs a build with -DUSE_LOCALITY=ON\n", opt.arm.c_str());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
#endif
    // -s only affects MPI Advance, so the MPI arm accepts and ignores it, and a campaign
    // can pass the same flags to every arm.
    if (!opt.split.empty() && opt.arm == "MPI")
    {
        opt.split.clear();
    }
    if (!opt.split.empty())
    {
        bool split_ok = opt.split == "SOCKET" || opt.split == "NUMA" || opt.split == "NODE";
#if defined(MPIL) && defined(MPIL_OLD_API)
        if (!split_ok)
        {
            if (rank == 0) printf("-s takes SOCKET, NUMA or NODE\n%s", usage);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        MPIL_Set_split(opt.split == "SOCKET" ? SOCKET : opt.split == "NUMA" ? NUMA : NODE);
#else
        if (rank == 0) printf("-s needs a build with -DUSE_LOCALITY=ON -DMPIL_OLD_API=ON%s\n",
                              split_ok ? "" : " (and takes SOCKET, NUMA or NODE)");
        MPI_Abort(MPI_COMM_WORLD, 1);
#endif
    }
    if (opt.name.empty())
    {
        opt.name = opt.file.substr(opt.file.find_last_of('/') + 1);
        if (opt.name.size() > 3 && opt.name.compare(opt.name.size() - 3, 3, ".pm") == 0)
            opt.name.resize(opt.name.size() - 3);
    }
    return opt;
}

// irregular-benchmarking's -W: an untimed exchange of 64 B and 128 KiB messages with
// rank +/- d (nearest: d = 1; all: every d), on a duplicate of MPI_COMM_WORLD that is
// freed afterwards, so those connections are open before anything is measured.
inline void run_warmup(const std::string& mode)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    std::vector<int> distances;
    if (mode == "nearest")
    {
        distances.push_back(1);
        if (num_procs - 1 != 1) distances.push_back(num_procs - 1);
    }
    else if (mode == "all")
    {
        for (int d = 1; d < num_procs; d++) distances.push_back(d);
    }
    if (distances.empty()) return;

    MPI_Comm comm;
    MPI_Comm_dup(MPI_COMM_WORLD, &comm);
    const int sizes[2] = {1 << 6, 1 << 17};
    std::vector<char> send_buf(sizes[1], 1), recv_buf(sizes[1]);
    for (int d : distances)
    {
        int to = (rank + d) % num_procs;
        int from = (rank - d + num_procs) % num_procs;
        for (int bytes : sizes)
        {
            MPI_Request requests[2];
            MPI_Irecv(recv_buf.data(), bytes, MPI_BYTE, from, 0, comm, &requests[0]);
            MPI_Isend(send_buf.data(), bytes, MPI_BYTE, to, 0, comm, &requests[1]);
            MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);
        }
    }
    MPI_Comm_free(&comm);
}

// MPI: ABCG's Isend/Irecv halo exchange. STANDARD and LOCALITY: MPI Advance's neighbor
// alltoallv, persistent (init once, then start/wait: the INIT methods
// irregular-benchmarking's arms use) or, with --persistent off, one call per exchange.
inline SpMVComm* make_spmv_comm(const PilotOptions& opt)
{
#ifdef MPIL
    bool standard = opt.arm == "STANDARD";
    if (opt.arm != "MPI" && opt.persistent)
    {
        mpil_set_neighbor_init(standard ? NEIGHBOR_ALLTOALLV_INIT_STANDARD
                                        : NEIGHBOR_ALLTOALLV_INIT_LOCALITY);
        return new SpMVCommMPILPers();
    }
    if (opt.arm != "MPI")
    {
        mpil_set_neighbor(standard ? NEIGHBOR_ALLTOALLV_STANDARD : NEIGHBOR_ALLTOALLV_LOCALITY);
        return new SpMVCommMPIL();
    }
#endif
    return new SpMVComm();
}

// MPI: MPI_Allreduce. STANDARD and LOCALITY: MPI Advance's allreduce with ALLREDUCE_PMPI,
// which hands the call to the MPI library, so only the SpMV communication differs. The
// old API has no MPI Advance allreduce, and AllreduceCommMPIL calls MPI_Allreduce there.
inline AllreduceComm* make_allreduce_comm(const PilotOptions& opt)
{
#ifdef MPIL
    if (opt.arm != "MPI")
    {
#ifndef MPIL_OLD_API
        MPIL_Set_allreduce_algorithm(ALLREDUCE_PMPI);
#endif
        return new AllreduceCommMPIL();
    }
#endif
    return new AllreduceComm();
}

// The run settings in irregular-benchmarking's -r format, as its parse_run() reads them.
// cg adds benchmark_cg_pilot's own settings.
inline void print_settings(const PilotOptions& opt, long n, bool cg = false)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    if (rank != 0) return;
    printf("------------------------------------------------------------\n");
    printf("-MPI: %s\n", opt.arm == "MPI" ? "MPI" : "MPIADVANCE");
    printf("-File: %s\n", opt.file.c_str());
    printf("-patterns: %s\n", opt.name.c_str());
    printf("-warmup: %s\n", opt.warmup.c_str());
    printf("-window (k): %d\n", opt.k);
    printf("-timed iterations (n): %ld\n", n);
    if (opt.arm != "MPI")
    {
        printf("-alltoallv: NEIGHBOR_ALLTOALLV_%s%s\n", opt.persistent ? "INIT_" : "",
               opt.arm.c_str());
        printf("-persistent: %s\n", opt.persistent ? "true" : "false");
    }
    if (!opt.split.empty()) printf("-split: %s\n", opt.split.c_str());
    if (cg)
    {
        printf("-max iter: %d\n", opt.max_iter);
        printf("-ignore convergence: %s\n", opt.ignore_convergence ? "true" : "false");
    }
    if (!opt.iter_prefix.empty())
    {
        printf("-iter prefix: %s\n", opt.iter_prefix.c_str());
        printf("-iter raw: %s\n", opt.iter_raw ? "true" : "false");
        printf("-iter compressed: %s\n", opt.iter_compressed ? "true" : "false");
    }
    printf("-ranks: %d\n", num_procs);
    // The timer is MPI_Wtime, CLOCK_MONOTONIC in Open MPI 4.1 on Linux.
    timespec monotonic_res{};
    clock_getres(CLOCK_MONOTONIC, &monotonic_res);
    printf("-MPI_Wtick: %g\n", MPI_Wtick());
    printf("-CLOCK_MONOTONIC resolution (ns): %ld\n",
           monotonic_res.tv_sec * 1000000000L + monotonic_res.tv_nsec);
    printf("-source commit: %s\n", ABCG_SOURCE_COMMIT);
    printf("-locality_aware commit: %s\n", LOCALITY_AWARE_COMMIT);
    printf("------------------------------------------------------------\n");
}

// One "(name)  label  min  max  average" row per value, reduced over ranks.
inline void print_rows(const std::string& name, int count, const char* labels[], double* local)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    std::vector<double> mins(count), maxs(count), sums(count);
    MPI_Reduce(local, mins.data(), count, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(local, maxs.data(), count, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(local, sums.data(), count, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank != 0) return;
    printf("%-20s %-12s %-12s %-12s\n", "Metric", "Min", "Max", "Average");
    for (int i = 0; i < count; i++)
    {
        printf("(%s)  %-20s %-.6f     %-.6f     %-.6f\n",
               name.c_str(), labels[i], mins[i], maxs[i], sums[i] / num_procs);
    }
}

// Solves window = startup + q*k and timed = q*n from the max-over-ranks totals (ms),
// and prints the n for which startup + q*(k + n) fills the budget, in the line format
// pilot.sbatch turns into targets.
inline void print_calc(const PilotOptions& opt, double window_ms, double timed_ms, long n)
{
    double window_max, timed_max;
    MPI_Reduce(&window_ms, &window_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&timed_ms, &timed_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank != 0) return;
    double q = timed_max / n;
    double startup = window_max - opt.k * q;
    double n_budget = std::floor((opt.calc_s * 1000.0 - startup) / q - opt.k);
    printf("(%s)  calc: q %.6f ms  startup %.6f ms  n %.0f  (budget %g s, k %d)\n",
           opt.name.c_str(), q, startup, n_budget, opt.calc_s, opt.k);
}

// Per-iteration times (--iter-prefix), kept in memory during the run and written after
// it, in irregular-benchmarking's format. start_s is wall-clock seconds since the epoch
// (system_clock), comparable across nodes and jobs up to the nodes' clock offsets; the
// durations come from the program's own timer. benchmark_spmv records apply_ms (stopped
// before the barrier) and iter_ms (barrier to barrier); benchmark_cg_pilot has no
// barrier, and its only synchronizations are CG's own allreduces, so it records iter_ms
// alone.
inline double wall_seconds()
{
    return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}

// Each rank's wall-clock (system_clock) offset from rank 0, so start_s can be compared
// across nodes (irregular-benchmarking's scripts/barrier-skew.py). Rank 0 ping-pongs each
// rank in turn; from the round with the smallest round trip, offset = reply time minus
// the midpoint of rank 0's send and receive, good to +/- rtt/2. Collective; rank 0 gets
// (t, offset, rtt) per rank, t being that midpoint in its own clock; the others get
// nothing.
inline std::vector<double> measure_clock_offsets()
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    const int rounds = 50, tag = 4242;
    std::vector<double> table;
    if (rank == 0)
    {
        table.assign(3 * num_procs, 0.0);
        table[0] = wall_seconds();
        for (int r = 1; r < num_procs; r++)
        {
            double best_rtt = 1e30;
            for (int k = 0; k < rounds; k++)
            {
                double reply, ping = 0.0;
                double t0 = wall_seconds();
                MPI_Send(&ping, 1, MPI_DOUBLE, r, tag, MPI_COMM_WORLD);
                MPI_Recv(&reply, 1, MPI_DOUBLE, r, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                double t1 = wall_seconds();
                if (t1 - t0 < best_rtt)
                {
                    best_rtt = t1 - t0;
                    table[3 * r] = 0.5 * (t0 + t1);
                    table[3 * r + 1] = reply - 0.5 * (t0 + t1);
                    table[3 * r + 2] = best_rtt;
                }
            }
        }
    }
    else
    {
        for (int k = 0; k < rounds; k++)
        {
            double ping, reply;
            MPI_Recv(&ping, 1, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            reply = wall_seconds();
            MPI_Send(&reply, 1, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD);
        }
    }
    return table;
}

struct IterLog
{
    bool on = false;
    bool has_apply = false;
    std::vector<double> start_s;
    std::vector<float> apply_ms;
    std::vector<float> iter_ms;
    // --iter-raw only: measure_clock_offsets() before and after the loop (rank 0).
    std::vector<double> offsets_before, offsets_after;

    void reserve(long count)
    {
        start_s.reserve(count);
        apply_ms.reserve(has_apply ? count : 0);
        iter_ms.reserve(count);
    }
    void record(double start, double apply, double iter)
    {
        start_s.push_back(start);
        if (has_apply) apply_ms.push_back((float)apply);
        iter_ms.push_back((float)iter);
    }
};

// A numpy .npy (format 1.0) header for packed little-endian records with these fields.
inline std::string npy_header(const std::vector<std::pair<std::string, std::string>>& fields,
                              const std::string& shape)
{
    std::string dict = "{'descr': [";
    for (size_t i = 0; i < fields.size(); i++)
        dict += (i ? ", ('" : "('") + fields[i].first + "', '" + fields[i].second + "')";
    dict += "], 'fortran_order': False, 'shape': " + shape + ", }";
    // magic, version and length take 10 bytes; the header ends with a newline at a
    // multiple of 64.
    dict.append((64 - (10 + dict.size() + 1) % 64) % 64, ' ');
    dict += '\n';
    std::string header("\x93NUMPY\x01\x00", 8);
    header += (char)(dict.size() & 0xff);
    header += (char)(dict.size() >> 8);
    return header + dict;
}

template <class T>
inline void append_bytes(std::string& out, T value)
{
    out.append(reinterpret_cast<const char*>(&value), sizeof(T));
}

// One gzip member; members written one after another read back as one gzip stream.
inline std::string gzip_bytes(const std::string& in)
{
    z_stream zs;
    memset(&zs, 0, sizeof(zs));
    deflateInit2(&zs, Z_DEFAULT_COMPRESSION, Z_DEFLATED, 15 + 16, 8, Z_DEFAULT_STRATEGY);
    std::string out(deflateBound(&zs, in.size()), '\0');
    zs.next_in = (Bytef*)in.data();
    zs.avail_in = in.size();
    zs.next_out = (Bytef*)&out[0];
    zs.avail_out = out.size();
    int status = deflate(&zs, Z_FINISH);
    out.resize(zs.total_out);
    deflateEnd(&zs);
    if (status != Z_STREAM_END)
    {
        printf("iter file: gzip failed (zlib status %d)\n", status);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return out;
}

// Writes PREFIX_NAME_sSAMPLE.iter.npy[.gz]: one row per iteration (window rows first),
// reduced over ranks, from rank 0. With raw, also PREFIX_NAME_sSAMPLE.raw.npy[.gz]:
// every rank's rows, shape (ranks, iterations), each rank's block (gzip member when
// compressed) sent to rank 0 in rank order, and the clock offsets. Collective.
inline void write_iter_files(const std::string& prefix, const std::string& name, int sample,
                             const IterLog& log, bool raw, bool compressed)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    const int count = (int)log.iter_ms.size();
    const std::string base = prefix + "_" + name + "_s" + std::to_string(sample);
    const std::string ext = compressed ? ".npy.gz" : ".npy";

    // slowest_rank has the largest apply (SpMV) or iteration (CG) time. iter_min_rank has
    // the smallest iteration time: a rank delayed outside its timed work (between
    // iterations, before its timer starts) arrives last and waits least, so it is the one
    // the others waited for even though its own times look normal.
    const std::vector<float>& primary = log.has_apply ? log.apply_ms : log.iter_ms;
    struct FloatInt { float value; int rank; };
    std::vector<FloatInt> local_primary(count), max_loc(count), local_iter(count), min_loc(count);
    std::vector<double> local_sum(primary.begin(), primary.end()), sum(count);
    std::vector<float> apply_min(count), iter_max(count);
    for (int i = 0; i < count; i++)
    {
        local_primary[i] = {primary[i], rank};
        local_iter[i] = {log.iter_ms[i], rank};
    }
    MPI_Reduce(local_primary.data(), max_loc.data(), count, MPI_FLOAT_INT, MPI_MAXLOC, 0, MPI_COMM_WORLD);
    MPI_Reduce(local_iter.data(), min_loc.data(), count, MPI_FLOAT_INT, MPI_MINLOC, 0, MPI_COMM_WORLD);
    MPI_Reduce(local_sum.data(), sum.data(), count, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (log.has_apply)
    {
        MPI_Reduce(log.apply_ms.data(), apply_min.data(), count, MPI_FLOAT, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(log.iter_ms.data(), iter_max.data(), count, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
    }

    if (rank == 0)
    {
        std::vector<std::pair<std::string, std::string>> fields = {{"start_s", "<f8"}};
        if (log.has_apply)
            fields.insert(fields.end(), {{"apply_max_ms", "<f4"}, {"apply_min_ms", "<f4"},
                                         {"apply_mean_ms", "<f4"}, {"slowest_rank", "<i4"},
                                         {"iter_max_ms", "<f4"}, {"iter_min_ms", "<f4"},
                                         {"iter_min_rank", "<i4"}});
        else
            fields.insert(fields.end(), {{"iter_max_ms", "<f4"}, {"iter_min_ms", "<f4"},
                                         {"iter_mean_ms", "<f4"}, {"slowest_rank", "<i4"},
                                         {"iter_min_rank", "<i4"}});
        std::string out = npy_header(fields, "(" + std::to_string(count) + ",)");
        for (int i = 0; i < count; i++)
        {
            append_bytes(out, log.start_s[i]);
            append_bytes(out, max_loc[i].value);
            append_bytes(out, log.has_apply ? apply_min[i] : min_loc[i].value);
            append_bytes(out, (float)(sum[i] / num_procs));
            append_bytes(out, (int32_t)max_loc[i].rank);
            if (log.has_apply)
            {
                append_bytes(out, iter_max[i]);
                append_bytes(out, min_loc[i].value);
            }
            append_bytes(out, (int32_t)min_loc[i].rank);
        }
        if (compressed) out = gzip_bytes(out);
        const std::string path = base + ".iter" + ext;
        FILE* file = fopen(path.c_str(), "wb");
        if (!file || fwrite(out.data(), 1, out.size(), file) != out.size() || fclose(file) != 0)
        {
            printf("iter file: cannot write %s\n", path.c_str());
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    if (!raw) return;
    std::vector<std::pair<std::string, std::string>> fields = {{"start_s", "<f8"}};
    if (log.has_apply) fields.push_back({"apply_ms", "<f4"});
    fields.push_back({"iter_ms", "<f4"});
    const std::string header =
        npy_header(fields, "(" + std::to_string(num_procs) + ", " + std::to_string(count) + ")");
    std::string block = rank == 0 ? header : "";
    for (int i = 0; i < count; i++)
    {
        append_bytes(block, log.start_s[i]);
        if (log.has_apply) append_bytes(block, log.apply_ms[i]);
        append_bytes(block, log.iter_ms[i]);
    }
    if (compressed) block = gzip_bytes(block);
    // Rank 0 writes the file, taking each rank's block in rank order. Collective MPI-IO
    // (MPI_File_write_at_all) took about 10 minutes for 14-26 MB on dane's NFS workspace,
    // from its file locking; this is the same plain write the reduced file uses.
    const int tag = 4243;
    const std::string path = base + ".raw" + ext;
    if (rank != 0)
    {
        long long size = block.size();
        MPI_Send(&size, 1, MPI_LONG_LONG, 0, tag, MPI_COMM_WORLD);
        MPI_Send(block.data(), (int)block.size(), MPI_BYTE, 0, tag, MPI_COMM_WORLD);
    }
    else
    {
        FILE* file = fopen(path.c_str(), "wb");
        bool ok = file && fwrite(block.data(), 1, block.size(), file) == block.size();
        for (int r = 1; r < num_procs; r++)
        {
            long long size;
            MPI_Recv(&size, 1, MPI_LONG_LONG, r, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            block.resize(size);
            MPI_Recv(&block[0], (int)size, MPI_BYTE, r, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            ok = ok && fwrite(block.data(), 1, block.size(), file) == block.size();
        }
        if (!ok || fclose(file) != 0)
        {
            printf("iter file: cannot write %s\n", path.c_str());
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // With raw: PREFIX_NAME_sSAMPLE.offsets.npy[.gz], one row per rank, the clock offsets
    // measured before and after the loop.
    if (rank != 0) return;
    std::string out = npy_header({{"t_before_s", "<f8"}, {"offset_before_s", "<f8"},
                                  {"rtt_before_s", "<f8"}, {"t_after_s", "<f8"},
                                  {"offset_after_s", "<f8"}, {"rtt_after_s", "<f8"}},
                                 "(" + std::to_string(num_procs) + ",)");
    for (int r = 0; r < num_procs; r++)
    {
        for (int c = 0; c < 3; c++) append_bytes(out, log.offsets_before[3 * r + c]);
        for (int c = 0; c < 3; c++) append_bytes(out, log.offsets_after[3 * r + c]);
    }
    if (compressed) out = gzip_bytes(out);
    const std::string offsets_path = base + ".offsets" + ext;
    FILE* offsets_file = fopen(offsets_path.c_str(), "wb");
    if (!offsets_file || fwrite(out.data(), 1, out.size(), offsets_file) != out.size() ||
        fclose(offsets_file) != 0)
    {
        printf("iter file: cannot write %s\n", offsets_path.c_str());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

#endif
