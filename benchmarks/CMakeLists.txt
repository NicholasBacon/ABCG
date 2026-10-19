# benchmark_cg_mpi uses AllreduceCommPers, which needs MPI_Allreduce_init (MPI 4, or Open
# MPI's MPIX_Allreduce_init); MPICH 3-based MPIs such as MVAPICH2 2.3 have neither.
# Same test as src/comm/allreduce_comm_pers.hpp.
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_INCLUDES ${MPI_CXX_INCLUDE_DIRS})
check_cxx_source_compiles("
#include <mpi.h>
#if !(MPI_VERSION >= 4 || defined(OPEN_MPI))
#error no MPI_Allreduce_init
#endif
int main() { return 0; }" ABCG_HAVE_ALLREDUCE_INIT)
unset(CMAKE_REQUIRED_INCLUDES)
if (ABCG_HAVE_ALLREDUCE_INIT)
    add_executable(benchmark_cg_mpi benchmark_cg_mpi.cpp)
    target_link_libraries(benchmark_cg_mpi abcg)
endif()

# benchmark_cg_mpil drives the new API's allreduce algorithms, which MPIL_OLD_API lacks.
if (USE_LOCALITY AND NOT MPIL_OLD_API)
    add_executable(benchmark_cg_mpil benchmark_cg_mpil.cpp)
    target_link_libraries(benchmark_cg_mpil abcg)
endif()

# Halo exchange only, and CG iteration timing, both with a --calc pilot mode. zlib
# compresses their per-iteration files (--iter-prefix).
find_package(ZLIB REQUIRED)
add_executable(benchmark_spmv benchmark_spmv.cpp)
target_link_libraries(benchmark_spmv abcg ZLIB::ZLIB)
if (ABCG_VERNIER)
    target_compile_definitions(benchmark_spmv PRIVATE ABCG_VERNIER)
    target_link_libraries(benchmark_spmv vernier::vernier)
endif()
add_executable(benchmark_cg_pilot benchmark_cg_pilot.cpp)
target_link_libraries(benchmark_cg_pilot abcg ZLIB::ZLIB)
