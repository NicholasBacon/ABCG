add_executable(benchmark_cg_mpi benchmark_cg_mpi.cpp)
target_link_libraries(benchmark_cg_mpi abcg)

# benchmark_cg_mpil drives the new API's allreduce algorithms, which MPIL_OLD_API lacks.
if (USE_LOCALITY AND NOT MPIL_OLD_API)
    add_executable(benchmark_cg_mpil benchmark_cg_mpil.cpp)
    target_link_libraries(benchmark_cg_mpil abcg)
endif()

# Halo exchange only, and CG iteration timing, both with a --calc pilot mode.
add_executable(benchmark_spmv benchmark_spmv.cpp)
target_link_libraries(benchmark_spmv abcg)
if (ABCG_VERNIER)
    target_compile_definitions(benchmark_spmv PRIVATE ABCG_VERNIER)
    target_link_libraries(benchmark_spmv vernier::vernier)
endif()
add_executable(benchmark_cg_pilot benchmark_cg_pilot.cpp)
target_link_libraries(benchmark_cg_pilot abcg)
