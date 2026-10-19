cmake_minimum_required(VERSION 3.24)

project(ABCG CXX)

# Text color settings taken from here
# https://stackoverflow.com/questions/18968979/how-to-make-colorized-message-with-cmake
string(ASCII 27 Esc)
set(ColorReset "${Esc}[m")
set(BoldRed    "${Esc}[1;31m")
set(BoldYellow "${Esc}[1;33m")
set(BoldBlue   "${Esc}[1;34m")

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Default CMake build type if not specified (release includes -O3)
if(NOT CMAKE_BUILD_TYPE)
        message(STATUS "No ${BoldBlue}CMAKE_BUILD_TYPE${ColorReset} specified, setting build type to ${BoldYellow}RELEASE${ColorReset}.")
        set(CMAKE_BUILD_TYPE RELEASE CACHE STRING "Build type for project." FORCE)
endif()

# Find necessary packages
find_package(MPI REQUIRED)

option(USE_LOCALITY "Use MPI Advance Locality-Aware" OFF)
option(MPIL_OLD_API "MPI Advance is an older locality_aware (e.g. 752a19b): alogorithm setters, no MPIL_COMM_WORLD, no MPIL_Allreduce" OFF)
option(ABCG_VERNIER "Build benchmark_spmv with a Vernier pattern, named by -P (default the matrix), around each halo exchange; captures the MPI arm only" OFF)
if (ABCG_VERNIER)
    # Vernier records every MPI call in the process, so with MPI Advance linked it would
    # capture the library's internal traffic instead of ABCG's own exchange.
    if (USE_LOCALITY)
        message(FATAL_ERROR "ABCG_VERNIER captures the MPI arm only; build it without USE_LOCALITY")
    endif()
    find_package(vernier REQUIRED)
endif()

if (USE_LOCALITY)
    # Older locality_aware installs (e.g. 752a19b) link NUMA::numa in their exported
    # targets without defining it; define it as irregular-benchmarking does.
    find_library(NUMA_LIBRARY numa)
    if (NUMA_LIBRARY AND NOT TARGET NUMA::numa)
        add_library(NUMA::numa UNKNOWN IMPORTED)
        set_target_properties(NUMA::numa PROPERTIES IMPORTED_LOCATION "${NUMA_LIBRARY}")
    endif()
    find_package(locality_aware REQUIRED)
    message(STATUS "Building ${BoldBlue}with${ColorReset} MPI Advance")
else()
    message(STATUS "Building ${BoldRed}without${ColorReset} MPI Advance")
endif()

add_library(abcg src/cg.cpp)
target_include_directories(abcg PUBLIC src)
target_link_libraries(abcg PUBLIC MPI::MPI_CXX)

if (USE_LOCALITY)
    target_compile_definitions(abcg PUBLIC MPIL)
    if (MPIL_OLD_API)
        target_compile_definitions(abcg PUBLIC MPIL_OLD_API)
    endif()
    target_link_libraries(abcg PUBLIC MPIAdvance::locality_aware)
endif()

# The commits the binaries are built from, compiled in as one string that strings(1)
# finds (irregular-benchmarking's scripts/provenance.sh) and printed by the pilot
# programs. irregular-benchmarking's scripts/dane/build_abcg_pilots.sh passes both.
# Otherwise SOURCE_COMMIT is read from git, at configure time only, and only when the
# source is the top of its own checkout: a git-archive export inside another repository
# (a home directory that is a repo) must not report that repository's commit.
set(SOURCE_COMMIT "" CACHE STRING "Commit this source is; empty: read from git at configure time")
set(LOCALITY_AWARE_COMMIT "" CACHE STRING "locality_aware commit linked; empty: unknown, or none without USE_LOCALITY")
set(resolved_commit "${SOURCE_COMMIT}")
if (resolved_commit STREQUAL "")
    set(resolved_commit "unknown (not a git checkout)")
    find_package(Git QUIET)
    if (GIT_FOUND)
        execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse --show-toplevel
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR} RESULT_VARIABLE git_status
                OUTPUT_VARIABLE git_top OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
        get_filename_component(git_top "${git_top}" REALPATH)
        get_filename_component(source_dir "${PROJECT_SOURCE_DIR}" REALPATH)
        if (git_status EQUAL 0 AND git_top STREQUAL source_dir)
            execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
                    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                    OUTPUT_VARIABLE resolved_commit OUTPUT_STRIP_TRAILING_WHITESPACE)
            execute_process(COMMAND ${GIT_EXECUTABLE} status --porcelain --untracked-files=no
                    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR} OUTPUT_VARIABLE git_changes)
            if (NOT git_changes STREQUAL "")
                string(APPEND resolved_commit "-dirty")
            endif()
        endif()
    endif()
endif()
set(resolved_la_commit "${LOCALITY_AWARE_COMMIT}")
if (resolved_la_commit STREQUAL "" AND USE_LOCALITY)
    set(resolved_la_commit "unknown")
elseif (resolved_la_commit STREQUAL "")
    set(resolved_la_commit "none")
endif()
message(STATUS "Provenance: source ${resolved_commit}, locality_aware ${resolved_la_commit}")
target_compile_definitions(abcg PUBLIC
        "ABCG_SOURCE_COMMIT=\"${resolved_commit}\""
        "LOCALITY_AWARE_COMMIT=\"${resolved_la_commit}\"")

add_subdirectory(benchmarks)

