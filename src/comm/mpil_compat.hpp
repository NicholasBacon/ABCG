#ifndef MPIL_COMPAT_HPP
#define MPIL_COMPAT_HPP

#include "locality_aware.h"

// locality_aware changed its API. Older versions (e.g. 752a19b, the fast one dane's
// campaigns use) spell the neighbor setters "alogorithm", have no MPIL_COMM_WORLD and
// no MPI Advance allreduce, and have MPIL_Set_split. Build those with -DMPIL_OLD_API=ON.
#ifdef MPIL_OLD_API
// One MPIL_Comm over MPI_COMM_WORLD for the whole run, created on first use.
inline MPIL_Comm* mpil_world()
{
    static MPIL_Comm* world = nullptr;
    if (world == nullptr) MPIL_Comm_init(&world, MPI_COMM_WORLD);
    return world;
}
inline void mpil_set_neighbor(enum NeighborAlltoallvMethod method)
{
    MPIL_Set_alltoallv_neighbor_alogorithm(method);
}
inline void mpil_set_neighbor_init(enum NeighborAlltoallvInitMethod method)
{
    MPIL_Set_alltoallv_neighbor_init_alogorithm(method);
}
inline void mpil_init() {}
inline void mpil_finalize() {}
#else
// The new API's MPIL_COMM_WORLD is null until MPIL_Init, so call mpil_init() after
// MPI_Init and mpil_finalize() before MPI_Finalize.
inline void mpil_init() { MPIL_Init(MPI_COMM_WORLD); }
inline void mpil_finalize() { MPIL_Finalize(); }
inline MPIL_Comm* mpil_world() { return MPIL_COMM_WORLD; }
inline void mpil_set_neighbor(enum NeighborAlltoallvMethod method)
{
    MPIL_Set_alltoallv_neighbor_algorithm(method);
}
inline void mpil_set_neighbor_init(enum NeighborAlltoallvInitMethod method)
{
    MPIL_Set_alltoallv_neighbor_init_algorithm(method);
}
#endif

#endif
