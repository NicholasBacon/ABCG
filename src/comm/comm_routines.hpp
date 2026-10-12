#ifndef COMM_ROUTINES_HPP
#define COMM_ROUTINES_HPP

#include "spmv_comm.hpp"
#include "spmv_comm_pers.hpp"
#include "allreduce_comm.hpp"
#include "allreduce_comm_pers.hpp"

#ifdef MPIL
#include "spmv_comm_mpil.hpp"
#include "spmv_comm_mpil_pers.hpp"
#include "allreduce_comm_mpil.hpp"
#ifndef MPIL_OLD_API  // no MPIL_Allreduce_init in the old API
#include "allreduce_comm_mpil_pers.hpp"
#endif
#endif

#endif
