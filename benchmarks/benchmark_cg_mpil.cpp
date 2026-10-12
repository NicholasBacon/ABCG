#include "cg.hpp"
#include "timer.hpp"

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    mpil_init();
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Init matrix from file
    std::string filename = argv[1];
    ParMat A;
    readParMatrix(filename.c_str(), A);
    form_comm(A);

    // Create linear system
    std::vector<double> x(A.local_cols);
    std::vector<double> b(A.local_rows);
    std::vector<double> r(A.local_rows);      
    std::vector<double> sendbuf(A.send_comm.size_msgs);
    std::vector<double> recvbuf(A.recv_comm.size_msgs);
    double local_sum, global_sum;

    SpMVComm* spmv_comm = new SpMVComm();
    AllreduceComm* allreduce_comm = new AllreduceComm();
    spmv_comm->init(A, sendbuf.data(), recvbuf.data());
    allreduce_comm->init(&local_sum, &global_sum);

    // Set x to random values, b = A*x 
    srand(time(NULL) + rank);
    std::generate(x.begin(), x.end(), [&]() { return (double)(rand()) / RAND_MAX; });
    spmv(1.0, A, x, 0.0, b, spmv_comm);
    double norm_b = sqrt(inner_product(b, b, allreduce_comm));
    double norm_r;

    // Declare book keeping variables
    int n_iter, n_tests;
    double time;

    // MPIL Setters
    std::vector<std::pair<enum NeighborAlltoallvMethod, std::string>> spmv_configs = {
        {NEIGHBOR_ALLTOALLV_STANDARD, "Standard"}, 
        {NEIGHBOR_ALLTOALLV_LOCALITY, "Locality"}
    };
    std::vector<std::pair<enum NeighborAlltoallvInitMethod, std::string>> pers_spmv_configs = {
        {NEIGHBOR_ALLTOALLV_INIT_STANDARD, "Standard"},
        {NEIGHBOR_ALLTOALLV_INIT_LOCALITY, "Locality"}
    };
    std::vector<std::pair<enum AllreduceMethod, std::string>> allreduce_configs = {
        {ALLREDUCE_PMPI, "PMPI"},
        {ALLREDUCE_RECURSIVE_DOUBLING, "Recursive Doubling"},
        {ALLREDUCE_DISSEMINATION_LOC, "Dissemination + Locality"},
        {ALLREDUCE_DISSEMINATION_ML, "Dissemination + ML"},
        {ALLREDUCE_DISSEMINATION_RADIX, "High-Radix"}
    };
    std::vector<std::pair<enum AllreduceInitMethod, std::string>> pers_allreduce_configs = {
        {ALLREDUCE_INIT_RECURSIVE_DOUBLING, "Recursive Doubling"},
        {ALLREDUCE_INIT_DISSEMINATION_LOC,"Dissemination + Locality"},
        {ALLREDUCE_INIT_DISSEMINATION_ML, "Dissemination + ML"},
        {ALLREDUCE_INIT_DISSEMINATION_RADIX, "High-Radix"}
    };

    /*********************************************************
     * Test CG with Standard MPIL
     ********************************************************/
    if (rank == 0)
        printf("Testing CG with Standard MPIL:\n");

    for (size_t i = 0; i < spmv_configs.size(); i++)
    {
        MPIL_Set_alltoallv_neighbor_algorithm(spmv_configs[i].first);
        for (size_t j = 0; j < allreduce_configs.size(); j++)
        {
            MPIL_Set_allreduce_algorithm(allreduce_configs[j].first);        
            if (rank == 0)
                printf("Testing with SpMV %s and Allreduce %s\n", 
                        spmv_configs[i].second.c_str(), 
                        allreduce_configs[j].second.c_str());

            // Set x to zero, test CG
            std::fill(x.begin(), x.end(), 0);
            n_iter = CG<SpMVCommMPIL, AllreduceCommMPIL>(A, x, b);
            r = b;
            spmv(1.0, A, x, -1.0, r, spmv_comm);
            norm_r = sqrt(inner_product(r, r, allreduce_comm));
            if (rank == 0) 
                printf("After %d iterations, final res is %e\n", 
                        n_iter, norm_r / norm_b);
            n_tests = calc_iters<SpMVCommMPIL, AllreduceCommMPIL>(A, x, b);
            time = time_function<SpMVCommMPIL, AllreduceCommMPIL>(n_tests, A, x, b);
            if (rank == 0) 
                printf("CG takes %e seconds\n", time);
        }
    }

    /*********************************************************
     * Test CG with Persistent SpMVs
     ********************************************************/
    if (rank == 0) 
        printf("Testing CG with Persistent SpMVs:\n");

    for (size_t i = 0; i < pers_spmv_configs.size(); i++)
    {
        MPIL_Set_alltoallv_neighbor_init_algorithm(pers_spmv_configs[i].first);
        for (size_t j = 0; j < allreduce_configs.size(); j++)
        {
            MPIL_Set_allreduce_algorithm(allreduce_configs[j].first);        
            if (rank == 0) 
                printf("Testing with SpMV %s and Allreduce %s\n", 
                        pers_spmv_configs[i].second.c_str(), 
                        allreduce_configs[j].second.c_str());

            // Set x to zero, test CG
            std::fill(x.begin(), x.end(), 0);
            n_iter = CG<SpMVCommMPILPers, AllreduceCommMPIL>(A, x, b);
            r = b;
            spmv(1.0, A, x, -1.0, r, spmv_comm);
            norm_r = sqrt(inner_product(r, r, allreduce_comm));
            if (rank == 0) 
                printf("After %d iterations, final res is %e\n", 
                        n_iter, norm_r / norm_b);

            n_tests = calc_iters<SpMVCommMPILPers, AllreduceCommMPIL>(A, x, b);
            time = time_function<SpMVCommMPILPers, AllreduceCommMPIL>(n_tests, A, x, b);
            if (rank == 0) 
                printf("CG takes %e seconds\n", time);
        }
    }


    /*********************************************************
     * Test CG with Persistent Allreduce
     ********************************************************/
    if (rank == 0) 
        printf("Testing CG with Persistent Allreduce:\n");

    for (size_t i = 0; i < spmv_configs.size(); i++)
    {
        MPIL_Set_alltoallv_neighbor_algorithm(spmv_configs[i].first);
        for (size_t j = 0; j < pers_allreduce_configs.size(); j++)
        {
            MPIL_Set_allreduce_init_algorithm(pers_allreduce_configs[j].first);        
            if (rank == 0) 
                printf("Testing with SpMV %s and Allreduce %s\n", 
                        spmv_configs[i].second.c_str(), 
                        pers_allreduce_configs[j].second.c_str());

            // Set x to zero, test CG
            std::fill(x.begin(), x.end(), 0);
            n_iter = CG<SpMVCommMPIL, AllreduceCommMPILPers>(A, x, b);
            r = b;
            spmv(1.0, A, x, -1.0, r, spmv_comm);
            norm_r = sqrt(inner_product(r, r, allreduce_comm));
            if (rank == 0) 
                printf("After %d iterations, final res is %e\n", 
                        n_iter, norm_r / norm_b);

            n_tests = calc_iters<SpMVCommMPIL, AllreduceCommMPILPers>(A, x, b);
            time = time_function<SpMVCommMPIL, AllreduceCommMPILPers>(n_tests, A, x, b);
            if (rank == 0) 
                printf("CG takes %e seconds\n", time);
        }
    }

    /*********************************************************
     * Test CG with Persistent SpMVs and Allreduce
     ********************************************************/
    if (rank == 0) 
        printf("Testing CG with Persistent SpMVs and Allreduce:\n");

    for (size_t i = 0; i < pers_spmv_configs.size(); i++)
    {
        MPIL_Set_alltoallv_neighbor_init_algorithm(pers_spmv_configs[i].first);
        for (size_t j = 0; j < pers_allreduce_configs.size(); j++)
        {
            MPIL_Set_allreduce_init_algorithm(pers_allreduce_configs[j].first);        
            if (rank == 0) 
                printf("Testing with SpMV %s and Allreduce %s\n", 
                        pers_spmv_configs[i].second.c_str(), 
                        pers_allreduce_configs[j].second.c_str());

            // Set x to zero, test CG
            std::fill(x.begin(), x.end(), 0);
            n_iter = CG<SpMVCommMPILPers, AllreduceCommMPILPers>(A, x, b);
            r = b;
            spmv(1.0, A, x, -1.0, r, spmv_comm);
            norm_r = sqrt(inner_product(r, r, allreduce_comm));
            if (rank == 0) 
                printf("After %d iterations, final res is %e\n", n_iter, norm_r / norm_b);

            n_tests = calc_iters<SpMVCommMPILPers, AllreduceCommMPILPers>(A, x, b);
            time = time_function<SpMVCommMPILPers, AllreduceCommMPILPers>(n_tests, A, x, b);
            if (rank == 0) 
                printf("CG takes %e seconds\n", time);
        }
    }


    delete spmv_comm;
    delete allreduce_comm;

    mpil_finalize();
    MPI_Finalize();

    return 0;
}


