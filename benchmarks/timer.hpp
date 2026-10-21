template <typename SComm, typename AComm>
double time_function(int iter, 
        ParMat& A, 
        std::vector<double>& x, 
        std::vector<double>& b)
{
    // CG() overwrites x, so restore the starting guess before each solve (untimed);
    // otherwise every solve after the first starts from the previous solution.
    std::vector<double> x0 = x;
    MPI_Barrier(MPI_COMM_WORLD);
    double t0, tfinal = 0;
    for (int i = 0; i < iter; i++)
    {
        x = x0;
        t0 = MPI_Wtime();
        CG<SComm, AComm>(A, x, b);
        tfinal += MPI_Wtime() - t0;
    }
    tfinal /= iter;
    MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return t0;
}

template <typename SComm, typename AComm>
int calc_iters(ParMat& A, 
        std::vector<double>& x, 
        std::vector<double>& b)
{
    int iters = 1;
    double time;
    time = time_function<SComm, AComm>(iters, A, x, b);
    if (time > 0.1) 
        return iters;
    else
    {
        iters = 10;
        time = time_function<SComm, AComm>(iters, A, x, b);
        iters = 1.0 / time;
        if (iters < 1)
            iters = 1;
    }

    return iters;
}
