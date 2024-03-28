/**
 * sweep_mpi_csl.cpp
 *
 * Sweep MPI OWA logistic regression with CSL updates across a range of
 * lambda values, storing the train/test accuracies in a CSV file.
 */
#include <mpi.h>
#include "mpi_csl.hpp"
#include "../libsvm.hpp"
#include "../data_utils.hpp"
#include <iostream>

using namespace std;

void help(char** argv)
{
  cout << "Usage: " << argv[0] << " <input_data_basename> <test_data_basename> "
       << "<extension> data_dim output_file.csv seed min_reg max_reg count [start "
       << " [verbose]]" << endl
       << endl
       << " - note: first two arguments should be basename of data;"
       << endl
       << "     data should be stored as input_data.partition.ext"
       << endl
       << "     for some extension ext (svm/csv/etc.)"
       << endl
       << " - note: data dimensionality must be provided"
       << endl
       << " - note: lambda values are between 10^{min_reg} and 10^{max_reg}"
       << endl
       << " - if start is given, the grid will start from that index"
       << endl
       << " - verbose output is given if *any* argument is given"
       << endl
       << " - number of partitions is set by number of MPI workers"
       << endl;
}

int main(int argc, char** argv)
{
  // Make sure we got the right number of arguments.
  if (argc != 10 && argc != 11 && argc != 12)
  {
    help(argv);
    exit(1);
  }

  const std::string inputFilePrefix(argv[1]);
  const std::string testFilePrefix(argv[2]);
  const std::string extension(argv[3]);
  const size_t dataDim = atoi(argv[4]);
  const std::string outputFile(argv[5]);
  const size_t seed = atoi(argv[6]);
  const double minReg = atof(argv[7]);
  const double maxReg = atof(argv[8]);
  const size_t count = atoi(argv[9]);
  const size_t start = (argc == 11) ? atoi(argv[10]) : 0;
  const bool verbose = (argc == 12);

  // We don't load the data here but instead let each MPI worker do that.
  srand(seed);
  arma::arma_rng::set_seed(seed);

  MPI_Init(NULL, NULL);
  int worker, worldSize;
  MPI_Comm_rank(MPI_COMM_WORLD, &worker);
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

  fstream f;
  if (worker == 0)
  {
    f = fstream(outputFile, fstream::out | fstream::ate);
    if (!f.is_open())
    {
      std::cerr << "Failed to open output file '" << outputFile << "'!"
          << std::endl;
      exit(1);
    }

    if (start == 0)
    {
      f << "method,index,lambda,lambda_pow,nnz,train_acc,test_acc,train_time" << endl;
    }
  }

  arma::wall_clock c;
  c.tic();
  const double step = (maxReg - minReg) / (count - 1);
  double pow = maxReg - (step * start);
  double lambda = std::pow(10.0, pow);

  // This creates the LIBLINEAR features.
  MPICSL owa(inputFilePrefix, testFilePrefix, extension, dataDim, lambda,
      verbose);
  for (size_t i = start + 1; i < count; ++i)
  {
    // Make sure no workers are hung up on something, so that our timing is
    // accurate.
    MPI_Barrier(MPI_COMM_WORLD);

    pow = maxReg - (step * i);
    lambda = std::pow(10.0, pow);

    // Step 0: Train OWA
    arma::wall_clock c;
    c.tic();
    owa.lambda = lambda;
    owa.TrainNaive();

    MPI_Barrier(MPI_COMM_WORLD);

    // Step 1: CSL Update 1
    owa.Update();

    MPI_Barrier(MPI_COMM_WORLD);

    // Step 2: CSL Update 2
    owa.Update();
    const double trainTime = c.toc();

    MPI_Barrier(MPI_COMM_WORLD);

    // Distribute models to all workers for accurate train/test accuracy
    // calculations.
    owa.DistributeModel();

    const double trainAcc2 = owa.TrainAccuracy();
    const double testAcc2 = owa.TestAccuracy();
    const size_t nonzeros2 = arma::accu(owa.model != 0.0);

    if (worker == 0)
    {
      f << "csl-" << worldSize << "," << i << "," << lambda << ","
          << pow << "," << nonzeros2 << ","
          << trainAcc2 << "," << testAcc2 << "," << trainTime << endl;
      cout << "CSL, " << worldSize << " partitions, lambda 10^"
          << pow << ": " << trainTime << "s training time; "
          << nonzeros2 << " nonzeros; "
          << trainAcc2 << " training accuracy; "
          << testAcc2 << " testing accuracy." << endl;
    }
  }

  MPI_Finalize();
}
