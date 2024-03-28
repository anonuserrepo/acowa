/**
 * sweep_csl.cpp
 *
 * Sweep distributed lasso + CSL updates
 * across a range of lambda values,
 * storing the train/test accuracies in a CSV file.
 */
#include "global_step.hpp"
#include "libsvm.hpp"
#include "data_utils.hpp"
#include <iostream>

using namespace std;

void help(char** argv)
{
  cout << "Usage: " << argv[0] << " input_data.svm output_file.csv seed "
       << "min_reg max_reg count partitions start_mode" << endl
       << endl
       << " - note: lambda values are between 10^{min_reg} and 10^{max_reg}"
       << endl
       << " - start_mode determines initial solution: 0=zeros, 1=naive, 2=owa"
       << endl;
}

int main(int argc, char** argv)
{
  // Make sure we got the right number of arguments.
  if (argc != 9)
  {
    help(argv);
    exit(1);
  }

  const std::tuple<std::string, std::string> inputFiles =
      SplitDatasetArgument(argv[1]);
  const std::string inputFile(std::get<0>(inputFiles));
  const std::string testFile(std::get<1>(inputFiles));
  const std::string outputFile(argv[2]);
  const size_t seed = atoi(argv[3]);
  const double minReg = atof(argv[4]);
  const double maxReg = atof(argv[5]);
  const size_t count = atoi(argv[6]);
  const size_t partitions = atoi(argv[7]);
  const size_t start_mode = atoi(argv[8]);
  const double alpha = 0;
  const size_t start = 0;
  const bool verbose = false;

  const std::tuple<arma::sp_mat, arma::rowvec> t = load_libsvm<arma::sp_mat>(
      inputFile);

  const arma::sp_mat& data = std::get<0>(t);
  const arma::rowvec& labels = std::get<1>(t);

  srand(seed);
  arma::arma_rng::set_seed(seed);

  // Split into training and test sets.
  arma::sp_mat trainData, testData;
  arma::rowvec trainLabels, testLabels;
  if (testFile.empty())
  {
    TrainTestSplit(data, labels, 0.8, trainData, trainLabels, testData,
        testLabels);
  }
  else
  {
    // Load the test file separately.
    const std::tuple<arma::sp_mat, arma::rowvec> t2 = load_libsvm<arma::sp_mat>(
        testFile);

    trainData = std::move(data);
    trainLabels = std::move(labels);
    testData = std::move(std::get<0>(t2));
    testLabels = std::move(std::get<1>(t2));

    // Ensure matrices have the same dimension.
    const size_t maxDimension = std::max(trainData.n_rows, testData.n_rows);
    trainData.resize(maxDimension, trainData.n_cols);
    testData.resize(maxDimension, testData.n_cols);
  }

  // Check if we will be able to split into partitions easily.  If not, just
  // drop some extra columns...
  const size_t partitionPoints = (trainData.n_cols + partitions - 1) /
      partitions;
  if (partitionPoints * (partitions - 1) >= trainData.n_cols)
  {
    // Just drop the extra points until it divides evenly...
    const size_t evenPartitionPoints = trainData.n_cols / partitions;
    std::cout << "Things don't divide evenly; dropping points "
        << evenPartitionPoints * partitions << " to "
        << trainData.n_cols - 1 << "; this gives ";
    trainData.shed_cols(evenPartitionPoints * partitions,
                        trainData.n_cols - 1);
    std::cout << trainData.n_cols << " points overall." << std::endl;
    trainLabels.shed_cols(evenPartitionPoints * partitions,
                          trainLabels.n_elem - 1);
  }

  fstream f(outputFile, fstream::out | fstream::ate);
  if (!f.is_open())
  {
    std::cerr << "Failed to open output file '" << outputFile << "'!"
        << std::endl;
    exit(1);
  }

  if (start == 0)
  {
    f << "method,index,lambda,lambda_pow,nnz,train_acc,test_acc,time" << endl;
  }

  const size_t totalThreads = omp_get_max_threads();
  std::cout << "Total threads: " << totalThreads << "." << std::endl;

  arma::wall_clock c;
  const double step = (maxReg - minReg) / (count - 1);
  const double pow = maxReg - (step * start);
  double lambda = std::pow(10.0, pow);
  
  GlobalStep<arma::sp_mat> Gs(lambda, partitions);
  Gs.numThreads = totalThreads;
  Gs.verbose = verbose;
  Gs.seed = seed;

  // The first run needs to be done separately to populate the LIBLINEAR
  // features members.

  // Set the initialization for w_0
  if (start_mode == 1)
    Gs.NaiveTrain(trainData, trainLabels);
  else if (start_mode == 2)
    Gs.Train(trainData, trainLabels);
  else if (start_mode == 0)
  {
    Gs.NaiveTrain(trainData, trainLabels);
    for (int j = 0; j < trainData.n_rows; ++j)
      Gs.model[j] = 0;
    Gs.modelNonzeros = arma::accu(Gs.model != 0);
  }

  // Instead of having a 3-level nested parallel run (over lambdas, partitions,
  // and then LIBLINEAR runs), we'll treat lambdas and partitions as OpenMP
  // tasks, allowing us a little more flexibility to assign them.
  for (size_t i = start; i < count; ++i)
  {
    const double powThread = maxReg - (step * i);
    const double lambdaThread = std::pow(10.0, powThread);

    GlobalStep<arma::sp_mat> GsThread(Gs);

    arma::wall_clock cThread;
    cThread.tic();
    GsThread.lambda = lambdaThread;
    GsThread.numThreads = totalThreads;

    // Set the initialization for w_0
    if (start_mode == 1)
      GsThread.NaiveRetrain();
    else if (start_mode == 2)
      GsThread.Retrain();
    else if (start_mode == 0)
    {
      GsThread.NaiveRetrain();
      for (int j = 0; j < trainData.n_rows; ++j)
        GsThread.model[j] = 0;
        GsThread.modelNonzeros = arma::accu(GsThread.model != 0);
    }

    // global update 1
    GsThread.CSLUpdate(-1, alpha);

    // global update 2
    GsThread.CSLUpdate(-1, alpha);
    const double trainTimeThread = cThread.toc();

    const std::tuple<double, double> accsThread2 = TrainTestAccuracy(
        GsThread, trainData, trainLabels, testData, testLabels);
    
    #pragma omp critical
    {
      f << "csl-" << partitions << "," << i << "," << lambdaThread << ","
          << powThread << "," << GsThread.modelNonzeros << ","
          << std::get<0>(accsThread2) << "," << std::get<1>(accsThread2)
          << "," << trainTimeThread << endl;
      cout << "CSL, " << partitions << " partitions, lambda 10^"
          << powThread << ": " << trainTimeThread << "s training time; "
          << GsThread.modelNonzeros << " nonzeros; "
          << std::get<0>(accsThread2) << " training accuracy; "
          << std::get<1>(accsThread2) << " testing accuracy." << endl;
    }
  }
}
