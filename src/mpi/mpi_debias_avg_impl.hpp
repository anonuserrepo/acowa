/**
 * mpi_debias_avg_impl.hpp
 *
 * MPI-based implementation of debiased averaging.
 */
#ifndef MPI_DEBIAS_AVG_IMPL_HPP
#define MPI_DEBIAS_AVG_IMPL_HPP

#include "mpi_debias_avg.hpp"
#include <mpi.h>
#include "../libsvm.hpp"
#include "../data_utils.hpp"

inline MPIDebiasAvg::MPIDebiasAvg(const std::string& trainBaseFilename,
                                const std::string& testBaseFilename,
                                const std::string& extension,
				                const size_t dataDim,
                                const double lambda,
                                const bool verbose,
                                const size_t seed) :
    lambda(lambda),
    verbose(verbose),
    seed(seed)
{
  // Get world information from MPI.
  int sizeIn, rankIn;
  MPI_Comm_size(MPI_COMM_WORLD, &sizeIn);
  MPI_Comm_rank(MPI_COMM_WORLD, &rankIn);
  worldSize = (size_t) sizeIn;
  worldRank = (size_t) rankIn;

  // Load data according to the pattern.
  // By convention we expect `baseFilename.worldRank.extension`.
  std::ostringstream oss;
  oss << trainBaseFilename << "." << worldRank << "." << extension;
  const std::tuple<arma::sp_mat, arma::rowvec> t = load_libsvm<arma::sp_mat>(
      oss.str());

  trainData = std::move(std::get<0>(t));
  trainLabels = std::move(std::get<1>(t));

  oss.str("");
  oss.clear();
  oss << testBaseFilename << "." << worldRank << "." << extension;
  const std::tuple<arma::sp_mat, arma::rowvec> t2 = load_libsvm<arma::sp_mat>(
      oss.str());

  testData = std::move(std::get<0>(t2));
  testLabels = std::move(std::get<1>(t2));

  // Make sure the dimensionalities of the training and test data are correct.
  if (trainData.n_rows != dataDim)
    trainData.resize(dataDim, trainData.n_cols);
  if (testData.n_rows != dataDim)
    testData.resize(dataDim, testData.n_cols);

  // Preprocess data into LIBLINEAR format.  (Use only one partition.)
  liblinearTrainData = liblinear::MatToTransposedFeatureNodes(trainData, 1);

  // generate subsampled training data locally
  subsampleTrainData = liblinear::SubsampleMatToTransposedFeatureNodes(
                                trainData, 1, 0.1);
  size_t n_sub = 0.1 * trainData.n_cols;
  subsampleLabels = trainLabels.head(n_sub);

  // Set model to the correct size.
  model.set_size(trainData.n_rows);
}

inline MPIDebiasAvg::~MPIDebiasAvg()
{
  if (liblinearTrainData)
  {
    CleanFeatureNode(liblinearTrainData[0], trainData.n_rows);
    delete[] liblinearTrainData;
  }
}

inline void MPIDebiasAvg::Train()
{
  // Train the model locally.
  arma::wall_clock c;
  c.tic();
  liblinear::Train((const liblinear::feature_node**) liblinearTrainData[0],
                   trainData.n_cols,
                   trainData.n_rows,
                   trainLabels.memptr(),
                   model.memptr(),
                   lambda,
                   20, // max outer iterations
                   50, // max inner iterations
                   0.001,
                   true, // L1 regularization
                   verbose,
                   seed + worldRank /* different seed for each partition */);

  int n_sub = 0.1 * trainData.n_cols;
  arma::rowvec submodel(model.n_elem);
  liblinear::Train((const liblinear::feature_node**) subsampleTrainData[0],
                   n_sub,
                   trainData.n_rows,
                   subsampleLabels.memptr(),
                   submodel.memptr(),
                   lambda,
                   20, // max outer iterations
                   50, // max inner iterations
                   0.001,
                   true, // L1 regularization
                   verbose,
                   seed + worldRank /* different seed for each partition */);
  const double trainTime = c.toc();
  if (verbose)
  {
    std::cout << "Worker " << worldRank << " took " << trainTime << "s to train"
        << " its local model on " << trainData.n_cols << " points."
        << std::endl;
  }


  // Collect all models on the main node.  `models` is only used on the main
  // node.
  arma::mat models;
  arma::mat submodels;
  if (worldRank == 0)
  {
    models.set_size(model.n_elem, worldSize);
    submodels.set_size(model.n_elem, worldSize);
  }

  // Send the model (if a worker node); receive the model (if the main node).
  MPI_Gather(model.memptr(), model.n_elem, MPI_DOUBLE,
      models.memptr() /* models.col(i) will be model i */, model.n_elem,
      MPI_DOUBLE, 0 /* gather on the main node */, MPI_COMM_WORLD);

  // Send the model (if a worker node); receive the model (if the main node).
  MPI_Gather(submodel.memptr(), submodel.n_elem, MPI_DOUBLE,
      submodels.memptr() /* models.col(i) will be model i */, submodel.n_elem,
      MPI_DOUBLE, 0 /* gather on the main node */, MPI_COMM_WORLD);

  if (worldRank == 0)
  {
    // Now, we have all the models, so we can simply average them.
    model = arma::mean(models, 1).t();
    submodel = arma::mean(submodels, 1).t();

    // compute SAVGM estimate
    model = (model - 0.1 * submodel) / (1 - 0.1);
  }
}

inline double MPIDebiasAvg::TrainAccuracy()
{
  // Compute accuracy on local system.
  const size_t correctCount = CountCorrect(trainData, trainLabels);

  // Gather all accuracies.
  arma::Col<uint64_t> correctCounts(worldSize);
  MPI_Gather(&correctCount, 1, MPI_UINT64_T, correctCounts.memptr(), 1,
      MPI_UINT64_T, 0, MPI_COMM_WORLD);

  // Gather data sizes.
  arma::Col<uint64_t> sizes(worldSize);
  const uint64_t localSize = trainData.n_cols;
  MPI_Gather(&localSize, 1, MPI_UINT64_T, sizes.memptr(), 1, MPI_UINT64_T, 0,
      MPI_COMM_WORLD);

  // Compute overall accuracy.
  const uint64_t totalSize = arma::accu(sizes);
  const uint64_t totalCorrect = arma::accu(correctCounts);
  const double trainAcc = ((double) totalCorrect) / ((double) totalSize);
  return trainAcc;
}

inline double MPIDebiasAvg::TestAccuracy()
{
  // Compute accuracy on local system.
  const size_t correctCount = CountCorrect(testData, testLabels);

  // Gather all accuracies.
  arma::Col<uint64_t> correctCounts(worldSize);
  MPI_Gather(&correctCount, 1, MPI_UINT64_T, correctCounts.memptr(), 1,
      MPI_UINT64_T, 0, MPI_COMM_WORLD);

  // Gather data sizes.
  arma::Col<uint64_t> sizes(worldSize);
  const uint64_t localSize = testData.n_cols;
  MPI_Gather(&localSize, 1, MPI_UINT64_T, sizes.memptr(), 1, MPI_UINT64_T, 0,
      MPI_COMM_WORLD);

  // Compute overall accuracy.
  const uint64_t totalSize = arma::accu(sizes);
  const uint64_t totalCorrect = arma::accu(correctCounts);
  const double testAcc = ((double) totalCorrect) / ((double) totalSize);
  return testAcc;
}

inline size_t MPIDebiasAvg::CountCorrect(const arma::sp_mat& data,
                                        const arma::rowvec& labels) const
{
  const arma::rowvec z(-model * data);
  const arma::rowvec predictions = 2.0 *
      arma::round(1.0 / (1.0 + arma::exp(z))) - 1.0;

  const size_t result = arma::accu(labels == predictions);
  return result;
}

inline void MPIDebiasAvg::DistributeModel()
{
  // Send model from main worker to all other workers.
  // It seems like MVAPICH2 segfaults when we try to send the entire buffer at
  // once, so we break it up into messages of 128 elements (1 kb).
  for (size_t i = 0; i < model.n_elem; i += 128)
  {
    const size_t len = (i + 128 <= model.n_elem) ? 128 : (model.n_elem - i);
    MPI_Bcast(model.memptr() + i,
              len,
              MPI_DOUBLE,
              0,
              MPI_COMM_WORLD);
  }
}

#endif
