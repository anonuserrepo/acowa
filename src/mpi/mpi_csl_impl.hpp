/**
 * mpi_csl_impl.hpp
 *
 * MPI-based implementation of CSL.
 */
#ifndef MPI_CSL_IMPL_HPP
#define MPI_CSL_IMPL_HPP

#include "mpi_csl.hpp"
#include <mpi.h>
#include "../libsvm.hpp"
#include "../data_utils.hpp"
#include "../filter_dims.hpp"
#include "../csl_solvers.hpp"

inline MPICSL::MPICSL(const std::string& trainBaseFilename,
                      const std::string& testBaseFilename,
                      const std::string& extension,
                      const size_t dataDim,
                      const double lambda,
                      const bool verbose,
                      const size_t seed,
                      const size_t cvFolds,
                      const size_t cvPoints) :
    lambda(lambda),
    verbose(verbose),
    seed(seed),
    cvFolds(cvFolds),
    cvPoints(cvPoints)
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

  if (trainData.n_rows != dataDim)
    trainData.resize(dataDim, trainData.n_cols);
  if (testData.n_rows != dataDim)
    testData.resize(dataDim, testData.n_cols);

  // Preprocess data into LIBLINEAR format.  (Use only one partition.)
  liblinearTrainData = liblinear::MatToTransposedFeatureNodes(trainData, 1);

  // Set model to the correct size.
  model.set_size(trainData.n_rows);
}

inline MPICSL::~MPICSL()
{
  if (liblinearTrainData)
  {
    CleanFeatureNode(liblinearTrainData[0], trainData.n_rows);
    delete[] liblinearTrainData;
  }
}

inline void MPICSL::TrainNaive()
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
  double* modelPointers;
  if (worldRank == 0)
  {
    models.set_size(model.n_elem, worldSize);
  }

  // Send the model (if a worker node); receive the model (if the main node).
  MPI_Gather(model.memptr(), model.n_elem, MPI_DOUBLE,
      models.memptr() /* models.col(i) will be model i */, model.n_elem,
      MPI_DOUBLE, 0 /* gather on the main node */, MPI_COMM_WORLD);

  if (worldRank == 0)
  {
    // Now, we have all the models, so we can simply average them.
    model = arma::mean(models, 1).t();
  }
}

inline void MPICSL::Train()
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
  const double trainTime = c.toc();
  if (verbose)
  {
    std::cout << "Worker " << worldRank << " took " << trainTime << "s to train"
        << " its local model on " << trainData.n_cols << " points."
        << std::endl;
  }

  // Collect all models on the main node.  `models` is only used on the main
  // node.
  c.tic();
  arma::mat models;
  if (worldRank == 0)
  {
    models.set_size(model.n_elem, worldSize);
  }

  // Send the model (if a worker node); receive the model (if the main node).
  MPI_Gather(model.memptr(), model.n_elem, MPI_DOUBLE,
      models.memptr() /* models.col(i) will be model i */, model.n_elem,
      MPI_DOUBLE, 0 /* gather on the main node */, MPI_COMM_WORLD);
  const double modelCollectionTime = c.toc();

  if (worldRank != 0)
  {
    return; // For any workers that are not the main worker, we are done now.
  }

  // Now, we have all the models, so we can do the second round of learning.
  // We simply reuse our partition of the data.
  c.tic();
  const arma::uvec nonzeroDims = arma::find(arma::sum(models, 1));
  arma::mat wHat = models.rows(nonzeroDims);
  arma::sp_mat secondRoundRawData = FilterDims(trainData, nonzeroDims);
  const double filterTime = c.toc();

  // This will be dense.
  c.tic();
  arma::mat secondRoundData = wHat.t() * secondRoundRawData;
  const double secondRoundDataTime = c.toc();

  // Now, we want to do k-fold cross-validation over a grid of possible lambda
  // values.  Because any other workers will be idle, we will use OpenMP.
  c.tic();
  liblinear::feature_node*** secondRoundFeatures =
      liblinear::MatToKFoldFeatureNodes(secondRoundData, cvFolds);
  arma::mat accuracies(cvFolds, cvPoints); // track accuracy of each fold
  const arma::vec cvLambdaPowers = arma::linspace<arma::vec>(-4, 4, cvPoints);
  const double secondRoundConversionTime = c.toc();

  c.tic();
  const size_t totalCVTrials = cvFolds * cvPoints;
  #pragma omp parallel for schedule(dynamic)
  for (size_t cvIndex = 0; cvIndex < totalCVTrials; ++cvIndex)
  {
    const size_t fold = cvIndex / cvPoints;
    const size_t lIndex = cvIndex % cvPoints;

    arma::rowvec cvModel(secondRoundData.n_rows);

    // Compute start and end of held-out set, and its size.
    const size_t batchSize = (secondRoundData.n_cols + cvFolds - 1) / cvFolds;
    const size_t heldOutStart = fold * batchSize;
    const size_t heldOutEnd = (((fold + 1) * batchSize) > secondRoundData.n_cols) ?
        (secondRoundData.n_cols - 1) : ((fold + 1) * batchSize - 1);
    const size_t heldOutSize = (heldOutEnd - heldOutStart + 1);
    arma::rowvec foldResponses(trainLabels.n_elem - heldOutSize);
    if (heldOutStart > 0)
    {
      foldResponses.subvec(0, heldOutStart - 1) =
          trainLabels.subvec(0, heldOutStart - 1);
    }

    if (heldOutEnd < trainLabels.n_elem - 1)
    {
      foldResponses.subvec(heldOutStart, foldResponses.n_elem - 1) =
          trainLabels.subvec(heldOutEnd + 1, trainLabels.n_elem - 1);
    }

    const double cvLambda = std::pow(10.0, cvLambdaPowers[lIndex]);
    liblinear::Train(
        (const liblinear::feature_node**) secondRoundFeatures[fold],
        secondRoundData.n_cols - heldOutSize,
        secondRoundData.n_rows,
        foldResponses.memptr(),
        cvModel.memptr(),
        cvLambda,
        20, // max outer iterations
        50, // max inner iterations
        0.001,
        false, // l2 regularization
        verbose,
        seed + lIndex);

    const arma::rowvec z(-cvModel * secondRoundData.cols(heldOutStart,
        heldOutEnd));
    const arma::rowvec predictions = 2.0 *
        arma::round(1.0 / (1.0 + arma::exp(z))) - 1.0;

    const double accuracy = arma::accu(predictions ==
        trainLabels.cols(heldOutStart, heldOutEnd)) / (double) heldOutSize;
    accuracies(fold, lIndex) = accuracy;
  }
  const size_t bestLambdaIndex = arma::index_max(arma::mean(accuracies, 0));
  const double bestLambda = std::pow(10.0, cvLambdaPowers[bestLambdaIndex]);
  const double cvTime = c.toc();

  // Now train the final model on all the data.
  c.tic();
  arma::rowvec tmpModel(secondRoundData.n_rows, arma::fill::none);
  liblinear::Train(
      (const liblinear::feature_node**) secondRoundFeatures[cvFolds],
      secondRoundData.n_cols,
      secondRoundData.n_rows,
      trainLabels.memptr(),
      tmpModel.memptr(),
      bestLambda,
      20, // max outer iterations
      50, // max inner iterations
      0.001,
      false, // l2 regularization
      verbose,
      seed + cvPoints);
  const double finalTrainTime = c.toc();

  c.tic();
  CleanKFoldFeatureNodes(secondRoundFeatures, cvFolds, secondRoundData.n_cols);

  // Lastly, we need to unpack the "compressed" tmpModel (which was not trained
  // on all dimensions) into the true model.
  arma::vec compressedModel = wHat * tmpModel.t();
  model.zeros();
  for (size_t i = 0; i < compressedModel.n_elem; ++i)
  {
    model[nonzeroDims[i]] = compressedModel[i];
  }
  const double cleanupTime = c.toc();

  //if (verbose)
  //{
    std::cout << "OWA model timing for lambda " << lambda << ":\n";
    std::cout << " - First round training time: " << trainTime << "s.\n";
    std::cout << " - Model collection time: " << modelCollectionTime << "s.\n";
    std::cout << " - Dimension filtering time: " << filterTime << "s.\n";
    std::cout << " - Second round data computation time: " << secondRoundDataTime << "s.\n";
    std::cout << " - LIBLINEAR second round conversion time: " << secondRoundConversionTime << "s.\n";
    std::cout << " - Cross-validation time: " << cvTime << "s.\n";
    std::cout << " - Final training time: " << finalTrainTime << "s.\n";
    std::cout << " - Cleanup time: " << cleanupTime << "s.\n";
  //}
}

inline void MPICSL::Update(double alpha)
{
  // Send model from main worker to all other workers.
  // Note: This is done already to calculate accuracy but needs to be ensured
  // regardless. (and timed as part of the procedure)
  arma::wall_clock c;
  c.tic();
  this->DistributeModel();
  const double distributeModelTime = c.toc();

  // Don't filter any features in subsequent procedures
  size_t* active_dims = nullptr;
  size_t active_size = -1;

  // Compute the local gradients.
  c.tic();
  arma::rowvec grad(model.n_elem, arma::fill::none);
  liblinear::lr_gradient((const liblinear::feature_node**) liblinearTrainData[0],
                         trainData.n_cols,
                         trainData.n_rows,
                         trainLabels.memptr(),
                         model.memptr(),
                         grad.memptr(),
                         false,
                         active_dims,
                         active_size);
  const double gradTime = c.toc();
  if (verbose)
  {
    std::cout << "Worker " << worldRank << " took " << gradTime << "s to compute"
        << " the local gradient on " << trainData.n_cols << " points."
        << std::endl;
  }

  // Collect all gradients on the main node.
  c.tic();
  arma::mat grads;      // only used on main node
  if (worldRank == 0)
  {
    grads.set_size(model.n_elem, worldSize);
  }

  // Send the grad (if a worker node); receive the grad (if the main node).
  MPI_Gather(grad.memptr(), grad.n_elem, MPI_DOUBLE,
      grads.memptr() /* grads.col(i) will be grad i */, grad.n_elem,
      MPI_DOUBLE, 0 /* gather on the main node */, MPI_COMM_WORLD);

  // also gather train data sizes to normalize.
  arma::Col<uint64_t> sizes(worldSize);
  const uint64_t localSize = trainData.n_cols;
  MPI_Gather(&localSize, 1, MPI_UINT64_T, sizes.memptr(), 1, MPI_UINT64_T, 0,
      MPI_COMM_WORLD);
  const uint64_t totalPoints = arma::accu(sizes);
  const double gradCollectionTime = c.toc();

  // Compute CSL update on the main node.
  if (worldRank == 0)
  {
    c.tic();
    // Compute normalized global and local gradients.
    arma::rowvec ugrad = arma::sum(grads / (double) totalPoints, 1).t();
    arma::rowvec igrad = grad / trainData.n_cols;

    liblinear::solve_csl_owlqn(
        (const liblinear::feature_node**) liblinearTrainData[0],
        trainData.n_cols,
        trainData.n_rows,
        trainLabels.memptr(),
        model.memptr(),
        igrad.memptr(),
        ugrad.memptr(),
        lambda,
        100,
        active_dims,
        active_size,
        alpha);
    const double CSLTime = c.toc();

    std::cout << "CSL model timing for lambda " << lambda << ":\n";
    std::cout << " - Broadcast model time: " << distributeModelTime << "s.\n";
    std::cout << " - Gradient collection time: " << gradCollectionTime << "s.\n";
    std::cout << " - CSL solve time: " << CSLTime << "s.\n";
  }
}

inline double MPICSL::TrainAccuracy()
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

inline double MPICSL::TestAccuracy()
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

inline size_t MPICSL::CountCorrect(const arma::sp_mat& data,
                                   const arma::rowvec& labels) const
{
  const arma::rowvec z(-model * data);
  const arma::rowvec predictions = 2.0 *
      arma::round(1.0 / (1.0 + arma::exp(z))) - 1.0;

  return arma::accu(labels == predictions);
}

inline void MPICSL::DistributeModel()
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
