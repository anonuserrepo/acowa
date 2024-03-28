/**
 * acowa_impl.hpp
 *
 * Implementation of feature-weighted second-round centroid augmented OWA.
 */
#ifndef ACOWA_IMPL_HPP
#define ACOWA_IMPL_HPP

#include "acowa.hpp"
#include "liblinear_interface.hpp"
#include "filter_dims.hpp"
#include "reorder_partition_data.hpp"

template<typename MatType>
ACOWA<MatType>::ACOWA(const double lambda,
                                const size_t partitions,
                                const size_t minSecondRoundPoints,
                                const size_t cvFolds,
                                const size_t cvPoints) :
    partitions(partitions),
    lambda(lambda),
    minSecondRoundPoints(minSecondRoundPoints),
    cvFolds(cvFolds),
    cvPoints(cvPoints),
    verbose(false),
    numThreads(1),
    seed(0),
    modelNonzeros(0),
    liblinearFeatures(NULL),
    points(0),
    lastPartitionPoints(0),
    dims(0),
    alias(false)
{
  // Nothing else to do.
}

template<typename MatType>
ACOWA<MatType>::~ACOWA()
{
  if (!alias)
  {
    for (size_t i = 0; i < partitions; ++i)
      liblinear::CleanFeatureNode(this->liblinearFeatures[i], this->dims);
    delete[] this->liblinearFeatures;
  }
}

template<typename MatType>
ACOWA<MatType>::ACOWA(const ACOWA<MatType>& other) :
    partitions(other.partitions),
    lambda(other.lambda),
    minSecondRoundPoints(other.minSecondRoundPoints),
    cvFolds(other.cvFolds),
    cvPoints(other.cvPoints),
    verbose(other.verbose),
    numThreads(other.numThreads),
    seed(other.seed),
    model(other.model),
    modelNonzeros(other.modelNonzeros),
    liblinearFeatures(other.liblinearFeatures), // alias!
    liblinearResponses((double*) other.liblinearResponses.memptr(),
                       other.liblinearResponses.n_rows,
                       other.liblinearResponses.n_cols,
                       false,
                       true),
    liblinearWeights((double*) other.liblinearWeights.memptr(),
                     other.liblinearWeights.n_rows,
                     other.liblinearWeights.n_cols,
                     false,
                     true),
    points(other.points),
    lastPartitionPoints(other.lastPartitionPoints),
    localData(other.localData), // TODO: use alias here?
    localResponses(other.localResponses),
    dims(other.dims),
    alias(true)
{
  // Nothing else to do.
}

template<typename MatType>
ACOWA<MatType>::ACOWA(ACOWA<MatType>&& other) :
    partitions(std::move(other.partitions)),
    lambda(std::move(other.lambda)),
    minSecondRoundPoints(std::move(minSecondRoundPoints)),
    cvFolds(std::move(other.cvFolds)),
    cvPoints(std::move(other.cvPoints)),
    verbose(std::move(other.verbose)),
    numThreads(std::move(other.numThreads)),
    seed(std::move(other.seed)),
    model(std::move(other.model)),
    modelNonzeros(std::move(other.modelNonzeros)),
    liblinearFeatures(other.liblinearFeatures), // take ownership
    liblinearResponses(std::move(other.liblinearResponses)),
    liblinearWeights(std::move(other.liblinearWeights)),
    points(std::move(other.points)),
    lastPartitionPoints(std::move(other.lastPartitionPoints)),
    localData(std::move(other.localData)),
    localResponses(std::move(other.localResponses)),
    dims(std::move(other.dims)),
    alias(false)
{
  other.liblinearFeatures = NULL;
  other.alias = false;
}

template<typename MatType>
ACOWA<MatType>& ACOWA<MatType>::operator=(
    const ACOWA<MatType>& other)
{
  if (this != &other)
  {
    this->partitions = other.partitions;
    this->lambda = other.lambda;
    this->minSecondRoundPoints = other.minSecondRoundPoints;
    this->cvFolds = other.cvFolds;
    this->cvPoints = other.cvPoints;
    this->verbose = other.verbose;
    this->numThreads = other.numThreads;
    this->seed = other.seed;
    this->model = other.model;
    this->modelNonzeros = other.modelNonzeros;
    this->liblinearFeatures = other.liblinearFeatures; // alias
    this->liblinearResponses = arma::mat(other.liblinearResponses.memptr(),
                                         other.liblinearResponses.n_rows,
                                         other.liblinearResponses.n_cols,
                                         false,
                                         true);
    this->liblinearWeights = arma::mat(other.liblinearWeights.memptr(),
                                       other.liblinearWeights.n_rows,
                                       other.liblinearWeights.n_cols,
                                       false,
                                       true);
    this->points = other.points;
    this->lastPartitionPoints = other.lastPartitionPoints;
    this->localData = other.localData; // TODO: use alias here?
    this->localResponses = other.localResponses;
    this->dims = other.dims;
    this->alias = true;
  }

  return *this;
}

template<typename MatType>
ACOWA<MatType>& ACOWA<MatType>::operator=(
    ACOWA<MatType>&& other)
{
  if (this != &other)
  {
    this->partitions = std::move(other.partitions);
    this->lambda = std::move(other.lambda);
    this->minSecondRoundPoints = std::move(other.minSecondRoundPoints);
    this->cvFolds = std::move(other.cvFolds);
    this->cvPoints = std::move(other.cvPoints);
    this->verbose = std::move(other.verbose);
    this->numThreads = std::move(other.numThreads);
    this->seed = std::move(other.seed);
    this->model = std::move(other.model);
    this->modelNonzeros = std::move(other.modelNonzeros);
    this->liblinearFeatures = other.liblinearFeatures; // take ownership
    this->liblinearResponses = std::move(other.liblinearResponses);
    this->liblinearWeights = std::move(other.liblinearWeights);
    this->points = std::move(other.points);
    this->lastPartitionPoints = std::move(other.lastPartitionPoints);
    this->localData = std::move(other.localData);
    this->localResponses = std::move(other.localResponses);
    this->dims = std::move(other.dims);
    this->alias = false;

    other.liblinearFeatures = NULL;
    other.alias = false;
  }

  return *this;
}

template<typename MatType>
void ACOWA<MatType>::Train(MatType& data,
                                arma::rowvec& labels)
{
  // The first step is to partition into subsets.
  points = (data.n_cols + partitions - 1) / partitions;

  // We don't want to copy out all the different data partitions, so instead we
  // re-order data inside each partition.  The idea is that each partition will
  // first contain the 0 points, then it will contain the 1 points.
  arma::Col<size_t> posCounts(partitions);
  ReorderPartitionData(data, labels, partitions, posCounts);

  arma::mat allCentroids(data.n_rows, 2 * partitions);
  arma::mat centroidCounts(2, partitions, arma::fill::zeros);

  #pragma omp parallel for schedule(dynamic) num_threads(numThreads)
  for (size_t p = 0; p < partitions; ++p)
  {
    const size_t start = p * points;
    const size_t startNeg = start + posCounts[p];
    const size_t end = std::min((p + 1) * points, (size_t) data.n_cols);

    allCentroids.col(2 * p) = arma::mean(data.cols(start, startNeg - 1), 1);
    allCentroids.col(2 * p + 1) = arma::mean(data.cols(startNeg, end - 1), 1);
    centroidCounts[2 * p] = (startNeg - start);
    centroidCounts[2 * p + 1] = (end - startNeg);
  }

  // First, convert into a series of LIBLINEAR problems.
  liblinearFeatures = liblinear::KMeansAugMatToTransposedFeatureNodes(data,
      allCentroids, partitions);
  liblinearResponses.set_size(points + 2 * partitions, partitions);
  liblinearWeights.set_size(points + 2 * partitions, partitions);
  for (size_t p = 0; p < partitions - 1; ++p)
  {
    liblinearResponses.submat(0, p, posCounts[p] - 1, p).fill(1);
    labels.subvec(p * points, p * points + posCounts[p] - 1).fill(1);
    liblinearResponses.submat(posCounts[p], p, points - 1, p).fill(-1);
    labels.subvec(p * points + posCounts[p], (p + 1) * points - 1).fill(-1);

    liblinearWeights.submat(0, p, points - 1, p).fill(1);

    // Set responses for all centroid points.
    liblinearResponses.row(points + 2 * p).fill(1);
    liblinearResponses.row(points + 2 * p + 1).fill(-1);

    // The weights for each centroid should be the number of points assigned to
    // them.
    liblinearWeights.row(points + 2 * p).fill(0.1 * centroidCounts(0, p));
    liblinearWeights.row(points + 2 * p + 1).fill(0.1 * centroidCounts(1, p));
  }

  lastPartitionPoints = (data.n_cols - (partitions - 1) * points);
  liblinearResponses.submat(0,
                            (partitions - 1),
                            posCounts[partitions - 1] - 1,
                            (partitions - 1)).fill(1);
  labels.subvec((partitions - 1) * points,
                (partitions - 1) * points + posCounts[partitions - 1] - 1).fill(1);
  liblinearResponses.submat(posCounts[partitions - 1], (partitions - 1),
                            lastPartitionPoints - 1, (partitions - 1)).fill(-1);
  labels.subvec((partitions - 1) * points + posCounts[partitions - 1],
                (partitions - 1) * points + lastPartitionPoints - 1).fill(-1);

  liblinearWeights.submat(0,
                          (partitions - 1),
                          points - 1,
                          (partitions - 1)).fill(1.0);

  // Set responses for all centroid points.
  liblinearResponses.row(points + 2 * (partitions - 1)).fill(1);
  liblinearResponses.row(points + 2 * (partitions - 1) + 1).fill(-1);

  // The weights for each centroid should be the number of points assigned to
  // them.
  liblinearWeights.row(points + 2 * (partitions - 1)).fill(
      centroidCounts(0, partitions - 1));
  liblinearWeights.row(points + 2 * (partitions - 1) + 1).fill(
      centroidCounts(1, partitions - 1));

  // The centroid responses and weights for the last partition need to be
  // shifted up, since the last partition has fewer points.
  if (lastPartitionPoints != points)
  {
    liblinearResponses.submat(lastPartitionPoints,
                              (partitions - 1),
                              lastPartitionPoints + 2 * partitions - 1,
                              (partitions - 1)) =
        liblinearResponses.submat(points,
                                  (partitions - 1),
                                  liblinearResponses.n_rows - 1,
                                  (partitions - 1));

    liblinearWeights.submat(lastPartitionPoints,
                            (partitions - 1),
                            lastPartitionPoints + 2 * partitions - 1,
                            (partitions - 1)) =
        liblinearWeights.submat(points,
                                (partitions - 1),
                                liblinearWeights.n_rows - 1,
                                (partitions - 1));
  }

  // Account for augmented points.
  points += 2 * partitions;
  lastPartitionPoints += 2 * partitions;

  localData = data;
  localResponses = labels;
  dims = data.n_rows;

  Retrain();
}

template<typename MatType>
void ACOWA<MatType>::Retrain()
{
  if (this->liblinearFeatures == nullptr)
  {
    throw std::invalid_argument("OWA::Retrain(): you must first call Train()!");
  }

  // First train each individual OWA learner.
  arma::mat models(dims, partitions, arma::fill::none);
  #pragma omp parallel for schedule(dynamic) num_threads(numThreads) \
      default(shared)
  for (size_t i = 0; i < partitions; ++i)
  {
    liblinear::Train((const liblinear::feature_node**) liblinearFeatures[i],
                     (i == (partitions - 1)) ? lastPartitionPoints : points,
                     dims,
                     liblinearResponses.colptr(i),
                     liblinearWeights.colptr(i),
                     models.colptr(i),
                     lambda,
                     20, // max outer iterations
                     50, // max inner iterations
                     0.001,
                     true, // l1 regularization
                     false, // instance weighting
                     verbose,
                     seed + i);
  }

  // Now compute feature importances for reweighting during the second round.
  // Each feature is scaled between 1 and 101; features that were seen more in
  // `models` are given higher weight.
  arma::vec featureScaling =
      arma::mean(arma::conv_to<arma::mat>::from(models != 0), 1);

  #pragma omp parallel for schedule(dynamic) num_threads(numThreads) \
      default(shared)
  for (size_t i = 0; i < partitions; ++i)
  {
    arma::vec featureWeights = 1.0 + 10 * featureScaling;

    const double lambda2 = lambda * featureWeights.n_elem /
        arma::accu(1.0 / featureWeights);

    // Feature-weighted training this time.
    liblinear::Train((const liblinear::feature_node**) liblinearFeatures[i],
                     (i == (partitions - 1)) ? lastPartitionPoints : points,
                     dims,
                     liblinearResponses.colptr(i),
                     featureWeights.memptr(),
                     models.colptr(i),
                     lambda2,
                     20, // max outer iterations
                     50, // max inner iterations
                     0.001,
                     true, // l1 regularization
                     true, // feature weighting (not instance weighting)
                     verbose,
                     seed + partitions + i);
  }

  // Compute the number of points to use for the second round.
  // We use 10% of the dataset size.
  const size_t secondRoundPoints = size_t(0.1 * localData.n_cols);
  arma::uvec secondRoundSamples;
  if (secondRoundPoints == localData.n_cols)
  {
    secondRoundSamples = arma::linspace<arma::uvec>(0, localData.n_cols - 1,
        localData.n_cols);
  }
  else
  {
    secondRoundSamples = arma::randperm(localData.n_cols, secondRoundPoints);
  }

  // We will filter down to dimensions that are nonzero in any individual
  // learner.  This is a simple optimization that doesn't affect the result.
  const arma::uvec nonzeroDims = arma::find(arma::sum(models, 1));
  arma::mat wHat = models.rows(nonzeroDims);
  MatType secondRoundRawData = FilterDims(localData, nonzeroDims,
      secondRoundSamples);

  // This will be dense.
  arma::mat secondRoundData = wHat.t() * secondRoundRawData;
  arma::rowvec secondRoundLabels = localResponses.cols(secondRoundSamples);

  // Now, we want to do k-fold cross-validation over a grid of possible lambda
  // values.
  liblinear::feature_node*** secondRoundFeatures =
      liblinear::MatToKFoldFeatureNodes(secondRoundData, cvFolds);
  arma::mat accuracies(cvFolds, cvPoints); // track accuracy of each fold
  const arma::vec cvLambdaPowers = arma::linspace<arma::vec>(-4, 4, cvPoints);

  const size_t totalCVPoints = cvFolds * cvPoints;

  #pragma omp parallel for schedule(dynamic) num_threads(numThreads)
  for (size_t cvIndex = 0; cvIndex < totalCVPoints; ++cvIndex)
  {
    const size_t fold = cvIndex / cvPoints;
    const size_t lIndex = cvIndex % cvPoints;

    arma::rowvec cvModel(secondRoundData.n_rows);

    // Compute start and end of held-out set, and its size.
    const size_t batchSize = (secondRoundPoints + cvFolds - 1) / cvFolds;
    const size_t heldOutStart = fold * batchSize;
    const size_t heldOutEnd = (((fold + 1) * batchSize) > secondRoundPoints) ?
        (secondRoundPoints - 1) : ((fold + 1) * batchSize - 1);
    const size_t heldOutSize = (heldOutEnd - heldOutStart + 1);
    arma::rowvec foldResponses(secondRoundLabels.n_elem - heldOutSize);
    if (heldOutStart > 0)
    {
      foldResponses.subvec(0, heldOutStart - 1) =
          secondRoundLabels.subvec(0, heldOutStart - 1);
    }

    if (heldOutEnd < secondRoundLabels.n_elem - 1)
    {
      foldResponses.subvec(heldOutStart, foldResponses.n_elem - 1) =
          secondRoundLabels.subvec(heldOutEnd + 1,
                                   secondRoundLabels.n_elem - 1);
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
        seed + partitions + lIndex);

    const arma::rowvec z(-cvModel * secondRoundData.cols(heldOutStart,
        heldOutEnd));
    const arma::rowvec predictions = 2.0 *
        arma::round(1.0 / (1.0 + arma::exp(z))) - 1.0;

    const double accuracy = arma::accu(predictions ==
        localResponses.cols(heldOutStart, heldOutEnd)) / (double) heldOutSize;
    accuracies(fold, lIndex) = accuracy;
  }
  const size_t bestLambdaIndex = arma::index_max(arma::mean(accuracies, 0));
  const double bestLambda = std::pow(10.0, cvLambdaPowers[bestLambdaIndex]);

  // Now train the final model on all the data.
  arma::rowvec tmpModel(secondRoundData.n_rows, arma::fill::none);
  liblinear::Train(
      (const liblinear::feature_node**) secondRoundFeatures[cvFolds],
      secondRoundData.n_cols,
      secondRoundData.n_rows,
      secondRoundLabels.memptr(),
      tmpModel.memptr(),
      bestLambda,
      20, // max outer iterations
      50, // max inner iterations
      0.001,
      false, // l2 regularization
      verbose,
      seed + partitions + cvPoints);

  CleanKFoldFeatureNodes(secondRoundFeatures, cvFolds, secondRoundPoints);

  // Lastly, we need to unpack the "compressed" tmpModel (which was not trained
  // on all dimensions) into the true model.
  arma::vec compressedModel = wHat * tmpModel.t();
  model.zeros(dims);
  for (size_t i = 0; i < compressedModel.n_elem; ++i)
  {
    model[nonzeroDims[i]] = compressedModel[i];
  }
  modelNonzeros = arma::accu(compressedModel != 0);
}

template<typename MatType>
void ACOWA<MatType>::Classify(const MatType& data,
                                     arma::rowvec& labels) const
{
  const arma::rowvec z(-model * data);
  labels = 2.0 * arma::round(1.0 / (1.0 + arma::exp(z))) - 1.0;
}

#endif
