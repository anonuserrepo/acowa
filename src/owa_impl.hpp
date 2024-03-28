/**
 * owa_impl.hpp
 *
 * Implementation of reference (unmodified) OWA, as in the original paper.
 */
#ifndef OWA_IMPL_HPP
#define OWA_IMPL_HPP

#include "owa.hpp"
#include "liblinear_interface.hpp"
#include "filter_dims.hpp"

template<typename MatType>
OWA<MatType>::OWA(const double lambda,
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
    seed(0),
    numThreads(1),
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
OWA<MatType>::~OWA()
{
  if (!alias)
  {
    for (size_t i = 0; i < partitions; ++i)
      liblinear::CleanFeatureNode(this->liblinearFeatures[i], this->dims);
    delete[] this->liblinearFeatures;
  }
}

template<typename MatType>
OWA<MatType>::OWA(const OWA<MatType>& other) :
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
OWA<MatType>::OWA(OWA<MatType>&& other) :
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
OWA<MatType>& OWA<MatType>::operator=(const OWA<MatType>& other)
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
OWA<MatType>& OWA<MatType>::operator=(OWA<MatType>&& other)
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
void OWA<MatType>::Train(const MatType& data,
                         const arma::rowvec& labels)
{
  // First, convert into a series of LIBLINEAR problems.
  liblinearFeatures = liblinear::MatToTransposedFeatureNodes(data, partitions);

  points = (data.n_cols + partitions - 1) / partitions;
  liblinearResponses.set_size(points, partitions);
  for (size_t p = 0; p < partitions - 1; ++p)
  {
    liblinearResponses.col(p) = labels.subvec(p * points,
                                              (p + 1) * points - 1).t();
  }
  lastPartitionPoints = (data.n_cols - (partitions - 1) * points);
  liblinearResponses.submat(0, (partitions - 1), lastPartitionPoints - 1,
      (partitions - 1)) = labels.subvec((partitions - 1) * points,
                                        (partitions - 1) * points +
                                            lastPartitionPoints - 1).t();

  localData = data;
  localResponses = labels;
  dims = data.n_rows;

  Retrain();
}

template<typename MatType>
void OWA<MatType>::Retrain()
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
                     models.colptr(i),
                     lambda,
                     20, // max outer iterations
                     50, // max inner iterations
                     0.001, // inner initial tol
                     true, // l1 regularization
                     verbose,
                     seed + i /* different seed per partition */);
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

  const size_t outerThreads = (cvFolds + numThreads - 1) / numThreads;
  const size_t innerThreads = numThreads / outerThreads;

  const size_t totalCVTrials = cvFolds * cvPoints;
  #pragma omp parallel for schedule(dynamic) num_threads(numThreads)
  for (size_t cvIndex = 0; cvIndex < totalCVTrials; ++cvIndex)
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
void OWA<MatType>::Classify(const MatType& data,
                            arma::rowvec& labels) const
{
  const arma::rowvec z(-model * data);
  labels = 2.0 * arma::round(1.0 / (1.0 + arma::exp(z))) - 1.0;
}

#endif
