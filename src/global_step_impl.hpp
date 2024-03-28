/**
 * global_step_impl.hpp
 *
 * Implementation of >=2 round distributed optimization with global gradients
 */
#ifndef GLOBAL_STEP_IMPL_HPP
#define GLOBAL_STEP_IMPL_HPP

#include "global_step.hpp"
#include "liblinear_interface.hpp"
#include "filter_dims.hpp"
#include "csl_solvers.hpp"

template<typename MatType>
GlobalStep<MatType>::GlobalStep(const double lambda,
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
    totalPoints(0),
    alias(false)
{
  // Nothing else to do.
}

template<typename MatType>
GlobalStep<MatType>::~GlobalStep()
{
  if (!alias)
  {
    for (size_t i = 0; i < partitions; ++i)
      liblinear::CleanFeatureNode(this->liblinearFeatures[i], this->dims);
    delete[] this->liblinearFeatures;
  }
}

template<typename MatType>
GlobalStep<MatType>::GlobalStep(const GlobalStep<MatType>& other) :
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
    totalPoints(other.totalPoints),
    alias(true)
{
  // Nothing else to do.
}

template<typename MatType>
GlobalStep<MatType>::GlobalStep(GlobalStep<MatType>&& other) :
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
    totalPoints(std::move(other.totalPoints)),
    alias(false)
{
  other.liblinearFeatures = NULL;
  other.alias = false;
}

template<typename MatType>
GlobalStep<MatType>& GlobalStep<MatType>::operator=(const GlobalStep<MatType>& other)
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
    this->totalPoints = other.totalPoints;
    this->alias = true;
  }

  return *this;
}

template<typename MatType>
GlobalStep<MatType>& GlobalStep<MatType>::operator=(GlobalStep<MatType>&& other)
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
    this->totalPoints = std::move(other.totalPoints);
    this->alias = false;

    other.liblinearFeatures = NULL;
    other.alias = false;
  }

  return *this;
}

template<typename MatType>
void GlobalStep<MatType>::Train(const MatType& data,
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
  totalPoints = data.n_cols;

  Retrain();
}

template<typename MatType>
void GlobalStep<MatType>::Retrain()
{
  if (this->liblinearFeatures == nullptr)
  {
    throw std::invalid_argument("OwaGs::Retrain(): you must first call Train()!");
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
                     0.001,
                     true, // l1 regularization
                     verbose,
                     seed + i /* different seed per partition */);
  }

  // Compute the number of points to use for the second round.
  // We use 10% of the dataset size.
  const size_t secondRoundPoints = size_t(0.1 * localData.n_cols);
  // arma::uvec secondRoundSamples;
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
  // const arma::uvec nonzeroDims = arma::find(arma::sum(models, 1));
  nonzeroDims = arma::find(arma::sum(models, 1));
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

  #pragma omp parallel for schedule(dynamic) num_threads(numThreads)
  for (size_t fold = 0; fold < cvFolds; ++fold)
  {
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

    #pragma omp parallel for schedule(dynamic) num_threads(innerThreads)
    for (size_t lIndex = 0; lIndex < cvPoints; ++lIndex)
    {
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
  // TODO should this be model rather than compressedModel?
  modelNonzeros = arma::accu(compressedModel != 0);
}

template<typename MatType>
void GlobalStep<MatType>::NaiveTrain(const MatType& data,
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
  totalPoints = data.n_cols;

  NaiveRetrain();
}

template<typename MatType>
void GlobalStep<MatType>::NaiveRetrain()
{
  if (this->liblinearFeatures == nullptr)
  {
    throw std::invalid_argument("OwaGs::NaiveRetrain(): you must first call Train()!");
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
                     0.001,
                     true, // l1 regularization
                     verbose,
                     seed + i /* different seed per partition */);
  }

  // Now compute an average of all the individual models to get the final model.
  model = arma::mean(models, 1).t();
  modelNonzeros = arma::accu(model != 0);

  // store nonzeros information for global steps
  nonzeroDims = arma::find(arma::sum(models, 1));
}

template<typename MatType>
void GlobalStep<MatType>::CSLUpdate(double lambda_t, double alpha)
{
  if (this->liblinearFeatures == nullptr)
  {
    throw std::invalid_argument("GlobalStep::CSLUpdate(): you must first call Train()!");
  }

  double lambda_use = (lambda_t < 0) ? lambda : lambda_t;

  // PARALLEL: get gradient of initial model from each partition
  arma::mat grads(dims, partitions, arma::fill::none);
  #pragma omp parallel for schedule(dynamic) num_threads(numThreads) \
      default(shared)
  for (size_t i = 0; i < partitions; ++i)
  {
    liblinear::lr_gradient(
            (const liblinear::feature_node**) liblinearFeatures[i],
            (i == (partitions - 1)) ? lastPartitionPoints : points,
            dims,
            liblinearResponses.colptr(i),
            model.colptr(0),
            grads.colptr(i),
            false,    // normalize in following lines
            nullptr,
            -1);
  }
  // average the distributed gradients to get global gradient
  arma::rowvec ugrad = arma::sum(grads / totalPoints, 1).t();

  // compute normalized local gradient
  arma::rowvec igrad = (grads.col(0) / points).t();

  // LOCAL: solve CSL objective on first node
  liblinear::solve_csl_owlqn(
    (const liblinear::feature_node**) liblinearFeatures[0],
    points,
    dims,
    liblinearResponses.colptr(0),
    model.colptr(0),
    igrad.colptr(0),
    ugrad.colptr(0),
    lambda_use,
    100,
    nullptr,
    -1,
    alpha,
    false);    // verbose

  // update nonzeros information for global steps
  modelNonzeros = arma::accu(model != 0);
  nonzeroDims = arma::find(model);
}

template<typename MatType>
void GlobalStep<MatType>::DaneUpdate(double alpha)
{
  if (this->liblinearFeatures == nullptr)
  {
    throw std::invalid_argument("GlobalStep::DaneUpdate(): you must first call Train()!");
  }

  // PARALLEL: get gradient of initial model from each partition
  arma::mat grads(dims, partitions, arma::fill::none);
  #pragma omp parallel for schedule(dynamic) num_threads(numThreads) \
      default(shared)
  for (size_t i = 0; i < partitions; ++i)
  {
    liblinear::lr_gradient(
            (const liblinear::feature_node**) liblinearFeatures[i],
            (i == (partitions - 1)) ? lastPartitionPoints : points,
            dims,
            liblinearResponses.colptr(i),
            model.colptr(0),
            grads.colptr(i),
            true,     // normalize here
            nullptr,
            -1);
  }
  
  // average the distributed gradients to get global gradient (correct)
  arma::rowvec ugrad(dims, arma::fill::zeros);
  for (size_t i = 0; i < partitions; ++i)
  {
    size_t correctPoints = (i == (partitions - 1)) ? lastPartitionPoints : points;
    ugrad += grads.col(i).t() * correctPoints;
  }
  ugrad /= totalPoints;

  // PARALLEL: solve shifted objective on each node
  arma::mat models(dims, partitions, arma::fill::none);
  #pragma omp parallel for schedule(dynamic) num_threads(numThreads) \
      default(shared)
  for (size_t i = 0; i < partitions; ++i)
  {
    liblinear::solve_csl_owlqn(
      (const liblinear::feature_node**) liblinearFeatures[i],
      (i == (partitions - 1)) ? lastPartitionPoints : points,
      dims,
      liblinearResponses.colptr(i),
      models.colptr(i),
      grads.colptr(i),
      ugrad.colptr(0),
      lambda,
      100,
      nullptr,
      -1,
      alpha);
  }

  // average the individual models to get the final model
  model = arma::mean(models, 1).t();

  // update nonzeros information for global steps
  modelNonzeros = arma::accu(model != 0);
  nonzeroDims = arma::find(arma::sum(models, 1));
}

template<typename MatType>
double GlobalStep<MatType>::getObjective(arma::rowvec anyModel)
{
  arma::vec objs(partitions, arma::fill::none);
  #pragma omp parallel for schedule(dynamic) num_threads(numThreads) \
      default(shared)
  for (size_t i = 0; i < partitions; ++i)
  {
    objs[i] = liblinear::lr_obj((const liblinear::feature_node**) liblinearFeatures[i],
                                points,
                                dims,
                                liblinearResponses.colptr(i),
                                anyModel.colptr(0),
                                lambda);
  }
  // renormalize because the last partition has different points
  double meanObj = arma::accu(objs * points) / totalPoints;
  return meanObj;
}

template<typename MatType>
void GlobalStep<MatType>::Classify(const MatType& data,
                            arma::rowvec& labels) const
{
  const arma::rowvec z(-model * data);
  labels = 2.0 * arma::round(1.0 / (1.0 + arma::exp(z))) - 1.0;
}

#endif
