/**
 * debias_avg_impl.hpp
 *
 * Implementation of subsampled debias averaging distributed logistic regression strategy.
 */
#ifndef DEBIAS_AVG_IMPL_HPP
#define DEBIAS_AVG_IMPL_HPP

#include "debias_avg.hpp"
#include "liblinear_interface.hpp"

inline DebiasAvg::DebiasAvg(const double lambda,
                            const size_t partitions,
                            const double ratio) :
    lambda(lambda),
    partitions(partitions),
    ratio(ratio),
    verbose(false),
    numThreads(1),
    seed(0),
    modelNonzeros(0),
    liblinearFeatures(NULL),
    points(0),
    lastPartitionPoints(0),
    dims(0),
    subsampleFeatures(NULL),
    alias(false)
{
  // Nothing else to do.
}

inline DebiasAvg::~DebiasAvg()
{
  if (!alias)
  {
    for (size_t i = 0; i < partitions; ++i)
    {
      CleanFeatureNode(this->liblinearFeatures[i], this->dims);
      CleanFeatureNode(this->subsampleFeatures[i], this->dims);
    }
    delete[] this->liblinearFeatures;
    delete[] this->subsampleFeatures;
  }
}

inline DebiasAvg::DebiasAvg(const DebiasAvg& other) :
    lambda(other.lambda),
    partitions(other.partitions),
    ratio(other.ratio),
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
                       true), // alias!
    subsampleFeatures(other.subsampleFeatures),
    subsampleResponses((double*) other.subsampleResponses.memptr(),
                       other.subsampleResponses.n_rows,
                       other.subsampleResponses.n_cols,
                       false,
                       true),
    points(other.points),
    lastPartitionPoints(other.lastPartitionPoints),
    dims(other.dims),
    alias(true)
{
  // Nothing else to do.
}

inline DebiasAvg::DebiasAvg(DebiasAvg&& other) :
    lambda(std::move(other.lambda)),
    partitions(std::move(other.partitions)),
    ratio(std::move(other.ratio)),
    verbose(std::move(other.verbose)),
    numThreads(std::move(other.numThreads)),
    seed(std::move(other.seed)),
    model(std::move(other.model)),
    modelNonzeros(std::move(other.modelNonzeros)),
    liblinearFeatures(other.liblinearFeatures), // take ownership
    liblinearResponses(std::move(other.liblinearResponses)),
    subsampleFeatures(other.subsampleFeatures),
    subsampleResponses(std::move(other.subsampleResponses)),
    points(std::move(other.points)),
    lastPartitionPoints(std::move(other.lastPartitionPoints)),
    dims(std::move(other.dims)),
    alias(false)
{
  other.liblinearFeatures = NULL;
  other.subsampleFeatures = NULL;
  other.alias = false;
}

inline DebiasAvg& DebiasAvg::operator=(const DebiasAvg& other)
{
  if (this != &other)
  {
    this->lambda = other.lambda;
    this->partitions = other.partitions;
    this->ratio = other.ratio;
    this->verbose = other.verbose;
    this->numThreads = other.numThreads;
    this->seed = other.seed;
    this->model = other.model;
    this->modelNonzeros = other.modelNonzeros;
    this->liblinearFeatures = other.liblinearFeatures; // alias!
    this->liblinearResponses = arma::mat(
        (double*) other.liblinearResponses.memptr(),
        other.liblinearResponses.n_rows,
        other.liblinearResponses.n_cols,
        false,
        true);
    this->subsampleFeatures = other.subsampleFeatures; // alias!
    this->subsampleResponses = arma::mat(
        (double*) other.subsampleResponses.memptr(),
        other.subsampleResponses.n_rows,
        other.subsampleResponses.n_cols,
        false,
        true);
    this->points = other.points;
    this->lastPartitionPoints = other.lastPartitionPoints;
    this->dims = other.dims;
    this->alias = true;
  }

  return *this;
}

inline DebiasAvg& DebiasAvg::operator=(DebiasAvg&& other)
{
  if (this != &other)
  {
    this->lambda = std::move(other.lambda);
    this->partitions = std::move(other.partitions);
    this->ratio = std::move(other.ratio);
    this->verbose = std::move(other.verbose);
    this->numThreads = std::move(other.numThreads);
    this->seed = std::move(other.seed);
    this->model = std::move(other.model);
    this->modelNonzeros = std::move(other.modelNonzeros);
    this->liblinearFeatures = other.liblinearFeatures; // take ownership
    this->liblinearResponses = std::move(other.liblinearResponses);
    this->subsampleFeatures = other.subsampleFeatures; // take ownership
    this->subsampleResponses = std::move(other.subsampleResponses);
    this->points = std::move(other.points);
    this->lastPartitionPoints = std::move(other.lastPartitionPoints);
    this->dims = std::move(other.dims);
    this->alias = false;

    other.liblinearFeatures = NULL;
    other.subsampleFeatures = NULL;
    other.alias = false;
  }

  return *this;
}

template<typename MatType>
void DebiasAvg::Train(const MatType& data, const arma::rowvec& labels)
{
  liblinearFeatures = liblinear::MatToTransposedFeatureNodes(data, partitions);

  points = (data.n_cols + partitions - 1) / partitions;
  liblinearResponses.set_size(points, partitions);
  for (size_t p = 0; p < partitions - 1; ++p)
  {
    liblinearResponses.col(p) = labels.subvec(p * points,
                                              (p + 1) * points - 1).t();
  }
  // The last partition may have a different size.
  lastPartitionPoints = (data.n_cols - (partitions - 1) * points);
  liblinearResponses.submat(0, (partitions - 1), lastPartitionPoints - 1,
      (partitions - 1)) = labels.subvec((partitions - 1) * points,
                                        (partitions - 1) * points +
                                            lastPartitionPoints - 1).t();
  dims = data.n_rows;

  // repeat above process for subsampled data set on each partition
  subsampleFeatures = liblinear::SubsampleMatToTransposedFeatureNodes(
                                                data, partitions, ratio);

  size_t n_sub = ratio * points;
  subsampleResponses.set_size(n_sub, partitions);
  for (size_t p = 0; p < partitions - 1; ++p)
  {
    subsampleResponses.col(p) = labels.subvec(p * points,
                                              p * points + n_sub - 1).t();
  }
  // The last partition may have a different size.
  n_sub = ratio * lastPartitionPoints;
  subsampleResponses.submat(0, (partitions - 1), n_sub - 1,
      (partitions - 1)) = labels.subvec((partitions - 1) * points,
                                        (partitions - 1) * points +
                                            n_sub - 1).t();

  Retrain();
}

inline void DebiasAvg::Retrain()
{
  if (this->liblinearFeatures == nullptr)
  {
    throw std::invalid_argument("DebiasAvg::Retrain(): you must first call "
        "Train()!");
  }

  arma::mat models(dims, partitions);
  arma::mat submodels(dims, partitions);

  #pragma omp parallel for num_threads(numThreads) default(none) \
                           shared(models, submodels, std::cout) schedule(static)
  for (size_t p = 0; p < partitions; ++p)
  {
    liblinear::Train((const liblinear::feature_node**) liblinearFeatures[p],
                     (p == (partitions - 1)) ? lastPartitionPoints : points,
                     dims,
                     liblinearResponses.colptr(p),
                     models.colptr(p),
                     lambda,
                     20, // max outer iterations
                     50, // max inner iterations
                     0.001,
                     true, // l1 regularization
                     verbose,
                     seed + p /* different seed for each partition */);
  

    // fit subsampled model into submodel
    int n_sub = (p == (partitions - 1)) ? ratio * lastPartitionPoints : ratio * points;
    liblinear::Train((const liblinear::feature_node**) subsampleFeatures[p],
                     n_sub,
                     dims,
                     subsampleResponses.colptr(p),
                     submodels.colptr(p),
                     lambda,
                     20, // max outer iterations
                     50, // max inner iterations
                     0.001,
                     true, // l1 regularization
                     verbose,
                     seed + p /* different seed for each partition */);    
  }

  // Now compute an average of all the individual models to get the final model.
  model = arma::mean(models, 1).t();
  arma::rowvec submodel = arma::mean(submodels, 1).t();

  // compute SAVGM estimate
  model = (model - ratio * submodel) / (1 - ratio);

  modelNonzeros = arma::accu(model != 0);
}

template<typename MatType>
void DebiasAvg::Classify(const MatType& data, arma::rowvec& labels) const
{
  const arma::rowvec z(-model * data);
  labels = 2.0 * arma::round(1.0 / (1.0 + arma::exp(z))) - 1.0;
}

#endif
