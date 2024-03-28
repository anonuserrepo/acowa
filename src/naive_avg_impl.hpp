/**
 * naive_avg_impl.hpp
 *
 * Implementation of naive averaging distributed logistic regression strategy.
 */
#ifndef NAIVE_AVG_IMPL_HPP
#define NAIVE_AVG_IMPL_HPP

#include "naive_avg.hpp"
#include "liblinear_interface.hpp"

inline NaiveAvg::NaiveAvg(const double lambda, const size_t partitions) :
    lambda(lambda),
    partitions(partitions),
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

inline NaiveAvg::~NaiveAvg()
{
  if (!alias)
  {
    for (size_t i = 0; i < partitions; ++i)
      CleanFeatureNode(this->liblinearFeatures[i], this->dims);
    delete[] this->liblinearFeatures;
  }
}

inline NaiveAvg::NaiveAvg(const NaiveAvg& other) :
    lambda(other.lambda),
    partitions(other.partitions),
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
    points(other.points),
    lastPartitionPoints(other.lastPartitionPoints),
    dims(other.dims),
    alias(true)
{
  // Nothing else to do.
}

inline NaiveAvg::NaiveAvg(NaiveAvg&& other) :
    lambda(std::move(other.lambda)),
    partitions(std::move(other.partitions)),
    verbose(std::move(other.verbose)),
    numThreads(std::move(other.numThreads)),
    seed(std::move(other.seed)),
    model(std::move(other.model)),
    modelNonzeros(std::move(other.modelNonzeros)),
    liblinearFeatures(other.liblinearFeatures), // take ownership
    liblinearResponses(std::move(other.liblinearResponses)),
    points(std::move(other.points)),
    lastPartitionPoints(std::move(other.lastPartitionPoints)),
    dims(std::move(other.dims)),
    alias(false)
{
  other.liblinearFeatures = NULL;
  other.alias = false;
}

inline NaiveAvg& NaiveAvg::operator=(const NaiveAvg& other)
{
  if (this != &other)
  {
    this->lambda = other.lambda;
    this->partitions = other.partitions;
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
    this->points = other.points;
    this->lastPartitionPoints = other.lastPartitionPoints;
    this->dims = other.dims;
    this->alias = true;
  }

  return *this;
}

inline NaiveAvg& NaiveAvg::operator=(NaiveAvg&& other)
{
  if (this != &other)
  {
    this->lambda = std::move(other.lambda);
    this->partitions = std::move(other.partitions);
    this->verbose = std::move(other.verbose);
    this->numThreads = std::move(other.numThreads);
    this->seed = std::move(other.seed);
    this->model = std::move(other.model);
    this->modelNonzeros = std::move(other.modelNonzeros);
    this->liblinearFeatures = other.liblinearFeatures; // take ownership
    this->liblinearResponses = std::move(other.liblinearResponses);
    this->points = std::move(other.points);
    this->lastPartitionPoints = std::move(other.lastPartitionPoints);
    this->dims = std::move(other.dims);
    this->alias = false;

    other.liblinearFeatures = NULL;
    other.alias = false;
  }

  return *this;
}

template<typename MatType>
void NaiveAvg::Train(const MatType& data, const arma::rowvec& labels)
{
  arma::wall_clock c;
  c.tic();
  liblinearFeatures = liblinear::MatToTransposedFeatureNodes(data, partitions);
  const double featureConvTime = c.toc();
  std::cout << "Converting features to LIBLINEAR format took " <<
      featureConvTime << "s." << std::endl;

  c.tic();
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
  const double responseConvTime = c.toc();
  std::cout << "Converting responses to LIBLINEAR format took " <<
      responseConvTime << "s." << std::endl;

  Retrain();
}

inline void NaiveAvg::Retrain()
{
  if (this->liblinearFeatures == nullptr)
  {
    throw std::invalid_argument("NaiveAvg::Retrain(): you must first call "
        "Train()!");
  }

  arma::mat models(dims, partitions);
  #pragma omp parallel for num_threads(numThreads) default(none) \
                           shared(models, std::cout) schedule(static)
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
                     0.001, // inner initial tol
                     true, // l1 regularization
                     verbose,
                     seed + p /* different seed for each partition */);
  }

  // Now compute an average of all the individual models to get the final model.
  model = arma::mean(models, 1).t();
  modelNonzeros = arma::accu(model != 0);
}

template<typename MatType>
void NaiveAvg::Classify(const MatType& data, arma::rowvec& labels) const
{
  const arma::rowvec z(-model * data);
  labels = 2.0 * arma::round(1.0 / (1.0 + arma::exp(z))) - 1.0;
}

#endif
