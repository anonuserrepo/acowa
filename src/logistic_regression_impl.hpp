/**
 * logistic_regression_impl.hpp
 *
 * Definition of L1-regularized logistic regression on the full dataset.
 */
#ifndef LOGISTIC_REGRESSION_IMPL_HPP
#define LOGISTIC_REGRESSION_IMPL_HPP

#include "logistic_regression.hpp"
#include "liblinear_interface.hpp"

inline LogisticRegression::LogisticRegression(const double lambda) :
    lambda(lambda),
    verbose(false),
    seed(0),
    modelNonzeros(0),
    liblinearFeatures(nullptr),
    points(0),
    dims(0),
    alias(false)
{
  // Nothing else to do.
}

inline LogisticRegression::~LogisticRegression()
{
  if (!alias)
    CleanFeatureNode(this->liblinearFeatures, this->dims);
}

inline LogisticRegression::LogisticRegression(const LogisticRegression& other) :
    lambda(other.lambda),
    verbose(other.verbose),
    seed(other.seed),
    model(other.model),
    modelNonzeros(other.modelNonzeros),
    liblinearFeatures(other.liblinearFeatures), // alias!
    liblinearResponses((double*) other.liblinearResponses.memptr(),
                       other.liblinearResponses.n_elem,
                       false,
                       true), // alias!
    points(other.points),
    dims(other.dims),
    alias(true)
{
  // Nothing else to do.
}

inline LogisticRegression::LogisticRegression(LogisticRegression&& other) :
    lambda(std::move(other.lambda)),
    verbose(std::move(other.verbose)),
    seed(std::move(other.seed)),
    model(std::move(other.model)),
    modelNonzeros(std::move(other.modelNonzeros)),
    liblinearFeatures(other.liblinearFeatures), // take ownership
    liblinearResponses(std::move(other.liblinearResponses)),
    points(std::move(other.points)),
    dims(std::move(other.dims)),
    alias(false)
{
  other.liblinearFeatures = NULL;
  other.alias = false;
}

inline LogisticRegression& LogisticRegression::operator=(
    const LogisticRegression& other)
{
  if (this != &other)
  {
    this->lambda = other.lambda;
    this->verbose = other.verbose;
    this->seed = other.seed;
    this->model = other.model;
    this->modelNonzeros = other.modelNonzeros;
    this->liblinearFeatures = other.liblinearFeatures; // alias!
    this->liblinearResponses = arma::rowvec(
        (double*) other.liblinearResponses.memptr(),
        other.liblinearResponses.n_elem,
        false,
        true);
    this->points = other.points;
    this->dims = other.dims;
    this->alias = true;
  }

  return *this;
}

inline LogisticRegression& LogisticRegression::operator=(
    LogisticRegression&& other)
{
  if (this != &other)
  {
    this->lambda = std::move(other.lambda);
    this->verbose = std::move(other.verbose);
    this->seed = std::move(other.seed);
    this->model = std::move(other.model);
    this->modelNonzeros = std::move(other.modelNonzeros);
    this->liblinearFeatures = other.liblinearFeatures; // take ownership
    this->liblinearResponses = std::move(other.liblinearResponses);
    this->points = std::move(other.points);
    this->dims = std::move(other.dims);
    this->alias = false;

    other.liblinearFeatures = NULL;
    other.alias = false;
  }

  return *this;
}

template<typename MatType>
void LogisticRegression::Train(const MatType& data,
                               const arma::rowvec& responses,
                               bool fast_train)
{
  arma::wall_clock c;
  c.tic();
  liblinear::feature_node*** liblinearFeatureSets =
      liblinear::MatToTransposedFeatureNodes(data, 1);
  const double convertTime = c.toc();
  std::cout << "Converting to LIBLINEAR format took " << convertTime << "s."
      << std::endl;
  this->liblinearFeatures = liblinearFeatureSets[0];
  delete[] liblinearFeatureSets;
  this->liblinearResponses = arma::rowvec((double*) responses.memptr(),
      responses.n_elem, false, true); // make an alias
  this->points = data.n_cols;
  this->dims = data.n_rows;

  Retrain(fast_train);
}

inline void LogisticRegression::Retrain(bool fast_train)
{
  if (this->liblinearFeatures == nullptr)
  {
    throw std::invalid_argument("LogisticRegression::Retrain(): you must first "
                                "call Train()!");
  }

  if (this->lambda >= 0)
  {
    int max_outer_itr = (fast_train) ? 20 : 100;
    int max_inner_itr = (fast_train) ? 50 : 1000;
    double tol = (fast_train) ? 0.001 : 0.001;

    this->model.set_size(dims);
    liblinear::Train((const liblinear::feature_node**) this->liblinearFeatures,
                    this->points,
                    this->dims,
                    this->liblinearResponses.memptr(),
                    this->model.memptr(),
                    this->lambda,
                    max_outer_itr, // max outer iterations
                    max_inner_itr, // max inner iterations
                    tol,
                    true, // l1 regularization
                    this->verbose,
                    this->seed);
  }
  else
  {
    throw std::invalid_argument("LogisticRegression::Retrain(): "
                                "lambda must be >= 0");
  }

  // Compute the number of nonzeros for convenience.
  this->modelNonzeros = arma::accu(this->model != 0);
}

template<typename MatType>
void LogisticRegression::Classify(const MatType& data,
                                  arma::rowvec& labels) const
{
  const arma::rowvec z(-model * data);
  labels = 2.0 * arma::round(1.0 / (1.0 + arma::exp(z))) - 1.0;
}

#endif
