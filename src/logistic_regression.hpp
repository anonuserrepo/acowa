/**
 * logistic_regression.hpp
 *
 * Definition of L1-regularized logistic regression on the full dataset.
 */
#ifndef LOGISTIC_REGRESSION_HPP
#define LOGISTIC_REGRESSION_HPP

#include <armadillo>
#include "liblinear/linear.hpp"

class LogisticRegression
{
 public:
  LogisticRegression(const double lambda = 0.0);
  ~LogisticRegression();

  LogisticRegression(const LogisticRegression& other);
  LogisticRegression(LogisticRegression&& other);

  LogisticRegression& operator=(const LogisticRegression& other);
  LogisticRegression& operator=(LogisticRegression&& other);

  template<typename MatType>
  void Train(const MatType& data,
             const arma::rowvec& labels,
             bool fast_train = true);

  void Retrain(bool fast_train = false);

  template<typename MatType>
  void Classify(const MatType& data,
                arma::rowvec& predictions) const;

  // Parameters for model training.
  double lambda;
  bool verbose;
  size_t seed;

  // Trained model.
  arma::rowvec model;
  size_t modelNonzeros;

 private:
  // Cached features for retraining.
  liblinear::feature_node** liblinearFeatures;
  arma::rowvec liblinearResponses;
  size_t points;
  size_t dims;
  bool alias;
};

#include "logistic_regression_impl.hpp"

#endif
