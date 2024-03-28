/**
 * naive_avg.hpp
 *
 * Definition of naive averaging distributed logistic regression strategy.
 */
#ifndef NAIVE_AVG_HPP
#define NAIVE_AVG_HPP

#include <armadillo>
#include "liblinear/linear.hpp"

class NaiveAvg
{
 public:
  NaiveAvg(const double lambda = 0.0, const size_t partitions = 1);
  ~NaiveAvg();

  NaiveAvg(const NaiveAvg& other);
  NaiveAvg(NaiveAvg&& other);

  NaiveAvg& operator=(const NaiveAvg& other);
  NaiveAvg& operator=(NaiveAvg&& other);

  template<typename MatType>
  void Train(const MatType& data,
             const arma::rowvec& labels);

  void Retrain();

  template<typename MatType>
  void Classify(const MatType& data,
                arma::rowvec& predictions) const;

  // Parameters for model training.
  double lambda;
  size_t partitions;
  bool verbose;
  size_t numThreads;
  size_t seed;

  // Trained model.
  arma::rowvec model;
  size_t modelNonzeros;

 private:
  // Cached features for retraining.
  liblinear::feature_node*** liblinearFeatures;
  arma::mat liblinearResponses; // one column per partition
  size_t points; // number of points in each partition
  size_t lastPartitionPoints; // number of points in last partition
  size_t dims;
  bool alias;
};

#include "naive_avg_impl.hpp"

#endif
