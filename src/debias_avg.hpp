/**
 * naive_avg.hpp
 *
 * Definition of subsampled debias averaging distributed logistic regression strategy.
 */
#ifndef DEBIAS_AVG_HPP
#define DEBIAS_AVG_HPP

#include <armadillo>
#include "liblinear/linear.hpp"

class DebiasAvg
{
 public:
  DebiasAvg(const double lambda = 0.0,
            const size_t partitions = 1,
            const double ratio = 0.5);
  ~DebiasAvg();

  DebiasAvg(const DebiasAvg& other);
  DebiasAvg(DebiasAvg&& other);

  DebiasAvg& operator=(const DebiasAvg& other);
  DebiasAvg& operator=(DebiasAvg&& other);

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
  double ratio;     // debias subsample fraction in (0, 1)

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

  // subsampled copies for debias method
  liblinear::feature_node*** subsampleFeatures;
  arma::mat subsampleResponses;
};

#include "debias_avg_impl.hpp"

#endif
