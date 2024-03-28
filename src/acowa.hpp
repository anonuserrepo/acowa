/**
 * acowa.hpp
 *
 * Definition of feature-weighted second-round centroid augmented OWA.
 */
#ifndef ACOWA_HPP
#define ACOWA_HPP

#include <armadillo>
#include "liblinear/linear.hpp"

template<typename MatType>
class ACOWA
{
 public:
  ACOWA(const double lambda = 0.0,
                 const size_t partitions = 1,
                 const size_t minSecondRoundPoints = 100,
                 const size_t cvFolds = 5,
                 const size_t cvPoints = 10);

  ~ACOWA();

  ACOWA(const ACOWA& other);
  ACOWA(ACOWA&& other);
  ACOWA& operator=(const ACOWA& other);
  ACOWA& operator=(ACOWA&& other);

  void Train(MatType& data,
             arma::rowvec& responses);

  void Retrain();

  void Classify(const MatType& data,
                arma::rowvec& predictions) const;

  size_t partitions;
  double lambda;
  size_t minSecondRoundPoints;
  size_t cvFolds;
  size_t cvPoints;
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
  arma::mat liblinearWeights; // one column per partition
  size_t points; // number of points in each partition
  size_t lastPartitionPoints; // number of points in last partition

  MatType localData;
  arma::rowvec localResponses;
  size_t dims;
  bool alias;
};

#include "acowa_impl.hpp"

#endif
