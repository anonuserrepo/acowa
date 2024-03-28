/**
 * owa.hpp
 *
 * Definition of reference (unmodified) OWA, as in the original paper.
 */
#ifndef OWA_HPP
#define OWA_HPP

#include <armadillo>
#include "liblinear/linear.hpp"

template<typename MatType>
class OWA
{
 public:
  OWA(const double lambda = 0.0,
      const size_t partitions = 1,
      const size_t minSecondRoundPoints = 100,
      const size_t cvFolds = 5,
      const size_t cvPoints = 10);

  ~OWA();

  OWA(const OWA& other);
  OWA(OWA&& other);
  OWA& operator=(const OWA& other);
  OWA& operator=(OWA&& other);

  void Train(const MatType& data,
             const arma::rowvec& responses);

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
  size_t points; // number of points in each partition
  size_t lastPartitionPoints; // number of points in last partition

  MatType localData;
  arma::rowvec localResponses;
  size_t dims;
  bool alias;
};

#include "owa_impl.hpp"

#endif
