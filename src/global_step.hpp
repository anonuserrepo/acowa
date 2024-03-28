/**
 * global_step.hpp
 *
 * Definition of >=2 round distributed optimization with global gradients
 */
#ifndef GLOBAL_STEP_HPP
#define GLOBAL_STEP_HPP

#include <armadillo>
#include "liblinear/linear.hpp"
#include "csl_solvers.hpp"

template<typename MatType>
class GlobalStep
{
 public:
  GlobalStep(const double lambda = 0.0,
      const size_t partitions = 1,
      const size_t minSecondRoundPoints = 100,
      const size_t cvFolds = 5,
      const size_t cvPoints = 10);

  ~GlobalStep();

  GlobalStep(const GlobalStep& other);
  GlobalStep(GlobalStep&& other);
  GlobalStep& operator=(const GlobalStep& other);
  GlobalStep& operator=(GlobalStep&& other);

  // fits standard OWA
  void Train(const MatType& data,
             const arma::rowvec& responses);

  void Retrain();

  // fits naive model
  void NaiveTrain(const MatType& data,
                  const arma::rowvec& responses);

  void NaiveRetrain();

  void CSLUpdate(double lambda_t = -1.0, double alpha = 0.0);

  void DaneUpdate(double alpha = 0.0);

  double getObjective(arma::rowvec anyModel);

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
  arma::uvec nonzeroDims;

 private:
  // Cached features for retraining.
  liblinear::feature_node*** liblinearFeatures;
  arma::mat liblinearResponses; // one column per partition
  size_t points; // number of points in each partition
  size_t lastPartitionPoints; // number of points in last partition

  MatType localData;
  arma::rowvec localResponses;
  size_t dims;
  size_t totalPoints;
  bool alias;

  // additional info for CSL steps
  arma::uvec secondRoundSamples;

};

#include "global_step_impl.hpp"

#endif
