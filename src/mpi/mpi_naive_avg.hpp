/**
 * mpi_naive_avg.hpp
 *
 * MPI-based implementation of naive averaging.
 */
#ifndef MPI_NAIVE_AVG_HPP
#define MPI_NAIVE_AVG_HPP

#include <armadillo>
#include "../liblinear_interface.hpp"

class MPINaiveAvg
{
 public:
  MPINaiveAvg(const std::string& trainBaseFilename,
              const std::string& testBaseFilename,
              const std::string& extension,
	      const size_t dataDim, // we can't trust the individual files
              const double lambda,
              const bool verbose = false,
              const size_t seed = 0);

  ~MPINaiveAvg();

  // Train a new model.
  void Train();

  // Evaluate training set and test set accuracy.  The returned value is only
  // correct on the main node.
  double TrainAccuracy();
  double TestAccuracy();

  // Distribute the model to all workers.
  void DistributeModel();

  double lambda;
  bool verbose;
  size_t seed;

  // On workers, this is the local model.  On the main node, this is the final
  // merged model.  (All of this only applies after Train() is called.)
  arma::rowvec model;

 private:
  // Note that these are only one shard of the data.
  arma::sp_mat trainData;
  arma::rowvec trainLabels;
  arma::sp_mat testData;
  arma::rowvec testLabels;

  liblinear::feature_node*** liblinearTrainData;

  // Internal accuracy computation function.
  size_t CountCorrect(const arma::sp_mat& data, const arma::rowvec& labels)
      const;

  size_t worldSize;
  size_t worldRank;
};

// Include implementation.
#include "mpi_naive_avg_impl.hpp"

#endif
