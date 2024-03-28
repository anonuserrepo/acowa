/**
 * mpi_owa.hpp
 *
 * MPI-based implementation of OWA.
 */
#ifndef MPI_OWA_HPP
#define MPI_OWA_HPP

#include <armadillo>
#include "../liblinear_interface.hpp"

class MPIOWA
{
 public:
  MPIOWA(const std::string& trainBaseFilename,
         const std::string& testBaseFilename,
         const std::string& extension,
         const size_t dataDim,
         const double lambda,
         const bool verbose = false,
         const size_t seed = 0,
         const size_t cvFolds = 5,
         const size_t cvPoints = 10);

  ~MPIOWA();

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
  size_t cvFolds;
  size_t cvPoints;

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
#include "mpi_owa_impl.hpp"

#endif
