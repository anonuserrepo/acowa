/**
 * data_utils.hpp
 *
 * Miscellaneous utilities for data preprocessing.
 */
#ifndef DATA_UTILS_HPP
#define DATA_UTILS_HPP

// Get training and test files, if they were specified as one parameter.
// The test file will be empty if not given.
std::tuple<std::string, std::string> SplitDatasetArgument(
    const std::string& arg);

// Shuffle the data, and split it into training and test set.
void TrainTestSplit(const arma::sp_mat& data,
                    const arma::rowvec& labels,
                    const double trainPct,
                    arma::sp_mat& trainData,
                    arma::rowvec& trainLabels,
                    arma::sp_mat& testData,
                    arma::rowvec& testLabels);

// Compute training and test set accuracies for a given model.
template<typename ModelType>
std::tuple<double, double> TrainTestAccuracy(const ModelType& m,
                                             const arma::sp_mat& trainData,
                                             const arma::rowvec& trainLabels,
                                             const arma::sp_mat& testData,
                                             const arma::rowvec& testLabels);

#include "data_utils_impl.hpp"

#endif
