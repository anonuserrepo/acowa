/**
 * data_utils_impl.hpp
 *
 * Implementation of miscellaneous data preprocessing utilities.
 */
#ifndef DATA_UTILS_IMPL_HPP
#define DATA_UTILS_IMPL_HPP

#include "data_utils.hpp"

// Get training and test files, if they were specified as one parameter.
// The test file will be empty if not given.
std::tuple<std::string, std::string> SplitDatasetArgument(
    const std::string& arg)
{
  size_t idx = arg.find_first_of(',');
  if (idx == std::string::npos)
  {
    return std::make_tuple(arg, std::string(""));
  }
  else
  {
    std::string trainFile = arg.substr(0, idx);
    std::string testFile = arg.substr(idx + 1);

    return std::make_tuple(trainFile, testFile);
  }
}

// Shuffle the data, and split it into training and test set.
inline void TrainTestSplit(const arma::sp_mat& data,
                           const arma::rowvec& labels,
                           const double trainPct,
                           arma::sp_mat& trainData,
                           arma::rowvec& trainLabels,
                           arma::sp_mat& testData,
                           arma::rowvec& testLabels)
{
  const size_t lastTrainIndex = 0.8 * data.n_cols;
  // Column i in the output comes from column order[i] in the input.
  arma::uvec order = arma::shuffle(arma::linspace<arma::uvec>(0,
      data.n_cols - 1, data.n_cols));

  trainLabels = labels.cols(order.subvec(0, lastTrainIndex - 1));
  testLabels = labels.cols(order.subvec(lastTrainIndex,
      order.n_elem - 1));

  trainData.zeros(data.n_rows, lastTrainIndex);
  testData.zeros(data.n_rows, (data.n_cols - lastTrainIndex));

  // Compute the size we will need for the train and test sets.
  size_t trainNonzeros = 0;
  size_t testNonzeros = 0;
  for (size_t i = 0; i < order.n_elem; ++i)
  {
    const size_t inCol = order[i];
    if (i < lastTrainIndex)
      trainNonzeros += (data.col_ptrs[inCol + 1] - data.col_ptrs[inCol]);
    else
      testNonzeros += (data.col_ptrs[inCol + 1] - data.col_ptrs[inCol]);
  }

  trainData.mem_resize(trainNonzeros);
  testData.mem_resize(testNonzeros);

  size_t currentTrainCol = 0;
  size_t currentTrainPos = 0;
  size_t currentTestCol = 0;
  size_t currentTestPos = 0;
  for (size_t i = 0; i < order.n_elem; ++i)
  {
    const size_t inCol = order[i];
    const size_t colNonzeros =
        (data.col_ptrs[inCol + 1] - data.col_ptrs[inCol]);
    const size_t startPos = data.col_ptrs[inCol];
    if (i < lastTrainIndex)
    {
      for (size_t j = 0; j < colNonzeros; ++j)
      {
        arma::access::rw(trainData.row_indices[currentTrainPos + j]) =
            data.row_indices[startPos + j];
        arma::access::rw(trainData.values[currentTrainPos + j]) =
            data.values[startPos + j];
      }

      arma::access::rw(trainData.col_ptrs[currentTrainCol + 1]) = colNonzeros;
      currentTrainPos += colNonzeros;
      currentTrainCol++;
    }
    else
    {
      for (size_t j = 0; j < colNonzeros; ++j)
      {
        arma::access::rw(testData.row_indices[currentTestPos + j]) =
            data.row_indices[startPos + j];
        arma::access::rw(testData.values[currentTestPos + j]) =
            data.values[startPos + j];
      }

      arma::access::rw(testData.col_ptrs[currentTestCol + 1]) = colNonzeros;
      currentTestPos += colNonzeros;
      currentTestCol++;
    }
  }

  // Now turn the column counts into column pointers.
  for (size_t i = 1; i <= trainData.n_cols; ++i)
    arma::access::rw(trainData.col_ptrs[i]) += trainData.col_ptrs[i - 1];
  for (size_t i = 1; i <= testData.n_cols; ++i)
    arma::access::rw(testData.col_ptrs[i]) += testData.col_ptrs[i - 1];
}

// Compute training and test set accuracies for a given model.
template<typename ModelType>
inline
std::tuple<double, double> TrainTestAccuracy(const ModelType& m,
                                             const arma::sp_mat& trainData,
                                             const arma::rowvec& trainLabels,
                                             const arma::sp_mat& testData,
                                             const arma::rowvec& testLabels)
{
  arma::rowvec predictions;
  m.Classify(trainData, predictions);
  const double trainAcc = arma::accu(trainLabels == predictions) /
      (double) trainLabels.n_elem;

  m.Classify(testData, predictions);
  const double testAcc = arma::accu(testLabels == predictions) /
      (double) testLabels.n_elem;

  return std::make_tuple(trainAcc, testAcc);
}

#endif
