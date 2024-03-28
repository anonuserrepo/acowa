/**
 * libsvm.hpp
 *
 * Trivially simple loader for libsvm data.
 */
#ifndef LIBSVM_HPP
#define LIBSVM_HPP

#include <armadillo>

/**
 * Given a filename, return a matrix containing the data and the labels.
 * The size is inferred from the dataset.
 * No comment lines are allowed.
 */
template<typename MatType>
typename std::tuple<MatType, arma::rowvec>
load_libsvm(const std::string& filename);

// Include implementation.
#include "libsvm_impl.hpp"

#endif
