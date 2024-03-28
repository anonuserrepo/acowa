/**
 * filter_dims.hpp
 *
 * Implementation of utility function to filter zero dimensions.
 */
#ifndef FILTER_DIMS_HPP
#define FILTER_DIMS_HPP

#include <armadillo>

template<typename eT>
inline
arma::Mat<eT> FilterDims(const arma::Mat<eT>& data,
                         const arma::uvec& nonzeroDims,
                         const arma::uvec& cols);

template<typename eT>
inline
arma::SpMat<eT> FilterDims(const arma::SpMat<eT>& data,
                           const arma::uvec& nonzeroDims,
                           const arma::uvec& cols);

template<typename eT>
inline
arma::Mat<eT> FilterDims(const arma::Mat<eT>& data,
                         const arma::uvec& nonzeroDims);

template<typename eT>
inline
arma::SpMat<eT> FilterDims(const arma::SpMat<eT>& data,
                           const arma::uvec& nonzeroDims);

#include "filter_dims_impl.hpp"

#endif
