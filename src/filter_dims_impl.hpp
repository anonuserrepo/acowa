/**
 * filter_dims_impl.hpp
 *
 * Implementation of utilities to filter zero dimensions.
 */
#ifndef FILTER_DIMS_IMPL_HPP
#define FILTER_DIMS_IMPL_HPP

#include "filter_dims.hpp"

template<typename eT>
inline
arma::Mat<eT> FilterDims(const arma::Mat<eT>& data,
                         const arma::uvec& nonzeroDims,
                         const arma::uvec& cols)
{
  return data.submat(nonzeroDims, cols);
}

template<typename eT>
inline
arma::SpMat<eT> FilterDims(const arma::SpMat<eT>& data,
                           const arma::uvec& nonzeroDims,
                           const arma::uvec& cols)
{
  arma::uvec dimMapping(data.n_rows);
  dimMapping.fill(data.n_rows);
  for (size_t i = 0; i < nonzeroDims.n_elem; ++i)
  {
    dimMapping[nonzeroDims[i]] = i;
  }

  // First count the number of nonzeros in the result.
  size_t nonzeros = 0;
  for (size_t i = 0; i < cols.n_elem; ++i)
  {
    size_t pos = data.col_ptrs[cols[i]];
    size_t end = data.col_ptrs[cols[i] + 1];
    while (pos < end)
    {
      const size_t dim = data.row_indices[pos];
      const size_t newDim = dimMapping[dim];
      if (newDim != data.n_rows)
        ++nonzeros;

      ++pos;
    }
  }

  arma::SpMat<eT> result(nonzeroDims.n_elem, cols.n_elem);
  result.mem_resize(nonzeros);

  // Now copy over the relevant data.
  size_t curPos = 0;
  for (size_t i = 0; i < cols.n_elem; ++i)
  {
    size_t startPos = curPos;
    size_t pos = data.col_ptrs[cols[i]];
    size_t end = data.col_ptrs[cols[i] + 1];
    while (pos < end)
    {
      const size_t dim = data.row_indices[pos];
      const size_t newDim = dimMapping[dim];
      if (newDim != data.n_rows)
      {
        arma::access::rw(result.row_indices[curPos]) = newDim;
        arma::access::rw(result.values[curPos]) = data.values[pos];
        ++curPos;
      }

      ++pos;
    }

    arma::access::rw(result.col_ptrs[i + 1]) = (curPos - startPos);
  }

  // Sum all column counts to make them column pointers.
  for (size_t i = 1; i <= result.n_cols; ++i)
  {
    arma::access::rw(result.col_ptrs[i]) += result.col_ptrs[i - 1];
  }

  return result;
}

template<typename eT>
inline
arma::Mat<eT> FilterDims(const arma::Mat<eT>& data,
                         const arma::uvec& nonzeroDims)
{
  return data.rows(nonzeroDims);
}

template<typename eT>
inline
arma::SpMat<eT> FilterDims(const arma::SpMat<eT>& data,
                           const arma::uvec& nonzeroDims)
{
  arma::uvec dimMapping(data.n_rows);
  dimMapping.fill(data.n_rows);
  for (size_t i = 0; i < nonzeroDims.n_elem; ++i)
  {
    dimMapping[nonzeroDims[i]] = i;
  }

  // First count the number of nonzeros in the result.
  size_t nonzeros = 0;
  for (size_t i = 0; i < data.n_cols; ++i)
  {
    size_t pos = data.col_ptrs[i];
    size_t end = data.col_ptrs[i + 1];
    while (pos < end)
    {
      const size_t dim = data.row_indices[pos];
      const size_t newDim = dimMapping[dim];
      if (newDim != data.n_rows)
        ++nonzeros;

      ++pos;
    }
  }

  arma::SpMat<eT> result(nonzeroDims.n_elem, data.n_cols);
  result.mem_resize(nonzeros);

  // Now copy over the relevant data.
  size_t curPos = 0;
  for (size_t i = 0; i < data.n_cols; ++i)
  {
    size_t startPos = curPos;
    size_t pos = data.col_ptrs[i];
    size_t end = data.col_ptrs[i + 1];
    while (pos < end)
    {
      const size_t dim = data.row_indices[pos];
      const size_t newDim = dimMapping[dim];
      if (newDim != data.n_rows)
      {
        arma::access::rw(result.row_indices[curPos]) = newDim;
        arma::access::rw(result.values[curPos]) = data.values[pos];
        ++curPos;
      }

      ++pos;
    }

    arma::access::rw(result.col_ptrs[i + 1]) = (curPos - startPos);
  }

  // Sum all column counts to make them column pointers.
  for (size_t i = 1; i <= result.n_cols; ++i)
  {
    arma::access::rw(result.col_ptrs[i]) += result.col_ptrs[i - 1];
  }

  return result;
}

#endif
