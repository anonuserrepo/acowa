/**
 * reorder_partition_data_impl.hpp
 *
 * A utility to reorder the points in a dataset in a better way for k-means
 * clustering each individual partition.
 */
#ifndef REORDER_PARTITION_DATA_IMPL_HPP
#define REORDER_PARTITION_DATA_IMPL_HPP

template<typename eT>
void ReorderPartitionData(arma::Mat<eT>& data,
                          const arma::Row<eT>& labels,
                          const size_t partitions,
                          arma::Col<size_t>& posCounts)
{
  // We can do this in-place on a dense matrix.
  const size_t points = (data.n_cols + partitions - 1) / partitions;
  for (size_t p = 0; p < partitions; ++p)
  {
    const size_t start = p * points;
    const size_t end = std::min((p + 1) * points, data.n_cols);

    size_t left = start;
    size_t right = end;

    while (left < right)
    {
      while (labels[left] == 1.0 && left < right)
        ++left;
      while (labels[right] == -1.0 && right > left)
        --right;

      if (left < right)
      {
        // Do a swap.
        data.swap_cols(left, right);
      }
    }

    posCounts[p] = left; // This is now the number of positive points.
  }
}

template<typename eT>
void ReorderPartitionData(arma::SpMat<eT>& data,
                          const arma::Row<eT>& labels,
                          const size_t partitions,
                          arma::Col<size_t>& posCounts)
{
  // We can't do this in-place easily for a sparse matrix.
  const size_t points = (data.n_cols + partitions - 1) / partitions;
  arma::SpMat<eT> sortedData(data.n_rows, data.n_cols);
  // Force CSC sync.
  data += sortedData;

  sortedData.mem_resize(data.n_nonzero);

  size_t totalNonzeros = 0;
  for (size_t p = 0; p < partitions; ++p)
  {
    const size_t start = p * points;
    const size_t end = std::min((p + 1) * points, (size_t) data.n_cols);

    // In the first pass, we want to simply count the number of nonzeros
    // associated with positive points and negative points.
    size_t posNonzeros = 0;
    size_t negNonzeros = 0;
    size_t posCols = 0;
    size_t negCols = 0;
    for (size_t i = start; i < end; ++i)
    {
      const size_t colNonzeros = (data.col_ptrs[i + 1] - data.col_ptrs[i]);

      if (labels[i] == 1.0)
      {
        posNonzeros += colNonzeros;
        posCols++;
      }
      else
      {
        negNonzeros += colNonzeros;
        negCols++;
      }
    }

    posCounts[p] = posCols;

    // Now, we can start copying elements directly.
    // Note that directly modifying the Armadillo sparse matrix structure is
    // only an advisable thing to do if you happen to be the person who wrote
    // it!!
    size_t currentInCol = start;
    size_t currentLabel = (labels[start] == 1.0) ? 1 : 0;
    size_t currentPosCol = start;
    size_t currentPosPos = totalNonzeros;
    size_t currentNegCol = start + posCols;
    size_t currentNegPos = totalNonzeros + posNonzeros;
    typename arma::SpMat<eT>::const_iterator it = data.begin_col(start);
    while (it != data.end_col(end - 1))
    {
      if (it.col() != currentInCol)
      {
        // We just finished a column.
        // Update the column pointers (which are right now just column counts).
        if (currentLabel == 1)
        {
          arma::access::rw(sortedData.col_ptrs[currentPosCol + 1]) =
              (data.col_ptrs[currentInCol + 1] - data.col_ptrs[currentInCol]);
          ++currentPosCol;
        }
        else
        {
          arma::access::rw(sortedData.col_ptrs[currentNegCol + 1]) =
              (data.col_ptrs[currentInCol + 1] - data.col_ptrs[currentInCol]);
          ++currentNegCol;
        }

        while (currentInCol + 1 < it.col())
        {
          // We had at least one empty column.  Update the column pointers for
          // the empty columns.
          currentInCol++;
          currentLabel = (labels[currentInCol] == 1.0) ? 1 : 0;
          if (currentLabel == 1)
          {
            arma::access::rw(sortedData.col_ptrs[currentPosCol + 1]) = 0;
            ++currentPosCol;
          }
          else
          {
            arma::access::rw(sortedData.col_ptrs[currentNegCol + 1]) = 0;
            ++currentNegCol;
          }
        }

        currentInCol = it.col();
        currentLabel = (labels[currentInCol] == 1.0) ? 1 : 0;
      }

      if (currentLabel == 1)
      {
        arma::access::rw(sortedData.row_indices[currentPosPos]) = it.row();
        arma::access::rw(sortedData.values[currentPosPos]) = (*it);
        currentPosPos++;
      }
      else
      {
        arma::access::rw(sortedData.row_indices[currentNegPos]) = it.row();
        arma::access::rw(sortedData.values[currentNegPos]) = (*it);
        currentNegPos++;
      }

      ++it;
    }

    // We just finished the last column.
    // Update the column pointers (which are right now just column counts).
    if (currentLabel == 1)
    {
      arma::access::rw(sortedData.col_ptrs[currentPosCol + 1]) =
          (data.col_ptrs[currentInCol + 1] - data.col_ptrs[currentInCol]);
      ++currentPosCol;
    }
    else
    {
      arma::access::rw(sortedData.col_ptrs[currentNegCol + 1]) =
          (data.col_ptrs[currentInCol + 1] - data.col_ptrs[currentInCol]);
      ++currentNegCol;
    }

    totalNonzeros += posNonzeros + negNonzeros;
  }

  // Now sum all the column sums so that they are correctly column pointers.
  for (size_t c = 1; c <= sortedData.n_cols; ++c)
    arma::access::rw(sortedData.col_ptrs[c]) += sortedData.col_ptrs[c - 1];

  data.steal_mem(sortedData);
}

// Include implementation.
#include "reorder_partition_data_impl.hpp"

#endif
