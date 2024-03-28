/**
 * reorder_partition_data.hpp
 *
 * A utility to reorder the points in a dataset in a better way for k-means
 * clustering each individual partition.
 */
#ifndef REORDER_PARTITION_DATA_HPP
#define REORDER_PARTITION_DATA_HPP

template<typename eT>
void ReorderPartitionData(arma::Mat<eT>& data,
                          const arma::Row<eT>& labels,
                          const size_t partitions,
                          arma::Col<size_t>& posCounts);

template<typename eT>
void ReorderPartitionData(arma::SpMat<eT>& data,
                          const arma::Row<eT>& labels,
                          const size_t partitions,
                          arma::Col<size_t>& posCounts);

// Include implementation.
#include "reorder_partition_data_impl.hpp"

#endif
