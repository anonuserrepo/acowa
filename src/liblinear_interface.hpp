/**
 * liblinear_interface.hpp
 *
 * Convenience interface for dealing with LIBLINEAR.
 */
#ifndef LIBLINEAR_INTERFACE_HPP
#define LIBLINEAR_INTERFACE_HPP

#include "liblinear/linear.hpp"
#include <armadillo>

namespace liblinear {

template<typename MatType>
feature_node***
MatToTransposedFeatureNodes(const MatType& X,
                            const size_t partitions);

template<typename MatType>
feature_node***
SubsampleMatToTransposedFeatureNodes(const MatType& X,
                                     const size_t partitions,
                                     const double ratio);

template<typename MatType>
feature_node***
KMeansAugMatToTransposedFeatureNodes(const MatType& X,
                                     const arma::mat& centroids,
                                     const size_t partitions);

template<typename MatType>
feature_node***
MatToKFoldFeatureNodes(const MatType& X,
                       const size_t k);

template<typename eT>
feature_node*
ColToFeatureNode(const arma::Mat<eT>& X,
                 const size_t row);

template<typename eT>
feature_node*
ColToFeatureNode(const arma::SpMat<eT>& X,
                 const size_t row);

template<typename eT>
feature_node*
ColSubsetToFeatureNode(const arma::Mat<eT>& Xt,
                       const size_t col,
                       const size_t startRow,
                       const size_t endRow);

template<typename eT>
feature_node*
ColSubsetToFeatureNode(const arma::SpMat<eT>& Xt,
                       const size_t col,
                       const size_t startRow,
                       const size_t endRow);

template<typename eT>
feature_node* KMeansAugColSubsetToFeatureNode(const arma::Mat<eT>& Xt,
                                              const arma::mat& centroids,
                                              const size_t col,
                                              const size_t startRow,
                                              const size_t endRow);

template<typename eT>
feature_node* KMeansAugColSubsetToFeatureNode(const arma::SpMat<eT>& Xt,
                                              const arma::mat& centroids,
                                              const size_t col,
                                              const size_t startRow,
                                              const size_t endRow);

inline
feature_node***
CopyFeatureNodes(const feature_node*** features,
                 const size_t featureSets,
                 const size_t featureSetSize);

void Train(const feature_node** data,
           const size_t l,
           const size_t n,
           const double* responses,
           double* model,
           const double lambda,
           const size_t max_outer_iter,
           const size_t max_inner_iter,
           const double epsilon,
           const bool l1_regularization,
           const bool verbose,
           const int seed);

void Train(const feature_node** data,
           const size_t l,
           const size_t n,
           const double* responses,
           const double* weights,
           double* model,
           const double lambda,
           const size_t max_outer_iter,
           const size_t max_inner_iter,
           const double epsilon,
           const bool l1_regularization,
           const bool feature_weights,
           const bool verbose,
           const int seed);

void CleanFeatureNode(feature_node**& n);

} // namespace liblinear

#include "liblinear_interface_impl.hpp"

#endif
