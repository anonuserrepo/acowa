/**
 * liblinear_interface.hpp
 *
 * Convenience interface for dealing with LIBLINEAR.
 */
#ifndef LIBLINEAR_INTERFACE_IMPL_HPP
#define LIBLINEAR_INTERFACE_IMPL_HPP

namespace liblinear {

template<typename MatType>
feature_node***
MatToTransposedFeatureNodes(const MatType& X,
                            const size_t partitions)
{
  feature_node*** results = new feature_node**[partitions];
  const size_t batchSize = (X.n_cols + partitions - 1) / partitions;
  MatType Xt = X.t();
  for (size_t p = 0; p < partitions; ++p)
  {
    const size_t start = p * batchSize;
    const size_t end = ((p + 1) * batchSize >= X.n_cols) ? X.n_cols :
        (p + 1) * batchSize;

    // The L1R_LR solver will want data in a transposed form.
    feature_node** result = new feature_node*[Xt.n_cols];
    #pragma omp parallel for
    for (size_t i = 0; i < Xt.n_cols; ++i)
    {
      result[i] = ColSubsetToFeatureNode(Xt, i, start, end);
    }

    results[p] = result;
  }

  return results;
}

template<typename MatType>
feature_node***
SubsampleMatToTransposedFeatureNodes(const MatType& X,
                                     const size_t partitions,
                                     const double ratio)
{
  feature_node*** results = new feature_node**[partitions];
  const size_t batchSize = (X.n_cols + partitions - 1) / partitions;
  MatType Xt = X.t();
  for (size_t p = 0; p < partitions; ++p)
  {
    const size_t start = p * batchSize;
    size_t end = ((p + 1) * batchSize >= X.n_cols) ? X.n_cols :
        (p + 1) * batchSize;
    
    // shorten end to subsample by ratio
    const size_t n_sub = ratio * (end - start);
    end = start + n_sub;

    // The L1R_LR solver will want data in a transposed form.
    feature_node** result = new feature_node*[Xt.n_cols];
    #pragma omp parallel for
    for (size_t i = 0; i < Xt.n_cols; ++i)
    {
      result[i] = ColSubsetToFeatureNode(Xt, i, start, end);
    }

    results[p] = result;
  }

  return results;
}

template<typename MatType>
feature_node***
KMeansAugMatToTransposedFeatureNodes(const MatType& X,
                                     const arma::mat& centroids,
                                     const size_t partitions)
{
  feature_node*** results = new feature_node**[partitions];
  const size_t batchSize = (X.n_cols + partitions - 1) / partitions;
  MatType Xt = X.t();

  for (size_t p = 0; p < partitions; ++p)
  {
    const size_t start = p * batchSize;
    const size_t end = ((p + 1) * batchSize >= X.n_cols) ? X.n_cols :
        (p + 1) * batchSize;

    // The L1R_LR solver will want data in a transposed form.
    feature_node** result = new feature_node*[Xt.n_cols];
    #pragma omp parallel for
    for (size_t i = 0; i < Xt.n_cols; ++i)
    {
      result[i] = KMeansAugColSubsetToFeatureNode(Xt, centroids, i, start, end);
    }

    results[p] = result;
  }

  return results;
}

// Freeing this will be a little bit complicated!  Be sure to use the function
// written for this (below).
// Note that we return k + 1 sets of feature_nodes; the last one of these
// includes all the data.
template<typename MatType>
feature_node***
MatToKFoldFeatureNodes(const MatType& X,
                       const size_t k)
{
  feature_node*** results = new feature_node**[k + 1];
  const size_t batchSize = (X.n_cols + k - 1) / k;

  // First, we'll create a feature_node** for all the data.
  results[k] = new feature_node*[X.n_cols];
  for (size_t i = 0; i < X.n_cols; ++i)
  {
    results[k][i] = ColToFeatureNode(X, i);
  }

  // Now split that data into k folds.
  for (size_t i = 0; i < k; ++i)
  {
    // We'll skip this set of points for fold `i`.
    const size_t skipStart = i * batchSize;
    const size_t skipEnd = ((i + 1) * batchSize >= X.n_cols) ? X.n_cols :
        (i + 1) * batchSize;

    const size_t foldPoints = X.n_cols - (skipEnd - skipStart);
    results[i] = new feature_node*[foldPoints];
    size_t currentIndex = 0;
    for (size_t c = 0; c < X.n_cols; ++c)
    {
      if (c >= skipStart && c < skipEnd)
        continue;
      results[i][currentIndex] = results[k][c];
      ++currentIndex;
    }
  }

  return results;
}

template<typename eT>
feature_node* ColToFeatureNode(const arma::Mat<eT>& X,
                               const size_t col)
{
  // For a fully dense matrix we have to copy all elements.
  feature_node* c = new feature_node[X.n_rows + 1];
  size_t current_index = 0;
  for (size_t i = 0; i < X.n_rows; ++i)
  {
    // Skip zero elements...
    if (X(i, col) == eT(0))
      continue;

    c[current_index].index = i + 1; // dimensions start from 1
    c[current_index].value = (double) X(i, col);
    ++current_index;
  }
  c[current_index].index = -1;
  c[current_index].value = 0.0;

  if (current_index != X.n_rows)
  {
    // We can shrink the result.
    feature_node* c2 = new feature_node[current_index + 1];
    for (size_t i = 0; i <= current_index; ++i)
    {
      c2[i].index = c[i].index;
      c2[i].value = c[i].value;
    }

    delete[] c;
    c = c2;
  }

  return c;
}

template<typename eT>
feature_node* ColToFeatureNode(const arma::SpMat<eT>& X,
                               const size_t col)
{
  // For a sparse matrix we need to copy only nonzero elements.
  arma::sp_mat::const_iterator it = X.begin_col(col);
  size_t subset_nnz = 0;
  while (it != X.end_col(col))
  {
    ++subset_nnz;
    ++it;
  }

  feature_node* c = new feature_node[subset_nnz + 1];
  size_t index = 0;
  it = X.begin_col(col);
  while (it != X.end_col(col))
  {
    c[index].index = it.row() + 1; // dimensions start from 1
    c[index].value = (*it);
    ++index;
    ++it;
  }
  c[subset_nnz].index = -1;
  c[subset_nnz].value = 0.0;

  return c;
}

template<typename eT>
feature_node* ColSubsetToFeatureNode(const arma::Mat<eT>& Xt,
                                     const size_t col,
                                     const size_t startRow,
                                     const size_t endRow)
{
  // For a fully dense matrix we have to copy all elements.
  const size_t len = endRow - startRow + 1;
  feature_node* c = new feature_node[len];
  size_t current_index = 0;
  for (size_t i = startRow; i < endRow; ++i)
  {
    // Skip zero elements...
    if (Xt(i, col) == eT(0))
      continue;

    c[current_index].index = (i - startRow) + 1; // dimensions start from 1
    c[current_index].value = (double) Xt(i, col);
    ++current_index;
  }
  c[current_index].index = -1;
  c[current_index].value = 0.0;

  if (current_index != len)
  {
    // We can shrink the result.
    feature_node* c2 = new feature_node[current_index + 1];
    for (size_t i = 0; i <= current_index; ++i)
    {
      c2[i].index = c[i].index;
      c2[i].value = c[i].value;
    }

    delete[] c;
    c = c2;
  }

  return c;
}

template<typename eT>
feature_node* ColSubsetToFeatureNode(const arma::SpMat<eT>& Xt,
                                     const size_t col,
                                     const size_t startRow,
                                     const size_t endRow)
{
  // For a sparse matrix we need to copy only nonzero elements.
  arma::sp_mat::const_iterator it = Xt.begin_col(col);
  size_t subset_nnz = 0;
  while (it != Xt.end_col(col))
  {
    if (it.row() >= startRow && it.row() < endRow)
    {
      ++subset_nnz;
    }
    else if (it.row() >= endRow)
    {
      break;
    }
    ++it;
  }

  feature_node* c = new feature_node[subset_nnz + 1];
  size_t index = 0;
  it = Xt.begin_col(col);
  while (it != Xt.end_col(col))
  {
    if (it.row() >= startRow && it.row() < endRow)
    {
      c[index].index = (it.row() - startRow) + 1; // dimensions start from 1
      c[index].value = (*it);
      ++index;
    }
    else if (it.row() >= endRow)
    {
      break;
    }
    ++it;
  }
  c[subset_nnz].index = -1;
  c[subset_nnz].value = 0.0;

  return c;
}

template<typename eT>
feature_node* KMeansAugColSubsetToFeatureNode(const arma::Mat<eT>& Xt,
                                              const arma::mat& centroids,
                                              const size_t col,
                                              const size_t startRow,
                                              const size_t endRow)
{
  // For a fully dense matrix we have to copy all elements.
  const size_t len = endRow - startRow + 1 + centroids.n_cols;
  feature_node* c = new feature_node[len];
  size_t current_index = 0;
  for (size_t i = startRow; i < endRow; ++i)
  {
    // Skip zero elements...
    if (Xt(i, col) == eT(0))
      continue;

    c[current_index].index = (i - startRow) + 1; // dimensions start from 1
    c[current_index].value = (double) Xt(i, col);
    ++current_index;
  }

  // Now augment with all centroids.
  for (size_t i = 0; i < centroids.n_cols; ++i)
  {
    // Note that centroids has not been transposed, but Xt has!
    if (centroids(col, i) == eT(0))
      continue;

    c[current_index].index = (endRow - startRow) + i + 1;
    c[current_index].value = (double) centroids(col, i);
    ++current_index;
  }

  c[current_index].index = -1;
  c[current_index].value = 0.0;

  if (current_index != len)
  {
    // We can shrink the result.
    feature_node* c2 = new feature_node[current_index + 1];
    for (size_t i = 0; i <= current_index; ++i)
    {
      c2[i].index = c[i].index;
      c2[i].value = c[i].value;
    }

    delete[] c;
    c = c2;
  }

  return c;
}

template<typename eT>
feature_node* KMeansAugColSubsetToFeatureNode(const arma::SpMat<eT>& Xt,
                                              const arma::mat& centroids,
                                              const size_t col,
                                              const size_t startRow,
                                              const size_t endRow)
{
  // For a sparse matrix we need to copy only nonzero elements.
  arma::sp_mat::const_iterator it = Xt.begin_col(col);
  size_t subset_nnz = 0;
  while (it != Xt.end_col(col))
  {
    if (it.row() >= startRow && it.row() < endRow)
    {
      ++subset_nnz;
    }
    else if (it.row() >= endRow)
    {
      break;
    }
    ++it;
  }

  // Also count nonzeros in the centroids.
  for (size_t i = 0; i < centroids.n_cols; ++i)
  {
    if (centroids(col, i) != eT(0))
      ++subset_nnz;
  }

  feature_node* c = new feature_node[subset_nnz + 1];
  size_t index = 0;
  it = Xt.begin_col(col);
  while (it != Xt.end_col(col))
  {
    if (it.row() >= startRow && it.row() < endRow)
    {
      c[index].index = (it.row() - startRow) + 1; // dimensions start from 1
      c[index].value = (*it);
      ++index;
    }
    else if (it.row() >= endRow)
    {
      break;
    }
    ++it;
  }

  // Now also add the centroid values.
  for (size_t i = 0; i < centroids.n_cols; ++i)
  {
    if (centroids(col, i) != eT(0))
    {
      c[index].index = (endRow - startRow) + i + 1;
      c[index].value = (double) centroids(col, i);
      ++index;
    }
  }

  c[subset_nnz].index = -1;
  c[subset_nnz].value = 0.0;

  return c;
}

inline void Train(const feature_node** data,
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
                  const int seed)
{
  const double C = (1.0 / (l * lambda));

  // Set up LIBLINEAR parameters.
  parameter param;
  param.solver_type = L1R_LR;
  param.eps = epsilon;
  param.C = C; // not actually used internally
  param.nr_weight = 0;
  param.weight_label = NULL;
  param.weight = NULL;
  param.p = 0;
  param.nu = 0.0;
  param.init_sol = NULL;
  param.regularize_bias = 0;
  param.max_outer_iter = max_outer_iter;
  param.max_inner_iter = max_inner_iter;
  param.verbose = (verbose ? 1 : 0);
  param.seed = seed;

  problem prob;
  prob.l = l;
  prob.n = n;
  prob.y = (double*) responses;
  prob.x = (feature_node**) data;
  prob.bias = -1.0;
  prob.W = nullptr;

  // We assume the model is of the correct size already.
  if (l1_regularization)
    (void) solve_l1r_lr(&prob, &param, model, C, C, epsilon);
  else
    (void) solve_l2r_lr_dual(&prob, &param, model, C, C);
}

inline void Train(const feature_node** data,
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
                  const int seed)
{
  const double C = (1.0 / (l * lambda));

  // Set up LIBLINEAR parameters.
  parameter param;
  param.solver_type = L1R_LR;
  param.eps = epsilon;
  param.C = C; // not actually used internally...
  param.nr_weight = 0;
  param.weight_label = NULL;
  param.weight = NULL;
  param.p = 0;
  param.nu = 0.0;
  param.init_sol = NULL;
  param.regularize_bias = 0;
  param.max_outer_iter = max_outer_iter;
  param.max_inner_iter = max_inner_iter;
  param.verbose = (verbose ? 1 : 0);
  param.seed = seed;

  problem prob;
  prob.l = l;
  prob.n = n;
  prob.y = (double*) responses;
  prob.x = (feature_node**) data;
  prob.bias = -1.0;
  prob.W = (double*) weights;

  // We assume the model is of the correct size already.
  if (l1_regularization && !feature_weights)
    (void) solve_l1r_lr_weighted(&prob, &param, model, C, C, epsilon);
  else if (l1_regularization && feature_weights)
    (void) solve_l1r_lr_feature_weighted(&prob, &param, model, C, C, epsilon);
  else
    throw std::invalid_argument("l2 regularization not supported for weighted "
        "training");
}

inline void CleanFeatureNode(feature_node**& n, size_t l)
{
  for (size_t i = 0; i < l; ++i)
    delete[] n[i];
  delete[] n;
  n = NULL;
}

inline void CleanKFoldFeatureNodes(feature_node***& n, size_t k, size_t n_cols)
{
  // We can delete only the points in the last "dummy" fold (which contains all
  // points).
  for (size_t i = 0; i < n_cols; ++i)
    delete[] n[k][i];
  // Now delete all the pointers to each fold (including the last "dummy" fold).
  for (size_t i = 0; i <= k; ++i)
    delete[] n[i];
  delete[] n;
  n = NULL;
}

} // namespace liblinear

#endif
