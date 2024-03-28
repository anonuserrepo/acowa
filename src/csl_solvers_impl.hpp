#ifndef CSL_SOLVERS_IMPL_HPP
#define CSL_SOLVERS_IMPL_HPP

#include "csl_solvers.hpp"
#include <iostream>
#include <stdio.h>
#include <lbfgs.h>

namespace liblinear{


double log1p_exp(double s)
{
  double lls;
  if (s < 0)
  {
    lls = log(1 + exp(s));
  }
  else
  {
    // avoid overflow
    lls = s + log(1 + exp(-s));
  }
  return lls;
}

// logistic loss with L1 regularization
double lr_obj(const feature_node** data,
              const size_t l,
              const size_t n,
              const double* responses,
              double* w,
              double lambd,
              bool verbose)
{
  feature_node** X = (feature_node**) data;
  double* y = (double*) responses;
  feature_node* xj;

  double* Xtw = new double[l];

  for (size_t i = 0; i < l; ++i)
  {
     Xtw[i] = 0;
  }

  for (size_t j = 0; j < n; ++j)
  {
    if (w[j] == 0)
      continue;
    xj = X[j];
    while (xj->index != -1)
    {
      Xtw[xj->index-1] += w[j] * xj->value;
      xj++;
    }
  }

  double obj = 0;
  for (size_t i = 0; i < l; ++i)
  {
    obj += log1p_exp(Xtw[i]);
    if (y[i] == 1)
      obj += -1.0 * Xtw[i];
  }
  obj /= l;

  double wnorm = 0;
  for (size_t j = 0; j < n; ++j)
  {
    wnorm += fabs(w[j]);
  }

  if (verbose)
  {
    std::cout << "--> lr loss: " << obj << std::endl;
    std::cout << "--> reg loss: " << lambd * wnorm << std::endl;
  }

  delete [] Xtw;

  return obj + lambd * wnorm;
}

double csl_obj(const feature_node** data,
               const size_t l,
               const size_t n,
               const double* responses,
               double* w,
               double* igrad,
               double* ugrad,
               double lambd,
               bool verbose)
{
  double obj = lr_obj(data, l, n, responses, w, lambd, verbose);

  // compute gradient diff term here. Grads must be pre-normalized
  double gtw = 0;
  for (size_t j = 0; j < n; ++j)
  {
    gtw += (ugrad[j] - igrad[j]) * w[j];
  }

  if (verbose)
    std::cout << "--> csl loss: " << gtw << std::endl;

  return obj + gtw;
}

// overload for proximal regularized CSL
double csl_obj(const feature_node** data,
               const size_t l,
               const size_t n,
               const double* responses,
               double* w,
               double* igrad,
               double* ugrad,
               double lambd,
               double alpha,
               double* w_prev,
               bool verbose)
{
  double obj = lr_obj(data, l, n, responses, w, lambd, verbose);

  // compute gradient diff term here. Grads must be pre-normalized
  double gtw = 0;
  for (size_t j = 0; j < n; ++j)
  {
    gtw += (ugrad[j] - igrad[j]) * w[j];
  }

  double ad2 = 0;
  for (size_t j = 0; j < n; ++j)
  {
    ad2 += std::pow(w[j] - w_prev[j], 2);
  }
  ad2 *= 0.5 * alpha;

  if (verbose)
  {
    std::cout << "--> csl loss: " << gtw << std::endl;
    std::cout << "--> csl reg loss: " << ad2 << std::endl;
  }

  return obj + gtw + ad2;
}


struct lbfgsInfo {
    feature_node** X;
    double* y;
    size_t l;
    size_t n;
    double* igrad;
    double* ugrad;
    double lambda;
    size_t* active_dims;
    size_t active_size;
    double alpha;
    double* w_prev;
};

void lr_gradient(const feature_node** data,
                 const size_t l,
                 const size_t n,
                 const double* responses,
                 double* w,
                 double* grad,
                 bool normalize,
                 size_t* active_dims,
                 size_t active_size)
{
  double* resid = new double[l];
  feature_node** X = (feature_node**) data;
  double* y = (double*) responses;
  feature_node* xj;

  // handle feature subsetting, default to all if nullptr passed
  if (active_dims == nullptr)
    active_size = n;  

  // init residual vector to 0s
  for (size_t i = 0; i < l; ++i)
  {
     resid[i] = 0;
  }

  // init grad to 0 (since not all will be assigned)
  for (size_t j = 0; j < n; ++j)
  {
    grad[j] = 0;
  }

  // compute x dot w
  for (size_t s = 0; s < active_size; ++s)
  {
    size_t j = (active_dims == nullptr) ? s : active_dims[s];
    xj = X[j];
    // each linked list xj stores a feature column so we need transpose of standard dot fn
    while (xj->index != -1)
    {
      resid[xj->index-1] += w[j] * xj->value;
      xj++;
    }
  }

  for (size_t i = 0; i < l; ++i)
  {
    // convert x*w to logits
    resid[i] = 1 / (1 + exp(-resid[i]));
    
    // // we don't care if y is {0, 1} or {-1, 1}, just if y=1
    if (y[i] == 1)
      resid[i] -= 1;
  }

  // compute xj^t residual and normalize by sample size
  for (size_t s = 0; s < active_size; ++s)
  {
    size_t j = (active_dims == nullptr) ? s : active_dims[s];
    xj = X[j];
    grad[j] = sparse_operator::dot(resid, xj);
    if (normalize)
       grad[j] /= l;
  }

  delete [] resid;
}


// Obj + Gradient function for OWLQN CSL 
static lbfgsfloatval_t evaluate(void *instance,
                                const lbfgsfloatval_t *x,
                                lbfgsfloatval_t *g,
                                const int n,
                                const lbfgsfloatval_t step)
{
  // struct lbfgsInfo* info = static_cast<lbfgsInfo *>(instance);
  struct lbfgsInfo* info = (lbfgsInfo*) instance;

  double* w = new double[n];
  for (size_t j = 0; j < n; ++j)
  {
    w[j] = x[j];
  }

  double obj = csl_obj((const feature_node**) info->X,
                       (const size_t) info->l,
                       (const size_t) info->n,
                       (const double*) info->y,
                       w,
                       info->igrad,
                       info->ugrad,
                       info->lambda,
                       info->alpha,
                       info->w_prev);

  // all grads should be normalized. ugrad and igrad already normalized
  double* grad = new double[n];
  lr_gradient((const feature_node**) info->X,
              (const size_t) info->l,
              (const size_t) info->n,
              (const double*) info->y,
              w,
              grad,
              true,   // normalize
              info->active_dims,
              info->active_size);

  size_t active_size = (info->active_dims == nullptr) ? info->n : info->active_size;
  for (size_t s = 0; s < active_size; ++s)
  {
    size_t j = (info->active_dims == nullptr) ? s : info->active_dims[s];
    grad[j] += info->ugrad[j] - info->igrad[j];   // add CSL grad diff term

    // optional: alpha shift shrinkage
    grad[j] += info->alpha * (w[j] - info->w_prev[j]);
  }

  for (size_t j = 0; j < n; ++j)
  {
    g[j] = grad[j];
  }

  delete [] w;
  delete [] grad;
  
  return obj;
}

void solve_csl_owlqn(const feature_node** data,
                     const size_t l,
                     const size_t n,
                     const double* responses,
                     double* w,
                     double* igrad,
                     double* ugrad,
                     double lambda,
                     size_t max_itr,
                     size_t* active_dims,
                     size_t active_size,
                     double alpha,
                     bool verbose)
{
  if (verbose)
  {
    std::cout << "==== Solving CSL objective with OWL-QN ====" << std::endl;
  
    // print out starting objective info
    double cslLoss = csl_obj(data, l, n, responses, w, igrad, ugrad, lambda, alpha, w);
    std::cout << "CSL starting objective: " << cslLoss << std::endl;

    double lrLoss = lr_obj(data, l, n, responses, w, lambda);
    std::cout << "local lr starting objective: " << lrLoss << std::endl;
  }

  // make copy of w for proximal alpha regularization
  double* w_prev = new double[n];
  for (size_t j = 0; j < n; ++j)
  {
    w_prev[j] = w[j];
  }

  lbfgsfloatval_t fx;
  lbfgsfloatval_t *x = lbfgs_malloc(n);
  lbfgs_parameter_t param;
  lbfgs_parameter_init(&param);
  param.max_iterations = (int) max_itr;   // set max iter
  param.orthantwise_c = lambda;     // set L1 regularization
  param.linesearch = 2;     // OWL-QN requires Armijo backtracking
  param.orthantwise_start = 0;
  param.orthantwise_end = n;

  struct lbfgsInfo objInfo;
  objInfo.X = (feature_node**) data;
  objInfo.y = (double*) responses;
  objInfo.l = l;
  objInfo.n = n;
  objInfo.igrad = igrad;
  objInfo.ugrad = ugrad;
  objInfo.lambda = lambda;
  objInfo.active_dims = active_dims;
  objInfo.active_size = active_size;
  objInfo.alpha = alpha;
  objInfo.w_prev = w_prev;

  int status = lbfgs(n, x, &fx, evaluate, nullptr, &objInfo, &param);

  for (size_t j = 0; j < n; ++j)
  {
    w[j] = x[j];
  }

  if (verbose)
  {
    // print ending objectives
    double cslLoss = csl_obj(data, l, n, responses, w, igrad, ugrad, lambda, alpha, w_prev);
    std::cout << "CSL ending objective: " << cslLoss << std::endl;

    double lrLoss = lr_obj(data, l, n, responses, w, lambda);
    std::cout << "local lr ending objective: " << lrLoss << std::endl;
  }

  delete [] w_prev;
}

}

#endif
