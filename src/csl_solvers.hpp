#ifndef CSL_SOLVERS_HPP
#define CSL_SOLVERS_HPP

#include "liblinear/linear.hpp"
#include <armadillo>
#include <stdio.h>
#include <lbfgs.h>
#include <random>

namespace liblinear{

// safe implementation of log(1 + exp(x))
double log1p_exp(double s);

// objective function for logistic regression with l1-reg
double lr_obj(const feature_node** data,
              const size_t l,
              const size_t n,
              const double* responses,
              double* w,
              double lambd,
              bool verbose = false);

// csl objective function
double csl_obj(const feature_node** data,
               const size_t l,
               const size_t n,
               const double* responses,
               double* w,
               double* igrad,
               double* ugrad,
               double lambd,
               bool verbose = false);

// csl objective function with shift regularization
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
               bool verbose = false);

// computes gradient of unregularized lr obj
void lr_gradient(const feature_node** data,
                 const size_t l,
                 const size_t n,
                 const double* responses,
                 double* w,
                 double* grad,
                 bool normalize = true,
                 size_t* active_dims = nullptr,
                 size_t active_size = -1);

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
                     double alpha = 0.0,
                     bool verbose = false);
}

#include "csl_solvers_impl.hpp"

#endif
