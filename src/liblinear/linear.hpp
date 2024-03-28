#ifndef LIBLINEAR_LINEAR_HPP
#define LIBLINEAR_LINEAR_HPP

#define LIBLINEAR_VERSION 246

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <locale.h>

namespace liblinear {

struct feature_node
{
  int index;
  double value;
};

struct problem
{
  int l, n;
  double *y;
  struct feature_node **x;
  double bias;            /* < 0 if no bias term */
  double *W; /* instance weights */
};

// Mostly unused in this adaptation...
enum
{
  L2R_LR,
  L2R_L2LOSS_SVC_DUAL,
  L2R_L2LOSS_SVC,
  L2R_L1LOSS_SVC_DUAL,
  MCSVM_CS,
  L1R_L2LOSS_SVC,
  L1R_LR,
  L2R_LR_DUAL,
  L2R_L2LOSS_SVR = 11,
  L2R_L2LOSS_SVR_DUAL,
  L2R_L1LOSS_SVR_DUAL,
  ONECLASS_SVM = 21
}; /* solver_type */

struct parameter
{
  int solver_type;

  /* these are for training only */
  double eps;             /* stopping tolerance */
  double C;
  int nr_weight;
  int *weight_label;
  double* weight;
  double p;
  double nu;
  double *init_sol;
  int regularize_bias;
  int max_outer_iter;
  int max_inner_iter;
  int verbose;
  int seed;
};

struct model
{
  struct parameter param;
  int nr_class;           /* number of classes */
  int nr_feature;
  double *w;
  int *label;             /* label of each class */
  double bias;
  double rho;             /* one-class SVM only */
};

typedef signed char schar;

#define INF HUGE_VAL

static inline void print_string_stdout(const char *s)
{
  fputs(s,stdout);
  fflush(stdout);
}

static inline void info(const char *fmt,...)
{
  char buf[BUFSIZ];
  va_list ap;
  va_start(ap,fmt);
  vsprintf(buf,fmt,ap);
  va_end(ap);
  print_string_stdout(buf);
}

class sparse_operator
{
public:
  static inline double nrm2_sq(const feature_node *x)
  {
    double ret = 0;
    while(x->index != -1)
    {
      ret += x->value*x->value;
      x++;
    }
    return ret;
  }

  static inline double dot(const double *s, const feature_node *x)
  {
    double ret = 0;
    while(x->index != -1)
    {
      ret += s[x->index-1]*x->value;
      x++;
    }
    return ret;
  }

  static inline double sparse_dot(const feature_node *x1, const feature_node *x2)
  {
    double ret = 0;
    while(x1->index != -1 && x2->index != -1)
    {
      if(x1->index == x2->index)
      {
        ret += x1->value * x2->value;
        ++x1;
        ++x2;
      }
      else
      {
        if(x1->index > x2->index)
          ++x2;
        else
          ++x1;
      }
    }
    return ret;
  }

  static inline void axpy(const double a, const feature_node *x, double *y)
  {
    while(x->index != -1)
    {
      y[x->index-1] += a*x->value;
      x++;
    }
  }

  // with a weight on the feature
  static inline void waxpy(const double a,
                           const feature_node* x,
                           const double w,
                           double* y)
  {
    while(x->index != -1)
    {
      y[x->index-1] += a*w*x->value;
      x++;
    }
  }
};

// function definitions

int solve_l1r_lr(const problem* prob_col, // must be transposed
                 const parameter* param,
                 double* w,
                 double Cp,
                 double Cn,
                 double eps);

int solve_l1r_lr_weighted(const problem* prob_col, // must be transposed
                          const parameter* param,
                          double* w,
                          double Cp,
                          double Cn,
                          double eps);

int solve_l1r_lr_feature_weighted(const problem* prob_col, // must be transposed
                                  const parameter* param,
                                  double* w,
                                  double Cp,
                                  double Cn,
                                  double eps);

int solve_l2r_lr_dual(const problem* prob_col, // not transposed
                      const parameter* param,
                      double* w,
                      double Cp,
                      double Cn);

// include all necessary components now

#include "l1r_lr.hpp"
#include "l1r_lr_weighted.hpp"
#include "l1r_lr_feature_weighted.hpp"
#include "l2r_lr.hpp"

} // namsepace liblinear

#endif
