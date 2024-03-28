#ifndef L2R_LR_HPP
#define L2R_LR_HPP

#include "linear.hpp"
#include <random>

// A coordinate descent algorithm for
// the dual of L2-regularized logistic regression problems
//
//  min_\alpha  0.5(\alpha^T Q \alpha) + \sum \alpha_i log (\alpha_i) + (upper_bound_i - \alpha_i) log (upper_bound_i - \alpha_i),
//    s.t.      0 <= \alpha_i <= upper_bound_i,
//
//  where Qij = yi yj xi^T xj and
//  upper_bound_i = Cp if y_i = 1
//  upper_bound_i = Cn if y_i = -1
//
// Given:
// x, y, Cp, Cn
// eps is the stopping tolerance
//
// solution will be put in w
//
// this function returns the number of iterations
//
// See Algorithm 5 of Yu et al., MLJ 2010

// This version is adapted from the original LIBLINEAR:
//   * Instance weight support is removed.
//   * The maximum number of inner iterations is configurable.
//   * Use local RNG to avoid glibc rand() global lock..

inline int solve_l2r_lr_dual(const problem *prob, const parameter *param, double *w, double Cp, double Cn)
{
  int l = prob->l;
  int w_size = prob->n;
  double eps = param->eps;
  int i, s, iter = 0;
  double *xTx = new double[l];
  int *index = new int[l];
  double *alpha = new double[2*l]; // store alpha and C - alpha
  schar *y = new schar[l];
  int max_inner_iter = param->max_inner_iter;
  int max_iter = param->max_outer_iter;
  double innereps = 1e-2;
  double innereps_min = std::min(1e-8, eps);
  double upper_bound[3] = {Cn, 0, Cp};
  int verbose = param->verbose;
  int seed = param->seed;

  // Create a thread-local RNG for the optimization.
  std::mt19937 rng_gen(seed);
  std::uniform_int_distribution<int> rng(0, INT_MAX);

  for(i=0; i<l; i++)
  {
    if(prob->y[i] > 0)
    {
      y[i] = +1;
    }
    else
    {
      y[i] = -1;
    }
  }

  // Initial alpha can be set here. Note that
  // 0 < alpha[i] < upper_bound[GETI(i)]
  // alpha[2*i] + alpha[2*i+1] = upper_bound[GETI(i)]
  for(i=0; i<l; i++)
  {
    alpha[2*i] = std::min(0.001*upper_bound[y[i] + 1], 1e-8);
    alpha[2*i+1] = upper_bound[y[i] + 1] - alpha[2*i];
  }

  for(i=0; i<w_size; i++)
    w[i] = 0;
  for(i=0; i<l; i++)
  {
    feature_node * const xi = prob->x[i];
    xTx[i] = sparse_operator::nrm2_sq(xi);
    sparse_operator::axpy(y[i]*alpha[2*i], xi, w);
    index[i] = i;
  }

  while (iter < max_iter)
  {
    for (i=0; i<l; i++)
    {
      int j = i+rng(rng_gen)%(l-i);
      std::swap(index[i], index[j]);
    }
    int newton_iter = 0;
    double Gmax = 0;
    for (s=0; s<l; s++)
    {
      i = index[s];
      const schar yi = y[i];
      double C = upper_bound[y[i] + 1];
      double ywTx = 0, xisq = xTx[i];
      feature_node * const xi = prob->x[i];
      ywTx = yi*sparse_operator::dot(w, xi);
      double a = xisq, b = ywTx;

      // Decide to minimize g_1(z) or g_2(z)
      int ind1 = 2*i, ind2 = 2*i+1, sign = 1;
      if(0.5*a*(alpha[ind2]-alpha[ind1])+b < 0)
      {
        ind1 = 2*i+1;
        ind2 = 2*i;
        sign = -1;
      }

      //  g_t(z) = z*log(z) + (C-z)*log(C-z) + 0.5a(z-alpha_old)^2 + sign*b(z-alpha_old)
      double alpha_old = alpha[ind1];
      double z = alpha_old;
      if(C - z < 0.5 * C)
        z = 0.1*z;
      double gp = a*(z-alpha_old)+sign*b+log(z/(C-z));
      Gmax = std::max(Gmax, fabs(gp));

      // Newton method on the sub-problem
      const double eta = 0.1; // xi in the paper
      int inner_iter = 0;
      while (inner_iter <= max_inner_iter)
      {
        if(fabs(gp) < innereps)
          break;
        double gpp = a + C/(C-z)/z;
        double tmpz = z - gp/gpp;
        if(tmpz <= 0)
          z *= eta;
        else // tmpz in (0, C)
          z = tmpz;
        gp = a*(z-alpha_old)+sign*b+log(z/(C-z));
        newton_iter++;
        inner_iter++;
      }

      if(inner_iter > 0) // update w
      {
        alpha[ind1] = z;
        alpha[ind2] = C-z;
        sparse_operator::axpy(sign*(z-alpha_old)*yi, xi, w);
      }
    }

    iter++;
    if(iter % 10 == 0 && verbose == 1)
      info(".");

    if(Gmax < eps)
      break;

    if(newton_iter <= l/10)
      innereps = std::max(innereps_min, 0.1*innereps);

  }

  if (verbose == 1)
    info("\noptimization finished, #iter = %d\n",iter);

  // calculate objective value

  double v = 0;
  for(i=0; i<w_size; i++)
    v += w[i] * w[i];
  v *= 0.5;
  for(i=0; i<l; i++)
    v += alpha[2*i] * log(alpha[2*i]) + alpha[2*i+1] * log(alpha[2*i+1])
      - upper_bound[y[i] + 1] * log(upper_bound[y[i] + 1]);
  if (verbose == 1)
    info("Objective value = %lf\n", v);

  delete [] xTx;
  delete [] alpha;
  delete [] y;
  delete [] index;

  return iter;
}

#endif
