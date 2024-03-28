#ifndef LIBLINEAR_L1R_LR_HPP
#define LIBLINEAR_L1R_LR_HPP

#include "linear.hpp"
#include <random>

// A coordinate descent algorithm for
// L1-regularized logistic regression problems
//
//  min_w \sum |wj| + C \sum log(1+exp(-yi w^T xi)),
//
// Given:
// x, y, Cp, Cn
// eps is the stopping tolerance
//
// solution will be put in w
//
// this function returns the number of iterations
//
// See Yuan et al. (2011) and appendix of LIBLINEAR paper, Fan et al. (2008)
//
// To not regularize the bias (i.e., regularize_bias = 0), a constant feature = 1
// must have been added to the original data. (see -B and -R option)

// This version is adapted from the original LIBLINEAR:
//   * Instance weight support is removed.
//   * The maximum number of inner iterations is configurable.
//   * Use local RNG to avoid glibc rand() global lock..

inline int solve_l1r_lr(const problem *prob_col, const parameter *param, double *w, double Cp, double Cn, double eps)
{
  int l = prob_col->l;
  int w_size = prob_col->n;
  int regularize_bias = param->regularize_bias;
  int j, s, newton_iter=0, iter=0;
  int max_newton_iter = param->max_outer_iter;
  int max_iter = param->max_inner_iter;
  int max_num_linesearch = 20;
  int active_size;
  int QP_active_size;
  int verbose = param->verbose;
  int seed = param->seed;

  double nu = 1e-12;
  double inner_eps = 1;
  double sigma = 0.01;
  double w_norm, w_norm_new;
  double z, G, H;
  double Gnorm1_init = -1.0; // Gnorm1_init is initialized at the first iteration
  double Gmax_old = INF;
  double Gmax_new, Gnorm1_new;
  double QP_Gmax_old = INF;
  double QP_Gmax_new, QP_Gnorm1_new;
  double delta, negsum_xTd, cond;

  int *index = new int[w_size];
  schar *y = new schar[l];
  double *Hdiag = new double[w_size];
  double *Grad = new double[w_size];
  double *wpd = new double[w_size];
  double *xjneg_sum = new double[w_size];
  double *xTd = new double[l];
  double *exp_wTx = new double[l];
  double *exp_wTx_new = new double[l];
  double *tau = new double[l];
  double *D = new double[l];
  feature_node *x;

  int *nnz_per_feature = new int[w_size];

  // Create a thread-local RNG for the optimization.
  std::mt19937 rng_gen(seed);
  std::uniform_int_distribution<int> rng(0, INT_MAX);

  double C[3] = {Cn,0,Cp};

  // Initial w can be set here.
  for(j=0; j<w_size; j++)
    w[j] = 0;

  for(j=0; j<l; j++)
  {
    if(prob_col->y[j] > 0)
      y[j] = 1;
    else
      y[j] = -1;

    exp_wTx[j] = 0;
  }

  w_norm = 0;
  for(j=0; j<w_size; j++)
  {
    w_norm += fabs(w[j]);
    wpd[j] = w[j];
    index[j] = j;
    xjneg_sum[j] = 0;
    x = prob_col->x[j];
    nnz_per_feature[j] = 0;
    while(x->index != -1)
    {
      int ind = x->index-1;
      double val = x->value;
      exp_wTx[ind] += w[j]*val;
      if(y[ind] == -1)
        xjneg_sum[j] += C[y[ind] + 1]*val;
      nnz_per_feature[j]++;
      x++;
    }
  }
  if (regularize_bias == 0)
    w_norm -= fabs(w[w_size-1]);

  for(j=0; j<l; j++)
  {
    exp_wTx[j] = exp(exp_wTx[j]);
    double tau_tmp = 1/(1+exp_wTx[j]);
    tau[j] = C[y[j] + 1]*tau_tmp;
    D[j] = C[y[j] + 1]*exp_wTx[j]*tau_tmp*tau_tmp;
  }

  while(newton_iter < max_newton_iter)
  {
    Gmax_new = 0;
    Gnorm1_new = 0;
    active_size = w_size;

    for(s=0; s<active_size; s++)
    {
      j = index[s];
      Hdiag[j] = nu;
      Grad[j] = 0;

      double tmp = 0;
      x = prob_col->x[j];
      while(x->index != -1)
      {
        int ind = x->index-1;
        Hdiag[j] += x->value*x->value*D[ind];
        tmp += x->value*tau[ind];
        x++;
      }
      Grad[j] = -tmp + xjneg_sum[j];

      double violation = 0;
      if (j == w_size-1 && regularize_bias == 0)
        violation = fabs(Grad[j]);
      else
      {
        double Gp = Grad[j]+1;
        double Gn = Grad[j]-1;
        if(w[j] == 0)
        {
          if(Gp < 0)
            violation = -Gp;
          else if(Gn > 0)
            violation = Gn;
          //outer-level shrinking
          else if(Gp>Gmax_old/l && Gn<-Gmax_old/l)
          {
            active_size--;
            std::swap(index[s], index[active_size]);
            s--;
            continue;
          }
        }
        else if(w[j] > 0)
          violation = fabs(Gp);
        else
          violation = fabs(Gn);
      }
      Gmax_new = std::max(Gmax_new, violation);
      Gnorm1_new += violation;
    }

    if(newton_iter == 0)
      Gnorm1_init = Gnorm1_new;

    if(Gnorm1_new <= eps*Gnorm1_init)
      break;

    iter = 0;
    QP_Gmax_old = INF;
    QP_active_size = active_size;

    for(int i=0; i<l; i++)
      xTd[i] = 0;

    // optimize QP over wpd
    while(iter < max_iter)
    {
      QP_Gmax_new = 0;
      QP_Gnorm1_new = 0;

      for(j=0; j<QP_active_size; j++)
      {
        int i = j+rng(rng_gen)%(QP_active_size-j);
        std::swap(index[i], index[j]);
      }

      for(s=0; s<QP_active_size; s++)
      {
        j = index[s];
        H = Hdiag[j];

        x = prob_col->x[j];
        G = Grad[j] + (wpd[j]-w[j])*nu;
        while(x->index != -1)
        {
          int ind = x->index-1;
          G += x->value*D[ind]*xTd[ind];
          x++;
        }

        double violation = 0;
        if (j == w_size-1 && regularize_bias == 0)
        {
          // bias term not shrunken
          violation = fabs(G);
          z = -G/H;
        }
        else
        {
          double Gp = G+1;
          double Gn = G-1;
          if(wpd[j] == 0)
          {
            if(Gp < 0)
              violation = -Gp;
            else if(Gn > 0)
              violation = Gn;
            //inner-level shrinking
            else if(Gp>QP_Gmax_old/l && Gn<-QP_Gmax_old/l)
            {
              QP_active_size--;
              std::swap(index[s], index[QP_active_size]);
              s--;
              continue;
            }
          }
          else if(wpd[j] > 0)
            violation = fabs(Gp);
          else
            violation = fabs(Gn);

          // obtain solution of one-variable problem
          if(Gp < H*wpd[j])
            z = -Gp/H;
          else if(Gn > H*wpd[j])
            z = -Gn/H;
          else
            z = -wpd[j];
        }
        QP_Gmax_new = std::max(QP_Gmax_new, violation);
        QP_Gnorm1_new += violation;

        if(fabs(z) < 1.0e-12)
          continue;
        z = std::min(std::max(z,-10.0),10.0);

        wpd[j] += z;

        x = prob_col->x[j];
        sparse_operator::axpy(z, x, xTd);
      }

      iter++;

      if(QP_Gnorm1_new <= inner_eps*Gnorm1_init)
      {
        //inner stopping
        if(QP_active_size == active_size)
          break;
        //active set reactivation
        else
        {
          QP_active_size = active_size;
          QP_Gmax_old = INF;
          continue;
        }
      }

      QP_Gmax_old = QP_Gmax_new;
    }

    if(iter >= max_iter && verbose == 1)
      info("WARNING: reaching max number of inner iterations\n");

    delta = 0;
    w_norm_new = 0;
    for(j=0; j<w_size; j++)
    {
      delta += Grad[j]*(wpd[j]-w[j]);
      if(wpd[j] != 0)
        w_norm_new += fabs(wpd[j]);
    }
    if (regularize_bias == 0)
      w_norm_new -= fabs(wpd[w_size-1]);
    delta += (w_norm_new-w_norm);

    negsum_xTd = 0;
    for(int i=0; i<l; i++)
      if(y[i] == -1)
        negsum_xTd += C[y[i] + 1]*xTd[i];

    int num_linesearch;
    for(num_linesearch=0; num_linesearch < max_num_linesearch; num_linesearch++)
    {
      cond = w_norm_new - w_norm + negsum_xTd - sigma*delta;

      for(int i=0; i<l; i++)
      {
        double exp_xTd = exp(xTd[i]);
        exp_wTx_new[i] = exp_wTx[i]*exp_xTd;
        cond += C[y[i] + 1]*log((1+exp_wTx_new[i])/(exp_xTd+exp_wTx_new[i]));
      }

      if(cond <= 0)
      {
        w_norm = w_norm_new;
        for(j=0; j<w_size; j++)
          w[j] = wpd[j];
        for(int i=0; i<l; i++)
        {
          exp_wTx[i] = exp_wTx_new[i];
          double tau_tmp = 1/(1+exp_wTx[i]);
          tau[i] = C[y[i] + 1]*tau_tmp;
          D[i] = C[y[i] + 1]*exp_wTx[i]*tau_tmp*tau_tmp;
        }
        break;
      }
      else
      {
        w_norm_new = 0;
        for(j=0; j<w_size; j++)
        {
          wpd[j] = (w[j]+wpd[j])*0.5;
          if(wpd[j] != 0)
            w_norm_new += fabs(wpd[j]);
        }
        if (regularize_bias == 0)
          w_norm_new -= fabs(wpd[w_size-1]);
        delta *= 0.5;
        negsum_xTd *= 0.5;
        for(int i=0; i<l; i++)
          xTd[i] *= 0.5;
      }
    }

    // Recompute some info due to too many line search steps
    if(num_linesearch >= max_num_linesearch)
    {
      for(int i=0; i<l; i++)
        exp_wTx[i] = 0;

      for(int i=0; i<w_size; i++)
      {
        if(w[i]==0) continue;
        x = prob_col->x[i];
        sparse_operator::axpy(w[i], x, exp_wTx);
      }

      for(int i=0; i<l; i++)
        exp_wTx[i] = exp(exp_wTx[i]);
    }

    if(iter == 1)
      inner_eps *= 0.25;

    newton_iter++;
    Gmax_old = Gmax_new;

    if (verbose == 1)
      info("iter %3d  #CD cycles %d\n", newton_iter, iter);
  }

  if (verbose == 1)
  {
    info("=========================\n");
    info("optimization finished, #iter = %d\n", newton_iter);
    if(newton_iter >= max_newton_iter)
      info("WARNING: reaching max number of iterations\n");
  }

  // calculate objective value

  double v = 0;
  int nnz = 0;
  for(j=0; j<w_size; j++)
    if(w[j] != 0)
    {
      v += fabs(w[j]);
      nnz++;
    }
  if (regularize_bias == 0)
    v -= fabs(w[w_size-1]);
  for(j=0; j<l; j++)
    if(y[j] == 1)
      v += C[y[j] + 1]*log(1+1/exp_wTx[j]);
    else
      v += C[y[j] + 1]*log(1+exp_wTx[j]);

  if (verbose == 1)
  {
    info("Objective value = %lf\n", v);
    info("#nonzeros/#features = %d/%d\n", nnz, w_size);
  }

  delete [] index;
  delete [] y;
  delete [] Hdiag;
  delete [] Grad;
  delete [] wpd;
  delete [] xjneg_sum;
  delete [] xTd;
  delete [] exp_wTx;
  delete [] exp_wTx_new;
  delete [] tau;
  delete [] D;
  delete [] nnz_per_feature;

  return newton_iter;
}

#endif
