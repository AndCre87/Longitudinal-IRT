// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "log_dmvnorm.h"

//This function takes Omega as the precision if is_prec = true (and precision is preferred to avoid inversion)
double log_dmvnorm(arma::vec x, arma::vec mu, arma::mat Omega, bool is_prec){
  double out = 0.0, sign = 1.0, log_det_Omega = 0.0, p = Omega.n_rows;
  arma::log_det(log_det_Omega, sign, Omega);
  
  if(is_prec){
    out = - 0.5 * ( p * M_LN_2PI - log_det_Omega + as_scalar((x.t() - mu.t()) * Omega * (x - mu)) );
  }else{
    out = - 0.5 * ( p * M_LN_2PI + log_det_Omega + as_scalar((x.t() - mu.t()) * arma::inv_sympd(Omega) * (x - mu)) );
  }
  return(out);
}

