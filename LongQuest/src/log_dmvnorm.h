#ifndef LOG_DMVNORM_H
#define LOG_DMVNORM_H

#include <RcppArmadillo.h>

double log_dmvnorm(arma::vec x, arma::vec mu, arma::mat Omega, bool is_prec);

#endif