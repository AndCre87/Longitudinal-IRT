#ifndef SAMPLE_BNP_FIXT0I0_H
#define SAMPLE_BNP_FIXT0I0_H

#include <RcppArmadillo.h>
using namespace Rcpp;

typedef std::tuple< arma::vec, arma::vec, double, arma::mat, arma::cube > sample_BNP_biv_t;

sample_BNP_biv_t sample_BNP_biv(List Y_bis, arma::mat Z, arma::mat B_Z, arma::cube theta, arma::vec alpha_Y, arma::mat beta_Y, arma::mat X_ZY, arma::vec U_Z, arma::mat U_Y, arma::mat s, arma::vec nj, double K_N, bool isDP, double kappa, double sigma, double u, arma::mat b_Z_star, arma::cube theta0_star, arma::vec m_b_Z, arma::mat Sigma_b_Z, arma::mat Omega_b_Z, arma::mat m_theta0, arma::cube Sigma_theta0, double sig2_Z, double sig2_Y, arma::vec subscales_indices_main);

#endif