// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <tuple>
#include "log_dmvnorm.h"
#include "sample_BNP_biv.h"
using namespace Rcpp;

sample_BNP_biv_t sample_BNP_biv(List Y_bis, arma::mat Z, arma::mat B_Z, arma::cube theta, arma::vec alpha_Y, arma::mat beta_Y, arma::mat X_ZY, arma::vec U_Z, arma::mat U_Y, arma::mat s, arma::vec nj, double K_N, bool isDP, double kappa, double sigma, double u, arma::mat b_Z_star, arma::cube theta0_star, arma::vec m_b_Z, arma::mat Sigma_b_Z, arma::mat Omega_b_Z, arma::mat m_theta0, arma::cube Sigma_theta0, double sig2_Z, double sig2_Y, arma::vec subscales_indices_main){
  
  // Extract sizes
  int N = Z.n_rows, nT_Z = Z.n_cols, nT_Y = Sigma_theta0.n_cols, n_subscales_main = Sigma_theta0.n_slices;
  //Number of auxiliary variables for Neal's Alg8, sample from P0
  int N_aux = 50;
  
  // Alg8: Renovate auxiliary variables and initialise vector of uniform probabilities
  arma::mat eye_mat_theta0(nT_Y, nT_Y, arma::fill::eye);
  arma::cube theta0_aux(N_aux, n_subscales_main, nT_Y, arma::fill::zeros);
  for(int is = 0; is < n_subscales_main; is++){
    for(int i_aux = 0; i_aux < N_aux; i_aux++){
      theta0_aux.tube(i_aux,is) = arma::mvnrnd(m_theta0.row(is).t(), Sigma_theta0.slice(is));
    }
  }
  arma::mat b_Z_aux = arma::mvnrnd(m_b_Z, Sigma_b_Z, N_aux).t();
  arma::vec prob_aux(N_aux, arma::fill::zeros);
  for(int i_aux = 0; i_aux < N_aux; i_aux++){
    prob_aux(i_aux) = 1.0 / N_aux;
  }
  arma::vec prob_aux_cum = arma::cumsum(prob_aux);
  
  // Update allocation of i-th subject
  for(int i = 0; i < N; i++){ 
    arma::vec Z_i = Z.row(i).t();
    
    int s_i = s(i);
    arma::rowvec X_ZY_i = X_ZY.row(i);
    
    // Remove element from count
    nj(s_i) = nj(s_i) -  1;
    // It could be the only element with j-th label
    bool alone_i = false; 
    if(nj(s_i) == 0){
      alone_i = true;
    }
    
    // Re-use algorithm
    if(alone_i){
      double aux_runif = arma::randu();
      arma::uvec hh_vec = arma::find(prob_aux_cum >= aux_runif, 1, "first");
      b_Z_aux.row(hh_vec(0)) = b_Z_star.row(s_i);
      theta0_aux.row(hh_vec(0)) = theta0_star.row(s_i);
    }
    
    arma::vec f_k(K_N + N_aux, arma::fill::zeros), w(K_N + N_aux, arma::fill::zeros);
    
    // Probability of allocating a new cluster
    for(int i_aux = 0; i_aux < N_aux; i_aux++){
      
      //SPlines part (longitudinal)
      for(int t = 0; t < nT_Z; t++){
        f_k(i_aux) += arma::log_normpdf(Z_i(t), arma::accu(B_Z.row(t) % b_Z_aux.row(i_aux)) + arma::accu(X_ZY_i % U_Z.t()), sqrt(sig2_Z));
      }
      
      //theta part (questionnaire)
      for(int is = 0; is < n_subscales_main; is++){
        for(int t = 0; t < nT_Y; t++){
          double m_theta_i_t = theta0_aux(i_aux,is,t) + arma::accu(X_ZY_i % U_Y.col(is).t());
          f_k(i_aux) += arma::log_normpdf(theta(i,is,t), m_theta_i_t, sqrt(sig2_Y));
        }
      }
    }
    
    // Probability of allocating to an existing cluster
    for(int j_aux = 0; j_aux < K_N; j_aux++){
      
      //SPlines part (longitudinal)
      for(int t = 0; t < nT_Z; t++){
        f_k(N_aux + j_aux) += arma::log_normpdf(Z_i(t), arma::accu(B_Z.row(t) % b_Z_star.row(j_aux)) + arma::accu(X_ZY_i % U_Z.t()), sqrt(sig2_Z));
      }
      
      //theta part (questionnaire)
      for(int is = 0; is < n_subscales_main; is++){
        for(int t = 0; t < nT_Y; t++){
          double m_theta_i_t = theta0_star(j_aux,is,t) + arma::accu(X_ZY_i % U_Y.col(is).t());
          f_k(N_aux + j_aux) += arma::log_normpdf(theta(i,is,t), m_theta_i_t, sqrt(sig2_Y));
        }
      }
    }
    
    
    if(isDP){
      w.head(N_aux).fill(kappa / N_aux);
      w.tail(K_N) = nj;
    }else{//NGG
      w.head(N_aux).fill(kappa * pow(1 + u, sigma) / N_aux); 
      w.tail(K_N) = nj - sigma;
    }
    // Check for empty cluster conditions
    if(alone_i){ // Fix negative weight
      w(N_aux + s_i) = 0.0;
    }
    
    f_k = exp(f_k-max(f_k)) % w;
    f_k = f_k/sum(f_k);
    
    double aux_runif = arma::randu();
    arma::vec f_k_cum = arma::cumsum(f_k);
    arma::uvec hh_vec = arma::find(f_k_cum >= aux_runif, 1, "first");
    int hh = hh_vec(0);
    
    if(hh < N_aux){ // New cluster
      
      // Select unique value from N_aux available
      if(alone_i){ // Same number of clusters
        nj(s_i) = 1;
        b_Z_star.row(s_i) = b_Z_aux.row(hh);
        theta0_star.row(s_i) = theta0_aux.row(hh);
      }else{ // Additional cluster
        nj.insert_rows(K_N,1);
        nj(K_N) = 1;
        b_Z_star.insert_rows(K_N,b_Z_aux.row(hh));
        theta0_star.insert_rows(K_N,theta0_aux.row(hh));
        s(i) = K_N; // Allocations s have indexing from 0!
        K_N ++;
      }
      //Restore used auxiliary variable
      b_Z_aux.row(hh) = arma::mvnrnd(m_b_Z, Sigma_b_Z).t();
      for(int is = 0; is < n_subscales_main; is++){
        theta0_aux.tube(hh,is) = arma::mvnrnd(m_theta0.row(is).t(), Sigma_theta0.slice(is));
      }
    }else{ // Old cluster
      
      int hk = hh - N_aux;
      nj(hk) ++;
      s(i) = hk;  // Allocations s have indexing from 0!
      if(alone_i){ // Remove empty cluster
        K_N --;
        nj.shed_row(s_i);
        b_Z_star.shed_row(s_i);
        theta0_star.shed_row(s_i);
        for(int i2 = 0; i2 < N; i2 ++){
          if(s(i2) > s_i){ // Allocations s have indexing from 0!
            s(i2) = s(i2) - 1;
          }
        }
      }
    }
  }
  
  
  //////////////////////////
  // Update unique values //
  //////////////////////////
  for(int k = 0; k < K_N; k++){
    
    ///////
    //b_Z//
    ///////        
    arma::mat Spost_b_Z = Omega_b_Z + nj(k) * B_Z.t() * B_Z / sig2_Z;
    arma::vec mpost_b_Z = Omega_b_Z * m_b_Z;
    for(int i = 0; i < N; i ++){
      //Cluster assignment of subject i
      int s_i = s(i);
      if(s_i == k){
        arma::vec mpost_aux(nT_Z);
        mpost_aux.fill(arma::accu(X_ZY.row(i) % U_Z.t()));
        mpost_b_Z += B_Z.t() * (Z.row(i).t() - mpost_aux) / sig2_Z;
      }
    }
    Spost_b_Z = arma::inv_sympd(Spost_b_Z);
    mpost_b_Z = Spost_b_Z * mpost_b_Z;
    
    b_Z_star.row(k) = arma::mvnrnd(mpost_b_Z, Spost_b_Z).t();
    
    
    //////////
    //theta0// conjugate
    //////////
    for(int is = 0; is < n_subscales_main; is++){
      
      arma::rowvec U_Y_is = U_Y.col(is).t();
      
      arma::mat Prec_theta0 = arma::inv_sympd(Sigma_theta0.slice(is));
      arma::mat S_theta0_post = Prec_theta0 + nj(k) / sig2_Y * eye_mat_theta0;
      arma::vec m_theta0_post = Prec_theta0 * m_theta0.row(is).t();
      
      for(int i = 0; i < N; i++){
        int s_i = s(i);
        if(s_i == k){
          arma::rowvec X_ZY_i = X_ZY.row(i);
          
          for(int t = 0; t < nT_Y; t++){
            m_theta0_post(t) += (theta(i,is,t) - arma::accu(X_ZY_i % U_Y_is)) / sig2_Y;
          }
        }
      }
      //For numerical stability:
      S_theta0_post = (S_theta0_post + S_theta0_post.t()) / 2;
      S_theta0_post = arma::inv_sympd(S_theta0_post);
      m_theta0_post = S_theta0_post * m_theta0_post;
      theta0_star.tube(k,is) = arma::mvnrnd(m_theta0_post, S_theta0_post);   
    }
  }
  
  return sample_BNP_biv_t(s, nj, K_N, b_Z_star, theta0_star);
}
