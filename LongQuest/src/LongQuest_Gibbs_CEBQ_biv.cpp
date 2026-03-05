// Main Gibbs function for PCM BNP IRT model

// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <tuple>
#include "sample_BNP_biv_DPM.h"
#include <progress.hpp>
#include <progress_bar.hpp>
using namespace Rcpp;

// [[Rcpp::export]]
List LongQuest_Gibbs_CEBQ_biv(List data_list, List MCMC_list, List Param_list){
  
  //Extract data
  arma::mat Z = as<arma::mat>(data_list["Z"]), B_Z = as<arma::mat>(data_list["B_Z"]), X_Z = as<arma::mat>(data_list["X_Z"]), X_Y = as<arma::mat>(data_list["X_Y"]), is_NA_Z = as<arma::mat>(data_list["is_NA_Z"]);
  List Y = data_list["Y"], Y_bis = clone(Y);
  
  // Extract sizes
  int nT_Z = Z.n_cols, nT_Y = Y.size();
  arma::mat Y_0 = Y[0];
  int N = Y_0.n_rows, q_Z = X_Z.n_cols, q_Y = X_Y.n_cols;
  
  // Extract number of questions within BEBQ and CEBQ
  int JY = Y_0.n_cols;
  
  //Item parameters and Hyperparameters needed
  
  //Alpha's
  arma::vec subscales_indices = as<arma::vec>(data_list["subscales_indices"]), subscales_unique = unique(subscales_indices);
  int n_subscales = subscales_unique.n_elem;
  
  // Number of main subscales
  arma::vec subscales_indices_main = as<arma::vec>(data_list["subscales_indices_main"]), subscales_main_unique = unique(subscales_indices_main);
  int n_subscales_main = subscales_main_unique.n_elem;
  
  
  //For subscale-specific means of alpha
  bool alpha_Y_zeros = Param_list["alpha_Y_zeros"];
  double sig2_alpha_Y = Param_list["sig2_alpha_Y"], m_mu_alpha_Y = Param_list["m_mu_alpha_Y"], sig2_mu_alpha_Y = Param_list["sig2_mu_alpha_Y"];
  arma::vec mu_alpha_Y(n_subscales, arma::fill::zeros), mu_alpha_Y_vec(JY, arma::fill::zeros);
  for(int j = 0; j < JY; j++){
    mu_alpha_Y_vec(j) = mu_alpha_Y(subscales_indices(j));
  }
  
  //For update/adaptation
  arma::vec sum_mu_alpha_Y(n_subscales, arma::fill::zeros);
  arma::mat S_mu_alpha_Y(n_subscales, n_subscales, arma::fill::zeros), prod_mu_alpha_Y(n_subscales, n_subscales, arma::fill::zeros), eye_mat_mu_alpha_Y(n_subscales, n_subscales, arma::fill::eye);
  S_mu_alpha_Y.diag().fill(0.1);
  arma::vec log_alpha_Y(JY, arma::fill::zeros);
  arma::vec alpha_Y = exp(log_alpha_Y);
  arma::vec s_alpha_Y(JY, arma::fill::zeros), alpha_Y_accept(JY, arma::fill::zeros);
  s_alpha_Y.fill(0.01);
  if(alpha_Y_zeros){
    alpha_Y.zeros();
  }
  
  //Beta's
  arma::mat Sigma_beta_Y = as<arma::mat>(Param_list["Sigma_beta_Y"]), Omega_beta_Y = arma::inv_sympd(Sigma_beta_Y);
  arma::vec mu_beta_Y = as<arma::vec>(Param_list["mu_beta_Y"]);
  int m_Y_beta = mu_beta_Y.n_elem, m_Y = m_Y_beta + 1;
  // This is to introduce a constraint on the beta coefficients. Sum_h beta_jh = 0
  arma::uvec ind_beta_Y = arma::regspace<arma::uvec>(1, m_Y_beta);
  //For update/adaptation
  arma::cube S_beta_Y(m_Y_beta, m_Y_beta, JY, arma::fill::zeros), prod_beta_Y(m_Y_beta, m_Y_beta, JY, arma::fill::zeros);
  arma::mat beta_Y_tilde(JY, m_Y_beta, arma::fill::zeros), beta_Y(JY, m_Y, arma::fill::zeros), sum_beta_Y(JY, m_Y_beta, arma::fill::zeros), eye_mat_beta_Y(m_Y_beta, m_Y_beta, arma::fill::eye);
  arma::vec s_d_beta_Y(JY, arma::fill::zeros), beta_Y_accept(JY, arma::fill::zeros);
  s_d_beta_Y.fill(pow(2.4,2)/m_Y_beta);
  for(int j = 0; j < JY; j++){
    // We initialise to zero
    S_beta_Y.slice(j).diag().fill(0.1);
  }  
  
  
  // Initialize subject parameter
  arma::cube theta(N, n_subscales_main, nT_Y, arma::fill::zeros), s_theta(N, n_subscales_main, nT_Y, arma::fill::ones), theta_accept(N, n_subscales_main, nT_Y, arma::fill::zeros);
  s_theta.fill(0.01);
  
  // Prior on regression coefficients and variances
  bool use_horseshoe = Param_list["use_horseshoe"], U_Y_zeros = Param_list["U_Y_zeros"], update_sig2_Y = Param_list["update_sig2_Y"];
  double sig2_Z = 1.0, a_sig2_Z = Param_list["a_sig2_Z"], b_sig2_Z = Param_list["b_sig2_Z"];
  arma::vec sig2_Y = as<arma::vec>(Param_list["sig2_Y"]), a_sig2_Y = as<arma::vec>(Param_list["a_sig2_Y"]), b_sig2_Y = as<arma::vec>(Param_list["b_sig2_Y"]);
  double xi_Z = 1.0, s_xi_Z = 1.0, xi_Z_accept = 0.0;
  arma::vec xi_Y(JY, arma::fill::ones), s_xi_Y(JY, arma::fill::ones), xi_Y_accept(JY, arma::fill::zeros);
  arma::vec eta_Z(q_Z, arma::fill::ones);
  arma::mat eta_Y(q_Y, JY, arma::fill::ones);
  
  //Initialize diagonal covariance matrix for longitudinal part
  arma::mat Sigma_Z(nT_Z, nT_Z, arma::fill::zeros);
  Sigma_Z.diag().fill(sig2_Z);
  
  arma::vec m_U_Z(q_Z, arma::fill::zeros), U_Z(q_Z, arma::fill::ones);
  arma::mat Omega_U_Z(q_Z, q_Z, arma::fill::zeros);
  arma::mat m_U_Y(q_Y, JY, arma::fill::zeros), U_Y(q_Y, JY, arma::fill::zeros), sum_U_Y(q_Y, JY, arma::fill::zeros), eye_mat_U_Y(q_Y, q_Y, arma::fill::eye);
  if(U_Y_zeros){
    U_Y.zeros();
  }
  arma::cube Omega_U_Y(q_Y, q_Y, JY, arma::fill::zeros), S_U_Y(q_Y, q_Y, JY, arma::fill::zeros), prod_U_Y(q_Y, q_Y, JY, arma::fill::zeros);
  for(int j = 0; j < JY; j++){
    Omega_U_Y.slice(j).diag() = xi_Y(j) * eta_Y.col(j);
    S_U_Y.slice(j).diag().fill(0.01);
  }
  Omega_U_Z.diag() = xi_Z * eta_Z;
  
  //Adaptive on regression coefficient
  arma::vec s_d_U_Y(JY, arma::fill::zeros), U_Y_accept(JY, arma::fill::zeros);
  s_d_U_Y.fill(pow(2.4,2)/q_Y);
  
  // Process P
  List process_P_list = Param_list["process_P_list"];
  bool update_s = process_P_list["update_s"], isDP = process_P_list["isDP"];
  bool update_kappa = process_P_list["update_kappa"], update_sigma = process_P_list["update_sigma"];
  double kappa = process_P_list["kappa"], a_kappa = 0.0, b_kappa = 0.0;
  if(update_kappa){
    a_kappa = process_P_list["a_kappa"];
    b_kappa = process_P_list["b_kappa"];
  }
  double sigma = process_P_list["sigma"], a_sigma = 0.0, b_sigma = 0.0, s_sigma = 0.0, sigma_accept = 0.0;
  if(update_sigma){
    a_sigma = process_P_list["a_sigma"];
    b_sigma = process_P_list["b_sigma"];
    s_sigma = 0.1;
  }
  double u = 1.0, s_u = 0.1, u_accept = 0.0;
  
  //Initialize BNP lists
  arma::vec s = MCMC_list["s_init"], s_aux = unique(s); // allocation variables s
  double K_N = s_aux.n_elem;
  arma::vec nj(K_N,arma::fill::zeros);
  for(int i = 0; i < N; i++){
    nj(s(i)) ++;
  }
  arma::vec m_b_Z = as<arma::vec>(Param_list["m_b_Z"]);
  arma::mat m_theta0(n_subscales_main, nT_Y, arma::fill::zeros), eye_mat_theta0(nT_Y, nT_Y, arma::fill::eye), Omega_b_Z = as<arma::mat>(Param_list["Omega_b_Z"]), Sigma_b_Z = arma::inv_sympd(Omega_b_Z), b_Z_star = arma::mvnrnd(m_b_Z, Sigma_b_Z, K_N).t();
  arma::cube Sigma_theta0(nT_Y, nT_Y, n_subscales_main, arma::fill::zeros), theta0_star(K_N, n_subscales_main, nT_Y, arma::fill::zeros);
  //Prior on hyperparameters of P0
  bool update_P0 = process_P_list["update_P0"];
  arma::vec nu_Sigma_theta0 = as<arma::vec>(Param_list["nu_Sigma_theta0"]);
  arma::cube Psi_Sigma_theta0 = as<arma::cube>(Param_list["Psi_Sigma_theta0"]);
  arma::mat mu_m_theta0 = as<arma::mat>(Param_list["mu_m_theta0"]);
  for(int is = 0; is < n_subscales_main; is++){
    Sigma_theta0.slice(is) = arma::iwishrnd(Psi_Sigma_theta0.slice(is), nu_Sigma_theta0(is));
    // Sigma_theta0.slice(is).diag().fill(1.0);
    m_theta0.row(is) = arma::mvnrnd(mu_m_theta0.row(is).t(), Sigma_theta0.slice(is)).t();
  }
  //Initialise theta0_star
  for(int is = 0; is < n_subscales_main; is++){
    for(int k = 0; k < K_N; k++){
      theta0_star.tube(k,is) = arma::mvnrnd(m_theta0.row(is).t(), Sigma_theta0.slice(is));
    }
  }
  
  //Algorithm parameters
  double n_burn1 = MCMC_list["n_burn1"], n_burn2 = MCMC_list["n_burn2"], thin = MCMC_list["thin"], n_save = MCMC_list["n_save"];
  int n_tot = n_burn1 + n_burn2 + thin * n_save, iter;
  //Adaptation
  NumericVector ADAPT(4);
  ADAPT(0) = n_burn1; //"burn-in" for adaptation
  ADAPT(1) = 0.7; //exponent for adaptive step
  ADAPT(2) = 0.234; //reference acceptance rate
  ADAPT(3) = 0.001; //for multivariate updates
  
  //Output lists
  arma::vec K_N_out(n_save, arma::fill::zeros), kappa_out(n_save,arma::fill::zeros), sig2_Z_out(n_save, arma::fill::zeros), xi_Z_out(n_save,arma::fill::zeros), sigma_out(n_save,arma::fill::zeros), u_out(n_save,arma::fill::zeros);
  arma::mat alpha_Y_out(n_save, JY, arma::fill::zeros), mu_alpha_Y_out(n_save, n_subscales, arma::fill::zeros), s_out(n_save, N, arma::fill::zeros), U_Z_out(n_save, q_Z, arma::fill::zeros), eta_Z_out(n_save, q_Z, arma::fill::zeros), sig2_Y_out(n_save, n_subscales_main, arma::fill::zeros), xi_Y_out(n_save, JY, arma::fill::zeros);
  arma::cube Z_out(N, nT_Z, n_save, arma::fill::zeros), beta_Y_out(JY, m_Y, n_save, arma::fill::zeros), U_Y_out(q_Y, JY, n_save, arma::fill::zeros), eta_Y_out(q_Y, JY, n_save, arma::fill::zeros);
  arma::cube theta_1_out(N, nT_Y, n_save, arma::fill::zeros), theta_2_out(N, nT_Y, n_save, arma::fill::zeros), m_theta0_out(n_subscales_main, nT_Y, n_save, arma::fill::zeros);
  arma::field<arma::cube> Sigma_theta0_out(n_save);
  List Y_out(n_save), b_Z_star_out(n_save), theta0_star_out(n_save), nj_out(n_save);
  
  Progress progr(n_tot, true);
  
  for(int g = 0; g < n_tot; g ++){
    
    // Rcout << "Missing Z \n";
    //////////////////////////
    // Missing values for Z //
    //////////////////////////
    
    for(int i = 0; i < N; i++){
      int s_i = s(i);
      
      // Impute missing from longitudinal part
      arma::rowvec is_NA_Z_i = is_NA_Z.row(i);
      
      //Check if missing values are present
      int n_NA_i = arma::sum(is_NA_Z_i);
      if(n_NA_i > 0){
        
        arma::vec Z_i = Z.row(i).t();
        
        arma::vec m_i_aux(nT_Z);
        m_i_aux.fill(arma::accu(X_Z.row(i) % U_Z.t()));
        arma::vec m_i = B_Z * b_Z_star.row(s_i).t() + m_i_aux;
        
        // Missing time points
        arma::uvec miss_i = arma::find(is_NA_Z_i > 0.0);
        
        // Sample missing components (conditional independence so marginal is enough!)
        Z_i(miss_i) = arma::mvnrnd(m_i(miss_i), Sigma_Z(miss_i,miss_i));
        Z.row(i) = Z_i.t();
      }
    }
    
    
    // Rcout << "Missing Y \n";
    //////////////////////////
    // Missing values for Y //
    //////////////////////////
    
    for(int t = 0; t < nT_Y; t ++){
      
      // Data at time t
      arma::mat Y_t = Y[t];
      arma::mat Y_t_bis = Y_bis[t];
      
      for(int j = 0; j < JY; j ++){
        arma::rowvec U_Y_j = U_Y.col(j).t();
        
        // Index in the main subscale
        int is = subscales_indices_main(j);
        
        // Item parmeters for question j
        double alpha_Y_j = alpha_Y(j);
        arma::vec beta_Y_j = beta_Y.row(j).t();
        
        arma::vec prob_h(m_Y), beta_cumsum = arma::cumsum(beta_Y_j);
        for(int i = 0; i < N; i ++){
          
          if(!arma::is_finite(Y_t(i,j))){
            arma::rowvec X_Y_i = X_Y.row(i);
            
            prob_h.zeros();
            for(int h = 0; h < m_Y; h++){
              prob_h(h) = alpha_Y_j * (h * theta(i,is,t) - beta_cumsum(h)) + h * arma::accu(X_Y_i % U_Y_j);
            }
            prob_h = exp(prob_h - max(prob_h));
            prob_h = prob_h/sum(prob_h);
            
            double aux_runif = arma::randu();
            arma::vec prob_h_cum = arma::cumsum(prob_h);
            arma::uvec hh_vec = arma::find(prob_h_cum >= aux_runif, 1, "first");
            int hh = hh_vec(0);
            
            Y_t_bis(i,j) = hh + 1;
          }
        }
      }
      Y_bis[t] = Y_t_bis;
    }
    
    
    
    
    if(update_s){
      // Rcout << "BNP \n";
      /////////////////////
      // sample BNP part //
      /////////////////////
      
      std::tie(s, nj, K_N, b_Z_star, theta0_star) = sample_BNP_biv_DPM(Y_bis, Z, B_Z, theta, X_Z, X_Y, U_Z, s, nj, K_N, isDP, kappa, sigma, u, b_Z_star, theta0_star, m_b_Z, Sigma_b_Z, Omega_b_Z, m_theta0, Sigma_theta0, sig2_Z, sig2_Y, subscales_indices_main);
    }else{
      // Rcout << "Unique values \n";
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
            mpost_aux.fill(arma::accu(X_Z.row(i) % U_Z.t()));
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
          arma::mat Prec_theta0 = arma::inv_sympd(Sigma_theta0.slice(is));
          arma::mat S_theta0_post = Prec_theta0 + nj(k) / sig2_Y(is) * eye_mat_theta0;
          arma::vec m_theta0_post = Prec_theta0 * m_theta0.row(is).t();
          
          for(int i = 0; i < N; i++){
            int s_i = s(i);
            if(s_i == k){
              arma::vec theta_i_is = theta.tube(i,is);
              m_theta0_post += theta_i_is / sig2_Y(is);
            }
          }
          S_theta0_post = arma::inv_sympd(S_theta0_post);
          m_theta0_post = S_theta0_post * m_theta0_post;
          theta0_star.tube(k,is) = arma::mvnrnd(m_theta0_post, S_theta0_post); 
        }
      }
    }
    
    
    if(update_P0){
      // Rcout << "P0 hyperparameters\n";
      ///////////////
      // sample P0 //
      ///////////////
      for(int is = 0; is < n_subscales_main; is++){
        ////////////////////
        // sample P0 - m0 //
        ////////////////////
        arma::vec m_m_theta0_post = mu_m_theta0.row(is).t();
        for(int k = 0; k < K_N; k++){
          arma::vec theta0_aux_k = theta0_star.tube(k,is);
          m_m_theta0_post += theta0_aux_k;
        }     
        m_m_theta0_post = m_m_theta0_post / (K_N + 1);
        m_theta0.row(is) = arma::mvnrnd(m_m_theta0_post, Sigma_theta0.slice(is) / (K_N + 1)).t();   
        
        
        ////////////////////////
        // sample P0 - Sigma0 //
        ////////////////////////
        double nu_Sigma_theta0_post = nu_Sigma_theta0(is) + K_N + 1;
        arma::mat Psi_Sigma_theta0_post = Psi_Sigma_theta0.slice(is) + (m_theta0.row(is).t() - mu_m_theta0.row(is).t()) * (m_theta0.row(is) - mu_m_theta0.row(is));
        for(int k = 0; k < K_N; k++){
          arma::vec theta0_aux_k = theta0_star.tube(k,is);
          Psi_Sigma_theta0_post += (theta0_aux_k - m_theta0.row(is).t()) * (theta0_aux_k.t() - m_theta0.row(is));
        }   
        Sigma_theta0.slice(is) = arma::iwishrnd(Psi_Sigma_theta0_post, nu_Sigma_theta0_post);
      }
    }
    
    
    
    
    
    // Rcout << "theta \n";
    ////////////////////
    // sample theta's //
    ////////////////////
    for(int is = 0; is < n_subscales_main; is++){
      
      for(int i = 0; i < N; i++){
        arma::rowvec X_Y_i = X_Y.row(i);
        
        int s_i = s(i);
        for(int t = 0; t < nT_Y; t++){
          
          // Propose a new value
          double theta_new = theta(i,is,t) + arma::randn() * sqrt(s_theta(i,is,t));
          
          // Prior part (time t)
          double m_theta_it = theta0_star(s_i,is,t);
          double log_ratio_theta = - .5 * (pow(theta_new - m_theta_it, 2) - pow(theta(i,is,t) - m_theta_it, 2)) / sig2_Y(is);
          
          // Likelihood part
          arma::mat Y_t_bis = Y_bis[t];
          arma::vec prob_itj(m_Y);
          for(int j = 0; j < JY; j++){
            arma::rowvec U_Y_j = U_Y.col(j).t();
            
            //Contribution only from questions in this main subscale
            if(subscales_indices_main(j) == is){
              // Item parmeters for question j
              double alpha_Y_j = alpha_Y(j);
              arma::vec beta_Y_j = beta_Y.row(j).t();
              arma::vec beta_cumsum = arma::cumsum(beta_Y_j);
              
              prob_itj.zeros();
              for(int h = 0; h < m_Y; h++){
                prob_itj(h) = alpha_Y_j * (h * theta_new - beta_cumsum(h)) + h * arma::accu(X_Y_i % U_Y_j);
              }
              prob_itj = exp(prob_itj - max(prob_itj));
              prob_itj = prob_itj/sum(prob_itj);
              
              log_ratio_theta += log(prob_itj(Y_t_bis(i,j)-1));
              
              prob_itj.zeros();
              for(int h = 0; h < m_Y; h++){
                prob_itj(h) = alpha_Y_j * (h * theta(i,is,t) - beta_cumsum(h)) + h * arma::accu(X_Y_i % U_Y_j);
              }
              prob_itj = exp(prob_itj - max(prob_itj));
              prob_itj = prob_itj/sum(prob_itj);
              
              log_ratio_theta += - log(prob_itj(Y_t_bis(i,j)-1));
            }
          }
          
          double accept_theta = 1.0;
          if( arma::is_finite(log_ratio_theta) ){
            if(log_ratio_theta < 0){
              accept_theta = exp(log_ratio_theta);
            }
          }else{
            accept_theta = 0.0;
          }
          
          theta_accept(i,is,t) += accept_theta;
          if( arma::randu() < accept_theta ){
            theta(i,is,t) = theta_new;
          }
          
          s_theta(i,is,t) = s_theta(i,is,t) + pow(g+1,-ADAPT(1)) * (accept_theta - ADAPT(2));
          if(s_theta(i,is,t) > exp(50)){
            s_theta(i,is,t) = exp(50);
          }else{
            if(s_theta(i,is,t) < exp(-50)){
              s_theta(i,is,t) = exp(-50);
            }
          }
        }
      }
    }
    
    
    
    
    
    
    // Rcout << "beta_Y \n";
    ////////////////////
    // sample beta_Y //
    ////////////////////
    
    //Same item for different time points in CEBQ
    
    // For each question, update the corresponding item parameters
    for(int j = 0; j < JY; j++){
      arma::rowvec U_Y_j = U_Y.col(j).t();
      
      // Index in the main subscale
      int is = subscales_indices_main(j);
      
      // Current value of item parameters
      double alpha_Y_j = alpha_Y(j);
      
      arma::vec beta_Y_j = beta_Y.row(j).t(), beta_Y_j_new = beta_Y_j;
      arma::vec beta_Y_tilde_j = beta_Y_j.rows(ind_beta_Y);
      // Propose new value of item parameters
      arma::mat S_beta_Y_j = S_beta_Y.slice(j);
      arma::vec beta_Y_tilde_j_new = arma::mvnrnd(beta_Y_tilde_j, S_beta_Y_j);
      beta_Y_j_new.elem(ind_beta_Y) = beta_Y_tilde_j_new;
      
      //Computing MH ratio:
      double log_ratio_beta_Y = 0.0;
      //Prior and proposal (symmetric)
      log_ratio_beta_Y += - .5 * (arma::as_scalar((beta_Y_tilde_j_new.t() - mu_beta_Y.t()) * Omega_beta_Y * (beta_Y_tilde_j_new - mu_beta_Y) - arma::as_scalar((beta_Y_tilde_j.t() - mu_beta_Y.t()) * Omega_beta_Y * (beta_Y_tilde_j - mu_beta_Y))));
      
      // Consider contribution from all time points
      arma::vec prob_itj(m_Y), beta_cumsum = arma::cumsum(beta_Y_j), beta_new_cumsum = arma::cumsum(beta_Y_j_new);
      for(int t = 0; t < nT_Y; t ++){
        
        //Data at time t
        arma::mat Y_t_bis = Y_bis[t];
        arma::vec Y_tj = Y_t_bis.col(j);
        
        //Likelihood
        for(int i = 0; i < N; i ++){
          arma::rowvec X_Y_i = X_Y.row(i);
          
          prob_itj.zeros();
          for(int h = 0; h < m_Y; h++){
            prob_itj(h) = alpha_Y_j * (h * theta(i,is,t) - beta_new_cumsum(h)) + h * arma::accu(X_Y_i % U_Y_j);
          }
          prob_itj = exp(prob_itj - max(prob_itj));
          prob_itj = prob_itj/sum(prob_itj);
          
          log_ratio_beta_Y += log(prob_itj(Y_tj(i)-1));
          
          
          prob_itj.zeros();
          for(int h = 0; h < m_Y; h++){
            prob_itj(h) = alpha_Y_j * (h * theta(i,is,t) - beta_cumsum(h)) + h * arma::accu(X_Y_i % U_Y_j);
          }
          prob_itj = exp(prob_itj - max(prob_itj));
          prob_itj = prob_itj/sum(prob_itj);
          
          log_ratio_beta_Y += - log(prob_itj(Y_tj(i)-1));
        }
      }
      
      double accept_beta_Y = 1.0;
      if( arma::is_finite(log_ratio_beta_Y) ){
        if(log_ratio_beta_Y < 0){
          accept_beta_Y = exp(log_ratio_beta_Y);
        }
      }else{
        accept_beta_Y = 0.0;
      }
      
      beta_Y_accept(j) += accept_beta_Y;
      
      if( arma::randu() < accept_beta_Y ){
        // Item parameters at time t for question j
        beta_Y.row(j) = beta_Y_j_new.t();
        beta_Y_tilde_j = beta_Y_tilde_j_new;
      }
      
      sum_beta_Y.row(j) += beta_Y_tilde_j.t();
      prod_beta_Y.slice(j) += beta_Y_tilde_j * beta_Y_tilde_j.t();
      
      s_d_beta_Y(j) += pow(g+1,-ADAPT(1))*(accept_beta_Y - ADAPT(2));
      if(s_d_beta_Y(j) > exp(50)){
        s_d_beta_Y(j) = exp(50);
      }else{
        if(s_d_beta_Y(j) < exp(-50)){
          s_d_beta_Y(j) = exp(-50);
        }
      }
      if(g > (ADAPT(0) - 1)){
        S_beta_Y.slice(j) = s_d_beta_Y(j)/g * (prod_beta_Y.slice(j) - sum_beta_Y.row(j).t() * sum_beta_Y.row(j)/(g+1.0)) + s_d_beta_Y(j) * pow(0.1,2.0) / m_Y * eye_mat_beta_Y;
      }
    }
    
    
    
    
    if(!alpha_Y_zeros){
      // Rcout << "alpha_Y \n";
      ////////////////////
      // sample alpha_Y //
      ////////////////////
      
      //We have a constraint on the alpha's --> joint update for each sub-questionnaire
      for(int j = 0; j < JY; j++){
        arma::rowvec U_Y_j = U_Y.col(j).t();
        
        //subscale index
        int ind_s_j = subscales_indices(j), is = subscales_indices_main(j);
        
        //New value fo alphas
        double log_alpha_Y_j = log_alpha_Y(j), log_alpha_Y_new = log_alpha_Y_j + arma::randn() * sqrt(s_alpha_Y(j));
        double alpha_Y_new = exp(log_alpha_Y_new);
        
        //Computing MH ratio:
        double log_ratio_alpha_Y_j = 0.0;
        //Prior and proposal (symmetric)
        log_ratio_alpha_Y_j += - .5 * (pow(log_alpha_Y_new - mu_alpha_Y_vec(ind_s_j), 2.0) - pow(log_alpha_Y_j - mu_alpha_Y_vec(ind_s_j), 2.0)) / sig2_alpha_Y;
        
        // Consider contribution from all time points
        double alpha_Y_j = alpha_Y(j);
        arma::vec prob_itj(m_Y), beta_cumsum = arma::cumsum(beta_Y.row(j).t());
        for(int t = 0; t < nT_Y; t ++){
          
          //Data at time t
          arma::mat Y_t_bis = Y_bis[t];
          arma::vec Y_tj = Y_t_bis.col(j);
          
          //Likelihood
          for(int i = 0; i < N; i ++){
            arma::rowvec X_Y_i = X_Y.row(i);
            
            prob_itj.zeros();
            for(int h = 0; h < m_Y; h++){
              prob_itj(h) = alpha_Y_new * (h * theta(i,is,t) - beta_cumsum(h)) + h * arma::accu(X_Y_i % U_Y_j);
            }
            prob_itj = exp(prob_itj - max(prob_itj));
            prob_itj = prob_itj/sum(prob_itj);
            
            log_ratio_alpha_Y_j += log(prob_itj(Y_tj(i)-1));
            
            
            prob_itj.zeros();
            for(int h = 0; h < m_Y; h++){
              prob_itj(h) = alpha_Y_j * (h * theta(i,is,t) - beta_cumsum(h)) + h * arma::accu(X_Y_i % U_Y_j);
            }
            prob_itj = exp(prob_itj - max(prob_itj));
            prob_itj = prob_itj/sum(prob_itj);
            
            log_ratio_alpha_Y_j += - log(prob_itj(Y_tj(i)-1));
          }
        }
        
        double accept_alpha_Y_j = 1.0;
        if( arma::is_finite(log_ratio_alpha_Y_j) ){
          if(log_ratio_alpha_Y_j < 0){
            accept_alpha_Y_j = exp(log_ratio_alpha_Y_j);
          }
        }else{
          accept_alpha_Y_j = 0.0;
        }
        
        alpha_Y_accept(j) += accept_alpha_Y_j;
        
        if( arma::randu() < accept_alpha_Y_j ){
          // Item parameters at time t for question j
          log_alpha_Y(j) = log_alpha_Y_new;
          alpha_Y(j) = alpha_Y_new;
        }
        
        s_alpha_Y(j) += pow(g+1,-ADAPT(1))*(accept_alpha_Y_j - ADAPT(2));
        if(s_alpha_Y(j) > exp(50)){
          s_alpha_Y(j) = exp(50);
        }else{
          if(s_alpha_Y(j) < exp(-50)){
            s_alpha_Y(j) = exp(-50);
          }
        }
      }
      
      
      
      // Rcout << "mu_alpha_Y \n";
      ///////////////////////
      // sample mu_alpha_Y //
      ///////////////////////
      
      //Common mean for each subscale --> full-conditional is conjugate! (assuming independence a-priori)
      arma::vec m_mu_alpha_Y_post(n_subscales, arma::fill::zeros);
      arma::vec sig2_mu_alpha_Y_post(n_subscales, arma::fill::zeros);
      m_mu_alpha_Y_post.fill(m_mu_alpha_Y / sig2_mu_alpha_Y);
      sig2_mu_alpha_Y_post.fill(1 / sig2_mu_alpha_Y);
      for(int j = 0; j < JY; j++){
        
        //Subscale indices for question j
        int ind_s_j = subscales_indices(j);
        
        sig2_mu_alpha_Y_post(ind_s_j) += 1 / sig2_mu_alpha_Y;
        m_mu_alpha_Y_post(ind_s_j) += log(alpha_Y(j)) / sig2_mu_alpha_Y;
      }
      
      for(int is = 0; is < n_subscales; is++){
        sig2_mu_alpha_Y_post(is) = 1 / sig2_mu_alpha_Y_post(is);
        m_mu_alpha_Y_post(is) = sig2_mu_alpha_Y_post(is) * m_mu_alpha_Y_post(is);
        mu_alpha_Y(is) = m_mu_alpha_Y_post(is) + arma::randn() * sqrt(sig2_mu_alpha_Y_post(is));
      }
      for(int j = 0; j < JY; j++){
        int ind_is = subscales_indices(j);
        mu_alpha_Y_vec(j) = mu_alpha_Y(ind_is);
      }
    }
    
    
    
    
    
    if(use_horseshoe){
      // Rcout << "HS part \n";
      /////////////
      // HS part //
      /////////////
      
      // Slice sampler for eta's
      double mj = 0.0, Uj = 0.0, Tj = 0.0, uj = 0.0;
      for(int l = 0; l < q_Z; l++){
        
        mj = 0.5 * pow(U_Z(l) - m_U_Z(l), 2.0) * xi_Z;
        
        Uj = arma::randu() / (1 + eta_Z(l));
        Tj = (1/Uj - 1);
        uj = arma::randu();
        
        eta_Z(l) = - 1/mj * log(1 - uj*(1 - exp(-mj * Tj)));
      }
      //One for each question, covariate and sub-class of items
      for(int j = 0; j < JY; j++){
        for(int l = 0; l < q_Y; l++){
          mj = 0.0;
          mj += 0.5 * pow(U_Y(l,j) - m_U_Y(l,j), 2.0) * xi_Y(j);
          Uj = arma::randu() / (1 + eta_Y(l,j));
          Tj = (1/Uj - 1);
          uj = arma::randu();
          
          eta_Y(l,j) = -1/mj * log(1 - uj*(1 - exp(-mj * Tj)));
        }
      }
      
      
      // // Slice sampler for xi_Z and xi_Y (needs boost library which is heavy...but faster once it is uploaded)
      // mj = 0.0, Uj = 0.0, Tj = 0.0, uj = 0.0;
      // for(int l = 0; l < q_Z; l++){
      //   mj += 0.5 * pow(U_Z(l) - m_U_Z(l), 2.0) * eta_Z(l);
      // }
      // Uj = arma::randu() / (1 + xi_Z);
      // Tj = (1/Uj - 1);
      // uj = arma::randu();
      // 
      // if(q_Z == 1){
      //   xi_Z = - 1/mj * log(1 - uj*(1 - exp(-mj * Tj)));
      // }else{
      //   xi_Z = boost::math::gamma_p_inv((q_Z + 1.0)/2.0, uj * boost::math::gamma_p((q_Z + 1.0)/2.0, mj * Tj)  ) / mj;
      // }
      // 
      // for(int j = 0; j < JY; j++){
      //     double mj = 0.0, Uj = 0.0, Tj = 0.0, uj = 0.0;
      //     for(int l = 0; l < q_Y; l++){
      //       mj += 0.5 * pow(U_Y(l,j) - m_U_Y(l,j), 2.0) * eta_Y(l,j);
      //     }
      //     Uj = arma::randu() / (1 + xi_Y(j));
      //     Tj = (1/Uj - 1);
      //     uj = arma::randu();
      //     
      //     if(q_Z == 1){
      //       xi_Y(j) = - 1/mj * log(1 - uj*(1 - exp(-mj * Tj)));
      //     }else{
      //       xi_Y(j) = boost::math::gamma_p_inv((q_Y + 1.0)/2.0, uj * boost::math::gamma_p((q_Y + 1.0)/2.0, mj * Tj)  ) / mj;
      //     }
      //   }
      
      
      // Metropolis-Hastings step for xi_Z and xi_Y (adaptive)
      
      // Propose a new value
      double xi_Z_new = xi_Z * exp(arma::randn() * sqrt(s_xi_Z));
      
      // Compute MH ratio:
      
      //Proposal and prior
      double log_ratio_xi_Z = (q_Z + 1.0)/2.0 * (log(xi_Z_new) - log(xi_Z)) - (log(1 + xi_Z_new) - log(1 + xi_Z));
      
      //Regression coefficients part
      log_ratio_xi_Z += - 0.5 * arma::accu(eta_Z % (U_Z - m_U_Z) % (U_Z - m_U_Z)) * (xi_Z_new - xi_Z);
      
      double accept_xi_Z = 1.0;
      if( arma::is_finite(log_ratio_xi_Z) ){
        if(log_ratio_xi_Z < 0){
          accept_xi_Z = exp(log_ratio_xi_Z);
        }
      }else{
        accept_xi_Z = 0.0;
      }
      
      xi_Z_accept += accept_xi_Z;
      if( arma::randu() < accept_xi_Z ){
        xi_Z = xi_Z_new;
      }
      
      s_xi_Z += pow(g+1,-ADAPT(1))*(accept_xi_Z - ADAPT(2));
      if(s_xi_Z > exp(50)){
        s_xi_Z = exp(50);
      }else{
        if(s_xi_Z < exp(-50)){
          s_xi_Z = exp(-50);
        }
      }
      
      for(int j = 0; j < JY; j++){
        arma::vec U_Y_j = U_Y.col(j), m_U_Y_j = m_U_Y.col(j);
        // Propose a new value
        double xi_Y_new = xi_Y(j) * exp(arma::randn() * sqrt(s_xi_Y(j)));
        
        // Compute MH ratio:
        
        //Proposal and prior
        double log_ratio_xi_Y = (q_Y * m_Y + 1.0)/2.0 * (log(xi_Y_new) - log(xi_Y(j))) - (log(1 + xi_Y_new) - log(1 + xi_Y(j)));
        
        //Regression coefficients part
        //Start from category 1 beause category 0 is fixed
        for(int h = 1; h < m_Y; h++){
          log_ratio_xi_Y += - 0.5 * arma::accu(eta_Y.col(j) % (U_Y_j - m_U_Y_j) % (U_Y_j - m_U_Y_j)) * (xi_Y_new - xi_Y(j));
        }
        
        double accept_xi_Y = 1.0;
        if( arma::is_finite(log_ratio_xi_Y) ){
          if(log_ratio_xi_Y < 0){
            accept_xi_Y = exp(log_ratio_xi_Y);
          }
        }else{
          accept_xi_Y = 0.0;
        }
        
        xi_Y_accept(j) += accept_xi_Y;
        if( arma::randu() < accept_xi_Y ){
          xi_Y(j) = xi_Y_new;
        }
        
        s_xi_Y(j) += pow(g+1,-ADAPT(1))*(accept_xi_Y - ADAPT(2));
        if(s_xi_Y(j) > exp(50)){
          s_xi_Y(j) = exp(50);
        }else{
          if(s_xi_Y(j) < exp(-50)){
            s_xi_Y(j) = exp(-50);
          }
        }
      }
      
      // Fill matrices for update of coefficients
      Omega_U_Z.diag() = xi_Z * eta_Z;
      for(int j = 0; j < JY; j++){
        Omega_U_Y.slice(j).diag() = xi_Y(j) * eta_Y.col(j);
      }
    }
    
    
    // Rcout << "U_Z \n";
    ////////////////
    // sample U_Z //
    ////////////////
    
    arma::mat Spost_U_Z = Omega_U_Z;
    
    arma::vec mpost_U_Z = Omega_U_Z * m_U_Z;
    for(int i = 0; i < N; i ++){
      //Cluster assignment of subject i
      int s_i = s(i);
      
      //Covariates for subject i
      arma::rowvec X_Z_i = X_Z.row(i);
      
      Spost_U_Z += X_Z_i.t() * X_Z_i / sig2_Z * nT_Z; //Need to multiply by nT_Z because it is the same regression at every time point!
      mpost_U_Z += X_Z_i.t() * arma::accu(Z.row(i) - b_Z_star.row(s_i) * B_Z.t()) / sig2_Z;
    }
    Spost_U_Z = arma::inv_sympd(Spost_U_Z);
    mpost_U_Z = Spost_U_Z * mpost_U_Z;
    
    U_Z = arma::mvnrnd(mpost_U_Z, Spost_U_Z);
    
    
    
    if(!U_Y_zeros){
      // Rcout << "U_Y \n";
      ////////////////
      // sample U_Y // joint proposal over time points
      ////////////////
      for(int j = 0; j < JY; j++){
        arma::vec U_Y_j = U_Y.col(j), m_U_Y_j = m_U_Y.col(j);
        arma::mat S_U_Y_j = S_U_Y.slice(j), Omega_U_Y_j = Omega_U_Y.slice(j);
        
        //Propose new vector of regression coefficients with random walk
        arma::vec U_Y_new = arma::mvnrnd(U_Y_j, S_U_Y_j);
        
        //Computing MH ratio:
        double log_ratio_U_Y = 0.0;
        //Prior and proposal (symmetric)
        log_ratio_U_Y += - .5 * (arma::as_scalar((U_Y_new.t() - m_U_Y_j.t()) * Omega_U_Y_j * (U_Y_new - m_U_Y_j) - arma::as_scalar((U_Y_j.t() - m_U_Y_j.t()) * Omega_U_Y_j * (U_Y_j - m_U_Y_j))));
        
        // Index in the main subscale
        int is = subscales_indices_main(j);
        
        // Likelihood part
        for(int t = 0; t < nT_Y; t++){
          //Data at time t
          arma::mat Y_t_bis = Y_bis[t];
          
          for(int i = 0; i < N; i++){
            //Data for subject i
            arma::vec Y_it = Y_t_bis.row(i).t();
            arma::vec X_Y_i = X_Y.row(i).t();
            
            arma::vec prob_itj(m_Y);
            
            //Contribution only from questions in this main subscale
            if(subscales_indices_main(j) == is){
              
              // Item parameters for question j
              double alpha_Y_j = alpha_Y(j);
              arma::vec beta_Y_j = beta_Y.row(j).t();
              arma::vec beta_cumsum = arma::cumsum(beta_Y_j);
              
              prob_itj.zeros();
              for(int h = 0; h < m_Y; h++){
                prob_itj(h) = alpha_Y_j * (h * theta(i,is,t) - beta_cumsum(h)) + h * arma::accu(X_Y_i % U_Y_new);
              }
              prob_itj = exp(prob_itj - max(prob_itj));
              prob_itj = prob_itj/sum(prob_itj);
              
              log_ratio_U_Y += log(prob_itj(Y_it(j)-1));
              
              prob_itj.zeros();
              for(int h = 0; h < m_Y; h++){
                prob_itj(h) = alpha_Y_j * (h * theta(i,is,t) - beta_cumsum(h)) + h * arma::accu(X_Y_i % U_Y_j);
              }
              prob_itj = exp(prob_itj - max(prob_itj));
              prob_itj = prob_itj/sum(prob_itj);
              
              log_ratio_U_Y += - log(prob_itj(Y_it(j)-1));
            }
          }
        }
        
        double accept_U_Y = 1.0;
        if( arma::is_finite(log_ratio_U_Y) ){
          if(log_ratio_U_Y < 0){
            accept_U_Y = exp(log_ratio_U_Y);
          }
        }else{
          accept_U_Y = 0.0;
        }
        
        U_Y_accept(j) += accept_U_Y;
        
        if( arma::randu() < accept_U_Y ){
          U_Y.col(j) = U_Y_new;
        }
        
        sum_U_Y.col(j) += U_Y.col(j);
        
        prod_U_Y.slice(j) += U_Y.col(j) * U_Y.col(j).t();
        
        s_d_U_Y(j) += pow(g+1,-ADAPT(1))*(accept_U_Y - ADAPT(2));
        if(s_d_U_Y(j) > exp(50)){
          s_d_U_Y(j) = exp(50);
        }else{
          if(s_d_U_Y(j) < exp(-50)){
            s_d_U_Y(j) = exp(-50);
          }
        }
        if(g > (ADAPT(0) - 1)){
          S_U_Y.slice(j) = s_d_U_Y(j)/g * (prod_U_Y.slice(j) - sum_U_Y.col(j) * sum_U_Y.col(j).t()/(g+1.0)) + s_d_U_Y(j) * pow(0.1,2.0) / q_Y * eye_mat_U_Y;
        }
      }
    }
    
    
    
    
    // Rcout << "sig2_Z \n";
    ///////////////////
    // sample sig2_Z //
    ///////////////////
    
    // Conjugate inverse-gamma prior
    
    //Prior part
    double aux_sig2_Z = b_sig2_Z;
    
    // Likelihood part
    for(int i = 1; i < N; i++){
      int s_i = s(i);
      arma::rowvec X_Z_i = X_Z.row(i);
      for(int t = 0; t < nT_Z; t++){
        aux_sig2_Z += 0.5 * pow(Z(i,t) - arma::accu(b_Z_star.row(s_i) % B_Z.row(t)) - arma::accu(X_Z_i % U_Z.t()), 2);
      }
    }
    sig2_Z = 1 / arma::randg(arma::distr_param(a_sig2_Z + N * nT_Z / 2, 1 / aux_sig2_Z ));
    
    //Update diagonal covariance matrix for longitudinal part
    Sigma_Z.diag().fill(sig2_Z);
    
    
    
    
    if(update_sig2_Y){
      // Rcout << "sig2_Y \n";
      ///////////////////
      // sample sig2_Y //
      ///////////////////
      for(int is = 0; is < n_subscales_main; is++){
        double aux_sig2_Y = b_sig2_Y(is);
        for(int i = 1; i < N; i++){
          for(int t = 0; t < nT_Y; t++){
            aux_sig2_Y += 0.5 * pow(theta(i,is,t), 2.0);
          }
        }
        sig2_Y(is) = 1 / arma::randg(arma::distr_param(a_sig2_Y(is) + N * nT_Y / 2, 1 / aux_sig2_Y ));
      }
    }
    
    
    
    
    
    
    
    
    
    if(isDP){
      if(update_kappa){
        // Rcout << "kappa \n";
        //////////////////
        // sample kappa //
        //////////////////
        
        double kappa_aux = R::rbeta(kappa + 1, N);
        
        double pi_kappa1 = a_kappa + K_N - 1;
        double pi_kappa2 = b_kappa - log(kappa_aux);
        double pi_kappa = pi_kappa1 / (pi_kappa1 + N * pi_kappa2 );
        
        if( arma::randu() < pi_kappa){
          kappa = arma::randg(arma::distr_param(pi_kappa1 + 1, 1/pi_kappa2));
        }else{
          kappa = arma::randg(arma::distr_param(pi_kappa1, 1/pi_kappa2));
        }
      }
    }else{
      if(update_kappa){
        kappa = arma::randg(arma::distr_param(a_kappa + K_N, 1 / (b_kappa + (pow(1 + u, sigma) - 1) / sigma)));
      }
      
      if(update_sigma){
        // Rcout << "sigma\n";
        //////////////////
        // sample sigma //
        ////////////////// 
        
        double sigma_new =  log(sigma/(1 - sigma)) + sqrt(s_sigma) * arma::randn();
        sigma_new = 1/(1 + exp(-sigma_new));
        
        double log_ratio_sigma = a_sigma * (log(sigma_new) - log(sigma)) + b_sigma * (log(1 - sigma_new) - log(1 - sigma));
        log_ratio_sigma += K_N * (log(1 + u) * (sigma_new - sigma) + lgamma(1 - sigma) - lgamma(1 - sigma_new));
        log_ratio_sigma += - kappa * ((pow(1 + u, sigma_new) - 1) / sigma_new - (pow(1 + u, sigma) - 1) / sigma);
        log_ratio_sigma += arma::accu(lgamma(nj - sigma_new) - lgamma(nj - sigma));
        
        double accept_sigma = 1.0;
        if( arma::is_finite(log_ratio_sigma) ){
          if(log_ratio_sigma < 0){
            accept_sigma = exp(log_ratio_sigma);
          }
        }else{
          accept_sigma = 0.0;
        }
        
        sigma_accept = sigma_accept + accept_sigma;
        
        if( arma::randu() < accept_sigma ){
          sigma = sigma_new;
        }
        
        s_sigma = s_sigma + pow(g + 1,-ADAPT(1)) * (accept_sigma - ADAPT(2));
        if(s_sigma > exp(50)){
          s_sigma = exp(50);
        }else{
          if(s_sigma < exp(-50)){
            s_sigma = exp(-50);
          }
        }
      }
      
      
      // Rcout << "u\n";
      //////////////
      // sample u //
      ////////////// 
      
      double u_new = u * exp(sqrt(s_u) * arma::randn());
      
      double log_ratio_u = N * (log(u_new) - log(u)) - kappa/sigma * (pow(1 + u_new, sigma) - pow(1 + u, sigma)) - (N - K_N * sigma) * (log(1 + u_new) - log(1 + u));
      
      double accept_u = 1.0;
      if( arma::is_finite(log_ratio_u) ){
        if(log_ratio_u < 0){
          accept_u = exp(log_ratio_u);
        }
      }else{
        accept_u = 0.0;
      }
      
      u_accept = u_accept + accept_u;
      
      if( arma::randu() < accept_u ){
        u = u_new;
      }
      
      s_u = s_u + pow(g + 1,-ADAPT(1)) * (accept_u - ADAPT(2));
      if(s_u > exp(50)){
        s_u = exp(50);
      }else{
        if(s_u < exp(-50)){
          s_u = exp(-50);
        }
      }
    }
    
    
    
    if( (g + 1) % 100 == 0 ){
      Rcout << "g = " << g + 1 << "\n";
      
      Rcout << "min(theta) = " << min(theta) << "\n";
      Rcout << "max(theta) = " << max(theta) << "\n";
      
      Rcout << "U_Z = " << U_Z.t() << "\n";
      Rcout << "U_Y = " << U_Y << "\n";
      Rcout << "U_Y_accept = " << U_Y_accept.t()/(g + 1) << "\n";
      
      Rcout << "alpha_Y = " << alpha_Y.t() << "\n";
      
      Rcout << "K_N = " << K_N << "\n";
      Rcout << "nj = " << nj.t() << "\n";
      
      Rcout << "kappa = " << kappa << "\n";
      Rcout << "sigma = " << sigma << "\n";
      
      Rcout << "sig2_Y = " << sig2_Y.t() << "\n";
      Rcout << "sig2_Z = " << sig2_Z << "\n";
    }
    
    
    
    
    //Save output for this iteration
    if( (g + 1 > (n_burn1 + n_burn2)) & (((g + 1 - n_burn1 - n_burn2) / thin - floor((g + 1 - n_burn1 - n_burn2) / thin)) == 0 )){
      
      iter = (g + 1 - n_burn1 - n_burn2)/thin - 1;
      
      //Field
      Sigma_theta0_out(iter) = Sigma_theta0;
      
      // Lists
      // Rcout << "Lists\n";
      
      // CLONE is important with list of lists
      Y_out[iter] = clone(Y_bis);
      b_Z_star_out[iter] = b_Z_star;
      theta0_star_out[iter] = theta0_star;
      nj_out[iter] = nj;
      
      // Cubes 
      // Rcout << "Cubes\n";
      Z_out.slice(iter) = Z;
      if(n_subscales_main > 1){
        theta_1_out.slice(iter) = theta.col(0);
        theta_2_out.slice(iter) = theta.col(1);
      }else{
        theta_1_out.slice(iter) = theta.col(0);
      }
      beta_Y_out.slice(iter) = beta_Y;
      m_theta0_out.slice(iter) = m_theta0;
      U_Y_out.slice(iter) = U_Y;
      eta_Y_out.slice(iter) = eta_Y;
      
      // Matrices
      // Rcout << "Matrices\n";
      sig2_Y_out.row(iter) = sig2_Y.t();
      s_out.row(iter) = s.t();
      U_Z_out.row(iter) = U_Z.t();
      eta_Z_out.row(iter) = eta_Z.t();
      alpha_Y_out.row(iter) = alpha_Y.t();
      mu_alpha_Y_out.row(iter) = mu_alpha_Y.t();
      xi_Y_out.row(iter) = xi_Y.t();
      
      // Vectors
      // Rcout << "Vectors\n";
      sig2_Z_out(iter) = sig2_Z;
      xi_Z_out(iter) = xi_Z;
      K_N_out(iter) = K_N;
      
      kappa_out(iter) = kappa;
      if(!isDP){
        sigma_out(iter) = sigma;
        u_out(iter) = u;
      }
    }
    
    //Progress bar increment
    progr.increment(); 
  }
  
  //Print acceptance rates
  Rcout << "-- Accepptance Rates --\n";
  // Rcout << "Subject parameters:\n";
  // Rcout << "theta a.r. = " << theta_accept/n_tot << "\n";
  
  Rcout << "Item parameters:\n";
  Rcout << "alpha_Y a.r. = " << alpha_Y_accept.t()/n_tot << "\n";
  Rcout << "beta_Y a.r. = " << beta_Y_accept.t()/n_tot << "\n";
  
  Rcout << "xi_Z a.r. = " << xi_Z_accept/n_tot << "\n";
  Rcout << "xi_Y a.r. = " << xi_Y_accept.t()/n_tot << "\n";
  
  Rcout << "Regression coefficients (IRT):\n";
  Rcout << "U_Y a.r. = " << U_Y_accept.t()/n_tot << "\n";
  
  if(!isDP){
    Rcout << "u a.r. = " << u_accept/n_tot << "\n";
    if(update_sigma){
      Rcout << "sigma a.r. = " << sigma_accept/n_tot << "\n";
    }
  }
  
  
  //Create lists outside. Max number of element is 20
  List YZ_List;
  YZ_List = List::create(Named("Z_out") = Z_out, Named("Y_out") = Y_out, Named("theta_1_out") = theta_1_out, Named("theta_2_out") = theta_2_out, Named("U_Z_out") = U_Z_out, Named("U_Y_out") = U_Y_out, Named("eta_Z_out") = eta_Z_out, Named("eta_Y_out") = eta_Y_out, Named("alpha_Y_out") = alpha_Y_out, Named("mu_alpha_Y_out") = mu_alpha_Y_out, Named("beta_Y_out") = beta_Y_out, Named("sig2_Z_out") = sig2_Z_out, Named("sig2_Y_out") = sig2_Y_out, Named("xi_Y_out") = xi_Y_out, Named("xi_Z_out") = xi_Z_out);
  List BNP_List;
  if(isDP){
    BNP_List = List::create(Named("s_out") = s_out, Named("nj_out") = nj_out, Named("K_N_out") = K_N_out, Named("kappa_out") = kappa_out, Named("b_Z_star_out") = b_Z_star_out, Named("theta0_star_out") = theta0_star_out);
  }else{
    BNP_List = List::create(Named("s_out") = s_out, Named("nj_out") = nj_out, Named("K_N_out") = K_N_out, Named("kappa_out") = kappa_out, Named("sigma_out") = sigma_out, Named("u_out") = u_out, Named("b_Z_star_out") = b_Z_star_out, Named("theta0_star_out") = theta0_star_out);
  }
  
  return List::create(Named("YZ_List") = YZ_List, Named("BNP_List") = BNP_List);
}

