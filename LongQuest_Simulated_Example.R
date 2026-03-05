################################################################################
# Simulated example for the LongQuest joint model (Z-BMI trajectories + CEBQ PCM)
#
# This script replaces all real data loading in LongQuest_Simul.R with simulated data
# that mimic the structure described in the paper "Bayesian semi-parametric inference
# for joint modelling of childhood growth and appetite phenotypes".
#
# It prepares:
#   - Z: continuous longitudinal trajectories (Z-BMI) on 14 unequally spaced times (1–9y)
#   - Y_list: 4 time points of CEBQ ordinal responses (35 items, 5 categories)
#   - X_Z, X_Y: time-invariant covariates (with dummies, standardisation)
#   - B_Z: cubic B-spline basis for the longitudinal model
#   - indices for CEBQ subscales and the two main domains (Food Approach/Avoidance)
#
# Then it runs (from R package LongQuest):
#   LongQuest_Gibbs_CEBQ_biv(data_list, MCMC_list, Param_list)
#
# NOTE:
#   The sampler LongQuest_Gibbs_CEBQ_biv is not defined in this script; it is expected
#   to be provided by your local codebase / package (e.g., LongQuest).
################################################################################

rm(list = ls())

# --------------------------- Packages -----------------------------------------
req_pkgs <- c("mvnfast", "splines", "fastDummies")
for(p in req_pkgs){
  if(!requireNamespace(p, quietly = TRUE)){
    install.packages(p, repos = "https://cloud.r-project.org")
  }
}
library(mvnfast)
library(splines)
library(fastDummies)

# --------------------------- Helper: PCM sampler ------------------------------
# h in {0,1,2,3,4} corresponds to categories {1,2,3,4,5} in the returned Y.
# beta is a length m vector with beta[1]=0, other entries free; we use cumsum(beta).
rpcm_one <- function(theta, alpha, beta, x_eff = 0, m = 5){
  h <- 0:(m-1)
  beta_cum <- cumsum(beta)
  eta <- alpha * (h * theta - beta_cum) + h * x_eff
  eta <- eta - max(eta)
  pr <- exp(eta)
  pr <- pr / sum(pr)
  sample.int(m, size = 1, prob = pr) # returns 1..m
}

# --------------------------- 1) Dimensions & time grids ------------------------
set.seed(123)

N    <- 150                 # number of children
TZ   <- 14                  # Z-BMI time points (1–9y, unequally spaced)
TY   <- 4                   # questionnaire time points
J    <- 35                  # CEBQ items
m_Y  <- 5                   # Likert categories

Z_times <- c(c(12, 15, 18)/12, 2, 3, 4, 4.5, 5, 5.5, 6, 6.5, 7, 8, 9)  # years
Y_times <- c(1, 3, 5, 7)

# Cubic B-splines with knots fixed to extremes + median (as in the paper / original script)
B_Z <- bs(Z_times, degree = 3, knots = median(Z_times), intercept = FALSE)
df_B <- ncol(B_Z)

# --------------------------- 2) CEBQ subscales (as in original script) --------
FR_CEBQ  <- c(12, 14, 19, 28, 34)    # Food Responsiveness (FAp)
EOE_CEBQ <- c(2, 13, 15, 27)         # Emotional Over-Eating (FAp)
EF_CEBQ  <- c(5, 1, 20, 22)          # Enjoyment of Food (FAp)
DD_CEBQ  <- c(6, 29, 31)             # Desire to Drink (FAp)
SR_CEBQ  <- c(3, 17, 21, 26, 30)     # Satiety Responsiveness (FAv)
SE_CEBQ  <- c(4, 8, 18, 35)          # Slowness in Eating (FAv)
EUE_CEBQ <- c(9, 11, 23, 25)         # Emotional Under-Eating (FAv)
FF_CEBQ  <- c(7, 10, 16, 24, 32, 33) # Food Fussiness (FAv)

subScales_CEBQ <- list(FR_CEBQ, EOE_CEBQ, EF_CEBQ, DD_CEBQ, SR_CEBQ, SE_CEBQ, EUE_CEBQ, FF_CEBQ)
n_subscales <- length(subScales_CEBQ)

subscales_indices <- rep(NA_integer_, J)
for(j in 1:J){
  subscales_indices[j] <- which(sapply(subScales_CEBQ, function(x) j %in% x))
}

subscales_indices_main <- rep(1L, J)           # 1 = Food approach
subscales_indices_main[subscales_indices > 4] <- 2L # 2 = Food avoidance
n_subscales_main <- length(unique(subscales_indices_main)) # should be 2

# --------------------------- 3) Covariates -----------------------------------
GA  <- rnorm(N, mean = 39, sd = 1.5)
Edu <- sample(c("Secondary", "University+", "BelowSecondary"), N, replace = TRUE, prob = c(0.55, 0.30, 0.15))
Eth <- sample(c("Chinese", "Malay", "Indian"), N, replace = TRUE, prob = c(0.65, 0.20, 0.15))
Parity <- rbinom(N, 1, 0.45)
Sex    <- rbinom(N, 1, 0.52)

X_all <- data.frame(
  GA = GA,
  Maternal_Education_3_cat = Edu,
  Ethnicity = Eth,
  Parity = Parity,
  gender = Sex
)

X_all$GA <- as.numeric(scale(X_all$GA))

X_all_dum <- dummy_cols(
  X_all,
  select_columns = c("Maternal_Education_3_cat", "Ethnicity"),
  remove_first_dummy = TRUE,
  remove_selected_columns = TRUE
)

X_all_dum <- X_all_dum[, c("GA",
                           grep("^Maternal_Education_3_cat_", names(X_all_dum), value = TRUE),
                           grep("^Ethnicity_", names(X_all_dum), value = TRUE),
                           "Parity",
                           "gender")]

X_ZY <- as.matrix(X_all_dum)
q_ZY <- ncol(X_ZY)

# Exclude Sex from X_Z (as Z-BMI typically standardised by sex)
X_Z <- X_ZY[, -q_ZY, drop = FALSE]
X_Y <- X_ZY

q_Z <- ncol(X_Z)
q_Y <- ncol(X_Y)

# --------------------------- 4) Simulate Z trajectories -----------------------
b_mat <- rmvn(N, mu = rep(0, df_B), sigma = diag(df_B))
gamma_Z <- rnorm(q_Z, mean = 0, sd = 0.15)

sigma2_Z <- 0.30^2

Z_mean <- b_mat %*% t(B_Z) + (X_Z %*% gamma_Z) %*% t(rep(1, TZ))
Z <- Z_mean + matrix(rnorm(N * TZ, sd = sqrt(sigma2_Z)), N, TZ)

set.seed(1234)
miss_Z <- matrix(runif(N * TZ) < 0.08, N, TZ)
Z[miss_Z] <- NA
is_NA_Z <- 1L * is.na(Z)

# --------------------------- 5) Simulate questionnaire (longitudinal PCM) -----
theta1_0 <- rnorm(N, 0, 1)
theta2_0 <- rnorm(N, 0, 1)

theta_1 <- matrix(NA, N, TY)
theta_2 <- matrix(NA, N, TY)
for(t in 1:TY){
  theta_1[, t] <- theta1_0 + rnorm(N, 0, 0.6) + 0.10 * (t-1)
  theta_2[, t] <- theta2_0 + rnorm(N, 0, 0.6) - 0.05 * (t-1)
}

mu_alpha_sub <- rnorm(n_subscales, mean = 0, sd = 0.35)
alpha_j <- rep(NA, J)
for(j in 1:J){
  alpha_j[j] <- exp(rnorm(1, mean = mu_alpha_sub[subscales_indices[j]], sd = 0.4))
}

beta_j <- matrix(0, J, m_Y)
beta_j[, 2:m_Y] <- matrix(rnorm(J * (m_Y-1), mean = 0, sd = 0.7), J, m_Y-1)

gamma_Y <- matrix(rnorm(J * q_Y, mean = 0, sd = 0.08), q_Y, J)

Y_list <- vector("list", length = TY)
for(t in 1:TY){
  Y_t <- matrix(NA_integer_, N, J)
  for(i in 1:N){
    x_eff_i <- as.numeric(X_Y[i, ] %*% gamma_Y)  # length J
    for(j in 1:J){
      th <- if(subscales_indices_main[j] == 1L) theta_1[i, t] else theta_2[i, t]
      Y_t[i, j] <- rpcm_one(theta = th, alpha = alpha_j[j], beta = beta_j[j, ], x_eff = x_eff_i[j], m = m_Y)
    }
  }
  Y_list[[t]] <- Y_t
}

set.seed(4321)
for(t in 1:TY){
  miss_Y <- matrix(runif(N * J) < 0.04, N, J)
  Y_list[[t]][miss_Y] <- NA_integer_
}

# --------------------------- 6) Build lists for the sampler -------------------
data_list <- list(
  Z = Z,
  is_NA_Z = is_NA_Z,
  Y = Y_list,
  subscales_indices_main = subscales_indices_main - 1L,
  n_subscales_main = n_subscales_main,
  subscales_indices = subscales_indices - 1L,
  X_Z = X_Z,
  X_Y = X_Y,
  B_Z = B_Z
)

n_burn1 <- 100
n_burn2 <- 500
thin    <- 2
n_save  <- 250

s_init <- rep(0, N)  # 0-indexed initial allocations as in the original script
MCMC_list <- list(n_burn1 = n_burn1, n_burn2 = n_burn2, n_save = n_save, thin = thin, s_init = s_init)

# --------------------------- 7) Priors (mirroring original script defaults) ---
m_Y_beta <- m_Y - 1
mu_beta_Y <- rep(0, m_Y_beta)
Sigma_beta_Y <- diag(m_Y_beta)

sig2_alpha_Y <- 1
m_mu_alpha_Y <- 0
sig2_mu_alpha_Y <- 1

mm <- 1
ss <- 1
a_sig2_Z <- 2 + mm^2/ss
b_sig2_Z <- (1 + mm^2/ss) * mm

update_sig2_Y <- FALSE
a_sig2_Y <- rep(2 + mm^2/ss, n_subscales_main)
b_sig2_Y <- rep((1 + mm^2/ss) * mm, n_subscales_main)
sig2_Y <- rep(1, n_subscales_main)

update_P0 <- FALSE
nu_Sigma_theta0 <- rep(TY + 2, n_subscales_main)
Psi_Sigma_theta0 <- array(NA, dim = c(TY, TY, n_subscales_main))
for(is in 1:n_subscales_main){
  Psi_Sigma_theta0[,,is] <- 100 * diag(TY)
}
mu_m_theta0 <- matrix(0, n_subscales_main, TY)

m_b_Z <- rep(0, df_B)
Omega_b_Z <- 0.01 * diag(df_B)

isDP <- FALSE
update_kappa <- FALSE
update_sigma <- FALSE

kappa <- 1
a_kappa <- 1
b_kappa <- 1

sigma <- 0.75
a_sigma <- 18
b_sigma <- 2

update_s <- TRUE
process_P_list <- list(
  update_s = update_s,
  isDP = isDP,
  update_P0 = update_P0,
  update_kappa = update_kappa, kappa = kappa, a_kappa = a_kappa, b_kappa = b_kappa,
  update_sigma = update_sigma, sigma = sigma, a_sigma = a_sigma, b_sigma = b_sigma
)

alpha_Y_zeros <- FALSE
U_Y_zeros <- FALSE
use_horseshoe <- FALSE

Param_list <- list(
  alpha_Y_zeros = alpha_Y_zeros,
  U_Y_zeros = U_Y_zeros,
  use_horseshoe = use_horseshoe,
  mu_beta_Y = mu_beta_Y,
  Sigma_beta_Y = Sigma_beta_Y,
  m_mu_alpha_Y = m_mu_alpha_Y,
  sig2_mu_alpha_Y = sig2_mu_alpha_Y,
  sig2_alpha_Y = sig2_alpha_Y,
  a_sig2_Z = a_sig2_Z,
  b_sig2_Z = b_sig2_Z,
  update_sig2_Y = update_sig2_Y,
  sig2_Y = sig2_Y,
  a_sig2_Y = a_sig2_Y,
  b_sig2_Y = b_sig2_Y,
  process_P_list = process_P_list,
  m_b_Z = m_b_Z,
  Omega_b_Z = Omega_b_Z,
  mu_m_theta0 = mu_m_theta0,
  Psi_Sigma_theta0 = Psi_Sigma_theta0,
  nu_Sigma_theta0 = nu_Sigma_theta0
)

# --------------------------- 8) Run the model (requires LongQuest package installation) ------------------
library("LongQuest")


set.seed(8)
MCMC_output <- LongQuest_Gibbs_CEBQ_biv(data_list = data_list, MCMC_list = MCMC_list, Param_list = Param_list)
save(MCMC_output, file = "OUTPUT_LongQuest_SIMULATED.RData")
message("Done. Saved OUTPUT_LongQuest_SIMULATED.RData")


# --------------------------- 9) Quick sanity checks ---------------------------
cat("\nSimulated data summary:\n")
cat("N =", N, "\n")
cat("Z dims:", dim(Z)[1], "x", dim(Z)[2], "\n")
cat("Y dims at each time:", paste(sapply(Y_list, nrow), "x", sapply(Y_list, ncol), collapse = " | "), "\n")
cat("Missingness Z:", round(mean(is.na(Z))*100, 2), "%\n")
cat("Missingness Y:", round(mean(sapply(Y_list, function(x) mean(is.na(x))))*100, 2), "% (avg over times)\n")




#Plot Simulated BMI Trajectories
library(ggplot2)
library(reshape2)

# Convert BMI matrix to long format
df_bmi <- melt(Z)
colnames(df_bmi) <- c("id", "time_index", "BMI")

df_bmi$time <- Z_times[df_bmi$time_index]

# Spaghetti plot
ggplot(df_bmi, aes(x = time, y = BMI, group = id)) +
  geom_line(alpha = 0.15) +
  stat_summary(aes(group = 1), fun = mean, geom = "line",
               linewidth = 1.2) +
  labs(title = "Simulated BMI Trajectories",
       x = "Age / Time",
       y = "Z-BMI") +
  theme_minimal()



#Plot Distribution of Questionnaire Responses
wave <- 1
item <- 5

df_item <- data.frame(
  response = factor(Y_list[[wave]][,item])
)

ggplot(df_item, aes(x = response)) +
  geom_bar() +
  labs(title = paste("Distribution: Wave", wave,
                     "- Item", item),
       x = "Response Category",
       y = "Count") +
  theme_minimal()




#Heatmap of Mean Item Scores Over Time
library(dplyr)

mean_scores <- sapply(1:length(Y_list), function(t)
  colMeans(Y_list[[t]], na.rm = TRUE))

# Convert to long format
df_heat <- melt(mean_scores)
colnames(df_heat) <- c("Item", "Time", "MeanScore")

ggplot(df_heat, aes(x = Time, y = Item, fill = MeanScore)) +
  geom_tile() +
  scale_fill_viridis_c() +
  labs(title = "Mean Questionnaire Scores Over Time",
       x = "Time",
       y = "Item") +
  theme_minimal()






#Co-Clustering probabilities
library("salso")
BNP_List <- MCMC_output$BNP_List
YZ_List <- MCMC_output$YZ_List

# Binder partition
s_out <- BNP_List$s_out + 1

pij <- matrix(0, N, N)
for(it in 1:n_save){
  pij <- pij + outer(s_out[it,], s_out[it,], "==")
}
pij <- pij / n_save

Binder_partition <- c(salso(s_out, loss = binder()))

K_N_Binder <- length(unique(Binder_partition))
nj_Binder <- table(Binder_partition)
nj_Binder

# Re-label clusters based on size
Binder_partition <- match(Binder_partition, as.numeric(names(sort(nj_Binder, decreasing = TRUE))))
nj_Binder <- table(Binder_partition)
nj_Binder

sort_ind <- sort(Binder_partition, index.return = TRUE)$ix
pij_sorted <- pij[sort_ind,sort_ind]

par(mar = c(5,7.5,5,7.5))
image(pij_sorted, pty="s", col = rev(heat.colors(100)), main = "", cex.main = 3, cex.lab = 2, axes = FALSE)
image.plot(pij_sorted, pty="s", col = rev(heat.colors(100)), legend.only = TRUE, horizontal = TRUE, axis.args = list(cex.axis = 2), legend.width = 1)
