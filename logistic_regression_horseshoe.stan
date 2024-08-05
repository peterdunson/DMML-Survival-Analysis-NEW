data {
  int<lower=0> N; // number of observations
  int<lower=0> K; // number of predictors
  matrix[N, K] X; // predictor matrix
  array[N] int<lower=0, upper=1> y; // binary outcome
}

parameters {
  vector[K] beta; // coefficients
  real alpha; // intercept
  real<lower=0> tau; // global shrinkage parameter
  vector<lower=0>[K] lambda; // local shrinkage parameters
}

transformed parameters {
  vector[K] beta_tilde;
  for (k in 1:K)
    beta_tilde[k] = beta[k] * lambda[k] * tau;
}

model {
  // Horseshoe prior
  tau ~ cauchy(0, 1);
  lambda ~ cauchy(0, 1);
  beta ~ normal(0, 1);
  alpha ~ normal(0, 1);
  
  y ~ bernoulli_logit(alpha + X * beta_tilde);
}

generated quantities {
  array[N] real y_pred;
  for (n in 1:N) {
    y_pred[n] = bernoulli_logit_rng(alpha + dot_product(X[n], beta_tilde));
  }
}


