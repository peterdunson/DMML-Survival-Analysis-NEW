// logistic_regression_horseshoe.stan

data {
  int<lower=0> N; // number of data points
  int<lower=0> K; // number of predictors
  matrix[N, K] X; // predictor matrix
  array[N] int<lower=0, upper=1> y; // outcome vector
}
parameters {
  real alpha; // intercept
  vector[K] z; // auxiliary variables for horseshoe prior
  real<lower=0> tau; // global shrinkage parameter
  vector<lower=0>[K] lambda; // local shrinkage parameters
}
transformed parameters {
  vector[K] beta;
  beta = z .* (lambda * tau);
}
model {
  // Priors
  alpha ~ normal(0, 1); // Prior for intercept
  tau ~ cauchy(0, 1); // Prior for global shrinkage parameter
  lambda ~ cauchy(0, 1); // Prior for local shrinkage parameters
  z ~ normal(0, 1); // Prior for auxiliary variables
  
  // Likelihood
  y ~ bernoulli_logit(alpha + X * beta);
}
generated quantities {
  vector[N] y_pred;
  for (n in 1:N) {
    y_pred[n] = bernoulli_logit_rng(alpha + dot_product(X[n], beta));
  }
}
