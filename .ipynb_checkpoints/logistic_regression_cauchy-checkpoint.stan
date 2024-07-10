data {
  int<lower=0> N; // number of observations
  int<lower=0> K; // number of predictors
  matrix[N, K] X; // predictor matrix
  array[N] int<lower=0, upper=1> y; // outcome vector
}

parameters {
  real alpha; // intercept
  vector[K] beta; // coefficients
}

model {
  // Priors
  alpha ~ normal(0, 10); // prior for intercept, as suggested by Gelman et al.
  beta ~ cauchy(0, 2.5); // Cauchy prior for coefficients

  // Logistic regression likelihood
  y ~ bernoulli_logit(alpha + X * beta);
}

generated quantities {
  vector[N] y_pred;
  for (n in 1:N) {
    y_pred[n] = bernoulli_logit_rng(alpha + dot_product(X[n], beta));
  }
}

