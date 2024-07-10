// logistic_regression_cauchy.stan

data {
  int<lower=0> N; // number of observations
  int<lower=0> K; // number of predictors
  matrix[N, K] X; // predictor matrix
  array[N] int<lower=0, upper=1> y; // binary outcome
}

parameters {
  vector[K] beta; // coefficients
  real alpha; // intercept
}

model {
  beta ~ cauchy(0, 2.5);
  alpha ~ normal(0, 10);
  y ~ bernoulli_logit(alpha + X * beta);
}

generated quantities {
  array[N] real y_pred;
  for (n in 1:N) {
    y_pred[n] = bernoulli_logit_rng(alpha + dot_product(X[n], beta));
  }
}



