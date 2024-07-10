
data {
  int<lower=0> N; // number of observations
  int<lower=0> K; // number of predictors
  matrix[N, K] X; // predictor matrix
  int<lower=0, upper=1> y[N]; // outcome vector
}
parameters {
  real alpha; // intercept
  vector[K] beta; // coefficients for predictors
}
model {
  y ~ bernoulli_logit(alpha + X * beta); // likelihood
}
generated quantities {
  vector[N] y_pred_prob;
  for (n in 1:N)
    y_pred_prob[n] = inv_logit(alpha + dot_product(X[n], beta));
}
