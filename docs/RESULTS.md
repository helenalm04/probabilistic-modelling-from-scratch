# Results and Interpretation

This project contains two probabilistic modelling experiments: a Bernoulli mixture model for binary images and Gaussian Process regression for a one-dimensional signal.

## 1. Bernoulli mixture model on binarised MNIST

The first notebook implements a Bernoulli mixture model for binary image vectors. Each mixture component represents a latent digit-like cluster, with one Bernoulli probability per pixel.

### Main observations

- The E-step is implemented in log-space using log-sum-exp. This avoids numerical underflow when multiplying many small Bernoulli probabilities across 784 pixels.
- For a subset of three MNIST digits, using three mixture components gives a natural and interpretable model: the learned component means behave like prototype digit templates.
- Increasing the number of components can reduce the training loss, but it also increases flexibility and can make the model more sensitive to local optima.
- Different random initialisations can converge to different final losses, which is expected because EM optimises a non-convex objective.
- Sampling from the fitted model shows the limitation of the conditional independence assumption: generated digits are recognisable, but can look noisy because pixels are sampled independently given the component.

## 2. Gaussian Process regression on `hr2.txt`

The second notebook implements Gaussian Process regression from scratch and compares a squared exponential kernel with a Laplace-style kernel.

The included `hr2.txt` file contains a scalar time series. In the notebook, the series is mean-centred, then a subset of points is used as observations for GP fitting.

### Kernel interpretation

- The squared exponential kernel assumes very smooth functions, so posterior samples and posterior means are smoother.
- The Laplace-style kernel allows more local irregularity. This can be useful for rougher data where neighbouring observations change more sharply.
- The lengthscale controls how quickly correlation decays with distance; the variance controls vertical amplitude; the observation-noise variance controls how much deviation from the latent function is expected.

### Quantitative comparison

| Kernel | Negative log marginal likelihood | MSE |
|---|---:|---:|
| Squared exponential | 419.35 | 3.43 |
| Laplace-style | 439.57 | 2.41 |

The squared exponential kernel achieved the lower negative log marginal likelihood, suggesting a stronger overall probabilistic fit under the marginal likelihood objective. The Laplace-style kernel achieved the lower MSE, which is plausible because the signal is relatively rough and the Laplace-style covariance can follow more local variation.

## What this project demonstrates

The value of the project is not only the final metrics. It demonstrates how to:

- implement probabilistic models from first principles;
- handle numerical stability issues;
- optimise model hyperparameters;
- compare models using multiple metrics;
- interpret generative model behaviour rather than treating models as black boxes.
