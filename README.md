# Probabilistic Modelling from Scratch

A compact portfolio project implementing two probabilistic modelling pipelines directly in Python. The focus is on clear mathematical implementation, numerical stability, reproducible experiments, and interpretation of model behaviour.

## Projects

### 1. Bernoulli Mixture Model for Binarised MNIST

Notebook: `notebooks/01_bernoulli_mixture_em_mnist.ipynb`

This notebook implements a Bernoulli mixture model trained with the Expectation-Maximisation algorithm. It models binarised MNIST images as observations generated from latent mixture components.

Key ideas covered:

- E-step and M-step implementation in NumPy;
- stable responsibility calculations using log-sum-exp;
- negative log-likelihood monitoring;
- comparison of model capacity, sample size and random initialisation;
- visualisation of learned component templates;
- sampling synthetic digit-like images from the fitted model.

### 2. Gaussian Process Regression on a One-Dimensional Signal

Notebook: `notebooks/02_gaussian_process_regression_from_scratch.ipynb`

This notebook implements Gaussian Process regression from scratch and compares two covariance kernels on a one-dimensional time series stored in `data/hr2.txt`.

Key ideas covered:

- squared exponential and Laplace-style covariance kernels;
- sampling from GP priors;
- negative log marginal likelihood optimisation;
- posterior mean and uncertainty bands;
- model comparison using negative log marginal likelihood and MSE.

## Results summary

A fuller write-up is available in [`docs/RESULTS.md`](docs/RESULTS.md). The main findings are:

- EM learns interpretable digit-like Bernoulli templates, and the likelihood improves steadily during training.
- The Bernoulli mixture model can generate recognisable samples, but the conditional independence assumption makes the digits noisy and fragmented.
- For the GP experiment, the squared exponential kernel gives the better marginal likelihood fit, while the Laplace-style kernel achieves lower MSE on the rougher one-dimensional signal.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── data/
│   ├── README.md
│   └── hr2.txt
├── docs/
│   └── RESULTS.md
└── notebooks/
    ├── 01_bernoulli_mixture_em_mnist.ipynb
    └── 02_gaussian_process_regression_from_scratch.ipynb
```

## Setup

Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Then launch Jupyter:

```bash
jupyter lab
```

## Skills demonstrated

- probabilistic modelling;
- Expectation-Maximisation;
- mixture models;
- Gaussian Processes;
- marginal likelihood optimisation;
- numerical stability and Cholesky-based linear algebra;
- model comparison and visualisation;
- clean notebook organisation.

## Data note

The GP notebook uses `data/hr2.txt`, a one-dimensional numeric signal. If you want to reuse the project on a different dataset, replace this file with any one-dimensional series and rerun the notebook.
