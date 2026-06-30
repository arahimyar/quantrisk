# quantrisk Library

This library supports a variety of tools for risk analysis in financial time series. In particular, it features a manual implementation of ARMA(1,1)-GARCH(1,1) with support for both Gaussian and non-Gaussian (e.g. Student-T, Normal Inverse Gaussian) shocks for use in Value-at-Risk (VaR) and conditional Value-at-Risk (CVaR, also known as expected shortfall). Additionally, it features built-in routines for convenient rolling window backtests, data visualization, and rigorous statistical tests (e.g. binomial, Kupiec, Christoffersen, Conditional Coverage, and Kolmogorov-Smirnov).

The code is written with a clean, modular, object-oriented design; this framework decouples the optimization logic from the underlying statistical models, making it trivial for contributors and users to use additional distributions not already featured.

## Model Framework

### Notation

We use the following notation for the ARMA(1,1)-GARCH(1,1) model:

$$r_t = \mathbb{E}[r_{t} \mid \mathcal{F}_{t-1}] + \varepsilon_t = \xi + \phi r_{t-1} + \theta \varepsilon_{t-1} + \varepsilon_{t}$$
$$\sigma_{t}^2 = C + B \sigma_{t-1}^2 + A\varepsilon_{t-1}^2$$
$$\varepsilon_t = \sigma_t z_t$$
$$z_t \sim \mathcal{D}(0,1)$$
Here $\mathcal{F}_{t-1}$ denotes the filtration at time $t-1$, and $\mathcal{D}(0,1)$ refers to any standardized (i.e. mean zero and unit variance) continuous distribution. We refer to $\{\varepsilon_t\}$ as the **innovations** and $\{z_t\}$ as the **shocks**.

For tail risk, we let $\alpha$ denote the tail probability so that, for example, 99% VaR corresponds to $\alpha = 0.01$.

### Implementation

The ARMA(1,1)-GARCH(1,1) fitting is done in a two step procedure:
1. The ARMA-GARCH model parameters are fit via maximum likelihood esimation (MLE) with the assumption of Gaussian errors. 
2. The innovations from step 1 are extracted and used to estimate distribution parameters via MLE if a non-Gaussian distribution is being used. 

In addition to the standard Gaussian, the codebase currently supports the following distributions:
- **Student-t**, $lst(\mu, \tau^2, \nu)$
    - *Parameters*: $\mu$: location, $\tau^2$: scale, $\nu$, degrees-of-freedom.
    - *Optimization*: Since shocks must be standarized, we take $\mu = 0$, $\tau^2 = \frac{\nu-2}{\nu}$ and optimize over $\nu$.
- **Normal Inverse Gaussian (NIG)**, $NIG(\alpha, \beta, \delta, \mu)$
    - *Parameters*: $\alpha$: tail heaviness, $\beta$: asymmetry parameter, $\delta$: scale, $\mu$: location.
    - *Optimization*: Let $\gamma = \sqrt{\alpha^2 - \beta^2}. Then due to the standarization requirement we take $\delta = \gamma^3/\alpha^2$ and $\mu = -\delta\beta / \gamma$, and optimize over $\alpha$ and $\beta$.

Users can implement a distribution not listed above by adding to `quantrisk/distributions.py`. To do so, create a subclass inheriting from the base abstract class `Distribution` and implement all abstract methods.


## Empirical Results

## Usage Guide

### Installation
1. Clone the repository
```bash
git clone [https://github.com/your-username/quantrisk.git](https://github.com/your-username/quantrisk.git)
cd quantrisk
```
2. Install dependencies via pip: `pip install -r requirements.txt`.

### Requirements

See `requirements.txt` for a list of required packages and their versions. 

### Demo

See `demo.ipynb` for a walked-through example.
