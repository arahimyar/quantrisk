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
1. The ARMA-GARCH model parameters are fit via maximum likelihood estimation (MLE) with the assumption of Gaussian errors. 
2. The innovations from step 1 are extracted and used to estimate distribution parameters via MLE if a non-Gaussian distribution is being used. 

In addition to the standard Gaussian, the codebase currently supports the following distributions:
- **Student-t**, $lst(\mu, \tau^2, \nu)$
    - *Parameters*: $\mu$: location, $\tau^2$: scale, $\nu$, degrees-of-freedom.
    - *Optimization*: Since shocks must be standardized, we take $\mu = 0$, $\tau^2 = \frac{\nu-2}{\nu}$ and optimize over $\nu$.
- **Normal Inverse Gaussian (NIG)**, $NIG(\alpha, \beta, \delta, \mu)$
    - *Parameters*: $\alpha$: tail heaviness, $\beta$: asymmetry parameter, $\delta$: scale, $\mu$: location.
    - *Optimization*: Let $\gamma = \sqrt{\alpha^2 - \beta^2}$. Then due to the standardization requirement we take $\delta = \gamma^3/\alpha^2$ and $\mu = -\delta\beta / \gamma$, and optimize over $\alpha$ and $\beta$.

Users can implement a distribution not listed above by adding to `quantrisk/distributions.py`. To do so, create a subclass inheriting from the base abstract class `Distribution` and implement all abstract methods.


## Empirical Results

The model was evaluated on daily log returns of 100 S&P 500 stocks chosen from a variety of sectors. 

<details>
<summary><b>Click to view the 100 S&P 500 assets tested</b></summary>
<p>
<code>AAPL, ABBV, ABT, ACN, ADBE, AIG, AMAT, AMD, AMGN, AMT, AMZN, AVGO, AXP, BA, BAC, BK, BKNG, BLK, BMY, C, CAT, CL, CMCSA, COF, COP, COST, CRM, CSCO, CVS, CVX, DE, DHR, DIS, DUK, EMR, FDX, GE, GILD, GM, GOOG, GOOGL, GS, GD, HD, HON, IBM, INTC, INTU, ISRG, JNJ, JPM, KO, LLY, LMT, LOW, LRCX, MA, MCD, MDLZ, MDT, MET, META, MMM, MO, MS, MSFT, MU, NEE, NFLX, NKE, NOW, NVDA, ORCL, PEP, PFE, PG, PLTR, PM, QCOM, RTX, SBUX, SCHW, SO, SPG, T, TMO, TMUS, TSLA, TXN, UBER, UNH, UNP, UPS, USB, V, VZ, WFC, WMT, XOM</code>
</p>
</details>

For each asset we perform a rolling window backtest with a window size of 250 and a total of 750 out of sample tests. For each backtest we fit a ARMA(1,1)-GARCH(1,1) model using Gaussian, Student-t, and NIG errors. We assess model performance using various statistical tests. 

The results are given below. In each table we record the number of assets that passed the statistical test (at a 5% significance level) over the total number of assets. We also report the average run time.

### 95% VaR:
|Distribution | Kupiec | Binomial | Christoffersen | Conditional Coverage | KS | Average Run Time (all backtests) | Average Run Time (per backtest)|
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|**Gaussian** | 93/100 | 92/100 | 95/100 | 91/100 | 30/100 | 32.55s | 0.04s |
|**Student-t** | 92/100 | 89/100 | 94/100 | 92/100 | 96/100 | 36.35s | 0.05s |
|**NIG** | 98/100 | 95/100 | 92/100 | 97/100 | 99/100 | 53.85s | 0.07s |



### 99% VaR

|Distribution | Kupiec | Binomial | Christoffersen | Conditional Coverage | KS | Average Run Time (all backtests) | Average Run Time (per backtest)|
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|**Gaussian** | 40/100 | 40/100 | 98/100 | 49/100 | 30/100 | 32.6s | 0.04s |
|**Student-t** | 88/100 | 88/100 | 96/100 | 92/100 | 96/100 | 36.44s | 0.05s |
|**NIG** | 91/100 | 91/100 | 93/100 | 95/100 | 99/100 | 54.12s | 0.07s |

As expected, the Gaussian performs poorly at extreme VaR levels, primarily due to its light tails. We observe excellent results when using the NIG distribution while maintaining efficient runtimes.


## Usage Guide

### Installation
1. Clone the repository
```bash
git clone https://github.com/arahimyar/quantrisk.git
cd quantrisk
```
2. Install dependencies via pip: `pip install -r requirements.txt`.

### Requirements

See `requirements.txt` for a list of required packages and their versions. 

### Demo

See `demo.ipynb` for a walked-through example.
