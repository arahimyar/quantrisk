import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, norminvgauss, t
from scipy.optimize import minimize
import os


### X_t = xi + phi * X_{t-1} + \theta \epsilon_{t-1} + \epsilon_t
### sigma_t^2 = C + A * epsilon_{t-1}^2 + B * \sigma_{t-1}^2 +
### Assume xi = 0 since we're modelling log returns

def ARMAGARCH(data, phi, theta, A, B):
    data_nomean = data - np.mean(data)
    n = len(data)
    e_t = np.zeros(n)
    sigma2_t = np.zeros(n)
    sigma2_t[0] = np.var(data_nomean)
    C = sigma2_t[0] * (1 - A - B)
    for i in range(1, n):
        e_t[i] = data_nomean[i] - phi * data_nomean[i-1] - theta * e_t[i-1]
        sigma2_t[i] = C + A * e_t[i-1] * e_t[i-1] + B * sigma2_t[i-1]
    return e_t, sigma2_t

def ARMAGARCH_obj(params, data):
    phi, theta, A, B = params
    if A + B >= 1:
        return 1e10
    e_t, sigma2_t = ARMAGARCH(data, phi, theta, A, B)
    return -np.sum(norm.logpdf(e_t[1:], loc=0, scale=np.sqrt(sigma2_t[1:])))

def optimize(data, initial = None):
    bounds = [(-0.999, 0.999), (-0.999, 0.999), (0.001, 0.999), (0.001,  0.999)]
    if initial is None:
        initial = np.asarray([0.1, 0.1, 0.05, 0.8])
    result = minimize(ARMAGARCH_obj, initial, args=(data,), method='L-BFGS-B', bounds=bounds)
    return result.x

def one_day_forecast(data, phi, theta, A, B):
    sigma2 = np.var(data)
    C = sigma2 * (1 - A - B)
    xi = np.mean(data) * (1-phi)
    eps = 0
    for i in range(1, len(data)):
        sigma2 = C + A * eps**2 + B * sigma2
        eps = data[i] - xi - phi * data[i-1] - theta * eps
    forecast_mean = xi + phi*data[-1] + theta * eps
    forecast_variance = C + A * eps**2 + B * sigma2
    return forecast_mean, forecast_variance

### NIG with mean zero and unit variance
### This forces mu = -delta * beta / gamma = -beta * gamma^2/alpha^2 
### delta = gamma^3/alpha^2
def NIG_obj(params, data):
    alpha, beta = params
    if abs(beta) > alpha:
        return 1e10
    gamma = np.sqrt(alpha**2 - beta**2)
    scale = gamma**3 / alpha**2
    loc = - (scale * beta) / gamma
    log_likelihood = norminvgauss.logpdf(data, alpha, beta, loc = loc, scale = scale)
    return -np.sum(log_likelihood)

def optimize_NIG(data, initial = None):
    if initial is None:
        initial = [5, 0]
    bounds = [(0.1, np.inf), (-np.inf, np.inf)]
    result = minimize(NIG_obj, initial, args=(data,), method='L-BFGS-B', bounds=bounds)
    return result.x

def VaR_crit_NIG(q, alpha ,beta):
    gamma = np.sqrt(alpha**2 - beta**2)
    scale = gamma**3 / alpha**2
    loc = - (scale * beta) / gamma
    return norminvgauss.ppf(q, alpha, beta, loc=loc, scale=scale)

### Student t
def StudentT_obj(params, data):
    nu = params
    log_likelihood = t.logpdf(data, nu)
    return -np.sum(log_likelihood)

def optimize_StudentT(data, initial = None):
    if initial is None:
        initial = 5
    bounds = [(2.01, np.inf)]
    result = minimize(StudentT_obj, initial, args=(data,), method='L-BFGS-B', bounds=bounds)
    return result.x

def VaR_crit_StudentT(q, nu):
    return t.ppf(q, nu)



### Change this to include variable q
def backtest(data, backtest_length, window):
    pass


path = "/Users/hasibrahimyar/Desktop/ARMAGARCH2/DATA/2019_2025"
stock = "AAPL"
data = path + stock + ".npy"
X = np.load(data)
failed95 = 0
failed99 = 0
j = 0
files = sorted([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.endswith(".npy")])

for file in files:
    filepath = os.path.join(path, file)
    X = np.load(filepath)
    print(f"Loaded {file}, shape: {X.shape}")
    VaR95 = 0
    VaR99 = 0
    window = 250
    backtest_length = 1000
    start_index = len(X) - backtest_length - window
    end_index = len(X) - window
    for i in range(start_index, end_index):
        ARMAGARCH_params = optimize(X[i:i+window])
        e_t, sigma2_t = ARMAGARCH(X[i:i+window], *ARMAGARCH_params)
        z_t = e_t/np.sqrt(sigma2_t)
        params_NIG = optimize_NIG(z_t)
        params_StudentT = optimize_StudentT(z_t)
        forecast_mean, forecast_variance = one_day_forecast(X[i:i+window], *ARMAGARCH_params)

        qNIG_95 = VaR_crit_NIG(0.05, *params_NIG)
        qNIG_99 = VaR_crit_NIG(0.01, *params_NIG)
        qT_95 = VaR_crit_StudentT(0.05, *params_StudentT)
        qT_99 = VaR_crit_StudentT(0.01, *params_StudentT)
        qNorm_95 = norm.ppf(0.05)
        qNorm_99 = norm.ppf(0.01)

        zNIG_95 = forecast_mean + np.sqrt(forecast_variance) * qNIG_95
        zNIG_99 = forecast_mean + np.sqrt(forecast_variance) * qNIG_99
        zT_99 = forecast_mean + np.sqrt(forecast_variance) * qT_95
        zT_95 = forecast_mean + np.sqrt(forecast_variance) * qT_99
        zNorm_99 = forecast_mean + np.sqrt(forecast_variance) * qNorm_95
        zNorm_95 = forecast_mean + np.sqrt(forecast_variance) * qNorm_99

        if X[i+window] <= z_95:
            VaR95 += 1

        if X[i+window] <= z_99:
            VaR99 += 1
    print(f"VaR 95 Exceedances: {VaR95}/{backtest_length}")
    print(f"VaR 99 Exceedances: {VaR99}/{backtest_length}")
    if VaR95 < 36 or VaR95 > 64:
        failed95 += 1
    if VaR99 < 4 or VaR99 > 16:
        failed99 += 1
    j+=1
    print(f"Failed 95: {failed95}/{j}")
    print(f"Failed 99: {failed99}/{j}")
    print("*********")


### To do:
### (1) Implement Student T
### (2) Add Cython
### (3) Add statistics (Binomial, Kupiec, Christofferson, KS)
### (4) Add visualizations

# exceedances95 = data <= VaR95
# exceedances99 = data <= VaR99
# plt.plot(X[-1600:],  color = "black")
# plt.plot(VaR95_locs, color = "red")
# plt.plot(VaR99_locs, color = "blue")
# plt.scatter(np.arange(1600)[exceedances], data[exceedances], 
#             color='red', marker='x', s=100, label='Exceedance')
# plt.show()

# plt.figure(figsize=(8,5))
# plt.hist(z_t[1:], bins=50, density=True, alpha=0.6, color='blue', edgecolor='black')
# plt.title("Histogram of ARMA-GARCH Residuals")
# plt.xlabel("Residual (e_t)")
# plt.ylabel("Density")
# x = np.linspace(-7, 7, 500)
# alpha, beta = params_NIG
# gamma = np.sqrt(alpha**2 - beta**2)
# pdf = norminvgauss.pdf(x, alpha, beta, loc=-beta * gamma * gamma / (alpha ** 2), scale= gamma**3 / alpha**3)
# plt.plot(x, pdf, label=f'NIG α={alpha}, β={beta}')
# plt.show()

        # plt.figure(figsize=(8,5))
        # plt.hist(z_t[1:], bins=50, density=True, alpha=0.6, color='blue', edgecolor='black')
        # plt.title("Histogram of ARMA-GARCH Residuals")
        # plt.xlabel("Residual (e_t)")
        # plt.ylabel("Density")
        # x = np.linspace(-7, 7, 500)
        # alpha, beta = params_NIG
        # gamma = np.sqrt(alpha**2 - beta**2)
        # pdf = norminvgauss.pdf(x, alpha, beta, loc=-beta * gamma * gamma / (alpha ** 2), scale= gamma**3 / alpha**3)
        # plt.plot(x, pdf, label=f'NIG α={alpha}, β={beta}')
        # plt.show()