import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, norminvgauss, t, chi2
from scipy.optimize import minimize
import os
import time

#######
# Stats
#######

def kupiec(p, exceptions):
    num_exceptions = np.sum(exceptions)
    n = len(exceptions)
    if num_exceptions == 0:
        return 1.0

    ratio = num_exceptions / n
    test_stat = (n - num_exceptions) * np.log(1-p) + num_exceptions * np.log(p) - (n - num_exceptions) * np.log(1 - ratio) - num_exceptions * np.log(ratio)
    test_stat = -2*test_stat
    p_value = 1 - chi2.cdf(test_stat, df=1)

    return test_stat, p_value

def christoffersen(exceptions):
    n00, n01, n10, n11 = 0, 0, 0, 0
    for i in range(1, len(exceptions)):
        prev = exceptions[i-1]
        curr = exceptions[i]

        if prev == 0 and curr == 0:
            n00 += 1
        elif prev == 0 and curr == 1:
            n01 += 1
        elif prev == 1 and curr == 0:
            n10 += 1
        elif prev == 1 and curr == 1:
            n11 += 1

    pi_0 = n01/(n00 + n01)
    pi_1 = n11 / (n10 + n11)
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    test_stat_num = (1-pi)**(n00 + n10) * pi**(n01 + n11)
    test_stat_denom = (1-pi_0)** n00 * pi_0**n01 * (1-pi_1)**n10 * pi_1**n11
    test_stat = -2*np.log(test_stat_num / test_stat_denom)
    p_value = 1 - chi2.cdf(test_stat, df=1)

    return test_stat, p_value

def joint_kupiec_christoffersen(exceptions, p, test_stat_kupiec = None, test_stat_christoffersen = None):
    if test_stat_kupiec is None:
        test_stat_kupiec = kupiec(exceptions, p)[0]
    if test_stat_christoffersen is None:
        test_stat_christoffersen = christoffersen(exceptions)[0]
    test_stat = test_stat_kupiec + test_stat_christoffersen
    p_value = 1 - chi2.cdf(test_stat, df=2)
    return test_stat, p_value



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
    nu, = params
    scale = np.sqrt((nu - 2) / nu)
    log_likelihood = t.logpdf(data, df = nu, scale = scale)
    return -np.sum(log_likelihood)

def optimize_StudentT(data, initial = None):
    if initial is None:
        initial = 10
    bounds = [(2.01, np.inf)]
    result = minimize(StudentT_obj, initial, args=(data,), method='L-BFGS-B', bounds=bounds)
    return result.x

def VaR_crit_StudentT(q, nu):
    return t.ppf(q, df = nu, scale = np.sqrt((nu - 2) / nu))



### Change this to include variable q
def backtest(data, backtest_length, window):
    pass


path = "/Users/hasibrahimyar/Desktop/ARMAGARCH2/DATA/2019_2025"
stock = "AAPL"
data = path + stock + ".npy"
X = np.load(data)
failed95_NIG = 0
failed99_NIG = 0
failed95_T = 0
failed99_T = 0
failed95_Norm = 0
failed99_Norm = 0
j = 0
files = sorted([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.endswith(".npy")])

start_time = time.perf_counter()
for file in files:
    filepath = os.path.join(path, file)
    X = np.load(filepath)
    print(f"Loaded {file}, shape: {X.shape}")
    window = 250
    backtest_length = 1000
    start_index = len(X) - backtest_length - window
    end_index = len(X) - window
    exceptions_NIG_95 = np.zeros(backtest_length)
    exceptions_NIG_99 = np.zeros(backtest_length)
    exceptions_T_95 = np.zeros(backtest_length)
    exceptions_T_99 = np.zeros(backtest_length)
    exceptions_Norm_95 = np.zeros(backtest_length)
    exceptions_Norm_99 = np.zeros(backtest_length)
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
        nu, = params_StudentT

        zNIG_95 = forecast_mean + np.sqrt(forecast_variance) * qNIG_95
        zNIG_99 = forecast_mean + np.sqrt(forecast_variance) * qNIG_99
        zT_95 = forecast_mean + np.sqrt(forecast_variance) * qT_95 
        zT_99 = forecast_mean + np.sqrt(forecast_variance) * qT_99
        zNorm_95 = forecast_mean + np.sqrt(forecast_variance) * qNorm_95
        zNorm_99 = forecast_mean + np.sqrt(forecast_variance) * qNorm_99

        if X[i+window] <= zNIG_95:
            exceptions_NIG_95[i-start_index] += 1
        if X[i+window] <= zNIG_99:
            exceptions_NIG_99[i-start_index] += 1
        if X[i+window] <= zT_95:
            exceptions_T_95[i-start_index] += 1
        if X[i+window] <= zT_99:
            exceptions_T_99[i-start_index] += 1
        if X[i+window] <= zNorm_95:
            exceptions_Norm_95[i-start_index] += 1
        if X[i+window] <= zNorm_99:
            exceptions_Norm_99[i-start_index] += 1
    j += 1
    print(f"NIG VaR 95 Exceedances: {np.sum(exceptions_NIG_95)}/{backtest_length}")
    kupiac_95_NIG = kupiec(0.05, exceptions_NIG_95)
    christoffersen_95_NIG = christoffersen(exceptions_NIG_95)
    joint_95_NIG = joint_kupiec_christoffersen(exceptions_NIG_95, .05, kupiac_95_NIG[0], christoffersen_95_NIG[1])
    if kupiac_95_NIG[1] <= 0.05:
        failed95_NIG += 1
    print(f"Kupiac p-value NIG VaR95: {kupiac_95_NIG[1]}, Christofferson p-value NIG VaR95: {christoffersen_95_NIG[1]}, Joint p-value NIG VaR95: {joint_95_NIG[1]}")
    print(f"Failed NIG 95: {failed95_NIG}/{j}")

    print(f"NIG VaR 99 Exceedances: {np.sum(exceptions_NIG_99)}/{backtest_length}")
    kupiac_99_NIG = kupiec(0.01, exceptions_NIG_99)
    christoffersen_99_NIG = christoffersen(exceptions_NIG_99)
    joint_99_NIG = joint_kupiec_christoffersen(exceptions_NIG_99, .01, kupiac_99_NIG[0], christoffersen_99_NIG[1])
    if kupiac_99_NIG[1] <= 0.05:
        failed99_NIG += 1
    print(f"Kupiac p-value NIG VaR99: {kupiac_99_NIG[1]}, Christofferson p-value NIG VaR99: {christoffersen_99_NIG[1]}, Joint p-value NIG VaR99: {joint_99_NIG[1]}")
    print(f"Failed NIG 99: {failed99_NIG}/{j}")


    print(f"Student T VaR 95 Exceedances: {np.sum(exceptions_T_95)}/{backtest_length}")
    kupiac_95_T = kupiec(0.05, exceptions_T_95)
    christoffersen_95_T = christoffersen(exceptions_T_95)
    joint_95_T = joint_kupiec_christoffersen(exceptions_T_95, .05, kupiac_95_T[0], christoffersen_95_T[1])
    if kupiac_95_T[1] <= 0.05:
        failed95_T += 1
    print(f"Kupiac p-value Student T VaR95: {kupiac_95_T[1]}, Christofferson p-value Student T VaR95: {christoffersen_95_T[1]}, Joint p-value Student T VaR95: {joint_95_T[1]}")
    print(f"Failed Student T 95: {failed95_T}/{j}")

    print(f"Student T VaR 99 Exceedances: {np.sum(exceptions_T_99)}/{backtest_length}")
    kupiac_99_T = kupiec(0.01, exceptions_T_99)
    christoffersen_99_T = christoffersen(exceptions_T_99)
    joint_99_T = joint_kupiec_christoffersen(exceptions_T_99, .01, kupiac_99_T[0], christoffersen_99_T[1])
    if kupiac_99_T[1] <= 0.05:
        failed99_T += 1
    print(f"Kupiac p-value Student T VaR99: {kupiac_99_T[1]}, Christofferson p-value Student T VaR99: {christoffersen_99_T[1]}, Joint p-value Student T VaR99: {joint_99_T[1]}")
    print(f"Failed Student T 99: {failed99_T}/{j}")

    print(f"Student Norm VaR 95 Exceedances: {np.sum(exceptions_Norm_95)}/{backtest_length}")
    kupiac_95_Norm = kupiec(0.05, exceptions_Norm_95)
    christoffersen_95_Norm = christoffersen(exceptions_Norm_95)
    joint_95_Norm = joint_kupiec_christoffersen(exceptions_Norm_95, .05, kupiac_95_Norm[0], christoffersen_95_Norm[1])
    if kupiac_95_Norm[1] <= 0.05:
        failed95_Norm += 1
    print(f"Kupiac p-value Norm VaR95: {kupiac_95_Norm[1]}, Christofferson p-value Norm VaR95: {christoffersen_95_Norm[1]}, Joint p-value Norm VaR95: {joint_95_Norm[1]}")
    print(f"Failed Gaussian 95: {failed95_Norm}/{j}")

    print(f"Student Norm VaR 99 Exceedances: {np.sum(exceptions_Norm_99)}/{backtest_length}")
    kupiac_99_Norm = kupiec(0.01, exceptions_Norm_99)
    christoffersen_99_Norm = christoffersen(exceptions_Norm_99)
    joint_99_Norm = joint_kupiec_christoffersen(exceptions_Norm_99, .01, kupiac_99_Norm[0], christoffersen_99_Norm[1])
    if kupiac_99_Norm[1] <= 0.05:
        failed99_Norm += 1
    print(f"Kupiac p-value Norm VaR99: {kupiac_99_Norm[1]}, Christofferson p-value Norm VaR99: {christoffersen_99_Norm[1]}, Joint p-value Norm VaR99: {joint_99_Norm[1]}")
    print(f"Failed Gaussian 99: {failed99_Norm}/{j}")
    print("****")
end_time = time.perf_counter()
elapsed = end_time - start_time
print(f"Total Time: {elapsed}")

### To do:
### (1) Implement Student T
### (2) Add Cython
### (3) Add statistics (Binomial, Kupiec, Christofferson, KS)
### (4) Add visualizations
### (5) Implement warm starts in rolling window
### (6) Fix issue with unpacking nu
### (7) Add CVaR

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