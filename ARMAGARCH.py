import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

class ARMAGARCH:
    def __init__(self):
        self.ARMA_params = {}
        self.GARCH_params = {}

    def compute_ARMAGARCH(self, data, phi, theta, A, B):
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

    def ARMAGARCH_obj(self, params, data):
        phi, theta, A, B = params
        if A + B >= 1:
            return 1e10
        e_t, sigma2_t = self.compute_ARMAGARCH(data, phi, theta, A, B)
        return -np.sum(norm.logpdf(e_t[1:], loc=0, scale=np.sqrt(sigma2_t[1:])))

    def fit(self, data, initial = None):
        bounds = [(-0.999, 0.999), (-0.999, 0.999), (0.001, 0.999), (0.001,  0.999)]
        if initial is None:
            initial = np.asarray([0.1, 0.1, 0.05, 0.8])
        result = minimize(self.ARMAGARCH_obj, initial, args=(data,), method='L-BFGS-B', bounds=bounds)

        if not result.success and "LNSRCH" in result.message:
            cold_start = np.array([0.1, 0.1, 0.05, 0.8])
            result = minimize(self.ARMAGARCH_obj, cold_start, args=(data,), method='L-BFGS-B', bounds=bounds)

        self.ARMA_params['phi'] = result.x[0]
        self.ARMA_params['theta'] = result.x[1]
        self.ARMA_params['xi'] = np.mean(data) * (1-self.ARMA_params['phi'])
        self.GARCH_params['A'] = result.x[2]
        self.GARCH_params['B'] = result.x[3]
        self.GARCH_params['C'] = np.var(data) * (1 - self.GARCH_params['A'] - self.GARCH_params['B'])
        return None

    def get_innovations(self, data):
        e_t, sigma2_t = self.compute_ARMAGARCH(data, self.ARMA_params['phi'], self.ARMA_params['theta'], self.GARCH_params['A'], self.GARCH_params['B'])
        return e_t, sigma2_t

    def get_shocks(self, innovations, variances):
        return innovations/np.sqrt(variances)

    def one_day_forecast(self, data, xi = None, phi = None, theta = None, C = None, A = None, B = None):
        xi = self.ARMA_params['xi'] if xi is None else xi
        phi = self.ARMA_params['phi'] if phi is None else phi
        theta = self.ARMA_params['theta'] if theta is None else theta
        C = self.GARCH_params['C'] if C is None else C
        A = self.GARCH_params['A'] if A is None else A
        B = self.GARCH_params['B'] if B is None else B
        eps = 0
        sigma2 = np.var(data)
        for i in range(1, len(data)):
            sigma2 = C + A * eps**2 + B * sigma2
            eps = data[i] - xi - phi * data[i-1] - theta * eps
        forecast_mean = xi + phi*data[-1] + theta * eps
        forecast_variance = C + A * eps**2 + B * sigma2
        return forecast_mean, forecast_variance


### X_t = xi + phi * X_{t-1} + \theta \epsilon_{t-1} + \epsilon_t
### sigma_t^2 = C + A * epsilon_{t-1}^2 + B * \sigma_{t-1}^2 +
### Assume xi = 0 since we're modelling log returns




######### NIG with mean zero and unit variance ##########

### This forces mu = -delta * beta / gamma = -beta * gamma^2/alpha^2 
### delta = gamma^3/alpha^2
# def NIG_obj(params, data):
#     alpha, beta = params
#     if abs(beta) > alpha:
#         return 1e10
#     gamma = np.sqrt(alpha**2 - beta**2)
#     scale = gamma**3 / alpha**2
#     loc = - (scale * beta) / gamma
#     log_likelihood = norminvgauss.logpdf(data, alpha, beta, loc = loc, scale = scale)
#     return -np.sum(log_likelihood)

# def optimize_NIG(data, initial = None):
#     if initial is None:
#         initial = [5, 0]
#     bounds = [(0.1, np.inf), (-np.inf, np.inf)]
#     result = minimize(NIG_obj, initial, args=(data,), method='L-BFGS-B', bounds=bounds)
#     return result.x

# def VaR_crit_NIG(q, alpha ,beta):
#     gamma = np.sqrt(alpha**2 - beta**2)
#     scale = gamma**3 / alpha**2
#     loc = - (scale * beta) / gamma
#     return norminvgauss.ppf(q, alpha, beta, loc=loc, scale=scale)

# ######### Student t ###########
# def StudentT_obj(params, data):
#     nu, = params
#     scale = np.sqrt((nu - 2) / nu)
#     log_likelihood = t.logpdf(data, df = nu, scale = scale)
#     return -np.sum(log_likelihood)

# def optimize_StudentT(data, initial = None):
#     if initial is None:
#         initial = 10
#     bounds = [(2.01, np.inf)]
#     result = minimize(StudentT_obj, initial, args=(data,), method='L-BFGS-B', bounds=bounds)
#     return result.x

# def VaR_crit_StudentT(q, nu):
#     return t.ppf(q, df = nu, scale = np.sqrt((nu - 2) / nu))