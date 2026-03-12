from distributions import Distribution

class ARMAGARCH:
    def __init__(self, distribution = None):
        supported_distributions = {
            "Gaussian": Gaussian,
            "StudentT": StudentT,
            "NIG": NIG
        }

        if distribution is not None and distribution not in supported_distributions:
            raise ValueError(f"Distribution '{distribution}' not supported. "
                             f"Currently supported distributions: {list(supported_distributions.keys())}")

        if distribution:
            self.distribution = Distribution.supported_distributions[distribution]()
        else:
            self.distribution = Distribution.Gaussian()
        self.ARMA_params = {}
        self.GARCH_params = {}

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

    def fit_ARMAGARCH(data, initial = None):
        bounds = [(-0.999, 0.999), (-0.999, 0.999), (0.001, 0.999), (0.001,  0.999)]
        if initial is None:
            initial = np.asarray([0.1, 0.1, 0.05, 0.8])
        result = minimize(ARMAGARCH_obj, initial, args=(data,), method='L-BFGS-B', bounds=bounds)
        return result.x


### X_t = xi + phi * X_{t-1} + \theta \epsilon_{t-1} + \epsilon_t
### sigma_t^2 = C + A * epsilon_{t-1}^2 + B * \sigma_{t-1}^2 +
### Assume xi = 0 since we're modelling log returns


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

######### NIG with mean zero and unit variance ##########

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

######### Student t ###########
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