from abc import ABC, abstractmethod
import scipy.stats as stats
from scipy.optimize import minimize
from scipy.integrate import quad
from scipy.special import gammaln, kv
import numpy as np

class Distribution(ABC):
    def __init__(self, **kwargs):
        self.parameters = kwargs
        self.parameter_names = []

    @abstractmethod
    def get_init_params(self, initial_guess = None):
        pass

    @abstractmethod
    def CDF(self, x):
        pass

    @abstractmethod
    def objective(self, params, data):
        pass

    @abstractmethod
    def fit(self, data):
        pass

    @abstractmethod
    def get_VaR_crit(self, x):
        pass

    @abstractmethod
    def get_CVaR_crit(self, x):
        pass


class Gaussian(Distribution):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key in self.parameters:
            if key not in self.parameter_names:
                raise ValueError(f"Invalid parameter '{key}' for {self.__class__.__name__}")
        return

    def get_init_params(self, initial_guess = None):
        return None

    def CDF(self, x):
        return stats.norm.cdf(x)

    def objective(self, params, data):
        ### Vectorized so x can be a vector
        return -0.5 * np.log(2 * np.pi) - 0.5 * (data ** 2)

    def fit(self, data):
        return []

    def get_VaR_crit(self, x):
        return stats.norm.ppf(x)

    def get_CVaR_crit(self, x):
        q = self.get_VaR_crit(x)
        return -stats.norm.pdf(q)/x

class StudentT(Distribution):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.parameter_names.append("nu")
        for key in self.parameters:
            if key not in self.parameter_names:
                raise ValueError(f"Invalid parameter '{key}' for {self.__class__.__name__}")

    ### Input must be a list
    def get_init_params(self, initial_guess = None):
        return [10] if initial_guess is None else init_guess

    def CDF(self, x):
        nu = self.parameters["nu"]
        return stats.t.cdf(x, df = nu, loc = 0, scale = np.sqrt((nu - 2) / nu))

    # def logPDF(self, x, params = None):
    #     nu = self.parameters['nu'] if params is None else params[0]
    #     if nu <= 2.01:
    #         return -1e10 * np.ones_like(x)
    #     log_c = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu-2))
    #     return log_c - ((nu + 1) / 2) * np.log(1 + (x**2) / (nu-2))

    def objective(self, params, data):
        nu, = params
        scale = np.sqrt((nu - 2) / nu)
        log_likelihood = stats.t.logpdf(data, df = nu, scale = scale)
        return -np.sum(log_likelihood)

    # def objective(self, params, data):
    #     nu, = params
    #     log_likelihood = self.logPDF(data, params)
    #     return -np.sum(log_likelihood)

    def fit(self, data):
        bounds = [(2.01, np.inf)]
        initial = self.get_init_params()
        result = minimize(self.objective, initial, args=(data,), method='L-BFGS-B', bounds=bounds)
        self.parameters["nu"] = result.x[0]
        return [self.parameters["nu"]]

    def get_VaR_crit(self, x):
        nu = self.parameters["nu"]
        return stats.t.ppf(x, df = nu, scale = np.sqrt((nu - 2) / nu))

    def get_CVaR_crit(self, x):
        nu = self.parameters['nu']
        scale = np.sqrt((nu - 2) / nu)
        q = self.get_VaR_crit(x)
        prod1 = (nu + q*q / (scale * scale)) / (nu - 1)
        prod2 = stats.t.pdf(q, df = nu, scale = scale) * scale / x
        return -prod1 * prod2

class NIG(Distribution):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.parameter_names.append("alpha")
        self.parameter_names.append("beta")
        for key in self.parameters:
            if key not in self.parameter_names:
                raise ValueError(f"Invalid parameter '{key}' for {self.__class__.__name__}")

    def get_init_params(self, initial_guess = None):
        ### Come back to this, find a more dynamic way
        return [5,0] if initial_guess is None else initial_guess

    def CDF(self, x):
        alpha = self.parameters["alpha"]
        beta = self.parameters["beta"]
        gamma = np.sqrt(alpha**2 - beta**2)
        scale = gamma**3 / alpha**2
        loc = - (scale * beta) / gamma
        return stats.norminvgauss.cdf(x, a = alpha, b = beta, scale = scale, loc = loc)

    def logPDF(self, x, params = None):
        if params is None:
            alpha, beta = self.parameters["alpha"], self.parameters["beta"]
        else:
            alpha, beta = params
        if abs(beta) >= alpha or alpha <= 0.1:
            return -1e10 * np.ones_like(x)
        gamma = np.sqrt(alpha**2 - beta**2)
        scale = gamma**3 / alpha**2
        loc = - (scale * beta) / gamma
        sq_term = np.sqrt(1.0 + x**2)
        log_density = (np.log(alpha) + (alpha * scale) + np.log(scale)
            - 0.5 * np.log(2.0 * np.pi)
            + (beta * scale * x)
            - 0.5 * np.log(sq_term**3)
        )
        bessel_val = kv(1, alpha * scale * sq_term)
        bessel_term = np.log(np.where(bessel_val > 0, bessel_val, 1e-300))
        return log_density + bessel_term

    # def objective(self, params, data): 
    #     alpha, beta = params
    #     if abs(beta) > alpha:
    #         return 1e10
    #     log_likelihood = self.logPDF(data, (alpha, beta))
    #     return -np.sum(log_likelihood)

    def objective(self, params, data):
        alpha, beta = params
        if abs(beta) > alpha:
            return 1e10
        gamma = np.sqrt(alpha**2 - beta**2)
        scale = gamma**3 / alpha**2
        loc = - (scale * beta) / gamma
        log_likelihood = stats.norminvgauss.logpdf(data, alpha, beta, loc = loc, scale = scale)
        return -np.sum(log_likelihood)

    def fit(self, data):
        initial = self.get_init_params()
        bounds = [(0.1, np.inf), (-np.inf, np.inf)]
        result = minimize(self.objective, initial, args=(data,), method='L-BFGS-B', bounds=bounds)
        self.parameters["alpha"] = result.x[0]
        self.parameters["beta"] = result.x[1]
        alpha = self.parameters["alpha"]
        beta = self.parameters["beta"]
        gamma = np.sqrt(alpha**2 - beta**2)
        scale = gamma**3 / alpha**2
        loc = - (scale * beta) / gamma
        return [alpha, beta, scale, loc]

    def get_VaR_crit(self, x):
        alpha = self.parameters["alpha"]
        beta = self.parameters["beta"]
        gamma = np.sqrt(alpha**2 - beta**2)
        scale = gamma**3 / alpha**2
        loc = - (scale * beta) / gamma
        return stats.norminvgauss.ppf(x, alpha, beta, loc=loc, scale=scale)

    def get_CVaR_crit(self, x):
        q = self.get_VaR_crit(x)
        alpha = self.parameters["alpha"]
        beta = self.parameters["beta"]
        gamma = np.sqrt(alpha**2 - beta**2)
        scale = gamma**3 / alpha**2
        loc = - (scale * beta) / gamma
        integrand = lambda  y : y * stats.norminvgauss.pdf(y, alpha, beta, loc=loc, scale=scale)
        lower_bound = -50.0
        val, _ = quad(integrand, lower_bound, q, epsabs=1e-4)
        return val / x


