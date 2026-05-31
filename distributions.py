from abc import ABC, abstractmethod
import scipy.stats as stats
from scipy.optimize import minimize
import numpy as np

class Distribution(ABC):
    def __init__(self, **kwargs):
        self.parameters = kwargs
        self.parameter_names = []

    @abstractmethod
    def get_init_params(self, initial_guess = None):
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

    def fit(self, data):
        return

    def get_VaR_crit(self, x):
        return stats.norm.ppf(x)

    def get_CVaR_crit(self, x):
        pass

class StudentT(Distribution):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.parameter_names.append("nu")
        for key in self.parameters:
            if key not in self.parameter_names:
                raise ValueError(f"Invalid parameter '{key}' for {self.__class__.__name__}")

    def ppf(self, x):
        return stats.t.ppf(x)

    ### Input must be a list
    def get_init_params(self, initial_guess = None):
        return [10] if initial_guess is None else init_guess

    def objective(self, params, data):
        nu, = params
        scale = np.sqrt((nu - 2) / nu)
        log_likelihood = stats.t.logpdf(data, df = nu, scale = scale)
        return -np.sum(log_likelihood)

    def fit(self, data):
        bounds = [(2.01, np.inf)]
        initial = self.get_init_params()
        result = minimize(self.objective, initial, args=(data,), method='L-BFGS-B', bounds=bounds)
        self.parameters["nu"] = result.x[0]
        return 

    def get_VaR_crit(self, x):
        nu = self.parameters["nu"]
        return stats.t.ppf(x, df = nu, scale = np.sqrt((nu - 2) / nu))

    def get_CVaR_crit(self, x):
        pass

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
        return 

    def get_VaR_crit(self, x):
        alpha = self.parameters["alpha"]
        beta = self.parameters["beta"]
        gamma = np.sqrt(alpha**2 - beta**2)
        scale = gamma**3 / alpha**2
        loc = - (scale * beta) / gamma
        return stats.norminvgauss.ppf(x, alpha, beta, loc=loc, scale=scale)

    def get_CVaR_crit(self, x):
        pass


