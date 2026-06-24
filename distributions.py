from abc import ABC, abstractmethod
import scipy.stats as stats
from scipy.optimize import minimize
from scipy.integrate import quad
from scipy.special import gammaln, kv
import numpy as np
import numpy.typing as npt
from typing import Tuple, Sequence, Optional

class Distribution(ABC):
    """
    Abstract base class for distributions. 
    Any distribution class used to model ARMA-GARCH shocks must inherit from this class and implement all abstract methods to ensure compatibility with the Risk class.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.parameters = kwargs
        self.parameter_names: list[str] = []

    @abstractmethod
    def get_init_params(self, initial_guess: Optional[Sequence[float]] = None) -> Sequence[float]:
        """
        Get the initial guess for the parameters of the distribution.
        Returns a sequence containing all the parameters.
        """
        pass

    @abstractmethod
    def CDF(self, x: float | npt.ArrayLike) -> float | npt.NDArray[np.float64]:
        """
        CDF of the distribution evaluated at x.
        Input is either a float or a vector if the function allows for vectorization.
        Returns a float or a numpy array, if vectorized.
        """
        pass

    @abstractmethod
    def objective(self, params: Sequence[float], data: npt.ArrayLike) -> float:
        """
        Objective function used for optimization. Should contain the log PDF so that the negative log-likelihood can be returned
        params should always be a sequence (e.g. a list or a tuple), even if distribution has only one or no parameters.
        """
        pass

    @abstractmethod
    def fit(self, data: npt.ArrayLike) -> Sequence[float]:
        """
        Fits distribution to data. Optimization routine should be included here.
        Returns fitted parameters as a sequence.
        """
        pass

    @abstractmethod
    def get_VaR_crit(self, x: float) -> float:
        """
        Get critical value for VaR, i.e. the value corresponding to the xth tail probability. 
        For example, if one would like 99% VaR, x = 0.01.
        """
        pass

    @abstractmethod
    def get_CVaR_crit(self, x: float) -> float:
        """
        See comments for get_VaR_crit method.
        """
        pass


class Gaussian(Distribution):
    """
    Standard Gaussian distribution class
    """
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        for key in self.parameters:
            if key not in self.parameter_names:
                raise ValueError(f"Invalid parameter '{key}' for {self.__class__.__name__}")
        return

    def get_init_params(self, initial_guess: Optional[Sequence[float]] = None) -> Sequence[float]:
        return []

    def CDF(self, x: float | npt.ArrayLike) -> float | npt.NDArray[np.float64]:
        val = stats.norm.cdf(x)
        if np.isscalar(x):
            return float(val)
        return val

    def objective(self, params: Sequence[float], data: npt.ArrayLike) -> float:
        return float(-np.sum(stats.norm.logpdf(data)))

    def fit(self, data: npt.ArrayLike) -> Sequence[float]:
        return []

    def get_VaR_crit(self, x: float) -> float:
        return float(stats.norm.ppf(x))

    def get_CVaR_crit(self, x: float) -> float:
        q = self.get_VaR_crit(x)
        return float(-stats.norm.pdf(q)/x)

class StudentT(Distribution):
    """
    Standard (location-scale) Student's t-distribution class. 

    Parameters:
    nu: scale parameter (bounded below by 2 to ensure finite variance)

    We take nu = 10 as an initial guess in the optimization.
    In the code we often multiply the scale by (nu-2)/nu to ensure unit variance.

    """
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.parameter_names.append("nu")
        for key in self.parameters:
            if key not in self.parameter_names:
                raise ValueError(f"Invalid parameter '{key}' for {self.__class__.__name__}")

    def get_init_params(self, initial_guess: Optional[Sequence[float]] = None) -> Sequence[float]:
        return [10] if initial_guess is None else initial_guess

    def CDF(self, x: float | npt.ArrayLike) -> float | npt.NDArray[np.float64]:
        nu = self.parameters["nu"]
        val = stats.t.cdf(x, df = nu, loc = 0, scale = np.sqrt((nu - 2) / nu))
        if np.isscalar(x):
            return float(val)
        return val

    def objective(self, params: Sequence[float], data: npt.ArrayLike) -> float:
        nu, = params
        scale = np.sqrt((nu - 2) / nu)
        log_likelihood = stats.t.logpdf(data, df = nu, scale = scale)
        return float(-np.sum(log_likelihood))

    def fit(self, data: npt.ArrayLike) -> Sequence[float]:
        ### Set a lower bound of 2 to ensure finite variance
        bounds = [(2.01, np.inf)]
        initial = self.get_init_params()
        result = minimize(self.objective, initial, args=(data,), method='L-BFGS-B', bounds=bounds)
        self.parameters["nu"] = result.x[0]
        return [self.parameters["nu"]]

    def get_VaR_crit(self, x: float) -> float:
        nu = self.parameters["nu"]
        return float(stats.t.ppf(x, df = nu, scale = np.sqrt((nu - 2) / nu)))

    def get_CVaR_crit(self, x: float) -> float:
        q = self.get_VaR_crit(x)
        nu = self.parameters["nu"]
        scale = np.sqrt((nu - 2) / nu)
        integrand = lambda z: z * stats.t.pdf(z, df=nu, scale=scale)
        val, _ = quad(integrand, -np.inf, q, epsabs=1e-4)
        return float(val / x)

class NIG(Distribution):
    """
    Standard Normal Inverse Gaussian (NIG) distribution class.

    Parameters:
    alpha: Tail heaviness
    beta: Asymmetry parameter
    mu: location
    delta: scale parameter

    Parameters are stored as (alpha, beta) and during fitting are initially guessed to be (5,0).

    We compute mu and delta as a function of alpha and beta due to our zero-mean unit-variance requirement.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.parameter_names.append("alpha")
        self.parameter_names.append("beta")
        for key in self.parameters:
            if key not in self.parameter_names:
                raise ValueError(f"Invalid parameter '{key}' for {self.__class__.__name__}")

    def get_init_params(self, initial_guess: Optional[Sequence[float]] = None) -> Sequence[float]:
        return [5,0] if initial_guess is None else initial_guess

    def CDF(self, x: float | npt.ArrayLike) -> float | npt.NDArray[np.float64]:
        alpha = self.parameters["alpha"]
        beta = self.parameters["beta"]
        gamma = np.sqrt(alpha**2 - beta**2)
        scale = gamma**3 / alpha**2
        loc = - (scale * beta) / gamma
        val = stats.norminvgauss.cdf(x, a = alpha, b = beta, scale = scale, loc = loc)
        if np.isscalar(x):
            return float(val)
        return val

    def objective(self, params: Sequence[float], data: npt.ArrayLike) -> float:
        alpha, beta = params
        if abs(beta) > alpha:
            return 1e10
        gamma = np.sqrt(alpha**2 - beta**2)
        scale = gamma**3 / alpha**2
        loc = - (scale * beta) / gamma
        log_likelihood = stats.norminvgauss.logpdf(data, a = alpha, b = beta, loc = loc, scale = scale)
        return float(-np.sum(log_likelihood))

    def fit(self, data: npt.ArrayLike) -> Sequence[float]:
        initial = self.get_init_params()
        bounds = [(0.1, np.inf), (-np.inf, np.inf)]
        result = minimize(self.objective, initial, args=(data,), method='L-BFGS-B', bounds=bounds)
        self.parameters["alpha"] = result.x[0]
        self.parameters["beta"] = result.x[1]
        alpha = self.parameters["alpha"]
        beta = self.parameters["beta"]
        gamma = np.sqrt(alpha**2 - beta**2)
        return [alpha, beta]

    def get_VaR_crit(self, x: float) -> float:
        alpha = self.parameters["alpha"]
        beta = self.parameters["beta"]
        gamma = np.sqrt(alpha**2 - beta**2)
        scale = gamma**3 / alpha**2
        loc = - (scale * beta) / gamma
        return float(stats.norminvgauss.ppf(x, a = alpha, b = beta, loc=loc, scale=scale))

    def get_CVaR_crit(self, x: float) -> float:
        q = self.get_VaR_crit(x)
        alpha = self.parameters["alpha"]
        beta = self.parameters["beta"]
        gamma = np.sqrt(alpha**2 - beta**2)
        scale = gamma**3 / alpha**2
        loc = - (scale * beta) / gamma
        integrand = lambda  z : z * stats.norminvgauss.pdf(z, a = alpha, b = beta, loc=loc, scale=scale)
        val, _ = quad(integrand, -np.inf, q, epsabs=1e-4)
        return float(val / x)


