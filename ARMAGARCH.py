import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import numpy.typing as npt
from typing import Tuple, Sequence, Optional

class ARMAGARCH:
    def __init__(self):
        self.ARMA_params = {}
        self.GARCH_params = {}

    def compute_ARMAGARCH(self, data : npt.ArrayLike, phi : float, theta : float, A : float, B : float) -> Tuple[npt.NDArray, npt.NDArray]:
        """
        Computes ARMA(1,1)-GARCH(1,1) residuals and conditional variances.
        The ARMA constant (xi) and GARCH constant (C) are calculated implicitly as a function of sample statistics and the other paramters.

        Parameters:
        phi: AR coefficient.
        theta: MA coefficient.
        A: GARCH error coefficient.
        B: GARCH variance coefficient
        """

        data = np.asarray(data)
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

    def ARMAGARCH_obj(self, params : Sequence[float], data : npt.ArrayLike) -> float:
        """
        Objective function for ARMA(1,1)-GARCH(1,1) with Gaussian errors.

        Parameters:
        params: ARMA(1,1)-GARCH(1,1) paramters in (phi, theta, A, B) order
        """

        phi, theta, A, B = params
        if A + B >= 1:
            return 1e10
        e_t, sigma2_t = self.compute_ARMAGARCH(data, phi, theta, A, B)
        return -np.sum(norm.logpdf(e_t[1:], loc=0, scale=np.sqrt(sigma2_t[1:])))

    def fit(self, data : npt.ArrayLike, initial: Optional[Sequence[float]] = None) -> None:
        """
        Fit the ARMA(1,1)-GARCH(1,1) model to the data.

        Parameters:
        initial: Initial guess for ARMA(1,1)-GARCH(1,1) paramters in (phi, theta, A, B) order
        """
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

    def get_innovations(self, data : npt.ArrayLike) -> Tuple[npt.NDArray, npt.NDArray]:
        e_t, sigma2_t = self.compute_ARMAGARCH(data, self.ARMA_params['phi'], self.ARMA_params['theta'], self.GARCH_params['A'], self.GARCH_params['B'])
        return e_t, sigma2_t

    def get_shocks(self, innovations : npt.ArrayLike, variances : npt.ArrayLike) -> npt.NDArray[np.float64]:
        """
        Computes the shocks, z_t = e_t / sigma_t
        """
        return innovations/np.sqrt(variances)

    def one_day_forecast(self, 
        data : npt.ArrayLike, 
        xi : Optional[float] = None, 
        phi : Optional[float] = None, 
        theta : Optional[float] = None, 
        C : Optional[float] = None, 
        A : Optional[float] = None, 
        B : Optional[float] = None) -> Tuple[float, float]:
        """
        Compute a one day forecast for the mean and variance of a fitted ARMA(1,1)-GARCH(1,1) model.
        """
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
