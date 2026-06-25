import numpy as np
from .armagarch import ARMAGARCH
from .distributions import Distribution
import numpy.typing as npt
from typing import Tuple, Sequence, Optional

class Risk:
    """
    Risk class, fits ARMA GARCH shocks to chosen distribution.
    Here alpha is the tail probability, e.g. 99% VaR corresponds to alpha = 0.01
    """

    def __init__(self, dist: Optional[str] = None, ARMAGARCH_model: Optional[ARMAGARCH] = None) -> None:
        supported_distributions = {cls.__name__: cls for cls in Distribution.__subclasses__()}

        if dist is not None and dist not in supported_distributions:
            raise ValueError(f"Distribution '{dist}' not supported. "
                             f"Currently supported distributions: {list(supported_distributions.keys())}")

        # If no distribution is selected the model will default to Gaussian shocks
        if dist:
            self.distribution = supported_distributions[dist]()
        else:
            self.distribution = Distribution.Gaussian()

        self.ARMAGARCH = ARMAGARCH_model if ARMAGARCH_model is not None else ARMAGARCH()

    def fit_distribution(self, data: npt.ArrayLike) -> Sequence[float]:
        """
        Fits chosen distribution to ARMA GARCH shocks (i.e. standardized residuals)
        Returns the fitted parameters.
        """
        data = np.asarray(data)
        resids, variances = self.ARMAGARCH.get_innovations(data)
        shocks = self.ARMAGARCH.get_shocks(resids, variances)
        params = self.distribution.fit(shocks)
        return params
    
    def get_VaR(self, data: npt.ArrayLike, alpha: float) -> float:
        data = np.asarray(data)
        forecast_mean, forecast_variance = self.ARMAGARCH.one_day_forecast(data)
        q = self.distribution.get_VaR_crit(alpha)
        return float(forecast_mean + q * np.sqrt(forecast_variance))


    def get_CVaR(self, data: npt.ArrayLike, alpha: float) -> float:
        data = np.asarray(data)
        forecast_mean, forecast_variance = self.ARMAGARCH.one_day_forecast(data)
        q = self.distribution.get_CVaR_crit(alpha)
        return float(forecast_mean + q * np.sqrt(forecast_variance))

    
    