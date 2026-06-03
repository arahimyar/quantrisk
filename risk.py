import numpy as np
from ARMAGARCH import ARMAGARCH
import distributions as Distribution

class Risk:
    def __init__(self, dist = None, ARMAGARCH_model = None):
        supported_distributions = {
            "Gaussian": Distribution.Gaussian,
            "StudentT": Distribution.StudentT,
            "NIG": Distribution.NIG
        }

        if dist is not None and dist not in supported_distributions:
            raise ValueError(f"Distribution '{dist}' not supported. "
                             f"Currently supported distributions: {list(supported_distributions.keys())}")

        if dist:
            self.distribution = supported_distributions[dist]()
        else:
            self.distribution = Distribution.Gaussian()

        self.ARMAGARCH = ARMAGARCH_model if ARMAGARCH_model is not None else ARMAGARCH()

    def fit_distribution(self, data):
        resids, variances = self.ARMAGARCH.get_innovations(data)
        shocks = self.ARMAGARCH.get_shocks(resids, variances)
        params = self.distribution.fit(shocks)
        return params
    
    ### Alpha here is the tail probability
    def get_VaR(self, data, alpha):
        forecast_mean, forecast_variance = self.ARMAGARCH.one_day_forecast(data)
        q = self.distribution.get_VaR_crit(alpha)
        return forecast_mean + q * np.sqrt(forecast_variance)


    def get_CVaR(self, data):
        forecast_mean, forecast_variance = ARMAGARCH.one_day_forecast(data)
        ### FINISH THIS #####

    
    