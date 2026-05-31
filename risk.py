from ARMAGARCH import ARMAGARCH

class Risk:
    def __init__(self, dist = None, ARMAGARCH_model = None):
        supported_distributions = {
            "Gaussian": Gaussian,
            "StudentT": StudentT,
            "NIG": NIG
        }

        if dist is not None and dist not in supported_distributions:
            raise ValueError(f"Distribution '{dist}' not supported. "
                             f"Currently supported distributions: {list(supported_distributions.keys())}")

        if dist:
            self.distribution = Distribution.supported_distributions[dist]()
        else:
            self.distribution = Distribution.Gaussian()

        self.ARMAGARCH = ARMAGARCH_model if ARMAGARCH_model is not None else ARMAGARCH()

    def fit_distribution(self, data):
        resids, variances = self.ARMAGARCH.get_innovations(data)
        shocks = self.ARMAGARCH.get_shocks(resids, variances)
        self.distribution.fit(shocks)
        return None
    
    ### Alpha here is the tail probability
    def get_VaR(self, alpha):
        forecast_mean, forecast_variance = self.ARMAGARCH.one_day_forecast()
        q = self.distribution.get_VaR_crit(alpha)
        return forecast_mean + q * np.sqrt(forecast_variance)


    def get_CVaR(self):
        forecast_mean, forecast_variance = ARMAGARCH.one_day_forecast()


    
    