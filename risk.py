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

    def get_ARMAGARCH_params(self):
        return {"ARMA" : self.ARMAGARCH.ARMA_params, "GARCH" : self.ARMAGARCH.GARCH_params}

    def fit_distribution(self, data):
        resids, variances = self.ARMAGARCH.get_innovations(data)
        shocks = self.ARMAGARCH.get_shocks(resids, variances)
        self.distribution.fit(shocks)
        return None
    
    def get_VaR(self):
        pass

    def get_CVaR(self):
        pass

    
    