import numpy as np
from risk import Risk
from stats import Statistics
from plotter import Plotter

class Backtest:

    def __init__(self, data, alpha, distribution):
        self.data = data
        self.alpha = alpha
        self.distribution = distribution
        self.forecasts = []
        self.observed = []
        self.distribution_list = []

    def rolling_window(self, window_size):
        out_of_sample_size = len(self.data) - window_size
        if out_of_sample_size < 0:
            raise ValueError("Window size too large relative to data length.")
        forecasts = np.zeros(out_of_sample_size)
        observed = np.zeros(out_of_sample_size)
        distribution_list = []
        for i in range(out_of_sample_size):
            print(f"{i}/{out_of_sample_size}")
            training = self.data[i : i + window_size]
            model = Risk(self.distribution)
            if i > 0:
                model.ARMAGARCH.fit(training, warm_ARMA_GARCH_params)
            else:
                model.ARMAGARCH.fit(training)

            model.fit_distribution(training)

            forecasts[i] = model.get_VaR(training, self.alpha)
            observed[i] = self.data[i+window_size]

            distribution_list.append(model.distribution)

            warm_ARMA_GARCH_params = (model.ARMAGARCH.ARMA_params['phi'],
            model.ARMAGARCH.ARMA_params['theta'],
            model.ARMAGARCH.GARCH_params['A'],
            model.ARMAGARCH.GARCH_params['B'],
            )

        self.forecasts = forecasts
        self.observed = observed
        self.distribution_list = distribution_list
        return forecasts, observed

    def statistical_tests(self, p_threshold = 0.05):
        stats_obj = Statistics(self.forecasts, self.observed, self.alpha)
        binomial_stat, binomial_p = stats_obj.binomial()
        kupiec_stat, kupiec_p = stats_obj.kupiec()
        christoffersen_stat, christoffersen_p = stats_obj.christoffersen()
        cc_stat, cc_p = stats_obj.conditional_coverage()
        KS_stat, KS_p = stats_obj.KS(self.distribution_list)


        results = {"Binomial": binomial_p > p_threshold,
                    "Kupiec": kupiec_p > p_threshold,
                    "Christoffersen": christoffersen_p > p_threshold,
                    "Conditional Coverage": cc_p > p_threshold,
                    "KS Test": KS_p > p_threshold
                }
                
        return results
        
    
