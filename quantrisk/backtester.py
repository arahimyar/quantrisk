import numpy as np
from .risk import Risk
from .stats import Statistics
from .plotter import Plotter
from typing import Tuple, Sequence

class Backtest:
    """
    Backtesting class to asssess model performance
    """

    def __init__(self, data: Sequence[float], alpha: float, distribution: str) -> None:
        """
        Parameters:
        data: data
        alpha: tail probability, e.g. 99% VaR -> alpha = 0.01
        distribution: distribution name
        """
        self.data = np.asarray(data)
        self.alpha = alpha
        self.distribution = distribution
        self.forecasts = []
        self.observed = []
        self.CDF_evals = []

    def rolling_window(self, window_size: int) -> Tuple[Sequence[float], Sequence[float]]:
        """
        Computes rolling window back test

        Parameters:
        window_size: size of rolling window

        Returns: 
        tuple of forecasted and observed values
        """
        out_of_sample_size = len(self.data) - window_size
        if out_of_sample_size < 0:
            raise ValueError("Window size too large relative to data length.")
        forecasts = np.zeros(out_of_sample_size)
        observed = np.zeros(out_of_sample_size)
        CDF_evals = []
        warm_ARMA_GARCH_params = None
        for i in range(out_of_sample_size):
            training = self.data[i : i + window_size]
            model = Risk(self.distribution)
            if warm_ARMA_GARCH_params is not None:
                model.ARMAGARCH.fit(training, warm_ARMA_GARCH_params)
            else:
                model.ARMAGARCH.fit(training)

            model.fit_distribution(training)

            forecasts[i] = model.get_VaR(training, self.alpha)
            observed[i] = self.data[i+window_size]

            warm_ARMA_GARCH_params = (model.ARMAGARCH.ARMA_params['phi'],
            model.ARMAGARCH.ARMA_params['theta'],
            model.ARMAGARCH.GARCH_params['A'],
            model.ARMAGARCH.GARCH_params['B'],
            )


            forecast_mean, forecast_var = model.ARMAGARCH.one_day_forecast(training)
            shock = (observed[i] - forecast_mean)/np.sqrt(forecast_var)
            CDF_evals.append(model.distribution.CDF(shock))

        self.forecasts = forecasts
        self.observed = observed
        self.CDF_evals = CDF_evals
        return forecasts, observed

    def statistical_tests(self, p_threshold = 0.05) -> dict:
        """
        Performs all statistical tests

        Parameters:
        p_threshold: Significane threshold for statistical tests

        Returns:
        Dictonary whose key is a statistical test and whose value is a tuple consisting of (p-value, pass/fail)
        """
        stats_obj = Statistics(self.forecasts, self.observed, self.alpha)
        binomial_stat, binomial_p = stats_obj.binomial()
        kupiec_stat, kupiec_p = stats_obj.kupiec()
        christoffersen_stat, christoffersen_p = stats_obj.christoffersen()
        cc_stat, cc_p = stats_obj.conditional_coverage()
        KS_stat, KS_p = stats_obj.KS(self.CDF_evals)


        results = {"Binomial": (binomial_p, binomial_p > p_threshold),
                    "Kupiec": (kupiec_p, kupiec_p > p_threshold),
                    "Christoffersen": (christoffersen_p, christoffersen_p > p_threshold),
                    "Conditional Coverage": (cc_p, cc_p > p_threshold),
                    "KS": (KS_p, KS_p > p_threshold)
                }
                
        return results
        
    
