from backtester import Backtest
import numpy as np

data = np.load('/Users/hasibrahimyar/Desktop/ARMAGARCH2/DATA/2019_2025AAPL.npy')
data = data[-1000:]
alpha = 0.01
test = Backtest(data, 0.05, "NIG")
test.rolling_window(250)
print(test.statistical_tests())
Plotter.QQ(test.distribution_list, test.forecasts, test.observed, alpha)
Plotter.VaR(test.forecasts, teset.observed, alpha)