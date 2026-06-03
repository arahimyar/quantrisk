from backtester import Backtest
from plotter import Plotter
import numpy as np

data = np.load('/Users/hasibrahimyar/Desktop/ARMAGARCH2/DATA/2019_2025AAPL.npy')
data = data[-1000:]
alpha = 0.01
test = Backtest(data, alpha, "NIG")
test.rolling_window(250)
plotter_obj = Plotter()
plotter_obj.QQ(test.distribution_list, test.forecasts, test.observed, alpha)
plotter_obj.VaR(test.forecasts, test.observed, alpha)

print(test.statistical_tests())
