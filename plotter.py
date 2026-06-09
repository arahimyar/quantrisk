import matplotlib.pyplot as plt
import numpy as np
from stats import Statistics

class Plotter:

    def QQ(self, CDF_evals, prediction, actual, alpha):
        stat_obj = Statistics(prediction, actual, alpha)
        emperical, model = zip(*stat_obj.QQ(CDF_evals))
        plt.figure(figsize=(6, 6))
        plt.scatter(emperical, model, color = "red")
        plt.title("QQ")
        plt.xlabel("Emperical Quantiles")
        plt.ylabel("Modoel Quantiles")
        plt.plot([0, 1], [0, 1], color='black', linestyle='--', linewidth=3, label='Theoretical Uniform')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.legend(loc='lower right')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.tight_layout()
        plt.show()

    
    def VaR(self, prediction, actual, alpha):
        prediction = np.asarray(prediction)
        actual = np.asarray(actual)
        time = np.arange(len(actual))
        stat_obj = Statistics(prediction, actual, alpha)
        violations = stat_obj.mask
        plt.figure(figsize=(12, 5))
        plt.plot(time, actual, color='darkgray', alpha=0.7, label='Realized Returns')
        plt.plot(time, prediction, color='red', linewidth=2, label=f'{int((1-alpha)*100)}% VaR')

        if np.any(violations):
            plt.scatter(time[violations], actual[violations], 
                    color='black', marker='x', s=50, zorder=5, 
                    label=f'VaR Exceedances ({stat_obj.exceedances})'
                )
        plt.title(f"{(1.0-alpha)*100:g}% VaR Backtest", fontsize=12, fontweight='bold')
        plt.xlabel("Time")
        plt.ylabel("Returns")
        plt.legend(loc='upper right', frameon=True)
        plt.tight_layout()
        plt.show()