import scipy.stats as stats
import numpy as np
import numpy.typing as npt
from typing import Tuple, Sequence, Optional

class Statistics:
    """
    Statistics class containing all the statistical tests used to assess fit and VaR performance.

    Here alpha is the tail probability, e.g. 99% VaR implies alpha = 0.01.
    """
    def __init__(self, prediction: Sequence[float], actual: Sequence[float], alpha: float) -> None:
        """
        Parameters:
        prediction: forecasted values
        actual: observed values
        alpha: tail probability
        """
        self.prediction = np.asarray(prediction)
        self.actual = np.asarray(actual)
        self.alpha = alpha
        self.mask = self.prediction > self.actual
        self.exceedances = np.sum(self.mask)
        self.N = len(actual)

    def binomial(self, alpha: Optional[float] = None) -> tuple[float, float]:
        """
        Binomial test

        Returns: (test-statistic, p-value) tuple
        """
        alpha = self.alpha if alpha is None else alpha
        n = self.N
        e = self.exceedances
        lower = stats.binom.cdf(e, n, alpha)
        upper = stats.binom.sf(e - 1, n, alpha) if e > 0 else 1.0
        return float(e), float(min(1, 2 * min(lower, upper)))

    def kupiec(self, alpha: Optional[float] = None) -> tuple[float, float]:
        """
        Kupiec test

        Returns: (test-statistic, p-value) tuple
        """
        alpha = self.alpha if alpha is None else alpha
        n = self.N
        e = self.exceedances
        if e == 0:
            LR = -2 * (n * np.log(1.0 - alpha))
            return LR, 1 - stats.chi2.cdf(LR, df=1)
        if e == n:
            LR = -2 * (n * np.log(alpha))
            return LR, 1 - stats.chi2.cdf(LR, df=1)

        log_null = (n - e) * np.log(1 - alpha) + e * np.log(alpha)
        log_alt = (n - e) * np.log(1 - (e / n)) + e * np.log(e / n)
        LR = -2 * (log_null - log_alt)
        return float(LR), float(1 - stats.chi2.cdf(LR, df = 1))

    def christoffersen(self) -> tuple[float, float]:
        """
        Christoffersen test

        Returns: (test-statistic, p-value) tuple
        """
        if self.exceedances < 2:
            return 0.0, 1.0
        no_no = 0
        yes_no = 0
        no_yes = 0
        yes_yes = 0
        for i in range(1, self.N):
            prev = self.mask[i-1]
            curr = self.mask[i]
            if prev == 0 and curr == 0:
                no_no += 1
            elif prev == 0 and curr == 1:
                no_yes += 1
            elif prev == 1 and curr == 0:
                yes_no += 1
            elif prev == 1 and curr == 1:
                yes_yes += 1

        p0 = no_yes / (no_no + no_yes) if (no_no + no_yes) > 0 else 0
        p1 = yes_yes / (yes_no + yes_yes) if (yes_no + yes_yes) > 0 else 0
        p = (no_yes + yes_yes) / (no_no + no_yes + yes_no + yes_yes)

        num = 0
        if p > 0:   num += (no_yes + yes_yes) * np.log(p)
        if p < 1:   num += (no_no + yes_no) * np.log(1 - p)
            
        denom = 0
        if p0 > 0: denom += no_yes * np.log(p0)
        if p0 < 1: denom += no_no * np.log(1 - p0)
        if p1 > 0: denom += yes_yes * np.log(p1)
        if p1 < 1: denom += yes_no * np.log(1 - p1)

        LR = max(0.0, -2 * (num - denom))
        return float(LR), float(1 - stats.chi2.cdf(LR, df = 1))

    def conditional_coverage(self, alpha: Optional[float] = None) -> tuple[float, float]:
        """
        Conditional coverage test (i.e. combining Kupiec and Christoffersen)

        Returns: (test-stat, p-value) tuple
        """
        alpha = self.alpha if alpha is None else alpha
        LR_kupiec, _ = self.kupiec(alpha)
        LR_christoffersen, _ = self.christoffersen()
        LR = LR_kupiec + LR_christoffersen
        return float(LR), float(1 - stats.chi2.cdf(LR, df = 2))

    def QQ(self, CDF_evals: Sequence[float]) -> list[tuple[float, float]]:
        """
        Returns empirical vs theoretical probabilities pairs

        Parameters:
        CDF_evals: CDF evaluations
        """
        model = sorted(CDF_evals)
        n = len(model)
        empirical = [float(i+1) / float(n+1) for i in range(n)]
        return list(zip(empirical, model))
    
    def KS(self, CDF_evals: Sequence[float]) -> tuple[float, float]:
        """
        Kolmogorov Smirnov test against uniform distribution

        Parameters:
        CDF_evals: CDF evaluations

        Returns: (test-statistic, p-value) tuple
        """
        stat, pval = stats.kstest(CDF_evals, 'uniform')
        return float(stat), float(pval)
