from scipy.stats import binom

class Statistics:
    def __init__(self, prediction, actual, alpha):
        self.prediction = prediction
        self.actual = actual
        self.alpha = alpha
        self.exceedances = np.sum(self.prediction < self.actual)
        self.N = len(actual)

    def binomial(self):
        pass

    def kupiac(self):
        pass

    def christoffersen(self):
        pass

    def KS(self):
        pass
