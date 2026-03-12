from abc import ABC, abstractmethod
import scipy.stats

class Distribution(ABC):
    def __init__(self, **kwargs):
        self.parameters = kwargs
        self.paramter_names = []
    @abstractmethod
    def pdf():
        pass

    @abstractmethod
    def logpdf():
        pass

    @abstractmethod
    def ppf():
        pass

    @abstractmethod
    def get_init_params():
        pass

    @abstractmethod
    def objective():
        pass

    @abstractmethod
    def fit():
        pass

    @abstractmethod
    def get_VaR_crit():
        pass


class Gaussian(Distribution):
    def __init__(self):
        super().__init__()
        for key in self.parameters:
            if key not in self.parameter_names:
                raise ValueError(f"Invalid parameter '{key}' for {self.__class__.__name__}")
        return
        
    def pdf(self, x):
        return stats.norm.pdf(x)

    def logpdf(self, x):
        return stats.norm.logpdf(x)

    def ppf(self, x):
        return stats.norm.ppf(x)

    def get_init_params(self):
        return None

    def objective():
        pass

    def fit():
        pass

    def get_VaR_crit():
        pass

class StudentT(Distribution):
    def __init__(self):
        super().__init__()
        self.parameter_names.append("nu")
        for key in self.parameters:
            if key not in self.parameter_names:
                raise ValueError(f"Invalid parameter '{key}' for {self.__class__.__name__}")

    def pdf(self, x):
        return stats.t.pdf(x)

    def logpdf(self, x):
        return stats.t.logpdf(x)

    def ppf(self, x):
        return stats.t.ppf(x)

    def get_init_params(self, data, init_guess = None):
        if init_guess:
            if init_guess <= 2:
                raise ValueError(f"Invalid value  for nu")
            return init_guess
        else:
            ### Compute empirical kurtosis
            fourth_moment = np.mean((data) - np.mean(data))**4
            empir_var = np.vara(data)
        return [fourth_moment / empir_var]

    def objective():
        pass

    def fit():
        pass

    def get_VaR_crit():
        pass

class NIG(Distribution):
    def __init__(self):
        super().__init__()
        self.parameter_names.append("alpha", "beta")
        for key in self.parameters:
            if key not in self.parameter_names:
                raise ValueError(f"Invalid parameter '{key}' for {self.__class__.__name__}")

    def pdf(self, x):
        return stats.norminvgauss.pdf(x)

    def logpdf(self, x):
        return stats.norminvgauss.logpdf(x)

    def ppf(self, x):
        return stats.norminvgauss.ppf(x)

    def get_init_params(self):
        ### Come back to this, find a more dynamic way
        return [5,0]

    def objective():
        pass

    def fit():
        pass

    def get_VaR_crit():
        pass


