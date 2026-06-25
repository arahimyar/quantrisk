from .backtester import Backtest
from .plotter import Plotter
from .risk import Risk
from .stats import Statistics
from .distributions import Distribution

__all__ = ["Backtest", "Plotter", "Risk", "Statistics"]

### Dynamically add all supported distributions
for sub_class in Distribution.__subclasses__():
    class_name = sub_class.__name__
    globals()[class_name] = sub_class
    __all__.append(class_name)