import warnings
from contextlib import contextmanager
from statsmodels.tools.sm_exceptions import ConvergenceWarning, HessianInversionWarning

warnings.filterwarnings("ignore", message=".*is a deprecated alias.*")
warnings.filterwarnings("ignore", message=".*lbfgs failed to converge.*")
warnings.filterwarnings("ignore", message=".*non-invertible starting MA.*")

@contextmanager
def hush_statsmodels():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning, module=r"statsmodels\..*")
        warnings.filterwarnings("ignore", category=HessianInversionWarning, module=r"statsmodels\..*")
        warnings.filterwarnings(
            "ignore", category=UserWarning,
            module=r"statsmodels\.tsa\.statespace\.sarimax",
            message=r".*Non-stationary starting autoregressive parameters found.*",
        )
        yield
