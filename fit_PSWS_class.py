import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

#########################
# Global default definitions
#########################
DEFAULT_UNITS = {
    "A": "pH/GHz",
    "w": "GHz",
    "fres": "GHz",
    "fper": "GHz",
    "fref": "GHz",
    "D": "m",
    "re0": "pH",
    "im0": "pH"
}

DEFAULT_UNITS_2 = {
    "A1": "pH/GHz",
    "w1": "GHz",
    "fres1": "GHz",
    "fper1": "GHz",
    "fref1": "GHz",
    "D1": "m",
    "A2": "pH/GHz",
    "w2": "GHz",
    "fres2": "GHz",
    "fper2": "GHz",
    "fref2": "GHz",
    "D2": "m",
    "re0": "pH",
    "im0": "pH"
}

# Bounds stored as dicts keyed by parameter name
LOWER_BOUNDS = {"A":0, "w":0.005, "fres":0.1, "fper":0.001, "fref":0.1, "D":0.1*1e-6, "re0":-10, "im0":-10}
UPPER_BOUNDS = {"A":np.inf, "w":2, "fres":50, "fper":1, "fref":50, "D":1000*1e-6, "re0":10, "im0":10}
LOWER_BOUNDS_2 = {
            "A1":0, "w1":0.005, "fres1":0.1, "fper1":0.001, "fref1":0.1, "D1":0.1*1e-6,
            "A2":0, "w2":0.05, "fres2":0.1, "fper2":0.001, "fref2":0.1, "D2":0.1*1e-6,
            "re0":-10, "im0":-10
        }
UPPER_BOUNDS_2 = {
            "A1":np.inf, "w1":2, "fres1":50, "fper1":1, "fref1":50, "D1":1000*1e-6,
            "A2":np.inf, "w2":2, "fres2":50, "fper2":1, "fref2":50, "D2":1000*1e-6,
            "re0":10, "im0":10
        }
#########################
# Base Model Class
#########################
class BaseComplexModel:
    """
    Base class for all complex fitting models.
    Stores parameter names, default units, and allows easy extension.
    """
    def __init__(self, param_names):
        self.param_names = param_names
        # Units default for all models
        self.units = DEFAULT_UNITS.copy()

#########################
# Complex Gaussian Models
#########################
class ComplexGaussian(BaseComplexModel):
    def __init__(self):
        param_names = ["A", "w", "fres", "fper", "fref", "re0", "im0"]
        super().__init__(param_names)
        self.lower_bounds = LOWER_BOUNDS
        self.upper_bounds = UPPER_BOUNDS

    @staticmethod
    def model(f, A, w, fres, fper, fref, re0, im0):
        """
        Single complex Gaussian with sinusoidal modulation.
        Returns concatenated real + imaginary arrays for fitting.
        """
        exp_term = (A/(w*np.sqrt(np.pi/2)))*np.exp(-2*((f-fres)/w)**2)
        re = re0 + exp_term * np.cos(2*np.pi*(f-fref)/fper)
        im = im0 + exp_term * np.sin(2*np.pi*(f-fref)/fper)
        return np.concatenate([re, im])

class TwoComplexGaussian(BaseComplexModel):
    def __init__(self):
        param_names = ["A1", "w1", "fres1", "fper1", "fref1",
                       "A2", "w2", "fres2", "fper2", "fref2", "re0", "im0"]
        super().__init__(param_names)
        self.units = DEFAULT_UNITS_2
        self.lower_bounds = LOWER_BOUNDS_2
        self.upper_bounds = UPPER_BOUNDS_2

    @staticmethod
    def model(f, A1, w1, fres1, fper1, fref1,
                    A2, w2, fres2, fper2, fref2, re0, im0):
        """
        Sum of two complex Gaussians
        """
        z = (ComplexGaussian.model(f, A1, w1, fres1, fper1, fref1, 0, 0) +
             ComplexGaussian.model(f, A2, w2, fres2, fper2, fref2, re0, im0))
        return z

class ComplexGaussianVG(BaseComplexModel):
    def __init__(self):
        param_names = ["A", "w", "fres", "D", "fref", "re0", "im0"]
        super().__init__(param_names)
        self.lower_bounds = LOWER_BOUNDS
        self.upper_bounds = UPPER_BOUNDS

    @staticmethod
    def model(f, A, w, fres, D, fref, re0, im0):
        """
        Single Gaussian with vg(f)/D frequency scaling
        """
        vg_val = ComplexFitter.vg(f) / D
        return ComplexGaussian.model(f, A, w, fres, vg_val, fref, re0, im0)

class TwoComplexGaussianVG(BaseComplexModel):
    def __init__(self):
        param_names = ["A1", "w1", "fres1", "D1", "fref1",
                       "A2", "w2", "fres2", "D2", "fref2", "re0", "im0"]
        super().__init__(param_names)
        self.units = DEFAULT_UNITS_2
        self.lower_bounds = LOWER_BOUNDS_2
        self.upper_bounds = UPPER_BOUNDS_2

    @staticmethod
    def model(f, A1, w1, fres1, D1, fref1,
                    A2, w2, fres2, D2, fref2, re0, im0):
        z = (ComplexGaussianVG.model(f, A1, w1, fres1, D1, fref1, 0, 0) +
             ComplexGaussianVG.model(f, A2, w2, fres2, D2, fref2, re0, im0))
        return z

#########################
# Fitter Class
#########################
class ComplexFitter:
    """
    Handles fitting of complex models, including:
    - masking frequency range
    - fixing parameters
    - bounds management
    - plotting and summary
    """
    def __init__(self, model: BaseComplexModel):
        self.model = model
        self.fixed = {}        # fixed parameters {name: value}
        self.result = {}       # fitted parameter results
        self.cov = None        # covariance matrix

    #####################
    # Fix parameters
    #####################
    def fix(self, **kwargs):
        """
        Fix parameters before fitting. Example:
        fitter.fix(fper=0.25, re0=0.0)
        """
        for k, v in kwargs.items():
            if k not in self.model.param_names:
                raise ValueError(f"Parameter {k} not in model")
            self.fixed[k] = v

    #####################
    # Mask arrays by frequency window
    #####################
    @staticmethod
    def mask_arrays(f_min, f_max, f_array, y_re, y_im):
        mask = (f_array >= f_min) & (f_array <= f_max)
        f_fit = f_array[mask]
        y_fit_re = y_re[mask]
        y_fit_im = y_im[mask]
        y_fit = np.concatenate([y_fit_re, y_fit_im])
        return f_fit, y_fit, y_fit_re, y_fit_im

    #####################
    # Fit function
    #####################
    def fit(self, f_min, f_max, f_array, y_re, y_im, p0):
        f_fit, y_fit, y_fit_re, y_fit_im = self.mask_arrays(f_min, f_max, f_array, y_re, y_im)

        # Free parameters
        param_indices = [i for i, name in enumerate(self.model.param_names) if name not in self.fixed]

        # Wrap model
        def wrapped_model(f, *free_params):
            full_params = []
            free_idx = 0
            for name in self.model.param_names:
                if name in self.fixed:
                    full_params.append(self.fixed[name])
                else:
                    full_params.append(free_params[free_idx])
                    free_idx += 1
            return self.model.model(f, *full_params)

        # Initial guess
        p0_free = [p0[i] for i in param_indices]

        # Bounds conversion
        lower_free = [self.model.lower_bounds[name] for i, name in enumerate(self.model.param_names) if
                      name not in self.fixed]
        upper_free = [self.model.upper_bounds[name] for i, name in enumerate(self.model.param_names) if
                      name not in self.fixed]
        bounds_free = (lower_free, upper_free)

        # Fit
        params_free, cov = curve_fit(wrapped_model, f_fit, y_fit, p0=p0_free, bounds=bounds_free)

        # Build full results
        self.result = {}
        free_idx = 0
        for name in self.model.param_names:
            if name in self.fixed:
                self.result[name] = self.fixed[name]
            else:
                self.result[name] = params_free[free_idx]
                free_idx += 1

        self.cov = cov
        self.f_fit = f_fit
        self.y_fit_re = y_fit_re
        self.y_fit_im = y_fit_im

        return self.result, self.cov

    #####################
    # Summary with uncertainties, scientific formatting, units
    #####################
    def summary(self, digits=4):
        print("\nFit Results")
        print("-" * 70)
        print(f"{'Parameter':<12}{'Value':>20}{'± Error':>15}{'Unit':>10}{'Fixed':>10}")
        print("-" * 70)

        if self.cov is not None:
            errors = np.sqrt(np.diag(self.cov))
        else:
            errors = None

        free_index = 0

        for name in self.model.param_names:
            value = self.result[name]
            unit = self.model.units.get(name, "")

            # Scientific formatting for very small or large values
            if abs(value) < 1e-3 or abs(value) > 1e3:
                value_str = f"{value:.3e}"
            else:
                value_str = f"{value:.{digits}f}"

            if name in self.fixed:
                err_str = "-"
                fixed = "Yes"
            else:
                if errors is not None:
                    err_val = errors[free_index]
                    if abs(err_val) < 1e-3 or abs(err_val) > 1e3:
                        err_str = f"{err_val:.3e}"
                    else:
                        err_str = f"{err_val:.{digits}f}"
                else:
                    err_str = "-"
                free_index += 1
                fixed = "No"

            print(f"{name:<12}{value_str:>20}{err_str:>15}{unit:>10}{fixed:>10}")

        print("-" * 70)

    #####################
    # Plot the fit
    #####################
    @staticmethod
    def unconcatenate(z):
        N = len(z)//2
        return z[:N], z[N:]

    def plot_fit(self, f_model=None):
        """
        Plot real and imaginary parts with data and fit.
        """
        if f_model is None:
            f_model = self.model.model

        plt.scatter(self.f_fit, self.y_fit_re, s=50, facecolors='none', edgecolors='blue', label="Re")
        plt.scatter(self.f_fit, self.y_fit_im, s=50, facecolors='none', edgecolors='red', label="Im")

        fit_vals = f_model(self.f_fit, *[self.result[name] for name in self.model.param_names])
        fit_re, fit_im = self.unconcatenate(fit_vals)

        plt.plot(self.f_fit, fit_re, 'cyan', label="Re_fit")
        plt.plot(self.f_fit, fit_im, 'yellow', label="Im_fit")

        plt.axhline(0, color='black', linewidth=1)
        plt.gca().set_facecolor('#edf1f7')
        plt.xlabel("Frequency (GHz)")
        plt.ylabel("dLij (pH)")
        plt.title("Fitting Results")
        plt.legend()
        plt.show()

    #####################
    # Example vg function for DE spin waves
    #####################
    @staticmethod
    def vg(f):
        mu_o = 4*np.pi*1e-7
        t = 105*1e-9  # m
        k = 2.95*1e6  # rad/m
        M = 139.6*1e3*mu_o  # T
        H_eff = M
        gamma = 182.21/(2*np.pi)  # GHz/T

        return (2*np.pi) * gamma*gamma*M*H_eff*t*np.exp(-2*k*t)/(4*f)