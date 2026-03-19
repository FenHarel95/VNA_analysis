import numpy as np
from scipy.optimize import curve_fit, minimize
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact

#########################
# Global default definitions
#########################
DEFAULT_UNITS = {
    "A": "pH/GHz",
    "w": "GHz",
    "fres": "GHz",
    "fper": "GHz",
    "fref": "GHz",
    "phi" : "rad",
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
    "phi1" : "rad",
    "A2": "pH/GHz",
    "w2": "GHz",
    "fres2": "GHz",
    "fper2": "GHz",
    "fref2": "GHz",
    "D2": "m",
    "phi2" : "rad",
    "re0": "pH",
    "im0": "pH"
}

# Bounds stored as dicts keyed by parameter name
LOWER_BOUNDS = {"A":1e-1, "w":0.005, "fres":0.1, "fper":0.001, "fref":0.1, "phi":0, "D":0.1*1e-6, "re0":-10, "im0":-10}
UPPER_BOUNDS = {"A":np.inf, "w":2, "fres":50, "fper":1, "fref":50, "phi":2*np.pi, "D":1000*1e-6, "re0":10, "im0":10}
LOWER_BOUNDS_2 = {
            "A1":1e-1, "w1":0.005, "fres1":0.1, "fper1":0.001, "fref1":0.1, "D1":0.1*1e-6,
            "A2":1e-1, "w2":0.05, "fres2":0.1, "fper2":0.001, "fref2":0.1, "phi1":0, "phi2":0, "D2":0.1*1e-6,
            "re0":-10, "im0":-10
        }
UPPER_BOUNDS_2 = {
            "A1":np.inf, "w1":2, "fres1":50, "fper1":1, "fref1":50, "D1":1000*1e-6,
            "A2":np.inf, "w2":2, "fres2":50, "fper2":1, "fref2":50, "phi1":2*np.pi, "phi2":2*np.pi, "D2":1000*1e-6,
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

class ComplexGaussianSimple(BaseComplexModel):
    def __init__(self):
        param_names = ["A", "w", "fres", "fper", "phi", "re0", "im0"]
        super().__init__(param_names)
        self.lower_bounds = LOWER_BOUNDS
        self.upper_bounds = UPPER_BOUNDS

    @staticmethod
    def model(f, A, w, fres, fper, phi, re0, im0):
        exp_term = A * np.exp(-2 * ((f - fres) / w) ** 2)

        phase = 2 * np.pi * f / fper - phi

        re = re0 + exp_term * np.cos(phase)
        im = im0 + exp_term * np.sin(phase)

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

class TwoComplexGaussianSimple(BaseComplexModel):
    def __init__(self):
        param_names = ["A1", "w1", "fres1", "fper1", "phi1",
                       "A2", "w2", "fres2", "fper2", "phi2", "re0", "im0"]
        super().__init__(param_names)
        self.units = DEFAULT_UNITS_2
        self.lower_bounds = LOWER_BOUNDS_2
        self.upper_bounds = UPPER_BOUNDS_2

    @staticmethod
    def model(f, A1, w1, fres1, fper1, phi1,
                    A2, w2, fres2, fper2, phi2, re0, im0):
        """
        Sum of two complex Gaussians
        """
        z = (ComplexGaussianSimple.model(f, A1, w1, fres1, fper1, phi1, 0, 0) +
             ComplexGaussianSimple.model(f, A2, w2, fres2, fper2, phi2, re0, im0))
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
    def __init__(self,  p0, f_min, f_max, f_array, y_re, y_im, model:BaseComplexModel):
        self.p0=p0 #initial parameters
        self.p0_free = p0 #changed later by fix()

        self.f_min = f_min #minimum freq. for fit
        self.f_max = f_max #maxium freq. for fit
        self.f_array = f_array #frequency values
        self.y_re = y_re #real part of measurements values
        self.y_im = y_im #imag. part of measurements values

        self.mask_f, self.mask_y, self.mask_y_re, self.mask_y_im =\
            self.mask_arrays(f_min, f_max, f_array, y_re, y_im)
        #already mask the used arrays to the freq. span of interest

        self.fixed = {}        # fixed parameters {name: value}
        self.bounds_free = 0 #Initial variable to store bounds for remaining free variables only
        self.param_indices_free = []
        self.param_names_free = []

        self.model = model  # The model used for fitting (see above the model available)
        self.result_simplex_wpm={} # fitted parameter results using simplex
        self.result = []       # fitted parameter results
        self.cov = None        # covariance matrix

    #####################
    # Fixing parameters
    #####################
    def fix(self, **kwargs):

        self.fixed = {}
        self.bounds_free = 0
        self.param_indices_free = []
        self.param_names_free = []

        if kwargs == {}:
            self.fixed = {}
        else:
            for k, v in kwargs.items():
                if k not in self.model.param_names:
                    raise ValueError(f"Parameter {k} not in model")
                self.fixed[k] = v

        self.param_names_free = [
            name for name in self.model.param_names if name not in self.fixed
        ]

        self.param_indices_free = [
            i for i, name in enumerate(self.model.param_names)
            if name not in self.fixed
        ]

        self.p0_free = [self.p0[i] for i in self.param_indices_free]

        lower_free = [self.model.lower_bounds[n] for n in self.param_names_free]
        upper_free = [self.model.upper_bounds[n] for n in self.param_names_free]

        self.bounds_free = (lower_free, upper_free)
    #####################
    # Simplex for initial params
    #####################
    def simplex(self, f, params, x, y, method='Powell'):
        """
        Performs a simplex optimization of parameters param, for function f, with x and y values.
        Penalizes optimization (large error) when the parameters are out of bounds.
        For method, use: 'Nelder-Mead' or 'Powell'
        """
        simplex_bounds = [(self.model.lower_bounds[name], self.model.upper_bounds[name])
                          for name in self.param_names_free]
        def error(params, x, y):
            if method=='Nelder-Mead':
                for i, name in enumerate(self.param_names_free):
                    if params[i] < self.model.lower_bounds[name] or params[i] > self.model.upper_bounds[name]:
                       return 1e20 #Penalizes parameter when out of bounds

            model = f(x, *params)
            return np.sum(np.abs(y - model)**2)
        if method=='Nelder-Mead':
            minim = minimize(error, params, args=(x, y), method='Nelder-Mead')
        else:
            minim = minimize(error, params, args=(x, y), method='Powell', bounds= simplex_bounds)
        return minim.x

    #####################
    # Mask arrays by frequency window
    #####################
    @staticmethod
    def mask_arrays(f_min, f_max, f_array, y_re, y_im):
        """
        Mask the mesurement arrays to the freq. domain  of interest.
        """
        mask = (f_array >= f_min) & (f_array <= f_max)
        f_fit = f_array[mask]
        y_fit_re = y_re[mask]
        y_fit_im = y_im[mask]
        y_fit = np.concatenate([y_fit_re, y_fit_im])
        return f_fit, y_fit, y_fit_re, y_fit_im

    #####################
    # Wrap model (limited for free parameters)
    #####################
    def wrapped_model(self, f, *free_params):
        """
        Fixes some parameters in a general model and returns a simplified one.
        """
        full_params = []
        free_idx = 0
        for name in self.model.param_names:
            if name in self.fixed:
                full_params.append(self.fixed[name])
            else:
                full_params.append(free_params[free_idx])
                free_idx += 1
        return self.model.model(f, *full_params)

    def list_results(self, optimized):
        """
        Builds a dictionary with the parameters of the model using the values in optimized
        """
        results = {}
        free_idx = 0
        for name in self.model.param_names:
            if name in self.fixed:
                results[name] = self.fixed[name]
            else:
                results[name] = optimized[free_idx]
                free_idx += 1
        return results

    def simplex_wpm(self, method='Powell'):
        """
        Executes simplex on the constrained model (wrapped_model).
        For method, use: 'Nelder-Mead' or 'Powell'
        """
        new_p0 = self.simplex(self.wrapped_model, self.p0_free, self.mask_f, self.mask_y, method)
        self.result_simplex_wpm = self.list_results(new_p0)

        return  self.result_simplex_wpm

    def update_p0(self, new_p0, **kwargs):
        """
        Updates the initial parameters (new_p0) and the ones fixed (kwargs)
        """
        self.p0 = new_p0
        self.fix(**kwargs)

    #####################
    # Fit function
    #####################
    def fit(self, simplex_b:bool):
        """
        Fits the masked data (mask_...) to the constrained model (wrapped_model), using initial parameters (p0_fit).
        With option for an internal simplex (simplex_b) to obtain "better" initial parameters.
        """
        #Optional internal Simplex run
        if simplex_b:
            p0_fit = self.simplex_wpm()
        else:
            p0_fit = self.p0_free

        # Actual Fit
        params_free, cov = curve_fit(self.wrapped_model, self.mask_f, self.mask_y, p0=p0_fit, bounds=self.bounds_free)

        # Build full results
        self.result = self.list_results(params_free)
        self.cov = cov

        return self.result, self.cov

    #####################
    # Summary with uncertainties, scientific formatting, units
    #####################
    def summary(self, digits=4):
        """
        Summary table of the results of fit().
        """
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

    @staticmethod
    def unconcatenate(z):
        """
        Unconcatenate array z. In the current context, gives back the Re and Im parts of a concattenated Z array.
        """
        N = len(z)//2
        return z[:N], z[N:]

    #####################
    # Plotting fits
    #####################
    def plot_data_fit(self, param_set):
        """
        Plot real and imaginary parts with data and the model using param_set.
        """
        plt.scatter(self.mask_f, self.mask_y_re, s=50, facecolors='none', edgecolors='blue', label="Re")
        plt.scatter(self.mask_f, self.mask_y_im, s=50, facecolors='none', edgecolors='red', label="Im")

        param_vals = self.model.model(self.mask_f, *[param_set[name] for name in self.model.param_names])


        fit_re, fit_im = self.unconcatenate(param_vals)

        plt.plot(self.mask_f, fit_re, 'cyan', label="Re_fit")
        plt.plot(self.mask_f, fit_im, 'yellow', label="Im_fit")

        plt.axhline(0, color='black', linewidth=1)
        plt.gca().set_facecolor('#edf1f7')
        plt.xlabel("Frequency (GHz)")
        plt.ylabel("dLij (pH)")
        plt.title("Fitting Results")
        plt.legend()
        plt.show()

    def plot_fit_simplex_wpm(self):
        """
        Plots directly the result from the simplex_wpm().
        """
        self.plot_data_fit(self.result_simplex_wpm)

    def plot_fit(self):
        """
        Plots directly the result from the fit().
        """
        self.plot_data_fit(self.result)

    def param_manipulator(self, variation, step, param_set):
        """
        Creates a plot of data and model (param_set) with option to manipulate the parameters based on:
         variation (max. percentage of deviation from original value) and step (steps for manipulation)
        """
        relative_sliders = {}

        for name in param_set:
            relative_sliders[name] = widgets.FloatSlider(
                value=1.0,
                min=1 - variation,
                max=1 + variation,
                step=step,
                description="F*" + name + ", F=",
                readout_format='.4f',
                continuous_update=False
            )

        def wrapper(**multipliers):
            """
            Multiply parameters by widget-provided multipliers and plot the fit.
            """
            #Compute new parameter values
            new_params = {
                name: param_set[name] * multipliers.get(name, 1.0)
                for name in param_set
            }

            #Pass the dictionary directly (no **kwargs!)
            self.plot_data_fit(param_set=new_params)
            print(new_params)
        interact(wrapper, **relative_sliders)

    def manipulator_simplex_wpm(self, variation, step):
        """Manipulator for the results of simplex_wpm()"""
        self.param_manipulator(variation, step, self.result_simplex_wpm)

    def manipulator_fit(self, variation, step):
        """Manipulator for the results of simplex_wpm()"""
        self.param_manipulator(variation, step, self.result)

    #####################
    # Example vg function for DE spin waves
    #####################
    @staticmethod
    def vg(f):
        """
        Group velocity (m/s) of magnetostatic surface waves in YIG (n=0), as function of frequency in Hz
        """
        mu_o = 4*np.pi*1e-7
        t = 105*1e-9  # m
        k = 2.95*1e6  # rad/m
        M = 139.6*1e3*mu_o  # T
        H_eff = M
        gamma = 182.21/(2*np.pi)  # GHz/T

        return (2*np.pi) * gamma*gamma*M*H_eff*t*np.exp(-2*k*t)/(4*f)