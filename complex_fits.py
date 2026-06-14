import numpy as np
from scipy.optimize import curve_fit, minimize
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact

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
    def __init__(self,  p0, f_min, f_max, f_array, y_re, y_im, model):
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

        self.manip_parameters = {} #Stores parameters changed by param_manipulator()

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
                results[name] = float(self.fixed[name])
            else:
                results[name] = float(optimized[free_idx])
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

    def export_fit_dict(self):
        """Creates dictionary ready for storing fitting results."""
        result = self.result
        cov = self.cov

        if cov is not None:
            errors = np.sqrt(np.diag(cov))
        else:
            errors = None

        export = {
            "params": {},
            "errors": {},
            "units": {},
            "fixed": {}
        }

        free_index = 0

        for name in self.model.param_names:

            export["params"][name] = float(result[name])
            export["units"][name] = self.model.units.get(name, "")

            if name in self.fixed:
                export["errors"][name] = None
            else:
                export["errors"][name] = float(errors[free_index]) if errors is not None else None
                free_index += 1

        return export


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
        self.manip_parameters={} #reset parameters

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
            self.manip_parameters = new_params
        interact(wrapper, **relative_sliders)
        return self.manip_parameters

    def manipulator_simplex_wpm(self, variation, step):
        """Manipulator for the results of simplex_wpm()"""
        return self.param_manipulator(variation, step, self.result_simplex_wpm)

    def manipulator_fit(self, variation, step):
        """Manipulator for the results of simplex_wpm()"""
        return self.param_manipulator(variation, step, self.result)

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