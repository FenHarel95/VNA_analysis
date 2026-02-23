import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

class BaseComplexModel:

    def __init__(self):
        self.param_names = []
        self.lower_bounds = []
        self.upper_bounds = []

    def evaluate(self, f, *params):
        raise NotImplementedError

def _base_complex_gaussian(f, A, w, fres, fper, fref, re0, im0):
    re = re0 + (A/(w*np.sqrt(np.pi/2))) * \
         np.exp(-2*((f-fres)/w)**2) * \
         np.cos(2*np.pi*(f-fref)/fper)

    im = im0 + (A/(w*np.sqrt(np.pi/2))) * \
         np.exp(-2*((f-fres)/w)**2) * \
         np.sin(2*np.pi*(f-fref)/fper)

    return np.concatenate([re, im])

class ComplexGaussian(BaseComplexModel):

    def __init__(self):
        super().__init__()

        self.param_names = ["A","w","fres","fper","fref","re0","im0"]

        self.lower_bounds = [0,0.005,0.1,0.001,0.1,-10,-10]
        self.upper_bounds = [np.inf,2,50,1,50,10,10]

    def evaluate(self, f, *p):
        return _base_complex_gaussian(f, *p)

class TwoComplexGaussian(BaseComplexModel):

    def __init__(self):
        super().__init__()

        self.param_names = [
            "A1","w1","fres1","fper1","fref1",
            "A2","w2","fres2","fper2","fref2",
            "re0","im0"
        ]

        self.lower_bounds = [0,0.005,0.1,0.001,0.1,
                             0,0.005,0.1,0.001,0.1,
                             -10,-10]

        self.upper_bounds = [np.inf,2,50,1,50,
                             np.inf,2,50,1,50,
                             10,10]

    def evaluate(self, f, *p):
        return (
            _base_complex_gaussian(f, *p[:5], 0, 0)
            + _base_complex_gaussian(f, *p[5:10], p[10], p[11])
        )

class TwoComplexGaussianPhi(BaseComplexModel):

    def __init__(self):
        super().__init__()

        self.param_names = ["A","w","fres","fper","fref1","fref2","re0","im0"]

        self.lower_bounds = [0,0.005,0.1,0.001,0.1,0.1,-10,-10]
        self.upper_bounds = [np.inf,2,50,1,50,50,10,10]

    def evaluate(self, f, *p):
        return (
            _base_complex_gaussian(f, p[0],p[1],p[2],p[3],p[4],0,0)
            + _base_complex_gaussian(f, p[0],p[1],p[2],p[3],p[5],p[6],p[7])
        )

class ComplexGaussianVG(BaseComplexModel):

    def __init__(self):
        super().__init__()

        self.param_names = ["A","w","fres","D","fref","re0","im0"]

        self.lower_bounds = [0,0.005,0.1,0.5e-6,0.1,-10,-10]
        self.upper_bounds = [np.inf,2,50,50e-6,50,10,10]

    def evaluate(self, f, *p):
        A,w,fres,D,fref,re0,im0 = p
        return _base_complex_gaussian(f, A,w,fres, vg(f)/D, fref, re0, im0)

class ComplexFitter:

    def __init__(self, model: BaseComplexModel):
        self.model = model
        self.fixed = {}
        self.result = None
        self.cov = None

    def fix(self, **kwargs):
        self.fixed.update(kwargs)

    def _wrapper(self, f, *free_params):

        full = []
        idx = 0

        for name in self.model.param_names:
            if name in self.fixed:
                full.append(self.fixed[name])
            else:
                full.append(free_params[idx])
                idx += 1

        return self.model.evaluate(f, *full)

    def fit(self, f_min, f_max, f_array, y_re, y_im, p0_full):

        f_fit, y_fit, y_re_fit, y_im_fit = mask_arrays(
            f_min, f_max, f_array, y_re, y_im
        )

        p0, lower, upper = self._prepare(p0_full)

        params_free, cov = curve_fit(
            self._wrapper,
            f_fit,
            y_fit,
            p0=p0,
            bounds=(lower, upper),
            method='trf'
        )

        self.result = self._reconstruct(params_free)
        self.cov = cov

        self.plot(f_fit, y_re_fit, y_im_fit)

        return self.result, cov

    def _prepare(self, p0_full):
        p0 = []
        lower = []
        upper = []

        for i, name in enumerate(self.model.param_names):
            if name not in self.fixed:
                p0.append(p0_full[i])
                lower.append(self.model.lower_bounds[i])
                upper.append(self.model.upper_bounds[i])

        return p0, lower, upper

    def _reconstruct(self, free):
        full = {}
        idx = 0

        for name in self.model.param_names:
            if name in self.fixed:
                full[name] = self.fixed[name]
            else:
                full[name] = free[idx]
                idx += 1

        return full

    def plot(self, f, y_re, y_im):
        y_fit = self.model.evaluate(
            f,
            *[self.result[n] for n in self.model.param_names]
        )

        re_fit, im_fit = unconcatenate(y_fit)

        plt.scatter(f, y_re, s=50, marker='o', facecolors='none', edgecolors='blue', label="Re")
        plt.scatter(f, y_im, s=50, marker='o', facecolors='none', edgecolors='red', label="Im")
        plt.plot(f, re_fit, 'cyan', label="Re_fit")
        plt.plot(f, im_fit, 'yellow', label="Im_fit")
        plt.axhline(0, color='black')
        plt.gca().set_facecolor('#edf1f7')
        plt.legend()
        plt.show()

    def summary(self, digits=4):
        """
        Prints a formatted table of fit results with uncertainties, fixed parameters,
        automatic scientific formatting, and units.
        """
        print("\nFit Results")
        print("-" * 70)
        print(f"{'Parameter':<12}{'Value':>20}{'± Error':>15}{'Unit':>10}{'Fixed':>10}")
        print("-" * 70)

        # Get errors from covariance if available
        if self.cov is not None:
            errors = np.sqrt(np.diag(self.cov))
        else:
            errors = None

        free_index = 0

        for name in self.model.param_names:
            value = self.result[name]
            unit = getattr(self.model, "units", {}).get(name, "")

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
                    err = errors[free_index]
                    if abs(err) < 1e-3 or abs(err) > 1e3:
                        err_str = f"{err:.3e}"
                    else:
                        err_str = f"{err:.{digits}f}"
                else:
                    err_str = "-"
                free_index += 1
                fixed = "No"

            print(f"{name:<12}{value_str:>20}{err_str:>15}{unit:>10}{fixed:>10}")

        print("-" * 70)


#####################Tools
def unconcatenate(z):
    N = len(z) // 2
    real = z[:N]
    imag = z[N:]
    return real, imag

def mask_arrays(f_min, f_max, f_arry, y_array_re, y_array_im):
    mask = (f_arry >= f_min) & (f_arry <= f_max)
    f_fit = f_arry[mask]
    y_fit_re = y_array_re[mask]
    y_fit_im = y_array_im[mask]
    y_fit = np.concatenate([y_fit_re, y_fit_im])
    return f_fit, y_fit, y_fit_re, y_fit_im

def plot_fit(f, results):
    """Formated plot of the results (formated correctly) of fitting complex functions"""
    plt.scatter(results[2], results[3], s=50, marker='o', facecolors='none', edgecolors='blue', label="Re")
    plt.scatter(results[2], results[4], s=50, marker='o', facecolors='none', edgecolors='red', label="Im")
    plt.plot(results[2], unconcatenate(f(results[2], *results[0]))[0], 'cyan', label="Re_fit")
    plt.plot(results[2], unconcatenate(f(results[2], *results[0]))[1], 'yellow', label="Im_fit")
    plt.axhline(0, color='black', linewidth=1)
    plt.gca().set_facecolor('#edf1f7')
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("dLij (pH)")
    plt.title("Fitting Results")
    plt.legend()
    plt.show()

def vg(f):
    """For now, Vg for magnetostatic DE in YIG in GHz*m units"""
    mu_o = 4*np.pi*1e-7
    t = 105*1e-9 #m
    k = 2.95*1e6 # rad/m
    M = 139.6*1e3*mu_o #T
    H_eff = M
    gamma = 182.21/(2*np.pi) #GHz/T

    return (2*np.pi) * gamma*gamma*M*H_eff*t*np.exp(-2*k*t)/(4*f)