import numpy as np
import fitting_functions as func
from fit_PSWS_class import BaseComplexModel
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

pi = np.pi
# Bounds stored as dicts keyed by parameter name
LOWER_BOUNDS = {"Ar":-1e2, "Ai":-1e2, "fo":0, "df":0.0001, "re0": -0.5,  "im0": -0.5, "B":0, "tau":0, "phi":0,
                "Ho":0, "dH":0.0001, "m":-1000}
UPPER_BOUNDS = {"Ar":1e2, "Ai":1e2, "fo":70, "df":2, "re0": 0.5,  "im0": 0.5, "B":0.01, "tau":1000, "phi":2*pi,
                "Ho":2, "dH":0.2, "m":1000}

#########################
# Complex Lorentzian Models
#########################
class ComplexLorentzian_freq(BaseComplexModel):
    def __init__(self):
        param_names = ["Ar","Ai","fo","df","re0","im0"]
        super().__init__(param_names)
        self.lower_bounds = LOWER_BOUNDS
        self.upper_bounds = UPPER_BOUNDS

        self.units = {"Ar": "U",
                      "Ai": "U",
                      "fo": "GHz",
                      "df": "GHz",
                      "re0": "U",
                      "im0" : "U",
                      }
        self.fit_type = "frequency"

    @staticmethod
    def model(f, Ar, Ai, fo, df, re0, im0):
        """df is the half-linewidth at half-maximum."""
        re = re0 + Ar*func.antisym_lorentzian(f,fo,df) + Ai*func.sym_lorentzian(f,fo,df)
        im = im0 + Ar*func.sym_lorentzian(f,fo,df) - Ai*func.antisym_lorentzian(f,fo,df) #inverted signs compared to
        # normal lorentzian
        return np.concatenate([re, im])

    @staticmethod
    def phase_deg(Ar, Ai, fo, df, re0, im0):
        A_complex = Ar + 1j * Ai
        theta = np.angle(A_complex) # rad
        if theta < 0:
            theta += 2 * np.pi
        theta_deg = np.degrees(theta)  # degrees
        return theta_deg

class ComplexLorentzianRipple_freq(BaseComplexModel):

    def __init__(self):
        param_names = [
            "Ar","Ai",
            "fo","df",
            "re0","im0",
            "B","tau","phi"
        ]
        super().__init__(param_names)
        self.lower_bounds = LOWER_BOUNDS
        self.upper_bounds = UPPER_BOUNDS
        self.fit_type = "frequency"

    @staticmethod
    def model(f,
              Ar, Ai,
              fo, df,
              re0, im0,
              B, tau, phi):

        theta = 2*np.pi*tau*f + phi

        ripple_re = B*np.cos(theta)
        ripple_im = (-1)*B*np.sin(theta) #we artificially chnage teh sign of this imaginary part

        re = (
            re0
            + ripple_re
            + Ar*func.antisym_lorentzian(f,fo,df)
            + Ai*func.sym_lorentzian(f,fo,df)
        )

        im = (
            im0
            + ripple_im
            + Ar*func.sym_lorentzian(f,fo,df)
            - Ai*func.antisym_lorentzian(f,fo,df)
        )

        return np.concatenate([re, im])

    @staticmethod
    def phase_deg(Ar, Ai, fo, df, re0, im0, B, tau, phi):
        A_complex = Ar + 1j * Ai
        theta = np.angle(A_complex)  # rad
        if theta < 0:
            theta += 2 * np.pi
        theta_deg = np.degrees(theta)  # degrees
        return theta_deg

class ComplexLorentzian_field(BaseComplexModel):
    def __init__(self):
        param_names = ["Ar","Ai","Ho","dH","re0","im0"]
        super().__init__(param_names)
        self.lower_bounds = LOWER_BOUNDS
        self.upper_bounds = UPPER_BOUNDS

        self.units = {"Ar": "U",
                      "Ai": "U",
                      "Ho": "T",
                      "dH": "T",
                      "re0": "U",
                      "im0" : "U",
                      }
        self.fit_type = "field"

    @staticmethod
    def model(H, Ar, Ai, Ho, dH, re0, im0):
        """dH is the half-linewidth at half-maximum."""
        #H = np.asarray(H).ravel() to delete
        re = re0 + Ar*func.antisym_lorentzian(H,Ho,dH) + Ai*func.sym_lorentzian(H,Ho,dH)
        im = im0 - Ar*func.sym_lorentzian(H,Ho,dH) + Ai*func.antisym_lorentzian(H,Ho,dH)
        return np.concatenate([re, im])

    @staticmethod
    def phase_deg(Ar, Ai, Ho, dH, re0, im0):
        A_complex = Ar + 1j * Ai
        theta = np.angle(A_complex) # rad
        if theta < 0:
            theta += 2 * np.pi
        theta_deg = np.degrees(theta)  # degrees
        return theta_deg

class ComplexLorentzian_slope_field(BaseComplexModel):
    def __init__(self):
        param_names = ["Ar","Ai","Ho","dH","m","re0","im0"]
        super().__init__(param_names)
        self.lower_bounds = LOWER_BOUNDS
        self.upper_bounds = UPPER_BOUNDS

        self.units = {"Ar": "U",
                      "Ai": "U",
                      "Ho": "T",
                      "dH": "T",
                      "m": "1/T",
                      "re0": "U",
                      "im0" : "U",
                      }
        self.fit_type = "field"

    @staticmethod
    def model(H, Ar, Ai, Ho, dH, m, re0, im0):
        """dH is the half-linewidth at half-maximum."""
        #H = np.asarray(H).ravel() to delete
        re = re0 + Ar*( func.antisym_lorentzian(H,Ho,dH) + m*H) + Ai*( func.sym_lorentzian(H,Ho,dH) + m*H)
        im = im0 - Ar*( func.sym_lorentzian(H,Ho,dH) + m*H) + Ai*( func.antisym_lorentzian(H,Ho,dH) + m*H)
        return np.concatenate([re, im])

    @staticmethod
    def phase_deg(Ar, Ai, Ho, dH, m, re0, im0):
        A_complex = Ar + 1j * Ai
        theta = np.angle(A_complex)  # rad
        if theta < 0:
            theta += 2 * np.pi
        theta_deg = np.degrees(theta)  # degrees
        return theta_deg


#########################
# Linewidth models
#########################

class linear_dHvsf_FMR():
    def __init__(self, dH_data, dH_err_data, f_data, gamma):
        self.dH_data = dH_data #We expect dH in T
        self.dH_err_data = dH_err_data
        self.f_data = f_data #We expect f in GHz
        self.gamma = gamma #We expect gamma in GHz/T

    @staticmethod
    def linear(x, m, b):
        return m * x + b

    def fit(self):
        self.popt, self.pcov = curve_fit(self.linear, self.f_data, self.dH_data)
        self.slope, self.dH0 = self.popt

        self.alpha = self.gamma * self.slope / 2

    def results(self):
        self.fit()
        plt.rcParams.update({
            "font.size": 14,  # base font size
            "axes.labelsize": 16,  # x/y labels
            "axes.titlesize": 18,  # title
            "xtick.labelsize": 14,  # x tick labels
            "ytick.labelsize": 14,  # y tick labels
            "legend.fontsize": 14,
        })

        x_fit = np.linspace(np.min(self.f_data), np.max(self.f_data), 500)
        y_fit = self.linear(x_fit, *self.popt)
        plt.plot(x_fit, y_fit*1000, label=f"Fit: y = {self.popt[0]:.3g}x + {self.popt[1]:.3g}", color="red")
        plt.errorbar(self.f_data, self.dH_data*1000, yerr=self.dH_err_data*1000, fmt='o', capsize=3, markersize=5)
        #We assume dH in T
        plt.xlabel("f (GHz)")
        plt.ylabel(r"$\Delta$H (mT)")
        plt.show()

        print(f"alpha={self.alpha}, dH0={self.dH0 * 1000}mT")