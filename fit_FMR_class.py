import numpy as np
import fitting_functions as func
from fit_PSWS_class import BaseComplexModel

pi = np.pi
# Bounds stored as dicts keyed by parameter name
LOWER_BOUNDS = {"Ar":-1e2, "Ai":-1e2, "fo":0, "df":0.0001, "re0": -0.5,  "im0": -0.5}
UPPER_BOUNDS = {"Ar":1e2, "Ai":1e2, "fo":70, "df":2, "re0": 0.5,  "im0": 0.5}

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

    @staticmethod
    def model(f, Ar, Ai, fo, df, re0, im0):
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



