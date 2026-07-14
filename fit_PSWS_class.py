import numpy as np
from complex_fits import ComplexFitter

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
            "A1":1e-1, "w1":0.005, "fres1":0.1, "fper1":0.001, "fref1":0.1, "phi1":0, "D1":0.1*1e-6,
            "A2":1e-1, "w2":0.05, "fres2":0.1, "fper2":0.001, "fref2":0.1, "phi2":0, "D2":0.1*1e-6,
            "A3":1e-1, "w3":0.05, "fres3":0.1, "fper3":0.001, "fref3":0.1, "phi3":0, "D3":0.1*1e-6,
            "re0":-10, "im0":-10
        }
UPPER_BOUNDS_2 = {
            "A1":np.inf, "w1":2, "fres1":50, "fper1":1, "fref1":50, "phi1":2*np.pi, "D1":1000*1e-6,
            "A2":np.inf, "w2":2, "fres2":50, "fper2":1, "fref2":50, "phi2":2*np.pi, "D2":1000*1e-6,
            "A3":np.inf, "w3":2, "fres3":50, "fper3":1, "fref3":50, "phi3":2*np.pi, "D3":1000*1e-6,
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
        self.fit_type = "frequency"

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
        self.fit_type = "frequency"

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
        self.fit_type = "frequency"

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
        self.fit_type = "frequency"

    @staticmethod
    def model(f, A1, w1, fres1, fper1, phi1,
                    A2, w2, fres2, fper2, phi2, re0, im0):
        """
        Sum of two complex Gaussians
        """
        z = (ComplexGaussianSimple.model(f, A1, w1, fres1, fper1, phi1, 0, 0) +
             ComplexGaussianSimple.model(f, A2, w2, fres2, fper2, phi2, re0, im0))
        return z

class ThreeComplexGaussianSimple(BaseComplexModel):
    def __init__(self):
        param_names = ["A1", "w1", "fres1", "fper1", "phi1",
                       "A2", "w2", "fres2", "fper2", "phi2",
                       "A3", "w3", "fres3", "fper3", "phi3", "re0", "im0"]
        super().__init__(param_names)
        self.units = DEFAULT_UNITS_2
        self.lower_bounds = LOWER_BOUNDS_2
        self.upper_bounds = UPPER_BOUNDS_2
        self.fit_type = "frequency"

    @staticmethod
    def model(f, A1, w1, fres1, fper1, phi1,
                    A2, w2, fres2, fper2, phi2,
              A3, w3, fres3, fper3, phi3, re0, im0):
        """
        Sum of 3 complex Gaussians
        """
        z = (ComplexGaussianSimple.model(f, A1, w1, fres1, fper1, phi1, 0, 0) +
             ComplexGaussianSimple.model(f, A2, w2, fres2, fper2, phi2, 0,0) +
             ComplexGaussianSimple.model(f, A3, w3, fres3, fper3, phi3, re0, im0)
             )
        return z

class ComplexGaussianVG(BaseComplexModel):
    def __init__(self):
        param_names = ["A", "w", "fres", "D", "fref", "re0", "im0"]
        super().__init__(param_names)
        self.lower_bounds = LOWER_BOUNDS
        self.upper_bounds = UPPER_BOUNDS
        self.fit_type = "frequency"

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
        self.fit_type = "frequency"

    @staticmethod
    def model(f, A1, w1, fres1, D1, fref1,
                    A2, w2, fres2, D2, fref2, re0, im0):
        z = (ComplexGaussianVG.model(f, A1, w1, fres1, D1, fref1, 0, 0) +
             ComplexGaussianVG.model(f, A2, w2, fres2, D2, fref2, re0, im0))
        return z

