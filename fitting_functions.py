import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

####################Bounds
#A, w, fres, fper, fref, re0, im0
lower_bounds = [0, 0.005, 0.1, 0.001, 0.1, -10, -10 ]
upper_bounds = [np.inf, 2, 50, 1, 50, 10, 10 ]
#A1, w1, fres1, fper1, fref1, A2, w2, fres2, fper2, fref2, re0, im0
two_lower_bounds = [0, 0.005, 0.1, 0.001, 0.1, 0, 0.05, 0.1, 0.001, 0.1, -10, -10 ]
two_upper_bounds = [np.inf, 2, 50, 1, 50, np.inf, 2, 50, 1, 50, 10, 10 ]
#A1, w1, fres1, fper1, fref1, fref2, re0, im0
two_lower_bounds_phi = [0, 0.005, 0.1, 0.001, 0.1, 0.1, -10, -10 ]
two_upper_bounds_phi = [np.inf, 2, 50, 1, 50, 50, 10, 10 ]
#A, w, fres, D, fref, re0, im0
lower_bounds_vg = [0, 0.005, 0.1, 0.5*1e-6, 0.1, -10, -10 ]
upper_bounds_vg = [np.inf, 2, 50, 50*1e-6, 50, 10, 10 ]
#A1, w1, fres1, D1, fref1, A2, w2, fres2, D2, fref2, re0, im0
two_lower_bounds_vg = [0, 0.005, 0.1, 0.5*1e-6, 0.1, 0, 0.05, 0.1, 0.5*1e-6, 0.1, -10, -10 ]
two_upper_bounds_vg = [np.inf, 2, 50, 50*1e-6, 50, np.inf, 2, 50, 50*1e-6, 50, 10, 10]

bounds=(lower_bounds, upper_bounds)

#####################Constants
pi = np.pi

####################Models
def sym_lorentzian(x, xo, dx):
    amplitude = dx/((x-xo)**2 + dx**2)
    return amplitude

def antisym_lorentzian(x, xo, dx):
    amplitude = (x-xo)/((x-xo)**2 + dx**2)
    return amplitude

def complx_gaussian(f, A, w, fres, fper, fref, re0, im0):
    re = re0 + (A/(w*np.sqrt(np.pi/2)))*np.exp(-2*((f-fres)/w)*((f-fres)/w))*np.cos(2*np.pi*(f-fref)/fper)
    im = im0 + (A/(w*np.sqrt(np.pi/2)))*np.exp(-2*((f-fres)/w)*((f-fres)/w))*np.sin(2*np.pi*(f-fref)/fper)
    return np.concatenate([re, im])

def complx_gaussian_simple(f, A, w, fres, fper, phi, re0, im0):
    exp_term = A * np.exp(-2 * ((f - fres) / w) ** 2)

    phase = 2 * np.pi * f / fper - phi

    re = re0 + exp_term * np.cos(phase)
    im = im0 + exp_term * np.sin(phase)

    return np.concatenate([re, im])

def two_complx_gaussian(f, A1, w1, fres1, fper1, fref1,
                           A2, w2, fres2, fper2, fref2, re0, im0):
    z = (complx_gaussian(f, A1, w1, fres1, fper1, fref1, 0, 0)
         + complx_gaussian(f, A2, w2, fres2, fper2, fref2, re0, im0))
    return z

def two_complx_gaussian_simple(f, A1, w1, fres1, fper1, phi1,
                           A2, w2, fres2, fper2, phi2, re0, im0):
    z = (complx_gaussian_simple(f, A1, w1, fres1, fper1, phi1, 0, 0)
         + complx_gaussian_simple(f, A2, w2, fres2, fper2, phi2, re0, im0))
    return z

def two_complx_gaussian_phi(f, A1, w1, fres1, fper1, fref1,
                           fref2, re0, im0):
    z = (complx_gaussian(f, A1, w1, fres1, fper1, fref1, 0, 0)
         + complx_gaussian(f, A1, w1, fres1, fper1, fref2, re0, im0))
    return z

def complx_gaussian_vg(f, A, w, fres, D, fref, re0, im0):
    z = complx_gaussian(f, A, w, fres, vg(f)/D, fref, re0, im0)
    return z

def two_complx_gaussian_vg(f, A1, w1, fres1, D1, fref1,
                           A2, w2, fres2, D2, fref2, re0, im0):
    z = (complx_gaussian(f, A1, w1, fres1, vg(f)/D1, fref1, 0, 0)
         + complx_gaussian(f, A2, w2, fres2, vg(f)/D2, fref2, re0, im0))
    return z

####################Fitting functions
def fit_complx_gaussian(A, w, fres, fper, fref, re0, im0,
                        f_min, f_max, f_arry, y_array_re, y_array_im):
    p0 = [A, w, fres, fper, fref, re0, im0]
    f_to_fit, y_to_fit, y_to_fit_re, y_to_fit_im = mask_arrays(f_min, f_max, f_arry, y_array_re, y_array_im)

    params, cov = curve_fit( complx_gaussian, f_to_fit, y_to_fit, p0=p0, bounds=(lower_bounds, upper_bounds) )
    result = params, cov, f_to_fit, y_to_fit_re, y_to_fit_im
    plot_fit(complx_gaussian, result)
    print(f"A={params[0]:.2f}, w={params[1]:.2f} GHz, fres={params[2]:.2f} GHz, fper={params[3]:.3f}  GHz, "
          f"fref={params[4]:.2f}  GHz, re0={params[5]:.2f} pH, im0={params[6]:.2f} pH")
    return result

def fit_two_complx_gaussian(A1, w1, fres1, fper1, fref1,
                            A2, w2, fres2, fper2, fref2, re0, im0,
                            f_min, f_max, f_arry, y_array_re, y_array_im):
    p0 = [A1, w1, fres1, fper1, fref1, A2, w2, fres2, fper2, fref2, re0, im0]
    f_to_fit, y_to_fit, y_to_fit_re, y_to_fit_im = mask_arrays(f_min, f_max, f_arry, y_array_re, y_array_im)

    params, cov = curve_fit( two_complx_gaussian, f_to_fit, y_to_fit, p0=p0,
                             bounds=(two_lower_bounds, two_upper_bounds) )
    result = params, cov, f_to_fit, y_to_fit_re, y_to_fit_im
    plot_fit(two_complx_gaussian, result)
    print(f"A1={params[0]:.2f}, w1={params[1]:.2f} GHz, fres1={params[2]:.2f} GHz, fper1={params[3]:.3f}  GHz, "
          f"fref1={params[4]:.2f}  GHz,\n"
          f"A2={params[5]:.2f}, w2={params[6]:.2f} GHz, fres2={params[7]:.2f} GHz, fper2={params[8]:.3f}  GHz, "
          f"fref2={params[9]:.2f}  GHz, re0={params[10]:.2f} pH, im0={params[11]:.2f} pH")
    return result

def fit_two_complx_gaussian_phi(A1, w1, fres1, fper1, fref1,
                             fref2, re0, im0,
                            f_min, f_max, f_arry, y_array_re, y_array_im):
    p0 = [A1, w1, fres1, fper1, fref1, fref2, re0, im0]
    f_to_fit, y_to_fit, y_to_fit_re, y_to_fit_im = mask_arrays(f_min, f_max, f_arry, y_array_re, y_array_im)

    params, cov = curve_fit( two_complx_gaussian_phi, f_to_fit, y_to_fit, p0=p0,
                             bounds=(two_lower_bounds_phi, two_upper_bounds_phi) )
    result = params, cov, f_to_fit, y_to_fit_re, y_to_fit_im
    plot_fit(two_complx_gaussian_phi, result)
    print(f"A={params[0]:.2f}, w={params[1]:.2f} GHz, fres={params[2]:.2f} GHz, fper1={params[3]:.3f}  GHz, "
          f"fref2={params[4]:.2f}  GHz,\n"
          f"re0={params[5]:.2f}, im0={params[6]:.2f}")
    return result

def fit_complx_gaussian_vg(A, w, fres, D, fref, re0, im0,
                        f_min, f_max, f_arry, y_array_re, y_array_im):
    p0 = [A, w, fres, D, fref, re0, im0]
    f_to_fit, y_to_fit, y_to_fit_re, y_to_fit_im = mask_arrays(f_min, f_max, f_arry, y_array_re, y_array_im)

    params, cov = curve_fit(complx_gaussian_vg, f_to_fit, y_to_fit, p0=p0, bounds=(lower_bounds_vg, upper_bounds_vg))
    result = params, cov, f_to_fit, y_to_fit_re, y_to_fit_im
    plot_fit(complx_gaussian_vg, result)
    print(f"A={params[0]:.2f}, w={params[1]:.2f} GHz, fres={params[2]:.3f} GHz, D={params[3]*1E6:.1f} um, "
          f"fref={params[4]:.2f}  GHz, re0={params[5]:.2f} pH, im0={params[6]:.2f} pH")
    return result

def fit_two_complx_gaussian_vg(A1, w1, fres1, D1, fref1,
                            A2, w2, fres2, D2, fref2, re0, im0,
                            f_min, f_max, f_arry, y_array_re, y_array_im):
    p0 = [A1, w1, fres1, D1, fref1, A2, w2, fres2, D2, fref2, re0, im0]
    f_to_fit, y_to_fit, y_to_fit_re, y_to_fit_im = mask_arrays(f_min, f_max, f_arry, y_array_re, y_array_im)

    params, cov = curve_fit(two_complx_gaussian_vg, f_to_fit, y_to_fit, p0=p0,
                            bounds=(two_lower_bounds_vg, two_upper_bounds_vg))
    result = params, cov, f_to_fit, y_to_fit_re, y_to_fit_im
    plot_fit(two_complx_gaussian_vg, result)
    print(f"A1={params[0]:.2f}, w1={params[1]:.2f} GHz, fres1={params[2]:.2f} GHz, D1={params[3]*1e6:.1f}  um, "
          f"fref1={params[4]:.2f}  GHz,\n"
          f"A2={params[5]:.2f}, w2={params[6]:.2f} GHz, fres2={params[7]:.2f} GHz, D2={params[8]*1e6:.1f}  um, "
          f"fref2={params[9]:.2f}  GHz, re0_={params[10]:.2f} pH, im0_={params[11]:.2f} pH")
    return result

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