import numpy as np

#Constants
pi = np.pi
mu_0 = 4*pi*1e-7 #SI units

def params_cubic_ferromagnet(Ms,Hc,Hu,Hs,Hex,t):
    """Calculates the magnetic constants from the stifness fields relevant for a cubic ferromagnet.
    Ref. Solano_2024 Phd Thesis, (App. B, pg. 193)
    All fields should be in Teslas, t in m
    Ms (T), saturation magnetization
    Hc (T), uniaxial anisotropy field, parametrized with respect to axis per to film,
    Hs (T), perpendicular surface anisotropy field, axis perp. to film.
    Hex (Y), exchange stiffness field
    """
    Kc = (Hc/mu_0)*Ms/2
    Ku = (Hu/mu_0)*Ms/2
    Ks = (Hs/mu_0)*(Ms*t)/2
    A = (Hex/mu_0)*(Ms*t**2)/(2*pi**2)

    return Kc, Ku, Ks, A

def dH_from_df(df, gamma, f, H, Hx,Hy, inP=True):
    """Calculated the field linewidth dH from the frequency linewidth df"""
    if inP:
        dH = 2*f*df/(gamma*(2*H+Hx+Hy))
    else:
        dH = df/(gamma*(H+Hx))

    return dH