import numpy as np

#Constants
mu_0 = 4*np.pi*1e-7 #SI units

def kalinikos(B_0, k, gamma, M_s, B_u, B_s, A, T,  kind="SW", n=0):  # in GHz, Ms in A/m, B in T
    """Based on Olga Gladii_2016_thesis (https://theses.hal.science/tel-01724624v1/file/Gladii_Olga_2016_ED182.pdf)
    k (rad/m), is the wave vector
    gamma (rad/T), is the gyromagnetic ratio
    M_s (A/m), saturation magnetization
    B_u (T), uniaxial anisotropy, parametrized with respect to axis per to film,
    B_s (T), perpendicular surface anisotropy, axis perp. to film.
    A (J/m), exchange stiffness constant,
    T (m), thickness
    B_0 (T), external magnetic field.
    """

    H0 = B_0 / mu_0  # A/m
    Hu = B_u / mu_0  # A/m
    Hs = B_s / mu_0  # A/m

    Lambda = np.sqrt(2 * A / (mu_0 * M_s ** 2))
    k_n = np.sqrt(k ** 2 + (n * np.pi / T) ** 2)
    rat = k ** 2 / k_n ** 2
    kron_n = (n == 0)
    P_nn = rat * (1 - (2 / (1 + kron_n)) * rat * (1 - (-1) ** n * np.exp(-k * T)) / (k * T))

    if kind == "BVW":
        return gamma * mu_0 * np.sqrt(
            (H0 + M_s * Lambda ** 2 * k_n ** 2) * \
            (H0 - Hu - (2) ** n * Hs + M_s * Lambda ** 2 * k_n ** 2 + M_s * (1 - P_nn))
        ) * (1e-9 / (2 * np.pi))
    if kind == "DE":
        return gamma * mu_0 * np.sqrt(
            (H0 + M_s * Lambda ** 2 * k_n ** 2 + M_s * P_nn) * \
            (H0 - Hu - (2) ** n * Hs + M_s * Lambda ** 2 * k_n ** 2 + M_s * (1 - P_nn))
        ) * (1e-9 / (2 * np.pi))
    if kind == "outP":
        return gamma * mu_0 * np.sqrt(
            (H0 - (M_s - Hu)) * \
            (H0 - (M_s - Hu))
        ) * (1e-9 / (2 * np.pi))
    else:
        print("No valid spin wave kind specified in dispersion. "
              "Please use BVW, SW or outP.")
        return k * 0


def hematite_modes(H, gamma, k, HD, HK, Hex):
    """Based on supplementary of ElKanj2023 (https://www.science.org/doi/10.1126/sciadv.adh1601)"""
    # all fields in T, k in rad/m
    # frequencies in GHz
    gamma_Hema = gamma  # GHz/T
    aex = 1.7E-10  # m, Wang2023_Supp
    MsRHex = 0.013 / (4 * np.pi)  # ElKanj2023

    fmr = gamma_Hema * np.sqrt(H * (H + HD) + HK * Hex) + 0 * k
    sswm = gamma_Hema * np.sqrt(H * (H + HD) + HK * Hex + (Hex * aex * k) * (Hex * aex * k))
    Kang_bulk_klln = gamma_Hema * np.sqrt(
        H * (H + HD) + HK * Hex + (MsRHex) * (H + HD) * (H + HD) + ((Hex * aex * k) * (Hex * aex * k)))
    Kang_sw = gamma_Hema * (1 / 2) * (H * (H + HD) + HK * Hex + (Hex * aex * k) * (Hex * aex * k)) / (
    (H + HD)) + gamma_Hema * (1 / 2) * (1 + MsRHex) * (H + HD)
    modes = {
        'FMR': fmr,
        'SSWM1': sswm,
        'Bulk//n': Kang_bulk_klln,
        'SW': Kang_sw
    }
    return modes

def paramagnetic_resonance( H, m, Ha):
    f = m * np.sqrt(H*H + Ha*Ha)
    return f

def kittel_general(H, gamma, Hx, Hy):
    """
    General FMR
    gamma (GHz/T)
    H (T)
    """
    f = gamma*np.sqrt( abs(H+Hx)*abs(H+Hy) ) #GHz
    return f

def kittel_cubic_inP(H,gamma,Ms,Hc,Hu,Hs,Hex,loo=True, n1=False):
    """
    FMR for films with cubic anisotropy. For Only valid for modes n=0,1
    gamma (GHz/T), is the gyromagnetic ratio. Ref. Solano_2024 Phd Thesis
    H (T), external magnetic field.
    Ms (T), saturation magnetization
    Hc (T), uniaxial anisotropy field, parametrized with respect to axis per to film,
    Hs (T), perpendicular surface anisotropy field, axis perp. to film.
    Hex (Y), exchange stiffness field,
    loo, True when Ms & H // to [100], False when Ms & H // to [110]
    """
    if n1:
        n=1
    else:
        n=0
    if loo:
        fc1=1
        fc2=1
    else:
        fc1=-1
        fc2=1/2
    hx = fc1*Hc + n*Hex
    hy = fc2*Hc + Ms + - Hu - (1+n)*Hs + n*Hex
    f = kittel_general(H, gamma, hx, hy)
    return f

def kittel_cylinder_longA(H, gamma, Ms, Ku, Nz, Nx, Ny):
    """Ms and H in T, Ku in J/m, gamma in GHz/T"""
    Hu = 2*Ku/Ms
    hx = mu_0*Hu + (Ny-Nz)*Ms
    hy = mu_0*Hu + (Nx-Nz)*Ms
    f = kittel_general(H, gamma, hx, hy)
    return f

def inverse_kittel_general(f, gamma, Hx, Hy):
    """
        Returns the largest positive resonance field
        gamma (GHz/T)
        f (GHz)
        Hx, Hy (T)
    """
    H1 = ( -(Hx+Hy) + np.sqrt((Hx+Hy)**2 - 4*(Hx*Hy-(f/gamma)**2)))/2
    H2 = ( -(Hx+Hy) - np.sqrt((Hx+Hy)**2 - 4*(Hx*Hy-(f/gamma)**2)))/2

    if H1>H2:
        H = H1
    else:
        H = H2
    return H

def inverse_bulkHematite(f, gamma, HD, HK, Hex):
    """
        Returns the largest positive resonance field
        gamma (GHz/T)
        f (GHz)
        HD, HK, Hex (T)
    """
    H1 = ( -HD + np.sqrt((HD)**2 - 4*(HK*Hex-(f/gamma)**2)))/2
    H2 = ( -HD - np.sqrt((HD)**2 - 4*(HK*Hex-(f/gamma)**2)))/2

    if H1>H2:
        H = H1
    else:
        H = H2
    return H