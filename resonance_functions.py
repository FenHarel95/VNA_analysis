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
    f = gamma*np.sqrt( (H+Hx)*(H+Hy) ) #GHz
    return f

def kittel_cubic(H,gamma,Ms,Hc,Hu,Hs,A,T):
    """
    FMR for films with cubic anisotropy.
    gamma (GHz/T), is the gyromagnetic ratio
    M_s (T), saturation magnetization
    B_u (T), uniaxial anisotropy, parametrized with respect to axis per to film,
    B_s (T), perpendicular surface anisotropy, axis perp. to film.
    A (J/m), exchange stiffness constant,
    T (m), thickness
    B_0 (T), external magnetic field.
    """