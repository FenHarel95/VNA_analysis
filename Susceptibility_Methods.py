import numpy as np

c0 = 299_792_458 # in meters per second (m/s)

def reflexion_p(s11_m, s21_m, s12_m, s22_m, deem=False, deem_phase=1):
    # This model (Bilzer_2007_Thesis) assumes a reciprocal network. So we use S12=S21= (S12_m+S21_m)/2
    s21 = (s21_m + s12_m) / 2

    # Geometric mean to correct for sample position
    s1122 = np.sqrt(s11_m * s22_m)

    # Deembedding Sij's to account for empty parts of the waveguide.
    if deem: #manual deembedding of empty CPW, requires phase
        s21b = deem_phase * s21
        s1122b = deem_phase * s1122
    else: #Assumes deembedding is included in VNA's Channel/Offset/Delay when loading state
        s21b = s21
        s1122b = s1122

    # Calculate parameter K
    k = (1 + s1122b * s1122b - s21b * s21b) / (2 * s1122b)

    # Calculate reflexion parameter
    reflexion_1 = k + np.sqrt(k*k-1)
    reflexion_2 = k - np.sqrt(k*k-1)

    if np.abs(reflexion_1)<1:
        reflexion = reflexion_1
    else:
        reflexion = reflexion_2

    # Calculate the parameter P
    p = (s1122b+s21b-reflexion)/( 1-(s1122b+s21b)*reflexion)

    return reflexion, p


def permitt_permeab(
    ls, # ls=sample length in m
    reflexion,
    propagation,
    frequency, #in GHz
    epsaverage=17  # Should be adjusted according to each CPW+sample combination.
):
    prop_amp = np.abs(propagation)
    prop_phase = np.angle(propagation)
    g = 1 # is a coefficient related to the sample/CPW geometry. Arbitrarily we choose unit.

    # Calculations
    # Gamma calculation
    gamma = -np.log(prop_amp) / ls - 1j * (prop_phase / ls)

    # Gamma_fs calculation
    gamma_fs = 0 + 1j * (2 * np.pi * frequency*1e9 / c0)

    # Gamma / Gamma_fs
    gamma_ratio = gamma / gamma_fs

    # (1 + reflexion) and (1 - reflexion)
    one_plus_r = 1 + reflexion
    one_minus_r = 1 - reflexion

    # (1 + R) / (1 - R) and (1 - R) / (1 + R)
    ratio_1 = one_plus_r / one_minus_r
    ratio_2 = one_minus_r / one_plus_r

    # Permeability and Permittivity calculations
    permitt = gamma_ratio * ratio_2 * g
    first_eval_permeab = gamma_ratio * ratio_1 * (1/g)

    gamma_squared = gamma_ratio*gamma_ratio
    second_eval_permeab = gamma_squared / epsaverage

    # Return results
    return {
        "permitt": permitt,
        "first_eval_permeab": first_eval_permeab,
        "second_eval_permeab": second_eval_permeab
    }

