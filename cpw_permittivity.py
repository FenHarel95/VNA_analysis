"""
CPW Superstrate Permittivity Extraction — Unified Multi-Sample Code
====================================================================
Handles all combinations of:
  - Materials : GGG (ε known = 12) and Hematite (ε unknown)
  - Thicknesses: 0.2 mm and 0.5 mm
  - Lengths   : 5.7 mm, 3.725 mm, 5.0 mm
  - Conditions: Pressed (d = d_slot, known) and Free (d = d_slot + d_contact)

Structure:
  Section 1  — Paths and output directory
  Section 2  — CPW geometry (edit once)
  Section 3  — Measurement registry (one dict per s2p file)
  Section 4  — CBCPW/superstrate filling factors
  Section 5  — Core extraction engine
  Section 6  — Gap correction models
  Section 7  — Cross-validation checks
  Section 8  — Plotly interactive HTML plots
  Section 9  — CSV export
  Section 10 — Entry point

Dependencies: pip install scikit-rf plotly scipy numpy matplotlib
"""

import numpy as np
import warnings
import csv
from pathlib import Path

import skrf as rf
from scipy.special import ellipk
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# =============================================================================
# SECTION 1 — PATHS
# =============================================================================

# Directory containing all .s2p files
DATA_DIR = Path(r"C:\Users\jose1\OneDrive\PostDoc_FZU\DATA\joseS\Permitivity_extraction")
# ← change to your folder, e.g. Path("/home/user/data")

# Output directory for HTML plots and CSV
OUTPUT_DIR = Path(r"C:\Users\jose1\OneDrive\PostDoc_FZU\DATA\joseS\Permitivity_extraction\Results")
# ← change as needed

# =============================================================================
# SECTION 2 — CPW GEOMETRY  (edit once, applies to all measurements)
# =============================================================================

MIL = 0.0254          # 1 mil in mm

# Physical dimensions [mm]
W    = 16.5 * MIL     # signal strip width
G    = 5.0  * MIL     # slot (gap) width
T_CU = 0.7  * MIL     # copper conductor thickness  = base air gap in slot
H_RO = 10.0  * MIL     #  Isola Astra MT77 substrate thickness
EPS_RO = 3.0         #  Isola Astra MT77 dielectric constant

# Structure: Conductor-Backed CPW (CBCPW)
#   back copper ground plane present → substrate = RO4003 only, FR4 irrelevant

# Frequency band for reporting and cross-checks [GHz]
F_LOW  = 10.0
F_HIGH = 40.0

# Speed of light [m/s]
C_LIGHT = 3e8

# =============================================================================
# SECTION 3 — MEASUREMENT REGISTRY
# =============================================================================
# Each entry maps an s2p filename stem to its physical parameters.
#
# Keys:
#   material   : "GGG" | "Hematite" | "bare"
#   h_mm       : sample thickness [mm]  (None for bare)
#   L_mm       : sample length covering CPW [mm]  (None for bare)
#   eps_known  : known ε value (GGG=12.0, Hematite=None, bare=None)
#   is_pressed : True → d_contact=0; False → d_contact unknown
#
# Filename convention:
#   <sample>_[Pressed_]on<CPW>_<date>.s2p
#   "Pressed" present → pressed; absent → free (not pressed)
#   Bare CPW: starts with CPW name directly, no sample prefix
#
# ── HOW TO EXTEND ─────────────────────────────────────────────────────────
# Add a new dict entry with the filename stem (no .s2p) as the key.
# Set eps_known=None for unknown materials → code extracts ε for you.
# Set eps_known=<value> for reference materials → code uses for validation.
# ──────────────────────────────────────────────────────────────────────────

REGISTRY = {

    # ── Bare CPW reference ────────────────────────────────────────────────
    "SgMNMSA_300526": {
        "material":   "bare",
        "h_mm":       None,
        "L_mm":       None,
        "eps_known":  None,
        "is_pressed": False,   # irrelevant for bare
    },

    # ── GGG long (5.7 mm), ε = 12.0 ──────────────────────────────────────
    "GGG_large_Pressed_onSgMNMSA_300526": {
        "material":   "GGG",
        "h_mm":       0.5,
        "L_mm":       5.7,
        "eps_known":  12.0,
        "is_pressed": True,
    },
    "GGG_large_onSgMNMSA_300526": {
        "material":   "GGG",
        "h_mm":       0.5,
        "L_mm":       5.7,
        "eps_known":  12.0,
        "is_pressed": False,
    },

    # ── GGG short (3.725 mm), ε = 12.0 ───────────────────────────────────
    "GGG_short_Pressed_onSgMNMSA_300526": {
        "material":   "GGG",
        "h_mm":       0.5,
        "L_mm":       3.725,
        "eps_known":  12.0,
        "is_pressed": True,
    },
    "GGG_short_onSgMNMSA_300526": {
        "material":   "GGG",
        "h_mm":       0.5,
        "L_mm":       3.725,
        "eps_known":  12.0,
        "is_pressed": False,
    },

    # ── Hematite, serial SN1, thickness 0.2 mm ───────────────────────────
    # Replace SN1 with the actual serial number in the filename
    "ml20240308b2_Pressed_onSgMNMSA_300526": {
        "material":   "Hematite",
        "h_mm":       0.2,
        "L_mm":       5.0,
        "eps_known":  None,
        "is_pressed": True,
    },
    "ml20240308b2_onSgMNMSA_300526": {
        "material":   "Hematite",
        "h_mm":       0.2,
        "L_mm":       5.0,
        "eps_known":  None,
        "is_pressed": False,
    },

    # ── Hematite, serial SN2, thickness 0.5 mm ───────────────────────────
    # Replace SN2 with the actual serial number in the filename
    "Fe2O3_14023080525_Pressed_onSgMNMSA_300526": {
        "material":   "Hematite",
        "h_mm":       0.5,
        "L_mm":       5.0,
        "eps_known":  None,
        "is_pressed": True,
    },
    "Fe2O3_14023080525_onSgMNMSA_300526": {
        "material":   "Hematite",
        "h_mm":       0.5,
        "L_mm":       5.0,
        "eps_known":  None,
        "is_pressed": False,
    },
}

# Map registry stems to internal result keys
# Adjust these keys if your filename stems differ from the defaults in REGISTRY
STEM_TO_KEY = {
    "GGG_large_Pressed_onSgMNMSA_300526":   "GGG_long_pressed",
    "GGG_large_onSgMNMSA_300526":           "GGG_long_free",
    "GGG_short_Pressed_onSgMNMSA_300526":  "GGG_short_pressed",
    "GGG_short_onSgMNMSA_300526":          "GGG_short_free",
    "ml20240308b2_Pressed_onSgMNMSA_300526":  "Hem_t02_pressed",
    "ml20240308b2_onSgMNMSA_300526":          "Hem_t02_free",
    "Fe2O3_14023080525_Pressed_onSgMNMSA_300526":  "Hem_t05_pressed",
    "Fe2O3_14023080525_onSgMNMSA_300526":          "Hem_t05_free",
}
BARE_STEM = "SgMNMSA_300526"

# =============================================================================
# SECTION 4 — FILLING FACTORS
# =============================================================================

def _K(k):
    """Complete elliptic integral of the first kind K(k), k = modulus."""
    k = np.clip(np.abs(float(k)), 0.0, 1.0 - 1e-12)
    return ellipk(k ** 2)


def q_cbcpw_substrate(w, g, h):
    """
    Substrate filling factor for Conductor-Backed CPW (CBCPW).
    Uses sinh-based modulus (Simons 2001, Ch. 4).
    Ground plane at depth h below the conductors confines the field.
    """
    a, b = w / 2, w / 2 + g
    k0 = a / b
    K0 = _K(k0);  K0p = _K(np.sqrt(1 - k0 ** 2))
    k1 = np.sinh(np.pi * a / (2 * h)) / np.sinh(np.pi * b / (2 * h))
    k1 = np.clip(k1, 0.0, 1.0 - 1e-12)
    K1 = _K(k1);  K1p = _K(np.sqrt(1 - k1 ** 2))
    return (K1 / K1p) / (2 * K0 / K0p)


def q_superstrate(w, g, h):
    """
    Superstrate filling factor (open above conductors).
    Uses tanh-based modulus — same formula as open CPW substrate.
    """
    a, b = w / 2, w / 2 + g
    k0 = a / b
    K0 = _K(k0);  K0p = _K(np.sqrt(1 - k0 ** 2))
    kh = np.tanh(np.pi * a / (2 * h)) / np.tanh(np.pi * b / (2 * h))
    kh = np.clip(kh, 0.0, 1.0 - 1e-12)
    Kh = _K(kh);  Khp = _K(np.sqrt(1 - kh ** 2))
    return (Kh / Khp) / (2 * K0 / K0p)


# Pre-compute substrate values (same for all measurements)
Q_SUB        = q_cbcpw_substrate(W, G, H_RO)
EPS_EFF_BARE = 1.0 + Q_SUB * (EPS_RO - 1.0)

# Slot fraction: the gap region is where the air gap actually lives
# (sample rests on conductors; gap exists only in the slot between them)
SLOT_FRACTION = 2 * G / (W + 2 * G)

# Effective air gap for pressed condition:
#   d_slot_full     : conservative upper bound (uniform slab)
#   d_slot_weighted : best estimate (weighted to slot region only)
D_SLOT_FULL     = T_CU
D_SLOT_WEIGHTED = T_CU * SLOT_FRACTION


# =============================================================================
# SECTION 5 — CORE EXTRACTION ENGINE
# =============================================================================

def load_network(data_dir: Path, stem: str) -> rf.Network | None:
    """Load a .s2p file by stem. Returns None if file not found."""
    path = data_dir / (stem + ".s2p")
    if not path.exists():
        print(f"  ⚠  File not found: {path}")
        return None
    return rf.Network(str(path))


def extract_eps_app(nw_bare: rf.Network,
                    nw_load: rf.Network,
                    L_mm: float,
                    h_mm: float) -> dict:
    """
    Extract apparent permittivity ε_app of the superstrate stack
    from the differential phase between loaded and bare S21.

    The differential phase Δφ = φ_loaded − φ_bare cancels:
      ✓ connector transitions
      ✓ all CPW sections outside the sample region
      ✓ systematic VNA errors
    Leaving only the phase shift from the L_mm loaded section.

    Returns dict with freq, f_GHz, eps_app, eps_eff_load,
    tan_delta, delta_phi, r_squared, q_sup.
    """
    L_m = L_mm * 1e-3

    # Interpolate bare to loaded frequency grid if needed
    if not np.allclose(nw_bare.f, nw_load.f):
        nw_bare = nw_bare.interpolate(nw_load.frequency)

    freq  = nw_load.f
    f_GHz = freq * 1e-9
    S21_b = nw_bare.s[:, 1, 0]
    S21_l = nw_load.s[:, 1, 0]

    # Differential phase — unwrap to remove ±π jumps
    delta_phi = np.unwrap(np.angle(S21_l) - np.angle(S21_b))

    # Effective permittivity of loaded section
    # Δφ = −(2πf/c)·L·(√ε_eff_load − √ε_eff_bare)
    # → √ε_eff_load = √ε_eff_bare − Δφ·c/(2πfL)
    sqrt_load    = np.sqrt(EPS_EFF_BARE) - delta_phi * C_LIGHT / (2 * np.pi * freq * L_m)
    eps_eff_load = sqrt_load ** 2

    # Superstrate filling factor for this sample thickness
    q_sup = q_superstrate(W, G, h_mm)

    # Apparent permittivity of the superstrate stack
    # (includes any air gap between Cu top and sample bottom)
    eps_app = 1.0 + (eps_eff_load - EPS_EFF_BARE) / q_sup

    # Linearity of Δφ(f) — diagnostic: should be ~1.0 for non-dispersive sample
    mask = (freq >= F_LOW * 1e9) & (freq <= F_HIGH * 1e9)
    if mask.sum() > 2:
        p       = np.polyfit(freq[mask], delta_phi[mask], 1)
        phi_fit = np.polyval(p, freq[mask])
        ss_res  = np.sum((delta_phi[mask] - phi_fit) ** 2)
        ss_tot  = np.sum((delta_phi[mask] - delta_phi[mask].mean()) ** 2)
        r2      = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        residual_deg = np.std(delta_phi[mask] - phi_fit) * 180 / np.pi
    else:
        r2 = np.nan;  residual_deg = np.nan

    # Loss tangent from differential attenuation
    # Δα = α_loaded − α_bare carries dielectric loss of sample
    alpha_b   = -np.log(np.clip(np.abs(S21_b), 1e-12, None)) / L_m
    alpha_l   = -np.log(np.clip(np.abs(S21_l), 1e-12, None)) / L_m
    delta_alpha = alpha_l - alpha_b
    eps_dbl   = np.abs(delta_alpha * C_LIGHT * np.sqrt(np.abs(eps_eff_load))
                       / (np.pi * freq)) / q_sup
    tan_delta = np.where(eps_app > 0.5, eps_dbl / eps_app, np.nan)

    return {
        "freq":         freq,
        "f_GHz":        f_GHz,
        "eps_app":      eps_app,
        "eps_eff_bare": np.full_like(eps_app, EPS_EFF_BARE),
        "eps_eff_load": eps_eff_load,
        "tan_delta":    tan_delta,
        "delta_phi":    delta_phi,
        "r_squared":    r2,
        "residual_deg": residual_deg,
        "q_sup":        q_sup,
    }


# =============================================================================
# SECTION 6 — GAP CORRECTION MODELS
# =============================================================================

def eps_true_from_app(h_mm: float, d_mm: float, eps_app: float) -> float:
    """
    Invert the series capacitor model to recover the true sample permittivity.

    Model: ε_app = (h + d) / (d/ε_gap + h/ε_true)
    For air gap: ε_gap = 1.0
    → ε_true = h · ε_app / (h − d·(ε_app − 1))

    Returns inf if d is at or past the pole (h/(ε_app−1)).
    """
    denom = h_mm - d_mm * (eps_app - 1.0)
    if denom <= 1e-12:
        return np.inf
    return h_mm * eps_app / denom


def apply_gap_correction(eps_app_arr: np.ndarray,
                         h_mm: float,
                         is_pressed: bool,
                         d_contact_mm: float = 0.0) -> dict:
    """
    Apply gap correction to an array of ε_app values.

    Pressed:
      d_eff = D_SLOT_WEIGHTED  (slot-weighted Cu thickness, best estimate)
      d_eff_upper = D_SLOT_FULL (conservative upper bound)

    Free (not pressed):
      d_eff = D_SLOT_WEIGHTED + d_contact_mm
      d_contact_mm = 0.0 if unknown (returns uncorrected upper/lower bounds)

    Returns dict with eps_true (best), eps_true_upper, eps_true_lower.
    """
    if is_pressed:
        d_best  = D_SLOT_WEIGHTED
        d_upper = D_SLOT_FULL
        d_lower = 0.0                   # absolute lower: no gap at all
    else:
        d_best  = D_SLOT_WEIGHTED + d_contact_mm
        d_upper = D_SLOT_FULL    + d_contact_mm
        d_lower = d_contact_mm          # lower: only contact gap, no slot gap

    def correct(arr, d):
        return np.array([eps_true_from_app(h_mm, d, float(ea)) for ea in arr])

    return {
        "eps_true":       correct(eps_app_arr, d_best),
        "eps_true_upper": correct(eps_app_arr, d_upper),
        "eps_true_lower": correct(eps_app_arr, d_lower),
        "d_best_um":      d_best  * 1000,
        "d_upper_um":     d_upper * 1000,
        "d_lower_um":     d_lower * 1000,
    }


def solve_d_contact(h_mm: float,
                    eps_app_pressed: float,
                    eps_app_free: float) -> float:
    """
    Solve for d_contact (unknown contact gap when not pressed) from
    the pressed and free apparent permittivities of the SAME sample.

    Pressed:  ε_app_p = f(h, D_SLOT_WEIGHTED,              ε_true)
    Free:     ε_app_f = f(h, D_SLOT_WEIGHTED + d_contact,  ε_true)

    From pressed: ε_true = h·ε_app_p / (h − D_SLOT_WEIGHTED·(ε_app_p−1))
    Substitute into free equation and solve for d_contact.

    Returns d_contact in mm, or nan if system is inconsistent.
    """
    d0 = D_SLOT_WEIGHTED
    denom_p = h_mm - d0 * (eps_app_pressed - 1.0)
    if denom_p <= 1e-12:
        return np.nan
    eps_true = h_mm * eps_app_pressed / denom_p

    # From free equation: ε_app_f = (h+d_tot)/(d_tot + h/eps_true)
    # where d_tot = D_SLOT_WEIGHTED + d_contact
    # → d_tot·(ε_app_f − 1) = h·(1 − ε_app_f/eps_true)
    # → d_tot = h·(1 − ε_app_f/eps_true) / (ε_app_f − 1)
    if abs(eps_app_free - 1.0) < 1e-9:
        return np.nan
    d_tot = h_mm * (1.0 - eps_app_free / eps_true) / (eps_app_free - 1.0)
    d_contact = d_tot - d0
    return d_contact  # mm


# =============================================================================
# SECTION 7 — CROSS-VALIDATION CHECKS
# =============================================================================

def run_cross_validation(results: dict) -> list[str]:
    """
    Run all cross-validation checks on the extracted results.
    Returns a list of report lines.

    Checks:
      A. GGG length ratio: Δφ(5.7mm) / Δφ(3.725mm) should equal 5.7/3.725
      B. GGG pipeline: pressed GGG with known ε=12 should recover ε≈12
      C. Hematite thickness: 0.2mm and 0.5mm pressed should give same ε
      D. d_contact extraction: pressed vs free for each sample
    """
    report = []
    mask_key = "mask"   # boolean mask for F_LOW–F_HIGH band

    def mean_in_band(arr, freq):
        m = (freq >= F_LOW*1e9) & (freq <= F_HIGH*1e9)
        return np.nanmean(arr[m])

    report.append("=" * 62)
    report.append("  CROSS-VALIDATION REPORT")
    report.append("=" * 62)

    # ── Check A: GGG length ratio ──────────────────────────────────────
    report.append("\n  A. GGG length ratio check (verifies L values)")
    report.append(     "  ─" * 31)
    ggg_long_p  = results.get("GGG_long_pressed")
    ggg_short_p = results.get("GGG_short_pressed")

    if ggg_long_p and ggg_short_p:
        freq = ggg_long_p["freq"]
        # Interpolate short to long frequency grid
        nw_interp = np.interp(freq,
                              ggg_short_p["freq"],
                              ggg_short_p["delta_phi"])
        mask = (freq >= F_LOW*1e9) & (freq <= F_HIGH*1e9)
        ratio_meas = np.nanmean(
            np.abs(ggg_long_p["delta_phi"][mask]) /
            np.abs(nw_interp[mask]))
        ratio_expected = 5.7 / 3.725
        error_pct = (ratio_meas - ratio_expected) / ratio_expected * 100
        report.append(f"  Expected Δφ ratio (5.7/3.725) = {ratio_expected:.4f}")
        report.append(f"  Measured Δφ ratio             = {ratio_meas:.4f}")
        report.append(f"  Discrepancy                   = {error_pct:+.2f}%")
        if abs(error_pct) < 3:
            report.append("  → ✓ L values consistent")
        else:
            # Estimate corrected L from ratio
            L_long_corr = ratio_meas * 3.725
            report.append(f"  → ⚠ Discrepancy > 3%")
            report.append(f"     Effective L_long ≈ {L_long_corr:.2f} mm "
                          f"(stated 5.7 mm)")
    else:
        report.append("  ⚠ GGG long and/or short pressed results not available")

    # ── Check B: GGG pipeline validation ──────────────────────────────
    report.append("\n  B. GGG pipeline validation (pressed, ε_known = 12.0)")
    report.append(     "  ─" * 31)
    for key, label in [("GGG_long_pressed",  "GGG long  (5.700mm)"),
                        ("GGG_short_pressed", "GGG short (3.725mm)")]:
        res = results.get(key)
        if res:
            freq = res["freq"]
            eps_t = mean_in_band(res["eps_true"], freq)
            eps_a = mean_in_band(res["eps_app"],  freq)
            err   = (eps_t - 12.0) / 12.0 * 100
            flag  = "✓" if abs(err) < 5 else "⚠"
            report.append(f"  {label}: ε_app={eps_a:.3f}  "
                          f"ε_corrected={eps_t:.3f}  "
                          f"error={err:+.1f}%  {flag}")
        else:
            report.append(f"  {label}: not available")

    # ── Check C: Hematite thickness consistency ────────────────────────
    report.append("\n  C. Hematite thickness consistency (pressed)")
    report.append(     "  ─" * 31)
    hem_02 = results.get("Hem_t02_pressed")
    hem_05 = results.get("Hem_t05_pressed")
    if hem_02 and hem_05:
        freq   = hem_02["freq"]
        eps_02 = mean_in_band(hem_02["eps_true"], freq)
        freq5  = hem_05["freq"]
        eps_05 = mean_in_band(hem_05["eps_true"], freq5)
        diff   = eps_05 - eps_02
        report.append(f"  h=0.2mm pressed: ε = {eps_02:.3f}")
        report.append(f"  h=0.5mm pressed: ε = {eps_05:.3f}")
        report.append(f"  Difference      : {diff:+.3f}")
        if abs(diff) < 0.5:
            report.append("  → ✓ Thickness-consistent result")
        else:
            report.append("  → ⚠ Thickness inconsistency — check h_mm or gap model")
    else:
        report.append("  ⚠ One or both hematite pressed results not available")

    # ── Check D: d_contact extraction ─────────────────────────────────
    report.append("\n  D. Contact gap d_contact (pressed vs free)")
    report.append(     "  ─" * 31)
    pairs = [
        ("GGG_long_pressed",  "GGG_long_free",  "GGG long",    0.5),
        ("GGG_short_pressed", "GGG_short_free", "GGG short",   0.5),
        ("Hem_t02_pressed",   "Hem_t02_free",   "Hematite 0.2mm", 0.2),
        ("Hem_t05_pressed",   "Hem_t05_free",   "Hematite 0.5mm", 0.5),
    ]
    for key_p, key_f, label, h_mm in pairs:
        rp = results.get(key_p)
        rf_ = results.get(key_f)
        if rp and rf_:
            freq = rp["freq"]
            mask = (freq >= F_LOW*1e9) & (freq <= F_HIGH*1e9)
            ea_p = np.nanmean(rp["eps_app"][mask])
            # Interpolate free eps_app to same grid
            ea_f_arr = np.interp(freq, rf_["freq"], rf_["eps_app"])
            ea_f = np.nanmean(ea_f_arr[mask])
            d_c  = solve_d_contact(h_mm, ea_p, ea_f)
            if np.isfinite(d_c) and d_c >= 0:
                report.append(f"  {label:20s}: d_contact = {d_c*1000:.1f} μm")
            elif np.isfinite(d_c) and d_c < 0:
                report.append(f"  {label:20s}: d_contact = {d_c*1000:.1f} μm "
                              f"(negative → free may be pressing more than pressed?)")
            else:
                report.append(f"  {label:20s}: d_contact could not be solved "
                              f"(pole condition)")
        else:
            report.append(f"  {label:20s}: pressed or free result not available")

    report.append("\n" + "=" * 62)
    return report


# =============================================================================
# SECTION 8 — PLOTLY INTERACTIVE HTML PLOTS
# =============================================================================

# Colour palette — one colour per material+condition combination
PALETTE = {
    "GGG_long_pressed":   "#1f77b4",
    "GGG_long_free":      "#aec7e8",
    "GGG_short_pressed":  "#ff7f0e",
    "GGG_short_free":     "#ffbb78",
    "Hem_t02_pressed":    "#2ca02c",
    "Hem_t02_free":       "#98df8a",
    "Hem_t05_pressed":    "#d62728",
    "Hem_t05_free":       "#ff9896",
}

LABELS = {
    "GGG_long_pressed":   "GGG 5.7mm pressed",
    "GGG_long_free":      "GGG 5.7mm free",
    "GGG_short_pressed":  "GGG 3.725mm pressed",
    "GGG_short_free":     "GGG 3.725mm free",
    "Hem_t02_pressed":    "Hematite 0.2mm pressed",
    "Hem_t02_free":       "Hematite 0.2mm free",
    "Hem_t05_pressed":    "Hematite 0.5mm pressed",
    "Hem_t05_free":       "Hematite 0.5mm free",
}


def _line_style(key: str) -> dict:
    """Pressed = solid, free = dashed."""
    dash = "solid" if "pressed" in key else "dash"
    return dict(color=PALETTE.get(key, "#888"), width=2, dash=dash)


def plot_epsilon(results: dict, save_path: Path) -> None:
    """
    Two-panel interactive HTML:
      Top    : ε_app (apparent, as measured) for all samples
      Bottom : ε_true (gap-corrected) for all samples
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=("Apparent ε (superstrate stack, includes air gap)",
                        "Corrected ε (gap model applied — best estimate)"),
    )

    for key, res in results.items():
        if key in ("bare",):
            continue
        color = PALETTE.get(key, "#888")
        dash  = "solid" if "pressed" in key else "dash"
        label = LABELS.get(key, key)
        f     = res["f_GHz"]

        # Top panel: eps_app
        fig.add_trace(go.Scatter(
            x=f, y=res["eps_app"], name=label,
            mode="lines", line=dict(color=color, width=2, dash=dash),
            legendgroup=key,
            hovertemplate=f"<b>{label}</b><br>f=%{{x:.2f}} GHz<br>"
                          f"ε_app=%{{y:.4f}}<extra></extra>",
        ), row=1, col=1)

        # Bottom panel: eps_true (best estimate)
        if "eps_true" in res:
            fig.add_trace(go.Scatter(
                x=f, y=res["eps_true"], name=label,
                mode="lines", line=dict(color=color, width=2, dash=dash),
                legendgroup=key, showlegend=False,
                hovertemplate=f"<b>{label}</b><br>f=%{{x:.2f}} GHz<br>"
                              f"ε_true=%{{y:.4f}}<extra></extra>",
            ), row=2, col=1)

            # Shaded uncertainty band (lower to upper)
            if "eps_true_upper" in res and "eps_true_lower" in res:
                fig.add_trace(go.Scatter(
                    x=np.concatenate([f, f[::-1]]),
                    y=np.concatenate([res["eps_true_upper"],
                                      res["eps_true_lower"][::-1]]),
                    fill="toself",
                    fillcolor=color.replace(")", ",0.10)").replace("rgb", "rgba")
                              if color.startswith("rgb") else color,
                    line=dict(width=0),
                    legendgroup=key, showlegend=False,
                    hoverinfo="skip",
                ), row=2, col=1)

    fig.update_yaxes(title_text="ε_app",  rangemode="tozero", row=1, col=1)
    fig.update_yaxes(title_text="ε_true", rangemode="tozero", row=2, col=1)
    fig.update_xaxes(title_text="Frequency (GHz)", row=2, col=1)
    fig.add_vrect(x0=F_LOW, x1=F_HIGH, fillcolor="grey",
                  opacity=0.06, line_width=0,
                  annotation_text=f"{F_LOW}–{F_HIGH} GHz analysis band",
                  annotation_position="top left")

    fig.update_layout(
        title="α-Fe₂O₃ & GGG — Permittivity vs Frequency",
        hovermode="x unified", template="plotly_white", height=700,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
    )
    fig.write_html(str(save_path), include_plotlyjs="cdn")
    print(f"  Epsilon plot → {save_path}")


def plot_tandelta(results: dict, save_path: Path) -> None:
    """Single-panel interactive HTML: tan δ for all samples."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig = go.Figure()
    for key, res in results.items():
        if key == "bare" or "tan_delta" not in res:
            continue
        color = PALETTE.get(key, "#888")
        dash  = "solid" if "pressed" in key else "dash"
        label = LABELS.get(key, key)
        f     = res["f_GHz"]
        fig.add_trace(go.Scatter(
            x=f, y=res["tan_delta"], name=label,
            mode="lines", line=dict(color=color, width=2, dash=dash),
            hovertemplate=f"<b>{label}</b><br>f=%{{x:.2f}} GHz<br>"
                          f"tan δ=%{{y:.5f}}<extra></extra>",
        ))

    fig.add_vrect(x0=F_LOW, x1=F_HIGH, fillcolor="grey", opacity=0.06,
                  line_width=0)
    fig.update_layout(
        title="α-Fe₂O₃ & GGG — Loss Tangent (tan δ) vs Frequency",
        xaxis_title="Frequency (GHz)", yaxis_title="tan δ",
        yaxis=dict(rangemode="tozero"),
        hovermode="x unified", template="plotly_white", height=480,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
    )
    fig.write_html(str(save_path), include_plotlyjs="cdn")
    print(f"  Tan δ plot  → {save_path}")


def plot_delta_phi(results: dict, save_path: Path) -> None:
    """
    Differential phase Δφ(f) for all samples.
    Linearity = non-dispersive sample. Useful diagnostic.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig = go.Figure()
    for key, res in results.items():
        if key == "bare" or "delta_phi" not in res:
            continue
        color = PALETTE.get(key, "#888")
        dash  = "solid" if "pressed" in key else "dash"
        label = LABELS.get(key, key)
        r2    = res.get("r_squared", np.nan)
        r2_str = f", R²={r2:.4f}" if np.isfinite(r2) else ""
        fig.add_trace(go.Scatter(
            x=res["f_GHz"],
            y=np.degrees(res["delta_phi"]),
            name=f"{label}{r2_str}",
            mode="lines",
            line=dict(color=color, width=2, dash=dash),
            hovertemplate=f"<b>{label}</b><br>f=%{{x:.2f}} GHz<br>"
                          f"Δφ=%{{y:.2f}}°<extra></extra>",
        ))

    fig.add_vrect(x0=F_LOW, x1=F_HIGH, fillcolor="grey", opacity=0.06,
                  line_width=0)
    fig.update_layout(
        title="Differential Phase Δφ (loaded − bare) — Linearity Check",
        xaxis_title="Frequency (GHz)", yaxis_title="Δφ (°)",
        hovermode="x unified", template="plotly_white", height=480,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
    )
    fig.write_html(str(save_path), include_plotlyjs="cdn")
    print(f"  Δφ plot     → {save_path}")


# =============================================================================
# SECTION 9 — CSV EXPORT
# =============================================================================

def export_csv(results: dict, save_path: Path) -> None:
    """
    Export one CSV with all results.
    Columns: key, freq_Hz, freq_GHz, eps_app, eps_true,
             eps_true_lower, eps_true_upper, tan_delta, delta_phi_deg
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with save_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "sample_key", "material", "h_mm", "L_mm", "pressed",
            "freq_Hz", "freq_GHz",
            "eps_app", "eps_true", "eps_true_lower", "eps_true_upper",
            "tan_delta", "delta_phi_deg",
        ])
        for key, res in results.items():
            if key == "bare":
                continue
            props = res.get("props", {})
            mat   = props.get("material", "")
            h_mm  = props.get("h_mm", "")
            L_mm  = props.get("L_mm", "")
            press = props.get("is_pressed", "")
            freq  = res["freq"]
            f_GHz = res["f_GHz"]
            ea    = res["eps_app"]
            et    = res.get("eps_true",       np.full_like(ea, np.nan))
            etl   = res.get("eps_true_lower", np.full_like(ea, np.nan))
            etu   = res.get("eps_true_upper", np.full_like(ea, np.nan))
            td    = res.get("tan_delta",      np.full_like(ea, np.nan))
            dp    = np.degrees(res.get("delta_phi", np.zeros_like(ea)))

            for i in range(len(freq)):
                def _fmt(v):
                    return f"{v:.6f}" if np.isfinite(v) else "NaN"
                writer.writerow([
                    key, mat, h_mm, L_mm, press,
                    f"{freq[i]:.6e}", f"{f_GHz[i]:.6f}",
                    _fmt(ea[i]), _fmt(et[i]), _fmt(etl[i]), _fmt(etu[i]),
                    _fmt(td[i]), f"{dp[i]:.4f}",
                ])
    print(f"  CSV         → {save_path}")


# =============================================================================
# SECTION 10 — ENTRY POINT
# =============================================================================


def main() -> None:

    print("=" * 62)
    print("  CPW Permittivity Extraction — Multi-Sample")
    print("=" * 62)
    print(f"""
  CPW geometry (CBCPW):
    w      = {W/MIL:.1f} mil = {W:.4f} mm
    g      = {G/MIL:.1f} mil = {G:.4f} mm
    t_Cu   = {T_CU/MIL:.1f} mil = {T_CU*1000:.1f} μm
    h_RO   = {H_RO/MIL:.1f} mil = {H_RO:.4f} mm
    ε_RO   = {EPS_RO}

  Filling factors:
    q_sub  (CBCPW) = {Q_SUB:.4f}
    ε_eff_bare     = {EPS_EFF_BARE:.4f}

  Gap model (slot region only):
    Slot fraction  = {SLOT_FRACTION:.3f}  (2g / (w+2g))
    d_slot_weighted= {D_SLOT_WEIGHTED*1000:.2f} μm  (best estimate)
    d_slot_full    = {D_SLOT_FULL*1000:.1f} μm  (upper bound)

  Analysis band   : {F_LOW}–{F_HIGH} GHz
""")

    # ── Load bare CPW ──────────────────────────────────────────────────
    print(f"  Loading bare CPW: {BARE_STEM}.s2p")
    nw_bare = load_network(DATA_DIR, BARE_STEM)
    if nw_bare is None:
        print("  ✗ Bare CPW file missing — cannot proceed.")
        return

    # ── Process each sample ────────────────────────────────────────────
    results = {}

    for stem, key in STEM_TO_KEY.items():
        props = REGISTRY.get(stem)
        if props is None:
            print(f"  ⚠ No registry entry for stem: {stem}")
            continue

        print(f"\n  Processing [{key}]  ← {stem}.s2p")
        nw_load = load_network(DATA_DIR, stem)
        if nw_load is None:
            continue

        # Core extraction
        res = extract_eps_app(
            nw_bare  = nw_bare,
            nw_load  = nw_load,
            L_mm     = props["L_mm"],
            h_mm     = props["h_mm"],
        )

        # Gap correction
        gap = apply_gap_correction(
            eps_app_arr = res["eps_app"],
            h_mm        = props["h_mm"],
            is_pressed  = props["is_pressed"],
        )
        res.update(gap)
        res["props"] = props

        # Band statistics
        mask = (res["freq"] >= F_LOW*1e9) & (res["freq"] <= F_HIGH*1e9)
        ea_mean  = np.nanmean(res["eps_app"][mask])
        et_mean  = np.nanmean(res["eps_true"][mask])
        td_mean  = np.nanmean(res["tan_delta"][mask])
        r2       = res["r_squared"]

        print(f"    ε_eff_bare         = {EPS_EFF_BARE:.4f}")
        print(f"    q_super            = {res['q_sup']:.4f}")
        print(f"    ε_app ({F_LOW}–{F_HIGH}GHz) = {ea_mean:.4f}")
        print(f"    ε_true (corrected) = {et_mean:.4f}  "
              f"[{gap['d_lower_um']:.1f}–{gap['d_upper_um']:.1f} μm gap bounds]")
        print(f"    tan δ              = {td_mean:.4f}")
        print(f"    Δφ linearity R²    = {r2:.5f}  "
              f"(residual {res['residual_deg']:.2f}°)")

        # Validation: if ε is known, report error
        if props["eps_known"] is not None:
            err = (et_mean - props["eps_known"]) / props["eps_known"] * 100
            flag = "✓" if abs(err) < 5 else "⚠"
            print(f"    Known ε = {props['eps_known']}  → "
                  f"error = {err:+.1f}%  {flag}")

        results[key] = res

    # ── Cross-validation ───────────────────────────────────────────────
    print()
    report = run_cross_validation(results)
    for line in report:
        print(line)

    # ── Plots ──────────────────────────────────────────────────────────
    print()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_epsilon(  results, OUTPUT_DIR / "epsilon_all_samples.html")
    plot_tandelta( results, OUTPUT_DIR / "tandelta_all_samples.html")
    plot_delta_phi(results, OUTPUT_DIR / "delta_phi_all_samples.html")

    # ── CSV ────────────────────────────────────────────────────────────
    export_csv(results, OUTPUT_DIR / "permittivity_results.csv")

    print("\n  Done.")


if __name__ == "__main__":
    main()
