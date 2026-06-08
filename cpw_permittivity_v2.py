"""
CPW Superstrate Permittivity Extraction — Unified Multi-Sample Code
====================================================================
GGG reference measurements (known ε = 12) are used as a mandatory
calibration step that determines the effective electrical length L_eff
for each sample placement. This corrects for the systematic discrepancy
between stated physical length and the electrically active length
(caused by connector launches, bond pads, or partial sample coverage).

All non-reference samples are then extracted using:
  1. Stated L (uncorrected) — as-measured, no GGG calibration
  2. Calibrated L_eff       — after GGG length correction
  3. Gap-corrected ε        — applying the slot air gap model

Both uncorrected and corrected results are reported side-by-side so the
impact of each correction step is transparent.

Pipeline:
  Step 1 — Load bare CPW reference
  Step 2 — Load GGG pressed measurements → solve L_eff for each GGG piece
  Step 3 — Compute scale factor = mean(L_eff / L_stated) across GGG pieces
  Step 4 — Apply scale factor to all non-reference samples
  Step 5 — Extract ε for all samples with stated L AND calibrated L_eff
  Step 6 — Apply gap correction (pressed: d = d_slot; free: d = d_slot + d_contact)
  Step 7 — Cross-validation report
  Step 8 — Plotly HTML plots (overlay stated vs calibrated)
  Step 9 — CSV export

Structure:
  Section 1  — Paths
  Section 2  — CPW geometry
  Section 3  — Measurement registry
  Section 4  — Filling factors
  Section 5  — Core extraction engine
  Section 6  — GGG calibration (L_eff solver)
  Section 7  — Gap correction
  Section 8  — Cross-validation report
  Section 9  — Plotly plots
  Section 10 — CSV export
  Section 11 — Entry point

Dependencies: pip install scikit-rf plotly scipy numpy
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

DATA_DIR   = Path(".")    # ← folder containing all .s2p files
OUTPUT_DIR = Path(".")    # ← folder for HTML plots and CSV

# =============================================================================
# SECTION 2 — CPW GEOMETRY
# =============================================================================

MIL    = 0.0254       # 1 mil in mm
W      = 16.5 * MIL   # signal strip width        [mm]
G      = 5.0  * MIL   # slot (gap) width           [mm]
T_CU   = 0.7  * MIL   # copper thickness = base gap in slot  [mm]
H_RO   = 8.0  * MIL   # RO4003 substrate thickness [mm]
EPS_RO = 3.55          # RO4003 dielectric constant

# Structure: Conductor-Backed CPW (CBCPW)
# Back copper ground plane → FR4 backer is fully shielded and irrelevant.

F_LOW   = 10.0   # analysis band start [GHz]
F_HIGH  = 40.0   # analysis band end   [GHz]
C_LIGHT = 3e8    # speed of light      [m/s]

# =============================================================================
# SECTION 3 — MEASUREMENT REGISTRY
# =============================================================================
# One dict entry per s2p file (keyed by filename stem, no .s2p extension).
#
# Fields:
#   role       : "reference" | "sample" | "bare"
#                "reference" = known-ε material used for L_eff calibration (GGG)
#                "sample"    = unknown material to be characterised (Hematite)
#                "bare"      = empty CPW, used as phase reference for all
#   material   : human label, e.g. "GGG", "Hematite", "bare"
#   h_mm       : sample thickness [mm]          (None for bare)
#   L_mm       : stated physical sample length   [mm]  (None for bare)
#   eps_known  : known dielectric constant (reference only; None for samples)
#   is_pressed : True  → d_contact = 0 (sample rests on Cu conductors)
#                False → d_contact unknown (sample placed without pressure)
#
# HOW TO ADD NEW FILES: add one dict with the filename stem as key.
# HOW TO ADD NEW REFERENCE MATERIAL: set role="reference", eps_known=<value>.
# HOW TO ADD NEW SAMPLE: set role="sample", eps_known=None.
# The bare CPW stem is declared separately as BARE_STEM below.

BARE_STEM = "SgMNMSA_010626"     # ← update date to match your session

REGISTRY = {

    # ── GGG large (5.7 mm, h=0.5 mm) — reference ─────────────────────
    "GGG_large_Pressed_onSgMNMSA_010626": {
        "role":       "reference",
        "material":   "GGG",
        "h_mm":       0.5,
        "L_mm":       5.700,
        "eps_known":  12.0,
        "is_pressed": True,
    },
    "GGG_large_onSgMNMSA_010626": {
        "role":       "reference",
        "material":   "GGG",
        "h_mm":       0.5,
        "L_mm":       5.700,
        "eps_known":  12.0,
        "is_pressed": False,
    },

    # ── GGG short (3.725 mm, h=0.5 mm) — reference ───────────────────
    "GGG_short_Pressed_onSgMNMSA_010626": {
        "role":       "reference",
        "material":   "GGG",
        "h_mm":       0.5,
        "L_mm":       3.725,
        "eps_known":  12.0,
        "is_pressed": True,
    },
    "GGG_short_onSgMNMSA_010626": {
        "role":       "reference",
        "material":   "GGG",
        "h_mm":       0.5,
        "L_mm":       3.725,
        "eps_known":  12.0,
        "is_pressed": False,
    },

    # ── Hematite SN1 (h=0.2 mm, L=5.0 mm) — sample ──────────────────
    # Replace 14023080525 with the actual serial number in your filenames
    "Fe2O3_14023080525_Pressed_onSgMNMSA_010626": {
        "role":       "sample",
        "material":   "Hematite",
        "h_mm":       0.2,
        "L_mm":       5.000,
        "eps_known":  None,
        "is_pressed": True,
    },
    "Fe2O3_14023080525_onSgMNMSA_010626": {
        "role":       "sample",
        "material":   "Hematite",
        "h_mm":       0.2,
        "L_mm":       5.000,
        "eps_known":  None,
        "is_pressed": False,
    },

    # ── Hematite SN2 (h=0.5 mm, L=5.0 mm) — sample ──────────────────
    # Replace ml20240308b2 with the actual serial number in your filenames
    "ml20240308b2_Pressed_onSgMNMSA_010626": {
        "role":       "sample",
        "material":   "Hematite",
        "h_mm":       0.5,
        "L_mm":       5.000,
        "eps_known":  None,
        "is_pressed": True,
    },
    "ml20240308b2_onSgMNMSA_010626": {
        "role":       "sample",
        "material":   "Hematite",
        "h_mm":       0.5,
        "L_mm":       5.000,
        "eps_known":  None,
        "is_pressed": False,
    },
}

# =============================================================================
# SECTION 4 — FILLING FACTORS
# =============================================================================

def _K(k: float) -> float:
    """Complete elliptic integral of the first kind K(k), k = modulus."""
    k = np.clip(np.abs(float(k)), 0.0, 1.0 - 1e-12)
    return float(ellipk(k ** 2))


def q_cbcpw_substrate(w: float, g: float, h: float) -> float:
    """
    Substrate filling factor for Conductor-Backed CPW (CBCPW).
    sinh-based modulus (Simons 2001, Ch. 4, eq. 4.5–4.9).
    The ground plane at depth h terminates the substrate field,
    making FR4 below the ground plane completely irrelevant.
    """
    a, b = w / 2.0, w / 2.0 + g
    k0  = a / b
    K0  = _K(k0);   K0p = _K(np.sqrt(1.0 - k0 ** 2))
    k1  = np.sinh(np.pi * a / (2.0 * h)) / np.sinh(np.pi * b / (2.0 * h))
    k1  = np.clip(k1, 0.0, 1.0 - 1e-12)
    K1  = _K(k1);   K1p = _K(np.sqrt(1.0 - k1 ** 2))
    return (K1 / K1p) / (2.0 * K0 / K0p)


def q_superstrate(w: float, g: float, h: float) -> float:
    """
    Superstrate filling factor (open half-space above conductors).
    tanh-based modulus — identical formula to open CPW substrate.
    """
    a, b = w / 2.0, w / 2.0 + g
    k0  = a / b
    K0  = _K(k0);   K0p = _K(np.sqrt(1.0 - k0 ** 2))
    kh  = np.tanh(np.pi * a / (2.0 * h)) / np.tanh(np.pi * b / (2.0 * h))
    kh  = np.clip(kh, 0.0, 1.0 - 1e-12)
    Kh  = _K(kh);   Khp = _K(np.sqrt(1.0 - kh ** 2))
    return (Kh / Khp) / (2.0 * K0 / K0p)


# Pre-computed substrate constants (same for all measurements)
Q_SUB         = q_cbcpw_substrate(W, G, H_RO)
EPS_EFF_BARE  = 1.0 + Q_SUB * (EPS_RO - 1.0)

# Gap geometry
# Pressed sample rests on signal strip AND ground conductors.
# Air gap exists ONLY in the slot regions between conductors.
SLOT_FRACTION  = 2.0 * G / (W + 2.0 * G)   # fraction of active width that is slot
D_SLOT_BEST    = T_CU * SLOT_FRACTION        # weighted gap [mm] — best estimate
D_SLOT_UPPER   = T_CU                        # full Cu thickness [mm] — upper bound


# =============================================================================
# SECTION 5 — CORE EXTRACTION ENGINE
# =============================================================================

def load_network(data_dir: Path, stem: str) -> rf.Network | None:
    """Load a .s2p file by stem. Returns None with a warning if not found."""
    path = data_dir / (stem + ".s2p")
    if not path.exists():
        print(f"  ⚠  Not found: {path.name}")
        return None
    return rf.Network(str(path))


def extract_eps_app(nw_bare: rf.Network,
                    nw_load: rf.Network,
                    L_mm: float,
                    h_mm: float) -> dict:
    """
    Extract apparent permittivity ε_app of the superstrate stack from the
    differential phase Δφ = φ_loaded − φ_bare.

    The differential phase cancels all contributions outside the loaded
    section (connector transitions, unloaded CPW, systematic VNA errors),
    leaving only the phase shift from the L_mm covered region.

    Physical model:
      Δφ = −(2πf/c)·L·[√ε_eff_load − √ε_eff_bare]
      ε_app = 1 + (ε_eff_load − ε_eff_bare) / q_sup

    Returns dict with freq, f_GHz, eps_app, eps_eff_load, tan_delta,
                       delta_phi, r_squared, residual_deg, q_sup.
    """
    L_m = L_mm * 1e-3

    # Align frequency grids
    if not np.allclose(nw_bare.f, nw_load.f):
        nw_bare = nw_bare.interpolate(nw_load.frequency)

    freq  = nw_load.f
    f_GHz = freq * 1e-9
    S21_b = nw_bare.s[:, 1, 0]
    S21_l = nw_load.s[:, 1, 0]

    # Differential phase — unwrap to remove ±π discontinuities
    delta_phi = np.unwrap(np.angle(S21_l) - np.angle(S21_b))

    # Effective permittivity of the loaded section
    sqrt_load    = np.sqrt(EPS_EFF_BARE) - delta_phi * C_LIGHT / (2.0 * np.pi * freq * L_m)
    eps_eff_load = sqrt_load ** 2

    # Superstrate filling factor for this sample thickness
    q_sup = q_superstrate(W, G, h_mm)

    # Apparent permittivity of the superstrate stack (includes any air gap)
    eps_app = 1.0 + (eps_eff_load - EPS_EFF_BARE) / q_sup

    # Linearity of Δφ(f) — should be R²≈1 for non-dispersive sample
    mask = (freq >= F_LOW * 1e9) & (freq <= F_HIGH * 1e9)
    if mask.sum() > 2:
        p         = np.polyfit(freq[mask], delta_phi[mask], 1)
        phi_fit   = np.polyval(p, freq[mask])
        ss_res    = np.sum((delta_phi[mask] - phi_fit) ** 2)
        ss_tot    = np.sum((delta_phi[mask] - delta_phi[mask].mean()) ** 2)
        r2        = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        resid_deg = float(np.std(delta_phi[mask] - phi_fit) * 180.0 / np.pi)
    else:
        r2 = np.nan;  resid_deg = np.nan

    # Loss tangent from differential attenuation
    alpha_b     = -np.log(np.clip(np.abs(S21_b), 1e-12, None)) / L_m
    alpha_l     = -np.log(np.clip(np.abs(S21_l), 1e-12, None)) / L_m
    delta_alpha = alpha_l - alpha_b
    eps_dbl     = (np.abs(delta_alpha) * C_LIGHT
                   * np.sqrt(np.abs(eps_eff_load))
                   / (np.pi * freq)) / q_sup
    tan_delta   = np.where(eps_app > 0.5, eps_dbl / eps_app, np.nan)

    return {
        "freq":         freq,
        "f_GHz":        f_GHz,
        "eps_app":      eps_app,
        "eps_eff_load": eps_eff_load,
        "tan_delta":    tan_delta,
        "delta_phi":    delta_phi,
        "r_squared":    r2,
        "residual_deg": resid_deg,
        "q_sup":        q_sup,
        "L_mm_used":    L_mm,
    }


def band_mean(arr: np.ndarray, freq: np.ndarray) -> float:
    """Mean of arr over the analysis band F_LOW–F_HIGH."""
    m = (freq >= F_LOW * 1e9) & (freq <= F_HIGH * 1e9)
    return float(np.nanmean(arr[m]))


# =============================================================================
# SECTION 6 — GGG CALIBRATION  (L_eff solver)
# =============================================================================

def solve_L_eff(nw_bare: rf.Network,
                nw_ref: rf.Network,
                h_mm: float,
                eps_known: float) -> float:
    """
    Back-solve the effective electrical length L_eff from a pressed
    reference measurement with known dielectric constant.

    Method:
      The true ε_eff_load is known from geometry:
        ε_eff_load_true = EPS_EFF_BARE + q_sup·(ε_known − 1)

      From the measured differential phase:
        Δφ = −(2πf/c)·L_eff·[√ε_eff_load_true − √ε_eff_bare]
        → L_eff = −Δφ·c / [2πf·(√ε_eff_load_true − √ε_eff_bare)]

      Return the mean L_eff over the analysis band [m], converted to mm.

    Physical meaning:
      L_eff < L_stated means the sample does not fully load the CPW
      over its entire physical length. The ratio L_eff/L_stated is the
      coverage factor, capturing any combination of:
        - connector launches / bond pads at the CPW ends
        - partial sample overlap with the active CPW region
        - fringe effects at sample edges
    """
    if not np.allclose(nw_bare.f, nw_ref.f):
        nw_bare = nw_bare.interpolate(nw_ref.frequency)

    freq  = nw_ref.f
    S21_b = nw_bare.s[:, 1, 0]
    S21_r = nw_ref.s[:, 1, 0]

    delta_phi = np.unwrap(np.angle(S21_r) - np.angle(S21_b))

    q_sup               = q_superstrate(W, G, h_mm)
    eps_eff_load_true   = EPS_EFF_BARE + q_sup * (eps_known - 1.0)
    delta_sqrt          = np.sqrt(eps_eff_load_true) - np.sqrt(EPS_EFF_BARE)

    # L_eff at each frequency point
    L_eff_arr = -delta_phi * C_LIGHT / (2.0 * np.pi * freq * delta_sqrt)

    mask = (freq >= F_LOW * 1e9) & (freq <= F_HIGH * 1e9)
    return float(np.nanmean(L_eff_arr[mask])) * 1e3  # → mm


def calibrate_from_ggg(nw_bare: rf.Network, registry: dict,
                       data_dir: Path) -> dict:
    """
    Load all reference (GGG pressed) measurements, solve L_eff for each,
    and compute the scale factor = L_eff / L_stated.

    Only PRESSED reference measurements are used for calibration because:
      - pressed → d_contact = 0 → gap correction is known and small
      - free GGG measurements have an additional unknown contact gap

    Returns calibration dict with:
      "scale_factor"    : mean L_eff/L_stated across all pressed references
      "scale_std"       : standard deviation (consistency indicator)
      "per_reference"   : list of per-measurement details for reporting
      "L_eff_by_stem"   : {stem: L_eff_mm} for every pressed reference
    """
    per_ref   = []
    scales    = []
    L_eff_map = {}

    for stem, props in registry.items():
        if props["role"] != "reference" or not props["is_pressed"]:
            continue

        nw_ref = load_network(data_dir, stem)
        if nw_ref is None:
            continue

        L_eff = solve_L_eff(nw_bare, nw_ref,
                             h_mm      = props["h_mm"],
                             eps_known = props["eps_known"])
        scale = L_eff / props["L_mm"]
        scales.append(scale)
        L_eff_map[stem] = L_eff
        per_ref.append({
            "stem":      stem,
            "material":  props["material"],
            "L_stated":  props["L_mm"],
            "L_eff":     L_eff,
            "scale":     scale,
        })

    scale_factor = float(np.mean(scales)) if scales else 1.0
    scale_std    = float(np.std(scales))  if scales else 0.0

    return {
        "scale_factor":  scale_factor,
        "scale_std":     scale_std,
        "per_reference": per_ref,
        "L_eff_by_stem": L_eff_map,
    }


# =============================================================================
# SECTION 7 — GAP CORRECTION
# =============================================================================

def eps_true_from_app(h_mm: float, d_mm: float, eps_app: float) -> float:
    """
    Invert the series capacitor air-gap model to recover ε_true.

    Model (air gap in slot region, sample pressed on conductors):
      ε_app = (h + d) / (d/1 + h/ε_true)
      → ε_true = h·ε_app / (h − d·(ε_app − 1))

    Returns inf when d is at or past the pole = h/(ε_app−1).
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
    Apply gap correction to a full frequency sweep of ε_app values.

    Pressed (is_pressed=True):
      d_best  = D_SLOT_BEST    (slot-weighted Cu thickness)
      d_upper = D_SLOT_UPPER   (full Cu thickness — conservative)
      d_lower = 0.0            (no gap at all — absolute lower bound)

    Free (is_pressed=False):
      d_best  = D_SLOT_BEST + d_contact_mm
      d_upper = D_SLOT_UPPER + d_contact_mm
      d_lower = d_contact_mm  (contact gap only, no slot contribution)

    Returns dict with eps_true (best), eps_true_upper, eps_true_lower,
    and the gap values used [μm].
    """
    d_best, d_upper, d_lower = (
        (D_SLOT_BEST,             D_SLOT_UPPER,             0.0)
        if is_pressed else
        (D_SLOT_BEST + d_contact_mm,
         D_SLOT_UPPER + d_contact_mm,
         d_contact_mm)
    )

    def _correct(arr, d):
        return np.array([eps_true_from_app(h_mm, d, float(v)) for v in arr])

    return {
        "eps_true":       _correct(eps_app_arr, d_best),
        "eps_true_upper": _correct(eps_app_arr, d_upper),
        "eps_true_lower": _correct(eps_app_arr, d_lower),
        "d_best_um":      d_best  * 1000.0,
        "d_upper_um":     d_upper * 1000.0,
        "d_lower_um":     d_lower * 1000.0,
    }


def solve_d_contact(h_mm: float,
                    eps_app_pressed: float,
                    eps_app_free: float) -> float:
    """
    Solve for d_contact from pressed and free measurements of the SAME sample.

    Pressed: ε_app_p → ε_true via d = D_SLOT_BEST
    Free:    ε_app_f → ε_true via d = D_SLOT_BEST + d_contact
    Same ε_true → solve for d_contact.

    Returns d_contact [mm], or nan if system is inconsistent.
    """
    d0    = D_SLOT_BEST
    denom = h_mm - d0 * (eps_app_pressed - 1.0)
    if denom <= 1e-12:
        return np.nan
    eps_t = h_mm * eps_app_pressed / denom

    if abs(eps_app_free - 1.0) < 1e-9:
        return np.nan
    d_tot = h_mm * (1.0 - eps_app_free / eps_t) / (eps_app_free - 1.0)
    return d_tot - d0  # mm


# =============================================================================
# SECTION 8 — CROSS-VALIDATION REPORT
# =============================================================================

def build_report(results: dict, calibration: dict) -> list[str]:
    """
    Build a human-readable cross-validation report covering:
      A. GGG calibration summary (scale factor, consistency)
      B. GGG recovery check: corrected ε should match eps_known=12
      C. Hematite thickness consistency (0.2mm vs 0.5mm, same condition)
      D. Contact gap d_contact per sample (pressed vs free pairs)
      E. Final hematite results — uncorrected, L-corrected, gap-corrected
    """
    R = []

    def sep(title=""):
        line = "─" * 62
        R.append(line if not title else f"  {title}")

    def h1(title):
        R.append("=" * 62)
        R.append(f"  {title}")
        R.append("=" * 62)

    def freq_mean(key, field):
        res = results.get(key)
        if res is None or field not in res:
            return np.nan
        return band_mean(res[field], res["freq"])

    # ── A. GGG calibration ─────────────────────────────────────────────
    h1("A. GGG CALIBRATION — Effective Length Correction")
    R.append(f"""
  Physical motivation:
  The stated sample length L is the physical dimension of the piece.
  The electrically active length L_eff is shorter because connector
  launches at the CPW ends and/or partial sample overlap reduce the
  fraction of the sample that uniformly loads the CPW field.

  Method: for each pressed GGG measurement (ε_GGG = 12.0 known),
  back-solve L_eff from the measured differential phase Δφ(f):
    L_eff = −Δφ·c / [2πf·(√ε_eff_load_true − √ε_eff_bare)]
  averaged over {F_LOW:.0f}–{F_HIGH:.0f} GHz.
""")

    R.append(f"  {'Reference stem':42} | {'L_stated':>9} | "
             f"{'L_eff':>7} | {'Scale':>7}")
    sep()
    for pr in calibration["per_reference"]:
        R.append(f"  {pr['stem']:42} | {pr['L_stated']:>9.3f} | "
                 f"{pr['L_eff']:>7.3f} | {pr['scale']:>7.4f}")

    sf  = calibration["scale_factor"]
    std = calibration["scale_std"]
    R.append("")
    R.append(f"  Scale factor (mean ± std) = {sf:.4f} ± {std:.4f}")
    if std / sf < 0.05:
        R.append("  → ✓ Consistent across GGG pieces (< 5% spread)")
    else:
        R.append("  → ⚠ Inconsistent scale — check sample positioning")
    R.append(f"""
  Interpretation:
  Scale factor ≈ {sf:.2f} means only {sf*100:.0f}% of the stated sample
  length is electrically active. This single factor is applied to
  ALL non-reference sample lengths for the calibrated extraction.
""")

    # ── B. GGG recovery check ──────────────────────────────────────────
    h1("B. GGG PIPELINE VALIDATION — Recover ε = 12.0")
    R.append(f"\n  {'Key':>22} | {'ε_app(stated L)':>16} | "
             f"{'ε_app(cal. L)':>14} | {'ε_true(cal+gap)':>15} | {'Error':>7}")
    sep()
    for key in ["GGG_large_pressed", "GGG_short_pressed",
                "GGG_large_free",   "GGG_short_free"]:
        res = results.get(key)
        if res is None:
            continue
        ea_s = freq_mean(key, "eps_app_stated")
        ea_c = freq_mean(key, "eps_app")
        et   = freq_mean(key, "eps_true")
        err  = (et - 12.0) / 12.0 * 100 if np.isfinite(et) else np.nan
        flag = "✓" if abs(err) < 5 else "⚠"
        R.append(f"  {key:>22} | {ea_s:>16.3f} | {ea_c:>14.3f} | "
                 f"{et:>15.3f} | {err:>+6.1f}% {flag}")

    # ── C. Hematite thickness consistency ─────────────────────────────
    h1("C. HEMATITE THICKNESS CONSISTENCY")
    R.append(f"""
  Same material, different thickness → different q_sup.
  After L-calibration and gap correction, both should give same ε.
  Disagreement indicates a residual gap or q_sup modelling error.
""")
    R.append(f"  {'Key':>24} | {'ε_app(stated)':>14} | "
             f"{'ε_app(cal.)':>12} | {'ε_true(cal+gap)':>15}")
    sep()
    for key in ["Hem_t02_pressed", "Hem_t02_free",
                "Hem_t05_pressed", "Hem_t05_free"]:
        res = results.get(key)
        if res is None:
            continue
        ea_s = freq_mean(key, "eps_app_stated")
        ea_c = freq_mean(key, "eps_app")
        et   = freq_mean(key, "eps_true")
        R.append(f"  {key:>24} | {ea_s:>14.3f} | {ea_c:>12.3f} | {et:>15.3f}")

    p02 = freq_mean("Hem_t02_pressed", "eps_true")
    p05 = freq_mean("Hem_t05_pressed", "eps_true")
    if np.isfinite(p02) and np.isfinite(p05):
        diff = abs(p05 - p02)
        R.append(f"\n  Pressed 0.2mm vs 0.5mm difference: {diff:.3f}")
        R.append("  → ✓ Thickness-consistent" if diff < 1.0
                 else "  → ⚠ Residual thickness inconsistency")

    # ── D. Contact gap d_contact ───────────────────────────────────────
    h1("D. CONTACT GAP d_contact (Pressed vs Free)")
    R.append(f"""
  For each sample pair (same piece, pressed and free), solving the
  series capacitor model gives d_contact: the additional gap that
  appears when the sample is placed without pressure.

  d_total_free = d_slot (Cu thickness) + d_contact
""")
    R.append(f"  {'Sample':>20} | {'h(mm)':>6} | "
             f"{'ε_app pressed':>14} | {'ε_app free':>11} | {'d_contact(μm)':>14}")
    sep()
    pairs = [
        ("GGG_large_pressed",  "GGG_large_free",  0.5, "GGG large"),
        ("GGG_short_pressed",  "GGG_short_free",  0.5, "GGG short"),
        ("Hem_t02_pressed",    "Hem_t02_free",    0.2, "Hem 0.2mm"),
        ("Hem_t05_pressed",    "Hem_t05_free",    0.5, "Hem 0.5mm"),
    ]
    for kp, kf, h, label in pairs:
        ea_p = freq_mean(kp, "eps_app")
        ea_f = freq_mean(kf, "eps_app")
        if np.isnan(ea_p) or np.isnan(ea_f):
            continue
        dc = solve_d_contact(h, ea_p, ea_f)
        dc_str = f"{dc*1000:.1f}" if np.isfinite(dc) and dc >= 0 else "—"
        R.append(f"  {label:>20} | {h:>6.1f} | {ea_p:>14.4f} | "
                 f"{ea_f:>11.4f} | {dc_str:>14}")

    # ── E. Final hematite results ──────────────────────────────────────
    h1("E. HEMATITE RESULTS — Three Levels of Correction")
    R.append(f"""
  Level 1 — Uncorrected (stated L, no gap correction):
    ε = ε_app using L_stated directly. Shows raw measurement.

  Level 2 — L-corrected (calibrated L_eff, no gap correction):
    ε_app using L_eff = L_stated × {sf:.4f}. Shows L correction impact.

  Level 3 — Fully corrected (calibrated L_eff + slot gap correction):
    ε_true using d_slot = {D_SLOT_BEST*1000:.2f} μm (weighted slot gap).
    Upper bound uses d = {D_SLOT_UPPER*1000:.1f} μm (full Cu thickness).
""")
    R.append(f"  {'Sample':>22} | {'L1: ε_app(stated)':>18} | "
             f"{'L2: ε_app(cal.)':>16} | {'L3: ε_true(cal+gap)':>20} | {'tan δ':>7}")
    sep()
    for key in ["Hem_t02_pressed", "Hem_t02_free",
                "Hem_t05_pressed", "Hem_t05_free"]:
        res = results.get(key)
        if res is None:
            continue
        ea_s = freq_mean(key, "eps_app_stated")
        ea_c = freq_mean(key, "eps_app")
        et   = freq_mean(key, "eps_true")
        td   = freq_mean(key, "tan_delta")
        R.append(f"  {key:>22} | {ea_s:>18.3f} | {ea_c:>16.3f} | "
                 f"{et:>20.3f} | {td:>7.4f}")

    R.append("\n" + "=" * 62)
    return R


# =============================================================================
# SECTION 9 — PLOTLY HTML PLOTS
# =============================================================================

PALETTE = {
    "GGG_large_pressed":  "#1f77b4",
    "GGG_large_free":     "#aec7e8",
    "GGG_short_pressed":  "#ff7f0e",
    "GGG_short_free":     "#ffbb78",
    "Hem_t02_pressed":    "#2ca02c",
    "Hem_t02_free":       "#98df8a",
    "Hem_t05_pressed":    "#d62728",
    "Hem_t05_free":       "#ff9896",
}

LABELS = {
    "GGG_large_pressed":  "GGG 5.7mm pressed",
    "GGG_large_free":     "GGG 5.7mm free",
    "GGG_short_pressed":  "GGG 3.725mm pressed",
    "GGG_short_free":     "GGG 3.725mm free",
    "Hem_t02_pressed":    "Hematite 0.2mm pressed",
    "Hem_t02_free":       "Hematite 0.2mm free",
    "Hem_t05_pressed":    "Hematite 0.5mm pressed",
    "Hem_t05_free":       "Hematite 0.5mm free",
}


def _ls(key: str) -> dict:
    """Line style: solid = pressed, dash = free."""
    return dict(color=PALETTE.get(key, "#888"), width=2,
                dash="solid" if "pressed" in key else "dash")


def _vrect(fig) -> None:
    fig.add_vrect(x0=F_LOW, x1=F_HIGH, fillcolor="lightgrey",
                  opacity=0.15, line_width=0)


def plot_epsilon(results: dict, sf: float, save_path: Path) -> None:
    """
    Three-panel plot:
      Top    — ε_app with stated L (uncorrected)
      Middle — ε_app with calibrated L_eff
      Bottom — ε_true (calibrated L + gap correction) with uncertainty band
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        subplot_titles=(
            f"Level 1 — ε_app  (stated L, uncorrected)",
            f"Level 2 — ε_app  (calibrated L_eff,  scale = {sf:.4f})",
            f"Level 3 — ε_true (calibrated L_eff + slot gap correction)",
        ),
    )

    for key, res in results.items():
        if not any(key.startswith(p) for p in
                   ["GGG", "Hem"]):
            continue
        f     = res["f_GHz"]
        color = PALETTE.get(key, "#888")
        dash  = "solid" if "pressed" in key else "dash"
        label = LABELS.get(key, key)
        ls    = dict(color=color, width=2, dash=dash)

        # Panel 1 — stated L
        if "eps_app_stated" in res:
            fig.add_trace(go.Scatter(
                x=f, y=res["eps_app_stated"], name=label,
                mode="lines", line=ls, legendgroup=key,
                hovertemplate=(f"<b>{label} [stated L]</b><br>"
                               f"f=%{{x:.2f}} GHz<br>ε=%{{y:.4f}}<extra></extra>"),
            ), row=1, col=1)

        # Panel 2 — calibrated L
        fig.add_trace(go.Scatter(
            x=f, y=res["eps_app"], name=label,
            mode="lines", line=ls, legendgroup=key, showlegend=False,
            hovertemplate=(f"<b>{label} [cal. L]</b><br>"
                           f"f=%{{x:.2f}} GHz<br>ε=%{{y:.4f}}<extra></extra>"),
        ), row=2, col=1)

        # Panel 3 — gap corrected with uncertainty band
        if "eps_true" in res:
            fig.add_trace(go.Scatter(
                x=f, y=res["eps_true"], name=label,
                mode="lines", line=ls, legendgroup=key, showlegend=False,
                hovertemplate=(f"<b>{label} [corrected]</b><br>"
                               f"f=%{{x:.2f}} GHz<br>ε=%{{y:.4f}}<extra></extra>"),
            ), row=3, col=1)

            # Uncertainty band
            eu = res.get("eps_true_upper", res["eps_true"])
            el = res.get("eps_true_lower", res["eps_true"])
            hex_col  = PALETTE.get(key, "#888888").lstrip("#")
            r, g_c, b = int(hex_col[0:2],16), int(hex_col[2:4],16), int(hex_col[4:6],16)
            fill_col = f"rgba({r},{g_c},{b},0.10)"
            fig.add_trace(go.Scatter(
                x=np.concatenate([f, f[::-1]]),
                y=np.concatenate([eu, el[::-1]]),
                fill="toself", fillcolor=fill_col,
                line=dict(width=0), legendgroup=key, showlegend=False,
                hoverinfo="skip",
            ), row=3, col=1)

    for row in [1, 2, 3]:
        _vrect(fig)
    fig.update_yaxes(title_text="ε_app (stated L)",  rangemode="tozero", row=1, col=1)
    fig.update_yaxes(title_text="ε_app (cal. L)",    rangemode="tozero", row=2, col=1)
    fig.update_yaxes(title_text="ε_true",            rangemode="tozero", row=3, col=1)
    fig.update_xaxes(title_text="Frequency (GHz)", row=3, col=1)
    fig.update_layout(
        title=("α-Fe₂O₃ & GGG — Permittivity vs Frequency "
               "(3-level correction comparison)"),
        hovermode="x unified", template="plotly_white", height=900,
        legend=dict(orientation="h", yanchor="bottom",
                    y=1.01, xanchor="right", x=1),
    )
    fig.write_html(str(save_path), include_plotlyjs="cdn")
    print(f"  Epsilon plot   → {save_path}")


def plot_tandelta(results: dict, save_path: Path) -> None:
    """Loss tangent for all samples (calibrated L)."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig = go.Figure()
    for key, res in results.items():
        if "tan_delta" not in res:
            continue
        fig.add_trace(go.Scatter(
            x=res["f_GHz"], y=res["tan_delta"],
            name=LABELS.get(key, key), mode="lines", line=_ls(key),
            hovertemplate=(f"<b>{LABELS.get(key,key)}</b><br>"
                           f"f=%{{x:.2f}} GHz<br>tan δ=%{{y:.5f}}<extra></extra>"),
        ))
    _vrect(fig)
    fig.update_layout(
        title="Loss Tangent (tan δ) vs Frequency — calibrated L",
        xaxis_title="Frequency (GHz)", yaxis_title="tan δ",
        yaxis=dict(rangemode="tozero"),
        hovermode="x unified", template="plotly_white", height=480,
        legend=dict(orientation="h", yanchor="bottom",
                    y=1.02, xanchor="right", x=1),
    )
    fig.write_html(str(save_path), include_plotlyjs="cdn")
    print(f"  Tan δ plot     → {save_path}")


def plot_delta_phi(results: dict, save_path: Path) -> None:
    """Raw differential phase Δφ(f) — linearity diagnostic."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig = go.Figure()
    for key, res in results.items():
        if "delta_phi" not in res:
            continue
        r2  = res.get("r_squared", np.nan)
        lbl = LABELS.get(key, key) + (f" R²={r2:.4f}" if np.isfinite(r2) else "")
        fig.add_trace(go.Scatter(
            x=res["f_GHz"], y=np.degrees(res["delta_phi"]),
            name=lbl, mode="lines", line=_ls(key),
            hovertemplate=(f"<b>{LABELS.get(key,key)}</b><br>"
                           f"f=%{{x:.2f}} GHz<br>Δφ=%{{y:.2f}}°<extra></extra>"),
        ))
    _vrect(fig)
    fig.update_layout(
        title="Differential Phase Δφ (loaded − bare) — Linearity Check",
        xaxis_title="Frequency (GHz)", yaxis_title="Δφ (°)",
        hovermode="x unified", template="plotly_white", height=480,
        legend=dict(orientation="h", yanchor="bottom",
                    y=1.02, xanchor="right", x=1),
    )
    fig.write_html(str(save_path), include_plotlyjs="cdn")
    print(f"  Δφ plot        → {save_path}")


# =============================================================================
# SECTION 10 — CSV EXPORT
# =============================================================================

def export_csv(results: dict, save_path: Path) -> None:
    """
    One CSV row per (sample, frequency).
    Includes stated-L and calibrated-L columns side-by-side.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "sample_key", "material", "h_mm",
        "L_stated_mm", "L_eff_mm", "scale_factor", "is_pressed",
        "freq_Hz", "freq_GHz",
        "eps_app_stated_L",        # Level 1
        "eps_app_calibrated_L",    # Level 2
        "eps_true_best",           # Level 3 best estimate
        "eps_true_lower",          # Level 3 lower bound (d=0)
        "eps_true_upper",          # Level 3 upper bound (full Cu gap)
        "tan_delta",
        "delta_phi_deg",
        "r_squared",
    ]

    with save_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for key, res in results.items():
            props  = res.get("props", {})
            freq   = res["freq"]
            f_GHz  = res["f_GHz"]
            ea_s   = res.get("eps_app_stated",  np.full_like(freq, np.nan))
            ea_c   = res.get("eps_app",         np.full_like(freq, np.nan))
            et     = res.get("eps_true",         np.full_like(freq, np.nan))
            etl    = res.get("eps_true_lower",   np.full_like(freq, np.nan))
            etu    = res.get("eps_true_upper",   np.full_like(freq, np.nan))
            td     = res.get("tan_delta",        np.full_like(freq, np.nan))
            dp     = np.degrees(res.get("delta_phi", np.zeros_like(freq)))
            r2     = res.get("r_squared", np.nan)

            def f_(v):
                return f"{v:.6f}" if np.isfinite(float(v)) else "NaN"

            for i in range(len(freq)):
                w.writerow([
                    key,
                    props.get("material", ""),
                    props.get("h_mm", ""),
                    props.get("L_mm", ""),
                    res.get("L_mm_used", ""),
                    res.get("scale_factor", ""),
                    props.get("is_pressed", ""),
                    f"{freq[i]:.6e}",
                    f"{f_GHz[i]:.6f}",
                    f_(ea_s[i]),
                    f_(ea_c[i]),
                    f_(et[i]),
                    f_(etl[i]),
                    f_(etu[i]),
                    f_(td[i]),
                    f"{dp[i]:.4f}",
                    f"{r2:.6f}" if np.isfinite(r2) else "NaN",
                ])
    print(f"  CSV            → {save_path}")


# =============================================================================
# SECTION 11 — ENTRY POINT
# =============================================================================

# Internal key mapping — update stems to match your actual filenames
STEM_TO_KEY = {
    # GGG reference
    "GGG_large_Pressed_onSgMNMSA_010626":  "GGG_large_pressed",
    "GGG_large_onSgMNMSA_010626":          "GGG_large_free",
    "GGG_short_Pressed_onSgMNMSA_010626":  "GGG_short_pressed",
    "GGG_short_onSgMNMSA_010626":          "GGG_short_free",
    # Hematite samples
    "Fe2O3_14023080525_Pressed_onSgMNMSA_010626": "Hem_t02_pressed",
    "Fe2O3_14023080525_onSgMNMSA_010626":         "Hem_t02_free",
    "ml20240308b2_Pressed_onSgMNMSA_010626":      "Hem_t05_pressed",
    "ml20240308b2_onSgMNMSA_010626":              "Hem_t05_free",
}


def main() -> None:

    print("=" * 62)
    print("  CPW Permittivity Extraction — Multi-Sample with GGG Cal.")
    print("=" * 62)
    print(f"""
  CPW (CBCPW):  w={W/MIL:.1f}mil  g={G/MIL:.1f}mil  "
              t_Cu={T_CU/MIL:.1f}mil  h_RO={H_RO/MIL:.0f}mil  ε_RO={EPS_RO}
  q_sub={Q_SUB:.4f}   ε_eff_bare={EPS_EFF_BARE:.4f}
  Slot fraction={SLOT_FRACTION:.3f}   d_slot_best={D_SLOT_BEST*1000:.2f}μm  "
              d_slot_upper={D_SLOT_UPPER*1000:.1f}μm
  Analysis band: {F_LOW:.0f}–{F_HIGH:.0f} GHz
""")

    # ── Step 1: bare CPW ──────────────────────────────────────────────
    print(f"  [Step 1] Loading bare CPW: {BARE_STEM}.s2p")
    nw_bare = load_network(DATA_DIR, BARE_STEM)
    if nw_bare is None:
        print("  ✗ Bare CPW missing — cannot proceed.")
        return

    # ── Step 2: GGG calibration ───────────────────────────────────────
    print("\n  [Step 2] GGG calibration — solving L_eff ...")
    calibration = calibrate_from_ggg(nw_bare, REGISTRY, DATA_DIR)
    sf  = calibration["scale_factor"]
    std = calibration["scale_std"]
    print(f"    Scale factor = {sf:.4f} ± {std:.4f}  "
          f"(from {len(calibration['per_reference'])} pressed GGG measurements)")
    for pr in calibration["per_reference"]:
        print(f"    {pr['stem']}: L_stated={pr['L_stated']:.3f}mm  "
              f"L_eff={pr['L_eff']:.3f}mm  scale={pr['scale']:.4f}")

    # ── Steps 3–6: process all measurements ──────────────────────────
    print("\n  [Steps 3–6] Extracting ε for all measurements ...")
    results = {}

    for stem, key in STEM_TO_KEY.items():
        props = REGISTRY.get(stem)
        if props is None:
            print(f"  ⚠ No registry entry: {stem}")
            continue

        nw_load = load_network(DATA_DIR, stem)
        if nw_load is None:
            continue

        L_stated = props["L_mm"]
        L_cal    = L_stated * sf          # calibrated length

        # Extract with stated L (Level 1)
        res_stated = extract_eps_app(nw_bare, nw_load, L_stated, props["h_mm"])

        # Extract with calibrated L (Level 2 + 3)
        res_cal = extract_eps_app(nw_bare, nw_load, L_cal, props["h_mm"])

        # Gap correction on calibrated result (Level 3)
        gap = apply_gap_correction(res_cal["eps_app"], props["h_mm"],
                                   props["is_pressed"])

        # Merge into single result dict
        res = res_cal.copy()
        res["eps_app_stated"]  = res_stated["eps_app"]   # Level 1
        # res["eps_app"] is already Level 2 (calibrated L)
        res.update(gap)                                   # Level 3 fields
        res["props"]           = props
        res["L_mm_used"]       = L_cal
        res["scale_factor"]    = sf

        # Band statistics
        mask   = (res["freq"] >= F_LOW*1e9) & (res["freq"] <= F_HIGH*1e9)
        ea_s   = float(np.nanmean(res["eps_app_stated"][mask]))
        ea_c   = float(np.nanmean(res["eps_app"][mask]))
        et     = float(np.nanmean(res["eps_true"][mask]))
        td     = float(np.nanmean(res["tan_delta"][mask]))
        r2     = res["r_squared"]

        press_str = "pressed" if props["is_pressed"] else "free   "
        print(f"\n  [{key}]  {props['material']} "
              f"h={props['h_mm']}mm  L_stated={L_stated:.3f}mm  "
              f"L_eff={L_cal:.3f}mm  {press_str}")
        print(f"    Level 1  ε_app (stated L)   = {ea_s:.4f}")
        print(f"    Level 2  ε_app (cal. L)     = {ea_c:.4f}  "
              f"[Δ = {ea_c-ea_s:+.4f}]")
        print(f"    Level 3  ε_true (cal+gap)   = {et:.4f}  "
              f"[d_gap={gap['d_best_um']:.2f}μm]")
        print(f"             tan δ              = {td:.4f}")
        print(f"             Δφ linearity R²    = {r2:.5f}  "
              f"(±{res['residual_deg']:.2f}°)")

        if props.get("eps_known"):
            err  = (et - props["eps_known"]) / props["eps_known"] * 100
            flag = "✓" if abs(err) < 5 else "⚠"
            print(f"             Known ε={props['eps_known']}  "
                  f"error={err:+.1f}%  {flag}")

        results[key] = res

    # ── Step 7: cross-validation report ──────────────────────────────
    print()
    report = build_report(results, calibration)
    for line in report:
        print(line)

    # ── Step 8: plots ─────────────────────────────────────────────────
    print()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_epsilon(  results, sf, OUTPUT_DIR / "epsilon_all_samples.html")
    plot_tandelta( results,     OUTPUT_DIR / "tandelta_all_samples.html")
    plot_delta_phi(results,     OUTPUT_DIR / "delta_phi_all_samples.html")

    # ── Step 9: CSV ───────────────────────────────────────────────────
    export_csv(results, OUTPUT_DIR / "permittivity_results.csv")

    print("\n  Done.")


if __name__ == "__main__":
    main()
