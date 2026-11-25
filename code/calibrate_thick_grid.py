#!/usr/bin/env python3
"""
calibrate_thick_grid.py

Calibrate the Faraday-depth grid that ThickComponent.compute_polarization uses
to integrate super-Gaussian (N != 2) FDFs numerically, for this source's band
and priors.

Method
------
For each super-Gaussian order N in N_LIST, the forward transform
    P(lambda^2) = integral F(phi) exp(2i phi lambda^2) dphi
is computed on a very fine, wide "fiducial" grid (treated as truth) and on a
sweep of coarser grids (half-width N_SIG * sigma_phi, N_PHI points). The cheapest
grid (fewest N_PHI, then smallest N_SIG) whose flux-weighted error
    rel_fw = sum|P - P_fid| / sum|P_fid|
is below REL_THRESHOLD is kept; grids are tried in that order and the search
stops at the first one that passes. Then
    N_SIG(N) = a * N^b + c     and     N_PHI(N) = a * N^b + c
are fitted (pinned through N = 2, N_PHI inflated by NPHI_INFLATE_FACTOR) and
checked against the fiducial again.

Inputs
------
* lambda^2 sampling: every channel in files/<SOURCE_DIR>/QU_text/*.txt. If there
  are no QU text files yet, MeerKAT L-band (0.85-1.8 GHz) is used, with a warning.
* sigma_phi = SIGMA_PHI_FACTOR (2) x W_max, where W_max ~ 0.67 (1/lambda^2_min +
  1/lambda^2_max) rad/m² is the widest Gaussian Faraday-thick FWHM the band is
  sensitive to (polarised power down by half; Rudnick & Cotton 2023,
  arXiv:2304.02728). The MeerKAT L-band W_max is printed for reference.
* phi_peak = max(|PHI_MIN|, |PHI_MAX|) from source_config.py: the fastest-
  rotating thick component the priors allow, i.e. the hardest case for the grid.

Output
------
THICK_NSIG_FIT / THICK_NPHI_FIT lines printed at the end, to paste into
source_config.py by hand, plus
    results/thick_grid_calibration/thick_grid_sweep.txt   (best grid per N)
    plots/thick_grid_calibration/thick_grid_vs_N.png

Run from code/:  python3 calibrate_thick_grid.py
Self-contained: every run calibrates from scratch and nothing is read back from
earlier runs. Kept fast by stopping each N's search at the first passing grid,
integrating each reference once (reused by the final check), and rotating the
phase factor along the uniform phi grid instead of calling sin/cos at every
point; numba uses all cores.
"""

import sys
import time
import warnings

import numpy as np
import matplotlib
matplotlib.use('Agg')  # save files only, no plot windows
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from numba import njit, prange

from faraday_utils import ThickComponent, C_LIGHT
from source_config import BASE_DIR, SOURCE_DIR, PHI_MIN, PHI_MAX


# =============================================================================
# SETTINGS
# =============================================================================
QU_TEXT_DIR = BASE_DIR / 'files' / SOURCE_DIR / 'QU_text'
RESULTS_FILE = BASE_DIR / 'results' / 'thick_grid_calibration' / 'thick_grid_sweep.txt'
PLOT_FILE = BASE_DIR / 'plots' / 'thick_grid_calibration' / 'thick_grid_vs_N.png'

# MeerKAT L-band: used when there are no QU text files yet, and as the W_max reference
FALLBACK_FREQ_GHZ = np.linspace(0.85, 1.8, 512)
L_BAND_LAMBDA_SQ = np.sort((C_LIGHT / (FALLBACK_FREQ_GHZ * 1e9)) ** 2)

# FDF used for the sweep (hardest case): sigma_phi = SIGMA_PHI_FACTOR x W_max of the
# band (worked out from the data in main), phi_peak = the largest |phi| the priors allow
SIGMA_PHI_FACTOR = 2.0
PHI_PEAK = max(abs(PHI_MIN), abs(PHI_MAX))     # rad/m²
A, B = 0.7, -0.2                               # complex amplitude (sets the phase only)

# Super-Gaussian orders to calibrate: the full N prior range of the fits (2-50),
# log-spaced since behaviour changes fastest at low N
N_LIST = np.geomspace(2.0, 50.0, 100)

# Candidate grids: half-width in units of sigma_phi, and point count (made odd
# so the profile peak falls on a grid node)
N_SIG_VALUES = np.arange(1.0, 7.0 + 1e-9, 0.2)
N_PHI_CANDIDATES = np.arange(30, 2001, 10)   # search stops at the first pass, so the top end only costs time if needed

# Fiducial ("truth") grid: very wide and very fine
N_SIG_FID = 10.0
N_PHI_FID = 200001

REL_THRESHOLD = 1e-6        # accept a grid if rel_fw is below this
NPHI_INFLATE_FACTOR = 1.20  # safety margin on N_PHI before fitting


# =============================================================================
# BAND (lambda^2 SAMPLING)
# =============================================================================
def load_lambda_sq():
    """Sorted lambda^2 (m²) of every channel in the QU text files, or of L-band if there are none."""
    qu_files = sorted(QU_TEXT_DIR.glob('*.txt'))
    if qu_files:
        freq_hz = np.unique(np.concatenate([np.loadtxt(f, usecols=0, ndmin=1) for f in qu_files]))
        freq_hz = freq_hz[np.isfinite(freq_hz) & (freq_hz > 0)]
        print(f"Band from {len(qu_files)} QU text file(s) in {QU_TEXT_DIR}")
    else:
        warnings.warn(f"No QU text files in {QU_TEXT_DIR}; calibrating for MeerKAT L-band "
                      f"(0.85-1.8 GHz) instead. Re-run once your QU text files are in place.",
                      UserWarning, stacklevel=2)
        return L_BAND_LAMBDA_SQ
    print(f"  {freq_hz.size} channels, {freq_hz.min() / 1e9:.3f}-{freq_hz.max() / 1e9:.3f} GHz")
    return np.sort((C_LIGHT / freq_hz) ** 2)


def w_max(lambda_sq):
    """Widest Gaussian Faraday-thick FWHM (rad/m²) the band is sensitive to, where the polarised
    power falls by half: W_max ~ 0.67 (1/lambda^2_min + 1/lambda^2_max) (Rudnick & Cotton 2023)."""
    return 0.67 * (1.0 / lambda_sq.min() + 1.0 / lambda_sq.max())


def print_widths(lambda_sq):
    """W_max and the sweep sigma_phi for this band, with the MeerKAT L-band values for reference."""
    print(f"W_max (Rudnick & Cotton 2023): this band {w_max(lambda_sq):.1f} rad/m² "
          f"-> sweep sigma_phi = {SIGMA_PHI_FACTOR:g} x W_max = {SIGMA_PHI_FACTOR * w_max(lambda_sq):.1f} rad/m²")
    print(f"  reference, MeerKAT L-band 0.85-1.8 GHz: W_max = {w_max(L_BAND_LAMBDA_SQ):.1f} rad/m², "
          f"{SIGMA_PHI_FACTOR:g} x W_max = {SIGMA_PHI_FACTOR * w_max(L_BAND_LAMBDA_SQ):.1f} rad/m²")


# =============================================================================
# INTEGRATION
# =============================================================================
@njit(parallel=True, cache=True)
def _transform_kernel(phi, F_real, F_imag, lambda_sq):
    """Trapezoid-rule P(lambda^2) = integral F(phi) exp(2i phi lambda^2) dphi on a uniform
    phi grid, parallel over channels (same rule as faraday_utils.fourier_transform_fdf_numba).

    The phase factor is advanced by a fixed rotation exp(2i lambda^2 dphi) from one grid
    point to the next instead of calling cos/sin at every point. The rounding this adds
    (~1e-11 relative even over the 2e5-point reference grid) is far below REL_THRESHOLD.
    """
    n_l2, n_phi = lambda_sq.shape[0], phi.shape[0]
    dphi = phi[1] - phi[0]
    P_real, P_imag = np.zeros(n_l2), np.zeros(n_l2)

    for j in prange(n_l2):
        c, s = np.cos(2.0 * phi[0] * lambda_sq[j]), np.sin(2.0 * phi[0] * lambda_sq[j])   # phase at the first point
        dc, ds = np.cos(2.0 * dphi * lambda_sq[j]), np.sin(2.0 * dphi * lambda_sq[j])     # rotation per step
        re_sum, im_sum = 0.0, 0.0
        for k in range(n_phi):
            w = 0.5 if (k == 0 or k == n_phi - 1) else 1.0   # trapezoid end weights
            re_sum += w * (F_real[k] * c - F_imag[k] * s)
            im_sum += w * (F_real[k] * s + F_imag[k] * c)
            c, s = c * dc - s * ds, s * dc + c * ds           # advance the phase one step
        P_real[j], P_imag[j] = re_sum * dphi, im_sum * dphi

    return P_real, P_imag


def transform_on_grid(N, n_sig, n_phi, lambda_sq, sigma_phi):
    """P(lambda^2) of the sweep FDF (order N, width sigma_phi) integrated on a grid of half-width
    n_sig * sigma_phi with n_phi points. The FDF comes from ThickComponent itself."""
    phi = np.linspace(PHI_PEAK - n_sig * sigma_phi, PHI_PEAK + n_sig * sigma_phi, n_phi)
    fdf = ThickComponent(A, B, PHI_PEAK, sigma_phi, N).compute_fdf(phi)
    P_real, P_imag = _transform_kernel(phi, np.ascontiguousarray(fdf.real),
                                       np.ascontiguousarray(fdf.imag), lambda_sq)
    return P_real + 1j * P_imag


def error_metrics(P, P_fid):
    """Flux-weighted relative error sum|dP| / sum|P_fid| (primary) and mean |dP|."""
    d = np.abs(P - P_fid)
    return float(np.sum(d) / np.sum(np.abs(P_fid))), float(np.mean(d))


def round_to_odd(n):
    """Nearest odd integer >= 3 (rounding up from even)."""
    n = max(int(round(n)), 3)
    return n if n % 2 == 1 else n + 1


# =============================================================================
# SWEEP
# =============================================================================
def best_grid_for_N(N, lambda_sq, n_phi_list, sigma_phi, P_fid):
    """Cheapest (n_sig, n_phi, rel_fw) for order N with rel_fw < REL_THRESHOLD against P_fid.

    Grids are tried fewest points first, then narrowest, and the search stops at the
    first that passes. If none passes, the most accurate grid tested is returned.
    """
    best_tested = None
    for n_phi in n_phi_list:
        for n_sig in N_SIG_VALUES:
            rel_fw, _ = error_metrics(transform_on_grid(N, n_sig, n_phi, lambda_sq, sigma_phi), P_fid)
            if rel_fw < REL_THRESHOLD:
                return float(n_sig), int(n_phi), rel_fw
            if best_tested is None or rel_fw < best_tested[2]:
                best_tested = (float(n_sig), int(n_phi), rel_fw)

    print(f"  N = {N:.4g}: no grid met rel_fw < {REL_THRESHOLD:.0e}; using the most accurate tested")
    return best_tested


def run_sweep(lambda_sq, sigma_phi):
    """Best grid for every N in N_LIST. Returns rows (N, n_sig, n_phi, rel_fw) and the
    reference P(lambda^2) for each N (reused by validate_fit)."""
    n_phi_list = sorted({round_to_odd(n) for n in N_PHI_CANDIDATES})
    print(f"\nSweep: {N_LIST.size} N values, {N_SIG_VALUES.size} x {len(n_phi_list)} grids each; "
          f"sigma_phi = {sigma_phi:.1f}, phi_peak = {PHI_PEAK:g} rad/m², threshold rel_fw < {REL_THRESHOLD:.0e}")

    transform_on_grid(2.5, 1.0, 5, lambda_sq[:2], sigma_phi)   # compile the numba kernel once
    rows, fiducials, t_start = [], [], time.perf_counter()
    for N in N_LIST:
        P_fid = transform_on_grid(N, N_SIG_FID, N_PHI_FID, lambda_sq, sigma_phi)   # reference ("truth")
        n_sig, n_phi, rel_fw = best_grid_for_N(N, lambda_sq, n_phi_list, sigma_phi, P_fid)
        rows.append((float(N), n_sig, n_phi, rel_fw))
        fiducials.append(P_fid)
        print(f"  N = {N:7.4f}: N_SIG = {n_sig:.1f}, N_PHI = {n_phi:4d}, rel_fw = {rel_fw:.2e}")
    print(f"Sweep took {time.perf_counter() - t_start:.0f} s")
    return np.array(rows), fiducials


def save_sweep(rows, lambda_sq, sigma_phi):
    """Save the best grid per N, with the band and FDF it was calibrated for."""
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    freq_ghz = C_LIGHT / np.sqrt(lambda_sq) / 1e9
    header = (f"Thick-component grid sweep: {lambda_sq.size} channels, "
              f"{freq_ghz.min():.3f}-{freq_ghz.max():.3f} GHz; sigma_phi = {sigma_phi:.1f}, "
              f"phi_peak = {PHI_PEAK:g} rad/m²; threshold rel_fw < {REL_THRESHOLD:.0e}\n"
              f"N  best_N_SIG  best_N_PHI  best_rel_fw")
    np.savetxt(RESULTS_FILE, rows, header=header, fmt=['%.6f', '%.4f', '%d', '%.6e'])
    print(f"Saved sweep: {RESULTS_FILE}")


# =============================================================================
# FIT
# =============================================================================
def monotone_envelope(values):
    """Raise each point to the maximum of everything at larger N, so the fit never underestimates."""
    return np.maximum.accumulate(values[::-1])[::-1]


def fit_pinned_power_law(Ns, ys, p0, bounds):
    """Fit y = a N^b + c with c fixed so the curve passes through the first point. Returns (a, b, c)."""
    N0, y0 = Ns[0], ys[0]
    popt, _ = curve_fit(lambda N, a, b: a * N ** b + (y0 - a * N0 ** b),
                        Ns, ys, p0=p0, bounds=bounds, maxfev=10000)
    a, b = popt
    return float(a), float(b), float(y0 - a * N0 ** b)


def fit_grid_curves(rows):
    """Fitted (a, b, c) for N_SIG(N) and N_PHI(N), plus the envelopes they were fitted to."""
    Ns, n_sig, n_phi = rows[:, 0], rows[:, 1], rows[:, 2]
    n_sig_env = monotone_envelope(n_sig)
    n_phi_env = np.array([round_to_odd(v * NPHI_INFLATE_FACTOR) for v in monotone_envelope(n_phi)], dtype=float)

    nsig_fit = fit_pinned_power_law(Ns, n_sig_env, p0=(20.0, -2.0), bounds=((0.0, -10.0), (1e5, -0.01)))
    nphi_fit = fit_pinned_power_law(Ns, n_phi_env, p0=(500.0, -2.0), bounds=((0.0, -10.0), (1e7, -0.01)))
    return nsig_fit, nphi_fit, n_sig_env, n_phi_env


def validate_fit(nsig_fit, nphi_fit, lambda_sq, sigma_phi, fiducials):
    """Integration error at every N when the fitted grid is used (what the fits will actually
    see), against the reference P(lambda^2) already computed in run_sweep."""
    print("\nFitted grid check (rel_fw should stay near or below the threshold):")
    worst = 0.0
    for N, P_fid in zip(N_LIST, fiducials):
        n_sig = nsig_fit[0] * N ** nsig_fit[1] + nsig_fit[2]
        n_phi = round_to_odd(nphi_fit[0] * N ** nphi_fit[1] + nphi_fit[2])
        rel_fw, _ = error_metrics(transform_on_grid(N, n_sig, n_phi, lambda_sq, sigma_phi), P_fid)
        worst = max(worst, rel_fw)
        print(f"  N = {N:7.4f}: N_SIG = {n_sig:.3f}, N_PHI = {n_phi:4d}, rel_fw = {rel_fw:.2e}")
    print(f"Worst rel_fw on the fitted grid: {worst:.2e}")


# =============================================================================
# PLOT
# =============================================================================
def plot_calibration(rows, nsig_fit, nphi_fit, n_sig_env, n_phi_env, sigma_phi):
    """Best N_SIG and N_PHI vs N with their fitted curves, and the error reached."""
    Ns = rows[:, 0]
    N_fine = np.linspace(Ns[0], Ns[-1], 500)
    fig, axes = plt.subplots(3, 1, figsize=(7.5, 9.0), sharex=True)

    panels = [(rows[:, 1], n_sig_env, nsig_fit, "monotone envelope", r"Grid half-width N_SIG ($\sigma_\phi$)"),
              (rows[:, 2], n_phi_env, nphi_fit, f"envelope x{NPHI_INFLATE_FACTOR:.2f}, odd", "Grid points N_PHI")]
    for ax, (raw, env, (a, b, c), env_label, ylabel) in zip(axes[:2], panels):
        ax.plot(Ns, raw, 'o', ms=4, color='C0', alpha=0.3, label='best per N')
        ax.plot(Ns, env, 'o', ms=4, color='C0', label=env_label)
        ax.plot(N_fine, a * N_fine ** b + c, lw=1.8, color='C1',
                label=f"{a:.3f} N$^{{{b:.3f}}}$ + {c:.3f}")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8, frameon=False)

    axes[2].plot(Ns, rows[:, 3], 'o', ms=4, color='C0')
    axes[2].axhline(REL_THRESHOLD, color='grey', ls=':', lw=1)
    axes[2].set_yscale('log')
    axes[2].set_ylabel('Best rel_fw')
    axes[2].set_xlabel('Super-Gaussian order N')
    axes[0].set_title(f"Thick-component grid calibration "
                      f"($\\sigma_\\phi$ = {sigma_phi:.1f}, $\\phi_{{peak}}$ = {PHI_PEAK:g} rad/m²)")

    fig.tight_layout()
    PLOT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_FILE, dpi=200)
    plt.close(fig)
    print(f"Saved plot: {PLOT_FILE}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    lambda_sq = load_lambda_sq()
    print_widths(lambda_sq)
    sigma_phi = SIGMA_PHI_FACTOR * w_max(lambda_sq)

    rows, fiducials = run_sweep(lambda_sq, sigma_phi)
    save_sweep(rows, lambda_sq, sigma_phi)

    nsig_fit, nphi_fit, n_sig_env, n_phi_env = fit_grid_curves(rows)
    validate_fit(nsig_fit, nphi_fit, lambda_sq, sigma_phi, fiducials)
    plot_calibration(rows, nsig_fit, nphi_fit, n_sig_env, n_phi_env, sigma_phi)

    print()
    print_widths(lambda_sq)
    print("Paste into code/source_config.py (Thick-component integration grid):")
    print(f"THICK_NSIG_FIT = ({nsig_fit[0]:.3f}, {nsig_fit[1]:.3f}, {nsig_fit[2]:.3f})")
    print(f"THICK_NPHI_FIT = ({nphi_fit[0]:.3f}, {nphi_fit[1]:.3f}, {nphi_fit[2]:.3f})")


if __name__ == "__main__":
    sys.exit(main())
