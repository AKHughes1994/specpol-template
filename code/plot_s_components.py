#!/usr/bin/env python3
"""
Plot Stokes I, spectral index, p0, psi0, RM and reduced chi2 of the S components of every
epoch in results/<TARGET>, and save the values to a table.

RM_intrinsic is the fitted RM corrected for the spinifex ionosphere and PHI_SYS; the ionospheric error is added in quadrature.
"""

import re
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# =============================================================================
# CONFIGURATION
# =============================================================================
# Source name, PHI_SYS, iono window and JET_PA are set in source_config.py
from source_config import SOURCE_DIR as TARGET, SOURCE_NAME, PHI_SYS, IONO_HALF_WINDOW_MIN, JET_PA

MODEL_TYPE           = 'S'         # fit to read; every S component in it is plotted
CHI2_REF             = 1.0         # reference line on the reduced chi² panel

BASE = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE / 'results' / TARGET
PLOTS_DIR = BASE / 'plots' / TARGET

# Serif font, thick spines and inward ticks on all sides
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'], 'mathtext.fontset': 'dejavuserif',
    'axes.linewidth': 1.5, 'xtick.major.width': 1.5, 'ytick.major.width': 1.5,
    'xtick.major.size': 6, 'ytick.major.size': 6, 'xtick.minor.size': 3, 'ytick.minor.size': 3,
    'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True,
    'xtick.minor.visible': True, 'ytick.minor.visible': True, 'font.size': 12,
})


def read_header(qu_file):
    """Return the epoch (datetime) and MFS Stokes I (mJy, error) from a QU text header."""
    with open(qu_file) as fh:
        text = ''.join(line for line in fh if line.startswith('#'))
    date = re.search(r'Date \(ISO\):\s*(\S+)', text).group(1)
    flux, flux_err = re.search(r'MFS Stokes I \(mJy\):\s*(\S+)\s*\+/-\s*(\S+)', text).groups()
    return datetime.fromisoformat(date), float(flux), float(flux_err)


def iono_rm(epoch):
    """Mean spinifex RM and error within ±IONO_HALF_WINDOW_MIN of the epoch, from every file in spinifex_iono/."""
    epoch_mjd = (epoch - datetime(1858, 11, 17)).total_seconds() / 86400
    rows = {}  # keyed by (source, time) so rows repeated across files count once
    for f in sorted((BASE / 'files' / 'spinifex_iono').glob('*.txt')):
        with open(f) as fh:
            for line in fh:
                p = line.split()
                if not p or p[0].startswith('#') or SOURCE_NAME.lower() not in p[0].lower():
                    continue
                mjd, rm, err = map(float, p[-3:])  # last three columns: Time_MJD, RM, RM_err
                if abs(mjd - epoch_mjd) <= IONO_HALF_WINDOW_MIN / 1440:
                    rows[(p[0], round(mjd * 86400))] = (rm, err)
    if not rows:
        raise ValueError(f"No spinifex rows for '{SOURCE_NAME}' within ±{IONO_HALF_WINDOW_MIN} min of {epoch}")
    rm, err = np.array(list(rows.values())).T
    return rm.mean(), err.mean()


def stat(par):
    """Median and (lower, upper) 68% errors of a fitted parameter."""
    lo, hi = par['eti_68']
    return par['median'], par['median'] - lo, hi - par['median']


def weighted_median(x, w):
    """Value at which the cumulative weight reaches half of the total."""
    order = np.argsort(x)
    cum = np.cumsum(w[order])
    return x[order][np.searchsorted(cum, cum[-1] / 2)]


# =============================================================================
# EXTRACT
# =============================================================================
rows = []
for res_file in sorted(RESULTS_DIR.glob(f'*/*_{MODEL_TYPE}_results.json')):
    stem = res_file.parent.name
    with open(res_file) as fh:
        fit = json.load(fh)

    epoch, flux, flux_err = read_header(BASE / 'files' / TARGET / 'QU_text' / f'{stem}.txt')
    rm_iono, rm_iono_err = iono_rm(epoch)

    # Processed values hold the unwrapped psi_0, so they take priority
    pars = {**fit['parameters'], **fit.get('processed_parameters', {})}
    alpha, alpha_err = fit['stokes_i_fit']['alpha_nu'], fit['stokes_i_fit']['alpha_nu_err']
    gof = fit['goodness_of_fit']
    chi2, chi2_err = gof['importance_weighted']['reduced_chi_squared_mean'], gof['importance_weighted']['chi_squared_std'] / gof['dof']

    for name in fit['metadata']['component_names']:
        if not name.startswith('S'):
            continue
        rm_fit = stat(pars[f'{name}_phi_rm'])
        rows.append(dict(
            epoch=epoch, stem=stem, comp=name, flux=(flux, flux_err, flux_err), alpha=(alpha, alpha_err, alpha_err),
            p0=stat(pars[f'{name}_p0']), psi0=stat(pars[f'{name}_psi_0']), chi2=(chi2, chi2_err, chi2_err),
            # Remove the ionosphere and the offset; add the ionospheric error in quadrature to both sides
            rm=(rm_fit[0] - (rm_iono - PHI_SYS), np.hypot(rm_fit[1], rm_iono_err), np.hypot(rm_fit[2], rm_iono_err)),
            rm_fit=rm_fit[0], rm_iono=rm_iono, rm_iono_err=rm_iono_err))

if not rows:
    raise SystemExit(f"No '{MODEL_TYPE}' results found in {RESULTS_DIR}")

# Weighted (weights 1/sigma²) and plain median RM; errors from bootstrapping the epochs
rm = np.array([r['rm'][0] for r in rows])
w = 1 / np.array([np.mean(r['rm'][1:]) for r in rows]) ** 2
rng = np.random.default_rng(0)
boots = [rng.integers(0, len(rm), len(rm)) for _ in range(2000)]
rm_wmed = weighted_median(rm, w)
rm_wmed_err = np.std([weighted_median(rm[b], w[b]) for b in boots])
rm_med = np.median(rm)
rm_med_err = np.std([np.median(rm[b]) for b in boots])
print(f"Weighted median RM = {rm_wmed:.2f} ± {rm_wmed_err:.2f} rad/m² (N = {len(rows)})")
print(f"Median RM          = {rm_med:.2f} ± {rm_med_err:.2f} rad/m²")

# =============================================================================
# SAVE
# =============================================================================
table = RESULTS_DIR / f'{TARGET}_{MODEL_TYPE}_components.csv'
# Column labels: every quantity is a value followed by its lower and upper errors
labels = {'flux': 'I_mJy', 'alpha': 'alpha', 'p0': 'p0', 'psi0': 'psi0_deg', 'rm': 'RM_intrinsic_radm2', 'chi2': 'chi2_red'}
header = (['epoch', 'component', 'time_isot'] + [f"{name}{s}" for name in labels.values() for s in ('', '_err_lo', '_err_hi')]
          + ['RM_fit_radm2', 'RM_iono_radm2', 'RM_iono_err_radm2'])
with open(table, 'w', newline='') as out:
    # Comment lines carry the median RMs; read with pandas.read_csv(table, comment='#')
    out.write(f"# RM_intrinsic = RM_fit - (RM_iono - PHI_SYS) (PHI_SYS = {PHI_SYS}); its errors include RM_iono_err in quadrature\n")
    out.write("# Median errors are from bootstrapping the epochs\n")
    out.write(f"# Weighted median RM_intrinsic = {rm_wmed:.3f} +/- {rm_wmed_err:.3f} rad/m2\n")
    out.write(f"# Median RM_intrinsic = {rm_med:.3f} +/- {rm_med_err:.3f} rad/m2\n")
    writer = csv.writer(out)
    writer.writerow(header)
    for r in rows:
        writer.writerow([r['stem'], r['comp'], r['epoch'].isoformat()] + [f"{v:.6g}" for k in labels for v in r[k]]
                        + [f"{r[k]:.6g}" for k in ('rm_fit', 'rm_iono', 'rm_iono_err')])
print(f"Saved table: {table}")

# =============================================================================
# PLOT
# =============================================================================
panels = [('flux', 'Stokes I (mJy)'), ('alpha', r'$\alpha$'), ('p0', r'$p_0$'), ('psi0', r'$\psi_0$ (deg)'),
          ('rm', r'RM (rad m$^{-2}$)'), ('chi2', r'$\chi^2_{\rm red}$')]
fig, axes = plt.subplots(len(panels), 1, figsize=(9, 14), sharex=True)
x = [r['epoch'] for r in rows]

for ax, (key, label) in zip(axes, panels):
    y = np.array([r[key][0] for r in rows])
    yerr = np.array([[r[key][1] for r in rows], [r[key][2] for r in rows]])
    # One C0 circle with a black border per value, with its error bars
    ax.errorbar(x, y, yerr=yerr, fmt='o', markersize=8, markerfacecolor='C0', markeredgecolor='black',
                markeredgewidth=1.2, ecolor='black', elinewidth=1.2, capsize=3, zorder=3)
    ax.set_ylabel(label)
    ax.grid(True, alpha=0.3)

# psi0 is defined modulo 180 deg, so show the full range and repeat the jet PA bands at every wrap
if JET_PA is not None:
    for shift, color, name in [(0, 'C2', 'Jet PA'), (90, 'C1', r'Jet PA $+\,90^\circ$')]:
        for k in range(-360, 361, 180):
            axes[3].axhspan(JET_PA[0] + shift + k, JET_PA[1] + shift + k, color=color, alpha=0.25, linewidth=0,
                            zorder=0, label=name if k == -360 else None)
    axes[3].legend(loc='best', ncol=2, framealpha=0.9, fontsize=10)
axes[3].set_ylim(-100, 100)

# Weighted and plain median RM, each with its error band
for val, err, color, ls, name in [(rm_wmed, rm_wmed_err, 'gray', '--', 'Weighted median'),
                                  (rm_med, rm_med_err, 'C3', ':', 'Median')]:
    axes[4].axhline(val, color=color, linestyle=ls, linewidth=1.5, zorder=1,
                    label=rf'{name}: ${val:.1f} \pm {err:.1f}$ rad m$^{{-2}}$')
    axes[4].axhspan(val - err, val + err, color=color, alpha=0.2, zorder=0)

# Widen the RM axis to cover the data, both median bands and the legend
rm_lo_err, rm_hi_err = (np.array([r['rm'][k] for r in rows]) for k in (1, 2))
lows = np.append(rm - rm_lo_err, [rm_wmed - rm_wmed_err, rm_med - rm_med_err])
highs = np.append(rm + rm_hi_err, [rm_wmed + rm_wmed_err, rm_med + rm_med_err])
span = highs.max() - lows.min()
axes[4].set_ylim(lows.min() - 0.15 * span, highs.max() + 0.55 * span)
axes[4].legend(loc='upper center', ncol=2, framealpha=0.9, fontsize=10)

# Reduced chi² reference line
axes[5].axhline(CHI2_REF, color='gray', linestyle='--', linewidth=1.5, zorder=1)

# Date ticks every 5 days at multiples of MJD 5 (matplotlib dates are days since 1970-01-01)
mjd_zero = mdates.date2num(datetime(1858, 11, 17))
lo, hi = axes[-1].get_xlim()
mjd_ticks = np.arange(np.ceil((lo - mjd_zero) / 5) * 5, hi - mjd_zero, 5)
axes[-1].set_xticks(mjd_ticks + mjd_zero)
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
axes[-1].set_xlabel('Date')
plt.setp(axes[-1].get_xticklabels(), rotation=30, ha='right')

# Top axis of the top panel: the same tick positions, labelled in MJD
ax_mjd = axes[0].secondary_xaxis('top', functions=(lambda d: d - mjd_zero, lambda m: m + mjd_zero))
ax_mjd.set_xticks(mjd_ticks)
ax_mjd.set_xticklabels([f'{m:.0f}' for m in mjd_ticks])
ax_mjd.set_xlabel('MJD')
ax_mjd.tick_params(which='both', direction='in')
fig.subplots_adjust(hspace=0.08)
fig.align_ylabels()

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
plot_file = PLOTS_DIR / f'{TARGET}_{MODEL_TYPE}_components.png'
fig.savefig(plot_file, dpi=200, bbox_inches='tight')
print(f"Saved plot: {plot_file}")
