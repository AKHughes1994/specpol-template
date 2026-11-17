import warnings
warnings.filterwarnings('ignore', message='.*pkg_resources.*')

import re
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict
try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        raise ImportError("Python ≥3.11 required for tomllib, or install tomli: pip install tomli")
from faraday_plot import plot_flagging_diagnostic
from faraday_data import DataLoader, FlaggingConfig
from faraday_utils import load_iono_rm
from source_config import (SOURCE_NAME, SOURCE_DIR, PHI_SYS, IONO_LOC, IONO_HALF_WINDOW_MIN,
                           ISM_EXCLUDE, MIN_PEAK_SNR)
import subprocess

FLAGGING_CONFIG_FILE = Path(__file__).parent / 'flagging_config.toml'

# =============================================================================
# SOURCE-SPECIFIC SETTINGS — review/change these when adapting to a new source
# =============================================================================
# Source name, iono and ISM settings are set in source_config.py.
# TARGETS: subdirectories of ../files/ that contain a QU_text/ folder of input spectra.
# Defaults to SOURCE_DIR; can be overridden on the command line: python3 faraday_synthesis.py target other_source
TARGETS = sys.argv[1:] or [SOURCE_DIR]

# rmclean1d thresholds (log10 of the CLEAN cutoff), applied to every file
DEFAULT_PEAK  = '-7'   # -c : peak threshold
DEFAULT_FINER = '-3'   # -w : finer-mask threshold

# Optional per-epoch overrides: {frozenset of substrings of the file stem: threshold};
# first match wins. e.g. PEAK_OVERRIDES = {frozenset(['20240204']): '-14'}
PEAK_OVERRIDES = {}
FINER_OVERRIDES = {}


def get_peak_threshold(base_name):
    for keys, val in PEAK_OVERRIDES.items():
        if any(k in base_name for k in keys):
            return val
    return DEFAULT_PEAK


def get_finer_threshold(base_name):
    for keys, val in FINER_OVERRIDES.items():
        if any(k in base_name for k in keys):
            return val
    return DEFAULT_FINER


def _date_key(stem):
    """Extract YYYYMMDD from a file stem; fall back to the full stem."""
    m = re.search(r'(\d{8})', stem)
    return m.group(1) if m else stem


def _resolve_files(files_path):
    """
    Return the ordered list of files to process.

    For dates where multiple files exist and at least one has 'scan'
    (case-insensitive) in the stem, keep every scan file individually.
    For all other dates, keep the file(s) as-is.
    """
    all_files = sorted(files_path.glob('*.txt'))
    groups = defaultdict(list)
    for f in all_files:
        groups[_date_key(f.stem)].append(f)

    resolved = []
    for date in sorted(groups):
        group = groups[date]
        scan_files = [f for f in group if re.search(r'scan', f.stem, re.IGNORECASE)]
        if len(group) > 1 and scan_files:
            print(f"  [{date}] {len(scan_files)} scan file(s) found — running per-scan")
            resolved.extend(sorted(scan_files))
        else:
            resolved.extend(group)
    return resolved


def read_rmtools_dat(path):
    """Return the key=value entries of an RM-tools summary file as a dict of strings."""
    with open(path) as fh:
        return dict(line.strip().split('=', 1) for line in fh if '=' in line)


def summarise_ism_rm(target, input_files, output_path):
    """
    Write <output_path>/<target>_ism_rm.txt with, for every epoch whose FDF peak has S/N >= MIN_PEAK_SNR:
      RM_peak  : peak Faraday depth of the CLEANed FDF (RM-tools phiPeakPIfit_rm2) and its error
      RM_iono  : mean spinifex ionospheric RM within ±IONO_HALF_WINDOW_MIN of the epoch
      RM_ISM   : RM_peak - (RM_iono - PHI_SYS)
    plus the median RM_ISM over epochs and the error on that median, which can be used
    as the baseline PHI_ISM (and its uncertainty) in faraday_obs_fit.py.
    """
    rows = []
    for f in input_files:
        clean_file = output_path / f.stem / f"{f.stem}_rmsynth_RMclean.dat"
        if not clean_file.exists():
            continue

        # A peak below the S/N threshold is just noise, whether or not CLEAN ran
        info = read_rmtools_dat(clean_file)
        snr = float(info['snrPIfit'])
        if snr < MIN_PEAK_SNR:
            print(f"  {f.stem}: peak S/N {snr:.1f} < {MIN_PEAK_SNR}, skipping")
            continue
        rm_peak, rm_peak_err = float(info['phiPeakPIfit_rm2']), float(info['dPhiPeakPIfit_rm2'])

        try:
            rm_iono, rm_iono_err = load_iono_rm(IONO_LOC, SOURCE_NAME, f, IONO_HALF_WINDOW_MIN)
        except ValueError as e:
            print(f"  {f.stem}: {e}, skipping")
            continue

        used = not any(k in f.stem for k in ISM_EXCLUDE)
        rows.append((f.stem, rm_peak, rm_peak_err, rm_iono, rm_iono_err, rm_peak - (rm_iono - PHI_SYS), used))

    if not rows:
        print("No epochs with a significant peak and an ionospheric RM, no ISM summary written.")
        return

    # Median and its error, using a robust sigma (1.4826 * MAD) so outlier epochs don't inflate it;
    # the standard error of a median for ~Gaussian data is 1.2533 * sigma / sqrt(N)
    ism = np.array([r[5] for r in rows if r[6]])
    median = np.median(ism)
    sigma = 1.4826 * np.median(np.abs(ism - median))
    median_err = 1.2533 * sigma / np.sqrt(len(ism)) if len(ism) > 1 else np.nan

    out_file = output_path / f"{target}_ism_rm.txt"
    with open(out_file, 'w') as out:
        out.write(f"# PHI_SYS = {PHI_SYS} rad/m2; iono window = +/-{IONO_HALF_WINDOW_MIN} min\n")
        out.write(f"# Median ISM RM = {median:.3f} +/- {median_err:.3f} rad/m2 (N = {len(ism)})\n")
        out.write("# Ionosphere- and bias-corrected: RM_ISM = RM_peak - (RM_iono (spinifex) - PHI_SYS)\n")
        out.write("# Epoch RM_peak RM_peak_err RM_iono RM_iono_err RM_ISM Used(1=in median)\n")
        for stem, rm_peak, rm_peak_err, rm_iono, rm_iono_err, rm_ism, used in rows:
            out.write(f"{stem} {rm_peak:.3f} {rm_peak_err:.3f} {rm_iono:.3f} {rm_iono_err:.3f} {rm_ism:.3f} {int(used)}\n")

    print(f"\nMedian ISM RM = {median:.3f} +/- {median_err:.3f} rad/m² (N = {len(ism)}), "
          f"corrected for spinifex iono and PHI_SYS, written to {out_file}")


def process_target(target, flagging_cfg):
    files_path = Path(__file__).parent.parent / 'files' / target / 'QU_text'
    output_path = Path(__file__).parent.parent / 'rmsynth' / target
    input_files = _resolve_files(files_path)

    if not input_files:
        print(f"No input files found for {target}, skipping.")
        return

    # Ask once if any already-fit outputs exist
    any_existing = any(
        (output_path / f.stem / f"{f.stem}_rmsynth_FDFmodel.dat").exists()
        for f in input_files
    )
    refit_existing = False
    if any_existing:
        response = input(f"\nSome outputs already exist for {target}.\nRe-fit already processed files? (y/n): ").strip().lower()
        refit_existing = response == 'y'

    for input_file in input_files:
        print(f"\nProcessing: {input_file}")
        base_name = input_file.stem
        output_subdir = output_path / base_name
        output_subdir.mkdir(parents=True, exist_ok=True)

        already_fit = (output_subdir / f"{base_name}_rmsynth_FDFmodel.dat").exists()
        if already_fit and not refit_existing:
            print(f"Skipping {base_name} (already fit).")
            continue

        stokes_i = np.median(np.loadtxt(input_file)[:, 1])

        # Build flagging params: default + first matching override
        fparams = dict(flagging_cfg['default'])
        for override in flagging_cfg.get('overrides', []):
            if any(m in base_name for m in override['match']):
                fparams.update({k: v for k, v in override.items() if k != 'match'})
                print(f"Flagging: applied override (match={override['match']})")
                break
        else:
            print("Flagging: using default configuration")

        spectrum_jitter = fparams['jitter_frac'] * stokes_i
        print(f"Stokes I median: {stokes_i:.6f}, jitter: {spectrum_jitter:.6f}")

        flagging = FlaggingConfig(
            sigma_clip_extreme=fparams['sigma_clip_extreme'],
            freq_ranges=fparams['freq_ranges'],
            clip_stokes_i=None,
            max_iter=100,
            sigma_clip_errors=fparams['sigma_clip_errors'],
            sigma_clip_errors_space='log',
            sigma_clip_spectrum=fparams['sigma_clip_spectrum'],
            spectrum_deg=fparams['spectrum_deg'],
            spectrum_jitter=spectrum_jitter,
            spectrum_clip_sided=fparams['spectrum_clip_sided'],
            spectrum_stokes=fparams['spectrum_stokes'],
            sigma_clip_v=fparams['sigma_clip_v'],
            clip_v_against='median',
            verbose=True,
        )

        load_result = DataLoader.load_physical_with_fit(input_file, I_model='polynomial',
                                                        flagging=flagging, load_stokes_v=True)
        data, I_fitter, flagging_info = load_result
        plot_flagging_diagnostic(flagging_info['raw_data'], flagging_info,
                                 filename=f'{base_name}_flagging_diagnostic.png',
                                 output_dir=output_subdir)

        # Save flagged data for RMSynthesis
        flagged_cols = [flagging_info['raw_data'][k][~flagging_info['mask']]
                        for k in ['freq', 'I', 'Q', 'U', 'dI', 'dQ', 'dU']]
        rmsynth_file = output_subdir / f"{base_name}_rmsynth.txt"
        np.savetxt(rmsynth_file, np.column_stack(flagged_cols))

        # RMSynthesis
        subprocess.run([
            'rmsynth1d',
            str(rmsynth_file),
            '-S', '-o', '4', '-l', '2500', '--super-resolution'
        ])

        # RMClean
        subprocess.run([
            'rmclean1d',
            str(rmsynth_file),
            '-S', '-c', get_peak_threshold(base_name), '-w', get_finer_threshold(base_name)
        ])

    # Baseline ISM RM from all CLEANed epochs (including ones skipped above as already fit)
    summarise_ism_rm(target, input_files, output_path)

def main():
    with open(FLAGGING_CONFIG_FILE, 'rb') as f:
        flagging_cfg = tomllib.load(f)

    for target in TARGETS:
        print(f"\n{'='*60}\nTarget: {target}\n{'='*60}")
        process_target(target, flagging_cfg)


if __name__ == "__main__":
    main()
