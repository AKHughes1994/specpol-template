# =============================================================================
# SOURCE CONFIGURATION — the one file to edit when adapting to a new source
#
# Every script in code/ (and files/<SOURCE_DIR>/convert_json_to_text.py)
# imports its source-specific settings from here, so they can't drift apart.
# Per-epoch flagging lives separately in flagging_config.toml, and the model
# lists per epoch live in run_all_model.sh / run_best_models.sh.
# =============================================================================
from pathlib import Path

import astropy.units as u
from astropy.coordinates import EarthLocation

BASE_DIR = Path(__file__).resolve().parent.parent   # this file lives in code/


# -----------------------------------------------------------------------------
# Source identity
# -----------------------------------------------------------------------------
SOURCE_NAME = ''         # e.g. 'GX_339-4'. Prefix of the QU text file names, written to their
                         # headers, and the substring used to find the source in the spinifex files
SOURCE_DIR  = 'target'   # subdirectory of files/, results/, plots/ and rmsynth/


# -----------------------------------------------------------------------------
# Telescope / band
# -----------------------------------------------------------------------------
# Observatory location, used by run_spinifex.py for the ionospheric RM (default: MeerKAT)
OBSERVATORY = EarthLocation(lon=21.44326 * u.deg, lat=-30.712455 * u.deg, height=1059.662443 * u.m)
CENTRAL_FREQ_GHZ = 1.28   # band centre, used as the Stokes I power-law reference (MeerKAT L-band)
LAMBDA_SQ_REF    = 0.05   # m²; ~band centre. psi_0 is sampled referenced to this lambda² (Faraday
                          # rotation removed; p0 unchanged), and power-law p0 is quoted here.
                          # 0.0 -> sample psi_0 at lambda² = 0 (not valid with 'P' components)
                          # None -> disable (power-law components fall back to lambda_sq_0 = 0.05 m²)


# -----------------------------------------------------------------------------
# Ionosphere (spinifex) and ISM foreground RM
# -----------------------------------------------------------------------------
IONO_LOC             = BASE_DIR / 'files' / 'spinifex_iono'   # spinifex dir or single file; None -> iono_rm = 0
IONO_HALF_WINDOW_MIN = 30     # iono RM = mean of spinifex rows within ± this many minutes of the epoch
PHI_SYS              = -0.3   # systematic offset of the spinifex iono RM (rad/m²): true iono = RM_iono - PHI_SYS

# Optional prior pinning the first S (or P) component's RM to the ISM + iono value.
# Run faraday_synthesis.py first: it writes rmsynth/<SOURCE_DIR>/<SOURCE_DIR>_ism_rm.txt,
# whose median (already corrected for iono and PHI_SYS) is what PHI_ISM should be set to.
CONSTRAIN_ISM_RM = False
PHI_ISM          = 0.0                  # expected Galactic ISM Faraday depth (rad/m²)
PHI_ISM_SIGMA    = 3.0                  # Gaussian std of the truncated prior (rad/m²)
PHI_ISM_RANGE    = 3.5 * PHI_ISM_SIGMA  # truncation half-width; 0 -> fixed at PHI_ISM

ISM_EXCLUDE  = ('ejecta',)   # epochs with any of these in the file stem are left out of the ISM median
MIN_PEAK_SNR = 6.0           # epochs whose FDF peak S/N (RM-tools snrPIfit) is below this are skipped


# -----------------------------------------------------------------------------
# RM synthesis (faraday_synthesis.py: RM-Tools rmsynth1d + rmclean1d)
# -----------------------------------------------------------------------------
RMSYNTH_WEIGHTING        = 'variance'  # channel weighting (-w): 'variance' (inverse-variance) or 'uniform' (all equal)
RMSYNTH_PHI_MAX          = 2500        # largest |Faraday depth| sampled, rad/m² (-l)
RMSYNTH_I_POLY_ORDER     = 4           # Stokes I polynomial order fitted before synthesis, 0-5 (-o)
RMSYNTH_SUPER_RESOLUTION = True        # sharpest RMSF, as per Rudnick & Cotton 2023 (--super-resolution)

# rmclean1d cutoffs: negative = multiples of the FDF noise (sigma), positive = absolute
RMCLEAN_CUTOFF        = -7   # initial CLEAN cutoff (-c)
RMCLEAN_WINDOW_CUTOFF = -3   # deeper CLEAN in windows around the components found first (-w)

# Optional per-epoch cutoff overrides: {(substrings of the file stem,): cutoff}; first match wins,
# e.g. RMCLEAN_CUTOFF_OVERRIDES = {('20240204',): -14}
RMCLEAN_CUTOFF_OVERRIDES        = {}
RMCLEAN_WINDOW_CUTOFF_OVERRIDES = {}


# -----------------------------------------------------------------------------
# Fitting (faraday_obs_fit.py)
# -----------------------------------------------------------------------------
TIE_RM         = False   # share one phi_rm between all S/P components (switched on automatically for all-P models)
SKIP_IF_EXISTS = False   # True: skip fitting and load the existing .pkl if there is one

# Faraday-depth prior range from the RM-CLEAN components (run faraday_synthesis.py first)
USE_RMCLEAN_PHI_RANGE  = True   # derive phi_min/phi_max from the CLEAN components
REQUIRE_RMCLEAN        = True   # exit if no RM-CLEAN file is found, instead of falling back to PHI_MIN/PHI_MAX
RMCLEAN_THRESHOLD_FRAC = 1e-6   # CLEAN components above this fraction of the peak count as real
RMCLEAN_PHI_PADDING    = 10.0   # rad/m² added to each side of the derived range


# -----------------------------------------------------------------------------
# Priors
# -----------------------------------------------------------------------------
P0_MAX = 0.50   # upper bound of the fractional-polarisation prior

# Faraday depth prior range, rad/m². Only used when faraday_obs_fit.py can't
# derive the range from the RM-CLEAN components (USE_RMCLEAN_PHI_RANGE).
PHI_MIN = -500.0
PHI_MAX = 500.0

# Thick-component sigma_phi prior min, rad/m² (~1/6 the L-band RMTF FWHM of ~17 rad/m²).
# Stops a T component shrinking to a point source, breaking its degeneracy with an S component
SIGMA_PHI_MIN = 3.0

# Thick-component sigma_phi prior max, rad/m² (~2x the RMTF FWHM, where you lose
# ~50% sensitivity to thick components)
SIGMA_PHI_MAX = 50.0

# Mixed S/T models: confine the first S component's phi_rm to this window (to break
# the S/T degeneracy), e.g. ('uniform', -20.0, 20.0). None -> full [phi_min, phi_max] range.
MIXED_ST_CONSTRAINED_PHI_RM = None

# psi_0 prior: peaked near PEAKED_ANGLE_DEG (and +90°) when a preferred EVPA is
# expected a priori, otherwise uniform
USE_PEAKED_ANGLE_PRIOR = False
PEAKED_ANGLE_K         = 1.0   # concentration; higher = more peaked
PEAKED_ANGLE_DEG       = 0.0   # preferred psi_0, degrees


# -----------------------------------------------------------------------------
# Thick-component integration grid
# -----------------------------------------------------------------------------
# For N != 2, ThickComponent integrates the super-Gaussian FDF on a phi grid with
# half-width N_SIG * sigma_phi and N_PHI points, where N_SIG(N) and N_PHI(N) =
# a * N**b + c. Defaults: MeerKAT L-band (0.85-1.8 GHz), sigma_phi = 50 and
# phi_peak = 500 rad/m². For another band or wider priors run
# code/calibrate_thick_grid.py and paste the two lines it prints here.
THICK_NSIG_FIT = (19.556, -2.190, 1.316)     # (a, b, c): grid half-width in sigma_phi units
THICK_NPHI_FIT = (7252.291, -4.310, 55.407)  # (a, b, c): grid points (rounded to odd)


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
JET_PA = None   # jet position angle (lower, upper) in deg, e.g. (140.0, 150.0); shown on the
                # psi0 panel of plot_s_components.py (with +90 deg). None hides it
