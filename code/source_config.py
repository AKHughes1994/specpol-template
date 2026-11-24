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
# the S/T degeneracy). None -> full [phi_min, phi_max] range.
MIXED_ST_CONSTRAINED_PHI_RM = ('uniform', -20.0, 20.0)

# psi_0 prior: peaked near PEAKED_ANGLE_DEG (and +90°) when a preferred EVPA is
# expected a priori, otherwise uniform
USE_PEAKED_ANGLE_PRIOR = False
PEAKED_ANGLE_K         = 1.0   # concentration; higher = more peaked
PEAKED_ANGLE_DEG       = 0.0   # preferred psi_0, degrees


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
JET_PA = None   # jet position angle (lower, upper) in deg, e.g. (140.0, 150.0); shown on the
                # psi0 panel of plot_s_components.py (with +90 deg). None hides it
