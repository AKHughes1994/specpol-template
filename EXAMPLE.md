# QU-Fitting: Worked Example

A step-by-step guide to loading data, building a model, specifying priors, and running a Bayesian nested sampling fit. This document is intended as a practical reference; a more detailed treatment of the formalism, flagging options, post-processing, and model comparison is in [PLACEHOLDER — fitting pipeline guide].

All examples assume you are running from `code/` and that the five core modules (`faraday_data`, `faraday_utils`, `faraday_model`, `faraday_plot`, `faraday_processing`) are importable from the same directory.

---

## 1. Loading Data

Data files are plain text with columns `freq[Hz]  I  Q  U  dI  dQ  dU` in physical flux units (e.g. mJy — the units do not matter as long as they are consistent, since the loader fits Stokes I and divides Q and U by it to produce a fractional polarisation spectrum).

```python
from faraday_data import DataLoader

data, I_fitter = DataLoader.load_physical_with_fit(
    '../files/target/QU_text/<SOURCE_NAME>_QU_<YYYYMMDD>.txt',
    I_model='power-law',   # fit Stokes I with a power-law I ∝ λ^α
    poly_degree=4,         # only used if I_model='polynomial'
    central_freq=1.28,     # reference frequency in GHz (CENTRAL_FREQ_GHZ in source_config.py)
)
```

`data` is a `PolarizationData` object containing fractional `Q`, `U` and their propagated errors. `I_fitter` holds the Stokes I fit parameters.

If you want to pass a channel flagging configuration, see `FlaggingConfig` in `faraday_data.py` and `flagging_config.toml` for the per-epoch overrides — this is handled automatically by `faraday_obs_fit.py` and you normally do not need to configure it by hand.

---

## 2. Building a Model

A `CompositeModel` is a sum of one or more Faraday components. Three component types are available:

| Class | Symbol | Description |
|---|---|---|
| `ThinComponent` | `S` | Thin Faraday screen. Two free amplitudes `(a, b)` and a rotation measure `φ_rm`. |
| `ThickComponent` | `T` | Thick screen / internal Faraday dispersion (super-Gaussian FDF). Amplitudes `(a, b)`, peak depth `φ_peak`, dispersion `σ_φ`, and number of cells `N`. |
| `ThinPowerLawComponent` | `P` | Thin screen with wavelength-dependent depolarisation. Adds spectral index `β` and a reference wavelength `λ²_0`. |

The initial parameter values passed to each component are arbitrary when fitting real data — dynesty samples the full prior volume from scratch and ignores them entirely. The names you assign (`S1`, `T1`, `P1`, etc.) are used as keys in the prior dictionary.

```python
from faraday_utils import CompositeModel, ThinComponent, ThickComponent, ThinPowerLawComponent

model = CompositeModel()

# Add a thin Faraday screen (e.g., the ISM foreground)
model.add_component(ThinComponent(a=0.1, b=0.0, phi_rm=0.0, name='S1'))

# Add a thick screen (e.g., emission from within the jet)
model.add_component(ThickComponent(a=0.1, b=0.0, phi_peak=0.0,
                                   sigma_phi=1.0, N=2.0, name='T1'))
```

For a pure power-law model:

```python
model = CompositeModel()
model.add_component(ThinPowerLawComponent(a=0.1, b=0.0, phi_rm=0.0,
                                          beta=0.0, lambda_sq_0=None, name='P1'))  # None → FitSetup uses lambda_sq_ref
```

---

## 3. Defining Priors

Priors are passed as a plain Python dictionary. The top-level keys are the component names you assigned above. Each component entry is itself a dict specifying the prior on each free parameter.

### 3.1 Available prior types

Scalar parameters (`phi_rm`, `phi_peak`, `sigma_phi`, `N`, `beta`, `a`, `b`) accept a tuple:

| Syntax | Description |
|---|---|
| `('uniform', min, max)` | Uniform between `min` and `max` |
| `('log_uniform', min, max)` | Log-uniform (Jeffreys prior); both bounds must be > 0 |
| `('gaussian', mean, std)` | Unbounded Gaussian |
| `('truncated_gaussian', mean, std, min, max)` | Gaussian truncated to `[min, max]` |
| `('fixed', value)` | Fix parameter to `value` (removes it from the fit entirely) |

### 3.2 Amplitude priors via `polar_conversion`

Rather than placing priors directly on the Cartesian amplitudes `(a, b)`, it is far more natural to work in polar coordinates `(p0, ψ₀)` where `p0 = √(a²+b²)` is the fractional polarisation amplitude and `ψ₀` is the intrinsic EVPA. This is controlled by the `polar_conversion` sub-dict:

```python
'polar_conversion': {
    'p0_bounds':   (p0_min, p0_max, p0_distribution),
    'psi_0_bounds': (psi_min, psi_max, angle_prior),
}
```

**`p0_distribution`** options:

| Value | Description |
|---|---|
| `'radial_uniform'` | Uniform in `p0` — i.e., `p(p0) ∝ 1`. Recommended: physically motivated and avoids the ring bias of uniform Cartesian priors. |
| `'area_uniform'` | Uniform in area of the Poincaré disc — i.e., `p(p0) ∝ p0`. Equivalent to uniform Cartesian `(a, b)`. |
| `'log_uniform'` | Log-uniform in `p0`; useful if amplitudes span orders of magnitude. |

**`angle_prior`** options:

| Value | Description |
|---|---|
| `'uniform'` | Flat over `[-π/2, π/2]`. Use for sources where you have no prior knowledge of the EVPA. |
| `'peaked'` | Weakly informative prior that peaks at `peaked_angle_deg` (default 0°) and its orthogonal direction (90° away), with concentration `peaked_k` (default 1.0). Recommended for sources where the intrinsic EVPA is unlikely to be wildly large; reduces the sampler's tendency to explore unphysical EVPA solutions. |

### 3.3 Recommended priors (the faraday_obs_fit.py defaults)

These are taken directly from `faraday_obs_fit.py` and reflect the priors actually used to produce the published results. The `phi_rm` / `phi_peak` range `[phi_min, phi_max]` is derived automatically from the RMclean output for each epoch (default fallback `±160 rad/m²`).

```python
import numpy as np

phi_min = -160.0   # rad/m² — set per-epoch from RMclean output in practice
phi_max =  160.0

THIN_PRIOR = {
    'type': 'ThinComponent',
    'polar_conversion': {
        'p0_bounds':    (0.0, 0.50, 'radial_uniform'),
        'psi_0_bounds': (-np.pi/2, np.pi/2, 'peaked'),
        'peaked_k':          1.0,   # concentration parameter
        'peaked_angle_deg':  0.0,   # peak direction (degrees)
    },
    'phi_rm': ('uniform', phi_min, phi_max),
}

THICK_PRIOR = {
    'type': 'ThickComponent',
    'polar_conversion': {
        'p0_bounds':    (0.0, 0.50, 'radial_uniform'),
        'psi_0_bounds': (-np.pi/2, np.pi/2, 'peaked'),
    },
    'phi_peak':  ('uniform',     phi_min, phi_max),
    'sigma_phi': ('log_uniform', 0.1,     50.0),    # rad/m²; upper bound ~ 2× RMTF FWHM
    'N':         ('log_uniform', 2.0,     50.0),    # number of Faraday cells
    'effective_width_mode': False,
}

POWERLAW_PRIOR = {
    'type': 'ThinPowerLawComponent',
    'polar_conversion': {
        'p0_bounds':    (0.0, 0.50, 'radial_uniform'),
        'psi_0_bounds': (-np.pi/2, np.pi/2, 'peaked'),
    },
    'phi_rm': ('uniform', phi_min, phi_max),
    'beta':   ('uniform', -5.0,    5.0),
}
```

### 3.4 ISM RM constraint

For models containing a thin screen, the first `S` component is constrained to the expected Galactic ISM foreground RM using a truncated Gaussian. The values below are illustrative; in the pipeline they are set in `code/source_config.py` (`CONSTRAIN_ISM_RM`, `PHI_ISM`, `PHI_SYS`, `PHI_ISM_SIGMA`, `PHI_ISM_RANGE`), with `PHI_ISM` taken from the median written by `faraday_synthesis.py` to `rmsynth/target/target_ism_rm.txt`.

```python
PHI_ISM   = -0.5   # Galactic ISM Faraday depth (rad/m²)
PHI_SYS   = -0.3   # Systematic offset of the spinifex iono RM (rad/m²): true iono = iono_rm - PHI_SYS
iono_rm   =  0.0   # Ionospheric contribution from Spinifex; set per-epoch

phi_center = PHI_ISM + (iono_rm - PHI_SYS)

ism_phi_prior = ('truncated_gaussian',
                 phi_center,          # mean
                 0.65,                # std (rad/m²)
                 phi_center - 2.3,    # lower truncation (≈ 3.5σ)
                 phi_center + 2.3)    # upper truncation

priors['S1']['phi_rm'] = ism_phi_prior
```

### 3.5 Tying `phi_rm` across components

For all-power-law models (`PP`, `PPP`, …), it is often appropriate to tie the rotation measure across components — i.e., all components share a single `φ_rm` with separate amplitude and spectral index. Enable this via:

```python
priors = {'tied_phi_rm': ('uniform', phi_min, phi_max)}
# ... then add each component prior WITHOUT a 'phi_rm' key
```

and pass `tie_phi_rm=True` to `FitSetup` (see below). `faraday_obs_fit.py` auto-enables this for any model consisting entirely of `P` components.

### 3.6 Assembling the prior dict

```python
priors = {}
priors['S1'] = THIN_PRIOR.copy()
priors['T1'] = THICK_PRIOR.copy()

# Optionally override one parameter after the fact:
priors['S1']['phi_rm'] = ism_phi_prior
```

---

## 4. FitSetup

`FitSetup` validates that the model, priors, and data are mutually consistent and pre-computes the prior transform used by dynesty.

```python
from faraday_model import FitSetup

# lambda_sq_ref: the λ² that psi_0 is referenced to while sampling (and where a
# power-law p0 is quoted). Using the band centre is standard; here ≈ 1.28 GHz.
lambda_sq_ref = (0.2998 / 1.28) ** 2   # ≈ 0.0549 m²

setup = FitSetup(
    model=model,
    priors=priors,
    data=data,
    enforce_ordering=True,   # enforce φ_rm ordering between same-type components
    tie_phi_rm=False,        # set True for all-P models
    lambda_sq_ref=lambda_sq_ref,
    p0_physical_max=0.7,     # upper bound on each component's and the total fractional polarisation
)
```

`enforce_ordering=True` (recommended) imposes `φ_rm(S1) ≤ φ_rm(S2) ≤ …` (and similarly for `T`, `P` components) to break the label-switching degeneracy between interchangeable components — those of the same type whose priors are identical in every parameter. It is implemented exactly via order statistics in the prior transform, so there is no likelihood penalty. A component with its own prior (e.g. `S1` pinned to the ISM RM) is already identified by that prior and is left out of the ordering, so the other components can sit on either side of it.

---

## 5. Running the Fit

`FaradayFitter` wraps dynesty. The `dynesty_kwargs` argument is **mandatory** — no defaults are provided to ensure you are always explicit about the sampler settings.

### 5.1 Parallelisation

By default (`use_pool=True`, `n_cores=None`), the fitter uses all available CPUs minus two. Override with `n_cores=N` if needed.

### 5.2 Static sampler (recommended for most single models)

```python
from faraday_model import FaradayFitter

dynesty_kwargs = {
    'sampler': {
        'nlive':  1000,         # number of live points
        'bound':  'multi',      # bounding method ('multi' = multi-ellipsoid)
        'sample': 'rwalk',      # sampling method
        'walks':  150,          # steps per random walk
    },
    'run': {
        'dlogz':   0.1,         # stopping criterion on log-evidence
        'maxiter': None,        # no iteration cap
        'maxcall': None,        # no likelihood call cap
    },
}

fitter = FaradayFitter(setup, sampler='static', dynesty_kwargs=dynesty_kwargs)
results = fitter.fit()
```

### 5.3 Dynamic sampler (the faraday_obs_fit.py default)

The dynamic sampler adaptively allocates live points to maximise posterior accuracy. It is slower per iteration but more efficient for complex posteriors. This is what `faraday_obs_fit.py` uses:

```python
dynesty_kwargs = {
    'sampler': {
        'bound':  'multi',
        'sample': 'rwalk',
        'walks':  150,
    },
    'run': {
        'dlogz_init':   0.01,
        'nlive_init':   450,      # = N_LIVE / 2
        'nlive_batch':  150,      # = N_LIVE / 6
        'n_effective':  10_000,
        'maxiter':      None,
        'maxcall':      None,
    },
}

fitter = FaradayFitter(setup, sampler='dynamic', dynesty_kwargs=dynesty_kwargs,
                       n_cores=15)
results = fitter.fit()
```

### 5.4 Saving results

```python
results.save('my_results.pkl')           # full pickle (samples, weights, etc.)
results.save_json('my_results.json')     # human-readable summary
```

---

## 6. Complete Worked Example

A self-contained script for a single epoch, fitting a thin screen + thick screen (`ST`) model:

```python
import numpy as np
from faraday_data import DataLoader
from faraday_utils import CompositeModel, ThinComponent, ThickComponent
from faraday_model import FitSetup, FaradayFitter
from faraday_processing import process_posterior_modes, ProcessingConfig
from faraday_plot import plot_fit_results, plot_corner
from pathlib import Path

# -------------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------------
data_file  = '../files/target/QU_text/<SOURCE_NAME>_QU_<YYYYMMDD>.txt'
output_dir = Path('../results/target/example_ST')
output_dir.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------------
data, I_fitter = DataLoader.load_physical_with_fit(
    data_file,
    I_model='power-law',
    central_freq=1.28,
)

# -------------------------------------------------------------------------
# Build model
# -------------------------------------------------------------------------
model = CompositeModel()
model.add_component(ThinComponent(a=0.1, b=0.0, phi_rm=0.0,   name='S1'))
model.add_component(ThickComponent(a=0.1, b=0.0, phi_peak=0.0,
                                   sigma_phi=1.0, N=2.0,       name='T1'))

# -------------------------------------------------------------------------
# Priors
# -------------------------------------------------------------------------
phi_min, phi_max = -160.0, 160.0
lambda_sq_ref    = (0.2998 / 1.28) ** 2

priors = {
    'S1': {
        'type': 'ThinComponent',
        'polar_conversion': {
            'p0_bounds':    (0.0, 0.50, 'radial_uniform'),
            'psi_0_bounds': (-np.pi/2, np.pi/2, 'peaked'),
        },
        # ISM RM constraint: truncated Gaussian around known foreground
        'phi_rm': ('truncated_gaussian', -0.8, 0.65, -3.1, 1.5),
    },
    'T1': {
        'type': 'ThickComponent',
        'polar_conversion': {
            'p0_bounds':    (0.0, 0.50, 'radial_uniform'),
            'psi_0_bounds': (-np.pi/2, np.pi/2, 'peaked'),
        },
        'phi_peak':  ('uniform',     phi_min, phi_max),
        'sigma_phi': ('log_uniform', 0.1,     50.0),
        'N':         ('log_uniform', 2.0,     50.0),
        'effective_width_mode': False,
    },
}

# -------------------------------------------------------------------------
# Setup + fit
# -------------------------------------------------------------------------
setup = FitSetup(model, priors, data,
                 enforce_ordering=True,
                 lambda_sq_ref=lambda_sq_ref)

dynesty_kwargs = {
    'sampler': {'nlive': 1000, 'bound': 'multi', 'sample': 'rwalk', 'walks': 150},
    'run':     {'dlogz': 0.1, 'maxiter': None, 'maxcall': None},
}
fitter  = FaradayFitter(setup, sampler='static', dynesty_kwargs=dynesty_kwargs)
results = fitter.fit()

# -------------------------------------------------------------------------
# Post-process and save
# -------------------------------------------------------------------------
pconfig = ProcessingConfig(
    filename_prefix='example_ST',
    detect_multiple_modes=True,
    plot_diagnostics=True,
    plot_dir=output_dir / 'modes',
)
results = process_posterior_modes(results, pconfig)
results.save(output_dir / 'example_ST.pkl')
results.save_json(output_dir / 'example_ST.json')
results.print_summary()

# -------------------------------------------------------------------------
# Plots
# -------------------------------------------------------------------------
plot_fit_results(results, I_fitter=I_fitter,
                 filename='example_ST_fit.png', output_dir=output_dir)
plot_corner(samples=results.samples_processed,
            param_names=results.param_names_processed,
            param_summary=results.param_summary,
            processed_parameters=results.processed_parameters,
            weights=results.weights,
            setup=results.setup,
            filename='example_ST_corner.png', output_dir=output_dir)
```

For running every model type across every epoch (for full model comparison), see `run_all_model.sh`. To regenerate only the favoured models and their posterior samples, use `run_best_models.sh`.

---

## 7. Simulating Data

The pipeline includes a self-contained simulation script that generates a synthetic polarised dataset from a user-defined model, fits it, and produces the full suite of diagnostic plots. This is useful for validating the fitting machinery, testing prior sensitivity, and producing demonstration figures.

The script is `code/faraday_example_simulate.py`. All configuration lives at the top of the file — you do not need to touch anything below the configuration block to run a simulation.

### 7.1 Defining the true model

Components are specified as lists of parameter dictionaries. Add, remove, or comment out entries to change the model:

```python
# Thin screen components (S1, S2, ...)
THIN_COMPONENTS = [
    {'p0': 0.02, 'psi_0_deg': -2.6, 'phi_rm': -5.76},   # S1
]

# Power-law components (P1, P2, ...)
POWERLAW_COMPONENTS = []  # empty = no power-law components

# Thick screen components (T1, T2, ...)
THICK_COMPONENTS = [
    {'p0': 0.05, 'psi_0_deg': -35.0, 'phi_peak': 53.0, 'sigma_phi': 35.0, 'N': 40},  # T1
]
```

### 7.2 Choosing what model to fit

```python
MODEL_TO_FIT = None   # None = fit the exact same model as the truth
                      # or set e.g. 'ST', 'S', 'T' to fit a different model
```

Setting `MODEL_TO_FIT` to something other than `None` lets you test whether the sampler can recover the correct model even when fitted with a misspecified one.

### 7.3 Other key settings

```python
QUICK_MODE = False   # True = skip corner/convergence/FDF plots (faster for quick checks)
```

By default the simulation uses every frequency channel in your QU text files (`files/target/QU_text/*.txt`, all epochs combined), so it matches your band:

```python
USE_DATA_CHANNELS = True   # False = always use FREQ_RANGE / N_CHANNELS below
```

If there are no QU text files (or `USE_DATA_CHANNELS = False`), it falls back to `FREQ_RANGE` / `N_CHANNELS`. These, and the other data generation parameters (SNR, spectral index), are set inside `run_test_scenario()` at the clearly marked `SECTION 2` block.

### 7.4 Running the script

```bash
cd code/
python faraday_example_simulate.py
```

All outputs (synthetic data file, fit results, plots) are written to `../simulated/`.

---

## 8. Caveats and Implementation Notes

### 8.1 The reference wavelength `lambda_sq_ref`

The Cartesian parameters `(a, b)` are each component's own complex polarisation amplitude — the intrinsic amplitude (before any Faraday rotation or depolarisation) for thin and thick components, and the amplitude at `λ²_0` for power-law components. The full model is:

$$P(\lambda^2) = (a + ib) \times K(\lambda^2)$$

where $K$ encodes the Faraday rotation and depolarisation of each component type.

When `lambda_sq_ref` is set in `FitSetup`, the sampler instead works with the amplitude referenced to the reference band by the Faraday rotation alone:

$$a_\text{ref} + ib_\text{ref} = (a + ib)\, e^{2i\phi\lambda^2_\text{ref}}$$

with $\phi = \phi_\text{rm}$ for thin and power-law components and $\phi = \phi_\text{peak}$ for thick components. The amplitude is recovered internally as $a + ib = (a_\text{ref} + ib_\text{ref})\, e^{-2i\phi\lambda^2_\text{ref}}$.

Because the rotation has unit modulus, `p0` and its prior are unchanged; only `ψ₀` is moved to the band centre, which removes the strong `ψ₀`–`φ` correlation during sampling. The forward model is not affected. Reported values are converted back (`ψ₀` at λ² = 0; power-law `p0` at `λ²_0`).

Earlier versions divided by the full transfer factor $K(\lambda^2_\text{ref})$ instead, which put the `p0` prior of thick (and of power-law, when `λ²_0 ≠ λ²_ref`) components on the amplitude observed at the reference band. For thick components with $N > 2$, $K(\lambda^2_\text{ref})$ passes through zero (near $\sigma_\phi \approx 30$ rad m$^{-2}$ for $\lambda^2_\text{ref} = 0.05$ m²) while the in-band response is still large, which left holes in the $(\sigma_\phi, N)$ posterior; for power laws it made the effective `p0` bound depend on `β`. `load_results` warns when it loads results made that way.

Set `lambda_sq_ref` to the band-centre wavelength squared — for MeerKAT L-band centred at 1.28 GHz this is `(0.2998 / 1.28)² ≈ 0.055 m²`.

### 8.2 Thick component numerical solver

For $N \neq 2$ (i.e., non-Gaussian FDFs), $P(\lambda^2)$ is computed by numerically Fourier-transforming the super-Gaussian FDF over a finite Faraday depth grid. The grid half-width (in units of $\sigma_\phi$) and point count follow power laws in $N$, $f(N) = a N^b + c$, whose coefficients are config variables in `source_config.py`:

```python
THICK_NSIG_FIT = (19.556, -2.190, 1.316)     # (a, b, c): grid half-width in σ_φ units
THICK_NPHI_FIT = (7252.291, -4.310, 55.407)  # (a, b, c): grid points (rounded to odd)
```

The defaults were calibrated for MeerKAT L-band (0.85–1.8 GHz) with $\sigma_\phi = 50$ and $\phi_\text{peak} = 500\ \text{rad/m}^2$. Wider components need a wider grid, and the integrand oscillates faster at larger Faraday depths and longer wavelengths, so for a different band or wider priors the default grid may be too coarse. Recalibrate with

```bash
cd code && python3 calibrate_thick_grid.py
```

For each $N$ it compares a sweep of grid sizes against a very fine reference integration, using the channels of your QU text files (MeerKAT L-band, with a warning, if there are none). The test component has $\sigma_\phi = 2 W_\text{max}$, twice the widest Gaussian Faraday-thick FWHM the band is sensitive to, $W_\text{max} \approx 0.67(\lambda_\text{min}^{-2} + \lambda_\text{max}^{-2})$ (Rudnick & Cotton 2023; $2 W_\text{max} \approx 59\ \text{rad/m}^2$ for MeerKAT L-band), and $\phi_\text{peak}$ = max(|`PHI_MIN`|, |`PHI_MAX`|). It keeps the cheapest grid with flux-weighted error $\sum|\Delta P| / \sum|P| < 10^{-6}$, fits the two power laws, checks the fitted grid, and prints the `THICK_NSIG_FIT` / `THICK_NPHI_FIT` lines to paste into `source_config.py`.
