# Specpol template — QU-fitting + RM synthesis

Source-agnostic template of the broadband QU-fitting and RM synthesis pipeline. Copy this whole directory for each new source, set the source in `code/source_config.py`, drop in the polkat JSON files, and run.

The pipeline decomposes broadband complex polarisation spectra $P(\lambda^2) = Q + iU$ into a sum of phenomenological Faraday components (thin screens, thick screens, power-law screens) using Bayesian nested sampling via [dynesty](https://dynesty.readthedocs.io), then selects the simplest model consistent with the data via the Bayesian evidence. See `EXAMPLE.md` for a worked example of the fitting API.

---

## Starting a new source

```bash
cp -r Specpol_template /path/to/<SOURCE>/Specpol
cd /path/to/<SOURCE>/Specpol
```

1. **Configure** — edit `code/source_config.py`. At minimum set `SOURCE_NAME` (e.g. `'GX_339-4'`). Check `OBSERVATORY`, `CENTRAL_FREQ_GHZ` and `LAMBDA_SQ_REF` if you're not using MeerKAT L-band.
2. **Input spectra** — copy the polkat `*_polarization.json` files into `files/target/polkat_json/`, then
   ```bash
   python3 files/target/convert_json_to_text.py
   ```
   which writes `files/target/QU_text/<SOURCE_NAME>_QU_<YYYYMMDD>[_HHMMSS].txt` (one per epoch; a `_HHMMSS` suffix is added when two epochs share a date, and an `_ejecta` file when polkat found a second component).
3. **Ionosphere** — `python3 code/run_spinifex.py` writes `files/spinifex_iono/spinifex_rm_all.txt` (needs [spinifex](https://github.com/lofar-astron/spinifex) and an internet connection). Skip this and set `IONO_LOC = None` to assume zero ionospheric RM.
4. **RM synthesis** — `python3 code/faraday_synthesis.py`. **Run this before fitting**: its RM-CLEAN components set the Faraday-depth prior range for each epoch (`REQUIRE_RMCLEAN` in `faraday_obs_fit.py`), and it writes the per-epoch ISM RM estimate to `rmsynth/target/target_ism_rm.txt`.
5. **ISM RM prior (optional)** — set `PHI_ISM` in `source_config.py` to the median from `target_ism_rm.txt` and `CONSTRAIN_ISM_RM = True` to pin the first S/P component's RM to ISM + ionosphere.
6. **Flagging** — inspect `plots/target/*/..._flagging_diagnostic.png` and add per-epoch `[[overrides]]` (RFI ranges, clip thresholds) to `code/flagging_config.toml` as needed.
7. **Thick-component grid** — only needed if you fit `T` components. `cd code && python3 calibrate_thick_grid.py` calibrates the Faraday-depth grid used to integrate thick components, for the band of your QU text files (MeerKAT L-band, with a warning, if there are none), using σ_φ = 2 × W_max of the band (W_max: the widest Faraday-thick width the band is sensitive to, Rudnick & Cotton 2023) and the largest |φ| allowed by `PHI_MIN`/`PHI_MAX`. Paste the `THICK_NSIG_FIT` / `THICK_NPHI_FIT` lines it prints into `code/source_config.py`. The defaults are for MeerKAT L-band, σ_φ ≤ 50 and |φ| ≤ 500 rad/m². Each run is self-contained and calibrates from scratch; the L-band W_max is printed alongside your band's for reference.
8. **Fit** — `code/run_all_model.sh` fits every model in `MODELS` to every epoch (slow: days for a full suite; parallelise across epochs on a cluster). Single fits: `cd code && python3 faraday_obs_fit.py <MODEL_TYPE> ../files/target/QU_text/<file>.txt`.
9. **Model selection** — `python3 code/compare_models.py` writes per-epoch summaries to `files/target/summary/`. Put the chosen models in `BEST_MODELS` in `code/run_best_models.sh` to regenerate just those.
10. **Summary plot** — `python3 code/plot_s_components.py` plots the S-component parameters vs time and writes a CSV to `results/target/`.

### Where the source-specific settings live

| File | What to set |
|---|---|
| `code/source_config.py` | Source name and directory, observatory, band centre, iono/ISM RM constraint, RM synthesis / CLEAN settings, fit options (tied RM, RM-CLEAN prior range), prior ranges, preferred EVPA, jet PA, thick-component grid coefficients. **Shared by every script.** |
| `code/flagging_config.toml` | Default clipping + per-epoch `[[overrides]]` (matched on substrings of the file stem) |
| `code/run_all_model.sh` | `MODELS` fitted to every epoch, optional `EXTRA_MODELS` per epoch |
| `code/run_best_models.sh` | `DEFAULT_MODELS` and per-epoch `BEST_MODELS` |
| `code/faraday_obs_fit.py` | Sampler/run settings (`N_CORES`, `NLIVE_FACTOR`, `SAMPLER`) — usually left alone |

---

## The five core modules (`code/`)

Library modules imported by everything else; not run directly.

| Module | Role |
|---|---|
| `faraday_utils.py` | Physical model components and spectral utilities: thin screens, power-law screens, thick screens (super-Gaussian, following Anderson et al. 2016), RM synthesis, RMTF, polarisation helpers, spinifex iono RM loading. |
| `faraday_data.py` | Data I/O and Stokes I fitting: loading QU text files, channel flagging, power-law and polynomial Stokes I fits, validation. |
| `faraday_model.py` | Bayesian fitting engine: dynesty wrapper, likelihood, prior transforms (Cartesian ↔ polar, RM ordering), result storage. |
| `faraday_plot.py` | Fit panels, corner plots, convergence diagnostics, FDF diagnostics, summary figures. |
| `faraday_processing.py` | Posterior post-processing: KDE mode detection, circular/axial angle handling, 68% HDIs per mode, $(a, b) \to (p_0, \psi_0)$. |

## Scripts (`code/`)

| Script | Role |
|---|---|
| `source_config.py` | Source-specific settings (see above). |
| `faraday_obs_fit.py` | Main fitting script: `python3 faraday_obs_fit.py <MODEL_TYPE> <QU_text_file>`. `MODEL_TYPE` is a string of `S` (thin), `T` (thick), `P` (power-law) components, e.g. `S`, `ST`, `SST`, `PP`. |
| `faraday_synthesis.py` | RM synthesis + RM-CLEAN on every epoch with [RM-Tools](https://github.com/CIRADA-Tools/RM-Tools) `rmsynth1d`/`rmclean1d`; outputs to `rmsynth/target/`. Optional args override the target directories. |
| `run_spinifex.py` | Ionospheric RMs from spinifex in a window around each epoch (position and time read from the QU text headers). |
| `compare_models.py` | Evidence-based model selection: simplest model within $\Delta \ln Z \leq 10$ of the best. |
| `plot_s_components.py` | Time series of S-component parameters (iono-corrected RM) for one model type. |
| `calibrate_thick_grid.py` | Calibrates the thick-component integration grid for your band and priors (step 7); prints `THICK_NSIG_FIT` / `THICK_NPHI_FIT` for `source_config.py` and writes `results/` and `plots/thick_grid_calibration/`. |
| `faraday_example_simulate.py` | Simulates a dataset from a user-specified model and fits it with the same pipeline (validation, prior tests). Uses the channels of your QU text files (all epochs combined), or a set frequency range if there are none. Outputs to `simulated/`. |
| `run_all_model.sh`, `run_best_models.sh` | Batch runners (above). |

---

## Directory structure

```
Specpol_template/
├── code/                   ← modules, scripts, source_config.py, flagging_config.toml
├── files/
│   ├── target/
│   │   ├── convert_json_to_text.py
│   │   ├── polkat_json/    ← polkat JSON outputs (input)
│   │   ├── QU_text/        ← per-epoch QU spectra (fitting input)
│   │   └── summary/        ← model-selection summaries (compare_models.py)
│   └── spinifex_iono/      ← ionospheric RMs (run_spinifex.py)
├── results/target/         ← per-epoch nested sampling outputs (JSON + pkl)
├── plots/target/           ← per-epoch diagnostic plots
├── rmsynth/target/         ← RM synthesis / CLEAN outputs + target_ism_rm.txt
└── simulated/              ← faraday_example_simulate.py outputs
```

The `target` subdirectory name is `SOURCE_DIR` in `source_config.py`; there's no need to rename it per source, since each source gets its own copy of the template. The empty directories hold a `.gitkeep`.

---

## Dependencies

A pinned environment is in `requirements.txt` (Python 3.12.3):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Not covered by `requirements.txt`:

- **[polkat](https://github.com/AKHughes1994/polkat)** — MeerKAT polarisation calibration and QU extraction pipeline that produces the input JSON files. Only needed to process raw visibilities.
- **[spinifex](https://github.com/lofar-astron/spinifex)** — only needed for `run_spinifex.py`.
- **RM-Tools CLI** — `RM-Tools` is in `requirements.txt`, but make sure `rmsynth1d` and `rmclean1d` are on your `PATH`.

## Caveat

The default thick-component integration grid (`THICK_NSIG_FIT`, `THICK_NPHI_FIT`) was calibrated for MeerKAT L-band (0.85–1.8 GHz), σ_φ ≤ 50 and |φ| ≤ 500 rad/m². For another band or wider priors, re-run `calibrate_thick_grid.py` (step 7) before trusting thick-component results (see `EXAMPLE.md` §8.2).
