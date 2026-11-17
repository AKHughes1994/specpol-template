#!/usr/bin/env python3
"""
Run spinifex for every epoch found in ../files/*/QU_text/*.txt and write one combined
file of ionospheric RMs, sampled in a time window around each epoch.

The source position and epoch time are read from the '#' header of each QU text
file (written by convert_json_to_text.py).
"""

import re
import tempfile
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time
from spinifex import get_rm

from source_config import BASE_DIR, OBSERVATORY, IONO_HALF_WINDOW_MIN

# =============================================================================
# CONFIGURATION
# =============================================================================
# Observatory location and HALF_WINDOW_MIN are set in source_config.py
FILES_DIR   = BASE_DIR / 'files'
OUTPUT_FILE = FILES_DIR / 'spinifex_iono' / 'spinifex_rm_all.txt'   # faraday_obs_fit.py reads every *.txt in spinifex_iono/

HALF_WINDOW_MIN = IONO_HALF_WINDOW_MIN   # RMs are evaluated from (epoch - this) to (epoch + this), minutes
STEP_MIN        = 1    # spacing of the evaluation times, minutes
IONO_MODEL      = 'ionex_iri'


def read_header(path):
    """Return (source_name, epoch Time, SkyCoord) from the '#' header of a QU text file."""
    text = ''
    with open(path) as f:
        for line in f:
            if not line.startswith('#'):
                break
            text += line

    source = re.search(r'#\s*(\S+) QU data', text).group(1)
    date   = re.search(r'Date \(ISO\):\s*(\S+)', text).group(1)
    ra, dec = re.search(r'RA, Dec \(deg\):\s*(\S+)\s+(\S+)', text).groups()

    epoch = Time(date, format='isot', scale='utc')
    coord = SkyCoord(ra=float(ra) * u.deg, dec=float(dec) * u.deg)
    return source, epoch, coord


def main():

    # Every QU text file in every <source>/QU_text subdirectory
    qu_files = sorted(FILES_DIR.glob('*/QU_text/*.txt'))
    print(f"Found {len(qu_files)} QU text files")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Temporary directory for the downloaded ionosphere/GPS data; deleted on exit
    with tempfile.TemporaryDirectory() as cache_dir, open(OUTPUT_FILE, 'w') as out:

        out.write("# Source RA_hms Dec_dms Epoch Time_ISOT Time_MJD RM_rad/m2 RM_err_rad/m2\n")

        for qu_file in qu_files:
            print(f"Processing {qu_file.name}")

            # Skip files whose header lacks a date or position (e.g. made by an older converter)
            try:
                source, epoch, coord = read_header(qu_file)
            except AttributeError:
                print("  Warning: header missing source/date/position, skipping")
                continue

            # Times around the epoch, e.g. -30 ... +30 min in 1 min steps
            offsets = np.arange(-HALF_WINDOW_MIN, HALF_WINDOW_MIN + 1, STEP_MIN) * u.min
            times = epoch + offsets

            # Run spinifex for this position and these times
            try:
                rm = get_rm.get_rm_from_skycoord(
                    loc=OBSERVATORY,
                    times=times,
                    output_directory=cache_dir,
                    source=coord,
                    iono_model_name=IONO_MODEL,
                )
            except Exception as e:
                print(f"  Warning: spinifex failed: {e}")
                continue

            # One row per time; the epoch column (file stem) identifies which observation it belongs to
            ra_hms  = coord.ra.to_string(unit=u.hour, sep=':', pad=True)
            dec_dms = coord.dec.to_string(unit=u.deg, sep=':', pad=True, alwayssign=True)
            for t, myrm, rm_err in zip(rm.times, rm.rm, rm.rm_error):
                out.write(f"{source} {ra_hms} {dec_dms} {qu_file.stem} {t.isot} {t.mjd} {myrm} {rm_err}\n")

    print(f"\nResults written to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
