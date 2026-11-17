"""
Simple script to convert the JSON output from polkat 
to text files compatible with the QU modelling code 
"""

import sys
import json
import glob
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from pathlib import Path

# =============================================================================
# CONFIGURATION
# SOURCE_NAME (the output filename/header prefix) is set in code/source_config.py.
# Paths are relative to this script, so it can be run from anywhere. Layout:
#   files/<SOURCE_DIR>/polkat_json/   <- input polkat JSON files
#   files/<SOURCE_DIR>/QU_text/       <- output text files
# =============================================================================
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / 'code'))
from source_config import SOURCE_NAME

INPUT_DIR  = HERE / 'polkat_json'
OUTPUT_DIR = HERE / 'QU_text'


def epoch_isot(mfs_component):
    """Epoch time of an MFS component; older polkat versions call the key 'time_ctr_isot'."""
    key = 'date_isot' if 'date_isot' in mfs_component else 'time_ctr_isot'
    return mfs_component[key][0]


def mfs_header(mfs_component):
    """Header lines with the Stokes I source position (degrees and hmsdms) and MFS flux density."""
    ra_deg = float(mfs_component['I_RA_deg'][0]) % 360.0   # polkat stores RA as negative degrees; wrap to [0, 360)
    dec_deg = float(mfs_component['I_DEC_deg'][0])
    coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
    return (f"# RA, Dec (deg): {ra_deg:.6f} {dec_deg:.6f}\n"
            f"# RA, Dec (hmsdms): {coord.to_string('hmsdms', precision=3)}\n"
            f"# MFS Stokes I (mJy): {float(mfs_component['I_flux_mJy'][0]):.4f} +/- "
            f"{float(mfs_component['I_err_mJy'][0]):.4f}\n")


def main():

    if not SOURCE_NAME:
        sys.exit("Please set SOURCE_NAME in code/source_config.py.")

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Construct sorted list of files
    json_files = sorted(glob.glob(f"{INPUT_DIR}/*.json"))
    
    # Several observations can fall on the same UTC date; those get a _HHMMSS
    # suffix so that they don't overwrite each other.
    dates = []
    for json_file in json_files:
        with open(json_file, 'r') as f:
            dates.append(epoch_isot(json.load(f)['MFS']['component0']).split('T')[0])

    # Iterate over files
    for json_file, date in zip(json_files, dates):

        print(f"Processing file: {json_file}")

        # Open json file
        with open(json_file, 'r') as f:
            data = json.load(f)

        # Take only the channelised array and the time
        time_ctr_isot = epoch_isot(data['MFS']['component0'])
        
        # Process component0
        data_array = data['CHAN']['component0']

        # Numpize the data array
        for key in data_array:
            data_array[key] = np.array(data_array[key])

        # Extract the data of interest
        text_array = [
            data_array['freq_GHz'][0] * 1e9,
            data_array['I_flux_mJy'][0] / 1e3,
            data_array['Q_flux_mJy'][0] / 1e3,
            data_array['U_flux_mJy'][0] / 1e3,
            data_array['V_flux_mJy'][0] / 1e3,
            data_array['I_rms_mJy'][0] / 1e3,
            data_array['Q_rms_mJy'][0] / 1e3,
            data_array['U_rms_mJy'][0] / 1e3,
            data_array['V_rms_mJy'][0] / 1e3
        ]

        # Save array to text file where each row is a channel, split at the T
        stamp = date.replace('-', '')
        if dates.count(date) > 1:
            stamp += '_' + time_ctr_isot.split('T')[1][:8].replace(':', '')
        output_file = f"{OUTPUT_DIR}/{SOURCE_NAME}_QU_{stamp}.txt"
        header = (f"# {SOURCE_NAME} QU data converted from polkat JSON output\n"
                  f"# Date (ISO): {time_ctr_isot}\n"
                  f"{mfs_header(data['MFS']['component0'])}"
                  f"# Columns: freq(Hz) I(Jy) Q(Jy) U(Jy) V(Jy) dI(Jy) dQ(Jy) dU(Jy) dV(Jy)\n")
        np.savetxt(output_file, np.column_stack(text_array), header=header, fmt='%.8e')
        
        # Check if component1 exists and process it for ejecta
        if 'component1' in data['CHAN']:
            print(f"  Found component1, creating ejecta file...")
            data_array_ejecta = data['CHAN']['component1']
            
            # Numpize the data array
            for key in data_array_ejecta:
                data_array_ejecta[key] = np.array(data_array_ejecta[key])
            
            # Extract the data of interest
            text_array_ejecta = [
                data_array_ejecta['freq_GHz'][0] * 1e9,
                data_array_ejecta['I_flux_mJy'][0] / 1e3,
                data_array_ejecta['Q_flux_mJy'][0] / 1e3,
                data_array_ejecta['U_flux_mJy'][0] / 1e3,
                data_array_ejecta['V_flux_mJy'][0] / 1e3,
                data_array_ejecta['I_rms_mJy'][0] / 1e3,
                data_array_ejecta['Q_rms_mJy'][0] / 1e3,
                data_array_ejecta['U_rms_mJy'][0] / 1e3,
                data_array_ejecta['V_rms_mJy'][0] / 1e3
            ]
            
            # Save ejecta array to text file
            output_file_ejecta = f"{OUTPUT_DIR}/{SOURCE_NAME}_QU_{stamp}_ejecta.txt"
            header_ejecta = (f"# {SOURCE_NAME} QU data (ejecta/component1) converted from polkat JSON output\n"
                            f"# Date (ISO): {time_ctr_isot}\n"
                            f"{mfs_header(data['MFS'].get('component1', data['MFS']['component0']))}"
                            f"# Columns: freq(Hz) I(Jy) Q(Jy) U(Jy) V(Jy) dI(Jy) dQ(Jy) dU(Jy) dV(Jy)\n")
            np.savetxt(output_file_ejecta, np.column_stack(text_array_ejecta), header=header_ejecta, fmt='%.8e')


if __name__ == "__main__":
    main()