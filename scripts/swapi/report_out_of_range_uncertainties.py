from pathlib import Path

import imap_data_access
import numpy as np
from spacepy.pycdf import CDF


def report_out_of_range_uncertainties(path: Path | str):
    path = Path(path)
    with CDF(str(path)) as cdf:
        uncertainty_vars = set()
        for var in cdf:
            if delta_minus := cdf[var].attrs.get("DELTA_MINUS_VAR"):
                uncertainty_vars.add(delta_minus)
            if delta_plus := cdf[var].attrs.get("DELTA_PLUS_VAR"):
                uncertainty_vars.add(delta_plus)
        for var in uncertainty_vars:
            fill_val = cdf[var].attrs["FILLVAL"]
            valid_min = cdf[var].attrs["VALIDMIN"]
            valid_max = cdf[var].attrs["VALIDMAX"]

            values = cdf[var][...]

            bad_vals = ((values < valid_min) | (values > valid_max)) & (
                values != fill_val
            )
            bad_indices = np.argwhere(bad_vals)
            if len(bad_indices) > 0:
                print(
                    f"{path.name} | {var} ({valid_min}-{valid_max}) | {np.squeeze(bad_indices)} | {values[bad_vals]}"
                )


def get_all_swapi_l3a():
    query_results = imap_data_access.query(
        instrument="swapi", data_level="l3a", version="latest"
    )
    paths_to_downloaded_files = []
    for result in sorted(
        query_results, key=lambda r: (r["descriptor"], r["start_date"])
    ):
        file_path = imap_data_access.download(result["file_path"])
        paths_to_downloaded_files.append(file_path)
    return paths_to_downloaded_files


if __name__ == "__main__":
    swapi_l3a_paths = get_all_swapi_l3a()
    for path in swapi_l3a_paths:
        report_out_of_range_uncertainties(path)

# report_out_of_range_uncertainties(
#     r"C:\Users\Petty\Development\imap_L3_processing\data\imap\swapi\l3a\2026\04\imap_swapi_l3a_pui-he_20260419_v001.0033.cdf"
# )