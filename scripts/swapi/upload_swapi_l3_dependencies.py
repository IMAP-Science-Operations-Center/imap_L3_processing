from pathlib import Path

import imap_data_access

files_to_upload = [
    "imap_swapi_l2_alpha-density-temperature-lut-text-not-cdf_20240920_v004.cdf",
    "imap_swapi_l2_clock-angle-and-flow-deflection-lut-text-not-cdf_20240918_v001.cdf",
    "imap_swapi_l2_density-of-neutral-helium-lut-text-not-cdf_20241023_v001.cdf",
    "imap_swapi_l2_density-temperature-lut-text-not-cdf_20240905_v002.cdf",
    "imap_swapi_l2_efficiency-lut-text-not-cdf_20241020_v003.cdf",
    "imap_swapi_l2_energy-gf-lut-not-cdf_20240923_v002.cdf",
    "imap_swapi_l2_instrument-response-lut-zip-not-cdf_20241023_v001.cdf",
]
for file in files_to_upload:
    try:
        imap_data_access.upload(Path(__file__).parent.parent / "swapi" / "test_data" / file)
        print("Successfully uploaded", file)
    except Exception as e:
        print(f'File "{file}" failed to upload with exception {e}')
