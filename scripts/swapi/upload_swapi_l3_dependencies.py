from pathlib import Path

import imap_data_access

files_to_upload = [
    "imap_swapi_alpha-density-temperature-lut_20240920_v000.dat",
    "imap_swapi_clock-angle-and-flow-deflection-lut_20240918_v000.dat",
    "imap_swapi_density-of-neutral-helium-lut_20241023_v000.dat",
    "imap_swapi_proton-density-temperature-lut_20240905_v000.dat",
    "imap_swapi_efficiency-lut_20241020_v000.dat",
    "imap_swapi_energy-gf-pui-lut_20100101_v001.csv",
    "imap_swapi_energy-gf-sw-lut_20100101_v001.csv",
    "imap_swapi_instrument-response-lut_20241023_v000.zip",
]
for file in files_to_upload:
    try:
        imap_data_access.upload(Path(__file__).parent.parent / "swapi" / "test_data" / file)
        print("Successfully uploaded", file)
    except Exception as e:
        print(f'File "{file}" failed to upload with exception {e}')
