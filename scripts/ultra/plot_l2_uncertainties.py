from datetime import datetime
from pathlib import Path

import numpy as np
from astropy_healpix import HEALPix
from matplotlib import pyplot as plt
from spiceypy import spiceypy

from imap_l3_processing.ultra.ultra_l3_dependencies import UltraL3CombinedDependencies
from tests.integration.test_map_processor_integration import INTEGRATION_TEST_DATA_PATH
from tests.test_helpers import get_run_local_data_path, get_test_data_path

ultra_imap_data_dir = get_run_local_data_path("ultra/integration_data")
ultra_path_90 = Path(
    r"C:\Users\Harrison\Downloads\ultra_combined_hf_validation\ULTRA-90sensor_helio-psets_20251201")
ultra_path_45 = Path(
    r"C:\Users\Harrison\Downloads\ultra_combined_hf_validation\ULTRA-45sensor_helio-psets_20251201"
)
glows_path = Path(
    r"C:\Users\Harrison\Downloads\ultra_combined_hf_validation\glows")

ancil_files = [
    INTEGRATION_TEST_DATA_PATH / "spice" / "naif020.tls",
    INTEGRATION_TEST_DATA_PATH / "spice" / "imap_science_108.tf",
    INTEGRATION_TEST_DATA_PATH / "spice" / "imap_sclk_008.tsc",
    INTEGRATION_TEST_DATA_PATH / "spice" / "imap_dps_2025_105_2026_105_009.ah.bc",
]

u45_paths = [Path(f) for f in list(ultra_path_45.glob("*.cdf"))]
u90_paths = [Path(f) for f in list(ultra_path_90.glob("*.cdf"))]
glows_paths = [Path(f) for f in list(glows_path.glob("*.cdf"))]
l2_u90 = Path(
    r"C:\Users\Harrison\Downloads\ultra_combined_hf_validation\imap_ultra_l2_u90-ena-h-hf-nsp-full-hae-2deg-6mo_20250416_v000.cdf")
l2_u45 = Path(
    r"C:\Users\Harrison\Downloads\ultra_combined_hf_validation\imap_ultra_l2_u45-ena-h-hf-nsp-full-hae-2deg-6mo_20250416_v000.cdf")
start_date = datetime(2025, 4, 16)
descriptor = "ulc-ena-h-hf-nsp-full-hae-2deg-6mo"

for f in ancil_files:
    spiceypy.furnsh(str(f))

ultra_nsp_combined_dep = UltraL3CombinedDependencies.from_file_paths(u45_pset_paths=u45_paths, u90_pset_paths=u90_paths,
                                                                     glows_l3e_pset_paths=glows_paths,
                                                                     u45_map_path=l2_u45, u90_map_path=l2_u90,
                                                                     energy_bin_group_sizes_path=get_test_data_path(
                                                                         "ultra/imap_ultra_l2-energy-bin-group-sizes_20250101_v000.csv"))
print(ultra_nsp_combined_dep.u45_l2_map.intensity_map_data.ena_intensity_stat_uncert.shape)

hp32 = HEALPix(32)
lon, lat = hp32.healpix_to_lonlat(np.arange(hp32.npix))
plt.scatter(lon, lat,
            c=ultra_nsp_combined_dep.u45_l2_map.intensity_map_data.ena_intensity_stat_uncert[0].reshape(12, -1)[0])
plt.show()
