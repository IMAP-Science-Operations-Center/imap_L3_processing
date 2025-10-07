from pathlib import Path

import imap_processing
from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes

from imap_l3_processing.cdf.imap_attribute_manager import ImapAttributeManager

imap_processing_metadata_path = Path(imap_processing.cdf.__file__).parent / "config"


l2_attr_manager = ImapCdfAttributes()

l2_attr_manager.add_instrument_variable_attrs("enamaps", "l2-common")
l2_attr_manager.add_instrument_variable_attrs("enamaps", "l2-rectangular")

l3_attr_manager = ImapAttributeManager()
l3_attr_manager.add_instrument_attrs("hi", "l3", "map-descriptor")


l2_attrs = l2_attr_manager.get_variable_attributes("ena_intensity")
l3_attrs = l3_attr_manager.get_variable_attributes("ena_intensity")


map_vars = [
    "epoch",
    "epoch_delta",
    "energy",
    "energy_label",
    "energy_delta_minus",
    "energy_delta_plus",
    "latitude",
    "latitude_label",
    "latitude_delta",
    "longitude",
    "longitude_label",
    "longitude_delta",
    "ena_intensity",
    "ena_intensity_stat_uncert",
    "ena_intensity_sys_err",
    "exposure_factor",
    "obs_date",
    "obs_date_range",
    "solid_angle",
    "ena_spectral_index",
    "ena_spectral_index_stat_uncert",
]

for var in map_vars:
    l3_attrs = l3_attr_manager.get_variable_attributes(var)
    try:
        l2_attrs = l2_attr_manager.get_variable_attributes(var)
    except KeyError:
        print(f"{var} not in L2 attributes!")
        continue

    print(var)
    for attr in l3_attrs.keys():
        if attr in l2_attrs and l2_attrs[attr] != l3_attrs[attr]:
            print(f"\t{attr}: {l2_attrs[attr]}\t{l3_attrs[attr]}")

    not_included_keys = [key for key in l3_attrs.keys() if key not in l2_attrs.keys()]
    print(var, not_included_keys)


