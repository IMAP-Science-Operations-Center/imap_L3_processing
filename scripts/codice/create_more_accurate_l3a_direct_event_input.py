from datetime import timedelta
from pathlib import Path

import numpy as np
from spacepy.pycdf import CDF

from imap_l3_processing.codice.l3.lo.direct_events.science.angle_lookup import PositionToElevationLookup
from imap_l3_processing.codice.l3.lo.models import CODICE_LO_L2_NUM_PRIORITIES
from imap_l3_processing.constants import ONE_SECOND_IN_NANOSECONDS
from tests.test_helpers import get_run_local_data_path, get_test_instrument_team_data_path


def extend_priority_counts_to_24_spin_angles(template_cdf_path, priority_count_variables: list[str], output_dir):
    rng = np.random.default_rng()

    output_path = output_dir / template_cdf_path.name
    output_path.unlink(missing_ok=True)

    with CDF(str(template_cdf_path)) as template_cdf:
        with CDF(str(output_path), masterpath="") as cdf:
            cdf["spin_sector_index"] = np.arange(24)
            for var in template_cdf:
                if var in priority_count_variables:
                    cdf[var] = np.repeat(template_cdf[var][...], 2, axis=2)
                elif var == "spin_sector_index":
                    pass
                else:
                    cdf[var] = template_cdf[var]
                cdf[var].attrs = template_cdf[var].attrs

            rgfo_half_spin = rng.integers(0, 33, size=template_cdf["rgfo_half_spin"].shape)
            randomly_fill_value = rng.choice([False, True], size=template_cdf["rgfo_half_spin"].shape)

            cdf["rgfo_half_spin"] = rgfo_half_spin
            # cdf["rgfo_half_spin"] = np.where(randomly_fill_value, cdf["rgfo_half_spin"].attrs["FILLVAL"],
            #                                  rgfo_half_spin)
    return output_path


def modify_l1a_priority_counts(instrument_team_l1a_nsw_path: Path, instrument_team_l1a_sw_path: Path) -> tuple[
    Path, Path]:
    output_dir = get_run_local_data_path("codice/lo")
    output_dir.mkdir(parents=True, exist_ok=True)

    l1a_nsw_priority_counts_vars = [
        "p5_heavies",
        "p6_hplus_heplusplus",
    ]

    modified_l1a_nsw_path = extend_priority_counts_to_24_spin_angles(instrument_team_l1a_nsw_path,
                                                                     l1a_nsw_priority_counts_vars, output_dir)

    l1a_sw_priority_counts_vars = [
        "p0_tcrs",
        "p1_hplus",
        "p2_heplusplus",
        "p3_heavies",
        "p4_dcrs",
    ]

    modified_l1a_sw_path = extend_priority_counts_to_24_spin_angles(instrument_team_l1a_sw_path,
                                                                    l1a_sw_priority_counts_vars,
                                                                    output_dir)

    return (modified_l1a_nsw_path,
            modified_l1a_sw_path)


def modify_l2_direct_events(instrument_team_l2_path: Path) -> Path:
    output_dir = get_run_local_data_path("codice/lo")
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng()
    elevation_lookup = PositionToElevationLookup()
    float_fillval = 1e-31
    int_fillval = 255

    output_path = output_dir / instrument_team_l2_path.name
    output_path.unlink(missing_ok=True)

    with CDF(str(instrument_team_l2_path)) as template_cdf:
        with CDF(str(output_path), masterpath="") as cdf:
            epoch_delta = timedelta(minutes=2).total_seconds() * ONE_SECOND_IN_NANOSECONDS
            cdf["epoch_delta_plus"] = np.full(template_cdf["epoch"].shape, epoch_delta)
            cdf["epoch_delta_minus"] = np.full(template_cdf["epoch"].shape, epoch_delta)

            for priority_i in range(CODICE_LO_L2_NUM_PRIORITIES):
                elevation_var = f"p{priority_i}_elevation"
                position_var = f"p{priority_i}_position"

                cdf[position_var] = rng.integers(1, 25, size=template_cdf[position_var].shape)
                cdf[position_var].attrs["FILLVAL"] = int_fillval

                cdf[elevation_var] = rng.choice(elevation_lookup.bin_centers, size=template_cdf[position_var].shape)
                cdf[elevation_var].attrs["FILLVAL"] = float_fillval

                for epoch_i in range(cdf[elevation_var].shape[0]):
                    number_of_events = template_cdf[f"p{priority_i}_num_events"][epoch_i]

                    filled_position = cdf[position_var][...]
                    filled_position[epoch_i, number_of_events:] = int_fillval
                    cdf[position_var] = filled_position

                    filled_elevation = cdf[elevation_var][...]
                    filled_elevation[epoch_i, number_of_events:] = float_fillval
                    cdf[elevation_var] = filled_elevation

            for var in template_cdf:
                if var not in cdf:
                    cdf[var] = template_cdf[var]
                    cdf[var].attrs = template_cdf[var].attrs

    return output_path


if __name__ == "__main__":
    print(modify_l2_direct_events(
        get_test_instrument_team_data_path("codice/lo/imap_codice_l2_lo-direct-events_20241110_v002.cdf")))
