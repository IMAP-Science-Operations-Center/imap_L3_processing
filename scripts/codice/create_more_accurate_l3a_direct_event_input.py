from pathlib import Path

import numpy as np
from spacepy.pycdf import CDF

from tests.test_helpers import get_run_local_data_path


def extend_priority_counts_to_24_spin_angles(template_cdf_path, priority_count_variables: list[str], output_dir):
    output_path = output_dir / template_cdf_path.name
    output_path.unlink(missing_ok=True)

    template_cdf = CDF(str(template_cdf_path))

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


if __name__ == "__main__":
    modify_l1a_priority_counts()
