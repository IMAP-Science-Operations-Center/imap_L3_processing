from datetime import datetime, timedelta
from pathlib import Path

from spacepy import pycdf
from spacepy.pycdf import CDF

from tests.test_helpers import get_test_data_path


def generate_fake_l1c(start_date: datetime, num_days: int, output_dir: Path) -> list[str]:
    output_files = []

    for i in range(num_days):
        date_to_set = start_date + timedelta(days=i)
        logical_source = f"imap_hi_l1c_90sensor-pset_{date_to_set.strftime('%Y%m%d')}_v001"
        output_path = output_dir / f"{logical_source}.cdf"
        output_path.unlink(missing_ok=True)
        with CDF(str(output_path),
                 masterpath=str(get_test_data_path("hi/imap_hi_l1c_90sensor-pset_20250415_v001.cdf"))) as cdf:
            del cdf['epoch']
            cdf.new("epoch", [date_to_set], type=pycdf.const.CDF_TIME_TT2000)
        output_files.append(str(output_path))
    return output_files


if __name__ == "__main__":
    generate_fake_l1c(datetime(month=4, year=2025, day=16, hour=12), num_days=90,
                      output_dir=get_test_data_path(f"hi/fake_l1c/90"))
