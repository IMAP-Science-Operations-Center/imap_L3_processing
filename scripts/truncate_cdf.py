import argparse
from pathlib import Path

from spacepy.pycdf import CDF


def truncate_cdf(cdf_path: Path, number_of_epochs: int = 2):
    output_path = cdf_path.parent / f"{cdf_path.name[:-4]}-truncated.cdf"
    output_path.unlink(missing_ok=True)
    original_cdf = CDF(str(cdf_path))

    with CDF(str(output_path), masterpath="") as cdf:
        cdf["epoch"] = original_cdf["epoch"][...][:number_of_epochs]
        cdf["epoch"].attrs = original_cdf["epoch"].attrs

        for var in original_cdf:
            if var in ["epoch"]:
                continue
            if "DEPEND_0" in original_cdf[var].attrs and original_cdf[var].attrs["DEPEND_0"] == "epoch":
                cdf[var] = original_cdf[var][...][:number_of_epochs]
            else:
                cdf[var] = original_cdf[var]

            cdf[var].attrs = original_cdf[var].attrs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cdf_path", required=True, type=Path)

    args = parser.parse_args()

    truncate_cdf(args.cdf_path)
