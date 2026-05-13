import argparse
from pathlib import Path

from spacepy.pycdf import CDF


def truncate_cdf(cdf_path: Path, number_of_epochs: int = 2) -> Path:
    output_path = cdf_path.parent / f"{cdf_path.stem}-truncated.cdf"
    output_path.unlink(missing_ok=True)
    original_cdf = CDF(str(cdf_path))

    with CDF(str(output_path), masterpath="") as cdf:
        for attr_name in original_cdf.attrs:
            cdf.attrs[attr_name] = list(original_cdf.attrs[attr_name])

        for var in original_cdf:
            depends_on_epoch = var == "epoch" or original_cdf[var].attrs.get("DEPEND_0") == "epoch"
            if depends_on_epoch:
                cdf.clone(original_cdf[var], var, data=False)
                cdf[var][:] = original_cdf[var][...][:number_of_epochs]
            else:
                cdf.clone(original_cdf[var], var, data=True)

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Truncate one or more CDF files to a smaller number of epochs along the epoch axis."
    )
    parser.add_argument("cdf_paths", nargs="+", type=Path, help="One or more CDF files to truncate.")
    parser.add_argument(
        "-n",
        "--number-of-epochs",
        type=int,
        default=2,
        help="Number of epochs to keep from the start of each file (default: 2).",
    )

    args = parser.parse_args()

    for cdf_path in args.cdf_paths:
        output_path = truncate_cdf(cdf_path, args.number_of_epochs)
        print(f"Wrote {output_path}")
