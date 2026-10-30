import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from imap_data_access.file_validation import ImapFilePath, ScienceFilePath
from spacepy.pycdf import CDF


def convert_epoch_time(
    filename,
    target=datetime(2026, 10, 20, 12),
    new_science_file_parts: dict[str, Any] | None = None,
) -> Path | None:
    file_path = Path(filename)
    new_science_file_parts = new_science_file_parts or {}

    with CDF(str(file_path), readonly=False) as cdf:
        delta = target - cdf["epoch"][0]
        cdf["epoch"][...] += delta

    try:
        science_file_path = ScienceFilePath(file_path)

        science_file_args = {
            "instrument": science_file_path.instrument,
            "data_level": science_file_path.data_level,
            "descriptor": science_file_path.descriptor,
            "start_time": target.strftime("%Y%m%d"),
            "version": science_file_path.version,
            "extension": science_file_path.extension,
            "repointing": science_file_path.repointing,
            "cr": science_file_path.cr,
        }

        new_science_file_path = ScienceFilePath.generate_from_inputs(
            **{**science_file_args, **new_science_file_parts}
        )

        new_name = new_science_file_path.construct_path().name
        new_path = file_path.rename(file_path.parent / new_name)

    except ImapFilePath.InvalidImapFileError:
        print(
            "Did not rename the file because it could not be converted into a science file path!"
        )
        return None

    return new_path


if __name__ == "__main__":
    convert_epoch_time(sys.argv[1])
