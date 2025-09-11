import sys
from datetime import datetime
from pathlib import Path

from imap_data_access.file_validation import generate_imap_file_path, ScienceFilePath, ImapFilePath
from spacepy.pycdf import CDF


def convert_epoch_time(filename, target=datetime(2025, 6, 7, 12)):
    file_path = Path(filename)

    with CDF(str(file_path), readonly=False) as cdf:
        delta = target - cdf['epoch'][0]
        cdf['epoch'][...] += delta

    try:
        science_file_path = ScienceFilePath(file_path)

        new_science_file_path = ScienceFilePath.generate_from_inputs(
            instrument=science_file_path.instrument,
            data_level=science_file_path.data_level,
            descriptor=science_file_path.descriptor,
            start_time=target.strftime("%Y%m%d"),
            version=science_file_path.version,
            extension=science_file_path.extension,
            repointing=science_file_path.repointing,
            cr=science_file_path.cr,
        )


        new_name = new_science_file_path.construct_path().name
        file_path.rename(file_path.parent / new_name)

    except ImapFilePath.InvalidImapFileError:
        print("Did not rename the file because it could not be converted into a science file path!")



if __name__ == '__main__':
    convert_epoch_time(sys.argv[1])
