import os
import sys
from pathlib import Path


def download_parents(path: Path):
    initial_cwd = os.getcwd()
    os.chdir(path)
    import imap_data_access
    from spacepy.pycdf import CDF

    try:
        files_to_download = set()

        for file in path.glob("*.cdf"):
            print(f"opening {path / file}")
            with CDF(str(path / file)) as cdf:
                for parent in cdf.attrs['Parents']:
                    files_to_download.add(parent)

        print(f"Downloading {len(files_to_download)} files")

        for file in files_to_download:
            print(f"downloading {file}")
            print(imap_data_access.download(file))

    except Exception as e:
        print(e)
        os.chdir(initial_cwd)


if __name__ == '__main__':
    download_parents(Path(sys.argv[1]))
