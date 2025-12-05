import os
import sys
import time
from pathlib import Path


def download_parents(path: Path):
    initial_cwd = os.getcwd()
    os.chdir(path)
    import imap_data_access
    from spacepy.pycdf import CDF
    retries = 10

    try:
        files_to_download = set()

        for file in path.glob("*.cdf"):
            print(f"opening {path / file}")
            with CDF(str(path / file)) as cdf:
                for parent in cdf.attrs['Parents']:
                    files_to_download.add(parent)

        print(f"Downloading {len(files_to_download)} files")

        for i, file in enumerate(files_to_download):

            print(f"[{i + 1}/{len(files_to_download)}] downloading {file}")
            failures = 0
            while failures < retries:
                try:
                    imap_data_access.download(file)
                    break
                except Exception as e:
                    failures += 1

                    print(f"failed to download file: {file} after {failures} attempts, retrying...")
                    if failures == retries:
                        raise Exception(
                            f"failed to download file: {file} after {retries} attempts, exiting").with_traceback(
                            e.__traceback__)
                    time.sleep(failures ** 2 * 0.5)

    except Exception as e:
        print(e)
        os.chdir(initial_cwd)


if __name__ == '__main__':
    download_parents(Path(sys.argv[1]))
