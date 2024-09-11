from pathlib import Path

from imap_processing import tests


def get_test_data_path(filename: str) -> Path:
    return Path(tests.__file__).parent / filename
