from pathlib import Path

import tests


def get_test_data_path(filename: str) -> Path:
    return Path(tests.__file__).parent / filename
