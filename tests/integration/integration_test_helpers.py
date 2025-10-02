import shutil
from pathlib import Path
from typing import Callable
from unittest.mock import patch

import imap_data_access
from imap_data_access import AncillaryFilePath
from imap_data_access.file_validation import generate_imap_file_path, ScienceFilePath

from tests.test_helpers import create_mock_query_results


def create_mock_query(input_files: list[Path | str], assert_uses_latest: bool = False) -> Callable:
    def fake_query(**kwargs):
        table = kwargs.get("table") or "science"
        if "table" in kwargs:
            del kwargs["table"]

        if assert_uses_latest:
            assert kwargs.get("version") == "latest"
        if kwargs.get("version") == "latest":
            del kwargs["version"]

        desired_attributes = set(kwargs.items())

        filtered_by_type = []
        for input_file in input_files:
            match generate_imap_file_path(Path(input_file).name), table:
                case ScienceFilePath(), "science":
                    filtered_by_type.append(input_file)
                case AncillaryFilePath(), "ancillary":
                    filtered_by_type.append(input_file)
                case _, "science" | "ancillary":
                    continue
                case _, _:
                    raise ValueError(f"Unexpected file type: {input_file}")

        query_results = create_mock_query_results(filtered_by_type)
        return [qr for qr in query_results if desired_attributes.issubset(set(qr.items()))]

    return fake_query


def fake_download(file: Path | str):
    filename = Path(file).name
    imap_file_path = generate_imap_file_path(filename)
    full_path = imap_file_path.construct_path()

    assert full_path.exists(), f"Expected {full_path} to exist, but it doesn't."
    return full_path


class mock_imap_data_access:
    def __init__(self, data_dir: Path, input_files: list[Path]):
        self.data_dir = data_dir
        self.input_files = input_files

    def __enter__(self):
        self.data_dir_patcher = patch.dict(imap_data_access.config, {"DATA_DIR": self.data_dir})
        self.download_patcher = patch.object(imap_data_access, "download", new=fake_download)
        self.query_patcher = patch.object(imap_data_access, "query", new=create_mock_query(self.input_files))

        self.data_dir_patcher.start()
        self.download_patcher.start()
        self.query_patcher.start()

        if self.data_dir.exists(): shutil.rmtree(self.data_dir)
        self.data_dir.mkdir(exist_ok=True, parents=True)

        for file_path in self.input_files:
            paths_to_generate = generate_imap_file_path(file_path.name).construct_path()
            paths_to_generate.parent.mkdir(exist_ok=True, parents=True)
            shutil.copy(src=file_path, dst=paths_to_generate)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.data_dir_patcher.stop()
        self.download_patcher.stop()
        self.query_patcher.stop()

        return False

    def __call__(self, fn: Callable):
        def wrapped(obj, *args, **kwargs):
            with self:
                fn(obj, *args, **kwargs)

        return wrapped
