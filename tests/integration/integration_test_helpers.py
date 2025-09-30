import shutil
from pathlib import Path
from typing import Callable
from unittest.mock import patch

import imap_data_access
from imap_data_access.file_validation import generate_imap_file_path, ScienceFilePath, AncillaryFilePath

from tests.test_helpers import create_mock_query_results


def create_mock_query(input_files: list[Path]) -> Callable:
    file_paths = [generate_imap_file_path(f.name) for f in input_files]

    def fake_query(**kwargs):
        table = kwargs.get("table") or "science"
        if "table" in kwargs:
            del kwargs["table"]
        if kwargs.get("version") == "latest":
            del kwargs["version"]

        query_result = []
        for file_path in file_paths:
            if table == "science" and isinstance(file_path, ScienceFilePath):
                file_dict = ScienceFilePath.extract_filename_components(file_path.filename)
            elif table == "ancillary" and isinstance(file_path, AncillaryFilePath):
                file_dict = AncillaryFilePath.extract_filename_components(file_path.filename)
            elif table not in ["science", "ancillary"]:
                raise NotImplementedError(f"Query for table: {table}")
            else:
                file_dict = None

            if file_dict is not None:
                keys = set(file_dict.items())
                if set(kwargs.items()).issubset(keys):
                    query_result.extend(create_mock_query_results("glows", [file_path.filename.name]))
        return query_result

    return fake_query


def mock_imap_data_access(data_dir: Path, input_files: list[Path]):
    def decorator(fn: Callable):
        def wrapped(self, *args):
            def fake_download(file: Path | str):
                filename = Path(file).name
                imap_file_path = generate_imap_file_path(filename)
                full_path = imap_file_path.construct_path()

                self.assertTrue(full_path.exists(), f"Expected {full_path} to exist, but it doesn't.")
                return full_path

            if data_dir.exists(): shutil.rmtree(data_dir)
            data_dir.mkdir(exist_ok=True, parents=True)

            with (
                patch.dict(imap_data_access.config, {"DATA_DIR": data_dir}),
                patch.object(imap_data_access, "download", new=fake_download),
                patch.object(imap_data_access, "query", new=create_mock_query(input_files))
            ):
                for file_path in input_files:
                    paths_to_generate = generate_imap_file_path(file_path.name).construct_path()
                    paths_to_generate.parent.mkdir(exist_ok=True, parents=True)
                    shutil.copy(src=file_path, dst=paths_to_generate)

                fn(self, *args)

        return wrapped

    return decorator
