import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Callable
from unittest.mock import patch, Mock
from urllib.parse import urlparse

import imap_data_access
import requests
from imap_data_access import AncillaryFilePath
from imap_data_access.file_validation import generate_imap_file_path, ScienceFilePath, SPICEFilePath
from requests import Response

from tests.test_helpers import create_mock_query_results


class ImapQueryPatcher:
    def __init__(self, input_files: list[Path | str]):
        self.input_files = input_files
        self.patcher = patch.object(imap_data_access, 'query', new=self)

    def start(self):
        self.patcher.start()

    def stop(self):
        self.patcher.stop()

    def __call__(self, **kwargs):
        table = kwargs.get("table") or "science"
        if "table" in kwargs:
            del kwargs["table"]

        use_latest = kwargs.get("version") == "latest"
        if kwargs.get("version") == "latest":
            del kwargs["version"]

        desired_attributes = set(kwargs.items())

        filtered_by_type = []
        for input_file in self.input_files:
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
        filtered_query_results = [qr for qr in query_results if desired_attributes.issubset(set(qr.items()))]
        if use_latest:
            filtered_query_results = self.filter_results_for_latest(filtered_query_results)
        return filtered_query_results

    @staticmethod
    def filter_results_for_latest(query_results: list[dict]):
        query_results_by_date = defaultdict(list)
        for qr in query_results:
            query_results_by_date[qr["start_date"]].append(qr)
        return [max(qrs, key=lambda qr: qr["version"]) for qrs in query_results_by_date.values()]

def fake_download(file: Path | str):
    filename = Path(file).name
    imap_file_path = generate_imap_file_path(filename)
    full_path = imap_file_path.construct_path()

    assert full_path.exists(), f"Expected {full_path} to exist, but it doesn't."
    return full_path

metakernel_text = """
\\begindata

\tKERNELS_TO_LOAD = ({kernels})

\\begintext
"""

class RequestsGetPatcher:
    def __init__(self, spice_inputs: list[Path]):
        self.spice_file_names = [Path(spice_input).name for spice_input in spice_inputs]
        self.original_requests_get = requests.get
        self.patcher = patch.object(requests, "get", new=self)

    def start(self):
        self.patcher.start()

    def stop(self):
        self.patcher.stop()

    def __call__(self, url, **kwargs):
        response = Mock(spec=Response)

        parsed_url = urlparse(url)
        if "imap-mission" in url:
            if parsed_url.path == "/metakernel":
                if 'params' in kwargs and kwargs['params'].get('list_files') == 'true':
                    response.text = json.dumps(self.spice_file_names)
                elif 'params' in kwargs:
                    prefix = kwargs['params'].get("spice_path") or ""

                    spice_file_paths = [SPICEFilePath(fn) for fn in self.spice_file_names]
                    all_spice_paths = []
                    for spice_file in spice_file_paths:
                        spice_subdir_name = spice_file.construct_path().parent.name
                        spice_file_name = spice_file.construct_path().name
                        spice_path = str(Path(prefix) / spice_subdir_name / spice_file_name)

                        chunked_spice_path = [spice_path[i:i+79] for i in range(0, len(spice_path), 79)]
                        for i, chunk in enumerate(chunked_spice_path[:-1]):
                            chunked_spice_path[i] = f"{chunk}+"

                        all_spice_paths.extend(chunked_spice_path)

                    formatted_spice_file_names = ',\n'.join([f"'{spice}'" for spice in all_spice_paths])
                    response.text = metakernel_text.format(kernels=formatted_spice_file_names)
                    response.content = response.text.encode()
            else:
                assert False, "I don't know how to mock that IMAP endpoint!"
        else:
            response = self.original_requests_get(url, **kwargs)

        return response

class mock_imap_data_access:
    def __init__(self, data_dir: Path, input_files: list[Path]):
        self.data_dir = data_dir
        self.input_files = input_files

        spice_file_paths = [input_file for input_file in input_files if isinstance(generate_imap_file_path(input_file.name), SPICEFilePath)]

        self.data_dir_patcher = patch.dict(imap_data_access.config, {"DATA_DIR": self.data_dir})
        self.download_patcher = patch.object(imap_data_access, "download", new=fake_download)
        self.query_patcher = ImapQueryPatcher(self.input_files)
        self.requests_get_patcher = RequestsGetPatcher(spice_file_paths)

    def __enter__(self):
        self.data_dir_patcher.start()
        self.download_patcher.start()
        self.query_patcher.start()
        self.requests_get_patcher.start()

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
        self.requests_get_patcher.stop()

        return False

    def __call__(self, fn: Callable):
        def wrapped(obj, *args, **kwargs):
            with self:
                fn(obj, *args, **kwargs)

        return wrapped
