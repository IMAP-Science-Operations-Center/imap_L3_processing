import json
from pathlib import Path


def parse_dependency_json_files(base_dir: Path) -> list[str]:
    if not base_dir.exists():
        raise ValueError(f"{base_dir} does not exist")
    dependency_files = list(base_dir.rglob("*.json"))

    all_files: set[str] = set()
    for dependency_file in dependency_files:
        dependencies = json.loads(dependency_file.read_text())["dependency"]
        files_to_download = [file for dependency in dependencies for file in dependency["files"]]
        all_files.update(files_to_download)

    return sorted(all_files)
