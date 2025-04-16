import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ESAStepLookup:
    esa_steps: list[tuple[int, float, float]]

    @classmethod
    def read_from_file(cls, file_path: Path):
        with open(file_path) as file:
            esa_step_lookup = csv.reader(file)
            next(esa_step_lookup)
            esa_step = []
            for index, entry in enumerate(esa_step_lookup):
                esa_step.append((int(entry[0]), float(entry[1]), float(entry[2])))

        return cls(esa_steps=esa_step)
