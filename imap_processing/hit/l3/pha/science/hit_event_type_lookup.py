import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Rule:
    range: str
    included_detector_groups: list[str]
    excluded_detector_groups: list[str]


@dataclass
class HitEventTypeLookup:
    _rules: list[Rule]

    def lookup_range(self, detectors: set[str]):
        for rule in self._rules:
            has_required_groups = all(included_group in detectors for included_group in rule.included_detector_groups)
            does_not_have_excluding_groups = all(
                excluded_group not in detectors for excluded_group in rule.excluded_detector_groups)

            if has_required_groups and does_not_have_excluding_groups:
                return rule

        return None

    @classmethod
    def from_csv(cls, file_path: Path):
        with open(file_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')

            detector_groups = next(csv_reader)
            data = list(csv_reader)

            rules = []
            for row in data:
                if row[-1] != "NOCALC":
                    included_detector_groups = []
                    excluded_detector_groups = []

                    for detector_group, is_included in zip(detector_groups[:-1], row[:-1]):
                        if is_included == "1":
                            included_detector_groups.append(detector_group)
                        elif is_included == "0":
                            excluded_detector_groups.append(detector_group)
                    rules.append(Rule(range=row[-1], included_detector_groups=included_detector_groups,
                                      excluded_detector_groups=excluded_detector_groups))

        return cls(_rules=rules)
