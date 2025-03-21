from dataclasses import dataclass


@dataclass
class GlowsL3BDependencies:
    @classmethod
    def fetch_dependencies(cls):
        return cls()
