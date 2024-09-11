from datetime import datetime
from typing import List, Optional

from imap_processing.models import UpstreamDataDependency


class Processor:
    def __init__(self, dependencies: List[UpstreamDataDependency], instrument: str, level: str, start_date: datetime,
                 end_date: datetime,
                 version: str):
        self.instrument = instrument
        self.level = level
        self.version = version
        self.end_date = end_date
        self.start_date = start_date
        self.dependencies = dependencies

    def format_time(self, t: Optional[datetime]) -> Optional[str]:
        if t is not None:
            return t.strftime("%Y%m%d")
        return None
