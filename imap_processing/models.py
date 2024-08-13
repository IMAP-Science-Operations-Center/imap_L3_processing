from dataclasses import dataclass
from datetime import datetime


@dataclass
class UpstreamDataDependency:
    instrument: str
    data_level: str
    descriptor: str
    start_date: datetime
    end_date: datetime
    version: str