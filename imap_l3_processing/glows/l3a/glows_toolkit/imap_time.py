"""@package docstring
Author: KPLabs with some modifications by Marek Strumik, maro at cbk.waw.pl
"""
from datetime import datetime, timedelta

class SpacecraftTime():
    """
    SpacecraftTime() class for epoch-datatime conversions
    """
    @staticmethod
    def date_to_spacecraft_epoch(time_tag: datetime) -> int:
        """Get number of seconds after 1 January 2010, midnight
        Arguments:
        time_tag: Datetime to convert
        Returns:
        Spacecraft Epoch (SCLK)
        """
        delta = time_tag - datetime(2010, 1, 1)
        timestamp = int(delta.total_seconds())
        assert timestamp >= 0
        return timestamp & 0xFFFFFFFF

    @staticmethod
    def spacecraft_epoch_to_date(epoch: int) -> datetime:
        """Converts spececraft epoch (SCLK) to real datetime.
        Arguments:
        epoch Spacecraft epoch (SCLK) value
        Returns:
        Corresponding datetime
        """
        delta = timedelta(seconds=epoch)
        return datetime(2010, 1, 1) + delta
