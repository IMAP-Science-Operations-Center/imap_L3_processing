from datetime import datetime, timedelta
from pathlib import Path

repoint_path = Path(r"C:\Users\Petty\Downloads\imap_2026_269_10.repoint.csv")


def add_repointings():
    spacecraft_clock_start = datetime(2010, 1, 1, 10, 30)

    repoint1000minus104start = datetime(2024, 12, 31, 23, 30)
    repoint1000minus104end = datetime(2025, 1, 1, 0, 30)

    last_repointing_exclusive = 1000
    first_repointing_inclusive = 1000 - 104

    for repointing in range(first_repointing_inclusive, last_repointing_exclusive):
        start_datetime = repoint1000minus104start + (repointing - (first_repointing_inclusive)) * timedelta(days=1)
        end_datetime = repoint1000minus104end + (repointing - (first_repointing_inclusive)) * timedelta(days=1)

        start_sclk = int((start_datetime - spacecraft_clock_start).total_seconds())
        repoint_start_subsec_sclk = 20

        end_sclk = int((end_datetime - spacecraft_clock_start).total_seconds())
        repoint_end_subsec_sclk = 999980

        formatted_start = start_datetime.strftime('%Y-%m-%d %H:%M:%S.%f')
        formatted_end = end_datetime.strftime('%Y-%m-%d %H:%M:%S.%f')
        csv_line = f"{start_sclk},{repoint_start_subsec_sclk},{end_sclk},{repoint_end_subsec_sclk},{formatted_start},{formatted_end},{repointing}"
        print(csv_line)


if __name__ == "__main__":
    add_repointings()
