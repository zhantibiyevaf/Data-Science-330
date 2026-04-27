from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = ["misspelled", "correct"]


def read_spelling_pairs(path):
    """Read generated spelling pairs from a CSV file."""
    csv_path = Path(path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find spelling data at {csv_path}")

    data = pd.read_csv(csv_path)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in data.columns]

    if missing_columns:
        raise ValueError(
            f"Spelling data is missing required column(s): {', '.join(missing_columns)}"
        )

    return data[REQUIRED_COLUMNS].dropna()
