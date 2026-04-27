import pandas as pd


def read_spelling_pairs(path: str) -> pd.DataFrame:
    """
    Read spelling correction pairs from a CSV file.

    Args:
        path: Path to the CSV file.

    Returns:
        DataFrame with misspelled and correct columns.
    """
    return pd.read_csv(path)