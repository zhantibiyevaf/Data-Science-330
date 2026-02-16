"""
Read in Human Activity Recognition (HAR) data for classification of sleep
"""

import os
import pandas as pd
import numpy as np


class HAR:
    def __init__(self, path: str, n_people: int):
        """Take in a path, read in the data files

        Args:
            path (str): the BASE path of the directory
            n_people (int): the number of people's data to read
        """
        hrs = self._read_hr(path, n_people)
        mots = self._read_motion(path, n_people)
        lbls = self._read_labels(path, n_people)
        self.df = self._combine(hrs, mots, lbls)

    def _read_hr(self, path: str, n_people: int = 1):
        """Read the heartrate from the base path"""
        path = os.path.join(path, "heart_rate")
        people = os.listdir(path)

        dfs = []
        count = 0
        for person in people:
            hrdf = pd.read_csv(
                os.path.join(path, person), names=["timestamp", "hr"]
            ).drop_duplicates("timestamp", keep="first")
            hrdf["person"] = count
            dfs.append(hrdf)
            count += 1
            if count >= n_people:
                break

        return pd.concat(dfs, ignore_index=True)

    def _read_motion(self, path: str, n_people: int = 1):
        """Read the motion from the base path"""
        path = os.path.join(path, "motion")
        people = os.listdir(path)

        dfs = []
        count = 0
        for person in people:
            motdf = pd.read_csv(
                os.path.join(path, person),
                delimiter=" ",
                names=["timestamp", "acc_x", "acc_y", "acc_z"],
            ).drop_duplicates("timestamp", keep="first")
            motdf["person"] = count
            dfs.append(motdf)
            count += 1
            if count >= n_people:
                break

        return pd.concat(dfs, ignore_index=True)

    def _read_labels(self, path: str, n_people: int = 1):
        """Read the labels from the base path"""
        path = os.path.join(path, "labels")
        people = os.listdir(path)

        dfs = []
        count = 0
        for person in people:
            labdf = pd.read_csv(
                os.path.join(path, person),
                names=["timestamp", "label"],
                delimiter=" ",
            ).drop_duplicates("timestamp", keep="first")
            labdf["person"] = count
            dfs.append(labdf)
            count += 1
            if count >= n_people:
                break

        comb = pd.concat(dfs, ignore_index=True)
        comb["is_sleep"] = comb["label"] == 0
        return comb.loc[comb["label"] > -1].copy()  # exclude missing data

    def _combine(self, hrs: pd.DataFrame, mots: pd.DataFrame, lbls: pd.DataFrame):
        """Combine three dataframes by interpolating to the highest sampling rate."""
        # If multiple people, combine each person separately (faster + avoids overlap issues)
        if len(pd.unique(hrs["person"])) > 1:
            hrs_groups = dict(tuple(hrs.groupby("person", sort=False)))
            mots_groups = dict(tuple(mots.groupby("person", sort=False)))
            lbls_groups = dict(tuple(lbls.groupby("person", sort=False)))

            out = []
            for person in hrs_groups.keys():
                out.append(
                    self._combine(
                        hrs_groups[person],
                        mots_groups[person],
                        lbls_groups[person],
                    )
                )

            return pd.concat(out, ignore_index=True)

        # ---- Single-person combine below ----

        min_interval = None
        min_interval_df = None
        last_start = None
        first_end = None

        dfs = {"hrs": hrs, "mots": mots, "lbls": lbls}

        for name, df in dfs.items():
            df_sorted = df.sort_values("timestamp")
            time_diffs = df_sorted["timestamp"].diff().dropna()
            sampling_interval = time_diffs.median()

            if min_interval is None or sampling_interval < min_interval:
                min_interval = sampling_interval
                min_interval_df = name

            start_time = df_sorted["timestamp"].min()
            end_time = df_sorted["timestamp"].max()
            if last_start is None or start_time > last_start:
                last_start = start_time
            if first_end is None or end_time < first_end:
                first_end = end_time

        # timestamp as index for interpolation
        for name in dfs:
            dfs[name] = dfs[name].copy()
            dfs[name] = dfs[name].set_index("timestamp").sort_index()

        shared_index = dfs[min_interval_df].index
        shared_index = shared_index[(shared_index >= last_start) & (shared_index <= first_end)]

        hrs_interp = (
            dfs["hrs"]
            .reindex(dfs["hrs"].index.union(shared_index))
            .interpolate(method="index")
            .loc[shared_index]
        )
        mots_interp = (
            dfs["mots"]
            .reindex(dfs["mots"].index.union(shared_index))
            .interpolate(method="index")
            .loc[shared_index]
        )
        lbls_interp = (
            dfs["lbls"]
            .reindex(dfs["lbls"].index.union(shared_index))
            .ffill()
            .loc[shared_index]
        )

        combined = (
            pd.concat(
                [
                    hrs_interp[["hr", "person"]],
                    mots_interp[["acc_x", "acc_y", "acc_z"]],
                    lbls_interp[["is_sleep"]],
                ],
                axis=1,
            )
            .dropna()
            .reset_index()
        )

        # keep your extra feature
        combined["acc_mag"] = np.sqrt(
            combined["acc_x"] ** 2 + combined["acc_y"] ** 2 + combined["acc_z"] ** 2
        )

        return combined


if __name__ == "__main__":
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    data_path = repo_root / "data" / "motion-and-heart-rate-from-a-wrist-worn-wearable-and-labeled-sleep-from-polysomnography-1.0.0"

    har = HAR(str(data_path), 10)
    print("columns:", list(har.df.columns))
    print("n_people:", har.df["person"].nunique())
    print("time range:", har.df["timestamp"].min(), "to", har.df["timestamp"].max())
    print(har.df[["timestamp", "person", "hr", "acc_mag", "is_sleep"]].head())