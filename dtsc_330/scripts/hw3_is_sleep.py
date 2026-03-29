"""
HAR classifier. 94.62% accuracy
"""

from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score

from dtsc330_26.readers.har import HAR
from dtsc330_26.reusable_classifier import ReusableClassifier


def main():
    repository = Path(__file__).resolve().parents[2]
    data_path = repository / "data" / "motion-and-heart-rate-from-a-wrist-worn-wearable-and-labeled-sleep-from-polysomnography-1.0.0"

    # load multiple people so we can evaluate between people instead of mixing them
    data = HAR(str(data_path), n_people=10)
    full_df = data.df.copy()

    full_df.index = pd.to_timedelta(full_df["timestamp"], unit="s")

    # get unique people so we can loop through each person separately
    people = pd.unique(full_df["person"])

    train_features, train_labels = [], []
    test_features, test_labels = [], []
    test_person = people[0]

    for person in people:
        print(f"Computing person {person + 1}")

        # work on one person at a time
        df = full_df.loc[full_df["person"] == person].copy().sort_index()

        # create rolling window features across different time scales
        for window in ["10s", "1min", "10min", "1h", "10h"]:
            for column in ["hr", "acc_x", "acc_y", "acc_z", "acc_mag"]:
                for fn in ["mean", "min", "max", "std"]:
                    df[f"{column}_{fn}_{window}"] = df[column].rolling(window).agg(fn)

        df = df.resample("10s").first().dropna(how="any")

        drop_cols = ["timestamp", "hr", "acc_x", "acc_y", "acc_z", "acc_mag", "is_sleep", "person"]
        Xp = df.drop(columns=drop_cols)
        yp = df["is_sleep"]

        if person == test_person:
            test_features.append(Xp)
            test_labels.append(yp)
        else:
            train_features.append(Xp)
            train_labels.append(yp)

    X_train = pd.concat(train_features)
    y_train = pd.concat(train_labels)
    X_test = pd.concat(test_features)
    y_test = pd.concat(test_labels)

    # random forest
    rf = ReusableClassifier(model_type="random_forest")
    rf.train(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_acc = accuracy_score(y_test.astype(int), rf_preds.astype(int))

    # xgboost
    xgb = ReusableClassifier(model_type="xgboost")
    xgb.train(X_train, y_train)
    xgb_preds = xgb.predict(X_test)
    xgb_acc = accuracy_score(y_test.astype(int), xgb_preds.astype(int))

    print("Random Forest accuracy:", rf_acc)
    print("XGBoost accuracy:", xgb_acc)
    print("Difference (XGB - RF):", xgb_acc - rf_acc)


if __name__ == "__main__":
    main()