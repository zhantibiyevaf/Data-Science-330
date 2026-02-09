"""
HW3
"""

from pathlib import Path

from dtsc330_26.readers.har import HAR
import dtsc330_26.readers.har as har_module
from dtsc330_26.reusable_classifier import ReusableClassifier


def main():
    repository = Path(__file__).resolve().parent
    data_path = repository / "data" / "motion-and-heart-rate-from-a-wrist-worn-wearable-and-labeled-sleep-from-polysomnography-1.0.0"

    har = HAR(str(data_path), n_people=1)
    df = har.df

    print(df.shape)
    print(df.columns)
    print(df.head())
    
    #features
    feature_cols = ["hr_mean", "hr_std", "acc_mag_mean", "acc_mag_std"]
    #labels
    X = df[feature_cols]
    y = df["is_sleep"]

    # training and evaluating the model using the ReusableClassifier class
    rc = ReusableClassifier(model_type="random_forest")
    score = rc.assess(X, y)

    print("Model:", rc.model_type)
    print("Features:", feature_cols)
    print("Score:", score)




if __name__ == "__main__":
    main()