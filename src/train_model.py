"""Train the final earthquake-alert model and save it to models/alert_model.joblib.

Run from the project root:   python src/train_model.py

This reproduces the final model from notebooks/Earthquake_Alert_Prediction.ipynb
(balanced Random Forest on latitude, longitude, depth, magnitude and region).
Use it to regenerate the model if you get a scikit-learn version warning.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "earthquakes.csv"
MODEL_PATH = BASE_DIR / "models" / "alert_model.joblib"

RANDOM_STATE = 42
NUM_COLS = ["latitude", "longitude", "depth", "mag"]
CAT_COLS = ["region_grp"]
MIN_REGION_EVENTS = 20   # regions with fewer events are grouped as "Other"

# Best parameters found by RandomizedSearchCV on events up to 2019 (see notebook, section 7)
RF_PARAMS = dict(n_estimators=200, min_samples_leaf=5, max_features=0.5, max_depth=12)


def prepare_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df = df[df["type"] == "earthquake"].reset_index(drop=True)
    df["region"] = df["place"].str.split(",").str[-1].str.strip()
    counts = df["region"].value_counts()
    df["region_grp"] = np.where(df["region"].map(counts) >= MIN_REGION_EVENTS, df["region"], "Other")
    df["depth"] = df["depth"].clip(lower=0)
    return df[NUM_COLS + CAT_COLS], df["Alert"]


def build_pipeline():
    prep = ColumnTransformer([
        ("num", StandardScaler(), NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
    ])
    rf = RandomForestClassifier(class_weight="balanced_subsample", random_state=RANDOM_STATE,
                                n_jobs=-1, **RF_PARAMS)
    return Pipeline([("prep", prep), ("model", rf)])


if __name__ == "__main__":
    X, y = prepare_data()
    model = build_pipeline().fit(X, y)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Trained on {len(X)} events. Saved model to {MODEL_PATH}")
