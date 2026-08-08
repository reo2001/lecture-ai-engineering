"""Smoke test for the serialized Titanic model used in the exercise."""

import pickle
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "titanic_model.pkl"
DATA_PATH = BASE_DIR / "data" / "Titanic.csv"


def test_saved_model_can_predict():
    """The committed model should accept raw Titanic features and predict labels."""
    assert MODEL_PATH.is_file(), f"モデルファイルがありません: {MODEL_PATH}"
    assert DATA_PATH.is_file(), f"データファイルがありません: {DATA_PATH}"

    with MODEL_PATH.open("rb") as model_file:
        model = pickle.load(model_file)

    data = pd.read_csv(DATA_PATH)
    features = data.drop(columns=["Survived"]).head(10)
    predictions = model.predict(features)

    assert len(predictions) == len(features)
    assert set(predictions).issubset({0, 1, "0", "1"})
