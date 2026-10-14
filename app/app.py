"""Flask app: predict the alert level of an earthquake.

Run from the project root:   python app/app.py
Then open http://127.0.0.1:5000
"""
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "alert_model.joblib"

app = Flask(__name__)
model = joblib.load(MODEL_PATH)

ALERT_ORDER = ["green", "yellow", "orange", "red"]
REGIONS = sorted(model.named_steps["prep"].named_transformers_["cat"].categories_[0])

# (min, max) accepted for each numeric input
LIMITS = {"latitude": (-90, 90), "longitude": (-180, 180), "depth": (0, 800), "mag": (4.5, 10)}


def parse_and_validate(data):
    """Return (row_dict, error_message). Error is None when the input is valid."""
    row = {}
    for field, (low, high) in LIMITS.items():
        raw = data.get(field)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return None, f"'{field}' must be a number."
        if not low <= value <= high:
            return None, f"'{field}' must be between {low} and {high}."
        row[field] = value
    region = (data.get("region") or "Other").strip()
    row["region_grp"] = region if region in REGIONS else "Other"
    return row, None


def predict(row):
    frame = pd.DataFrame([row])
    proba = model.predict_proba(frame)[0]
    probs = {cls: float(p) for cls, p in zip(model.classes_, proba)}
    predicted = max(probs, key=probs.get)
    return {
        "predicted_alert": predicted,
        "probabilities": {a: round(probs.get(a, 0.0), 4) for a in ALERT_ORDER},
        # Practical view: chance that this event triggers ANY alert (yellow/orange/red)
        "any_alert_probability": round(1 - probs.get("green", 0.0), 4),
    }


@app.route("/", methods=["GET", "POST"])
def index():
    result, error, form = None, None, {}
    if request.method == "POST":
        form = request.form.to_dict()
        row, error = parse_and_validate(form)
        if error is None:
            result = predict(row)
    return render_template("index.html", regions=REGIONS, result=result, error=error,
                           form=form, alert_order=ALERT_ORDER)


@app.route("/predict", methods=["POST"])
def predict_api():
    """JSON API. Example body:
    {"latitude": 35.7, "longitude": 139.7, "depth": 10, "mag": 7.1, "region": "Japan"}"""
    data = request.get_json(silent=True) or {}
    row, error = parse_and_validate(data)
    if error:
        return jsonify({"error": error}), 400
    return jsonify(predict(row))


@app.route("/health")
def health():
    return jsonify({"status": "ok", "classes": list(model.classes_)})


if __name__ == "__main__":
    app.run(debug=False)
