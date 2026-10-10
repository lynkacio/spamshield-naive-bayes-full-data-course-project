"""Classify one email using the saved full-data SpamShield model."""

from pathlib import Path
import argparse, json

from spamshield.features import counts_for_text
from spamshield.model import MultinomialNaiveBayes

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "results" / "model.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True, help="Email text to classify")
    args = parser.parse_args()

    payload = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    feature_mode = payload["feature_mode"]
    model = MultinomialNaiveBayes.from_dict(payload["model"])
    base, extra = counts_for_text(args.text, feature_mode)
    prediction = model.predict_counts([base], [extra])[0]
    print(prediction)


if __name__ == "__main__":
    main()
