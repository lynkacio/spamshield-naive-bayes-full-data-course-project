from collections import Counter
from pathlib import Path

from spamshield.data import load_xlsx_records
from spamshield.features import tokenize_basic, extract_structural_features
from spamshield.model import MultinomialNaiveBayes

ROOT = Path(__file__).resolve().parents[1]


def test_raw_workbook_loads_full_dataset():
    records, removed = load_xlsx_records(ROOT / "data" / "raw" / "SpamAssassin_H2_Data_FULL.xlsx")
    assert len(records) == 5921
    assert sum(r["label"] == "ham" for r in records) == 4109
    assert sum(r["label"] == "spam" for r in records) == 1812
    assert removed == 102


def test_tokenizer_and_url_feature():
    assert tokenize_basic("FREE!!! offer-now") == ["free", "offernow"]
    assert extract_structural_features(
        "visit https://example.com NOW", use_url=True, use_caps=False, use_punct=False
    ) == ["__URL_1__"]


def test_from_scratch_model_learns_toy_problem():
    docs = [Counter({"meeting": 2, "project": 1}), Counter({"free": 3, "prize": 1})]
    labels = ["ham", "spam"]
    model = MultinomialNaiveBayes(alpha=1, max_words=10).fit(docs, labels)
    pred = model.predict_counts([Counter({"free": 2})])[0]
    assert pred == "spam"
