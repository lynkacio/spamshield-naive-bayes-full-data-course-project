"""Tokenization and structural email feature engineering."""

from __future__ import annotations
from collections import Counter
import re


def tokenize_basic(text: str) -> list[str]:
    """Match the original notebook tokenizer exactly."""
    words = []
    for raw_word in text.lower().split():
        word = "".join(ch for ch in raw_word if ch.isalnum())
        if len(word) > 1:
            words.append(word)
    return words


def frequency_bucket(count: int) -> str:
    if count == 0:
        return "0"
    if count == 1:
        return "1"
    if count <= 3:
        return "2_3"
    return "4PLUS"


def extract_structural_features(
    text: str,
    *,
    use_url: bool = True,
    use_caps: bool = True,
    use_punct: bool = True,
) -> list[str]:
    """Encode non-word properties as categorical pseudo-tokens."""
    features: list[str] = []

    if use_url:
        url_count = len(re.findall(r"https?://\S+|www\.\S+", text, flags=re.IGNORECASE))
        features.append(f"__URL_{frequency_bucket(url_count)}__")

    if use_punct:
        punctuation = {"!": "EXCLAMATION", "?": "QUESTION", "$": "DOLLAR", "%": "PERCENT"}
        for symbol, name in punctuation.items():
            features.append(f"__{name}_{frequency_bucket(text.count(symbol))}__")

    if use_caps:
        all_caps_count = 0
        alphabetic_word_count = 0
        for raw_word in text.split():
            letters = "".join(ch for ch in raw_word if ch.isalpha())
            if len(letters) >= 2:
                alphabetic_word_count += 1
                if letters.isupper():
                    all_caps_count += 1
        features.append(f"__ALLCAPS_{frequency_bucket(all_caps_count)}__")
        cap_ratio = all_caps_count / alphabetic_word_count if alphabetic_word_count else 0.0
        if cap_ratio == 0:
            ratio_bucket = "0"
        elif cap_ratio < 0.05:
            ratio_bucket = "LOW"
        elif cap_ratio < 0.10:
            ratio_bucket = "MEDIUM"
        else:
            ratio_bucket = "HIGH"
        features.append(f"__CAP_RATIO_{ratio_bucket}__")

    return features


FEATURE_MODES = {
    "Words only": (False, False, False),
    "+ URL frequency": (True, False, False),
    "+ Capitalisation frequency": (False, True, False),
    "+ Punctuation frequency": (False, False, True),
    "All structural features": (True, True, True),
}


def build_feature_caches(texts):
    """Precompute Counters once so full-data experiments remain fast."""
    base = []
    extras = {name: [] for name in FEATURE_MODES}
    for text in texts:
        base.append(Counter(tokenize_basic(text)))
        for name, (use_url, use_caps, use_punct) in FEATURE_MODES.items():
            if name == "Words only":
                extras[name].append(Counter())
            else:
                extras[name].append(Counter(extract_structural_features(
                    text, use_url=use_url, use_caps=use_caps, use_punct=use_punct
                )))
    return base, extras


def counts_for_text(text: str, feature_mode: str):
    base = Counter(tokenize_basic(text))
    use_url, use_caps, use_punct = FEATURE_MODES[feature_mode]
    extra = Counter()
    if feature_mode != "Words only":
        extra.update(extract_structural_features(
            text, use_url=use_url, use_caps=use_caps, use_punct=use_punct
        ))
    return base, extra
