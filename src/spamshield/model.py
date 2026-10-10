"""Efficient from-scratch Multinomial Naive Bayes using cached token counts."""

from __future__ import annotations
from collections import Counter
import math


class MultinomialNaiveBayes:
    """Multinomial Naive Bayes implemented without scikit-learn."""

    def __init__(self, alpha: float = 1.0, max_words: int = 1000):
        if alpha <= 0:
            raise ValueError("alpha must be > 0")
        if max_words <= 0:
            raise ValueError("max_words must be > 0")
        self.alpha = float(alpha)
        self.max_words = int(max_words)
        self.vocabulary = []
        self.log_prior = {}
        self.log_likelihood = {}

    def fit(self, base_counts, labels, extra_counts=None):
        labels = list(labels)
        if extra_counts is None:
            extra_counts = [Counter() for _ in base_counts]
        if len(base_counts) != len(labels) or len(extra_counts) != len(labels):
            raise ValueError("base_counts, extra_counts, and labels must have equal length")

        total_words = Counter()
        for counts in base_counts:
            total_words.update(counts)
        ordinary_vocab = [token for token, _ in total_words.most_common(self.max_words)]
        structural_vocab = sorted({token for counts in extra_counts for token in counts})
        self.vocabulary = ordinary_vocab + structural_vocab
        vocab_set = set(self.vocabulary)
        V = len(self.vocabulary)

        classes = sorted(set(labels))
        for label in classes:
            idx = [i for i, value in enumerate(labels) if value == label]
            self.log_prior[label] = math.log(len(idx) / len(labels))
            aggregate = Counter()
            for i in idx:
                aggregate.update({w: n for w, n in base_counts[i].items() if w in vocab_set})
                aggregate.update(extra_counts[i])
            total = sum(aggregate.values())
            denominator = total + self.alpha * V
            self.log_likelihood[label] = {
                token: math.log((aggregate.get(token, 0) + self.alpha) / denominator)
                for token in self.vocabulary
            }
        return self

    def predict_counts(self, base_counts, extra_counts=None):
        if not self.log_prior:
            raise RuntimeError("Model has not been fitted")
        if extra_counts is None:
            extra_counts = [Counter() for _ in base_counts]
        vocab_set = set(self.vocabulary)
        predictions = []
        for base, extra in zip(base_counts, extra_counts):
            observed = Counter({w: n for w, n in base.items() if w in vocab_set})
            observed.update(extra)
            scores = dict(self.log_prior)
            for token, count in observed.items():
                if token not in vocab_set:
                    continue
                for label in scores:
                    scores[label] += self.log_likelihood[label][token] * count
            predictions.append(max(scores, key=scores.get))
        return predictions

    def to_dict(self):
        return {
            "alpha": self.alpha,
            "max_words": self.max_words,
            "vocabulary": self.vocabulary,
            "log_prior": self.log_prior,
            "log_likelihood": self.log_likelihood,
        }

    @classmethod
    def from_dict(cls, payload):
        model = cls(alpha=payload["alpha"], max_words=payload["max_words"])
        model.vocabulary = list(payload["vocabulary"])
        model.log_prior = {str(k): float(v) for k, v in payload["log_prior"].items()}
        model.log_likelihood = {
            str(label): {str(k): float(v) for k, v in values.items()}
            for label, values in payload["log_likelihood"].items()
        }
        return model
