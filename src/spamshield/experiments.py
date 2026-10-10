"""Cached experiment runner for repeated Naive Bayes model selection."""

from __future__ import annotations
from collections import Counter
import math

from .metrics import classification_metrics, confusion_counts


class CachedExperimentRunner:
    """Pre-aggregate training counts so repeated experiments stay fast.

    The classifier equations are unchanged; only repeated preprocessing and
    aggregation are cached across alpha/vocabulary/feature sweeps.
    """

    def __init__(self, base_counts, extras, labels, train_idx):
        self.base_counts = base_counts
        self.extras = extras
        self.labels = labels
        self.train_idx = [int(i) for i in train_idx]
        self.classes = sorted(set(labels[self.train_idx]))

        self.train_total = Counter()
        self.train_class = {label: Counter() for label in self.classes}
        self.extra_class = {
            mode: {label: Counter() for label in self.classes}
            for mode in extras
        }

        for i in self.train_idx:
            self.train_total.update(base_counts[i])
            label = labels[i]
            self.train_class[label].update(base_counts[i])
            for mode in extras:
                self.extra_class[mode][label].update(extras[mode][i])

        self.ranked_words = [token for token, _ in self.train_total.most_common()]
        self.class_counts = {
            label: sum(labels[i] == label for i in self.train_idx)
            for label in self.classes
        }
        self.log_prior = {
            label: math.log(self.class_counts[label] / len(self.train_idx))
            for label in self.classes
        }

    def build_model_payload(self, alpha, max_words, feature_mode):
        ordinary = self.ranked_words[:max_words]
        structural = sorted({
            token
            for label in self.classes
            for token in self.extra_class[feature_mode][label]
        })
        vocabulary = ordinary + structural
        vocab_set = set(vocabulary)
        V = len(vocabulary)
        log_likelihood = {}

        for label in self.classes:
            aggregate = Counter({
                token: count
                for token, count in self.train_class[label].items()
                if token in vocab_set
            })
            aggregate.update(self.extra_class[feature_mode][label])
            total = sum(aggregate.values())
            denominator = total + alpha * V
            log_likelihood[label] = {
                token: math.log((aggregate.get(token, 0) + alpha) / denominator)
                for token in vocabulary
            }

        return {
            "alpha": float(alpha),
            "max_words": int(max_words),
            "vocabulary": vocabulary,
            "log_prior": dict(self.log_prior),
            "log_likelihood": log_likelihood,
        }

    def predict(self, payload, eval_idx, feature_mode):
        vocab_set = set(payload["vocabulary"])
        predictions = []
        for raw_i in eval_idx:
            i = int(raw_i)
            observed = Counter({
                token: count
                for token, count in self.base_counts[i].items()
                if token in vocab_set
            })
            observed.update(self.extras[feature_mode][i])

            scores = dict(payload["log_prior"])
            for token, count in observed.items():
                if token not in vocab_set:
                    continue
                for label in self.classes:
                    scores[label] += payload["log_likelihood"][label][token] * count
            predictions.append(max(scores, key=scores.get))
        return predictions

    def evaluate(self, alpha, max_words, feature_mode, eval_idx):
        payload = self.build_model_payload(alpha, max_words, feature_mode)
        pred = self.predict(payload, eval_idx, feature_mode)
        true = self.labels[eval_idx]
        return classification_metrics(true, pred), confusion_counts(true, pred), payload
