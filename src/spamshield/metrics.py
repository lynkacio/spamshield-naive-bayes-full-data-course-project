"""Binary classification metrics implemented without scikit-learn."""


def confusion_counts(y_true, y_pred):
    tp = tn = fp = fn = 0
    for truth, pred in zip(y_true, y_pred):
        if truth == "spam" and pred == "spam": tp += 1
        elif truth == "ham" and pred == "ham": tn += 1
        elif truth == "ham" and pred == "spam": fp += 1
        elif truth == "spam" and pred == "ham": fn += 1
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}


def classification_metrics(y_true, y_pred):
    c = confusion_counts(y_true, y_pred)
    tp, tn, fp, fn = c["TP"], c["TN"], c["FP"], c["FN"]
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
