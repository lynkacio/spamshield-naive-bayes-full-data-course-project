"""Data loading and deterministic stratified splitting.

The supplied raw workbook contains a small number of legacy XML control bytes
inside email bodies. They are removed *in memory* before XML parsing; the raw
file on disk is never modified.
"""

from __future__ import annotations

from pathlib import Path
import re
import zipfile
import xml.etree.ElementTree as ET
import numpy as np

_INVALID_XML_BYTES = re.compile(rb"[\x00-\x08\x0B\x0C\x0E-\x1F]")
_NS = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def _column_index(cell_ref: str) -> int:
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    value = 0
    for ch in letters:
        value = value * 26 + (ord(ch.upper()) - ord("A") + 1)
    return value - 1


def _coerce_number(value: str):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    return int(number) if number.is_integer() else number


def load_xlsx_records(path: str | Path, worksheet_xml: str = "xl/worksheets/sheet1.xml"):
    """Load the Data sheet directly from the project XLSX using stdlib only.

    Returns
    -------
    records : list[dict]
        One dictionary per email.
    removed_control_bytes : int
        Number of invalid XML 1.0 control bytes removed in memory.
    """
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        raw_xml = archive.read(worksheet_xml)

    clean_xml, removed = _INVALID_XML_BYTES.subn(b"", raw_xml)
    root = ET.fromstring(clean_xml)

    rows = []
    for row in root.findall(".//x:sheetData/x:row", _NS):
        cells = {}
        for cell in row.findall("x:c", _NS):
            col = _column_index(cell.attrib["r"])
            cell_type = cell.attrib.get("t")
            if cell_type == "inlineStr":
                text_nodes = cell.findall(".//x:t", _NS)
                value = "".join(node.text or "" for node in text_nodes)
            else:
                value_node = cell.find("x:v", _NS)
                value = None if value_node is None else _coerce_number(value_node.text)
            cells[col] = value
        if cells:
            max_col = max(cells)
            rows.append([cells.get(i) for i in range(max_col + 1)])

    if not rows:
        return [], removed

    headers = rows[0]
    records = []
    for row in rows[1:]:
        padded = row + [None] * (len(headers) - len(row))
        records.append(dict(zip(headers, padded)))
    return records, removed


def stratified_split_indices(labels, test_ratio: float = 0.20, seed: int = 42):
    """Split each class independently to preserve the ham/spam ratio."""
    labels = np.asarray(labels, dtype=object)
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []

    for label in np.unique(labels):
        idx = np.where(labels == label)[0].copy()
        rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * test_ratio)))
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])

    train_idx = np.asarray(train_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return train_idx, test_idx
