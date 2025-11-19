#!/usr/bin/env python3
"""Process medical product exports and assign catalog categories."""
from __future__ import annotations

import argparse
import csv
import io
import logging
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import chardet
import pandas as pd
from rapidfuzz import fuzz, process as rf_process

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PRODUCT_COLUMNS: Sequence[str] = [
    "NID",
    "Назва",
    "Міжнародне непатентоване найменування",
    "Форма випуску",
    "Склад (діючі)",
    "Фармакотерапевтична група",
    "Код АТС",
    "Виробник",
    "Виробник.1",
    "Інструкція",
    "Категорія",
]

CATEGORY_LEVELS: Sequence[str] = ["Категорія1", "Категорія2", "Категорія3"]
TEXT_FIELDS: Sequence[str] = [
    "Назва",
    "Міжнародне непатентоване найменування",
    "Форма випуску",
    "Склад (діючі)",
    "Фармакотерапевтична група",
]
FORM_FACTOR_KEYWORDS: Dict[str, str] = {
    "крапл": "краплі",
    "таблет": "таблетки",
    "капсул": "капсули",
    "сироп": "сироп",
    "маз": "мазь",
    "крем": "крем",
    "спрей": "спрей",
    "суспенз": "суспензія",
    "розчин": "розчин",
    "порош": "порошок",
    "супозитор": "супозиторії",
    "гель": "гель",
}
ATC_CATEGORY_HINTS: Dict[str, str] = {
    "A11": "Вітаміни",
    "A12": "Мінерали",
    "R05": "Протизастудні",
    "J01": "Антибіотики",
    "N02": "Знеболювальні",
}
NUMERIC_TOKEN_RE = re.compile(r"(\d+)")
NON_ALNUM_RE = re.compile(r"[^0-9A-Za-zА-Яа-яЇїІіЄєҐґ]+")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CategoryRecord:
    """Normalized representation of a single catalog row."""

    level1: str
    level2: str
    level3: str
    combined: str

    @classmethod
    def from_row(cls, row: pd.Series) -> "CategoryRecord":
        levels = [str(row.get(col, "") or "").strip() for col in CATEGORY_LEVELS]
        combined = " > ".join([value for value in levels if value])
        return cls(levels[0], levels[1], levels[2], combined)

    @property
    def depth(self) -> int:
        if self.level3:
            return 3
        if self.level2:
            return 2
        if self.level1:
            return 1
        return 0


@dataclass
class CategoryContext:
    """Reusable helper structures for matching strategies."""

    records: Sequence[CategoryRecord]
    exact_patterns: List[Tuple[CategoryRecord, List[re.Pattern]]]
    fuzzy_choices: List[str]
    fuzzy_map: Dict[str, CategoryRecord]
    form_factor_map: Dict[str, List[CategoryRecord]]


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------
def clean_text(value: Optional[str]) -> str:
    """Normalize text values (strip, collapse spaces, remove control chars)."""

    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = str(value)
    text = text.replace("\ufeff", " ").replace("\u200b", " ").replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text.strip())
    return text


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean every column in the DataFrame."""

    for column in df.columns:
        df[column] = df[column].apply(clean_text)
    return df


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------
def detect_encoding(raw: bytes) -> str:
    """Detect text encoding using chardet, fallback to UTF-8."""

    guess = chardet.detect(raw)
    return guess.get("encoding") or "utf-8"


def detect_separator(sample: str) -> str:
    """Detect CSV separator by inspecting the header sample."""

    lines = sample.splitlines()[:5]
    joined = "\n".join(lines)
    comma_count = joined.count(",")
    semicolon_count = joined.count(";")
    if semicolon_count > comma_count:
        return ";"
    if comma_count > 0:
        return ","
    try:
        dialect = csv.Sniffer().sniff(joined)
        return dialect.delimiter
    except Exception:
        return ","


def read_csv_from_bytes(raw: bytes, source: str) -> pd.DataFrame:
    """Decode bytes and load as pandas DataFrame with best-effort parsing."""

    encoding = detect_encoding(raw)
    text = raw.decode(encoding, errors="replace")
    sample = text[:2048]
    sep = detect_separator(sample)
    buffer = io.StringIO(text)
    try:
        df = pd.read_csv(
            buffer,
            sep=sep,
            dtype=str,
            on_bad_lines="skip",
            engine="python" if sep != "," else "c",
        )
    except Exception:
        buffer.seek(0)
        df = pd.read_csv(buffer, dtype=str, sep=None, engine="python", on_bad_lines="skip")
    df["source_file"] = source
    return df


def numeric_sort_key(value: str | Path) -> Tuple[int, str]:
    """Return numeric-aware key so export-2 comes before export-10."""

    name = Path(value).name
    match = NUMERIC_TOKEN_RE.search(name)
    order = int(match.group(1)) if match else sys.maxsize
    return order, name.lower()


def iter_csv_contents(input_path: Path) -> Iterator[Tuple[str, bytes]]:
    """Yield every CSV file (name, bytes) from directory/zip/single file."""

    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(input_path, "r") as archive:
            members = [
                info
                for info in archive.infolist()
                if not info.is_dir() and info.filename.lower().endswith(".csv")
            ]
            members.sort(key=lambda info: numeric_sort_key(info.filename))
            logging.info("ZIP archive contains %s CSV files", len(members))
            for info in members:
                logging.info("Reading %s from ZIP", info.filename)
                yield info.filename, archive.read(info)
    elif input_path.is_dir():
        files = sorted((p for p in input_path.rglob("*.csv") if p.is_file()), key=numeric_sort_key)
        logging.info("Directory contains %s CSV files", len(files))
        for file_path in files:
            logging.info("Reading %s", file_path)
            yield file_path.name, file_path.read_bytes()
    elif input_path.is_file() and input_path.suffix.lower() == ".csv":
        logging.info("Reading single CSV %s", input_path)
        yield input_path.name, input_path.read_bytes()
    else:
        raise FileNotFoundError(f"Unsupported input path: {input_path}")


def load_products(input_path: Path) -> pd.DataFrame:
    """Load and merge every CSV file from the provided source."""

    frames: List[pd.DataFrame] = []
    for name, raw in iter_csv_contents(input_path):
        try:
            frame = read_csv_from_bytes(raw, name)
            logging.info("Loaded %s rows from %s", len(frame), name)
            frames.append(frame)
        except Exception as exc:  # pragma: no cover - defensive
            logging.error("Failed to parse %s: %s", name, exc)
    if not frames:
        raise RuntimeError("No CSV data could be loaded from the input.")
    merged = pd.concat(frames, ignore_index=True)
    logging.info("Merged %s rows from %s files", len(merged), len(frames))
    return merged


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace/BOM from column names."""

    mapping = {column: column.replace("\ufeff", "").strip() for column in df.columns}
    return df.rename(columns=mapping)


def ensure_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Ensure columns exist (create empty ones if missing)."""

    for column in columns:
        if column not in df.columns:
            df[column] = ""
    return df


# ---------------------------------------------------------------------------
# Category helpers
# ---------------------------------------------------------------------------
def load_categories(category_path: Path) -> pd.DataFrame:
    """Load category structure CSV with cleaning."""

    raw = category_path.read_bytes()
    df = read_csv_from_bytes(raw, category_path.name)
    df = normalize_columns(df)
    df = ensure_columns(df, CATEGORY_LEVELS)
    df = df[list(CATEGORY_LEVELS)]
    df = clean_dataframe(df)
    df["CombinedCategory"] = df.apply(
        lambda row: " > ".join([row[level] for level in CATEGORY_LEVELS if row[level]]),
        axis=1,
    )
    return df


def build_category_records(category_df: pd.DataFrame) -> List[CategoryRecord]:
    """Convert category rows into dataclass records."""

    return [CategoryRecord.from_row(row) for _, row in category_df.iterrows()]


def build_word_pattern(phrase: str) -> re.Pattern:
    """Create regex that matches the phrase as whole words."""

    normalized = clean_text(phrase).lower()
    if not normalized:
        return re.compile(r"^$")
    tokens = [re.escape(token) for token in normalized.split() if token]
    joined = r"\b" + r"\s+".join(tokens) + r"\b"
    return re.compile(joined)


def build_category_context(records: Sequence[CategoryRecord]) -> CategoryContext:
    """Pre-compute match helpers for every category record."""

    exact_patterns: List[Tuple[CategoryRecord, List[re.Pattern]]] = []
    for record in records:
        patterns: List[re.Pattern] = []
        for level in (record.level3, record.level2, record.level1):
            if level:
                patterns.append(build_word_pattern(level))
        exact_patterns.append((record, patterns))

    fuzzy_choices = [record.combined for record in records if record.combined]
    fuzzy_map = {record.combined: record for record in records if record.combined}

    form_factor_map: Dict[str, List[CategoryRecord]] = {}
    for keyword in {value.lower() for value in FORM_FACTOR_KEYWORDS.values()}:
        form_factor_map[keyword] = []
    for record in records:
        combined_lower = record.combined.lower()
        for keyword in form_factor_map:
            if keyword and keyword in combined_lower:
                form_factor_map[keyword].append(record)

    return CategoryContext(records, exact_patterns, fuzzy_choices, fuzzy_map, form_factor_map)


# ---------------------------------------------------------------------------
# Matching strategies
# ---------------------------------------------------------------------------
def combine_product_text(row: pd.Series) -> str:
    """Combine key textual fields into a single search string."""

    parts = [row.get(field, "") or "" for field in TEXT_FIELDS]
    parts.append(row.get("source_file", ""))
    text = " ".join(parts)
    text = clean_text(text)
    return text.lower()


def exact_keyword_match(text: str, context: CategoryContext) -> Optional[CategoryRecord]:
    """Strategy A: exact keyword match using precompiled patterns."""

    for record, patterns in context.exact_patterns:
        for pattern in patterns:
            if pattern.search(text):
                return record
    return None


def fuzzy_match(text: str, context: CategoryContext) -> Tuple[Optional[CategoryRecord], Optional[float]]:
    """Strategy B: fuzzy match combined text against category names."""

    if not text.strip() or not context.fuzzy_choices:
        return None, None
    match = rf_process.extractOne(text, context.fuzzy_choices, scorer=fuzz.WRatio)
    if not match:
        return None, None
    name, score, _ = match
    if score >= 60:
        return context.fuzzy_map.get(name), float(score)
    return None, float(score)


def form_factor_match(text: str, context: CategoryContext) -> Optional[CategoryRecord]:
    """Strategy C: detect dosage/form keywords and choose deepest category."""

    candidates: List[Tuple[int, CategoryRecord]] = []
    for needle, keyword in FORM_FACTOR_KEYWORDS.items():
        if needle in text:
            keyword_lower = keyword.lower()
            for record in context.form_factor_map.get(keyword_lower, []):
                candidates.append((record.depth, record))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], item[1].combined))
    return candidates[0][1]


def normalize_atc_code(code: str) -> str:
    """Normalize ATC codes to a three-character uppercase prefix."""

    cleaned = re.sub(r"[^0-9A-Za-z]", "", code or "").upper()
    return cleaned[:3]


def atc_match(atc_code: str, context: CategoryContext) -> Optional[CategoryRecord]:
    """Strategy D: map ATC prefixes to top-level hints."""

    prefix = normalize_atc_code(atc_code)
    if not prefix:
        return None
    hint = ATC_CATEGORY_HINTS.get(prefix)
    if not hint:
        return None
    hint_lower = hint.lower()
    for record in context.records:
        if record.level1.lower() == hint_lower:
            return record
    return None


def assign_category_to_row(row: pd.Series, context: CategoryContext) -> Tuple[str, str, str, str, str, Optional[float]]:
    """Run all strategies for a single row and return category info."""

    text = combine_product_text(row)
    strategy = "unassigned"
    fuzzy_score: Optional[float] = None

    record = exact_keyword_match(text, context)
    if record:
        strategy = "exact"
    else:
        record, fuzzy_score = fuzzy_match(text, context)
        if record:
            strategy = "fuzzy"
    if not record:
        record = form_factor_match(text, context)
        if record:
            strategy = "form-factor"
    if not record:
        record = atc_match(row.get("Код АТС", ""), context)
        if record:
            strategy = "atc"

    if record:
        logging.debug(
            "Row text: %s -> %s via %s%s",
            text,
            record.combined or "(empty)",
            strategy,
            f" (score={fuzzy_score:.1f})" if fuzzy_score is not None else "",
        )
        return record.level1, record.level2, record.level3, record.combined, strategy, fuzzy_score

    logging.debug("Row text: %s -> Невизначено via unassigned", text)
    return "", "", "", "Невизначено", "unassigned", fuzzy_score


def assign_categories(df: pd.DataFrame, records: Sequence[CategoryRecord]) -> pd.DataFrame:
    """Assign categories to every row and log assignment stats."""

    context = build_category_context(records)
    assignments = df.apply(lambda row: assign_category_to_row(row, context), axis=1)
    df[["Категорія1", "Категорія2", "Категорія3", "CombinedCategory", "_strategy", "_score"]] = list(assignments)
    stats = df["_strategy"].value_counts().to_dict()
    logging.info("Category assignment stats: %s", stats)
    df.drop(columns=["_strategy", "_score"], inplace=True)
    return df


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------
def process_data(input_path: Path, category_path: Path, output_path: Path) -> None:
    """Full processing pipeline."""

    logging.info("Loading product CSVs from %s", input_path)
    products = load_products(input_path)
    products = normalize_columns(products)
    products = ensure_columns(products, PRODUCT_COLUMNS)
    products = clean_dataframe(products)
    if "NID" in products.columns:
        before = len(products)
        products.drop_duplicates(subset=["NID"], inplace=True)
        logging.info("Deduplicated by NID: %s -> %s rows", before, len(products))
    else:
        logging.warning("Column NID missing; skipping deduplication")

    logging.info("Loading category structure from %s", category_path)
    category_df = load_categories(category_path)
    category_records = build_category_records(category_df)
    logging.info("Loaded %s category records", len(category_records))

    logging.info("Assigning categories ...")
    products = assign_categories(products, category_records)

    ordered_columns = list(dict.fromkeys(list(PRODUCT_COLUMNS) + list(CATEGORY_LEVELS) + ["CombinedCategory"]))
    for column in ordered_columns:
        if column not in products.columns:
            products[column] = ""
    extra_columns = [col for col in products.columns if col not in ordered_columns]
    products = products[ordered_columns + extra_columns]

    logging.info("Writing %s rows to %s", len(products), output_path)
    products.to_csv(output_path, index=False)
    logging.info("Processing complete")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Process medical product exports and assign categories.")
    parser.add_argument("--input", required=True, help="Path to CSV file, directory, or ZIP archive")
    parser.add_argument("--categories", required=True, help="Path to category structure CSV")
    parser.add_argument("--output", required=True, help="Path to save the resulting CSV")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Entry point."""

    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")
    input_path = Path(args.input).expanduser().resolve()
    category_path = Path(args.categories).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    try:
        process_data(input_path, category_path, output_path)
    except Exception as exc:  # pragma: no cover - CLI safety net
        logging.error("Processing failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
