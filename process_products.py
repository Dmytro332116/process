#!/usr/bin/env python3
"""Process medical product exports and assign categories."""
from __future__ import annotations

import argparse
import csv
import io
import logging
import re
import sys
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple, Union

import chardet
import pandas as pd
from rapidfuzz import fuzz, process as rf_process

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PRODUCT_COLUMNS = [
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

CATEGORY_LEVELS = ["Категорія1", "Категорія2", "Категорія3"]
TEXT_FIELDS = [
    "Назва",
    "Міжнародне непатентоване найменування",
    "Форма випуску",
    "Склад (діючі)",
    "Фармакотерапевтична група",
]
FORM_FACTOR_KEYWORDS = {
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
ATC_CATEGORY_HINTS = {
    "A11": "Вітаміни",
    "A12": "Мінерали",
    "R05": "Протизастудні",
    "J01": "Антибіотики",
    "N02": "Знеболювальні",
}
NUMERIC_TOKEN_RE = re.compile(r"(\d+)")

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class CategoryRecord:
    """Represents a single category row."""

    level1: str
    level2: str
    level3: str
    combined: str

    @classmethod
    def from_row(cls, row: pd.Series) -> "CategoryRecord":
        level_values = [str(row.get(col, "") or "").strip() for col in CATEGORY_LEVELS]
        combined = " > ".join([val for val in level_values if val])
        return cls(level_values[0], level_values[1], level_values[2], combined)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def detect_encoding(raw: bytes) -> str:
    """Detect file encoding with chardet."""

    result = chardet.detect(raw)
    encoding = result.get("encoding") or "utf-8"
    return encoding


def detect_separator(sample: str) -> str:
    """Detect separator by counting separators in the header sample."""

    header = sample.splitlines()[0] if sample else ""
    comma_count = header.count(",")
    semicolon_count = header.count(";")
    if semicolon_count > comma_count:
        return ";"
    if comma_count > 0:
        return ","
    try:
        dialect = csv.Sniffer().sniff(sample)
        return dialect.delimiter
    except Exception:
        return ","


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names by stripping whitespace and removing BOM."""

    mapping = {}
    for col in df.columns:
        normalized = col.replace("\ufeff", "").strip()
        mapping[col] = normalized
    return df.rename(columns=mapping)


def clean_text(value: Optional[str]) -> str:
    """Clean text values removing artifacts and repeated whitespace."""

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value)
    text = text.replace("\u200b", " ")
    text = text.replace("\xa0", " ")
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean all text fields in a DataFrame."""

    for column in df.columns:
        df[column] = df[column].apply(clean_text)
    return df


def numeric_sort_key(value: Union[str, Path]) -> Tuple[int, str]:
    """Return a numeric-aware sort key for file names."""

    name = Path(value).name
    match = NUMERIC_TOKEN_RE.search(name)
    number = int(match.group(1)) if match else sys.maxsize
    return number, name.lower()


# ---------------------------------------------------------------------------
# File reading helpers
# ---------------------------------------------------------------------------
def read_csv_from_bytes(raw: bytes, source: str) -> pd.DataFrame:
    """Read CSV content from raw bytes."""

    encoding = detect_encoding(raw)
    text = raw.decode(encoding, errors="replace")
    text = text.replace("\ufeff", "")
    sep = detect_separator(text[:1024])
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
        df = pd.read_csv(buffer, dtype=str, on_bad_lines="skip", sep=None, engine="python")
    df["source_file"] = source
    return df


def iter_csv_contents(input_path: Path) -> Iterator[Tuple[str, bytes]]:
    """Yield (name, raw_bytes) for each CSV file in the input path/zip."""

    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(input_path, "r") as archive:
            csv_members = [
                member
                for member in archive.namelist()
                if member.lower().endswith(".csv") and not member.endswith("/")
            ]
            for member in sorted(csv_members, key=numeric_sort_key):
                logging.info("Reading %s from ZIP", member)
                yield member, archive.read(member)
    elif input_path.is_dir():
        csv_files = sorted(
            (path for path in input_path.rglob("*.csv") if path.is_file()),
            key=numeric_sort_key,
        )
        for file_path in csv_files:
            logging.info("Reading %s", file_path)
            yield file_path.name, file_path.read_bytes()
    elif input_path.is_file() and input_path.suffix.lower() == ".csv":
        logging.info("Reading single CSV %s", input_path)
        yield input_path.name, input_path.read_bytes()
    else:
        raise FileNotFoundError(f"Unsupported input path: {input_path}")


def load_products(input_path: Path) -> pd.DataFrame:
    """Load all product CSV files into a single DataFrame."""

    frames: List[pd.DataFrame] = []
    for name, raw in iter_csv_contents(input_path):
        try:
            frame = read_csv_from_bytes(raw, name)
            logging.info("Loaded %s rows from %s", len(frame), name)
            frames.append(frame)
        except Exception as exc:  # pragma: no cover - defensive
            logging.error("Failed to parse %s: %s", name, exc)
    if not frames:
        raise RuntimeError("No CSV data could be loaded from input.")
    df = pd.concat(frames, ignore_index=True)
    logging.info("Merged %s rows from %s files", len(df), len(frames))
    return df


def ensure_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Ensure expected columns exist in DataFrame."""

    for column in columns:
        if column not in df.columns:
            df[column] = ""
    return df


# ---------------------------------------------------------------------------
# Category helpers
# ---------------------------------------------------------------------------
def load_categories(category_path: Path) -> pd.DataFrame:
    """Load category structure CSV."""

    raw = category_path.read_bytes()
    df = read_csv_from_bytes(raw, category_path.name)
    df = normalize_columns(df)
    df = ensure_columns(df, CATEGORY_LEVELS)
    df = df[CATEGORY_LEVELS]
    df = clean_dataframe(df)
    df["CombinedCategory"] = df.apply(
        lambda row: " > ".join([row[col] for col in CATEGORY_LEVELS if row[col]]), axis=1
    )
    return df


def build_category_records(category_df: pd.DataFrame) -> List[CategoryRecord]:
    """Convert category DataFrame to records."""

    records = [CategoryRecord.from_row(row) for _, row in category_df.iterrows()]
    return records


# ---------------------------------------------------------------------------
# Category assignment strategies
# ---------------------------------------------------------------------------
def exact_keyword_match(text: str, records: Sequence[CategoryRecord]) -> Optional[CategoryRecord]:
    """Strategy A: exact keyword matching on category levels."""

    text_lower = text.lower()
    for record in records:
        for level in (record.level3, record.level2, record.level1):
            if level and level.lower() in text_lower:
                return record
    return None


def fuzzy_match(text: str, records: Sequence[CategoryRecord]) -> Optional[CategoryRecord]:
    """Strategy B: fuzzy matching against combined categories."""

    if not text.strip():
        return None
    combined_map = {record.combined: record for record in records if record.combined}
    if not combined_map:
        return None
    match = rf_process.extractOne(text, list(combined_map.keys()), scorer=fuzz.partial_ratio)
    if match and match[1] >= 70:
        return combined_map[match[0]]
    return None


def form_factor_match(text: str, records: Sequence[CategoryRecord]) -> Optional[CategoryRecord]:
    """Strategy C: match by detected form factor keywords."""

    text_lower = text.lower()
    for fragment, keyword in FORM_FACTOR_KEYWORDS.items():
        if fragment in text_lower:
            for record in records:
                if keyword.lower() in record.combined.lower():
                    return record
    return None


def atc_match(atc_code: str, records: Sequence[CategoryRecord]) -> Optional[CategoryRecord]:
    """Strategy D: map ATC code prefixes to categories."""

    if not atc_code:
        return None
    for prefix, category1 in ATC_CATEGORY_HINTS.items():
        if atc_code.upper().startswith(prefix):
            for record in records:
                if record.level1.lower() == category1.lower():
                    return record
    return None


def assign_category_to_row(
    row: pd.Series, records: Sequence[CategoryRecord]
) -> Tuple[str, str, str, str, str]:
    """Assign category levels to a single product row."""

    combined_text = " ".join([row.get(field, "") or "" for field in TEXT_FIELDS])
    combined_text = clean_text(combined_text)
    strategy = ""

    record = exact_keyword_match(combined_text, records)
    if record:
        strategy = "exact"
    else:
        record = fuzzy_match(combined_text, records)
        if record:
            strategy = "fuzzy"
    if not record:
        record = form_factor_match(combined_text, records)
        if record:
            strategy = "form-factor"
    if not record:
        record = atc_match(row.get("Код АТС", ""), records)
        if record:
            strategy = "atc"

    if record:
        return record.level1, record.level2, record.level3, record.combined, strategy
    return "", "", "", "Невизначено", "unassigned"


def assign_categories(df: pd.DataFrame, records: Sequence[CategoryRecord]) -> pd.DataFrame:
    """Assign categories to the full DataFrame using all strategies."""

    stats: Counter[str] = Counter()
    assignments = df.apply(lambda row: assign_category_to_row(row, records), axis=1)
    df[["Категорія1", "Категорія2", "Категорія3", "CombinedCategory", "_strategy"]] = list(assignments)
    stats.update(df["_strategy"].value_counts().to_dict())
    logging.info("Category assignment stats: %s", dict(stats))
    df.drop(columns=["_strategy"], inplace=True)
    return df


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------
def process_data(input_path: Path, category_path: Path, output_path: Path) -> None:
    """Main entry point for processing."""

    logging.info("Loading product data from %s", input_path)
    products = load_products(input_path)
    products = normalize_columns(products)
    products = ensure_columns(products, PRODUCT_COLUMNS)
    products = clean_dataframe(products)
    products.drop_duplicates(subset=["NID"], inplace=True)
    logging.info("Total unique products after deduplication: %s", len(products))

    logging.info("Loading category structure from %s", category_path)
    category_df = load_categories(category_path)
    category_records = build_category_records(category_df)
    logging.info("Loaded %s category records", len(category_records))

    logging.info("Assigning categories...")
    products = assign_categories(products, category_records)

    columns_order = list(dict.fromkeys(PRODUCT_COLUMNS + CATEGORY_LEVELS + ["CombinedCategory"]))
    for column in columns_order:
        if column not in products.columns:
            products[column] = ""
    products = products[columns_order + [col for col in products.columns if col not in columns_order]]

    logging.info("Saving results to %s", output_path)
    products.to_csv(output_path, index=False)
    logging.info("Processing complete. Final product count: %s", len(products))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Process medical product exports.")
    parser.add_argument("--input", required=True, help="Path to input CSV/ZIP folder")
    parser.add_argument("--categories", required=True, help="Path to category CSV file")
    parser.add_argument("--output", required=True, help="Path to output CSV file")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point."""

    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")
    input_path = Path(args.input).expanduser().resolve()
    category_path = Path(args.categories).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    try:
        process_data(input_path, category_path, output_path)
    except Exception as exc:  # pragma: no cover - CLI safety
        logging.error("Processing failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
