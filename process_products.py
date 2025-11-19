import os
import zipfile
import pandas as pd
import chardet
from rapidfuzz import process, fuzz
import argparse

# -------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------
EXPECTED_COLUMNS = [
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

CATEGORY_COLUMNS = ["Категорія1", "Категорія2", "Категорія3"]


# -------------------------------------------------------------
# Detect file encoding
# -------------------------------------------------------------
def detect_encoding(raw_bytes):
    result = chardet.detect(raw_bytes)
    return result["encoding"] or "utf-8"


# -------------------------------------------------------------
# Load a single CSV with encoding auto-detection
# -------------------------------------------------------------
def load_csv_safe(path):
    with open(path, "rb") as f:
        raw = f.read()

    encoding = detect_encoding(raw)

    try:
        df = pd.read_csv(path, encoding=encoding, dtype=str)
    except Exception:
        df = pd.read_csv(path, encoding="utf-8", dtype=str, errors="ignore")

    return df


# -------------------------------------------------------------
# Load CSVs from ZIP
# -------------------------------------------------------------
def load_products(input_zip):
    frames = []

    with zipfile.ZipFile(input_zip, "r") as z:
        csv_files = [f for f in z.namelist() if f.lower().endswith(".csv")]

        for name in sorted(csv_files):
            print(f"→ Читаю: {name}")
            try:
                raw = z.read(name)
                encoding = chardet.detect(raw)["encoding"] or "utf-8"
                df = pd.read_csv(pd.compat.StringIO(raw.decode(encoding)), dtype=str)
                df["source_file"] = name
                frames.append(df)
            except Exception:
                continue

    if not frames:
        raise Exception("У ZIP не знайдено жодного CSV-файлу")

    df = pd.concat(frames, ignore_index=True)
    return df


# -------------------------------------------------------------
# Normalize column names
# -------------------------------------------------------------
def normalize_columns(df):
    mapping = {}
    for col in df.columns:
        key = col.strip()
        mapping[col] = key
    df = df.rename(columns=mapping)
    return df


# -------------------------------------------------------------
# Convert "Категорія" → Категорія1 / Категорія2 / Категорія3
# -------------------------------------------------------------
def split_categories(df):
    if "Категорія" not in df.columns:
        for c in CATEGORY_COLUMNS:
            df[c] = None
        return df

    def expand(cat):
        if pd.isna(cat):
            return [None, None, None]
        parts = [p.strip() for p in str(cat).split("|")]
        parts += [None] * (3 - len(parts))
        return parts[:3]

    expanded = df["Категорія"].apply(expand)
    df[CATEGORY_COLUMNS] = pd.DataFrame(expanded.tolist(), index=df.index)
    return df


# -------------------------------------------------------------
# Main processing
# -------------------------------------------------------------
def process_products(input_zip, output_csv):
    print("🔍 Завантажую таблиці з ZIP...")
    df = load_products(input_zip)

    print("🔧 Нормалізую назви колонок...")
    df = normalize_columns(df)

    # Ensure all expected columns exist
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = None

    print("🔍 Розбиваю категорії...")
    df = split_categories(df)

    print("💾 Зберігаю у файл...")
    df.to_csv(output_csv, index=False)

    print(f"✔️ Готово! Збережено у {output_csv}")
    print(f"📦 Кількість товарів: {len(df)}")


# -------------------------------------------------------------
# CLI
# -------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    process_products(args.input, args.output)
