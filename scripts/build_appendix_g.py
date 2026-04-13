"""Build Appendix Г — database models (schemas, views, indexes)."""
import re
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

OUT = r"D:\Dev\ВКР\appendix_g_models.txt"

# Files to include (schema/model SQL — triggers/functions already in Appendix В)
FILES = {
    "sql/01_schema_raw.sql": "Схема RAW — хранилище сырых данных API",
    "sql/02_schema_stg.sql": "Схема STG — нормализованные данные",
    "sql/03_schema_dict.sql": "Схема DICT — справочники продавца",
    "sql/03a_schema_app.sql": "Схема APP — пользователи и аккаунты WB",
    "sql/04_schema_mart.sql": "Схема MART — витрины данных",
    "sql/05_views_and_marts.sql": "Аналитические представления (views)",
    "sql/06_indexes.sql": "Индексы базы данных",
}


def strip_comments(text: str) -> str:
    """Remove explanatory SQL comments but keep section headers."""
    lines = text.split("\n")
    result = []
    for line in lines:
        stripped = line.strip()

        if stripped.startswith("--"):
            bare = stripped.lstrip("- ").strip()
            is_header = (
                "===" in stripped
                or "---" in stripped
                or stripped.startswith("-- FUNCTION")
                or stripped.startswith("-- PROCEDURE")
                or stripped.startswith("-- TRIGGER")
                or stripped.startswith("-- Схема")
                or (bare.isupper() and len(bare) < 80)
            )
            if is_header:
                result.append(line)
            continue
        result.append(line)

    # Remove excessive blank lines (max 2 consecutive)
    cleaned = []
    blank_count = 0
    for line in result:
        if not line.strip():
            blank_count += 1
            if blank_count <= 2:
                cleaned.append(line)
        else:
            blank_count = 0
            cleaned.append(line)

    return "\n".join(cleaned)


# Build output
output_parts = []
output_parts.append("ПРИЛОЖЕНИЕ Г")
output_parts.append("")
output_parts.append("Код моделей базы данных (схемы, представления, индексы)")
output_parts.append("")

for filepath, title in FILES.items():
    full_path = f"D:/ВКР/{filepath}"
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            raw = f.read()
    except FileNotFoundError:
        print(f"WARNING: {full_path} not found, skipping")
        continue

    clean = strip_comments(raw)

    # Count lines
    line_count = len([l for l in clean.split("\n") if l.strip()])

    output_parts.append("=" * 72)
    output_parts.append(f"  {title}")
    output_parts.append(f"  Файл: {filepath} ({line_count} строк)")
    output_parts.append("=" * 72)
    output_parts.append("")
    output_parts.append(clean.rstrip())
    output_parts.append("")
    output_parts.append("")

final = "\n".join(output_parts)

with open(OUT, "w", encoding="utf-8") as f:
    f.write(final)

total_lines = len([l for l in final.split("\n") if l.strip()])
print(f"Appendix Г saved to {OUT}")
print(f"Total non-empty lines: {total_lines}")
print(f"Files included: {len(FILES)}")
