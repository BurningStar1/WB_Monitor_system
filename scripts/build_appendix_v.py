"""Build Appendix В — clean code without explanatory comments."""
import re
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

OUT = r"D:\Dev\ВКР\appendix_v_code.txt"

# Files to include
FILES = {
    # ETL modules
    "etl/pipeline.py": "Класс Pipeline — координатор ETL-процесса",
    "etl/raw_loader.py": "Класс RawLoader — загрузка сырых данных из WB API",
    "etl/stg_loader.py": "Класс StgLoader — нормализация данных (RAW → STG)",
    "etl/mart_loader.py": "Класс MartLoader — агрегация данных (STG → MART)",
    # SQL functions and procedures
    "sql/07_triggers_functions_procedures.sql": "Триггеры, функции и процедуры БД",
    # Streamlit pages
    "app/Home.py": "Главная страница Streamlit",
    "app/pages/01_KPI_Dashboard.py": "Страница KPI-дашборда",
    "app/pages/02_Weekly_Report.py": "Страница еженедельного отчёта",
    "app/pages/03_Article_Report.py": "Страница отчёта по артикулам",
    "app/pages/04_Stocks_Report.py": "Страница отчёта по остаткам",
    "app/pages/05_ABC_Analysis.py": "Страница ABC-анализа",
    "app/pages/06_Profit_Report.py": "Страница отчёта о прибыли",
    "app/pages/07_Statutory_Report.py": "Страница регламентного отчёта",
}


def strip_comments(text: str, lang: str) -> str:
    """Remove explanatory comments but keep section headers."""
    lines = text.split("\n")
    result = []
    for line in lines:
        stripped = line.strip()

        if lang == "py":
            # Keep lines that are NOT pure comments
            if stripped.startswith("#"):
                # Keep header-like comments (short, uppercase, dashes, section markers)
                bare = stripped.lstrip("# ").strip()
                # Keep: section headers with --- or === or ── or capitalized short titles
                is_header = (
                    "---" in stripped
                    or "===" in stripped
                    or "──" in stripped
                    or "═" in stripped
                    or (bare.startswith(("# ", "## ")) and len(bare) < 60)
                    or (bare.isupper() and len(bare) < 60)
                    or bare.startswith("---")
                )
                if is_header:
                    result.append(line)
                # Skip explanatory comments
                continue
            # Remove inline explanatory comments but keep short markers
            if "  #" in line and not any(kw in line for kw in ["# ---", "# ===", "# ──"]):
                # Remove inline comment
                code_part = line.split("  #")[0].rstrip()
                if code_part.strip():
                    result.append(code_part)
                    continue
            # Skip docstrings (triple-quoted single-line)
            if stripped.startswith('"""') and stripped.endswith('"""') and len(stripped) > 6:
                continue
            result.append(line)

        elif lang == "sql":
            # Keep section headers (lines with === or --- or block comments that are headers)
            if stripped.startswith("--"):
                bare = stripped.lstrip("- ").strip()
                is_header = (
                    "===" in stripped
                    or "---" in stripped
                    or stripped.startswith("-- FUNCTION")
                    or stripped.startswith("-- PROCEDURE")
                    or stripped.startswith("-- TRIGGER")
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
output_parts.append("ПРИЛОЖЕНИЕ В")
output_parts.append("")
output_parts.append("Код программной реализации ETL-модулей, функций БД и интерфейса")
output_parts.append("")

for filepath, title in FILES.items():
    full_path = f"D:/ВКР/{filepath}"
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            raw = f.read()
    except FileNotFoundError:
        print(f"WARNING: {full_path} not found, skipping")
        continue

    lang = "sql" if filepath.endswith(".sql") else "py"
    clean = strip_comments(raw, lang)

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
print(f"Appendix В saved to {OUT}")
print(f"Total non-empty lines: {total_lines}")
print(f"Files included: {len(FILES)}")
