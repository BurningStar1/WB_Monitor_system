from __future__ import annotations

import argparse

from etl import InternalDictLoader


def main() -> None:
    parser = argparse.ArgumentParser(description="Load internal dictionaries from XLSX")
    parser.add_argument("--cost", required=False, help="Path to cost_reference.xlsx")
    parser.add_argument("--expenses", required=False, help="Path to extra_expenses.xlsx")
    parser.add_argument("--tax", required=False, help="Path to tax_reference.xlsx")
    args = parser.parse_args()

    loader = InternalDictLoader()
    if args.cost:
        loader.load_cost_reference(args.cost)
    if args.expenses:
        loader.load_extra_expenses(args.expenses)
    if args.tax:
        loader.load_tax_reference(args.tax)


if __name__ == "__main__":
    main()
