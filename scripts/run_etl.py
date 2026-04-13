from __future__ import annotations

import argparse

from etl import Pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run WB ETL pipeline")
    parser.add_argument("--days-back", type=int, default=7, help="Days back for API dateFrom")
    args = parser.parse_args()

    Pipeline().run(days_back=args.days_back)


if __name__ == "__main__":
    main()
