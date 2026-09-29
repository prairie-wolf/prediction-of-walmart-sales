"""Command-line entry point: python -m walmart_sales."""

import argparse
from pathlib import Path

from .pipeline import run


def main() -> None:
    project = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Analyze weekly Walmart store sales.")
    parser.add_argument("--data", type=Path, default=project / "Walmart_Sales.csv")
    parser.add_argument("--output", type=Path, default=project / "reports")
    args = parser.parse_args()
    destination = run(args.data, args.output)
    print(f"Analysis written to {destination}")


if __name__ == "__main__":
    main()
