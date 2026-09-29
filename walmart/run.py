"""Run the project from its root without a package installation step."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from walmart_sales.__main__ import main  # noqa: E402


if __name__ == "__main__":
    main()
