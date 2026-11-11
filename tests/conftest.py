"""Make the repository root, starter/ and tests/ importable, as the starter scripts do."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for path in (HERE, os.path.join(ROOT, "starter"), ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)
