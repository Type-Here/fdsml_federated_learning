"""Make the repository root importable from the tests.

pytest puts the test file's own directory on `sys.path`, not the project root,
so without this `import aggregation_policy` fails when pytest is invoked as
`pytest` rather than `python -m pytest`.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))