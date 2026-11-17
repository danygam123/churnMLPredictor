"""Make the relocated production engine importable from the repository root.

The V2 ML engine now lives in ``vercel-app/ml`` next to the Vercel application it
serves (the Vercel project's Root Directory is ``vercel-app``, so the engine and
its model artifacts must be deployable from there). Tests still import ``ml.``
plainly; this conftest puts the application directory on ``sys.path`` for every
test run. The application's own directory is appended last so its copy of
``model_features.py`` (byte-identical to the root copy) resolves correctly.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
APP_DIR = REPO_ROOT / "vercel-app"

for path in (str(APP_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.append(path)
