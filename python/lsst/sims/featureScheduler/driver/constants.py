import os
from pathlib import Path

RUN_SCRIPT_NAME = 'scheduler.py'
BASE_PACKAGE_PATH = str(Path(__file__).parent.parent.parent.parent.parent.parent)
RUN_SCRIPT = os.path.join(BASE_PACKAGE_PATH, 'scripts/', RUN_SCRIPT_NAME)
CONFIG_NAME = 'feature_scheduler.py'
