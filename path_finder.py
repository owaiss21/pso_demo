import ultralytics, sys
from pathlib import Path
print(Path(ultralytics.__file__).parent / "cfg" / "trackers" / "botsort.yaml")
