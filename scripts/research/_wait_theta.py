import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.api.thetadata.thetadata_client import get_thetadata_client
c = get_thetadata_client()
for i in range(30):
    if c.is_ready():
        print("ready", i)
        raise SystemExit(0)
    time.sleep(2)
print("not ready")
raise SystemExit(1)
