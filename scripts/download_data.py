"""Fetch the raw dataset into data/raw and record provenance.

Options (pick one):
  python scripts/download_data.py kaggle                 # needs ~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY
  python scripts/download_data.py roboflow --version 1 --api_key XXXX
  python scripts/download_data.py zip --path /path/to/archive.zip

Kaggle mirror  : https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia  (CC BY 4.0)
Upstream       : Kermany, Zhang, Goldbaum (2018), Mendeley Data v2, doi:10.17632/rscbjbr9sj.2  (CC BY 4.0)
Roboflow copy  : https://universe.roboflow.com/mohamed-traore-2ekkp/chest-x-rays-qjmia  (CC BY 4.0)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

SOURCES = {
    "kaggle": dict(url="https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia", licence="CC BY 4.0",
                   upstream="Kermany et al. 2018, Mendeley Data v2 (doi:10.17632/rscbjbr9sj.2)"),
    "roboflow": dict(url="https://universe.roboflow.com/mohamed-traore-2ekkp/chest-x-rays-qjmia", licence="CC BY 4.0",
                     upstream="Kermany et al. 2018 (re-split and re-exported by Roboflow user Mohamed Traore)"),
    "zip": dict(url="(local archive)", licence="CC BY 4.0", upstream="Kermany et al. 2018"),
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", choices=list(SOURCES))
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--version", type=int, default=1, help="roboflow dataset version (use an UN-augmented one)")
    ap.add_argument("--api_key", default=os.environ.get("ROBOFLOW_API_KEY"))
    ap.add_argument("--path", help="zip archive for source=zip")
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    info = dict(SOURCES[a.source], source=a.source, downloaded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))

    if a.source == "kaggle":
        subprocess.run([sys.executable, "-m", "kaggle", "datasets", "download", "-d",
                        "paultimothymooney/chest-xray-pneumonia", "-p", str(out), "--unzip"], check=True)
        info["version"] = "Kaggle dataset version current at download date (5,863 files: 5,216 train / 16 val / 624 test)"
    elif a.source == "roboflow":
        if not a.api_key:
            raise SystemExit("--api_key or ROBOFLOW_API_KEY required")
        from roboflow import Roboflow  # pip install roboflow
        rf = Roboflow(api_key=a.api_key)
        ds = rf.workspace("mohamed-traore-2ekkp").project("chest-x-rays-qjmia").version(a.version)
        ds.download("folder", location=str(out))
        info["version"] = f"chest-x-rays-qjmia/{a.version}"
    else:
        if not a.path:
            raise SystemExit("--path required for source=zip")
        with zipfile.ZipFile(a.path) as z:
            z.extractall(out)
        info["version"] = f"archive {Path(a.path).name}"

    with open(out / "DOWNLOAD_INFO.json", "w") as f:
        json.dump(info, f, indent=2)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
