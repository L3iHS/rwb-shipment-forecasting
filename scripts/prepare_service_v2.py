"""Recalibrate the saved service for reconciled flow, without training a model."""

import argparse
import logging
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rwb.data import HORIZON, load_panel
from rwb.features import cumsums
from rwb.forecaster import Forecaster
from rwb.service import Service, ServiceConfig
from compare_service_models import sha256, write_json


def prepare(output: Path) -> Service:
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a fresh bundle directory; the original is never overwritten")
    panel = load_panel(cache=False)
    cums = cumsums(panel)
    forecaster = Forecaster.load(ROOT / "models/prod/service/model")
    with np.load(ROOT / "models/rolling/lean_fold0.npz", allow_pickle=False) as z:
        origins, pred = z["origins"], z["pred"].astype(np.float64)
        cutoff = int(z["label_cutoff"])
    if cutoff > origins.min() or origins.max() + HORIZON >= panel.n_times:
        raise ValueError("Calibration archive violates label availability")
    service = Service.calibrate(forecaster, cums, panel, origins, pred, ServiceConfig())
    service.calibration_metadata.update({"archive_sha256": sha256(ROOT / "models/rolling/lean_fold0.npz"),
        "training_cutoff_of_calibration_forecaster": cutoff,
        "train_sha256": sha256(ROOT / "data/raw/train_solo_track.parquet")})
    service.save(output)
    write_json(output / "provenance.json", {"trained_new_model": False, "calibration": service.calibration_metadata,
        "model_files": {p.name: sha256(p) for p in sorted((output / "model").glob("*"))}})
    logging.info("Saved recalibrated service to %s, using %d origins", output, len(origins))
    return service


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "models/prod/service_v2")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    prepare(args.output.resolve())
