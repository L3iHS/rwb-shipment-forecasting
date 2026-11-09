"""Average saved predictions of several runs (e.g. seeds) into one ensemble member.

Seed-averaging before blending keeps the blend optimizer from splitting weight
arbitrarily between near-identical members.

    python scripts/bag_oof.py xgb_lr xgb_lr_s1 xgb_lr_s2 --out xgb_lr_bag
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rwb.data import ROOT

log = logging.getLogger("bag_oof")
OOF = ROOT / "models/oof"


def bag(names: list[str], out: str) -> None:
    parts = [np.load(OOF / f"{n}.npz") for n in names]
    keys = parts[0].files
    merged = {}
    for k in keys:
        if k.startswith("pred_"):
            merged[k] = np.mean([p[k] for p in parts], axis=0)
        else:
            merged[k] = parts[0][k]  # truth is identical across runs
    np.savez_compressed(OOF / f"{out}.npz", **merged)
    log.info("wrote %s.npz from %s", out, names)
    tests = [OOF / f"{n}_test.npy" for n in names]
    if all(t.exists() for t in tests):
        np.save(OOF / f"{out}_test.npy", np.mean([np.load(t) for t in tests], axis=0))
        log.info("wrote %s_test.npy", out)
    else:
        log.warning("test predictions missing for %s; %s_test.npy not written",
                    [n for n, t in zip(names, tests) if not t.exists()], out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("names", nargs="+")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")
    bag(a.names, a.out)


if __name__ == "__main__":
    main()
