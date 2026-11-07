"""Create the fixed equal-weight blend of the team's v2 and weekly forecasts."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def blend(test, peer, weekly):
    if (
        test.empty
        or not {"id", "route_id", "timestamp"}.issubset(test)
        or test.id.isna().any()
        or not test.id.is_unique
        or test[["route_id", "timestamp"]].isna().any().any()
        or test.duplicated(["route_id", "timestamp"]).any()
    ):
        raise ValueError("Test requires unique nonmissing ID and route/time keys")
    aligned = []
    for forecast in (peer, weekly):
        if (
            forecast.columns.tolist() != ["id", "y_pred"]
            or len(forecast) != len(test)
            or not forecast.id.is_unique
            or forecast.id.isna().any()
            or set(forecast.id) != set(test.id)
            or not np.isfinite(forecast.y_pred).all()
            or (forecast.y_pred < 0).any()
        ):
            raise ValueError("Forecast must have one finite nonnegative value for every test ID")
        aligned.append(forecast.set_index("id").loc[test.id, "y_pred"].to_numpy(dtype=np.float64))
    prediction = 0.5 * aligned[0] + 0.5 * aligned[1]
    return (
        pd.DataFrame({"id": test.id.to_numpy(), "y_pred": prediction})
        .sort_values("id")
        .reset_index(drop=True)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--peer-csv", type=Path, default=ROOT / "submissions/teammate_ensemble_v2.csv"
    )
    parser.add_argument(
        "--weekly-csv", type=Path, default=ROOT / "submissions/weekly_v3_latent_half_shares.csv"
    )
    parser.add_argument("--test", type=Path, default=ROOT / "data/raw/test_solo_track.parquet")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "submissions/ensemble_v2_weekly50.csv"
    )
    args = parser.parse_args()
    sidecar = args.output.with_suffix(".metadata.json")
    if args.output.exists() or sidecar.exists() or args.output.resolve() == sidecar.resolve():
        parser.error("Choose a fresh CSV output path")
    inputs = {"peer": args.peer_csv, "weekly": args.weekly_csv, "test": args.test}
    hashes = {name: sha256(path) for name, path in inputs.items()}
    test = pd.read_parquet(args.test, columns=["id", "route_id", "timestamp"])
    result = blend(
        test,
        pd.read_csv(args.peer_csv, float_precision="round_trip"),
        pd.read_csv(args.weekly_csv, float_precision="round_trip"),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".v2_weekly_", dir=args.output.parent) as directory:
        csv = Path(directory) / "forecast.csv"
        metadata = Path(directory) / "metadata.json"
        result.to_csv(csv, index=False, float_format="%.17g")
        restored = pd.read_csv(csv, float_precision="round_trip")
        np.testing.assert_array_equal(restored.to_numpy(), result.to_numpy())
        if hashes != {name: sha256(path) for name, path in inputs.items()}:
            raise RuntimeError("An input changed during generation")
        record = dict(
            model=args.output.stem,
            recipe="equal_weight_peer_weekly",
            rows=len(result),
            input_sha256=hashes,
            weights=dict(peer=0.5, weekly=0.5),
            csv_sha256=sha256(csv),
            public_score=None,
            platform_submission=False,
        )
        metadata.write_text(json.dumps(record, indent=2) + "\n")
        os.link(metadata, sidecar)
        try:
            os.link(csv, args.output)
        except OSError:
            if sidecar.exists() and os.path.samefile(metadata, sidecar):
                sidecar.unlink()
            raise
    print(args.output, record["csv_sha256"])


if __name__ == "__main__":
    main()
