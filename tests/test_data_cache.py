import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rwb import data


@pytest.fixture
def raw_files(tmp_path, monkeypatch):
    raw = tmp_path / "data/raw"
    raw.mkdir(parents=True)
    monkeypatch.setattr(data, "ROOT", tmp_path)
    monkeypatch.setattr(data, "RAW", raw)
    frame = pd.DataFrame({"route_id": [1, 1], "timestamp": pd.date_range("2025-01-01", periods=2, freq="30min"),
                          "target_1h": [3, 4], **{f"status_{i}": [i, i] for i in range(1, 7)}})
    frame.to_parquet(raw / "train_solo_track.parquet", index=False)
    pd.DataFrame({"id": [0], "route_id": [1], "timestamp": [pd.Timestamp("2025-01-01 01:00")]}).to_parquet(
        raw / "test_solo_track.parquet", index=False)
    return raw


def test_train_change_invalidates_cache_even_when_mtime_is_preserved(raw_files):
    original = data.load_panel()
    path = raw_files / "train_solo_track.parquet"
    stat = path.stat()
    frame = pd.read_parquet(path)
    frame.loc[0, "target_1h"] = 7
    frame.to_parquet(path, index=False)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    updated = data.load_panel()
    assert original.y[0, 0] == 3
    assert updated.y[0, 0] == 7


def test_valid_cache_reused_but_test_file_is_fresh(raw_files, monkeypatch):
    data.load_panel()
    test_path = raw_files / "test_solo_track.parquet"
    test = pd.read_parquet(test_path)
    test["id"] = 42
    test.to_parquet(test_path, index=False)
    original_read = pd.read_parquet

    def read(path, *args, **kwargs):
        assert Path(path).name != "train_solo_track.parquet", "valid cache should avoid loading train"
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", read)
    assert data.load_panel().test.id.tolist() == [42]


@pytest.mark.parametrize("kind", ["junk", "truncated_zip", "legacy"])
def test_corrupt_and_legacy_caches_rebuilt(raw_files, kind):
    cache_path = data.ROOT / "data/processed/panel.npz"
    cache_path.parent.mkdir(parents=True)
    if kind == "legacy":
        np.savez(cache_path, y=np.zeros((1, 2)))
    else:
        cache_path.write_bytes(b"PK\x03\x04truncated" if kind == "truncated_zip" else b"truncated cache")
    assert data.load_panel().y.tolist() == [[3, 4]]
    with np.load(cache_path, allow_pickle=False) as z:
        assert z["cache_version"].item() == data.CACHE_VERSION


def test_cache_false_does_not_write(raw_files):
    data.load_panel(cache=False)
    assert not (data.ROOT / "data/processed/panel.npz").exists()
