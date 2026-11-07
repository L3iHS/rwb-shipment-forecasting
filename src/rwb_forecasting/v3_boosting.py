"""Explicit standalone T_h1s600 predictor attributed to Vlado's pinned source.

Statistical recipe: commit bd61c6d560f4a838dc436fbc13139fa42ccb99be,
experiments/exp_gbm.py, blob d09105b0cf3f31c442d7743db24f50c554cd6786.
The default native-thread change10->2 is a recorded resource override.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import platform
import resource
import tempfile

import numpy as np

from .v3_features import HORIZON, SOURCE_COMMIT, SOURCE_GIT_BLOB, SOURCE_SHA256, SOURCE_URL, V3Context, fixed_recipe

CANDIDATE = "vlado_t_h1s600_reference"


@dataclass(frozen=True)
class RuntimeConfig:
    native_threads: int = 2

    def __post_init__(self):
        if type(self.native_threads) is not int or self.native_threads < 1:
            raise ValueError("A positive native thread count is required")


def native_parameters(runtime=RuntimeConfig()):
    """Fresh exact pinned native parameters except the explicit thread override."""
    return dict(objective="tweedie", learning_rate=.05, num_leaves=63,
        min_data_in_leaf=200, feature_fraction=.8, bagging_fraction=.8,
        bagging_freq=1, lambda_l2=1., verbose=-1,
        num_threads=runtime.native_threads, tweedie_variance_power=1.3, seed=42)


@dataclass
class TrainingDesign:
    features: np.memmap
    raw_labels: np.memmap
    scales: np.memmap
    normalized_labels: np.memmap
    feature_names: list[str]
    sources: np.ndarray
    directory: Path

    @property
    def model_columns(self):
        return [column for column in self.feature_names if column != "route_id"]

    @property
    def model_matrix(self):
        if self.feature_names[-1] != "route_id":
            raise ValueError("Pinned final route column changed")
        return self.features[:, :-1]


@contextmanager
def training_design(context, work_directory):
    """Source-major float64 memmaps; one source dataframe at a time, exact pinned rows."""
    work_directory = Path(work_directory)
    work_directory.mkdir(parents=True, exist_ok=True)
    sources = context.training_sources()
    first, _ = context.features(int(sources[0]))
    columns = list(first)
    block_rows = len(context.routes) * HORIZON
    count = len(sources) * block_rows
    with tempfile.TemporaryDirectory(prefix="vlado_v3_memmap_", dir=work_directory) as folder:
        directory = Path(folder)
        feature = np.memmap(directory / "features.f64", mode="w+", dtype="float64", shape=(count, len(columns)))
        maps = [feature]
        try:
            vectors = []
            for name in ["raw_labels", "scales", "normalized_labels"]:
                vector = np.memmap(directory / (name + ".f64"), mode="w+", dtype="float64", shape=(count,))
                maps.append(vector); vectors.append(vector)
            labels, scales, normalized = vectors
            known = columns.index("known"); horizon = columns.index("h")
            del first
            for number, source in enumerate(sources):
                frame, scale = context.features(int(source))
                if list(frame) != columns:
                    raise ValueError("Source feature order changed")
                start = number * block_rows; end = start + block_rows
                feature[start:end] = frame.to_numpy(dtype=np.float64)
                scales[start:end] = np.tile(scale, HORIZON)
                labels[start:end] = np.concatenate([context.raw_target[source + step] for step in range(1, HORIZON + 1)])
                offset = np.where(feature[start:end, horizon] == 1, feature[start:end, known], 0.)
                normalized[start:end] = np.maximum(labels[start:end] / scales[start:end] - offset, 0.)
                del frame
            for array in maps:
                array.flush()
            yield TrainingDesign(feature, labels, scales, normalized, columns, sources, directory)
        finally:
            for array in maps:
                array.flush(); array._mmap.close()


def fit_native(design, current, runtime=RuntimeConfig(), *, dataset_factory=None, trainer=None):
    if dataset_factory is None or trainer is None:
        import lightgbm as lgb
        dataset_factory = dataset_factory or lgb.Dataset
        trainer = trainer or lgb.train
    if list(current) != design.feature_names:
        raise ValueError("Current feature order changed")
    dataset = dataset_factory(design.model_matrix, label=design.normalized_labels,
        weight=design.scales, feature_name=design.model_columns,
        categorical_feature=[], free_raw_data=True)
    booster = trainer(native_parameters(runtime), dataset, num_boost_round=600)
    if booster.current_iteration() != 600:
        raise ValueError("Native tree count differs from fixed600")
    raw = booster.predict(current[design.model_columns], num_iteration=600,
                          num_threads=runtime.native_threads)
    return np.asarray(raw, dtype=np.float64)


def predict(history, work_directory, runtime=RuntimeConfig(), *, fitter=fit_native):
    """Return standalone raw route-by-eight hourly predictions and a causal training audit."""
    context = V3Context.from_history(history)
    current, scale = context.features(len(context.times) - 1)
    current_scale = np.tile(scale, HORIZON)
    with training_design(context, work_directory) as design:
        raw = fitter(design, current, runtime)
        offsets = np.where(current.h.to_numpy() == 1, current.known.to_numpy(), 0.)
        if raw.shape != current_scale.shape or not np.isfinite(raw).all():
            raise ValueError("Eight finite route-aligned native horizons required")
        values = (np.maximum(raw, 0.) + offsets) * current_scale
        result = values.reshape(HORIZON, len(context.routes)).T
        audit = dict(candidate=CANDIDATE, attribution="Vlado pinned T_h1s600 reproduction",
            source_commit=SOURCE_COMMIT, source_git_blob=SOURCE_GIT_BLOB,
            source_sha256=SOURCE_SHA256, source_url=SOURCE_URL,
            recipe=fixed_recipe(), native_params=native_parameters(runtime),
            origin=str(context.origin), training_sources=len(design.sources),
            training_rows=len(design.features), source_feature_count=len(design.feature_names),
            model_feature_count=len(design.model_columns),
            first_source_timestamp=str(context.times[design.sources[0]]),
            last_source_timestamp=str(context.times[design.sources[-1]]),
            latest_label_timestamp=str(context.times[design.sources[-1] + HORIZON]),
            latest_feature_timestamp=str(context.origin),
            h1="Estimated16-step half midpoint added to learned residual",
            label="Raw hourly targets; only H1 residual target clipped nonnegative",
            source_dtype="float64", feature_storage="Temporary float64 memmap; one source dataframe at a time",
            feature_memmap_bytes=int(design.features.nbytes),
            model_matrix_c_contiguous=bool(design.model_matrix.flags.c_contiguous),
            native_contiguous_copy_possible=True,
            cumulative_process_peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * (1 if platform.system() == "Darwin" else 1024)),
            local_fits=1, native_threads=runtime.native_threads, n_estimators=600,
            resource_override=f"Original10 native threads changed to{runtime.native_threads}",
            comparable_to_teammate_validation=False, platform_submission=False)
    if not np.isfinite(result).all() or (result < 0).any():
        raise ValueError("Nonnegative finite output required")
    return result, audit
