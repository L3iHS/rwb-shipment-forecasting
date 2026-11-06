"""Plot saved local baseline comparisons."""

import argparse
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "rwb-shipment-matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "rwb-shipment-cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / ".local/runs/temporal_baselines")
    args = parser.parse_args()
    summary = pd.read_csv(args.run_dir / "summary.csv").sort_values("mean_score")
    horizons = pd.read_csv(args.run_dir / "by_horizon.csv").groupby(["model", "horizon_step"])["score"].mean()
    names = {
        "ridge_status": "Ridge: статусы + маршрут", "ridge_lags": "Ridge + лаги",
        "ridge_rolling": "Ridge + лаги + средние", "ridge_calendar": "Ridge + лаги + средние + календарь",
        "mean_last_3h": "Среднее последних 3 часов", "seasonal_day": "То же время вчера",
        "last_value": "Последнее значение", "seasonal_week": "То же время неделю назад",
    }
    enriched = "ridge_calendar" in summary["model"].to_list()
    highlight = "ridge_calendar" if enriched else "ridge_status"
    colors = ["#176b87" if name == highlight else "#aebcc8" for name in summary["model"]]
    metadata = json.loads((args.run_dir / "metadata.json").read_text())
    origins = pd.to_datetime(metadata["origins"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 6 if enriched else 4.8), constrained_layout=True)
    fig.suptitle("RWB: сравнение прогнозов на четырёхчасовых окнах", fontsize=16, fontweight="bold")
    axes[0].barh(summary["model"].map(names), summary["mean_score"], color=colors, xerr=summary["std_score"], capsize=3)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 0.65)
    axes[0].set_xlabel("Средний score; меньше — лучше")
    axes[0].set_title(f"Разброс: стандартное отклонение по {len(origins)} окнам", fontsize=10)
    axes[0].tick_params(axis="y", labelsize=9)
    for index, row in enumerate(summary.itertuples()):
        axes[0].text(row.mean_score + row.std_score + 0.014, index, f"{row.mean_score:.4f}", va="center", fontsize=10)
    curves = (
        [("ridge_status", "#7b8994"), ("ridge_rolling", "#d35b36"), ("ridge_calendar", "#176b87")]
        if enriched else [("ridge_status", "#176b87"), ("mean_last_3h", "#d35b36"), ("last_value", "#7b8994")]
    )
    for name, color in curves:
        axes[1].plot(np.arange(1, 9) * 30, horizons.loc[name], marker="o", color=color, label=names[name])
    axes[1].set_xlabel("Горизонт, минут")
    axes[1].set_ylabel("Средний score отдельно для горизонта")
    axes[1].set_xticks(np.arange(1, 9) * 30)
    axes[1].legend(frameon=False, fontsize=9, loc="upper left")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x" if ax is axes[0] else "y", alpha=0.15)
    fig.supxlabel(
        f"Локальная проверка: {origins.min():%d.%m.%Y}–{origins.max():%d.%m.%Y}, "
        f"начало в {origins[0]:%H:%M}. Это не оценка платформы.", fontsize=10,
    )
    path = args.run_dir / "comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(path)


if __name__ == "__main__":
    main()
