"""Plot saved local baseline comparisons."""

import argparse
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
    names = {"ridge_status": "Ridge: статусы + маршрут", "mean_last_3h": "Среднее последних 3 часов", "seasonal_day": "То же время вчера", "last_value": "Последнее значение", "seasonal_week": "То же время неделю назад"}
    colors = ["#176b87" if name == "ridge_status" else "#aebcc8" for name in summary["model"]]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    fig.suptitle("RWB: сравнение прогнозов на четырёхчасовых окнах", fontsize=16, fontweight="bold")
    axes[0].barh(summary["model"].map(names), summary["mean_score"], color=colors, xerr=summary["std_score"], capsize=3)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 0.65)
    axes[0].set_xlabel("Средний score; меньше — лучше")
    axes[0].set_title("Разброс: стандартное отклонение по 7 окнам", fontsize=10)
    for index, row in enumerate(summary.itertuples()):
        axes[0].text(row.mean_score + row.std_score + 0.014, index, f"{row.mean_score:.4f}", va="center", fontsize=10)
    for name, color in [("ridge_status", "#176b87"), ("mean_last_3h", "#d35b36"), ("last_value", "#7b8994")]:
        axes[1].plot(np.arange(1, 9) * 30, horizons.loc[name], marker="o", color=color, label=names[name])
    axes[1].set_xlabel("Горизонт, минут")
    axes[1].set_ylabel("Средний score отдельно для горизонта")
    axes[1].set_xticks(np.arange(1, 9) * 30)
    axes[1].legend(frameon=False, fontsize=9, loc="upper left")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x" if ax is axes[0] else "y", alpha=0.15)
    fig.supxlabel("Локальная проверка: 25–31.10.2025, начало в 10:30. Это не оценка платформы.", fontsize=10)
    path = args.run_dir / "comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(path)


if __name__ == "__main__":
    main()
