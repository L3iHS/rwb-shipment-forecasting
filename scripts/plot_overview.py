"""Plot descriptive data summaries; no fitted model or test predictions."""

import os
import tempfile
from pathlib import Path

# Matplotlib's default user cache can be outside the writable workspace.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "rwb-shipment-matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "rwb-shipment-cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    reports = ROOT / ".local/reports/data_audit"
    daily = pd.read_csv(reports / "daily_target_summary.csv", parse_dates=["date"])
    totals = pd.read_csv(reports / "target_by_timestamp.csv", parse_dates=["timestamp"])
    test = pd.read_parquet(ROOT / "data/raw/test_solo_track.parquet")
    last_train = totals["timestamp"].max()
    final_test = test["timestamp"].max()
    recent = totals[totals["timestamp"] >= last_train - pd.Timedelta(days=2)]

    fig, axes = plt.subplots(2, 1, figsize=(13, 7.6), constrained_layout=True)
    fig.set_facecolor("#f7f9fc")
    fig.suptitle("RWB: история отгрузок и горизонт прогноза", fontsize=18, fontweight="bold")
    for ax in axes:
        ax.set_facecolor("white")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.18)
        ax.tick_params(labelsize=10)

    # The final calendar day is incomplete; show it separately.
    full_days = daily[daily["target_points"] == 48_000]
    partial_days = daily[daily["target_points"] != 48_000]
    axes[0].plot(full_days["date"], full_days["target_mean"] / 1000, color="#176b87", linewidth=1.8)
    axes[0].scatter(partial_days["date"], partial_days["target_mean"] / 1000, color="#d35b36", label="Неполный день", zorder=3)
    axes[0].set_title("Среднее значение часового target по всем маршрутам и точкам дня", loc="left", fontsize=11)
    axes[0].set_ylabel("target_1h, тыс.")
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    axes[0].legend(frameon=False, fontsize=9)

    axes[1].plot(recent["timestamp"], recent["total_target_1h"] / 1e6, color="#176b87", linewidth=1.8, label="Известные значения train")
    axes[1].axvspan(last_train, final_test, alpha=0.13, color="#d35b36", label="Прогноз на 4 часа: факты неизвестны")
    axes[1].axvline(last_train, color="#d35b36", linewidth=1, linestyle="--")
    axes[1].set_xlim(recent["timestamp"].min(), final_test)
    axes[1].set_title("Сумма часового target по 1000 маршрутам в каждый момент, последние 48 часов train", loc="left", fontsize=11)
    axes[1].set_ylabel("Сумма target_1h, млн")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%d.%m %H:%M"))
    axes[1].legend(frameon=False, fontsize=9, loc="upper left")
    fig.supxlabel("Даты наблюдений: 2025 год. Часовой пояс в файлах не указан.", fontsize=10)

    path = reports / "figures/data_overview.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(path)


if __name__ == "__main__":
    main()
