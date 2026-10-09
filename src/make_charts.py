#!/usr/bin/env python3
"""使用本地日度合并数据绘制宏观因子与上证指数点位对照图。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager


REPO_ROOT = Path(__file__).resolve().parents[1]
PHASE_COLORS = {
    "phase1_2018-02_2019-06": "#4F81BD",
    "phase2_2020-07_2022-03": "#70AD47",
    "phase3_2023-10_2024-10": "#ED7D31",
}
LEGACY_PHASES = {
    "bear_2018-02_2019-06": "phase1_2018-02_2019-06",
    "period_2020-07_2022-03": "phase2_2020-07_2022-03",
    "bear_2023-10_2024-10": "phase3_2023-10_2024-10",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="绘制宏观因子与上证指数收盘点位的变化对照图。", add_help=False)
    parser.add_argument("-h", "--help", action="help", help="显示参数说明并退出。")
    parser._optionals.title = "可选参数"
    parser.add_argument(
        "--data-file",
        type=Path,
        default=Path("results/step/analysis_daily_data.csv"),
        help="使用--save-daily-data生成的日度合并CSV文件。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures"),
        help="图表输出目录，默认figures。",
    )
    parser.add_argument(
        "--study-only",
        action="store_true",
        help="只绘制三个研究阶段，阶段之间留空。",
    )
    parser.add_argument(
        "--font-path",
        type=Path,
        default=None,
        help="可选的中文字体文件路径，支持.ttf或.ttc。",
    )
    parser.add_argument("--dpi", type=int, default=180, help="图表输出分辨率，默认180。")
    parser.add_argument(
        "--scissor-mode",
        choices=("step", "linear", "lag1"),
        help="默认从run_config.json读取剪刀差模式；历史日度文件需显式指定。",
    )
    return parser.parse_args()


def repo_path(path: Path) -> Path:
    return path.expanduser().resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def configure_font(font_path: Path | None) -> None:
    if font_path is None:
        return
    resolved = repo_path(font_path)
    if not resolved.is_file():
        raise FileNotFoundError(f"找不到字体文件：{resolved}")
    font_manager.fontManager.addfont(str(resolved))
    family = font_manager.FontProperties(fname=str(resolved)).get_name()
    plt.rcParams["font.family"] = family
    plt.rcParams["axes.unicode_minus"] = False


def plot_series_by_scope(ax, data: pd.DataFrame, column: str, color: str, study_only: bool) -> None:
    if not study_only:
        ax.plot(data["date"], data[column], color=color, linewidth=1.6)
        return
    for phase in PHASE_COLORS:
        part = data.loc[data["phase"].eq(phase)]
        if not part.empty:
            ax.plot(part["date"], part[column], color=color, linewidth=1.6)


def shade_phases(ax, data: pd.DataFrame) -> None:
    for phase, color in PHASE_COLORS.items():
        part = data.loc[data["phase"].eq(phase)]
        if part.empty:
            continue
        ax.axvspan(part["date"].min(), part["date"].max(), color=color, alpha=0.10, lw=0)


def make_chart(
    data: pd.DataFrame,
    column: str,
    factor_title: str,
    factor_ylabel: str,
    filename: str,
    output_dir: Path,
    study_only: bool,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(18, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1.25], "hspace": 0.20},
    )
    plot_series_by_scope(axes[0], data, column, "#B66A00", study_only)
    plot_series_by_scope(axes[1], data, "close", "#3F2AA8", study_only)
    if column in ("scissor", "spread"):
        axes[0].axhline(0, color="#888888", linewidth=0.9, linestyle="--")
    axes[0].set_title(factor_title, loc="left", fontsize=16, pad=12)
    axes[0].set_ylabel(factor_ylabel, fontsize=12)
    axes[1].set_title("上证指数 · 日度收盘点位", loc="left", fontsize=16, pad=12)
    axes[1].set_ylabel("收盘点位", fontsize=12)

    for ax in axes:
        shade_phases(ax, data)
        ax.grid(True, color="#DCDCDC", linewidth=0.75)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#BDBDBD")
        ax.spines["bottom"].set_color("#BDBDBD")

    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[1].set_xlim(data["date"].min(), data["date"].max())
    fig.suptitle(f"{factor_title.split('·')[0].strip()}与上证指数点位变化", fontsize=22, y=0.98)
    scope = "三个研究阶段" if study_only else "完整有效样本"
    fig.text(0.075, 0.935, f"{scope} | 上下共用时间轴", fontsize=11, color="#777777")
    fig.subplots_adjust(left=0.075, right=0.985, top=0.86, bottom=0.10)
    fig.savefig(output_dir / filename, dpi=dpi, facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    configure_font(args.font_path)
    data_file = repo_path(args.data_file)
    output_dir = repo_path(args.output_dir)
    if not data_file.is_file():
        raise FileNotFoundError(
            f"找不到日度数据：{data_file}。请先运行run_analysis.py并传入--save-daily-data。"
        )
    data = pd.read_csv(data_file, parse_dates=["date"]).sort_values("date")
    required = {"date", "close", "scissor", "spread", "phase"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"日度数据缺少字段：{', '.join(missing)}")
    data["phase"] = data["phase"].replace(LEGACY_PHASES)
    config_file = data_file.parent / "run_config.json"
    mode = args.scissor_mode
    if mode is None and config_file.is_file():
        with config_file.open(encoding="utf-8") as handle:
            mode = json.load(handle)["scissor_mode"]
    if mode is None:
        raise ValueError("数据文件没有对应的run_config.json，请传入--scissor-mode")
    titles = {
        "step": "当月M1-M2货币剪刀差 · 阶梯型月度值",
        "linear": "M1-M2货币剪刀差 · 月末节点线性插值",
        "lag1": "上月M1-M2货币剪刀差 · 阶梯型月度值",
    }
    if args.study_only:
        data = data.loc[data["phase"].isin(PHASE_COLORS)].copy()
    if data.empty:
        raise ValueError("筛选后没有可绘制的数据")
    output_dir.mkdir(parents=True, exist_ok=True)

    make_chart(
        data,
        "scissor",
        titles[mode],
        "剪刀差（百分点）",
        "scissor_vs_sse_close.png",
        output_dir,
        args.study_only,
        args.dpi,
    )
    make_chart(
        data,
        "spread",
        "10Y-1Y国债收益率利差 · 日度值",
        "利差（百分点）",
        "yield_spread_vs_sse_close.png",
        output_dir,
        args.study_only,
        args.dpi,
    )
    print(f"图表已保存至：{output_dir}")


if __name__ == "__main__":
    main()
