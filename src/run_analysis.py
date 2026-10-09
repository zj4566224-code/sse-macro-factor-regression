#!/usr/bin/env python3
"""运行上证指数日度收盘点位回归。

支持三种M1-M2剪刀差处理方式：

* ``step``：当月观测复制到当月各交易日；
* ``linear``：以自然月末观测为节点进行线性插值；
* ``lag1``：当月各交易日使用上月观测。

本研究用于描述样本内相关关系，不直接估计因果效应或样本外预测能力。
项目不附带原始授权数据。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import statsmodels.api as sm


REPO_ROOT = Path(__file__).resolve().parents[1]

PHASES = (
    ("phase1_2018-02_2019-06", "2018-02", "2019-06"),
    ("phase2_2020-07_2022-03", "2020-07", "2022-03"),
    ("phase3_2023-10_2024-10", "2023-10", "2024-10"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用M1-M2剪刀差及10Y-1Y国债利差解释上证指数日度收盘点位。",
        add_help=False,
    )
    parser.add_argument("-h", "--help", action="help", help="显示参数说明并退出。")
    parser._optionals.title = "可选参数"
    parser.add_argument(
        "--scissor-mode",
        choices=("step", "linear", "lag1"),
        default="step",
        help="剪刀差处理方式：step当月阶梯型、linear线性插值、lag1滞后一个月；默认step。",
    )
    parser.add_argument(
        "--money-file",
        type=Path,
        default=Path("data/raw/CME_Mfinamkt1.csv"),
        help="M1/M2月度CSV文件；相对路径以项目根目录为基准。",
    )
    parser.add_argument(
        "--yield-10y-file",
        type=Path,
        default=Path("data/raw/BND_TreasYield10.csv"),
        help="10年期国债收益率CSV文件。",
    )
    parser.add_argument(
        "--yield-1y-file",
        type=Path,
        default=Path("data/raw/BND_TreasYield.csv"),
        help="1年期国债收益率CSV文件。",
    )
    parser.add_argument(
        "--index-files",
        type=Path,
        nargs="+",
        default=[
            Path("data/raw/IDX_Idxtrd_part1.csv"),
            Path("data/raw/IDX_Idxtrd_part2.csv"),
            Path("data/raw/IDX_Idxtrd_part3.csv"),
        ],
        help="一个或多个指数日行情CSV文件，重复日期优先保留先列出的文件记录。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="结果输出目录；默认results/<剪刀差模式>。",
    )
    parser.add_argument("--start", default="2016-09-01", help="样本起始日期，默认2016-09-01。")
    parser.add_argument("--end", default="2026-08-31", help="样本结束日期，默认2026-08-31。")
    parser.add_argument("--hac-lags", type=int, default=20, help="HAC最大滞后阶数，默认20。")

    # 当前确认的权重：阶段权重1.65／1.55／1.65；
    # 利差在[0.50, 0.75]内的乘数为0.5，其他区间为1.7。
    parser.add_argument("--phase1-weight", type=float, default=1.65, help="阶段一权重，默认1.65。")
    parser.add_argument("--phase2-weight", type=float, default=1.55, help="阶段二权重，默认1.55。")
    parser.add_argument("--phase3-weight", type=float, default=1.65, help="阶段三权重，默认1.65。")
    parser.add_argument("--low-spread-min", type=float, default=0.50, help="低权重利差区间下界，默认0.50个百分点，含端点。")
    parser.add_argument("--low-spread-max", type=float, default=0.75, help="低权重利差区间上界，默认0.75个百分点，含端点。")
    parser.add_argument("--low-spread-weight", type=float, default=0.50, help="区间内利差权重乘数，默认0.5。")
    parser.add_argument("--other-spread-weight", type=float, default=1.70, help="区间外利差权重乘数，默认1.7。")
    parser.add_argument(
        "--save-daily-data",
        action="store_true",
        help="额外保存日度合并数据；若来源为授权数据，仅供本地使用。",
    )
    return parser.parse_args()


def repo_path(path: Path) -> Path:
    """解析命令行路径，相对路径以项目根目录为基准。"""
    return path.expanduser().resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def validate_args(args: argparse.Namespace) -> None:
    if pd.Timestamp(args.start) > pd.Timestamp(args.end):
        raise ValueError("--start起始日期不能晚于--end结束日期")
    if args.hac_lags < 0:
        raise ValueError("--hac-lags必须为非负整数")
    if args.low_spread_min > args.low_spread_max:
        raise ValueError("--low-spread-min不能大于--low-spread-max")
    weights = (
        args.phase1_weight,
        args.phase2_weight,
        args.phase3_weight,
        args.low_spread_weight,
        args.other_spread_weight,
    )
    if any(not np.isfinite(weight) or weight <= 0 for weight in weights):
        raise ValueError("所有WLS权重必须为有限的正数")


def require_columns(frame: pd.DataFrame, columns: Iterable[str], source: Path) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} 缺少必需字段：{', '.join(missing)}")


def read_monthly_scissor(path: Path) -> pd.Series:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    require_columns(frame, ("Staper", "Ezm0110", "Ezm0109"), path)
    frame["month"] = pd.PeriodIndex(frame["Staper"], freq="M")
    frame["scissor"] = (
        pd.to_numeric(frame["Ezm0110"], errors="raise")
        - pd.to_numeric(frame["Ezm0109"], errors="raise")
    )
    if frame["month"].duplicated().any():
        duplicates = frame.loc[frame["month"].duplicated(False), "month"].astype(str).unique()
        raise ValueError(f"M1/M2存在重复月份：{', '.join(duplicates[:5])}")
    return frame.set_index("month")["scissor"].sort_index()


def read_yield(path: Path) -> pd.Series:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    require_columns(frame, ("Cvtype", "Trddt", "Yield"), path)
    frame = frame.loc[frame["Cvtype"].astype(str).str.strip() == "1"].copy()
    frame["date"] = pd.to_datetime(frame["Trddt"])
    frame = frame.drop_duplicates("date", keep="last").set_index("date").sort_index()
    return pd.to_numeric(frame["Yield"], errors="raise")


def read_sse_close(paths: list[Path]) -> pd.Series:
    frames = []
    for path in paths:
        frame = pd.read_csv(path, encoding="utf-8-sig")
        require_columns(frame, ("Indexcd", "Idxtrd01", "Idxtrd05"), path)
        frames.append(frame)
    index = pd.concat(frames, ignore_index=True)
    index = index.loc[
        index["Indexcd"].astype(str).str.strip().str.zfill(6) == "000001"
    ].copy()
    index["date"] = pd.to_datetime(index["Idxtrd01"])
    # 保留原项目口径：多个指数文件出现同一日期时，优先保留先列出的文件记录。
    index = index.drop_duplicates("date", keep="first").set_index("date").sort_index()
    return pd.to_numeric(index["Idxtrd05"], errors="raise")


def interpolate_scissor(monthly: pd.Series, daily_index: pd.DatetimeIndex) -> pd.Series:
    """保留原脚本以月末为节点的时间插值行为。

    Pandas保留首个节点之前的缺失值，并对末端节点之后的日期沿用最后观测。
    """
    nodes = monthly.copy()
    nodes.index = nodes.index.to_timestamp(how="end").normalize()
    union = nodes.index.union(daily_index).sort_values()
    return nodes.reindex(union).interpolate(method="time").reindex(daily_index)


def map_scissor(
    monthly: pd.Series, daily_index: pd.DatetimeIndex, mode: str
) -> pd.Series:
    months = daily_index.to_period("M")
    if mode == "step":
        return pd.Series(months.map(monthly), index=daily_index, dtype=float)
    if mode == "lag1":
        # 将每个观测匹配到下一个自然月，使t月使用t-1月的剪刀差。
        lagged = monthly.copy()
        lagged.index = lagged.index + 1
        return pd.Series(months.map(lagged), index=daily_index, dtype=float)
    return interpolate_scissor(monthly, daily_index)


def assign_phases(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["phase"] = "other"
    for label, start, end in PHASES:
        in_phase = frame["month"].between(
            pd.Period(start, freq="M"), pd.Period(end, freq="M")
        )
        frame.loc[in_phase, "phase"] = label
    frame["in_study_period"] = frame["phase"].ne("other").astype(int)
    return frame


def add_weights(frame: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    frame = frame.copy()
    phase_weights = {
        PHASES[0][0]: args.phase1_weight,
        PHASES[1][0]: args.phase2_weight,
        PHASES[2][0]: args.phase3_weight,
    }
    frame["phase_weight"] = frame["phase"].map(phase_weights).fillna(1.0)
    frame["low_spread"] = frame["spread"].between(
        args.low_spread_min, args.low_spread_max, inclusive="both"
    ).astype(int)
    frame["spread_weight"] = np.where(
        frame["low_spread"].eq(1),
        args.low_spread_weight,
        args.other_spread_weight,
    )
    frame["weight"] = frame["phase_weight"] * frame["spread_weight"]
    return frame


def load_data(args: argparse.Namespace) -> pd.DataFrame:
    money_path = repo_path(args.money_file)
    yield_10y_path = repo_path(args.yield_10y_file)
    yield_1y_path = repo_path(args.yield_1y_file)
    index_paths = [repo_path(path) for path in args.index_files]
    for path in (money_path, yield_10y_path, yield_1y_path, *index_paths):
        if not path.is_file():
            raise FileNotFoundError(f"找不到输入文件：{path}")

    monthly = read_monthly_scissor(money_path)
    close = read_sse_close(index_paths)
    spread = read_yield(yield_10y_path) - read_yield(yield_1y_path)
    frame = pd.DataFrame({"close": close, "spread": spread}).loc[
        pd.Timestamp(args.start) : pd.Timestamp(args.end)
    ].dropna()
    frame["month"] = frame.index.to_period("M")
    frame["scissor"] = map_scissor(monthly, frame.index, args.scissor_mode)
    frame = assign_phases(frame.dropna(subset=["scissor"]))
    return add_weights(frame, args).dropna().copy()


def fit_model(
    data: pd.DataFrame,
    predictors: list[str],
    hac_lags: int,
    weights: pd.Series | None = None,
):
    x = sm.add_constant(data[predictors], has_constant="add")
    if weights is None:
        fitted = sm.OLS(data["close"], x).fit()
    else:
        fitted = sm.WLS(data["close"], x, weights=weights).fit()
    robust = fitted.get_robustcov_results(cov_type="HAC", maxlags=hac_lags)
    return fitted, robust


def partial_r2(data: pd.DataFrame, predictors: list[str], variable: str, hac_lags: int) -> float:
    full, _ = fit_model(data, predictors, hac_lags)
    reduced_predictors = [name for name in predictors if name != variable]
    reduced, _ = fit_model(data, reduced_predictors, hac_lags)
    return max(0.0, (reduced.ssr - full.ssr) / reduced.ssr)


def report(
    name: str,
    data: pd.DataFrame,
    predictors: list[str],
    hac_lags: int,
    weights: pd.Series | None = None,
) -> pd.DataFrame:
    fitted, robust = fit_model(data, predictors, hac_lags, weights)
    names = ["const", *predictors]
    result = pd.DataFrame(
        {
            "term": names,
            "coefficient": robust.params,
            "hac_std_error": robust.bse,
            "hac_p_value": robust.pvalues,
        }
    )
    y_std = data["close"].std(ddof=0)
    result["standardized_coefficient"] = [
        np.nan
        if term == "const"
        else fitted.params[term] * data[term].std(ddof=0) / y_std
        for term in names
    ]
    result["model"] = name
    result["n_obs"] = len(data)
    result["r_squared"] = fitted.rsquared
    result["adjusted_r_squared"] = fitted.rsquared_adj
    result["rmse"] = np.sqrt(np.mean(fitted.resid**2))
    result["partial_r_squared"] = [
        np.nan
        if term == "const" or weights is not None
        else partial_r2(data, predictors, term, hac_lags)
        for term in names
    ]
    return result


def run_models(data: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    predictors = ["scissor", "spread"]
    study = data.loc[data["in_study_period"].eq(1)].copy()
    if study.empty:
        raise ValueError("三个研究阶段内均没有可用观测")
    study["weight"] /= study["weight"].mean()
    prefix = f"{args.scissor_mode}_scissor"
    reports = [
        report(
            f"{prefix}_three_periods_combined: scissor + spread",
            study,
            predictors,
            args.hac_lags,
        ),
        report(
            f"{prefix}_three_periods_combined: weighted scissor + spread",
            study,
            predictors,
            args.hac_lags,
            study["weight"],
        ),
    ]
    for phase, _, _ in PHASES:
        segment = study.loc[study["phase"].eq(phase)]
        if segment.empty:
            raise ValueError(f"研究阶段{phase}没有可用观测")
        reports.extend(
            [
                report(
                    f"{prefix}_{phase}: scissor + spread",
                    segment,
                    predictors,
                    args.hac_lags,
                ),
                report(
                    f"{prefix}_{phase}: weighted scissor + spread",
                    segment,
                    predictors,
                    args.hac_lags,
                    segment["weight"],
                ),
            ]
        )
    return pd.concat(reports, ignore_index=True)


def serializable_config(args: argparse.Namespace, output_dir: Path) -> dict:
    return {
        "scissor_mode": args.scissor_mode,
        "sample": {"start": args.start, "end": args.end},
        "hac_lags": args.hac_lags,
        "phase_weights": {
            PHASES[0][0]: args.phase1_weight,
            PHASES[1][0]: args.phase2_weight,
            PHASES[2][0]: args.phase3_weight,
        },
        "spread_weight_rule": {
            "inclusive_interval": [args.low_spread_min, args.low_spread_max],
            "inside_interval": args.low_spread_weight,
            "outside_interval": args.other_spread_weight,
        },
        "output_dir": str(output_dir.relative_to(REPO_ROOT))
        if output_dir.is_relative_to(REPO_ROOT)
        else str(output_dir),
        "note": "WLS权重=阶段权重×利差区间权重；在合并研究样本内标准化为均值1。",
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    output_dir = repo_path(args.output_dir or Path("results") / args.scissor_mode)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_data(args)
    results = run_models(data, args)
    results.to_csv(output_dir / "model_results.csv", index=False, encoding="utf-8-sig")
    if args.save_daily_data:
        data.to_csv(
            output_dir / "analysis_daily_data.csv",
            index_label="date",
            encoding="utf-8-sig",
        )
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(serializable_config(args, output_dir), handle, ensure_ascii=False, indent=2)

    overview = results.groupby("model", sort=False).first()[
        ["n_obs", "r_squared", "adjusted_r_squared", "rmse"]
    ]
    print("模型结果概览（R²以小数表示）：")
    print(overview.rename(columns={
        "n_obs": "样本量", "r_squared": "R²",
        "adjusted_r_squared": "调整后R²", "rmse": "RMSE（点）",
    }).to_string(float_format=lambda value: f"{value:.4f}"))
    print(f"\n结果已保存至：{output_dir}")


if __name__ == "__main__":
    main()
