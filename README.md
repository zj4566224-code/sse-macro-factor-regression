# M1-M2货币剪刀差、国债期限利差与上证指数点位研究

本项目研究M1-M2货币剪刀差、10年期减1年期国债收益率利差与上证指数日度收盘点位的样本内相关关系，提供数据处理、分阶段回归和稳健性检验代码。原始CSMAR数据及高校授权的指数行情数据不随项目公开。

## 模型设定

`收盘点位 = 常数项 + β₁ × M1-M2剪刀差 + β₂ × 10Y-1Y国债利差 + 误差项`

分别对三个阶段及其合并样本估计普通最小二乘法（OLS）和加权最小二乘法（WLS），采用20阶HAC稳健标准误。

| 模式 | 剪刀差处理方式 |
|---|---|
| `step` | 当月剪刀差复制到当月各交易日，形成阶梯型序列 |
| `linear` | 以月末观测为节点，按自然日时间间隔进行线性插值 |
| `lag1` | 当月各交易日使用上一个自然月的剪刀差 |

## 当前权重版本

三种模式共用以下WLS权重规则：

| 设置 | 数值 |
|---|---:|
| 阶段一：2018年2月—2019年6月 | 1.65 |
| 阶段二：2020年7月—2022年3月 | 1.55 |
| 阶段三：2023年10月—2024年10月 | 1.65 |
| 10Y-1Y利差位于 `[0.50, 0.75]` 个百分点，含端点 | 0.5 |
| 利差位于其他区间 | 1.7 |

最终权重为“阶段权重 × 利差区间权重”，在三个阶段合并样本内标准化为均值1。每次运行将参数写入 `run_config.json`，便于核对版本。

## 本地数据准备

将获得使用授权的数据放在 `data/raw/` 中，也可以通过命令行指定其他位置：

```text
data/raw/
  CME_Mfinamkt1.csv
  BND_TreasYield10.csv
  BND_TreasYield.csv
  IDX_Idxtrd_part1.csv
  IDX_Idxtrd_part2.csv
  IDX_Idxtrd_part3.csv
```

| 数据文件 | 必需字段与处理口径 |
|---|---|
| M1/M2月度数据 | `Staper`识别月份；`Ezm0110`减`Ezm0109`，即M1同比减M2同比 |
| 10年期和1年期国债数据 | 筛选`Cvtype=1`；按`Trddt`匹配`Yield`，10年期减1年期 |
| 指数行情数据 | 筛选`Indexcd=000001`；日期使用`Idxtrd01`，收盘点位使用`Idxtrd05` |

多个指数文件存在重复日期时，优先保留命令行中先列出的文件记录。

## 安装与运行

使用Python 3.10或更新版本，在项目根目录安装依赖：

```bash
python -m pip install -r requirements.txt
```

```bash
python src/run_analysis.py --scissor-mode step
python src/run_analysis.py --scissor-mode linear
python src/run_analysis.py --scissor-mode lag1
```

上面三条命令分别运行当月阶梯型、线性插值和滞后一个月模型。默认样本窗口为2016年9月1日至2026年8月31日，回归只使用三个研究阶段。可配置输入文件、输出目录、样本日期、HAC阶数和权重：

```bash
python src/run_analysis.py \
  --scissor-mode lag1 \
  --money-file data/raw/CME_Mfinamkt1.csv \
  --yield-10y-file data/raw/BND_TreasYield10.csv \
  --yield-1y-file data/raw/BND_TreasYield.csv \
  --index-files data/raw/IDX_Idxtrd_part1.csv data/raw/IDX_Idxtrd_part2.csv data/raw/IDX_Idxtrd_part3.csv \
  --output-dir results/lag1 \
  --save-daily-data
```

程序的输入、输出相对路径均以项目根目录为基准。命令中的脚本路径仍相对于终端当前目录；从其他目录运行时，需要正确指定脚本路径。

运行 `python src/run_analysis.py --help` 查看完整参数说明。

## 图表生成

先在本地保存日度合并数据，再生成剪刀差与点位、期限利差与点位两张对照图：

```bash
python src/run_analysis.py --scissor-mode step --save-daily-data
python src/make_charts.py --data-file results/step/analysis_daily_data.csv --study-only
```

`--study-only`只展示三个研究阶段，阶段之间留空。日度文件包含授权数据序列，已通过 `.gitignore` 排除。图标题自动读取数据文件同目录下 `run_config.json` 中的剪刀差模式；使用历史日度文件时，需要显式传入 `--scissor-mode`。中文显示缺字时，可通过 `--font-path` 指定支持中文的字体文件。

权重端点与阶段权重测试：

```bash
python -m unittest discover -s tests -v
```

## 结果复现与核对

重构后的三种模式已使用相同本地原始数据重跑。系数、HAC标准误与p值、样本量、R²、调整后R²及RMSE等全部数值列，与对应0.5／1.7权重历史脚本在 `1e-10` 容差内一致。三种模式均使用1,025个研究阶段交易日样本。

| 剪刀差处理方式 | 合并OLS R² | 合并WLS R² | 阶段一WLS R² | 阶段二WLS R² | 阶段三WLS R² |
|---|---:|---:|---:|---:|---:|
| 当月阶梯型 | 12.85% | 13.11% | 57.16% | 2.77% | 5.16% |
| 月末线性插值 | 13.57% | 12.20% | 51.99% | 1.93% | 4.44% |
| 滞后一个月阶梯型 | 12.15% | 9.81% | 45.26% | 1.34% | 1.67% |

旧版0.65／1.0权重脚本未纳入本项目。

## 研究结论与局限

因子的样本内解释力存在明显阶段差异，后两个阶段的拟合度较低。结果应作为阶段性相关特征解读，不能直接推断因果关系或稳定的样本外预测能力。

月度观测按统计月份匹配，未按真实发布日期构造可用信息集，因此滞后一个月也不等于已经验证的实时预测信号。线性插值使用后续月末观测，属于事后检验；原实现对末端节点之后的日期沿用最后一个观测值。

WLS权重由研究者指定，用于敏感性分析，并非估计得到的误差方差倒数。WLS R²采用加权口径，表中RMSE采用未加权口径。HAC处理标准误的序列相关问题，不能替代对点位序列平稳性、伪回归和样本外表现的检验。
