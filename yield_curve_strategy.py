#!/usr/bin/env python3
"""国债收益率曲线形态分析 + 2s10s 均值回归策略回测（纯标准库）"""

from __future__ import annotations

import csv
import math
import random
import statistics
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DATA_URL = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
    "daily-treasury-rates.csv?field_tdr_date_value=all&page&_format=csv"
)


@dataclass
class Row:
    date: datetime
    y2: float
    y5: float
    y10: float


@dataclass
class TradePoint:
    date: datetime
    spread: float
    signal: int
    ret_bp: float
    equity: float


def parse_float(cell: str) -> Optional[float]:
    cell = (cell or "").strip()
    if not cell:
        return None
    try:
        return float(cell)
    except ValueError:
        return None


def generate_synthetic_rows(days: int = 3000, seed: int = 42) -> List[Row]:
    """生成可复现实验数据：含周期与均值回归特征。"""
    random.seed(seed)
    rows: List[Row] = []
    base = datetime(2012, 1, 2)
    y2 = 2.0
    spread = 1.2
    belly_bias = 0.1
    for i in range(days):
        cyc = 0.6 * math.sin(i / 180)
        shock = random.gauss(0, 0.05)
        # 2Y 更受政策利率扰动
        y2 = 0.92 * y2 + 0.08 * (2.2 + 0.4 * math.sin(i / 70)) + random.gauss(0, 0.03)
        # 2s10s 含均值回归
        spread = 0.96 * spread + 0.04 * (1.0 + cyc) + shock
        y10 = y2 + spread
        y5 = (y2 + y10) / 2 + belly_bias + random.gauss(0, 0.02)
        rows.append(Row(base.fromordinal(base.toordinal() + i), y2, y5, y10))
    return rows


def download_rows(url: str = DATA_URL) -> Tuple[List[Row], str]:
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            content = resp.read().decode("utf-8")
        source = "US Treasury Daily Treasury Rates"
    except Exception:
        rows = generate_synthetic_rows()
        return rows, "synthetic (network fallback)"

    rows: List[Row] = []
    reader = csv.DictReader(content.splitlines())
    for item in reader:
        date_text = item.get("Date") or item.get("DATE")
        if not date_text:
            continue
        try:
            dt = datetime.strptime(date_text, "%m/%d/%Y")
        except ValueError:
            continue
        y2 = parse_float(item.get("2 Yr", ""))
        y5 = parse_float(item.get("5 Yr", ""))
        y10 = parse_float(item.get("10 Yr", ""))
        if y2 is None or y5 is None or y10 is None:
            continue
        rows.append(Row(dt, y2, y5, y10))

    rows.sort(key=lambda r: r.date)
    return rows


def rolling_stats(values: List[float], lookback: int) -> List[Tuple[Optional[float], Optional[float]]]:
    out: List[Tuple[Optional[float], Optional[float]]] = []
    for i in range(len(values)):
        if i < lookback:
            out.append((None, None))
            continue
        window = values[i - lookback : i]
        mu = statistics.mean(window)
        sd = statistics.pstdev(window)
        out.append((mu, sd if sd > 1e-9 else None))
    return out


def quantile(values: List[float], q: float) -> float:
    s = sorted(values)
    if not s:
        raise ValueError("empty values")
    if q <= 0:
        return s[0]
    if q >= 1:
        return s[-1]
    pos = q * (len(s) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return s[lo]
    w = pos - lo
    return s[lo] * (1 - w) + s[hi] * w


def max_drawdown(equity: List[float]) -> float:
    peak = -1e18
    mdd = 0.0
    for v in equity:
        peak = max(peak, v)
        dd = v - peak
        mdd = min(mdd, dd)
    return mdd


def run_backtest(rows: List[Row], lookback: int = 60, entry_z: float = 1.0, tc_bp: float = 0.2) -> Dict[str, object]:
    spread = [r.y10 - r.y2 for r in rows]
    curve = [2 * r.y5 - r.y2 - r.y10 for r in rows]
    stats = rolling_stats(spread, lookback)

    signal = [0] * len(rows)
    pos = 0
    for i in range(len(rows)):
        mu, sd = stats[i]
        if mu is None or sd is None:
            signal[i] = 0
            continue
        z = (spread[i] - mu) / sd
        if pos == 0:
            if z > entry_z:
                pos = 1  # flattening: profit from spread down
            elif z < -entry_z:
                pos = -1  # steepening: profit from spread up
        else:
            # 回到均值附近再平仓，降低换手
            if abs(z) < 0.25:
                pos = 0
        signal[i] = pos

    returns_bp: List[float] = [0.0]
    equity = [0.0]
    points: List[TradePoint] = [TradePoint(rows[0].date, spread[0], signal[0], 0.0, 0.0)]

    for i in range(1, len(rows)):
        d_spread = spread[i] - spread[i - 1]
        prev_sig = signal[i - 1]
        turnover = abs(signal[i] - signal[i - 1])
        # 头寸方向定义：+1=做平(押注spread下降)，-1=做陡(押注spread上升)
        ret = prev_sig * (-d_spread) - turnover * tc_bp
        returns_bp.append(ret)
        equity.append(equity[-1] + ret)
        points.append(TradePoint(rows[i].date, spread[i], signal[i], ret, equity[-1]))

    valid_rets = returns_bp[1:]
    ann_factor = 252
    avg = statistics.mean(valid_rets) if valid_rets else 0.0
    vol = statistics.pstdev(valid_rets) if len(valid_rets) > 1 else 0.0
    sharpe = (avg / vol) * math.sqrt(ann_factor) if vol > 1e-12 else 0.0
    ann_ret = avg * ann_factor
    ann_vol = vol * math.sqrt(ann_factor)
    win_rate = sum(1 for r in valid_rets if r > 0) / len(valid_rets) if valid_rets else 0.0
    mdd = max_drawdown(equity)

    # 形态分层：按2s10s分位数
    q20 = quantile(spread, 0.2)
    q80 = quantile(spread, 0.8)
    steep_idx = [i for i, s in enumerate(spread[:-20]) if s >= q80]
    flat_idx = [i for i, s in enumerate(spread[:-20]) if s <= q20]
    normal_idx = [i for i, s in enumerate(spread[:-20]) if q20 < s < q80]

    def fwd_change(indexes: List[int], horizon: int = 20) -> float:
        if not indexes:
            return 0.0
        vals = [spread[i + horizon] - spread[i] for i in indexes if i + horizon < len(spread)]
        return statistics.mean(vals) if vals else 0.0

    regime = {
        "steep_20d_bp": fwd_change(steep_idx),
        "flat_20d_bp": fwd_change(flat_idx),
        "normal_20d_bp": fwd_change(normal_idx),
        "q20": q20,
        "q80": q80,
        "avg_curvature": statistics.mean(curve),
    }

    return {
        "points": points,
        "metrics": {
            "total_bp": equity[-1],
            "ann_ret_bp": ann_ret,
            "ann_vol_bp": ann_vol,
            "sharpe": sharpe,
            "max_drawdown_bp": mdd,
            "win_rate": win_rate,
            "trades": sum(1 for i in range(1, len(signal)) if signal[i] != signal[i - 1]),
            "samples": len(valid_rets),
        },
        "regime": regime,
        "start": rows[0].date,
        "end": rows[-1].date,
    }


def build_report(result: Dict[str, object], source: str) -> str:
    m = result["metrics"]
    r = result["regime"]
    start = result["start"].strftime("%Y-%m-%d")
    end = result["end"].strftime("%Y-%m-%d")

    return f"""# 国债收益率曲线形态分析与交易策略回测

## 1) 数据与方法
- 数据源：**{source}**，样本区间 **{start} ~ {end}**（2Y/5Y/10Y）。
- 形态指标：
  - 斜率（Slope）= 10Y - 2Y（bp）
  - 曲率（Curvature）= 2×5Y - 2Y - 10Y（bp）
- 策略逻辑：2s10s 价差均值回归。
  - 当 z-score > 1：做平（押注 2s10s 下行）
  - 当 z-score < -1：做陡（押注 2s10s 上行）
  - 其他：空仓
- 参数：60日滚动窗口，单边换仓成本 0.2bp（按头寸变化计入）。

## 2) 曲线形态变化结论（20个交易日前瞻）
- 2s10s 处于**最陡20%**时，未来20日平均变化：**{r['steep_20d_bp']:.2f} bp**。
- 2s10s 处于**最平20%**时，未来20日平均变化：**{r['flat_20d_bp']:.2f} bp**。
- 中间区间（20%~80%）未来20日平均变化：**{r['normal_20d_bp']:.2f} bp**。
- 全样本平均曲率：**{r['avg_curvature']:.2f} bp**（反映 belly 相对两端的抬升/下沉）。

> 解读：若“陡峭区间”对应未来负变化、而“平坦区间”对应未来正变化，说明斜率存在一定均值回归特征，可支持区间反转型交易框架。

## 3) 回测表现（bp 口径）
- 累计收益：**{m['total_bp']:.2f} bp**
- 年化收益：**{m['ann_ret_bp']:.2f} bp/年**
- 年化波动：**{m['ann_vol_bp']:.2f} bp/年**
- 夏普比率：**{m['sharpe']:.2f}**
- 最大回撤：**{m['max_drawdown_bp']:.2f} bp**
- 胜率：**{m['win_rate']*100:.2f}%**
- 换仓次数：**{m['trades']}**
- 样本日数：**{m['samples']}**

## 4) 可执行交易建议
1. **基准策略（推荐）**：保持 2s10s z-score 反转规则 + 成本约束。
2. **风险控制**：
   - 斜率绝对值突破历史99%分位时降杠杆；
   - 宏观事件窗口（FOMC/非农/CPI）将仓位减半。
3. **增强方向**：
   - 用曲率过滤（例如仅在曲率同向极值时开仓）以降低噪声；
   - 引入期限溢价代理变量（如 ACM term premium）做状态切换。

---
*注：本回测为研究框架演示，未包含融资、保证金、滑点细化与真实可交易合约映射。若数据源为 synthetic，结论仅用于方法演示。*
"""


def main() -> None:
    rows, source = download_rows()
    if len(rows) < 300:
        raise RuntimeError("有效样本太少，无法回测。")
    result = run_backtest(rows)
    report = build_report(result, source)
    out = Path("report.md")
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n报告已写入: {out.resolve()}")


if __name__ == "__main__":
    main()
