"""
L2 描述性分析（RQ1）：情绪分布、季度演化趋势、标志词频率变化。

输入:  data/processed/answers_labeled.csv
输出:  analysis/figures/{emotion_dist,emotion_trend,marker_words}.png
       analysis/tables/emotion_share_by_quarter.csv

用法:  python -m analysis.rq1_descriptive
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analysis.common import (
    LABELED_ANSWERS, FIGURES_DIR, TABLES_DIR, EMOTIONS, TOPICS,
    ensure_dirs, setup_cn_font, read_csv, write_csv,
)

# 情绪固定配色（积极=绿，焦虑=橙，解构=红，中性=灰）
EMO_COLORS = {"积极": "#2ca02c", "焦虑": "#ff7f0e", "解构": "#d62728", "中性": "#999999"}

# 关键历史节点（季度 → 标注），用于在趋势图上做描述性解读
EVENTS = {
    "2021Q3": "双减政策",
    "2023Q1": "孔乙己文学",
    "2023Q3": "青年失业\n数据停更",
    "2024Q1": "AI就业\n讨论",
}

# 标志词：上升组（解构/焦虑话语）vs 下降组（积极话语）
RISING_WORDS = ["躺平", "摆烂", "内卷", "整活", "绷不住"]
FALLING_WORDS = ["梦想", "奋斗", "相信", "热爱", "未来"]


def _quarter_order(df: pd.DataFrame) -> list[str]:
    return sorted(df["quarter"].unique())


def fig_emotion_dist(df: pd.DataFrame) -> None:
    """图1：四类情绪总体占比 + 分议题占比（堆叠条形）。"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # 总体
    overall = df["emotion"].value_counts(normalize=True).reindex(EMOTIONS).fillna(0)
    ax1.bar(overall.index, overall.values, color=[EMO_COLORS[e] for e in overall.index])
    ax1.set_title("四类情绪总体占比",fontsize=18)
    ax1.set_ylabel("占比",fontsize=16)
    for i, v in enumerate(overall.values):
        ax1.text(i, v + 0.005, f"{v:.1%}", ha="center",fontsize=14)

    # 分议题（堆叠）
    by_topic = (
        df.groupby("topic")["emotion"].value_counts(normalize=True)
        .unstack().reindex(index=TOPICS, columns=EMOTIONS).fillna(0)
    )
    bottom = np.zeros(len(by_topic))
    for emo in EMOTIONS:
        ax2.bar(by_topic.index, by_topic[emo], bottom=bottom,
                label=emo, color=EMO_COLORS[emo])
        bottom += by_topic[emo].values
    ax2.set_title("分议题情绪占比",fontsize=18)
    ax2.set_ylabel("占比",fontsize=16)
    ax2.legend(loc="upper right", ncol=4, fontsize=8)
    plt.setp(ax2.get_xticklabels(), rotation=15,fontsize=14)

    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "emotion_dist.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"[L2] 图: {path}")


def fig_emotion_trend(df: pd.DataFrame) -> pd.DataFrame:
    """图2：季度情绪占比时间序列（折线 + 历史事件标注）。"""
    quarters = _quarter_order(df)
    share = (
        df.groupby("quarter")["emotion"].value_counts(normalize=True)
        .unstack().reindex(index=quarters, columns=EMOTIONS).fillna(0)
    )
    counts = df.groupby("quarter").size().reindex(quarters)

    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(quarters))
    for emo in EMOTIONS:
        ax.plot(x, share[emo].values, marker="o", ms=3, label=emo, color=EMO_COLORS[emo])

    # 事件竖线
    for q, name in EVENTS.items():
        if q in quarters:
            y_top = share.max().max()
            xi = quarters.index(q)
            ax.axvline(xi, color="black", ls="--", lw=0.8, alpha=0.5)
            ax.text(xi, y_top+0.06, name, rotation=0, fontsize=14, ha="center",
                    va="top", color="black",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8))
            ax.set_ylim(0, y_top+0.15)

    ax.set_xticks(x)
    ax.set_xticklabels(quarters, rotation=60, fontsize=12)
    ax.set_ylabel("情绪类型占比",fontsize=16)
    ax.set_title("知乎人生规划话语：四类情绪的季度演化（2020Q1–2025Q4）",fontsize=18)
    ax.legend(loc="upper left", ncol=4)
    ax.grid(alpha=0.3)

    # 副轴：每季度样本量（提示早期季度样本较少）
    ax2 = ax.twinx()
    ax2.bar(x, counts.values, alpha=0.12, color="steelblue")
    ax2.set_ylabel("每季度回答数（浅蓝柱）", color="steelblue",fontsize=16)

    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "emotion_trend.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"[L2] 图: {path}")

    # 同时落表
    tbl = share.copy()
    tbl["n_answers"] = counts.values
    tbl_path = os.path.join(TABLES_DIR, "emotion_share_by_quarter.csv")
    write_csv(tbl.reset_index(), tbl_path)
    print(f"[L2] 表: {tbl_path}")
    return share


def fig_marker_words(df: pd.DataFrame) -> None:
    """图3：标志词年度出现率对比（上升组 vs 下降组）。"""
    df = df.copy()
    df["content"] = df["content"].astype(str)
    years = sorted(df["year"].unique())

    def yearly_rate(word: str) -> list[float]:
        rates = []
        for y in years:
            sub = df[df["year"] == y]
            rates.append(sub["content"].str.contains(word, regex=False).mean())
        return rates

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for w in RISING_WORDS:
        ax1.plot(years, yearly_rate(w), marker="o", ms=3, label=w)
    ax1.set_title("上升组（解构/焦虑话语）",fontsize=18)
    ax1.set_xlabel("年份",fontsize=16)
    ax1.set_ylabel("含该词回答占比",fontsize=16)
    ax1.legend(fontsize=14)
    ax1.grid(alpha=0.3)

    for w in FALLING_WORDS:
        ax2.plot(years, yearly_rate(w), marker="o", ms=3, label=w)
    ax2.set_title("下降组（积极话语）",fontsize=18)
    ax2.set_xlabel("年份",fontsize=16)
    ax2.legend(fontsize=14)
    ax2.grid(alpha=0.3)

    fig.suptitle("标志性词汇的年度出现率变化",fontsize=16)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "marker_words.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"[L2] 图: {path}")


def run() -> None:
    ensure_dirs()
    setup_cn_font()
    df = read_csv(LABELED_ANSWERS)

    fig_emotion_dist(df)
    fig_emotion_trend(df)
    fig_marker_words(df)
    print("[L2] RQ1 描述性分析完成")


if __name__ == "__main__":
    run()
