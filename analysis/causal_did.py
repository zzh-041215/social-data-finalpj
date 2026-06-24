"""
L4 因果模块（对接大纲 §6.3，课程方法）：事件研究 + DiD + 作者固定效应。

研究设计（准实验）：
  以「孔乙己文学」2023Q1 作为对就业/学历叙事的外生话语冲击。
  处理组 = 职业与阶层议题（直接受冲击）；对照组 = 婚育决策 + 置业决策。
  （综合人生规划含躺平/摆烂等，本身即解构话语载体，内生，故排除出对照组。）
  结果变量 = 关键词×季度 cell 内「解构话语占比」（话语构成），
            以及 cell 平均 log(赞同)（传播强度）。

  ① 事件研究：相对事件季度的 leads/lags 系数，检验平行趋势。
  ② TWFE-DiD： Y_ut = α_unit + δ_quarter + β·(Treat×Post) + ε，按关键词聚类。
  ③ 作者固定效应：回答级 log_voteup ~ 情绪 + 控制 + 作者FE，缓解 OVB。

输入:  data/processed/answers_labeled.csv
输出:  analysis/figures/event_study.png
       analysis/tables/{did.csv, author_fe.csv}

用法:  python -m analysis.causal_did
"""

import os

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

from analysis.common import (
    LABELED_ANSWERS, FIGURES_DIR, TABLES_DIR,
    ensure_dirs, setup_cn_font, read_csv,
)
from crawler.config import KEYWORD_GROUPS

EVENT_QUARTER = "2023Q1"          # 孔乙己文学
TREAT_TOPIC = "职业与阶层"
CONTROL_TOPICS = ["婚育决策", "置业决策"]
EVENT_WINDOW = 4                  # 事件研究窗口：前后各 4 个季度（外侧合并）

# 关键词→议题映射（复用爬虫配置）
KW2TOPIC = {kw: topic for topic, kws in KEYWORD_GROUPS.items() for kw in kws}


def _quarter_index(quarters: pd.Series) -> dict[str, int]:
    uniq = sorted(quarters.unique())
    return {q: i for i, q in enumerate(uniq)}


def build_panel(df: pd.DataFrame) -> pd.DataFrame:
    """构造 关键词×季度 面板：cell 内解构占比、平均 log 赞同、样本量。"""
    df = df[df["topic"].isin([TREAT_TOPIC] + CONTROL_TOPICS)].copy()
    df["is_dec"] = (df["emotion"] == "解构").astype(float)

    panel = (
        df.groupby(["keyword_searched", "quarter"])
        .agg(dec_share=("is_dec", "mean"),
             mean_log_voteup=("log_voteup", "mean"),
             n=("answer_id", "size"))
        .reset_index()
        .rename(columns={"keyword_searched": "unit"})
    )
    panel["topic"] = panel["unit"].map(KW2TOPIC)
    panel["treat"] = (panel["topic"] == TREAT_TOPIC).astype(int)

    qidx = _quarter_index(df["quarter"])
    panel["qidx"] = panel["quarter"].map(qidx)
    panel["post"] = (panel["qidx"] >= qidx[EVENT_QUARTER]).astype(int)
    panel["did"] = panel["treat"] * panel["post"]
    panel["rel"] = panel["qidx"] - qidx[EVENT_QUARTER]  # 相对事件季度
    return panel


def run_did(panel: pd.DataFrame) -> pd.DataFrame:
    """TWFE-DiD，结果变量分别为解构占比与平均 log 赞同，按关键词聚类。"""
    rows = []
    for yvar, ylabel in [("dec_share", "解构话语占比"), ("mean_log_voteup", "平均log赞同")]:
        res = smf.wls(
            f"{yvar} ~ did + C(unit) + C(quarter)",
            data=panel, weights=panel["n"],
        ).fit(cov_type="cluster", cov_kwds={"groups": panel["unit"]})
        b = res.params["did"]
        p = res.pvalues["did"]
        star = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
        rows.append({
            "结果变量": ylabel,
            "DiD系数(Treat×Post)": round(b, 4),
            "标准误": round(res.bse["did"], 4),
            "p值": round(p, 4),
            "显著性": star,
            "N_cells": int(res.nobs),
            "聚类数(关键词)": panel["unit"].nunique(),
        })
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(TABLES_DIR, "did.csv"), index=False, encoding="utf-8-sig")
    return out


def fig_event_study(panel: pd.DataFrame) -> None:
    """事件研究：相对事件季度的 Treat 动态效应（解构占比），检验平行趋势。"""
    p = panel.copy()
    # 外侧合并到窗口边界，减少稀疏尾部噪声
    p["rel_w"] = p["rel"].clip(-EVENT_WINDOW, EVENT_WINDOW)

    # 生成 treat×相对季度 哑变量，省略 k=-1 作为基准
    coefs, lowers, uppers, ks = [], [], [], []
    dummy_cols = []
    for k in range(-EVENT_WINDOW, EVENT_WINDOW + 1):
        if k == -1:
            continue
        # patsy 安全列名（避免负号被当成减法）
        col = f"d_m{abs(k)}" if k < 0 else f"d_p{k}"
        p[col] = ((p["rel_w"] == k) & (p["treat"] == 1)).astype(int)
        dummy_cols.append((k, col))

    formula = "dec_share ~ " + " + ".join(c for _, c in dummy_cols) + " + C(unit) + C(quarter)"
    res = smf.wls(formula, data=p, weights=p["n"]).fit(
        cov_type="cluster", cov_kwds={"groups": p["unit"]})

    ci = res.conf_int()
    for k, col in dummy_cols:
        ks.append(k)
        coefs.append(res.params[col])
        lowers.append(ci.loc[col, 0])
        uppers.append(ci.loc[col, 1])
    # 基准点 k=-1 = 0
    ks.append(-1); coefs.append(0.0); lowers.append(0.0); uppers.append(0.0)
    order = np.argsort(ks)
    ks = np.array(ks)[order]; coefs = np.array(coefs)[order]
    lowers = np.array(lowers)[order]; uppers = np.array(uppers)[order]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(ks, coefs, yerr=[coefs - lowers, uppers - coefs],
                fmt="o-", capsize=3, color="#d62728")
    ax.axhline(0, color="gray", lw=0.8)
    ax.axvline(-0.5, color="black", ls="--", lw=1, alpha=0.6)
    ax.text(-0.4, ax.get_ylim()[1] * 0.9, "事件:孔乙己文学\n(2023Q1)", fontsize=14)
    ax.set_xlabel("相对事件的季度数（0=2023Q1；负=事前）",fontsize=14)
    ax.set_ylabel("处理组解构占比的动态效应",fontsize=14)
    ax.set_title("事件研究：职业议题解构话语占比相对冲击的动态变化\n",fontsize=16)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "event_study.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"[L4] 图: {path}")


def run_author_fe(df: pd.DataFrame) -> pd.DataFrame:
    """作者固定效应回归（缓解 OVB）：仅靠多产作者的组内变异识别情绪效应。"""
    from linearmodels.panel import PanelOLS

    d = df.copy()
    d["log_content_len"] = np.log1p(d["content_len"])
    # 仅保留出现≥2次的作者（FE 才有组内变异）
    counts = d["author_url_token"].value_counts()
    multi = counts[counts >= 2].index
    d = d[d["author_url_token"].isin(multi)].copy()

    # 情绪虚拟变量（中性为参照）
    for emo in ["积极", "焦虑", "解构"]:
        d[f"emo_{emo}"] = (d["emotion"] == emo).astype(float)

    # 面板索引：实体=作者，时间=数值化季度（linearmodels 要求时间维数值/日期型）
    qidx = _quarter_index(d["quarter"])
    d["qnum"] = d["quarter"].map(qidx)
    # 同一作者同一季度多篇会导致重复索引，用累加序号去重
    d["occ"] = d.groupby(["author_url_token", "qnum"]).cumcount()
    d["tnum"] = d["qnum"] * 100 + d["occ"]
    d = d.set_index(["author_url_token", "tnum"])
    exog_cols = ["emo_积极", "emo_焦虑", "emo_解构",
                 "log_follower", "log_content_len", "answer_age_days"]
    mod = PanelOLS(d["log_voteup"], d[exog_cols], entity_effects=True,
                   check_rank=False, drop_absorbed=True)
    res = mod.fit(cov_type="clustered", cluster_entity=True)

    rows = []
    for emo in ["积极", "焦虑", "解构"]:
        col = f"emo_{emo}"
        if col not in res.params.index:
            continue
        p = res.pvalues[col]
        star = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
        rows.append({
            "情绪(vs中性)": emo,
            "系数": round(res.params[col], 4),
            "标准误": round(res.std_errors[col], 4),
            "p值": round(p, 4),
            "显著性": star,
        })
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(TABLES_DIR, "author_fe.csv"), index=False, encoding="utf-8-sig")
    print(f"[L4] 作者FE: N={int(res.nobs)} (≥2篇作者), 实体数={d.index.get_level_values(0).nunique()}")
    return out


def run() -> None:
    ensure_dirs()
    setup_cn_font()
    df = read_csv(LABELED_ANSWERS)

    panel = build_panel(df)
    print(f"[L4] 面板: {len(panel)} 个 关键词×季度 cell, "
          f"处理组关键词 {panel[panel.treat==1]['unit'].nunique()} / "
          f"对照组 {panel[panel.treat==0]['unit'].nunique()}")

    did = run_did(panel)
    print("[L4] DiD 结果（孔乙己2023Q1，处理=职业议题）:")
    for _, r in did.iterrows():
        print(f"      {r['结果变量']}: β={r['DiD系数(Treat×Post)']:+.3f} {r['显著性']} (p={r['p值']})")

    fig_event_study(panel)

    afe = run_author_fe(df)
    print("[L4] 作者固定效应（情绪 vs 中性）:")
    for _, r in afe.iterrows():
        print(f"      {r['情绪(vs中性)']:<4} β={r['系数']:+.3f} {r['显著性']:<3} (p={r['p值']})")
    print("[L4] 因果模块完成（结果定位为对关联的准实验稳健性检验，非严格因果）")


if __name__ == "__main__":
    run()
