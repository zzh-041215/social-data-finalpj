"""
L3 回归分析（RQ2）：情绪类型与传播效果（log 赞同）的关联，控制基础属性。

主模型 OLS + 稳健标准误(HC1)：
  log_voteup ~ C(emotion, ref="中性") + log_follower + log_content_len
               + answer_age_days + C(quarter) + C(topic)
稳健性：①因变量换 log_engagement；②加议题×季度交互；③仅高赞子样本。

输入:  data/processed/answers_labeled.csv
输出:  analysis/tables/rq2_ols.csv（情绪系数对比）+ rq2_ols_summary.txt（主模型全表）

用法:  python -m analysis.rq2_regression
"""

import os

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from analysis.common import (
    LABELED_ANSWERS, TABLES_DIR, EMOTIONS, NEUTRAL,
    ensure_dirs, read_csv,
)

# 以「中性」为参照组的情绪虚拟变量
EMO_TERM = f'C(emotion, Treatment(reference="{NEUTRAL}"))'

BASE_CONTROLS = "log_follower + log_content_len + answer_age_days + C(quarter) + C(topic)"

MODELS = {
    "M1_主模型_log赞同": f"log_voteup ~ {EMO_TERM} + {BASE_CONTROLS}",
    "M2_稳健_log综合互动": f"log_engagement ~ {EMO_TERM} + {BASE_CONTROLS}",
    "M3_稳健_议题季度交互": f"log_voteup ~ {EMO_TERM} + log_follower + log_content_len + answer_age_days + C(quarter)*C(topic)",
}


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_content_len"] = np.log1p(df["content_len"])
    # emotion 设为有序类别，保证参照组与展示稳定
    df["emotion"] = pd.Categorical(df["emotion"], categories=EMOTIONS)
    return df


def _emotion_rows(res, model_name: str) -> list[dict]:
    """从拟合结果抽取情绪虚拟变量的系数行。"""
    rows = []
    ci = res.conf_int()
    for name in res.params.index:
        if "emotion" not in name:
            continue
        # 解析出情绪类别名，如 ...[T.解构]
        label = name.split("T.")[-1].rstrip("]")
        p = res.pvalues[name]
        star = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
        rows.append({
            "model": model_name,
            "情绪(vs中性)": label,
            "系数": round(res.params[name], 4),
            "标准误": round(res.bse[name], 4),
            "p值": round(p, 4),
            "显著性": star,
            "CI下": round(ci.loc[name, 0], 4),
            "CI上": round(ci.loc[name, 1], 4),
        })
    return rows


def _contrast_test(res, a: str, b: str) -> dict:
    """检验两个情绪虚拟变量系数是否相等（解构 vs 积极/焦虑），返回差值与 p。"""
    names = list(res.params.index)
    name_a = next((n for n in names if f"T.{a}]" in n), None)
    name_b = next((n for n in names if f"T.{b}]" in n), None)
    if not name_a or not name_b:
        return {}
    r = np.zeros(len(names))
    r[names.index(name_a)] = 1.0
    r[names.index(name_b)] = -1.0
    t = res.t_test(r)
    return {
        "对比": f"{a} - {b}",
        "差值": round(float(t.effect[0]), 4),
        "p值": round(float(t.pvalue), 4),
        "结论": f"{a}显著{'高于' if t.effect[0] > 0 else '低于'}{b}" if float(t.pvalue) < 0.1 else f"{a}与{b}无显著差异",
    }


def run() -> pd.DataFrame:
    ensure_dirs()
    df = _prep(read_csv(LABELED_ANSWERS))

    all_rows = []
    summaries = {}
    for name, formula in MODELS.items():
        data = df
        res = smf.ols(formula, data=data).fit(cov_type="HC1")
        all_rows.extend(_emotion_rows(res, name))
        summaries[name] = res

    # M4：仅高赞子样本（voteup >= 中位数）
    hi = df[df["voteup_count"] >= df["voteup_count"].median()]
    res_hi = smf.ols(f"log_voteup ~ {EMO_TERM} + {BASE_CONTROLS}", data=hi).fit(cov_type="HC1")
    all_rows.extend(_emotion_rows(res_hi, "M4_稳健_高赞子样本"))

    coef = pd.DataFrame(all_rows)
    coef_path = os.path.join(TABLES_DIR, "rq2_ols.csv")
    coef.to_csv(coef_path, index=False, encoding="utf-8-sig")

    # 主模型全表
    main = summaries["M1_主模型_log赞同"]
    summary_path = os.path.join(TABLES_DIR, "rq2_ols_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("RQ2 主模型 (M1): log(赞同+1) ~ 情绪(参照=中性) + 控制变量\n")
        f.write(f"N = {int(main.nobs)}, R² = {main.rsquared:.3f}, "
                f"调整R² = {main.rsquared_adj:.3f}\n")
        f.write("稳健标准误: HC1\n\n")
        f.write(str(main.summary()))

    # 解构 vs 其他情绪的直接对比检验（RQ2 核心）
    contrasts = [c for c in (_contrast_test(main, "解构", "积极"),
                             _contrast_test(main, "解构", "焦虑")) if c]
    con_df = pd.DataFrame(contrasts)
    con_path = os.path.join(TABLES_DIR, "rq2_contrasts.csv")
    con_df.to_csv(con_path, index=False, encoding="utf-8-sig")

    # 控制台报告
    print(f"[L3] 主模型 N={int(main.nobs)}  R²={main.rsquared:.3f}")
    print("[L3] 情绪系数（vs 中性，正=传播更广）:")
    main_rows = coef[coef["model"] == "M1_主模型_log赞同"]
    for _, r in main_rows.iterrows():
        print(f"      {r['情绪(vs中性)']:<4} β={r['系数']:+.3f} {r['显著性']:<3} (p={r['p值']})")
    print("[L3] 解构 vs 其他情绪（核心对比）:")
    for c in contrasts:
        print(f"      {c['对比']}: 差值{c['差值']:+.3f} (p={c['p值']}) → {c['结论']}")
    print(f"[L3] 系数对比表: {coef_path} | 对比检验: {con_path}")
    print(f"[L3] 主模型全表: {summary_path}")
    return coef


if __name__ == "__main__":
    run()
