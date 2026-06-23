"""
L0 预处理：清洗原始爬取数据，派生分析所需字段，聚合评论。

输入:  data/raw/answers.csv, data/raw/comments.csv
输出:  data/processed/answers_clean.csv, data/processed/comments_clean.csv

用法:  python -m analysis.preprocess
"""

import re
import sys

import numpy as np
import pandas as pd

from analysis.common import (
    RAW_ANSWERS, RAW_COMMENTS, CLEAN_ANSWERS, CLEAN_COMMENTS,
    ensure_dirs, read_csv, write_csv,
)

# 复用爬虫端的广告标题规则与最短长度门槛，保持清洗口径一致
from crawler.config import MIN_ANSWER_LENGTH, AD_PATTERNS

_AD_RE = re.compile("|".join(AD_PATTERNS))


def is_ad_title(title: str) -> bool:
    return bool(_AD_RE.search(str(title)))


def _aggregate_comments(comments: pd.DataFrame):
    """评论清洗 + 聚合到回答级别。返回 (回答级聚合表, 清洗后评论表)。"""
    cols = ["answer_id", "n_comments_collected", "mean_comment_like"]
    if comments.empty:
        return pd.DataFrame(columns=cols), comments

    c = comments.copy()
    # 去重 + 去空/纯表情（content_length 已在爬虫端计算，这里兜底）
    c = c.drop_duplicates(subset=["comment_id"])
    c["content"] = c["content"].fillna("").astype(str)
    c = c[c["content"].str.len() >= 2]

    agg = (
        c.groupby("answer_id")
        .agg(
            n_comments_collected=("comment_id", "count"),
            mean_comment_like=("like_count", "mean"),
        )
        .reset_index()
    )
    return agg, c


def run() -> pd.DataFrame:
    ensure_dirs()

    # ── 读取 ──
    ans = read_csv(RAW_ANSWERS)
    com = read_csv(RAW_COMMENTS)
    n0 = len(ans)

    # ── 去重（resume=False 造成的重复 + 任何意外重复）──
    ans = ans.drop_duplicates(subset=["answer_id"], keep="first").copy()
    n_dedup = n0 - len(ans)

    # ── 文本清洗与过滤 ──
    ans["content"] = ans["content"].fillna("").astype(str)
    ans["content_len"] = ans["content"].str.len()

    before_filter = len(ans)
    ans = ans[ans["content_len"] >= MIN_ANSWER_LENGTH]
    n_short = before_filter - len(ans)

    before_ad = len(ans)
    ans = ans[~ans["question_title"].apply(is_ad_title)]
    n_ad = before_ad - len(ans)

    # ── 时间派生 ──
    ans["publish_time"] = pd.to_datetime(ans["publish_time"], errors="coerce")
    ans["crawled_at"] = pd.to_datetime(ans["crawled_at"], errors="coerce")
    ans = ans[ans["publish_time"].notna()].copy()

    ans["year"] = ans["publish_time"].dt.year
    ans["quarter"] = ans["publish_time"].dt.to_period("Q").astype(str)
    ans["answer_age_days"] = (
        (ans["crawled_at"] - ans["publish_time"]).dt.total_seconds() / 86400.0
    )
    # 存续天数应为正；异常（爬取早于发布）置 NaN 再用中位数兜底
    ans.loc[ans["answer_age_days"] < 0, "answer_age_days"] = np.nan
    ans["answer_age_days"] = ans["answer_age_days"].fillna(ans["answer_age_days"].median())

    # ── 数值与对数变换（互动指标高度右偏）──
    for col in ["voteup_count", "comment_count", "thanks_count", "author_follower_count"]:
        ans[col] = pd.to_numeric(ans[col], errors="coerce").fillna(0)

    ans["engagement"] = ans["voteup_count"] + ans["thanks_count"] + ans["comment_count"]
    ans["log_voteup"] = np.log1p(ans["voteup_count"])
    ans["log_engagement"] = np.log1p(ans["engagement"])
    ans["log_follower"] = np.log1p(ans["author_follower_count"])

    # ── 评论聚合并合并 ──
    agg, com_clean = _aggregate_comments(com)
    ans = ans.merge(agg, on="answer_id", how="left")
    ans["n_comments_collected"] = ans["n_comments_collected"].fillna(0).astype(int)
    ans["mean_comment_like"] = ans["mean_comment_like"].fillna(0.0)

    # ── 输出 ──
    write_csv(ans, CLEAN_ANSWERS)
    write_csv(com_clean, CLEAN_COMMENTS)

    # ── 报告 ──
    print(f"[L0] 原始 {n0} 条 → 去重 -{n_dedup} → 短文本 -{n_short} → 广告 -{n_ad} → 清洗后 {len(ans)} 条")
    print(f"[L0] content长度 中位{ans['content_len'].median():.0f} 均值{ans['content_len'].mean():.0f}")
    print(f"[L0] 议题分布: {ans['topic'].value_counts().to_dict()}")
    print(f"[L0] 季度跨度: {ans['quarter'].min()} ~ {ans['quarter'].max()}  ({ans['quarter'].nunique()}个季度)")
    print(f"[L0] 评论(清洗后): {len(com_clean)} 条，覆盖 {agg['answer_id'].nunique()} 个回答")
    print(f"[L0] 输出: {CLEAN_ANSWERS}")
    return ans


if __name__ == "__main__":
    run()
