"""
L1 情绪分类（词典法）：对回答正文（主）+ 问题标题（辅）做四类情绪打分。

四类：积极 / 焦虑 / 解构 / 中性（中性=情绪信号稀薄的残差类）
方法：jieba 分词 → 三类情绪词典加权计分（含否定/程度副词规则）→ argmax；
      情绪密度过低或无主导类则判为「中性」。

输入:  data/processed/answers_clean.csv
输出:  data/processed/answers_labeled.csv（新增 emotion / s_积极 / s_焦虑 / s_解构 / emo_confidence）

用法:  python -m analysis.emotion_classify
"""

import os

import numpy as np
import pandas as pd
import jieba

from analysis.common import (
    LEXICON_DIR, CLEAN_ANSWERS, LABELED_ANSWERS,
    NEUTRAL, ensure_dirs, read_csv, write_csv,
)

# ============================================================
# 规则词表
# ============================================================
NEGATIONS = {"不", "没", "没有", "无", "别", "非", "未", "莫", "勿", "甭", "毫无", "并非", "不再"}
INTENSIFIERS = {"很", "非常", "特别", "极其", "极度", "太", "超", "超级", "巨", "贼",
                "十分", "格外", "相当", "挺", "更", "最", "尤其", "异常", "分外"}

CONTENT_WEIGHT = 1.0
TITLE_WEIGHT = 0.5
INTENSIFY_FACTOR = 1.5
WINDOW = 2  # 否定/程度副词的前向窗口

# 中性判定阈值（情绪词密度 = 情绪命中权重 / 词数）。低于此值判中性。
# 注：曾尝试改为与长度无关的计数法(HIT_FLOOR)以提升 RQ2 显著性，但 497 条人工
# 标注显示密度法 Kappa=0.79（人工中性占比 29.8% ≈ 本法 31%）显著优于计数法
# (Kappa≤0.69)。故保留密度法——以人工校验为准，不为显著性牺牲分类准确度。
DENSITY_MIN = 0.010
# 主导类份额阈值：最高情绪类占三类总分比例低于此值视为情绪混杂 → 中性
SHARE_MIN = 0.40

CLASS_FILES = {
    "积极": "positive.txt",
    "焦虑": "anxiety.txt",
    "解构": "deconstruct.txt",
}


def load_lexicons() -> dict[str, set[str]]:
    """加载三类情绪词典，并把多字词注册进 jieba 以保证整体切分。"""
    lex: dict[str, set[str]] = {}
    for cls, fname in CLASS_FILES.items():
        path = os.path.join(LEXICON_DIR, fname)
        words = set()
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                w = line.strip()
                if not w or w.startswith("#"):
                    continue
                words.add(w)
                if len(w) >= 2:
                    jieba.add_word(w, freq=10000)  # 防止被切碎，如「割韭菜」
        lex[cls] = words
    return lex


def _score_tokens(tokens: list[str], lex: dict[str, set[str]], weight: float,
                  scores: dict[str, float]) -> None:
    """对一段 token 序列累计三类情绪得分（就地更新 scores）。"""
    for i, tok in enumerate(tokens):
        # 命中哪一类
        hit_cls = None
        for cls, words in lex.items():
            if tok in words:
                hit_cls = cls
                break
        if hit_cls is None:
            continue

        w = weight
        # 前向窗口找程度副词 / 否定词
        negated = False
        for j in range(max(0, i - WINDOW), i):
            if tokens[j] in INTENSIFIERS:
                w *= INTENSIFY_FACTOR
            if tokens[j] in NEGATIONS:
                negated = True

        if negated:
            # 否定积极词 → 视为负面（计入焦虑）；否定焦虑/解构 → 显著削弱
            if hit_cls == "积极":
                scores["焦虑"] += w * 0.8
                continue
            else:
                w *= 0.25

        scores[hit_cls] += w


def classify_text(content: str, title: str, lex: dict[str, set[str]]) -> dict:
    """对单条回答打分并给出标签。"""
    c_tokens = jieba.lcut(str(content))
    t_tokens = jieba.lcut(str(title))

    scores = {"积极": 0.0, "焦虑": 0.0, "解构": 0.0}
    _score_tokens(c_tokens, lex, CONTENT_WEIGHT, scores)
    _score_tokens(t_tokens, lex, TITLE_WEIGHT, scores)

    total = sum(scores.values())
    n_tokens = max(len(c_tokens), 1)
    density = total / n_tokens

    if total <= 0 or density < DENSITY_MIN:
        # 情绪词密度过低 → 中性（信息型）
        label = NEUTRAL
        confidence = float(1.0 - min(density / DENSITY_MIN, 1.0)) if total > 0 else 1.0
    else:
        top_cls = max(scores, key=scores.get)
        share = scores[top_cls] / total
        if share < SHARE_MIN:
            label = NEUTRAL  # 多类情绪混杂、无主导 → 中性
            confidence = float(1.0 - share)
        else:
            label = top_cls
            confidence = float(share)

    return {
        "emotion": label,
        "s_积极": round(scores["积极"], 3),
        "s_焦虑": round(scores["焦虑"], 3),
        "s_解构": round(scores["解构"], 3),
        "emo_total": round(total, 3),
        "emo_density": round(total / n_tokens, 4),  # 仅诊断用
        "emo_confidence": round(confidence, 3),
    }


def run() -> pd.DataFrame:
    ensure_dirs()
    df = read_csv(CLEAN_ANSWERS)
    lex = load_lexicons()

    records = [
        classify_text(row.content, row.question_title, lex)
        for row in df.itertuples(index=False)
    ]
    res = pd.DataFrame.from_records(records)
    out = pd.concat([df.reset_index(drop=True), res], axis=1)

    write_csv(out, LABELED_ANSWERS)

    dist = out["emotion"].value_counts()
    print(f"[L1] 已标注 {len(out)} 条")
    print(f"[L1] 情绪分布: {dist.to_dict()}")
    print(f"[L1] 占比: { {k: round(v/len(out), 3) for k, v in dist.items()} }")
    print(f"[L1] 平均置信度: {out['emo_confidence'].mean():.3f}")
    print(f"[L1] 输出: {LABELED_ANSWERS}")
    return out


if __name__ == "__main__":
    run()
