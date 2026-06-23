"""
L1b 情绪分类人工校验：抽样导出 + Cohen's Kappa 一致性评估。

两种模式（自动判定）：
  1) 导出模式：若 annotation_sample.csv 不存在 → 分层随机抽样 500 条导出，
     供人工在 human_label 列填四类标签（积极/焦虑/解构/中性）。
  2) 评分模式：若样本文件已存在且 human_label 有填写 → 计算 Cohen's Kappa
     与混淆矩阵，写入 analysis/tables/kappa.txt。

用法:
  python -m analysis.validate_emotion           # 自动：先导出，填完后再跑即评分
  python -m analysis.validate_emotion --export  # 强制重新导出（不覆盖已填标签）
"""

import argparse
import os

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from analysis.common import (
    LABELED_ANSWERS, ANNOTATION_SAMPLE, TABLES_DIR, EMOTIONS,
    ensure_dirs, read_csv, write_csv,
)

SAMPLE_SIZE = 500
SEED = 42
CONTENT_CHARS = 600  # 导出正文截断长度，便于人工阅读


def export_sample() -> None:
    """分层随机抽样导出待标注样本（按模型情绪比例分层，保持代表性）。"""
    df = read_csv(LABELED_ANSWERS)

    frac = SAMPLE_SIZE / len(df)
    # 按模型情绪分层随机抽样，保持类别比例代表性
    sample = df.groupby("emotion", group_keys=False).sample(frac=frac, random_state=SEED)
    if len(sample) > SAMPLE_SIZE:
        sample = sample.sample(n=SAMPLE_SIZE, random_state=SEED)

    out = pd.DataFrame({
        "answer_id": sample["answer_id"].values,
        "topic": sample["topic"].values,
        "keyword_searched": sample["keyword_searched"].values,
        "question_title": sample["question_title"].values,
        "content": sample["content"].astype(str).str.slice(0, CONTENT_CHARS).values,
        "model_emotion": sample["emotion"].values,
        "emo_confidence": sample["emo_confidence"].values,
        "human_label": "",  # 待人工填写：积极/焦虑/解构/中性
    })
    write_csv(out, ANNOTATION_SAMPLE)
    print(f"[L1b] 已导出 {len(out)} 条待标注样本: {ANNOTATION_SAMPLE}")
    print(f"[L1b] 请在 human_label 列填写四类之一：{'/'.join(EMOTIONS)}")
    print(f"[L1b] 填完后重新运行本脚本即自动计算 Cohen's Kappa。")


def score_sample() -> None:
    """对已标注样本计算 Kappa 与混淆矩阵。"""
    df = read_csv(ANNOTATION_SAMPLE)
    df["human_label"] = df["human_label"].astype(str).str.strip()
    labeled = df[df["human_label"].isin(EMOTIONS)]

    if len(labeled) == 0:
        print(f"[L1b] 样本文件已存在但 human_label 尚未填写，跳过评分。")
        print(f"[L1b] 请编辑 {ANNOTATION_SAMPLE} 后重跑。")
        return

    kappa = cohen_kappa_score(
        labeled["human_label"], labeled["model_emotion"], labels=EMOTIONS
    )
    acc = (labeled["human_label"] == labeled["model_emotion"]).mean()
    confusion = pd.crosstab(
        labeled["human_label"], labeled["model_emotion"],
        rownames=["人工"], colnames=["模型"], dropna=False,
    ).reindex(index=EMOTIONS, columns=EMOTIONS, fill_value=0)

    ensure_dirs()
    out_path = os.path.join(TABLES_DIR, "kappa.txt")
    lines = [
        "情绪分类人工校验报告",
        "=" * 40,
        f"已标注样本量: {len(labeled)} / {len(df)}",
        f"准确率(Accuracy): {acc:.3f}",
        f"Cohen's Kappa: {kappa:.3f}   "
        f"({'极好' if kappa>=0.8 else '良好' if kappa>=0.6 else '中等' if kappa>=0.4 else '一般'})",
        "",
        "混淆矩阵（行=人工，列=模型）:",
        confusion.to_string(),
    ]
    report = "\n".join(lines)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(report)
    print(f"\n[L1b] 报告已写入: {out_path}")


def run(force_export: bool = False) -> None:
    ensure_dirs()
    if force_export or not os.path.exists(ANNOTATION_SAMPLE):
        export_sample()
    else:
        score_sample()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--export", action="store_true", help="强制重新导出抽样模板")
    args = p.parse_args()
    if args.export and os.path.exists(ANNOTATION_SAMPLE):
        print(f"[L1b] 警告：{ANNOTATION_SAMPLE} 已存在，--export 会覆盖已填标签。"
              f" 如确需重导，请先手动备份。已取消。")
    else:
        run(force_export=args.export)
