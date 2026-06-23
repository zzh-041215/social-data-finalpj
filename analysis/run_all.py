"""
L5 编排：一键运行完整分析流水线 L0 → L4。

用法:  python -m analysis.run_all

说明：validate_emotion 默认仅导出待标注样本；人工标注后单独运行该脚本算 Kappa。
"""

import time

from analysis import (
    preprocess, emotion_classify, validate_emotion,
    rq1_descriptive, rq2_regression, causal_did,
)


def _banner(title: str) -> None:
    print("\n" + "=" * 64)
    print(f"  {title}")
    print("=" * 64)


def main() -> None:
    t0 = time.time()

    _banner("L0 预处理")
    preprocess.run()

    _banner("L1 情绪分类（词典法）")
    emotion_classify.run()

    _banner("L1b 人工校验抽样（导出 / 若已标注则评分）")
    validate_emotion.run()

    _banner("L2 RQ1 描述性分析")
    rq1_descriptive.run()

    _banner("L3 RQ2 回归分析")
    rq2_regression.run()

    _banner("L4 因果模块（事件研究 + DiD + 作者FE）")
    causal_did.run()

    _banner(f"全流程完成，用时 {time.time() - t0:.1f}s")
    print("产物: data/processed/  analysis/figures/  analysis/tables/")
    print("结果综述见: analysis/RESULTS.md")


if __name__ == "__main__":
    main()
