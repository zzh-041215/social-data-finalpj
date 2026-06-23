"""
分析模块共享工具：路径常量、中文绘图字体、议题映射、I/O 辅助。
"""

import os

import pandas as pd

# ============================================================
# 路径
# ============================================================
ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ANALYSIS_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

FIGURES_DIR = os.path.join(ANALYSIS_DIR, "figures")
TABLES_DIR = os.path.join(ANALYSIS_DIR, "tables")
LEXICON_DIR = os.path.join(ANALYSIS_DIR, "lexicon")

# 原始数据
RAW_ANSWERS = os.path.join(RAW_DIR, "answers.csv")
RAW_COMMENTS = os.path.join(RAW_DIR, "comments.csv")

# 处理后数据
CLEAN_ANSWERS = os.path.join(PROCESSED_DIR, "answers_clean.csv")
CLEAN_COMMENTS = os.path.join(PROCESSED_DIR, "comments_clean.csv")
LABELED_ANSWERS = os.path.join(PROCESSED_DIR, "answers_labeled.csv")
ANNOTATION_SAMPLE = os.path.join(PROCESSED_DIR, "annotation_sample.csv")

# 四类情绪标签（中性为回归参照组）
EMOTIONS = ["积极", "焦虑", "解构", "中性"]
NEUTRAL = "中性"

# 议题顺序（用于稳定的图例/分组）
TOPICS = ["婚育决策", "置业决策", "职业与阶层", "综合人生规划"]


def ensure_dirs() -> None:
    """确保所有输出目录存在。"""
    for d in (PROCESSED_DIR, FIGURES_DIR, TABLES_DIR, LEXICON_DIR):
        os.makedirs(d, exist_ok=True)


def setup_cn_font() -> None:
    """配置 matplotlib 中文字体（Windows 优先 SimHei/微软雅黑）。"""
    import matplotlib
    matplotlib.use("Agg")  # 无界面后端，仅出图文件
    import matplotlib.pyplot as plt

    for font in ("Microsoft YaHei", "SimHei", "DengXian", "SimSun"):
        try:
            plt.rcParams["font.sans-serif"] = [font]
            break
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.bbox"] = "tight"


def read_csv(path: str) -> pd.DataFrame:
    """统一用 utf-8-sig 读取（与爬虫写出的编码一致）。"""
    return pd.read_csv(path, encoding="utf-8-sig")


def write_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
