"""
知乎人生规划话语数据采集器 — 全局配置

包含：关键词、API端点、限速参数、数据字段、文件路径、过滤规则
"""

import os

# ============================================================
# 项目根目录
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# 搜索关键词（~42个，四类情绪均衡覆盖）
# 标注仅为搜索阶段的粗略预期，最终分类由NLP模型完成
# ============================================================
KEYWORD_GROUPS: dict[str, list[str]] = {
    # ── 婚育决策 ──
    "婚育决策": [
        # 解构/批判 (Type 3)
        "不婚不育", "彩礼",
        # 焦虑/压力 (Type 2)
        "恐婚恐育", "生育成本", "结婚率下降",
        # 积极/理想主义 (Type 1)
        "结婚的意义", "为什么要生孩子",
        # 中性/信息型 (Type 4)
        "丁克", "生育率", "婚姻制度",
    ],

    # ── 置业决策 ──
    "置业决策": [
        # 解构
        "韭菜买房",
        # 焦虑
        "买不起房", "房价太高", "房贷压力",
        # 积极
        "买房上车",
        # 中性
        "要不要买房", "租房还是买房", "首付",
        "房价走势", "一线城市买房",
    ],

    # ── 职业与阶层 ──
    "职业与阶层": [
        # 解构
        "孔乙己", "学历贬值", "阶层固化", "做题家",
        # 焦虑
        "35岁危机", "就业难", "内卷", "996",
        # 积极
        "考研上岸", "考公上岸", "财务自由", "副业",
        # 中性
        "考公", "职业规划", "行业前景",
        "转行", "行业分析",
    ],

    # ── 综合人生规划 ──
    "综合人生规划": [
        # 解构
        "躺平", "摆烂", "佛系",
        # 焦虑
        "精神内耗", "迷茫", "年龄焦虑",
        # 积极
        "奋斗", "人生规划", "逆袭",
        "重启人生", "认知升级",
        # 中性
        "二十几岁", "人生建议",
    ],
}

# 扁平化为列表（用于迭代）
ALL_KEYWORDS: list[str] = [
    kw for kws in KEYWORD_GROUPS.values() for kw in kws
]


# ============================================================
# 知乎 API 端点
# ============================================================
API_SEARCH = "https://www.zhihu.com/api/v4/search_v3"
API_ANSWER = "https://www.zhihu.com/api/v4/answers"
API_QUESTION = "https://www.zhihu.com/api/v4/questions"
API_COMMENTS = "https://www.zhihu.com/api/v4/comments"
API_USER = "https://www.zhihu.com/api/v4/members"

# ============================================================
# HTTP 请求头（模拟浏览器）
# ============================================================
HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.zhihu.com/",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Cache-Control": "max-age=0",
}

# API请求头（用于JSON接口，额外包含知乎专用头）
API_HEADERS: dict[str, str] = {
    "User-Agent": HEADERS["User-Agent"],
    "Referer": "https://www.zhihu.com/",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "x-api-version": "3.0.40",
    "x-requested-with": "fetch",
}

# ============================================================
# 自适应限速参数
# ============================================================
DELAY_MIN = 0.8              # 正常请求最小延迟（秒）
DELAY_MAX = 1.5              # 正常请求最大延迟（秒）
RATE_LIMIT_BACKOFF_BASE = 10.0  # 收到429后初始退避秒数（加重惩罚，避免连续触发）
RATE_LIMIT_BACKOFF_MAX = 120.0  # 最大退避秒数
MAX_RETRIES = 3                # 单次请求最大重试次数
NETWORK_PAUSE_SEC = 300        # 连续网络错误后暂停5分钟
MAX_CONSECUTIVE_NETWORK_ERRORS = 10  # 超过此次数则退出

# ============================================================
# 搜索 & 分页参数
# ============================================================
SEARCH_LIMIT = 20               # 知乎每页返回数
MAX_OFFSET_PER_KEYWORD = 200    # 每个关键词最多搜索 offset=0,20,...,200（共11页）
SEARCH_TYPE = "general"         # 知乎搜索类型
MAX_COMMENTS_PER_ANSWER = 20    # 每回答最多取20条评论
DEFAULT_MAX_ANSWERS = 2500      # 默认回答采集上限（适合轻薄本 ~2h）

# ============================================================
# 过滤规则
# ============================================================
MIN_ANSWER_LENGTH = 50          # 最短回答文本长度（字符）
PUBLISH_AFTER = "2020-01-01"    # 最早时间
PUBLISH_BEFORE = "2025-12-31"   # 最晚时间

# 标题/摘要中匹配到以下关键词则跳过（广告/推广内容）
AD_PATTERNS = [
    r"广告", r"合作", r"推广", r"抽奖",
    r"福利", r"带货", r"限时", r"优惠",
    r"红包", r"秒杀", r"拼团",
    r"软文", r"营销",
]

# ============================================================
# 爬取批次 & Checkpoint
# ============================================================
BATCH_SIZE = 10                 # 每处理多少条回答flush一次CSV
CHECKPOINT_INTERVAL = 10        # 每处理多少条回答保存一次checkpoint
SEARCH_WORKERS = 4              # 并行搜索线程数（= 议题组数量）

# ============================================================
# 文件路径
# ============================================================
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
CHECKPOINT_DIR = os.path.join(DATA_DIR, "checkpoints")
LOG_DIR = os.path.join(DATA_DIR, "logs")

ANSWERS_CSV = os.path.join(RAW_DIR, "answers.csv")
COMMENTS_CSV = os.path.join(RAW_DIR, "comments.csv")
CHECKPOINT_STATE = os.path.join(CHECKPOINT_DIR, "state.json")
CHECKPOINT_ANSWER_IDS = os.path.join(CHECKPOINT_DIR, "crawled_answer_ids.txt")
COOKIES_FILE = os.path.join(PROJECT_ROOT, "crawler", "cookies.txt")

# ============================================================
# CSV 列定义
# ============================================================
ANSWER_COLUMNS = [
    "answer_id", "question_id", "question_title",
    "content", "excerpt",
    "publish_time", "created_time", "updated_time",
    "author_name", "author_url_token", "author_headline",
    "author_follower_count",
    "voteup_count", "comment_count", "view_count", "favorite_count",
    "keyword_searched", "topic",
    "answer_url",
    "crawled_at",
]

COMMENT_COLUMNS = [
    "comment_id", "answer_id", "question_id",
    "content", "content_length",
    "like_count", "reply_count",
    "publish_time",
    "parent_id",
    "crawled_at",
]


def ensure_dirs() -> None:
    """确保所有数据目录存在。"""
    for d in [DATA_DIR, RAW_DIR, PROCESSED_DIR, CHECKPOINT_DIR, LOG_DIR]:
        os.makedirs(d, exist_ok=True)
