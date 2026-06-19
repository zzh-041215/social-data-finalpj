"""
B站人生规划话语数据采集器 — 全局配置

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
        "恐婚恐育", "生不起孩子", "养不起娃",
        # 积极/理想主义 (Type 1)
        "结婚的意义", "为什么生孩子",
        # 中性/信息型 (Type 4)
        "婚育成本", "生育政策", "丁克",
    ],

    # ── 置业决策 ──
    "置业决策": [
        # 解构
        "韭菜买房",
        # 焦虑
        "买不起房", "房价太高",
        # 积极
        "买房上车", "终于买房了",
        # 中性
        "要不要买房", "租房还是买房", "买房攻略",
        "首付", "房价走势", "房产政策",
    ],

    # ── 职业与阶层 ──
    "职业与阶层": [
        # 解构
        "孔乙己", "学历贬值", "阶层固化", "做题家",
        # 焦虑
        "35岁危机", "中年危机", "就业难", "失业", "内卷", "996",
        # 积极
        "考研上岸", "考公上岸", "财务自由", "副业", "自律",
        # 中性
        "考公", "考研值不值", "职业规划", "行业分析",
        "求职技巧", "简历", "跳槽攻略",
    ],

    # ── 综合人生规划 ──
    "综合人生规划": [
        # 解构
        "躺平", "摆烂", "佛系", "草台班子",
        # 焦虑
        "精神内耗", "迷茫", "年龄焦虑", "出路", "负债",
        # 积极
        "奋斗", "人生规划", "逆袭", "认知升级",
        "重启人生", "被动收入",
        # 中性
        "二十几岁怎么活", "人生建议", "时间管理",
        "理财", "养老规划", "避坑指南", "底层逻辑",
    ],
}

# 扁平化为列表（用于迭代）
ALL_KEYWORDS: list[str] = [
    kw for kws in KEYWORD_GROUPS.values() for kw in kws
]


# ============================================================
# B站 API 端点（均无需登录）
# ============================================================
API_SEARCH = "https://api.bilibili.com/x/web-interface/search/type"
API_VIDEO_INFO = "https://api.bilibili.com/x/web-interface/view"
API_COMMENTS = "https://api.bilibili.com/x/v2/reply/main"
API_USER_INFO = "https://api.bilibili.com/x/space/acc/info"

# ============================================================
# HTTP 请求头（模拟浏览器）
# ============================================================
HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.bilibili.com/",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Cache-Control": "max-age=0",
}

# API请求头（用于JSON接口）
API_HEADERS: dict[str, str] = {
    "User-Agent": HEADERS["User-Agent"],
    "Referer": "https://www.bilibili.com/",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

# ============================================================
# 自适应限速参数
# ============================================================
DELAY_MIN = 0.8          # 正常请求最小延迟（秒）— 低于0.5会频繁429
DELAY_MAX = 1.5          # 正常请求最大延迟（秒）— 随机取 min~max
RATE_LIMIT_BACKOFF_BASE = 4.0   # 收到429后初始退避秒数
RATE_LIMIT_BACKOFF_MAX = 60.0   # 最大退避秒数
MAX_RETRIES = 3                # 单次请求最大重试次数
NETWORK_PAUSE_SEC = 300        # 连续网络错误后暂停5分钟
MAX_CONSECUTIVE_NETWORK_ERRORS = 10  # 超过此次数则退出

# ============================================================
# 搜索 & 分页参数
# ============================================================
SEARCH_PAGE_SIZE = 50           # B站每页最大返回数
MAX_PAGES_PER_KEYWORD = 20      # 每个关键词最多搜索20页（1000条）
# 使用多种排序方式确保时间覆盖（B站每种排序最多返回1000条）
# pubdate: 最新内容（2024-2026）; click: 最多播放（回溯到2021-2022）; dm: 最多弹幕（高互动）
SEARCH_ORDERS = ["pubdate", "click", "dm"]
COMMENT_SORT = 2                # 2=按热度排序
MAX_COMMENTS_PER_VIDEO = 20     # 每视频最多取20条评论

# ============================================================
# 过滤规则
# ============================================================
MIN_VIEW_COUNT = 1000           # 最低播放量
MIN_DURATION_SEC = 30           # 最短时长（秒）
PUBLISH_AFTER = "2020-01-01"    # 最早时间
PUBLISH_BEFORE = "2025-12-31"   # 最晚时间

# 标题中匹配到以下关键词则跳过（广告/非内容视频）
AD_PATTERNS = [
    r"广告", r"合作", r"推广", r"抽奖",
    r"福利", r"带货", r"限时", r"优惠",
    r"红包", r"秒杀", r"拼团",
]

# ============================================================
# 爬取批次 & Checkpoint
# ============================================================
BATCH_SIZE = 10                 # 每处理多少条视频flush一次CSV
CHECKPOINT_INTERVAL = 10        # 每处理多少条视频保存一次checkpoint
SEARCH_WORKERS = 4              # 并行搜索线程数（= 议题组数量）

# ============================================================
# 文件路径
# ============================================================
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
CHECKPOINT_DIR = os.path.join(DATA_DIR, "checkpoints")
LOG_DIR = os.path.join(DATA_DIR, "logs")

VIDEOS_CSV = os.path.join(RAW_DIR, "videos.csv")
COMMENTS_CSV = os.path.join(RAW_DIR, "comments.csv")
CHECKPOINT_STATE = os.path.join(CHECKPOINT_DIR, "state.json")
CHECKPOINT_BVIDS = os.path.join(CHECKPOINT_DIR, "crawled_bvids.txt")

# ============================================================
# CSV 列定义
# ============================================================
VIDEO_COLUMNS = [
    "bvid", "aid", "title", "description", "publish_time", "duration_sec",
    "tags", "tag_list",
    "category_id", "category_name",
    "keyword_searched", "topic",
    "uploader_name", "uploader_mid", "uploader_follower_count",
    "view_count", "like_count", "coin_count", "favorite_count",
    "share_count", "comment_count", "danmaku_count",
    "video_url",
    "crawled_at",
]

COMMENT_COLUMNS = [
    "comment_id", "bvid", "oid",
    "text", "text_length",
    "like_count", "reply_count",
    "publish_time", "is_hot",
    "parent_id",
    "crawled_at",
]


def ensure_dirs() -> None:
    """确保所有数据目录存在。"""
    for d in [DATA_DIR, RAW_DIR, PROCESSED_DIR, CHECKPOINT_DIR, LOG_DIR]:
        os.makedirs(d, exist_ok=True)
