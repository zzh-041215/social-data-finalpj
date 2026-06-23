"""
工具模块：自适应限速器、重试装饰器、网络中断检测、日志系统、辅助函数
"""

import time
import random
import logging
import re
import hashlib
import urllib.parse
import functools
import uuid
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Any, Optional

import requests

from crawler.config import (
    DELAY_MIN, DELAY_MAX,
    RATE_LIMIT_BACKOFF_BASE, RATE_LIMIT_BACKOFF_MAX,
    MAX_RETRIES, NETWORK_PAUSE_SEC, MAX_CONSECUTIVE_NETWORK_ERRORS,
    LOG_DIR, AD_PATTERNS,
    PUBLISH_AFTER, PUBLISH_BEFORE,
)

# ============================================================
# 日志系统
# ============================================================

def setup_logging(log_dir: str = LOG_DIR, name: str = "crawler") -> logging.Logger:
    """配置双输出日志：控制台（INFO）+ 文件（DEBUG）。"""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(log_dir) / f"crawl_{timestamp}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # 控制台 handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)-5s] %(message)s",
        datefmt="%H:%M:%S"
    ))
    logger.addHandler(ch)

    # 文件 handler
    fh = logging.FileHandler(str(log_path), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)-5s] %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(fh)

    logger.info(f"日志文件: {log_path}")
    return logger


# 模块级 logger（在 run.py 中调用 setup_logging 后替换）
logger = logging.getLogger("crawler")


# ============================================================
# 自适应限速器
# ============================================================

class AdaptiveRateLimiter:
    """
    自适应限速器：正常时使用短延迟，收到429后指数退避。

    用法：
        limiter = AdaptiveRateLimiter()
        for req in requests:
            limiter.wait()
            ...
            if response.status_code == 429:
                limiter.report_429()
    """

    def __init__(self, min_delay: float = DELAY_MIN, max_delay: float = DELAY_MAX):
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.last_request_time = 0.0
        self._backoff_until = 0.0       # 在此时间之前强制等待
        self._current_backoff = RATE_LIMIT_BACKOFF_BASE
        self._consecutive_429s = 0
        self._lock = threading.Lock()   # 多线程安全：防止竞态导致扎堆请求触发429

    def wait(self) -> float:
        """
        等待到可以发下一个请求，返回实际等待秒数。
        线程安全：用锁保证请求间隔均匀，防止多线程同时读到旧时间戳而扎堆发请求。
        锁只保护计时逻辑，HTTP请求本身仍然并行（锁在请求发出前已释放）。
        """
        with self._lock:
            now = time.time()

            # 如果处于退避状态，先完成退避
            if now < self._backoff_until:
                sleep = self._backoff_until - now
                time.sleep(sleep)
                now = time.time()

            # 计算自上次请求以来的间隔
            elapsed = now - self.last_request_time
            target_delay = random.uniform(self.min_delay, self.max_delay)
            wait_time = max(0.0, target_delay - elapsed)

            if wait_time > 0:
                time.sleep(wait_time)

            self.last_request_time = time.time()
            return wait_time

    def report_429(self) -> None:
        """收到429后调用，进入指数退避。线程安全。"""
        with self._lock:
            self._consecutive_429s += 1
            backoff = min(
                self._current_backoff * (2 ** (self._consecutive_429s - 1)),
                RATE_LIMIT_BACKOFF_MAX
            )
            self._backoff_until = time.time() + backoff
            logger.warning(f"收到429限速，退避 {backoff:.1f}s "
                           f"(连续第{self._consecutive_429s}次)")

    def report_success(self) -> None:
        """请求成功后调用，逐步恢复正常延迟。线程安全。"""
        with self._lock:
            if self._consecutive_429s > 0:
                self._consecutive_429s = max(0, self._consecutive_429s - 1)
            self._backoff_until = 0.0

    @property
    def is_backing_off(self) -> bool:
        return time.time() < self._backoff_until


# ============================================================
# 重试装饰器 + 网络中断检测
# ============================================================

class NetworkErrorCounter:
    """全局网络错误计数器（检测长时间断网）。"""

    def __init__(self):
        self.consecutive_errors = 0
        self.consecutive_successes = 0
        self.total_errors = 0

    def record_error(self) -> None:
        self.consecutive_errors += 1
        self.consecutive_successes = 0
        self.total_errors += 1

    def record_success(self) -> None:
        self.consecutive_successes += 1
        self.consecutive_errors = 0

    def should_pause(self) -> bool:
        """返回True表示应该暂停（3-9次连续错误）。"""
        return 3 <= self.consecutive_errors < MAX_CONSECUTIVE_NETWORK_ERRORS

    def should_abort(self) -> bool:
        """返回True表示应该退出（>=10次连续错误）。"""
        return self.consecutive_errors >= MAX_CONSECUTIVE_NETWORK_ERRORS


net_error_counter = NetworkErrorCounter()


def retry_with_backoff(
    max_retries: int = MAX_RETRIES,
    base_delay: float = RATE_LIMIT_BACKOFF_BASE,
    max_delay: float = RATE_LIMIT_BACKOFF_MAX,
):
    """
    装饰器：自动重试API调用，支持指数退避和网络中断检测。
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    result = func(*args, **kwargs)

                    if isinstance(result, requests.Response):
                        if result.status_code == 429:
                            net_error_counter.record_error()
                            if attempt < max_retries:
                                delay = min(
                                    base_delay * (2 ** attempt),
                                    max_delay
                                )
                                logger.debug(
                                    f"HTTP 429, 第{attempt+1}次重试, "
                                    f"等待{delay:.1f}s"
                                )
                                time.sleep(delay)
                                continue
                        if result.status_code >= 500:
                            net_error_counter.record_error()
                            if attempt < max_retries:
                                delay = min(base_delay * (2 ** attempt), max_delay)
                                time.sleep(delay)
                                continue
                        net_error_counter.record_success()
                        return result

                    if isinstance(result, tuple) and len(result) == 2:
                        success, data = result
                        if success:
                            net_error_counter.record_success()
                            return data
                        elif attempt < max_retries:
                            delay = min(base_delay * (2 ** attempt), max_delay)
                            time.sleep(delay)
                            continue
                        else:
                            net_error_counter.record_error()
                            return None

                    net_error_counter.record_success()
                    return result

                except (requests.exceptions.Timeout,
                        requests.exceptions.ConnectionError,
                        requests.exceptions.HTTPError) as e:
                    net_error_counter.record_error()
                    last_exception = e
                    if attempt < max_retries:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        logger.debug(f"网络错误 [{type(e).__name__}], "
                                     f"第{attempt+1}次重试, 等待{delay:.1f}s")
                        time.sleep(delay)
                    else:
                        logger.error(f"请求最终失败 [{type(e).__name__}]: {e}")

                except Exception as e:
                    logger.error(f"非网络错误: {type(e).__name__}: {e}")
                    raise

            logger.debug(f"请求经{max_retries}次重试后仍失败")
            return None

        return wrapper
    return decorator


# ============================================================
# 辅助函数
# ============================================================

def parse_timestamp(ts: int | float | None) -> str | None:
    """Unix时间戳 → ISO格式字符串（北京时间 UTC+8）。"""
    if ts is None or ts == 0:
        return None
    try:
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, OSError):
        return None


def is_ad_title(title: str) -> bool:
    """检查文本是否匹配广告/营销关键词。"""
    for pattern in AD_PATTERNS:
        if re.search(pattern, title):
            return True
    return False


def is_in_time_range(pubdate_ts: int | None) -> bool:
    """
    检查发布时间是否在 2020-01-01 ~ 2025-12-31 (UTC) 之间。
    知乎 API 返回 Unix UTC 时间戳。
    """
    if pubdate_ts is None or pubdate_ts == 0:
        return True  # 无法判断则不筛
    # 2020-01-01 00:00:00 UTC = 1577836800
    # 2025-12-31 23:59:59 UTC = 1767225599
    AFTER_TS = 1577836800
    BEFORE_TS = 1767225599
    return AFTER_TS <= pubdate_ts <= BEFORE_TS


def strip_html(text: str) -> str:
    """去除HTML标签和其他标记。"""
    if not text:
        return ""
    # 去除HTML标签
    text = re.sub(r"<[^>]+>", "", text)
    # 去除知乎特殊标记
    text = re.sub(r"\[.*?\]", "", text)
    # 去除多余空白
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def safe_get(d: dict, *keys, default=None) -> Any:
    """安全获取嵌套字典的值。"""
    for key in keys:
        if isinstance(d, dict):
            d = d.get(key, default)
        else:
            return default
    return d if d is not None else default


def format_number(n: int) -> str:
    """人性化数字显示：50000 → 5.0w, 1000 → 1.0k。"""
    if n is None:
        return "?"
    if n >= 10000:
        return f"{n/10000:.1f}w"
    elif n >= 1000:
        return f"{n/1000:.1f}k"
    return str(n)


# ============================================================
# 知乎 ZSE-96 签名（知乎 API 反爬机制）
# ============================================================

# x-zse-96 编码表（版本 3.0.40，社区逆向工程）
# 每个条目为3字符，共256条
_ZSE_ENC_TABLE_RAW = (
    "A4UsDdR0DJ4NR0M0z4L3l4u09uEl42Z42J4mQkt4uT4uW4sS4kU"
    "4uX4uY4s24u14wW4zE4K0D4HY4uZ4sR4uz4u34s74u54u64s84u9"
    "4sW4u04uE4uF4uG4uH4uI4uJ4uN4uO4uP4uQ4uV4u44ug4uw4u"
    "A4uB4uC4sH4sI4sJ4sK4uo4up4uq4ur4us4ut4uu4uv4sL4sM4s"
    "N4sO4sP4sQ4sT4rY4rv4rw4rx4ry4rz4rA4rB4rC4sU4sV4sX4"
    "sY4sZ4s04s14s34s44s54s64s94sa4sb4sc4sd4se4sf4sg4sh4"
    "si4sj4sk4sl4sm4sn4so4sp4sq4sr4ss4st4su4sv4sw4sx4sy"
    "4sz4sA4sB4sC4sD4sE4sF4sG4rj4rk4rl4rm4rn4ro4rp4rq4r"
    "r4rs4rt4ru4rR4rS4rT4rU4rV4rW4rX4rf4rg4rh4ri4sS4rZ4"
    "r04r14r24r34r44r54r64r74r84r94ra4rb4rc4rd4re4u84ta4"
    "tb4tc4td4te4tf4tg4th4ti4tj4tk4tl4tm4tn4to4tp4tq4tr"
    "4ts4tt4tu4tv4tw4u14u24u64u74rD4rE4rF4rG4rH4rI4rJ4r"
    "K4rL4rM4rN4rO4rP4rQ4tx4ty4tz4tA4tB4tC4tD4tE4tF4tG"
    "4tH4tI4tJ4tK4tL4tM4tN4tO4tP4tQ4tR4tS4tT4tU4tV4tW4"
    "tX4tY4tZ4t04t14t24t34t44t54t64t74rx4ry4rz4rA4rB4rC"
)
# 将原始表拆分为3字符组
_ZSE_ENC_TABLE = [
    _ZSE_ENC_TABLE_RAW[i:i+3]
    for i in range(0, len(_ZSE_ENC_TABLE_RAW), 3)
]


def _zse_encode(text: str) -> str:
    """将字符串编码为 x-zse-96 所用格式（使用3字符组编码表）。"""
    result = []
    table_len = len(_ZSE_ENC_TABLE)
    for ch in text:
        idx = ord(ch) % table_len
        result.append(_ZSE_ENC_TABLE[idx])
    return "".join(result)


class ZhihuZSE:
    """
    知乎 x-zse-96 签名生成器。

    为 API 请求（尤其是 /api/v4/search_v3）生成必需的 x-zse-96 header。
    基于社区逆向工程成果实现，版本 3.0.40。

    算法概要:
    1. 构建源字符串: URL path + query
    2. MD5 源字符串 → hex digest
    3. 拼接: md5_hex + d_c0 cookie
    4. 使用3字符组编码表逐个字符替换
    5. 输出: "2.0_{encoded}"

    用法:
        zse = ZhihuZSE(session)
        zse_96 = zse.generate("https://www.zhihu.com/api/v4/search_v3", params)
        headers["x-zse-96"] = zse_96
    """

    def __init__(self, session: Optional[requests.Session] = None):
        self._session = session or requests.Session()
        self._d_c0: str = ""

    def _get_d_c0(self) -> str:
        """从 session cookies 获取或生成 d_c0。"""
        if self._d_c0:
            return self._d_c0
        for cookie in self._session.cookies:
            if cookie.name == "d_c0":
                self._d_c0 = cookie.value
                return self._d_c0
        # 生成新的 d_c0
        self._d_c0 = str(uuid.uuid4()).replace("-", "")[:32]
        return self._d_c0

    def generate(self, url: str, params: dict | None = None) -> str:
        """
        生成 x-zse-96 header 值。

        Args:
            url: 完整 API URL
            params: 查询参数字典（可选）

        Returns:
            x-zse-96 字符串，格式: "2.0_{signature}"
        """
        from urllib.parse import urlparse

        parsed = urlparse(url)
        source_path = parsed.path

        # 构建 source 字符串：path + ? + sorted_query
        if params:
            sorted_params = sorted(params.items(), key=lambda x: x[0])
            query_str = urllib.parse.urlencode(sorted_params)
            source = source_path + "?" + query_str
        elif parsed.query:
            # 如果URL已包含query但params为空，对query排序
            query_pairs = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
            sorted_pairs = sorted(query_pairs, key=lambda x: x[0])
            query_str = urllib.parse.urlencode(sorted_pairs)
            source = source_path + "?" + query_str
        else:
            source = source_path

        # Step 1: MD5 source → hex digest (32 chars)
        md5_digest = hashlib.md5(source.encode("utf-8")).hexdigest()

        # Step 2: 拼接 d_c0
        combined = md5_digest + self._get_d_c0()

        # Step 3: 使用 ZSE 编码表编码（每个字符映射到3字符组）
        encoded = _zse_encode(combined)

        return f"2.0_{encoded}"

    def generate_from_path(self, path: str) -> str:
        """从纯路径生成 x-zse-96（用于无查询参数的请求）。"""
        md5_digest = hashlib.md5(path.encode("utf-8")).hexdigest()
        combined = md5_digest + self._get_d_c0()
        encoded = _zse_encode(combined)
        return f"2.0_{encoded}"
