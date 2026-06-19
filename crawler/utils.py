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
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Any, Optional

import requests

from crawler.config import (
    DELAY_MIN, DELAY_MAX,
    RATE_LIMIT_BACKOFF_BASE, RATE_LIMIT_BACKOFF_MAX,
    MAX_RETRIES, NETWORK_PAUSE_SEC, MAX_CONSECUTIVE_NETWORK_ERRORS,
    LOG_DIR, AD_PATTERNS,
    MIN_VIEW_COUNT, MIN_DURATION_SEC,
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
    自适应限速器：正常时使用短延迟（0.3-0.8s），收到429后指数退避。

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

    def wait(self) -> float:
        """等待到可以发下一个请求，返回实际等待秒数。"""
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
        """收到429后调用，进入指数退避。"""
        self._consecutive_429s += 1
        backoff = min(
            self._current_backoff * (2 ** (self._consecutive_429s - 1)),
            RATE_LIMIT_BACKOFF_MAX
        )
        self._backoff_until = time.time() + backoff
        logger.warning(f"收到429限速，退避 {backoff:.1f}s "
                       f"(连续第{self._consecutive_429s}次)")

    def report_success(self) -> None:
        """请求成功后调用，逐步恢复正常延迟。"""
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

    对被装饰函数的要求：
    - 返回 (success: bool, data: Any) 或直接返回 Response
    - 抛出 requests.RequestException 会自动重试
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    result = func(*args, **kwargs)

                    # 如果返回的是 requests.Response
                    if isinstance(result, requests.Response):
                        if result.status_code == 429 or (
                            hasattr(result, 'json') and callable(result.json)
                        ):
                            try:
                                body = result.json()
                                if body.get("code") == -412:
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
                            except Exception:
                                pass
                        if result.status_code >= 500:
                            net_error_counter.record_error()
                            if attempt < max_retries:
                                delay = min(base_delay * (2 ** attempt), max_delay)
                                time.sleep(delay)
                                continue
                        net_error_counter.record_success()
                        return result

                    # 如果返回的是 tuple (success, data)
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
                    # 非网络错误不重试
                    logger.error(f"非网络错误: {type(e).__name__}: {e}")
                    raise

            # 所有重试都失败了
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


def parse_duration(duration_str: str | None) -> int | None:
    """
    将B站时长格式转为秒数。
    支持格式："12:34" → 754, "01:02:03" → 3723
    """
    if not duration_str or duration_str == "--":
        return None
    parts = duration_str.strip().split(":")
    try:
        if len(parts) == 2:   # mm:ss
            return int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 3: # hh:mm:ss
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    except (ValueError, IndexError):
        pass
    return None


def is_ad_title(title: str) -> bool:
    """检查标题是否匹配广告/营销关键词。"""
    for pattern in AD_PATTERNS:
        if re.search(pattern, title):
            return True
    return False


def is_in_time_range(pubdate_ts: int | None) -> bool:
    """
    检查发布时间是否在 2020-01-01 ~ 2025-12-31 (UTC) 之间。
    B站API返回Unix UTC时间戳。
    """
    if pubdate_ts is None or pubdate_ts == 0:
        return True  # 无法判断则不筛
    # 使用显式UTC时间戳
    # 2020-01-01 00:00:00 UTC = 1577836800
    # 2025-12-31 23:59:59 UTC = 1767225599
    AFTER_TS = 1577836800
    BEFORE_TS = 1767225599
    return AFTER_TS <= pubdate_ts <= BEFORE_TS


def strip_html(text: str) -> str:
    """去除HTML标签和B站emoji标记。"""
    if not text:
        return ""
    # 去除HTML标签
    text = re.sub(r"<[^>]+>", "", text)
    # 去除B站表情标记 [xxx]
    text = re.sub(r"\[.*?\]", "", text)
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
    """人性化数字显示：50000 → 5.0w, 1000000 → 100w。"""
    if n is None:
        return "?"
    if n >= 10000:
        return f"{n/10000:.1f}w"
    elif n >= 1000:
        return f"{n/1000:.1f}k"
    return str(n)


# ============================================================
# WBI 签名（B站 API 反爬机制，自2023年起强制要求）
# ============================================================

class BilibiliWBI:
    """
    B站 WBI 签名生成器。

    为API请求参数计算 w_rid 和 wts，绕过 -1200 降级过滤。
    密钥每日轮换，首次调用时从 /x/web-interface/nav 获取。
    """

    # 固定64位索引排列表（自2023年引入后至今未变）
    MIXIN_KEY_ENC_TAB = [
        46, 47, 18, 2, 53, 8, 23, 32, 15, 50, 10, 31, 58, 3, 45, 35,
        27, 43, 5, 49, 33, 9, 42, 19, 29, 28, 14, 39, 12, 38, 41, 13,
        37, 48, 7, 16, 24, 55, 40, 61, 26, 17, 0, 1, 60, 51, 30, 4,
        22, 25, 54, 21, 56, 59, 6, 63, 57, 62, 11, 36, 20, 34, 44, 52,
    ]

    def __init__(self, session: Optional[requests.Session] = None):
        self.img_key: Optional[str] = None
        self.sub_key: Optional[str] = None
        self.mixin_key: Optional[str] = None
        self._keys_ts: float = 0.0  # 密钥获取时间
        self._session = session or requests.Session()

    def get_wbi_keys(self) -> tuple[str, str]:
        """
        从 /x/web-interface/nav 获取 img_key 和 sub_key。
        密钥每日轮换，缓存12小时。
        """
        now = time.time()
        # 缓存12小时（43200秒）
        if (self.mixin_key is not None
                and (now - self._keys_ts) < 43200):
            return self.img_key, self.sub_key

        try:
            resp = self._session.get(
                "https://api.bilibili.com/x/web-interface/nav",
                timeout=15,
            )
            data = resp.json()

            # code=-101 表示未登录，但仍返回wbi_img数据
            # 任何code都可以包含wbi_img，先检查数据是否存在
            wbi_img = safe_get(data, "data", "wbi_img")
            if not wbi_img:
                raise RuntimeError(
                    f"获取WBI密钥失败: code={data.get('code')}, "
                    f"message={data.get('message', '')}"
                )

            logger.debug(f"WBI密钥获取成功 (nav code={data.get('code')})")
            self.img_key = wbi_img["img_url"].split("/")[-1].split(".")[0]
            self.sub_key = wbi_img["sub_url"].split("/")[-1].split(".")[0]
            self._keys_ts = now

            # 生成 mixin key
            raw_key = self.img_key + self.sub_key
            self.mixin_key = "".join([
                raw_key[i]
                for i in self.MIXIN_KEY_ENC_TAB
                if i < len(raw_key)
            ])[:32]

            logger.debug(f"WBI密钥已更新: mixin_key={self.mixin_key[:8]}...")
            return self.img_key, self.sub_key

        except Exception as e:
            logger.error(f"获取WBI密钥异常: {e}")
            raise

    def enc_wbi(self, params: dict) -> dict:
        """
        为请求参数计算 WBI 签名，返回附加了 w_rid 和 wts 的参数字典。

        Args:
            params: 原始请求参数字典

        Returns:
            附加了 w_rid, wts 的新参数字典
        """
        if not self.mixin_key:
            self.get_wbi_keys()

        # 1. 添加时间戳
        signed_params = params.copy()
        wts = int(time.time())
        signed_params["wts"] = wts

        # 2. 按 key 字母序排序
        sorted_params = dict(sorted(signed_params.items(), key=lambda x: x[0]))

        # 3. 过滤特殊字符 !'()*
        filtered = {}
        for k, v in sorted_params.items():
            val_str = str(v)
            for ch in "!'()*":
                val_str = val_str.replace(ch, "")
            filtered[k] = val_str

        # 4. URL编码
        query_string = urllib.parse.urlencode(filtered)

        # 5. 拼接 mixin_key 并计算 MD5
        sign_str = query_string + self.mixin_key
        w_rid = hashlib.md5(sign_str.encode("utf-8")).hexdigest()

        # 6. 追加签名
        result = params.copy()
        result["w_rid"] = w_rid
        result["wts"] = str(wts)
        return result
