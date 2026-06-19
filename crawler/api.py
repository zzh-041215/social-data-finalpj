"""
B站API封装：搜索、视频详情、评论、用户信息 四个端点
- WBI 签名（绕过 -1200 降级过滤）
- requests.Session 复用TCP连接 + buvid3 Cookie
- 搜索结果预筛选（减少不必要的详情API调用）
- 自适应限速 + 429/412退避
"""

import time
import uuid
from typing import Optional

import requests

from crawler.config import (
    API_SEARCH, API_VIDEO_INFO, API_COMMENTS, API_USER_INFO,
    HEADERS, API_HEADERS, SEARCH_PAGE_SIZE, SEARCH_ORDERS,
    MAX_PAGES_PER_KEYWORD, MAX_COMMENTS_PER_VIDEO,
    MIN_VIEW_COUNT, MIN_DURATION_SEC,
)
from crawler.utils import (
    logger, AdaptiveRateLimiter, BilibiliWBI,
    net_error_counter,
    parse_duration, is_ad_title, is_in_time_range,
    safe_get, strip_html,
)


class BilibiliAPI:
    """封装B站公开API，管理Session、WBI签名和限速。"""

    def __init__(self, rate_limiter: AdaptiveRateLimiter):
        self.limiter = rate_limiter
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.session.timeout = 30

        # 用户信息缓存：mid -> {name, follower, ...}
        self._user_cache: dict[int, dict] = {}

        # Step 1: 先访问B站首页获取必要cookies（buvid3, b_nut）
        self._init_cookies()

        # Step 2: 初始化 WBI 签名器（使用已有cookies的session）
        self.wbi = BilibiliWBI(self.session)
        try:
            self.wbi.get_wbi_keys()
            logger.debug("WBI密钥初始化成功")
        except Exception as e:
            logger.warning(f"WBI密钥初始化失败（将继续尝试）: {e}")

    def _init_cookies(self) -> None:
        """访问B站首页获取buvid3等必需cookies。"""
        try:
            resp = self.session.get(
                "https://www.bilibili.com/",
                timeout=20,
            )
            logger.debug(
                f"首页cookies: {dict(self.session.cookies.get_dict())}"
            )
        except Exception as e:
            logger.warning(f"获取首页cookies失败: {e}")

    @staticmethod
    def _generate_buvid3() -> str:
        """生成随机 buvid3 cookie。"""
        prefix = str(uuid.uuid4()).replace("-", "")
        return f"{prefix[:8]}-{prefix[8:12]}-{prefix[12:16]}-{prefix[16:20]}-{prefix[20:32]}infoc"

    # ═══════════════════════════════════════════════════
    # 通用请求方法（带WBI签名）
    # ═══════════════════════════════════════════════════

    def _signed_get(self, url: str, params: dict) -> Optional[requests.Response]:
        """
        发送带WBI签名的GET请求。
        返回 Response 对象或 None（失败时）。
        """
        self.limiter.wait()

        # 临时切换Accept头为JSON（保留其他headers包括Sec-Fetch-*）
        old_accept = self.session.headers.get("Accept", "")
        self.session.headers["Accept"] = "application/json, text/plain, */*"

        try:
            signed_params = self.wbi.enc_wbi(params)
        except Exception as e:
            logger.debug(f"WBI签名失败: {e}")
            signed_params = params

        try:
            resp = self.session.get(url, params=signed_params)
        except requests.RequestException as e:
            net_error_counter.record_error()
            logger.debug(f"请求失败 [{url}]: {type(e).__name__}")
            resp = None
        finally:
            self.session.headers["Accept"] = old_accept

        return resp

    def _check_response(self, resp: requests.Response, context: str = "") -> bool:
        """
        检查API响应是否成功。
        处理 429、-412（签名失效）、-1200（降级过滤）等错误。
        返回 True 表示成功。
        """
        if resp is None:
            return False

        if resp.status_code == 429:
            self.limiter.report_429()
            logger.debug(f"HTTP 429 [{context}]")
            return False

        if resp.status_code != 200:
            net_error_counter.record_error()
            if resp.status_code == 404:
                logger.debug(f"HTTP 404 [{context}]")
            else:
                logger.debug(f"HTTP {resp.status_code} [{context}]")
            return False

        try:
            body = resp.json()
        except ValueError:
            net_error_counter.record_error()
            return False

        code = body.get("code", -1)

        # -412: 被拦截（签名无效或cookie问题）
        if code == -412:
            self.limiter.report_429()
            logger.debug(f"API -412 被拦截 [{context}]: {body.get('message', '')}")
            # 尝试刷新WBI密钥
            try:
                self.wbi.get_wbi_keys()
            except Exception:
                pass
            return False

        # -1200: 降级过滤的请求
        if code == -1200:
            self.limiter.report_429()
            logger.warning(f"API -1200 降级过滤 [{context}]，可能需要刷新WBI密钥或Cookie")
            try:
                self.wbi.get_wbi_keys()
            except Exception:
                pass
            return False

        # -799: 频率限制
        if code == -799:
            self.limiter.report_429()
            logger.debug(f"API -799 频率限制 [{context}]")
            return False

        if code != 0:
            # 62002: 视频不可见, -404: 不存在
            if code in (62002, -404):
                logger.debug(f"资源不可用 [{context}] code={code}")
            else:
                logger.debug(f"API code={code} [{context}]: "
                             f"{body.get('message', '')}")
            return False

        self.limiter.report_success()
        net_error_counter.record_success()
        return True

    # ═══════════════════════════════════════════════════
    # 搜索API
    # ═══════════════════════════════════════════════════

    def search_videos(
        self, keyword: str, page: int, page_size: int = SEARCH_PAGE_SIZE,
        order: str = "pubdate",
    ) -> Optional[dict]:
        """
        搜索视频，返回API原始响应（data字段）。
        order: pubdate(最新), click(最多播放), dm(最多弹幕), stow(最多收藏)
        """
        params = {
            "search_type": "video",
            "keyword": keyword,
            "page": page,
            "page_size": page_size,
            "order": order,
        }

        resp = self._signed_get(API_SEARCH, params)
        if not self._check_response(resp, f"搜索 {keyword} p{page} order={order}"):
            return None

        return resp.json().get("data")

    def search_and_filter(
        self, keyword: str, page: int, order: str = "pubdate"
    ) -> list[dict]:
        """
        搜索一页并做预筛选，返回通过初筛的视频摘要列表。
        预筛选：时间范围、广告标题、时长（播放量过滤移到详情阶段）
        """
        data = self.search_videos(keyword, page, order=order)
        if data is None:
            return []

        results = data.get("result", [])
        if not results:
            return []

        filtered = []
        for item in results:
            if item.get("type") != "video":
                continue

            bvid = item.get("bvid", "")
            if not bvid:
                continue

            title = strip_html(item.get("title", ""))
            play = item.get("play", 0)
            pubdate_ts = item.get("pubdate", 0)
            duration_str = item.get("duration", "0:00")

            # 注意：搜索API返回的play字段可能不准确（尤其对新视频显示0）
            # 播放量过滤移到获取视频详情之后（使用准确的stat.view）
            if not is_in_time_range(pubdate_ts):
                continue
            if is_ad_title(title):
                continue

            duration_sec = parse_duration(duration_str)
            if duration_sec is not None and duration_sec < MIN_DURATION_SEC:
                continue

            filtered.append({
                "bvid": bvid,
                "aid": item.get("id", 0),
                "title": title,
                "description": strip_html(item.get("description", "")),
                "play": play,
                "pubdate": pubdate_ts,
                "author": item.get("author", ""),
                "mid": item.get("mid", 0),
                "tag": item.get("tag", ""),
                "duration_str": duration_str,
                "duration_sec": duration_sec or 0,
                "video_review": item.get("video_review", 0),
            })

        return filtered

    # ═══════════════════════════════════════════════════
    # 视频详情API
    # ═══════════════════════════════════════════════════

    def get_video_info(self, bvid: str) -> Optional[dict]:
        """获取视频详细信息。"""
        params = {"bvid": bvid}
        resp = self._signed_get(API_VIDEO_INFO, params)

        if not self._check_response(resp, f"视频 {bvid}"):
            return None

        vdata = resp.json().get("data", {})
        if not vdata:
            return None

        stat = vdata.get("stat", {})
        owner = vdata.get("owner", {})

        return {
            "bvid": vdata.get("bvid", bvid),
            "aid": vdata.get("aid", 0),
            "title": vdata.get("title", ""),
            "description": strip_html(vdata.get("desc", "")),
            "publish_time": vdata.get("pubdate", 0),
            "duration_sec": vdata.get("duration", 0),
            "tags": "",
            "tag_list": "",
            "category_id": vdata.get("tid", 0),
            "category_name": vdata.get("tname", ""),
            "uploader_name": owner.get("name", ""),
            "uploader_mid": owner.get("mid", 0),
            "uploader_follower_count": 0,
            "view_count": stat.get("view", 0),
            "like_count": stat.get("like", 0),
            "coin_count": stat.get("coin", 0),
            "favorite_count": stat.get("favorite", 0),
            "share_count": stat.get("share", 0),
            "comment_count": stat.get("reply", 0),
            "danmaku_count": stat.get("danmaku", 0),
        }

    # ═══════════════════════════════════════════════════
    # 评论API
    # ═══════════════════════════════════════════════════

    def get_video_comments(
        self, oid: int, mode: int = 3
    ) -> list[dict]:
        """
        获取视频热门评论（使用新版 reply/main API）。
        oid=aid, type=1（视频）, mode=3（热度排序）/ mode=2（时间排序）。
        返回评论列表。
        """
        params = {
            "oid": oid,
            "type": 1,
            "mode": mode,
            "next": 0,
        }

        resp = self._signed_get(API_COMMENTS, params)
        if not self._check_response(resp, f"评论 oid={oid}"):
            return []

        data = resp.json().get("data", {})
        if not data:
            return []

        # 合并 top 评论和普通热评
        replies = data.get("replies", [])
        top_replies = data.get("top_replies", [])

        all_replies = list(top_replies) + list(replies)

        comments = []
        for r in all_replies:
            content = r.get("content", {})
            text = content.get("message", "")
            text = strip_html(text)

            if not text or len(text) < 2:
                continue

            comments.append({
                "comment_id": r.get("rpid", 0),
                "oid": oid,
                "text": text,
                "text_length": len(text),
                "like_count": r.get("like", 0),
                "reply_count": r.get("rcount", r.get("count", 0)),
                "publish_time": r.get("ctime", 0),
                "is_hot": 1 if r.get("attr", 0) & 2 else 0,  # attr bit 1 = hot
                "parent_id": r.get("parent", 0),
            })

        return comments

    # ═══════════════════════════════════════════════════
    # 用户信息API
    # ═══════════════════════════════════════════════════

    def get_user_info(self, mid: int) -> Optional[dict]:
        """获取UP主粉丝数等信息。结果缓存。"""
        if mid in self._user_cache:
            return self._user_cache[mid]

        params = {"mid": mid}
        resp = self._signed_get(API_USER_INFO, params)

        if not self._check_response(resp, f"用户 mid={mid}"):
            return None

        udata = resp.json().get("data", {})
        info = {
            "mid": udata.get("mid", mid),
            "name": udata.get("name", ""),
            "follower": udata.get("follower", 0),
        }
        self._user_cache[mid] = info
        return info

    def get_cached_user_follower(self, mid: int) -> int:
        """获取已缓存的粉丝数（不发起请求）。"""
        if mid in self._user_cache:
            return self._user_cache[mid].get("follower", 0)
        return 0
