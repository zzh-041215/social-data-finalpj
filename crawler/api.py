"""
知乎 API 封装：搜索、回答详情、评论、用户信息 四个端点
- requests.Session 复用TCP连接 + 自动cookie管理
- 可选 x-zse-96 签名（用于搜索等严格端点）
- 搜索结果预筛选（减少不必要的详情API调用）
- 自适应限速 + 429/403退避
"""

import time
import uuid
import threading
from typing import Optional

import requests

from crawler.config import (
    API_SEARCH, API_ANSWER, API_QUESTION, API_COMMENTS, API_COMMENTS_V5,
    API_USER, ANSWER_INCLUDE,
    HEADERS, API_HEADERS,
    SEARCH_LIMIT, SEARCH_TYPE,
    MAX_OFFSET_PER_KEYWORD, MAX_COMMENTS_PER_ANSWER,
    MIN_ANSWER_LENGTH, COOKIES_FILE,
)
from crawler.utils import (
    logger, AdaptiveRateLimiter, ZhihuZSE,
    net_error_counter,
    is_ad_title, is_in_time_range,
    safe_get, strip_html,
)


class ZhihuAPI:
    """封装知乎公开API，管理Session、ZSE签名和限速。"""

    def __init__(self, rate_limiter: AdaptiveRateLimiter):
        self.limiter = rate_limiter
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self._session_lock = threading.Lock()  # requests.Session不是线程安全的

        # 用户信息缓存：url_token -> {name, follower, ...}
        self._user_cache: dict[str, dict] = {}

        # 是否持有登录态 cookie（z_c0 是知乎登录凭证）
        self._has_auth = False

        # Step 1: 加载 cookie（优先 cookies.txt，否则访问首页获取游客cookie）
        self._init_cookies()

        # Step 2: 初始化 ZSE 签名器（仅在无登录态时需要）
        self.zse = ZhihuZSE(self.session)
        logger.debug("知乎API模块初始化完成")

    def _init_cookies(self) -> None:
        """初始化 session cookies：优先从 cookies.txt 加载，否则访问首页获取。"""
        # Step 1: 尝试从文件加载登录cookie
        cookies_loaded = self._load_cookies_from_file()

        if cookies_loaded:
            logger.info(f"已从 cookies.txt 加载 {len(cookies_loaded)} 个cookie")
        else:
            # Step 2: 回退到访问首页获取游客cookie
            logger.info("未找到 cookies.txt，访问首页获取游客cookie...")
            try:
                resp = self.session.get(
                    "https://www.zhihu.com/",
                    timeout=20,
                )
                cookies = dict(self.session.cookies.get_dict())
                logger.debug(f"知乎首页cookies: {cookies}")
                if "d_c0" not in cookies:
                    d_c0 = str(uuid.uuid4()).replace("-", "")[:32]
                    self.session.cookies.set("d_c0", d_c0, domain=".zhihu.com")
                    logger.debug(f"手动设置 d_c0: {d_c0}")
            except Exception as e:
                logger.warning(f"获取知乎首页cookies失败: {e}")
                d_c0 = str(uuid.uuid4()).replace("-", "")[:32]
                self.session.cookies.set("d_c0", d_c0, domain=".zhihu.com")

    def _load_cookies_from_file(self) -> dict[str, str]:
        """
        从 cookies.txt 加载登录cookie到 session。

        文件格式（每行一个 cookie）：
            cookie_name=cookie_value

        返回加载的 cookie 字典，若文件不存在或为空则返回 {}。
        """
        import os as _os

        if not _os.path.exists(COOKIES_FILE):
            return {}

        loaded = {}
        try:
            with open(COOKIES_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    # 支持 name=value 格式
                    if "=" in line:
                        key, _, value = line.partition("=")
                        key = key.strip()
                        value = value.strip()
                        if key and value:
                            self.session.cookies.set(key, value, domain=".zhihu.com")
                            loaded[key] = value[:30] + "..."  # 只记录部分值用于日志
        except Exception as e:
            logger.warning(f"读取 cookies.txt 失败: {e}")
            return {}

        if loaded:
            logger.info(f"已加载cookie: {list(loaded.keys())}")
            # 检测是否有登录态（z_c0 是知乎的核心登录凭证）
            if "z_c0" in loaded:
                self._has_auth = True
                logger.info("检测到登录cookie (z_c0)，将跳过 x-zse-96 签名")
        return loaded

    # ═══════════════════════════════════════════════════
    # 通用请求方法
    # ═══════════════════════════════════════════════════

    def _api_get(
        self, url: str, params: dict | None = None,
        needs_zse: bool = False,
    ) -> Optional[requests.Response]:
        """
        发送 API GET 请求。

        Args:
            url: API URL
            params: 查询参数字典
            needs_zse: 是否需要添加 x-zse-96 header（搜索端点建议启用）

        Returns:
            Response 对象或 None（失败时）
        """
        self.limiter.wait()

        if params is None:
            params = {}

        # 设置 API 专用 headers
        for key, value in API_HEADERS.items():
            self.session.headers[key] = value

        # 添加 x-zse-96 如果需要
        if needs_zse:
            try:
                zse_value = self.zse.generate(url, params)
                self.session.headers["x-zse-96"] = zse_value
            except Exception as e:
                logger.debug(f"ZSE签名生成失败: {e}")

        try:
            with self._session_lock:
                resp = self.session.get(url, params=params, timeout=30)
        except requests.RequestException as e:
            net_error_counter.record_error()
            logger.debug(f"请求失败 [{url}]: {type(e).__name__}")
            return None

        return resp

    def _check_response(
        self, resp: Optional[requests.Response], context: str = ""
    ) -> bool:
        """
        检查 API 响应是否成功。
        处理 429（限速）、403（被拦截）、5xx 等错误。
        返回 True 表示成功。
        """
        if resp is None:
            return False

        if resp.status_code == 429:
            self.limiter.report_429()
            logger.debug(f"HTTP 429 [{context}]")
            return False

        if resp.status_code == 403:
            self.limiter.report_429()
            logger.warning(
                f"HTTP 403 被拦截 [{context}]，"
                f"可能需要更新 cookie 或 x-zse-96 签名"
            )
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
            logger.debug(f"响应非JSON [{context}]")
            return False

        # 知乎 API 错误检查
        error = body.get("error")
        if error:
            error_code = error.get("code", 0)
            error_msg = error.get("message", "")
            if error_code in (403001, 403002):  # 频率限制
                self.limiter.report_429()
                logger.debug(f"知乎频率限制 [{context}]: {error_msg}")
                return False
            if error_code != 0:
                logger.debug(f"知乎 API 错误 [{context}] code={error_code}: {error_msg}")
                return False

        self.limiter.report_success()
        net_error_counter.record_success()
        return True

    # ═══════════════════════════════════════════════════
    # 搜索 API
    # ═══════════════════════════════════════════════════

    def search_content(
        self, keyword: str, offset: int = 0, limit: int = SEARCH_LIMIT
    ) -> Optional[dict]:
        """
        搜索知乎内容，返回 API 原始响应。

        Args:
            keyword: 搜索关键词
            offset: 分页偏移量（0, 20, 40, ...）
            limit: 每页数量（默认20）

        Returns:
            API 响应的 data 字段，或 None
        """
        params = {
            "q": keyword,
            "t": SEARCH_TYPE,
            "lc_idx": "0",
            "offset": str(offset),
            "limit": str(limit),
        }

        # 有登录cookie时不需要 x-zse-96，否则可能因编码表版本不匹配导致403
        needs_zse = not self._has_auth

        resp = self._api_get(
            API_SEARCH, params,
            needs_zse=needs_zse,
        )
        if not self._check_response(resp, f"搜索 {keyword} offset={offset}"):
            return None

        return resp.json()

    def search_and_filter(
        self, keyword: str, offset: int = 0
    ) -> list[dict]:
        """
        搜索一页并做预筛选，返回通过初筛的回答摘要列表。
        预筛选：时间范围、广告标题、内容长度。
        """
        data = self.search_content(keyword, offset)
        if data is None:
            return []

        results = data.get("data", [])
        if not results:
            return []

        filtered = []
        for item in results:
            obj = item.get("object", {})
            if not obj:
                continue

            # 只收集 answer 类型
            obj_type = obj.get("type", "")
            if obj_type != "answer":
                continue

            answer_id = obj.get("id", 0)
            if not answer_id:
                continue

            # 时间筛选
            created_time = obj.get("created_time", 0)
            if not is_in_time_range(created_time):
                continue

            # 内容长度筛选
            excerpt = strip_html(obj.get("excerpt", ""))
            if len(excerpt) < MIN_ANSWER_LENGTH:
                continue

            question = obj.get("question", {})
            author = obj.get("author", {})

            filtered.append({
                "answer_id": answer_id,
                "question_id": question.get("id", 0),
                "question_title": question.get("title", ""),
                "excerpt": excerpt,
                "voteup_count": obj.get("voteup_count", 0),
                "comment_count": obj.get("comment_count", 0),
                "created_time": created_time,
                "author_name": author.get("name", ""),
                "author_url_token": author.get("url_token", ""),
            })

        return filtered

    # ═══════════════════════════════════════════════════
    # 回答详情 API
    # ═══════════════════════════════════════════════════

    def get_answer_detail(self, answer_id: int) -> Optional[dict]:
        """获取回答详细信息。

        注意：必须传 include 参数，否则知乎不下发 content/voteup_count 等字段。
        """
        url = f"{API_ANSWER}/{answer_id}"
        resp = self._api_get(url, params={"include": ANSWER_INCLUDE}, needs_zse=False)

        if not self._check_response(resp, f"回答 {answer_id}"):
            return None

        adata = resp.json()
        question = adata.get("question", {})
        author = adata.get("author", {})

        return {
            "answer_id": adata.get("id", answer_id),
            "question_id": question.get("id", 0),
            "question_title": question.get("title", ""),
            "content": strip_html(adata.get("content", "")),
            "excerpt": adata.get("excerpt", ""),
            "publish_time": adata.get("created_time", 0),
            "created_time": adata.get("created_time", 0),
            "updated_time": adata.get("updated_time", 0),
            "author_name": author.get("name", ""),
            "author_url_token": author.get("url_token", ""),
            "author_headline": author.get("headline", ""),
            "author_follower_count": author.get("follower_count", 0),
            "voteup_count": adata.get("voteup_count", 0),
            "comment_count": adata.get("comment_count", 0),
            "thanks_count": adata.get("thanks_count", 0),
        }

    # ═══════════════════════════════════════════════════
    # 问题下的回答列表
    # ═══════════════════════════════════════════════════

    def get_question_answers(
        self, qid: int, offset: int = 0, limit: int = 20,
        sort_by: str = "default",
    ) -> list[dict]:
        """
        获取问题下的回答列表。

        Args:
            qid: 问题ID
            offset: 偏移量
            limit: 每页数量
            sort_by: 排序方式 (default / created_time)

        Returns:
            回答摘要列表
        """
        url = f"{API_QUESTION}/{qid}/answers"
        params = {
            "limit": str(limit),
            "offset": str(offset),
            "sort_by": sort_by,
            "include": "data[*]." + ANSWER_INCLUDE.replace(",", ",data[*]."),
        }
        resp = self._api_get(url, params, needs_zse=False)

        if not self._check_response(resp, f"问题 {qid} 回答 offset={offset}"):
            return []

        data = resp.json().get("data", [])
        answers = []
        for item in data:
            adata = item.get("target", item)  # 知乎有时包裹在 target 中
            if not adata.get("id"):
                continue

            author = adata.get("author", {})
            question = adata.get("question", {})

            answers.append({
                "answer_id": adata.get("id", 0),
                "question_id": question.get("id", qid),
                "question_title": question.get("title", ""),
                "content": strip_html(adata.get("content", "")),
                "excerpt": adata.get("excerpt", ""),
                "publish_time": adata.get("created_time", 0),
                "created_time": adata.get("created_time", 0),
                "updated_time": adata.get("updated_time", 0),
                "author_name": author.get("name", ""),
                "author_url_token": author.get("url_token", ""),
                "author_headline": author.get("headline", ""),
                "author_follower_count": author.get("follower_count", 0),
                "voteup_count": adata.get("voteup_count", 0),
                "comment_count": adata.get("comment_count", 0),
                "thanks_count": adata.get("thanks_count", 0),
            })

        return answers

    # ═══════════════════════════════════════════════════
    # 评论 API
    # ═══════════════════════════════════════════════════

    def get_answer_comments(
        self, answer_id: int, limit: int = MAX_COMMENTS_PER_ANSWER
    ) -> list[dict]:
        """
        获取回答的评论列表。

        Args:
            answer_id: 回答ID
            limit: 最大评论数

        Returns:
            评论列表
        """
        # 知乎新版评论API（旧的 /comments/{id}/root_comments 已废弃，返回404）
        url = f"{API_COMMENTS_V5}/{answer_id}/root_comment"
        params = {
            "order_by": "score",
            "limit": str(min(limit, 20)),
            "offset": "",
        }
        resp = self._api_get(url, params, needs_zse=False)

        if not self._check_response(resp, f"评论 answer={answer_id}"):
            return []

        data = resp.json().get("data", [])
        comments = []
        for c in data:
            content_text = strip_html(c.get("content", ""))
            if not content_text or len(content_text) < 2:
                continue
            comments.append({
                "comment_id": c.get("id", 0),
                "content": content_text,
                "content_length": len(content_text),
                "like_count": c.get("like_count", 0),
                "reply_count": c.get("child_comment_count", 0),
                "publish_time": c.get("created_time", 0),
                "parent_id": 0,  # root comment
            })

        return comments

    # ═══════════════════════════════════════════════════
    # 用户信息 API
    # ═══════════════════════════════════════════════════

    def get_user_info(self, url_token: str) -> Optional[dict]:
        """获取用户信息。结果缓存。"""
        if not url_token:
            return None
        if url_token in self._user_cache:
            return self._user_cache[url_token]

        url = f"{API_USER}/{url_token}"
        # members 端点同样需要 include 才会下发 follower_count
        resp = self._api_get(
            url, params={"include": "follower_count,headline,gender"},
            needs_zse=False,
        )

        if not self._check_response(resp, f"用户 {url_token}"):
            return None

        udata = resp.json()
        info = {
            "url_token": udata.get("url_token", url_token),
            "name": udata.get("name", ""),
            "headline": udata.get("headline", ""),
            "follower_count": udata.get("follower_count", 0),
        }
        self._user_cache[url_token] = info
        return info

    def get_cached_user_follower(self, url_token: str) -> int:
        """获取已缓存的粉丝数（不发起请求）。"""
        if url_token in self._user_cache:
            return self._user_cache[url_token].get("follower_count", 0)
        return 0
