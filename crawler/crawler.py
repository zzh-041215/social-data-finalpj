"""
爬虫编排模块：多线程搜索 + 串行视频详情处理 + 高频checkpoint

架构：
  4个搜索线程（每个负责一个议题组）
       ↓ (共享队列)
  统一去重集合
       ↓ (串行处理)
  逐条视频：获取详情 → 获取UP主粉丝数 → 获取评论 → 写入CSV
       ↓
  每10条保存checkpoint
"""

import signal
import sys
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from crawler.config import (
    KEYWORD_GROUPS, ALL_KEYWORDS,
    MAX_PAGES_PER_KEYWORD, MAX_COMMENTS_PER_VIDEO,
    CHECKPOINT_INTERVAL, BATCH_SIZE, SEARCH_WORKERS,
    MIN_VIEW_COUNT, SEARCH_ORDERS,
    NETWORK_PAUSE_SEC,
    MAX_CONSECUTIVE_NETWORK_ERRORS,
    ensure_dirs,
)
from crawler.utils import (
    logger, setup_logging,
    AdaptiveRateLimiter, net_error_counter,
    parse_timestamp, format_number,
)
from crawler.api import BilibiliAPI
from crawler.storage import DataStorage, ProgressLogger


class BilibiliCrawler:
    """B站人生规划话语数据采集器。"""

    def __init__(self):
        ensure_dirs()
        self.limiter = AdaptiveRateLimiter()
        self.api = BilibiliAPI(self.limiter)
        self.storage = DataStorage()

        # 运行时状态
        self.crawled_bvids: set[str] = set()
        self.keyword_progress: dict[str, dict] = {}  # {kw: {last_page, done}}
        self.stats: dict = {
            "total_videos_crawled": 0,
            "total_videos_skipped": 0,
            "total_comments_collected": 0,
            "api_errors": 0,
            "network_interruptions": 0,
        }
        self.progress: Optional[ProgressLogger] = None

        # 注册信号处理
        self._shutdown_requested = False
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    # ═══════════════════════════════════════════════════
    # 主入口
    # ═══════════════════════════════════════════════════

    def run(
        self,
        keywords: Optional[list[str]] = None,
        max_videos: Optional[int] = None,
        resume: bool = False,
    ) -> dict:
        """
        启动爬虫。

        Args:
            keywords: 指定关键词列表（None=全部）
            max_videos: 视频数量上限（None=不限制）
            resume: 是否从checkpoint恢复
        """
        logger.info("=" * 60)
        logger.info("B站人生规划话语数据采集器 启动")
        logger.info("=" * 60)

        # 恢复或初始化
        if resume:
            self._restore_state()

        # 确定关键词
        if keywords is None:
            keywords = ALL_KEYWORDS.copy()

        # 初始化关键词进度（每个 keyword+order 组合独立追踪）
        for kw in keywords:
            for order in SEARCH_ORDERS:
                progress_key = f"{kw}__{order}"
                if progress_key not in self.keyword_progress:
                    self.keyword_progress[progress_key] = {"last_page": 0, "done": False}

        # 过滤已完成的关键词
        pending_items = [
            (kw, order) for kw in keywords
            for order in SEARCH_ORDERS
            if not self.keyword_progress.get(f"{kw}__{order}", {}).get("done", False)
        ]
        logger.info(f"关键词: {len(keywords)} × 排序: {len(SEARCH_ORDERS)} = "
                     f"{len(keywords) * len(SEARCH_ORDERS)} 组搜索, "
                     f"待处理: {len(pending_items)}, "
                     f"上限: {max_videos or '无'}")

        if not pending_items:
            logger.info("所有搜索组合已完成，无需继续。")
            return self.progress.final_report() if self.progress else {}

        # ── Phase A: 多线程搜索 + 收集bvid ──
        all_bvids = self._search_all_keywords(keywords, max_videos)

        # ── Phase B: 串行处理视频详情 ──
        self._process_videos(all_bvids, keywords, max_videos)

        # ── 最终收尾 ──
        self._finalize()
        return self.progress.final_report() if self.progress else {}

    # ═══════════════════════════════════════════════════
    # Phase A: 多线程搜索
    # ═══════════════════════════════════════════════════

    def _search_all_keywords(
        self, keywords: list[str], max_videos: Optional[int]
    ) -> list[tuple[str, str, str]]:
        """
        多线程并行搜索所有(keyword × order)组合。
        使用多种排序方式（pubdate, click, dm）确保时间覆盖。
        返回去重后的 (bvid, keyword, topic) 列表。
        """
        # 按议题分组
        topic_groups: dict[str, list[str]] = {}
        for topic, kws in KEYWORD_GROUPS.items():
            topic_groups[topic] = [kw for kw in kws if kw in keywords]

        logger.info(f"Phase A: 多线程搜索（{SEARCH_WORKERS}线程 × {len(SEARCH_ORDERS)}排序）...")

        all_results: list[tuple[str, str, str, str, int, str, str]]
        all_results = []
        seen_bvids: set[str] = self.crawled_bvids.copy()

        with ThreadPoolExecutor(max_workers=SEARCH_WORKERS) as executor:
            futures = {}
            for topic, kws in topic_groups.items():
                for kw in kws:
                    for order in SEARCH_ORDERS:
                        progress_key = f"{kw}__{order}"
                        if self.keyword_progress.get(progress_key, {}).get("done", False):
                            continue
                        start_page = self.keyword_progress.get(progress_key, {}).get("last_page", 0) + 1
                        for page in range(start_page, MAX_PAGES_PER_KEYWORD + 1):
                            if max_videos and len(seen_bvids) >= max_videos:
                                break
                            if self._shutdown_requested:
                                break
                            fut = executor.submit(
                                self._search_one_page, kw, topic, page, order
                            )
                            futures[fut] = (kw, topic, page, order, progress_key)
                        # 标记该组合已调度
                        if progress_key in self.keyword_progress:
                            self.keyword_progress[progress_key]["last_page"] = MAX_PAGES_PER_KEYWORD
                            self.keyword_progress[progress_key]["done"] = True
                        else:
                            self.keyword_progress[progress_key] = {
                                "last_page": MAX_PAGES_PER_KEYWORD, "done": True
                            }

            for fut in as_completed(futures):
                kw, topic, page, order, progress_key = futures[fut]
                try:
                    results = fut.result()
                    if results:
                        for item in results:
                            bvid = item["bvid"]
                            if bvid not in seen_bvids:
                                seen_bvids.add(bvid)
                                all_results.append((
                                    bvid, kw, topic,
                                    item["title"], item["play"],
                                    item["tag"], item["pubdate"],
                                    item["aid"], item["author"],
                                    item["mid"], item["description"],
                                    item["duration_sec"], item["duration_str"],
                                    item["video_review"],
                                ))
                except Exception as e:
                    logger.error(f"搜索异常 [{kw} p{page} o={order}]: {e}")

                # 检查是否达到上限
                if max_videos and len(all_results) >= max_videos:
                    break

        logger.info(f"搜索完成: 发现 {len(all_results)} 个新视频 "
                     f"(已有 {len(self.crawled_bvids)} 个)")

        # 按发布时间排序（老→新），利于时间序列完整性
        all_results.sort(key=lambda x: x[6])  # pubdate

        return all_results

    def _search_one_page(
        self, keyword: str, topic: str, page: int, order: str = "pubdate"
    ) -> list[dict]:
        """搜索单个关键词单页（供线程池调用），返回通过预筛选的条目。"""
        if self._shutdown_requested:
            return []
        return self.api.search_and_filter(keyword, page, order=order)

    # ═══════════════════════════════════════════════════
    # Phase B: 串行处理视频
    # ═══════════════════════════════════════════════════

    def _process_videos(
        self,
        bvid_list: list[tuple],
        keywords: list[str],
        max_videos: Optional[int],
    ) -> None:
        """串行处理每条视频：详情 → 用户 → 评论 → 写入。"""
        total = len(bvid_list)
        self.progress = ProgressLogger(estimated_total=total)

        logger.info(f"Phase B: 处理 {total} 条视频详情...")

        video_buffer: list[dict] = []
        comment_buffer: list[dict] = []
        new_bvids_batch: list[str] = []

        for i, item in enumerate(bvid_list):
            # 检查上限
            if max_videos and self.stats["total_videos_crawled"] >= max_videos:
                logger.info(f"已达到视频上限 {max_videos}，停止处理。")
                break

            # 检查关闭信号
            if self._shutdown_requested:
                self._save_and_exit(video_buffer, comment_buffer, new_bvids_batch)
                return

            # 检查网络状态
            if net_error_counter.should_pause():
                self._handle_network_pause(video_buffer, comment_buffer, new_bvids_batch)
            if net_error_counter.should_abort():
                self._save_and_exit(video_buffer, comment_buffer, new_bvids_batch)
                return

            bvid, keyword, topic, title, play, tag, pubdate, \
                aid, author, mid, description, duration_sec, \
                duration_str, video_review = item

            # 跳过已爬
            if bvid in self.crawled_bvids:
                continue

            # ── 获取视频详情 ──
            video_info = self.api.get_video_info(bvid)
            if video_info is None:
                self.stats["api_errors"] += 1
                self.progress.log_error(bvid, keyword, "视频详情获取失败")
                self.crawled_bvids.add(bvid)  # 标记为已尝试
                new_bvids_batch.append(bvid)
                continue

            # 合并搜索结果中的字段
            video_info["tags"] = tag
            video_info["tag_list"] = tag
            video_info["keyword_searched"] = keyword
            video_info["topic"] = topic
            video_info["duration_sec"] = video_info["duration_sec"] or duration_sec

            # 使用搜索结果中的标题（有时比详情更完整）
            if title and len(title) > len(video_info.get("title", "")):
                video_info["title"] = title
            if description:
                video_info["description"] = (
                    video_info.get("description", "") + " " + description
                ).strip()

            # ── 获取UP主粉丝数 ──
            uploader_mid = video_info.get("uploader_mid", 0)
            if uploader_mid:
                user_info = self.api.get_user_info(uploader_mid)
                if user_info:
                    video_info["uploader_follower_count"] = user_info.get("follower", 0)
                    video_info["uploader_name"] = (
                        user_info.get("name") or video_info.get("uploader_name", "")
                    )
                else:
                    # 尝试缓存
                    video_info["uploader_follower_count"] = \
                        self.api.get_cached_user_follower(uploader_mid)

            # 时间戳转换
            pubdate_ts = video_info.get("publish_time", pubdate)
            video_info["publish_time"] = parse_timestamp(pubdate_ts) or ""
            video_info["video_url"] = f"https://www.bilibili.com/video/{bvid}"
            video_info["crawled_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # ── 获取评论 ──
            aid_val = video_info.get("aid", aid)
            comments = self.api.get_video_comments(aid_val)

            # ── 加入buffer ──
            video_buffer.append(video_info)
            self.crawled_bvids.add(bvid)
            new_bvids_batch.append(bvid)
            self.stats["total_videos_crawled"] += 1

            for c in comments:
                c["bvid"] = bvid
                c["publish_time"] = parse_timestamp(c.get("publish_time")) or ""
                c["crawled_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                comment_buffer.append(c)
                self.stats["total_comments_collected"] += 1

            self.progress.log_ok(
                bvid, keyword,
                video_info.get("view_count", 0),
                video_info.get("like_count", 0),
            )

            # ── Flush & Checkpoint ──
            if len(video_buffer) >= BATCH_SIZE:
                self._flush_buffers(video_buffer, comment_buffer, new_bvids_batch)
                video_buffer.clear()
                comment_buffer.clear()
                new_bvids_batch.clear()

            if (self.stats["total_videos_crawled"] % CHECKPOINT_INTERVAL == 0
                    and self.stats["total_videos_crawled"] > 0):
                self._save_checkpoint()
                # 每100条不重复输出（ProgressLogger已处理）
                if self.stats["total_videos_crawled"] % 100 == 0:
                    pass  # ProgressLogger._maybe_report 已输出

        # ── 处理剩余 ──
        self._flush_buffers(video_buffer, comment_buffer, new_bvids_batch)

    # ═══════════════════════════════════════════════════
    # Buffer & Checkpoint
    # ═══════════════════════════════════════════════════

    def _flush_buffers(
        self,
        video_buffer: list[dict],
        comment_buffer: list[dict],
        bvids_batch: list[str],
    ) -> None:
        """写入CSV并追加bvid到checkpoint文件。"""
        if video_buffer:
            self.storage.save_videos(video_buffer)
        if comment_buffer:
            self.storage.save_comments(comment_buffer)
        if bvids_batch:
            self.storage.append_bvids(bvids_batch)

    def _save_checkpoint(self) -> None:
        """保存运行状态。"""
        self.storage.save_checkpoint(
            self.keyword_progress,
            self.stats,
            len(self.crawled_bvids),
        )

    def _save_and_exit(
        self,
        video_buffer: list[dict],
        comment_buffer: list[dict],
        bvids_batch: list[str],
    ) -> None:
        """紧急保存并退出。"""
        logger.warning("正在保存进度...")
        self._flush_buffers(video_buffer, comment_buffer, bvids_batch)
        self._save_checkpoint()
        logger.info(f"进度已保存。"
                     f"视频 {self.stats['total_videos_crawled']}, "
                     f"评论 {self.stats['total_comments_collected']}")
        sys.exit(0)

    # ═══════════════════════════════════════════════════
    # 断网处理
    # ═══════════════════════════════════════════════════

    def _handle_network_pause(
        self,
        video_buffer: list[dict],
        comment_buffer: list[dict],
        bvids_batch: list[str],
    ) -> None:
        """连续网络错误时暂停并等待恢复。"""
        self.stats["network_interruptions"] += 1
        self._flush_buffers(video_buffer, comment_buffer, bvids_batch)
        self._save_checkpoint()

        logger.warning(
            f"连续 {net_error_counter.consecutive_errors} 次网络错误，"
            f"暂停 {NETWORK_PAUSE_SEC}s..."
        )
        time.sleep(NETWORK_PAUSE_SEC)

        if net_error_counter.consecutive_errors >= MAX_CONSECUTIVE_NETWORK_ERRORS:
            logger.critical("网络长时间不可用，保存进度并退出。")
            logger.critical("恢复网络后运行: python -m crawler.run --resume")
            sys.exit(0)

        logger.info("重试中...")
        net_error_counter.consecutive_errors = 0

    # ═══════════════════════════════════════════════════
    # Resume & Signal
    # ═══════════════════════════════════════════════════

    def _restore_state(self) -> None:
        """从checkpoint恢复状态。"""
        state = self.storage.load_checkpoint()
        if state:
            self.keyword_progress = state.get("keyword_progress", {})
            self.stats = state.get("stats", {
                "total_videos_crawled": 0,
                "total_videos_skipped": 0,
                "total_comments_collected": 0,
                "api_errors": 0,
                "network_interruptions": 0,
            })
            logger.info(
                f"从checkpoint恢复: "
                f"已爬 {state.get('bvid_count', 0)} 条视频, "
                f"时间戳 {state.get('timestamp', '?')}"
            )

        # 从 bvids.txt 重建已爬集合
        self.crawled_bvids = self.storage.load_crawled_bvids()
        logger.info(f"已爬bvid数量: {len(self.crawled_bvids)}")

    def _signal_handler(self, signum, frame) -> None:
        """Ctrl+C 信号处理。"""
        sig_name = signal.Signals(signum).name
        logger.warning(f"\n收到 {sig_name} 信号，正在优雅退出...")
        self._shutdown_requested = True

    def _finalize(self) -> None:
        """最终收尾。"""
        self._save_checkpoint()
        if self.progress:
            report = self.progress.final_report()
            logger.info("=" * 60)
            logger.info("爬取完成!")
            logger.info(f"  视频: {report['total_crawled']} 条")
            logger.info(f"  跳过: {report['total_skipped']} 条")
            logger.info(f"  评论: {self.stats['total_comments_collected']} 条")
            logger.info(f"  错误: {report['total_errors']} 次")
            logger.info(f"  耗时: {report['elapsed_hours']}h")
            logger.info(f"  速率: {report['rate_per_hour']:.0f} 条/h")
            logger.info("=" * 60)
