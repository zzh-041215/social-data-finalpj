"""
爬虫编排模块：多线程搜索 + 串行回答详情处理 + 高频checkpoint

架构：
  4个搜索线程（每个负责一个议题组）
       ↓ (共享队列)
  统一去重集合
       ↓ (串行处理)
  逐条回答：获取详情 → 获取答主信息 → 获取评论 → 写入CSV
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
    MAX_OFFSET_PER_KEYWORD, SEARCH_LIMIT,
    MAX_COMMENTS_PER_ANSWER,
    CHECKPOINT_INTERVAL, BATCH_SIZE, SEARCH_WORKERS,
    MIN_ANSWER_LENGTH,
    NETWORK_PAUSE_SEC,
    MAX_CONSECUTIVE_NETWORK_ERRORS,
    ensure_dirs,
)
from crawler.utils import (
    logger, setup_logging,
    AdaptiveRateLimiter, net_error_counter,
    parse_timestamp, format_number,
)
from crawler.api import ZhihuAPI
from crawler.storage import DataStorage, ProgressLogger


class ZhihuCrawler:
    """知乎人生规划话语数据采集器。"""

    def __init__(self):
        ensure_dirs()
        self.limiter = AdaptiveRateLimiter()
        self.api = ZhihuAPI(self.limiter)
        self.storage = DataStorage()

        # 运行时状态
        self.crawled_answer_ids: set[str] = set()
        self.keyword_progress: dict[str, dict] = {}  # {kw: {last_offset, done}}
        self.stats: dict = {
            "total_answers_crawled": 0,
            "total_answers_skipped": 0,
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
        max_answers: Optional[int] = None,
        resume: bool = False,
    ) -> dict:
        """
        启动爬虫。

        Args:
            keywords: 指定关键词列表（None=全部）
            max_answers: 回答数量上限（None=不限制）
            resume: 是否从checkpoint恢复
        """
        logger.info("=" * 60)
        logger.info("知乎人生规划话语数据采集器 启动")
        logger.info("=" * 60)

        # 恢复或初始化
        if resume:
            self._restore_state()

        # 确定关键词
        if keywords is None:
            keywords = ALL_KEYWORDS.copy()

        # 初始化关键词进度
        for kw in keywords:
            if kw not in self.keyword_progress:
                self.keyword_progress[kw] = {"last_offset": -SEARCH_LIMIT, "done": False}

        # 过滤已完成的关键词
        pending_kws = [
            kw for kw in keywords
            if not self.keyword_progress.get(kw, {}).get("done", False)
        ]
        logger.info(f"关键词: {len(keywords)} 个, "
                     f"待处理: {len(pending_kws)}, "
                     f"上限: {max_answers or '无'}")

        if not pending_kws:
            logger.info("所有关键词已完成，无需继续。")
            return self.progress.final_report() if self.progress else {}

        # ── Phase A: 多线程搜索 + 收集answer_id ──
        all_results = self._search_all_keywords(keywords, max_answers)

        # ── Phase B: 串行处理回答详情 ──
        self._process_answers(all_results, keywords, max_answers)

        # ── 最终收尾 ──
        self._finalize()
        return self.progress.final_report() if self.progress else {}

    # ═══════════════════════════════════════════════════
    # Phase A: 多线程搜索
    # ═══════════════════════════════════════════════════

    def _search_all_keywords(
        self, keywords: list[str], max_answers: Optional[int]
    ) -> list[tuple]:
        """
        多线程并行搜索所有关键词。
        使用 offset 分页。
        返回去重后的 (answer_id, keyword, topic, question_title, ...) 列表。
        """
        # 按议题分组
        topic_groups: dict[str, list[str]] = {}
        for topic, kws in KEYWORD_GROUPS.items():
            topic_groups[topic] = [kw for kw in kws if kw in keywords]

        logger.info(f"Phase A: 多线程搜索（{SEARCH_WORKERS}线程）...")

        all_results: list[tuple] = []
        seen_ids: set[str] = self.crawled_answer_ids.copy()

        with ThreadPoolExecutor(max_workers=SEARCH_WORKERS) as executor:
            futures = {}
            kw_pending: dict[str, int] = {}  # 每个关键词尚未完成的搜索任务数
            for topic, kws in topic_groups.items():
                for kw in kws:
                    if self.keyword_progress.get(kw, {}).get("done", False):
                        continue
                    start_offset = self.keyword_progress.get(kw, {}).get("last_offset", -SEARCH_LIMIT) + SEARCH_LIMIT
                    for offset in range(start_offset, MAX_OFFSET_PER_KEYWORD + 1, SEARCH_LIMIT):
                        if max_answers and len(seen_ids) >= max_answers:
                            break
                        if self._shutdown_requested:
                            break
                        fut = executor.submit(
                            self._search_one_offset, kw, topic, offset
                        )
                        futures[fut] = (kw, topic, offset)
                        kw_pending[kw] = kw_pending.get(kw, 0) + 1

            for fut in as_completed(futures):
                kw, topic, offset = futures[fut]
                try:
                    results = fut.result()
                    if results:
                        for item in results:
                            aid_str = str(item["answer_id"])
                            if aid_str not in seen_ids:
                                seen_ids.add(aid_str)
                                all_results.append((
                                    item["answer_id"],
                                    item["question_id"],
                                    item["question_title"],
                                    item["excerpt"],
                                    item["voteup_count"],
                                    item["comment_count"],
                                    item["created_time"],
                                    item["author_name"],
                                    item["author_url_token"],
                                    kw,
                                    topic,
                                ))
                except Exception as e:
                    logger.error(f"搜索异常 [{kw} offset={offset}]: {e}")

                # 仅在该关键词所有搜索任务都完成、且未收到中断信号时才标记 done，
                # 避免中途 Ctrl+C 后 --resume 误以为已完成而漏页。
                kw_pending[kw] -= 1
                if kw_pending[kw] == 0 and not self._shutdown_requested:
                    self.keyword_progress[kw] = {
                        "last_offset": MAX_OFFSET_PER_KEYWORD, "done": True
                    }

                # 检查是否达到上限
                if max_answers and len(all_results) >= max_answers:
                    break

        logger.info(f"搜索完成: 发现 {len(all_results)} 个新回答 "
                     f"(已有 {len(self.crawled_answer_ids)} 个)")

        # 按发布时间排序（老→新），利于时间序列完整性
        all_results.sort(key=lambda x: x[6])  # created_time

        return all_results

    def _search_one_offset(
        self, keyword: str, topic: str, offset: int
    ) -> list[dict]:
        """搜索单个关键词单个offset（供线程池调用），返回通过预筛选的条目。"""
        if self._shutdown_requested:
            return []
        return self.api.search_and_filter(keyword, offset)

    # ═══════════════════════════════════════════════════
    # Phase B: 串行处理回答
    # ═══════════════════════════════════════════════════

    def _process_answers(
        self,
        results_list: list[tuple],
        keywords: list[str],
        max_answers: Optional[int],
    ) -> None:
        """串行处理每条回答：详情 → 用户 → 评论 → 写入。"""
        total = len(results_list)
        self.progress = ProgressLogger(estimated_total=total)

        logger.info(f"Phase B: 处理 {total} 条回答详情...")

        answer_buffer: list[dict] = []
        comment_buffer: list[dict] = []
        new_ids_batch: list[int] = []

        for i, item in enumerate(results_list):
            # 检查上限
            if max_answers and self.stats["total_answers_crawled"] >= max_answers:
                logger.info(f"已达到回答上限 {max_answers}，停止处理。")
                break

            # 检查关闭信号
            if self._shutdown_requested:
                self._save_and_exit(answer_buffer, comment_buffer, new_ids_batch)
                return

            # 检查网络状态
            if net_error_counter.should_pause():
                self._handle_network_pause(answer_buffer, comment_buffer, new_ids_batch)
            if net_error_counter.should_abort():
                self._save_and_exit(answer_buffer, comment_buffer, new_ids_batch)
                return

            (answer_id, question_id, question_title, excerpt,
             voteup_count, comment_count, created_time,
             author_name, author_url_token,
             keyword, topic) = item

            # 跳过已爬
            if str(answer_id) in self.crawled_answer_ids:
                continue

            # ── 获取回答详情 ──
            answer_info = self.api.get_answer_detail(answer_id)
            if answer_info is None:
                self.stats["api_errors"] += 1
                self.progress.log_error(answer_id, keyword, "回答详情获取失败")
                self.crawled_answer_ids.add(str(answer_id))
                new_ids_batch.append(answer_id)
                continue

            # 合并搜索结果中的字段
            answer_info["keyword_searched"] = keyword
            answer_info["topic"] = topic

            # 如果搜索结果中的标题更完整，使用搜索结果的
            if question_title and not answer_info.get("question_title"):
                answer_info["question_title"] = question_title

            # 答主信息（粉丝数/简介/昵称）已由 ANSWER_INCLUDE 在回答详情里一并返回，
            # 无需再单独请求 members 端点（省去每条回答一次 API 调用，更礼貌、更快）。
            # 仅在回答详情未给出昵称时，用搜索结果里的昵称兜底。
            if not answer_info.get("author_name"):
                answer_info["author_name"] = author_name

            # 时间戳转换
            pubdate_ts = answer_info.get("publish_time", created_time)
            answer_info["publish_time"] = parse_timestamp(pubdate_ts) or ""
            answer_info["created_time"] = parse_timestamp(answer_info.get("created_time")) or ""
            answer_info["updated_time"] = parse_timestamp(answer_info.get("updated_time")) or ""
            answer_info["answer_url"] = (
                f"https://www.zhihu.com/question/{answer_info['question_id']}"
                f"/answer/{answer_id}"
            )
            answer_info["crawled_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # ── 获取评论 ──
            comments = self.api.get_answer_comments(answer_id)

            # ── 加入buffer ──
            answer_buffer.append(answer_info)
            self.crawled_answer_ids.add(str(answer_id))
            new_ids_batch.append(answer_id)
            self.stats["total_answers_crawled"] += 1

            for c in comments:
                c["answer_id"] = answer_id
                c["question_id"] = answer_info.get("question_id", question_id)
                c["publish_time"] = parse_timestamp(c.get("publish_time")) or ""
                c["crawled_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                comment_buffer.append(c)
                self.stats["total_comments_collected"] += 1

            self.progress.log_ok(
                answer_id, keyword,
                answer_info.get("voteup_count", 0),
                answer_info.get("comment_count", 0),
            )

            # ── Flush & Checkpoint ──
            if len(answer_buffer) >= BATCH_SIZE:
                self._flush_buffers(answer_buffer, comment_buffer, new_ids_batch)
                answer_buffer.clear()
                comment_buffer.clear()
                new_ids_batch.clear()

            if (self.stats["total_answers_crawled"] % CHECKPOINT_INTERVAL == 0
                    and self.stats["total_answers_crawled"] > 0):
                self._save_checkpoint()

        # ── 处理剩余 ──
        self._flush_buffers(answer_buffer, comment_buffer, new_ids_batch)

    # ═══════════════════════════════════════════════════
    # Buffer & Checkpoint
    # ═══════════════════════════════════════════════════

    def _flush_buffers(
        self,
        answer_buffer: list[dict],
        comment_buffer: list[dict],
        ids_batch: list[int],
    ) -> None:
        """写入CSV并追加answer_id到checkpoint文件。"""
        if answer_buffer:
            self.storage.save_answers(answer_buffer)
        if comment_buffer:
            self.storage.save_comments(comment_buffer)
        if ids_batch:
            self.storage.append_answer_ids(ids_batch)

    def _save_checkpoint(self) -> None:
        """保存运行状态。"""
        self.storage.save_checkpoint(
            self.keyword_progress,
            self.stats,
            len(self.crawled_answer_ids),
        )

    def _save_and_exit(
        self,
        answer_buffer: list[dict],
        comment_buffer: list[dict],
        ids_batch: list[int],
    ) -> None:
        """紧急保存并退出。"""
        logger.warning("正在保存进度...")
        self._flush_buffers(answer_buffer, comment_buffer, ids_batch)
        self._save_checkpoint()
        logger.info(f"进度已保存。"
                     f"回答 {self.stats['total_answers_crawled']}, "
                     f"评论 {self.stats['total_comments_collected']}")
        sys.exit(0)

    # ═══════════════════════════════════════════════════
    # 断网处理
    # ═══════════════════════════════════════════════════

    def _handle_network_pause(
        self,
        answer_buffer: list[dict],
        comment_buffer: list[dict],
        ids_batch: list[int],
    ) -> None:
        """连续网络错误时暂停并等待恢复。"""
        self.stats["network_interruptions"] += 1
        self._flush_buffers(answer_buffer, comment_buffer, ids_batch)
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
                "total_answers_crawled": 0,
                "total_answers_skipped": 0,
                "total_comments_collected": 0,
                "api_errors": 0,
                "network_interruptions": 0,
            })
            logger.info(
                f"从checkpoint恢复: "
                f"已爬 {state.get('answer_id_count', 0)} 条回答, "
                f"时间戳 {state.get('timestamp', '?')}"
            )

        # 从 answer_ids.txt 重建已爬集合
        self.crawled_answer_ids = self.storage.load_crawled_answer_ids()
        logger.info(f"已爬answer_id数量: {len(self.crawled_answer_ids)}")

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
            logger.info(f"  回答: {report['total_crawled']} 条")
            logger.info(f"  跳过: {report['total_skipped']} 条")
            logger.info(f"  评论: {self.stats['total_comments_collected']} 条")
            logger.info(f"  错误: {report['total_errors']} 次")
            logger.info(f"  耗时: {report['elapsed_hours']}h")
            logger.info(f"  速率: {report['rate_per_hour']:.0f} 条/h")
            logger.info("=" * 60)
