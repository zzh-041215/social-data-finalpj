"""
存储模块：CSV追加写入、高频checkpoint、已爬answer_ids管理、进度日志
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from crawler.config import (
    RAW_DIR, CHECKPOINT_DIR, LOG_DIR,
    ANSWERS_CSV, COMMENTS_CSV,
    CHECKPOINT_STATE, CHECKPOINT_ANSWER_IDS,
    ANSWER_COLUMNS, COMMENT_COLUMNS,
    SEARCH_WORKERS,
)
from crawler.utils import logger


# ============================================================
# CSV 写入
# ============================================================

class DataStorage:
    """管理CSV追加写入和checkpoint持久化。"""

    def __init__(self):
        # 确保目录存在
        for d in [RAW_DIR, CHECKPOINT_DIR, LOG_DIR]:
            os.makedirs(d, exist_ok=True)

        self.answers_path = ANSWERS_CSV
        self.comments_path = COMMENTS_CSV
        self.answer_ids_path = CHECKPOINT_ANSWER_IDS

    # ── 回答CSV ──

    def save_answers(self, rows: list[dict]) -> None:
        """追加回答行到CSV（首次写入含表头）。"""
        if not rows:
            return
        df = pd.DataFrame(rows, columns=ANSWER_COLUMNS)
        file_exists = os.path.exists(self.answers_path)
        df.to_csv(
            self.answers_path,
            mode="a",
            header=not file_exists,
            index=False,
            encoding="utf-8-sig",
        )

    def save_comments(self, rows: list[dict]) -> None:
        """追加评论行到CSV。"""
        if not rows:
            return
        df = pd.DataFrame(rows, columns=COMMENT_COLUMNS)
        file_exists = os.path.exists(self.comments_path)
        df.to_csv(
            self.comments_path,
            mode="a",
            header=not file_exists,
            index=False,
            encoding="utf-8-sig",
        )

    # ── 已爬answer_ids管理 ──

    def load_crawled_answer_ids(self) -> set[str]:
        """从 crawled_answer_ids.txt 重建已爬集合。"""
        if not os.path.exists(self.answer_ids_path):
            return set()
        with open(self.answer_ids_path, "r", encoding="utf-8") as f:
            ids_set = {line.strip() for line in f if line.strip()}
        logger.debug(f"从checkpoint恢复了 {len(ids_set)} 个已爬answer_id")
        return ids_set

    def append_answer_ids(self, ids_list: list[int]) -> None:
        """追加新answer_id到checkpoint文件。"""
        with open(self.answer_ids_path, "a", encoding="utf-8") as f:
            for aid in ids_list:
                f.write(f"{aid}\n")

    def get_answer_id_count(self) -> int:
        """快速统计已爬answer_id数量（不加载全部到内存）。"""
        if not os.path.exists(self.answer_ids_path):
            return 0
        count = 0
        with open(self.answer_ids_path, "r", encoding="utf-8") as f:
            for _ in f:
                count += 1
        return count

    # ── Checkpoint ──

    def save_checkpoint(
        self,
        keyword_progress: dict,
        stats: dict,
        answer_id_count: int,
    ) -> str:
        """保存运行状态到 state.json。"""
        state = {
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "keyword_progress": keyword_progress,
            "stats": stats,
            "last_successful_request": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "answer_id_count": answer_id_count,
        }
        with open(CHECKPOINT_STATE, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        return CHECKPOINT_STATE

    def load_checkpoint(self) -> Optional[dict]:
        """加载最近的checkpoint，不存在则返回None。"""
        if not os.path.exists(CHECKPOINT_STATE):
            return None
        try:
            with open(CHECKPOINT_STATE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Checkpoint文件损坏: {e}")
            return None

    # ── 统计 ──

    def get_answer_count_from_csv(self) -> int:
        """从CSV统计已有回答数（不加载全部数据）。"""
        if not os.path.exists(self.answers_path):
            return 0
        try:
            df = pd.read_csv(self.answers_path, usecols=["answer_id"])
            return len(df)
        except Exception:
            return 0

    def get_comment_count_from_csv(self) -> int:
        """从CSV统计已有评论数。"""
        if not os.path.exists(self.comments_path):
            return 0
        try:
            df = pd.read_csv(self.comments_path, usecols=["comment_id"])
            return len(df)
        except Exception:
            return 0


# ============================================================
# 进度日志
# ============================================================

class ProgressLogger:
    """每100条输出汇总，控制台可读的进度跟踪。"""

    def __init__(self, estimated_total: int = 12000):
        self.start_time = time.time()
        self.estimated_total = estimated_total
        self.crawled = 0
        self.skipped = 0
        self.errors = 0
        self.last_report_at = 0

    def log_ok(self, answer_id: int, keyword: str,
               voteup_count: int, comment_count: int) -> None:
        """记录一条成功爬取的回答。"""
        self.crawled += 1
        total = self.crawled + self.skipped
        voteup_str = f"{voteup_count/10000:.1f}w" if voteup_count and voteup_count >= 10000 else str(voteup_count or "?")
        logger.info(
            f"[{self.crawled}/{total}] {answer_id} | {keyword} | "
            f"赞同{voteup_str} 评论{comment_count or '?'} | OK"
        )
        self._maybe_report()

    def log_skip(self, answer_id: int, keyword: str, reason: str) -> None:
        self.skipped += 1
        total = self.crawled + self.skipped
        logger.info(
            f"[{self.crawled}/{total}] {answer_id} | {keyword} | "
            f"SKIP({reason})"
        )
        self._maybe_report()

    def log_error(self, answer_id: int, keyword: str, error: str) -> None:
        self.errors += 1
        total = self.crawled + self.skipped
        logger.warning(
            f"[{self.crawled}/{total}] {answer_id} | {keyword} | "
            f"ERROR: {error}"
        )

    def log_stats_line(self) -> None:
        """输出一条汇总统计行。"""
        elapsed = time.time() - self.start_time
        total = self.crawled + self.skipped
        if self.crawled > 0:
            rate = self.crawled / (elapsed / 3600) if elapsed > 0 else 0
            remaining = self.estimated_total - self.crawled
            eta_h = remaining / rate if rate > 0 else 0
            logger.info(
                f"[STATS] 已爬{self.crawled}条 | 跳过{self.skipped}条 | "
                f"错误{self.errors}次 | "
                f"速率{rate:.0f}条/h | "
                f"剩余约{remaining}条 | ETA {eta_h:.1f}h"
            )
        self.last_report_at = self.crawled

    def _maybe_report(self) -> None:
        """每100条自动输出统计。"""
        if self.crawled > 0 and self.crawled % 100 == 0:
            if self.crawled != self.last_report_at:
                self.log_stats_line()

    def final_report(self) -> dict:
        """爬取完成后的最终统计。"""
        elapsed = time.time() - self.start_time
        return {
            "total_crawled": self.crawled,
            "total_skipped": self.skipped,
            "total_errors": self.errors,
            "elapsed_hours": round(elapsed / 3600, 2),
            "rate_per_hour": round(self.crawled / (elapsed / 3600), 0) if elapsed > 0 else 0,
        }
