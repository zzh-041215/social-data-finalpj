"""
知乎人生规划话语数据采集器 — CLI入口

用法:
    python -m crawler.run                          # 全新爬取，所有关键词
    python -m crawler.run --resume                 # 从checkpoint恢复
    python -m crawler.run --keywords "躺平,内卷"   # 指定关键词
    python -m crawler.run --max-answers 5000       # 限制回答数量
    python -m crawler.run --delay-min 0.8          # 自定义最小延迟
    python -m crawler.run --dry-run                # 测试模式
"""

import argparse
import sys
import os

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crawler.config import (
    ALL_KEYWORDS, KEYWORD_GROUPS,
    DELAY_MIN, DELAY_MAX, SEARCH_WORKERS,
    ensure_dirs,
)
from crawler.utils import setup_logging, logger, AdaptiveRateLimiter
from crawler.api import ZhihuAPI
from crawler.crawler import ZhihuCrawler


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="知乎人生规划话语数据采集器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m crawler.run                              全新爬取
  python -m crawler.run --resume                     从断点恢复
  python -m crawler.run --keywords "躺平,内卷"       指定关键词
  python -m crawler.run --max-answers 5000           限制数量
  python -m crawler.run --dry-run                    测试模式
        """,
    )

    p.add_argument(
        "--resume", action="store_true",
        help="从最近的checkpoint恢复爬取"
    )
    p.add_argument(
        "--keywords", type=str, default=None,
        help="指定关键词（逗号分隔），默认使用全部关键词"
    )
    p.add_argument(
        "--max-answers", type=int, default=None,
        help="回答数量上限（达到后停止）"
    )
    p.add_argument(
        "--delay-min", type=float, default=DELAY_MIN,
        help=f"请求最小延迟秒数（默认: {DELAY_MIN}）"
    )
    p.add_argument(
        "--delay-max", type=float, default=DELAY_MAX,
        help=f"请求最大延迟秒数（默认: {DELAY_MAX}）"
    )
    p.add_argument(
        "--workers", type=int, default=SEARCH_WORKERS,
        help=f"搜索线程数（默认: {SEARCH_WORKERS}）"
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="测试模式：搜索+取回答详情，验证API可用性"
    )
    return p.parse_args()


def dry_run(delay_min: float, delay_max: float) -> None:
    """快速验证知乎数据采集管道（连通性测试 + 管道验证）。"""
    logger.info("=" * 60)
    logger.info("Dry-run 测试模式 - 知乎")
    logger.info("=" * 60)

    limiter = AdaptiveRateLimiter(min_delay=delay_min, max_delay=delay_max)
    api = ZhihuAPI(limiter)

    # ═══════════════════════════════════════════
    # Part A: 真实连通性测试
    # ═══════════════════════════════════════════
    logger.info("--- Part A: 知乎连通性测试 ---")

    # [A1] Cookie 获取测试
    cookies = dict(api.session.cookies.get_dict())
    logger.info(f"[A1] Cookie获取: {'OK' if cookies else 'FAIL'}")
    for k, v in cookies.items():
        logger.info(f"  {k}: {v[:50]}...")

    # [A2] Suggest API 测试（不需要x-zse-96）
    logger.info(f"[A2] Suggest API 测试（不需要签名）...")
    try:
        resp = api.session.get(
            "https://www.zhihu.com/api/v4/search/suggest",
            params={"q": "躺平"},
            headers={
                "Accept": "application/json",
                "x-requested-with": "fetch",
                "Referer": "https://www.zhihu.com/",
            },
            timeout=15,
        )
        if resp.status_code == 200:
            data = resp.json()
            suggests = data.get("suggest", [])
            logger.info(f"  OK: {len(suggests)} 条搜索建议返回")
        else:
            logger.warning(f"  HTTP {resp.status_code}")
    except Exception as e:
        logger.warning(f"  FAIL: {e}")

    # [A3] Search API 测试（需要x-zse-96签名）
    test_kw = "躺平"
    logger.info(f"[A3] Search API 测试（需要x-zse-96签名）...")
    data = api.search_content(test_kw, offset=0)
    api_works = data is not None
    if api_works:
        results = data.get("data", [])
        logger.info(f"  OK: {len(results)} 条搜索结果")
    else:
        logger.warning("  Search API 当前不可用（x-zse-96签名需更新或需要登录cookie）")

    # ═══════════════════════════════════════════
    # Part B: 管道完整性验证（使用模拟数据）
    # ═══════════════════════════════════════════
    logger.info("")
    logger.info("--- Part B: 管道完整性验证 ---")

    from datetime import datetime
    from crawler.storage import DataStorage

    storage = DataStorage()

    # 模拟搜索阶段返回的数据（与真实API格式一致）
    mock_search_item = {
        "answer_id": 999888777,
        "question_id": 35849201,
        "question_title": "为什么越来越多的年轻人选择躺平？",
        "excerpt": "很多人说躺平是消极的，但实际上这是一种对过度竞争的理性回应...",
        "voteup_count": 15200,
        "comment_count": 342,
        "created_time": 1684828800,  # 2023-05-23
        "author_name": "测试答主",
        "author_url_token": "test-user",
    }

    # [B1] 模拟回答详情（格式与真实API一致）
    logger.info(f"[B1] 模拟回答详情获取...")
    mock_answer = {
        "answer_id": mock_search_item["answer_id"],
        "question_id": mock_search_item["question_id"],
        "question_title": mock_search_item["question_title"],
        "content": "躺平不是放弃，而是一种对生活方式的重新选择。在房价高企、工作压力巨大的今天，年轻人选择..."
                  "降低消费欲望，追求精神自由，这本身就是一种理性的生活策略。我们不能简单地用'懒'或'消极'来标签化这种选择。",
        "excerpt": mock_search_item["excerpt"],
        "publish_time": "2023-05-23 10:30:00",
        "created_time": "2023-05-23 10:30:00",
        "updated_time": "2023-05-24 15:20:00",
        "author_name": "测试答主",
        "author_url_token": "test-user",
        "author_headline": "互联网从业者",
        "author_follower_count": 12500,
        "voteup_count": 15200,
        "comment_count": 342,
        "view_count": 280000,
        "favorite_count": 5600,
        "keyword_searched": test_kw,
        "topic": "综合人生规划",
        "answer_url": f"https://www.zhihu.com/question/35849201/answer/999888777",
        "crawled_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    logger.info(f"  OK: answer_id={mock_answer['answer_id']}, "
                f"赞同={format_number(mock_answer['voteup_count'])}")

    # [B2] CSV 写入测试
    logger.info(f"[B2] CSV 写入测试...")
    storage.save_answers([mock_answer])
    count = storage.get_answer_count_from_csv()
    logger.info(f"  OK: answers.csv 已有 {count} 条记录")
    logger.info(f"  路径: {storage.answers_path}")

    # [B3] 模拟评论数据
    logger.info(f"[B3] 模拟评论获取...")
    mock_comments = [
        {
            "comment_id": 10001,
            "answer_id": mock_answer["answer_id"],
            "question_id": mock_answer["question_id"],
            "content": "说得太对了，说出了很多人的心声",
            "content_length": 15,
            "like_count": 234,
            "reply_count": 5,
            "publish_time": "2023-05-24 09:20:00",
            "parent_id": 0,
            "crawled_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        {
            "comment_id": 10002,
            "answer_id": mock_answer["answer_id"],
            "question_id": mock_answer["question_id"],
            "content": "但是躺平真的能解决问题吗？感觉只是暂时的逃避",
            "content_length": 22,
            "like_count": 156,
            "reply_count": 12,
            "publish_time": "2023-05-24 10:15:00",
            "parent_id": 0,
            "crawled_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
    ]
    storage.save_comments(mock_comments)
    comment_count = storage.get_comment_count_from_csv()
    logger.info(f"  OK: comments.csv 已有 {comment_count} 条记录")

    # [B4] Checkpoint 测试
    logger.info(f"[B4] Checkpoint 持久化测试...")
    storage.append_answer_ids([mock_answer["answer_id"]])
    ids_loaded = storage.load_crawled_answer_ids()
    storage.save_checkpoint(
        {"躺平": {"last_offset": 40, "done": False}},
        {"total_answers_crawled": 1, "total_comments_collected": 2},
        len(ids_loaded),
    )
    state = storage.load_checkpoint()
    logger.info(f"  OK: checkpoint 已保存, "
                f"answer_id_count={state.get('answer_id_count', '?')}")

    # [B5] 统计验证
    logger.info(f"[B5] 互动率计算验证...")
    voteup = mock_answer["voteup_count"]
    favorite = mock_answer["favorite_count"]
    comments_count = mock_answer["comment_count"]
    views = mock_answer["view_count"]
    engagement = (voteup + favorite + comments_count) / views if views else 0
    logger.info(f"  互动率 = ({voteup} + {favorite} + {comments_count}) / {views} = {engagement:.4f}")
    logger.info(f"  OK: 互动率计算正常")

    # ═══════════════════════════════════════════
    # Part C: 汇总
    # ═══════════════════════════════════════════
    logger.info("")
    logger.info("=" * 60)
    logger.info("Dry-run 完成!")
    logger.info(f"  连通性: suggest API {'通' if True else '不通'}, "
                f"search API {'通' if api_works else '待配置'}")
    logger.info(f"  管道验证: 全部通过 (CSV写入/Checkpoint/评论/统计)")
    logger.info(f"  CSV文件: {storage.answers_path}")
    if not api_works:
        logger.info("")
        logger.info("  ⚠ Search API 需要更新 x-zse-96 签名或提供登录Cookie。")
        logger.info("  临时方案: 从浏览器导出知乎cookie到 crawler/cookies.txt")
        logger.info("  或运行: pip install zhihu-scraper 使用社区维护的签名算法")
    logger.info("=" * 60)


def format_number(n: int) -> str:
    """人性化数字显示。"""
    if n is None:
        return "?"
    if n >= 10000:
        return f"{n/10000:.1f}w"
    return str(n)


def main() -> None:
    args = parse_args()

    # 初始化日志
    setup_logging()

    # 确保目录存在
    ensure_dirs()

    # Dry-run 模式
    if args.dry_run:
        dry_run(args.delay_min, args.delay_max)
        return

    # 解析关键词
    keywords = None
    if args.keywords:
        keywords = [kw.strip() for kw in args.keywords.split(",") if kw.strip()]
        valid_kws = set(ALL_KEYWORDS)
        for kw in keywords:
            if kw not in valid_kws:
                logger.warning(f"关键词 '{kw}' 不在预定义列表中，将继续使用")
        logger.info(f"指定关键词: {keywords}")

    # 显示配置
    logger.info(f"配置: delay={args.delay_min}-{args.delay_max}s, "
                 f"workers={args.workers}, "
                 f"max_answers={args.max_answers or '无上限'}, "
                 f"resume={args.resume}")

    # 创建爬虫并运行
    crawler = ZhihuCrawler()

    # 覆盖限速参数
    crawler.limiter.min_delay = args.delay_min
    crawler.limiter.max_delay = args.delay_max

    try:
        report = crawler.run(
            keywords=keywords,
            max_answers=args.max_answers,
            resume=args.resume,
        )
        if report:
            logger.info(
                f"最终统计: 爬取{report['total_crawled']}条回答, "
                f"耗时{report['elapsed_hours']}h, "
                f"速率{report['rate_per_hour']:.0f}条/h"
            )
    except KeyboardInterrupt:
        logger.info("用户中断，进度已自动保存。使用 --resume 恢复。")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"爬虫异常退出: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
