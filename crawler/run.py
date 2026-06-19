"""
B站人生规划话语数据采集器 — CLI入口

用法:
    python -m crawler.run                          # 全新爬取，所有关键词
    python -m crawler.run --resume                 # 从checkpoint恢复
    python -m crawler.run --keywords "躺平,上岸"   # 指定关键词
    python -m crawler.run --max-videos 5000        # 限制视频数量
    python -m crawler.run --delay-min 0.5          # 自定义最小延迟
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
from crawler.api import BilibiliAPI
from crawler.crawler import BilibiliCrawler


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="B站人生规划话语数据采集器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m crawler.run                          全新爬取
  python -m crawler.run --resume                 从断点恢复
  python -m crawler.run --keywords "躺平,内卷"   指定关键词
  python -m crawler.run --max-videos 5000        限制数量
  python -m crawler.run --dry-run                测试模式
        """,
    )

    p.add_argument(
        "--resume", action="store_true",
        help="从最近的checkpoint恢复爬取"
    )
    p.add_argument(
        "--keywords", type=str, default=None,
        help="指定关键词（逗号分隔），默认使用全部42个关键词"
    )
    p.add_argument(
        "--max-videos", type=int, default=None,
        help="视频数量上限（达到后停止）"
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
        help="测试模式：搜索1页+取1条视频详情，验证API可用性"
    )
    return p.parse_args()


def dry_run(delay_min: float, delay_max: float) -> None:
    """快速验证API连通性。"""
    logger.info("=" * 60)
    logger.info("Dry-run 测试模式")
    logger.info("=" * 60)

    limiter = AdaptiveRateLimiter(min_delay=delay_min, max_delay=delay_max)
    api = BilibiliAPI(limiter)

    # 测试搜索 — 展示不同排序方式的时间覆盖
    test_kw = "躺平"
    logger.info(f"[1/4] 搜索关键词: {test_kw}（多排序时间覆盖测试）")

    from datetime import datetime, timezone
    for order in ["pubdate", "click", "dm"]:
        raw = api.search_videos(test_kw, page=1, order=order)
        if raw:
            results = raw.get("result", [])
            if results:
                dates = []
                for r in results[:5]:
                    ts = r.get("pubdate", 0)
                    if ts > 0:
                        dates.append(datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d"))
                logger.info(f"  order={order}: {len(results)}条/页, 日期范围: {min(dates) if dates else '?'} ~ {max(dates) if dates else '?'}")

    # 测试预筛选 — 用click排序找2020-2025数据
    logger.info(f"[2/4] 用 order=click 搜索并预筛选:")
    total = 0
    for pg in range(1, 4):
        results = api.search_and_filter(test_kw, page=pg, order="click")
        total += len(results)
        if results:
            r = results[0]
            logger.info(f"  page={pg}: {len(results)} 条通过预筛选, 首条={r['title'][:40]}...")
            break
    if total == 0:
        logger.info(f"  前3页click排序共 {total} 条（关键词'{test_kw}'可能视频量少或全在2026）")

    if results:
        r = results[0]
        logger.info(f"  首条: {r['bvid']} | {r['title'][:50]}... "
                     f"| 播放{format_number(r['play'])}")

        # 测试视频详情
        logger.info(f"[3/4] 获取视频详情: {r['bvid']}")
        info = api.get_video_info(r["bvid"])
        if info:
            logger.info(f"  OK: 点赞{format_number(info['like_count'])}, "
                         f"投币{format_number(info['coin_count'])}, "
                         f"分区: {info['category_name']}")

            # 测试评论
            logger.info(f"[4/4] 获取评论: oid={info['aid']}")
            comments = api.get_video_comments(info["aid"])
            logger.info(f"  评论: {len(comments)} 条")
            if comments:
                logger.info(f"  首条: {comments[0]['text'][:60]}...")

            # 测试用户信息
            mid = info.get("uploader_mid", 0)
            if mid:
                user = api.get_user_info(mid)
                if user:
                    logger.info(f"  UP主: {user['name']}, "
                                 f"粉丝: {format_number(user['follower'])}")
        else:
            logger.warning("  视频详情获取失败（可能需要cookie?）")
    else:
        logger.warning("  搜索结果为空！请检查网络或API可用性。")

    logger.info("=" * 60)
    logger.info("Dry-run 完成。如果以上输出正常，可以开始全量爬取。")
    logger.info("=" * 60)


def format_number(n: int) -> str:
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
        # 验证关键词
        valid_kws = set(ALL_KEYWORDS)
        for kw in keywords:
            if kw not in valid_kws:
                logger.warning(f"关键词 '{kw}' 不在预定义列表中，将继续使用")
        logger.info(f"指定关键词: {keywords}")

    # 显示配置
    logger.info(f"配置: delay={args.delay_min}-{args.delay_max}s, "
                 f"workers={args.workers}, "
                 f"max_videos={args.max_videos or '无上限'}, "
                 f"resume={args.resume}")

    # 创建爬虫并运行
    crawler = BilibiliCrawler()

    # 覆盖限速参数
    crawler.limiter.min_delay = args.delay_min
    crawler.limiter.max_delay = args.delay_max

    try:
        report = crawler.run(
            keywords=keywords,
            max_videos=args.max_videos,
            resume=args.resume,
        )
        if report:
            logger.info(
                f"最终统计: 爬取{report['total_crawled']}条视频, "
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
