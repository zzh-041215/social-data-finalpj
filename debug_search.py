"""Minimal debug: test search API directly, bypassing limiter and thread pool."""
import sys, os, time, requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crawler.config import COOKIES_FILE, HEADERS, API_HEADERS, API_SEARCH, SEARCH_LIMIT
from crawler.utils import logger, AdaptiveRateLimiter

# Load cookies
session = requests.Session()
session.headers.update(HEADERS)

if os.path.exists(COOKIES_FILE):
    with open(COOKIES_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip()
                if key and value:
                    session.cookies.set(key, value, domain=".zhihu.com")
    print(f"Loaded cookies from {COOKIES_FILE}")
else:
    print("No cookies.txt found, using guest mode")
    resp = session.get("https://www.zhihu.com/", timeout=20)
    print(f"Guest cookies: {dict(session.cookies.get_dict())}")

has_auth = "z_c0" in [c.name for c in session.cookies]

# Set API headers
for key, value in API_HEADERS.items():
    session.headers[key] = value

# Test 1: Sequential search with different offsets
test_kws = ["躺平", "内卷", "买房"]
for kw in test_kws:
    for offset in [0, 20]:
        params = {
            "q": kw,
            "t": "general",
            "lc_idx": "0",
            "offset": str(offset),
            "limit": str(SEARCH_LIMIT),
        }
        t0 = time.time()
        try:
            resp = session.get(API_SEARCH, params=params, timeout=30)
            elapsed = time.time() - t0
            print(f"[{kw} offset={offset}] HTTP {resp.status_code} in {elapsed:.1f}s")

            if resp.status_code == 200:
                data = resp.json()
                results = data.get("data", [])
                answer_count = sum(1 for r in results if r.get("object", {}).get("type") == "answer")
                print(f"  Total items: {len(results)}, answers: {answer_count}")
                if results:
                    first = results[0]
                    obj = first.get("object", {})
                    print(f"  First item type: {obj.get('type', '?')}, "
                          f"title: {obj.get('question',{}).get('title','?')[:50]}")
            elif resp.status_code == 403:
                print(f"  403 FORBIDDEN — cookie may be invalid or need x-zse-96")
            else:
                print(f"  Body preview: {resp.text[:200]}")
        except requests.Timeout:
            elapsed = time.time() - t0
            print(f"[{kw} offset={offset}] TIMEOUT after {elapsed:.1f}s")
        except Exception as e:
            elapsed = time.time() - t0
            print(f"[{kw} offset={offset}] ERROR after {elapsed:.1f}s: {type(e).__name__}: {e}")

        # Small delay between requests
        time.sleep(0.5)

print("\n=== Debug complete ===")
