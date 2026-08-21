"""抓语料：GitHub 公开 API 拉 jnMetaCode 高 star 仓库的 README 到 data/。

无需 token（匿名限流 60 次/时，30 个仓库足够）。
用法：uv run python scripts/fetch_corpus.py [用户名] [仓库数]
"""

import sys
from pathlib import Path

import httpx

USER = sys.argv[1] if len(sys.argv) > 1 else "jnMetaCode"
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 20
OUT = Path("data")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    with httpx.Client(timeout=30, follow_redirects=True) as c:
        repos = c.get(
            f"https://api.github.com/users/{USER}/repos",
            params={"per_page": 100, "sort": "pushed"},
        ).json()
        if isinstance(repos, dict):  # rate limit / 错误
            print("GitHub API 返回异常：", repos.get("message"))
            return
        repos = sorted(repos, key=lambda r: r.get("stargazers_count", 0), reverse=True)[:LIMIT]
        for r in repos:
            name = r["name"]
            resp = c.get(
                f"https://api.github.com/repos/{USER}/{name}/readme",
                headers={"Accept": "application/vnd.github.raw+json"},
            )
            if resp.status_code != 200:
                print(f"  跳过 {name}（无 README）")
                continue
            (OUT / f"{name}.md").write_text(resp.text, encoding="utf-8")
            print(f"  ✓ {name} ({r.get('stargazers_count', 0)}★, {len(resp.text)} 字符)")
    print(f"语料已写入 {OUT}/")


if __name__ == "__main__":
    main()
