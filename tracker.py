"""arXiv 신규 논문 수집 → papers/YYYY-MM-DD.md, docs/papers.json(누적) 갱신. 사용법: python tracker.py [lookback_days]"""
import json, re, sys, time, datetime as dt, pathlib, urllib.error, urllib.parse, urllib.request
import xml.etree.ElementTree as ET
from graph import load_cache, save_cache, vectors

ROOT = pathlib.Path(__file__).parent
NS = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}
API = "https://export.arxiv.org/api/query?"


def parse(xml):
    for e in ET.fromstring(xml).findall("a:entry", NS):
        text = lambda tag: " ".join(e.findtext(f"a:{tag}", "", NS).split())
        yield {
            "id": text("id").rsplit("/abs/", 1)[-1].split("v")[0],
            "url": text("id"),
            "title": text("title"),
            "abstract": text("summary"),
            "published": text("published")[:10],
            "authors": [a.findtext("a:name", "", NS) for a in e.findall("a:author", NS)],
            "cats": [c.get("term") for c in e.findall("a:category", NS)],
            "primary": e.find("x:primary_category", NS).get("term"),
        }


def norm(t):
    """소문자화하고 하이픈·공백을 공백 하나로 통일 → "long-term" / "long term" 모두 일치."""
    return re.sub(r"[\s\-\u2010-\u2014]+", " ", t.lower())


def matched(p, cfg):
    hay = norm(p["title"] + " " + p["abstract"])
    return [" + ".join(k) for k in cfg["keywords"] if all(norm(w) in hay for w in k)]


def to_md(p):
    authors = ", ".join(p["authors"][:5]) + (" et al." if len(p["authors"]) > 5 else "")
    how = ", ".join(p["matched"]) or "semantic"
    return (f"### [{p['title']}]({p['url']})\n"
            f"- {authors}\n- {p['published']} · {', '.join(p['cats'])} · Matched: {how} · Relevance: {p['relevance']:.2f}\n\n"
            f"<details><summary>Abstract</summary>\n\n{p['abstract']}\n</details>\n")


def fetch(params, tries=5):
    """arXiv API는 간헐적으로 5xx·429·타임아웃을 반환하므로 지수 백오프(5, 10, 20, 40초)로 재시도."""
    for i in range(tries):
        try:
            with urllib.request.urlopen(API + urllib.parse.urlencode(params), timeout=120) as r:
                return list(parse(r.read()))
        except (urllib.error.URLError, TimeoutError) as e:
            client_error = isinstance(e, urllib.error.HTTPError) and e.code < 500 and e.code != 429
            if client_error or i == tries - 1:
                raise
            print(f"  arXiv API error ({e}), retry {i + 1}/{tries - 1}")
            time.sleep(5 * 2 ** i)


def fetch_recent(cfg, cutoff):
    """카테고리 전체 신규 논문을 제출일 하루 단위로 조회.
    arXiv API는 깊은 페이지(start 수천 이상)에서 HTTP 500을 반환하므로, 하루씩 나눠 얕은 페이지만 사용."""
    cats = " OR ".join(f"cat:{c}" for c in cfg["categories"])
    out, size, day = [], cfg["page_size"], dt.date.today()
    while day.isoformat() >= cutoff:
        nxt = day + dt.timedelta(days=1)
        query = f"({cats}) AND submittedDate:[{day:%Y%m%d}0000 TO {nxt:%Y%m%d}0000]"
        n, start = 0, 0
        while True:
            params = {"search_query": query, "start": start, "max_results": size}
            page = fetch(params)
            if not page and start == 0:  # arXiv API가 간헐적으로 빈 결과를 반환하므로 1회 재시도
                time.sleep(5)
                page = fetch(params)
            out, n = out + page, n + len(page)
            time.sleep(3)  # arXiv API 호출 간격 권장
            if len(page) < size:
                break
            start += size
        print(f"  {day}: {n} papers")
        day -= dt.timedelta(days=1)
    return out


def main():
    cfg = json.loads((ROOT / "config.json").read_text())
    days = int(sys.argv[1]) if len(sys.argv) > 1 else cfg["lookback_days"]
    today, rel_cfg = dt.date.today().isoformat(), cfg["relevance"]
    db_path = ROOT / "docs" / "papers.json"
    db = json.loads(db_path.read_text()) if db_path.exists() else []
    seen = {p["id"] for p in db}

    # anchor: 키워드·기간 필터 없이 config에 지정한 기준 논문을 1회만 추가하고, config에서 빠진 anchor는 제거
    removed = [p for p in db if p.get("anchor") and p["id"] not in cfg["anchors"]]
    db = [p for p in db if p not in removed]
    for p in removed:
        print(f"  [anchor removed] {p['title'][:90]}")
    todo = [a for a in cfg["anchors"] if a not in seen]
    anchors = fetch({"id_list": ",".join(todo), "max_results": len(todo)}) if todo else []
    for p in anchors:
        p.update(anchor=True, added=today, matched=matched(p, cfg))
    db += anchors

    cutoff = (dt.date.today() - dt.timedelta(days=days)).isoformat()
    cands = {p["id"]: p for p in fetch_recent(cfg, cutoff)
             if p["published"] >= cutoff and p["id"] not in seen and p["primary"] in cfg["categories"]}
    cands = list(cands.values())

    # 관련성 = anchor 논문 중 가장 가까운 것과의 코사인 유사도
    cache = load_cache()
    A = vectors([p for p in db if p.get("anchor")], cfg["embedding_model"], cache)
    rel = lambda ps: (vectors(ps, cfg["embedding_model"], cache) @ A.T).max(1) if ps and len(A) else [1.0] * len(ps)
    for p, r in zip(cands, rel(cands)):
        # 백필(기본보다 긴 기간) 실행에서는 수집일 대신 제출일을 기록해 New this week에 몰리지 않게 함
        p.update(relevance=round(float(r), 3), matched=matched(p, cfg),
                 added=today if days <= cfg["lookback_days"] else p["published"])
    # 키워드 일치 + 최소 관련성, 또는 핵심어(recall_terms)가 있으면서 관련성이 높은 논문(semantic match)
    term = lambda p: any(norm(t) in norm(p["title"] + " " + p["abstract"]) for t in rel_cfg["recall_terms"])
    keep = lambda p: bool(p["matched"] and p["relevance"] >= rel_cfg["min"]) or (p["relevance"] >= rel_cfg["recall"] and term(p))
    new = [p for p in cands if keep(p)]

    # 기존 논문도 같은 기준으로 재평가해 기준 미달(노이즈)은 제거
    old = [p for p in db if not p.get("anchor")]
    for p, r in zip(old, rel(old)):
        p.update(relevance=round(float(r), 3), matched=matched(p, cfg))
    dropped = [p for p in old if not keep(p)]
    db = [p for p in db if p not in dropped] + new

    print(f"candidates={len(cands)} keyword={sum(bool(p['matched']) for p in cands)} "
          f"new={len(new)} semantic_only={sum(not p['matched'] for p in new)} dropped={len(dropped)}")
    for p in sorted((p for p in cands if not p["matched"]), key=lambda p: -p["relevance"])[:15]:
        print(f"  [no-keyword] {p['relevance']:.3f} {'KEEP' if keep(p) else 'skip'} {p['title'][:90]}")
    for p in dropped:
        print(f"  [dropped] {p['relevance']:.3f} {p['title'][:90]}")

    save_cache({p["id"]: cache[p["id"]] for p in db if p["id"] in cache})
    if new:
        out = ROOT / "papers" / f"{today}.md"
        out.parent.mkdir(exist_ok=True)
        head = "" if out.exists() else f"# {today}\n\n"
        with out.open("a") as f:
            f.write(head + "\n".join(to_md(p) for p in new))
    db_path.write_text(json.dumps(db, ensure_ascii=False, indent=0))


if __name__ == "__main__":
    main()
