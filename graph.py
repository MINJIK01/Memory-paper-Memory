"""docs/papers.json → docs/graph.json: 임베딩 kNN 이웃, k-means 클러스터(+TF-IDF 라벨), 벤치마크 태그."""
import json, math, re, pathlib, collections
import numpy as np

ROOT = pathlib.Path(__file__).parent
SHARDS = ROOT / "data" / "embeddings"
LEGACY = ROOT / "data" / "embeddings.npz"
CENTROIDS = ROOT / "data" / "centroids.npy"
STOP = set("""a an the and or of to in on for with by from as at is are was were be been this that these those we our
it its can which via into than not but also such their they has have more both while how what when where using
based use new show shows propose proposed approach method methods paper results model models task tasks
respectively respective various existing however often across within without further thus present first two three""".split())


def stem(w):
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    return w[:-1] if w.endswith("s") and not w.endswith("ss") and len(w) > 4 else w


def tfidf(texts):
    docs = [collections.Counter(stem(w) for w in re.findall(r"[a-z][a-z0-9-]+", t.lower()) if w not in STOP)
            for t in texts]
    df = collections.Counter(w for d in docs for w in d)
    vecs = []
    for d in docs:
        v = {w: c * (math.log((1 + len(docs)) / (1 + df[w])) + 1) for w, c in d.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        vecs.append({w: x / norm for w, x in v.items()})
    return vecs


def embed(texts, model):
    from fastembed import TextEmbedding
    return np.array(list(TextEmbedding(model, cache_dir=str(ROOT / ".cache" / "fastembed")).embed(texts)), dtype=np.float32)


def shard(pid):
    """arXiv ID 앞 4자리(YYMM)별로 임베딩 파일을 나눔 → 매일 바뀌는 파일은 이번 달 것 하나뿐 (git 히스토리 증가 억제)."""
    return pid[:4] if re.match(r"\d{4}\.", pid) else "other"


def load_cache():
    files = sorted(SHARDS.glob("*.npz")) + ([LEGACY] if LEGACY.exists() else [])
    return {i: v for f in files for z in [np.load(f)] for i, v in zip(z["ids"].tolist(), z["vecs"])}


def save_cache(cache):
    """구성이 바뀐 shard만 float16으로 다시 쓰고, 빈 shard와 예전 단일 파일은 삭제."""
    SHARDS.mkdir(parents=True, exist_ok=True)
    groups = collections.defaultdict(list)
    for i in cache:
        groups[shard(i)].append(i)
    for name, ids in groups.items():
        f = SHARDS / f"{name}.npz"
        if f.exists() and set(np.load(f)["ids"].tolist()) == set(ids):
            continue
        np.savez_compressed(f, ids=np.array(sorted(ids)), vecs=np.stack([cache[i] for i in sorted(ids)]).astype(np.float16))
    for f in SHARDS.glob("*.npz"):
        if f.stem not in groups:
            f.unlink()
    LEGACY.unlink(missing_ok=True)


def vectors(papers, model, cache):
    """cache에 없는 논문만 임베딩해 cache에 추가하고, 정규화된 행렬을 반환."""
    todo = [p for p in papers if p["id"] not in cache]
    if todo:
        cache.update(zip((p["id"] for p in todo), embed([p["title"] + ". " + p["abstract"] for p in todo], model)))
    if not papers:
        return np.zeros((0, 1), np.float32)
    X = np.stack([cache[p["id"]] for p in papers]).astype(np.float32)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def kmeans(X, k, C0=None, iters=100):
    """C0(이전 실행의 중심)로 시작해 클러스터 번호·색이 실행마다 유지되도록 함. 빈 클러스터는 가장 동떨어진 논문으로 재시드."""
    C0 = np.zeros((0, X.shape[1])) if C0 is None or C0.shape[1] != X.shape[1] else C0[:k]
    C = np.vstack([C0, X[np.linspace(0, len(X) - 1, k - len(C0)).astype(int)]])
    for _ in range(iters):
        lab = (X @ C.T).argmax(1)
        C2 = np.stack([X[lab == j].mean(0) if (lab == j).any() else X[(X @ C.T).max(1).argmin()] for j in range(k)])
        C2 /= np.linalg.norm(C2, axis=1, keepdims=True)
        if np.allclose(C2, C):
            break
        C = C2
    return (X @ C.T).argmax(1), C


def cluster_labels(texts, lab, k, n=3):
    vecs = tfidf(texts)
    total = collections.Counter()
    for v in vecs:
        total.update(v)
    labels = []
    for j in range(k):
        members = [v for v, l in zip(vecs, lab) if l == j]
        inside, df = collections.Counter(), collections.Counter()
        for v in members:
            inside.update(v)
            df.update(v.keys())
        # 클러스터 내 2편 이상에 나오는 단어 중, 전체 평균 대비 비중이 큰 단어 (논문 고유 이름 배제)
        score = {w: inside[w] / len(members) - total[w] / len(vecs) for w in inside if df[w] >= min(2, len(members))}
        labels.append(sorted(score, key=score.get, reverse=True)[:n])
    return labels


def main():
    cfg = json.loads((ROOT / "config.json").read_text())
    papers = json.loads((ROOT / "docs" / "papers.json").read_text())
    ids = [p["id"] for p in papers]
    texts = [p["title"] + " " + p["abstract"] for p in papers]
    cache = load_cache()
    X = vectors(papers, cfg["embedding_model"], cache)
    save_cache({i: cache[i] for i in ids})

    S = X @ X.T
    np.fill_diagonal(S, -1)
    top = np.argsort(-S, axis=1)[:, :cfg["graph_top_k"]]
    neighbors = {ids[i]: [[ids[j], round(float(S[i, j]), 3)] for j in top[i]] for i in range(len(ids))}

    k = min(8, len(papers), max(2, round(math.sqrt(len(papers) / 2))))
    lab, C = kmeans(X, k, np.load(CENTROIDS) if CENTROIDS.exists() else None)
    np.save(CENTROIDS, C)

    pats = {name: re.compile("|".join(rf"(?<![A-Za-z]){re.escape(a)}(?![A-Za-z])" for a in aliases))
            for name, aliases in cfg["benchmarks"].items()}
    benchmarks = {p["id"]: b for p, t in zip(papers, texts) if (b := [n for n, r in pats.items() if r.search(t)])}

    # semantic match 논문(키워드 없음)은 키워드 그룹별 임베딩 중심이 가장 가까운 키워드로 배정
    groups = collections.defaultdict(list)
    for i, p in enumerate(papers):
        if p["matched"] and not p.get("anchor"):
            groups[p["matched"][0]].append(i)
    names = list(groups)
    C = np.stack([X[groups[n]].mean(0) for n in names]) if names else np.zeros((0, X.shape[1]))
    C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-9
    inferred = {p["id"]: names[int((C @ X[i]).argmax())] for i, p in enumerate(papers)
                if names and not p["matched"] and not p.get("anchor")}

    out = {"neighbors": neighbors, "inferred": inferred, "cluster": {i: int(l) for i, l in zip(ids, lab)},
           "clusters": cluster_labels(texts, lab, k), "benchmarks": benchmarks, "window_days": cfg["graph_window_days"]}
    (ROOT / "docs" / "graph.json").write_text(json.dumps(out))
    print(f"nodes={len(papers)} clusters={k} benchmark_tagged={len(benchmarks)}")


if __name__ == "__main__":
    main()
