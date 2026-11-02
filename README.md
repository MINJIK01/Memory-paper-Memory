# arXiv Paper Tracker

`config.json`의 카테고리·키워드로 arXiv 신규 논문을 평일마다 자동 수집해 `papers/YYYY-MM-DD.md`로 커밋합니다. API 키는 필요 없습니다.
Collects new arXiv papers matching the categories and keywords in `config.json` every weekday and commits them to `papers/YYYY-MM-DD.md`. No API keys are required.

**Paper graph / 논문 그래프:** https://minjik01.github.io/Memory-paper-Memory/

## 설정 / Configuration (`config.json`)
| Key | 설명 | Description |
|---|---|---|
| `categories` | arXiv 카테고리. 주 카테고리(primary)가 이 목록에 있는 논문만 저장 | arXiv categories. Only papers whose primary category is listed are kept |
| `keywords` | 각 항목은 제목·abstract 내 AND 조건, 항목끼리는 OR. 대소문자·하이픈·공백 차이 무시 | Each entry is an AND of terms in the title/abstract; entries are OR-ed. Case, hyphens and spacing are ignored |
| `lookback_days` | 수집 대상 기간 (제출일 기준) | Collection window in days (by submission date) |
| `page_size` | arXiv API 페이지당 조회 수 (기간 끝까지 페이지 단위로 조회) | Results per arXiv API page (pages are fetched until the window ends) |
| `relevance.min` | 키워드가 맞아도 이 값 미만이면 제외. 수집 규칙은 기존 논문에도 매 실행 다시 적용 | Keyword matches below this are dropped; the rules are re-applied to stored papers on every run |
| `relevance.recall` | 키워드가 없어도 이 값 이상이고 `recall_terms` 중 하나가 있으면 semantic match로 수집 | Papers at or above this that contain one of `recall_terms` are kept as semantic matches even without a keyword |
| `relevance.recall_terms` | semantic match에 필요한 핵심어 (부분 일치) | Core terms a semantic match must contain (substring match) |
| `graph_top_k` | 그래프에서 논문별로 저장할 유사 논문 수 | Number of similar papers stored per paper for the graph |
| `graph_window_days` | 그래프 기본 표시 기간(일). 이전 논문은 All time에서 표시 | Default graph window in days; older papers appear under All time |
| `benchmarks` | 벤치마크 이름 → abstract에서 찾을 표기 목록 | Benchmark name → spellings to look for in the title/abstract |
| `anchors` | 키워드·기간과 무관하게 추가할 기준 논문 arXiv ID. 목록에서 빼면 다음 실행 때 제거 | arXiv IDs of reference papers added regardless of keywords or dates; removed on the next run once taken off the list |
| `embedding_model` | 유사도 계산용 임베딩 모델 (fastembed) | Embedding model used for similarity (fastembed) |

관련성(relevance)은 논문 임베딩과 anchor 논문 임베딩 사이 코사인 유사도의 최댓값입니다. 수집한 논문은 `docs/papers.json`에 누적되며 중복 수집되지 않습니다.
Relevance is the highest cosine similarity between a paper's embedding and any anchor paper's embedding. Collected papers accumulate in `docs/papers.json` and are never collected twice.

## 실행 / Running
- 자동 / Scheduled: 평일 02:23 UTC (`.github/workflows/track.yml`) / weekdays at 02:23 UTC
- 수동 / Manual: Actions → arXiv tracker → Run workflow (`days`로 과거 기간 백필 / backfill with `days`)
- 로컬 / Local: `pip install -r requirements.txt && python tracker.py [days] && python graph.py`

## Anchor 논문 / Anchor papers
Anchor 논문은 에이전트 메모리, 장기 대화 메모리, 페르소나 대화, 롤플레이 캐릭터 분야의 대표 선행 연구로, `config.json`의 `anchors`에 직접 지정합니다. 두 가지 역할을 합니다.
Anchor papers are well-known prior work on agent memory, long-term conversational memory, persona dialogue, and role-play characters, listed by hand under `anchors` in `config.json`. They serve two purposes:

1. **기준점 / Orientation**: 기간 설정과 관계없이 그래프에 항상 표시되어(마름모), 새 논문이 어느 연구 계열 가까이에 있는지 보여 줍니다. / Always shown on the graph (diamonds), regardless of the time window, so you can see which line of work a new paper sits next to.
2. **관련성 기준 / Relevance scoring**: 논문의 관련성은 anchor 논문들과의 임베딩 유사도 최댓값입니다. 수집 단계에서 주제와 무관한 논문을 거르고, 키워드는 없지만 관련 있는 논문을 찾는 데 쓰입니다. / A paper's relevance is its highest embedding similarity to any anchor. Collection uses it to drop off-topic papers and to pick up related papers that miss the keywords.

현재 anchor: 메모리 — A-MEM, LongMemEval, LoCoMo / 페르소나 대화 — PersonaChat, MSC (Beyond Goldfish Memory), PersonaGym, Instruction (In)Stability(페르소나 드리프트) / 롤플레이 — Character-LLM, RoleLLM, CharacterEval, InCharacter, Ditto(Superpositions of All Characters), Role-Playing Language Agents survey. ID를 추가하거나 빼면 다음 실행 때 반영되며, 관련성 기준도 함께 바뀝니다.
Current anchors are listed above. Adding or removing an ID takes effect on the next run and also shifts what counts as relevant.

## 논문 그래프 / Paper graph
`graph.py`가 GitHub Actions에서 제목+abstract 임베딩(`BAAI/bge-small-en-v1.5`, CPU)을 계산하고, 코사인 유사도 상위 k개 이웃, k-means 주제 클러스터, 벤치마크 태그를 `docs/graph.json`에 저장합니다. 임베딩은 `data/embeddings/<YYMM>.npz`(arXiv ID 월별, float16)에 캐시되어 새 논문만 계산하고, 매일 바뀌는 파일은 이번 달 것 하나뿐입니다.
On GitHub Actions, `graph.py` embeds each title+abstract (`BAAI/bge-small-en-v1.5`, CPU) and writes top-k cosine neighbors, k-means topic clusters, and benchmark tags to `docs/graph.json`. Embeddings are cached per arXiv month in `data/embeddings/<YYMM>.npz` (float16), so only new papers are embedded and only the current month's file changes each day.

- 키워드 / 주제 클러스터 색상 전환. 클러스터 라벨은 클러스터 내 특징적인 TF-IDF 단어 / Color by keyword or topic cluster; cluster labels are their most distinctive TF-IDF terms
- 벤치마크 칩으로 해당 벤치마크를 언급한 논문 강조 / Benchmark chips highlight papers that mention them
- 마름모 노드는 anchor(기준) 논문 / Diamond nodes are anchor papers
- 기본은 최근 `graph_window_days`일 + anchor만 표시, All time으로 전체 보기. 유사 논문 목록은 항상 전체 기준 / Shows the last `graph_window_days` days plus anchors by default; All time shows everything. Similar-paper lists always search all papers
- 논문당 연결 수 1–5, 키워드 그룹 표시·숨김 / 1–5 links per paper; toggle keyword groups
- 최근 7일 추가 논문은 빨간 테두리 표시, 개요 패널에 New this week 목록 / Papers added in the last 7 days are outlined in red and listed under New this week
- 클러스터 번호는 이전 실행 결과에서 이어서 계산해 매일 크게 바뀌지 않음 / Cluster numbering continues from the previous run, so it stays stable day to day
- 노드 클릭 시 abstract, 주제, 태그, 유사 논문 목록 / Select a node to see its abstract, topic, tags, and most similar papers
- 개요 패널의 주간 키워드별 논문 수 차트(표 보기 포함) / Weekly papers-per-keyword chart (with a table view) in the overview panel
- GitHub Pages 설정 / Pages setup: Settings → Pages → Branch `main` / `/docs`

## 요약 / Summaries
Claude Code 세션에서 "papers/ 이번 주 논문 요약해줘"처럼 요청하면 요약할 수 있습니다.
Ask a Claude Code session, for example "summarize this week's papers in papers/", to get summaries.
