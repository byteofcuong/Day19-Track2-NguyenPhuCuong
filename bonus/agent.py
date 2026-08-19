"""HybridMemoryAgent — episodic memory (vector) + stable profile (feature store).

Bonus challenge for Day 19. `bonus/ARCHITECTURE.md` holds the decisions and the
tradeoffs; this file is the runnable POC of those decisions.

    from bonus.agent import HybridMemoryAgent
    agent = HybridMemoryAgent()
    agent.remember("Đọc xong bài về Kubernetes HPA", user_id="u_001", topic="cloud")
    print(agent.recall("tôi đã đọc gì về Kubernetes?", user_id="u_001"))

Three stores, three different reasons:

  Qdrant  episodic memory. Grows every session, re-indexed continuously.
  BM25    the same memories, lexically. Vietnamese proper nouns, error codes
          and vi/en code-switching ("deploy con service này") are exactly where
          a 384-dim English-leaning embedding is weakest — NB2 measured that
          (semantic 24.0% vs BM25 33.3% Precision@10 on paraphrase queries).
  Feast   the stable profile + recent-activity counters. Different write
          cadence, different read path (point lookup by key), so it does not
          belong in the vector store — ARCHITECTURE.md §Rejected.

Isolation: every memory carries `user_id` in its Qdrant payload and every read
goes through a `must` filter on it — filtered-ANN, not post-filter. NB5 showed
post-filter recall collapsing to 0.00 once a filter gets selective, and a
per-user filter over a shared index is the most selective filter there is.
"""
from __future__ import annotations

import re
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.embeddings import Embedder  # noqa: E402

COLLECTION = "bonus_episodic_memory"
FEAST_DIR = ROOT / "app" / "feast_repo"

# Whitespace + punctuation split, keeping Vietnamese diacritics as word chars.
# Deliberately NOT pyvi/underthesea — see ARCHITECTURE.md §Vietnamese context
# for what that costs and why the POC accepts the cost.
_TOKEN_RE = re.compile(r"[^\wÀ-ỹ]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.split(text.lower()) if t]


@dataclass
class Memory:
    mem_id: int
    user_id: str
    text: str
    topic: str | None
    ts: float


@dataclass
class RecalledContext:
    """`recall()` returns the rendered string; this is the same content before
    rendering, so a caller can build its own prompt shape instead."""
    user_id: str
    query: str
    profile: dict[str, Any]
    memories: list[Memory]
    sources: dict[int, list[str]] = field(default_factory=dict)
    degraded: str | None = None


class HybridMemoryAgent:
    """remember() writes episodic memory; recall() assembles LLM context."""

    def __init__(
        self,
        feast_repo: Path | str | None = FEAST_DIR,
        top_k: int = 4,
        rrf_k: int = 60,
        depth: int = 20,
        affinity_weight: float = 0.35,
    ) -> None:
        self.embedder = Embedder()
        self.top_k, self.rrf_k, self.depth = top_k, rrf_k, depth
        # Weighted RRF. Plain RRF treats every ranker as equally trustworthy,
        # and affinity is not: it answers "what does this user usually care
        # about", not "what did they ask". At weight 1.0 a user with only three
        # cloud memories gets those same three back for every question,
        # including one about security. 0.35 keeps affinity as a tie-breaker —
        # decisive between two similarly relevant memories, never able to
        # outvote a direct lexical or semantic hit on its own.
        self.affinity_weight = affinity_weight

        self.client = QdrantClient(":memory:")
        self.client.create_collection(
            collection_name=COLLECTION,
            vectors_config=models.VectorParams(
                size=self.embedder.dim, distance=models.Distance.COSINE),
        )
        # Payload index on user_id — on a Qdrant *server* this is what lets the
        # tenant filter ride inside the ANN traversal instead of being applied
        # after it (NB5). Local/in-memory Qdrant warns that the index is a
        # no-op; the filter itself is still enforced by the engine during
        # search, so isolation holds either way. Declaring it here means the
        # POC ports to a server without a code change.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.client.create_payload_index(
                collection_name=COLLECTION,
                field_name="user_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

        self._memories: dict[int, Memory] = {}
        self._by_user: dict[str, list[int]] = {}
        self._bm25: dict[str, tuple[BM25Okapi, list[int]]] = {}   # per-user, lazy
        self._next_id = 0

        self._store = self._open_feature_store(feast_repo)
        self.degraded: str | None = None if self._store else "feature store unavailable"

    # ── feature store ───────────────────────────────────────────────────
    def _open_feature_store(self, repo: Path | str | None):
        """A missing registry must not take the agent down.

        Episodic recall is the load-bearing half; the profile only personalises
        it. A POC that dies because `feast apply` has not run yet would hide the
        part that does work.
        """
        if repo is None:
            return None
        try:
            from feast import FeatureStore
            return FeatureStore(repo_path=str(repo))
        except Exception as exc:                              # noqa: BLE001
            print(f"[warn] Feast disabled ({type(exc).__name__}: {exc}). "
                  f"Run NB4 to populate {repo}.", file=sys.stderr)
            return None

    def profile(self, user_id: str) -> dict[str, Any]:
        """Point lookup — NB4 measured P99 = 0.46 ms against the SQLite online store."""
        if self._store is None:
            return {}
        try:
            raw = self._store.get_online_features(
                features=[
                    "user_profile_features:topic_affinity",
                    "user_profile_features:preferred_language",
                    "user_profile_features:reading_speed_wpm",
                    "query_velocity_features:queries_last_hour",
                    "query_velocity_features:distinct_topics_24h",
                ],
                entity_rows=[{"user_id": user_id}],
            ).to_dict()
        except Exception as exc:                              # noqa: BLE001
            self.degraded = f"{type(exc).__name__}: {exc}"
            return {}
        return {k: v[0] for k, v in raw.items() if k != "user_id"}

    # ── write path ──────────────────────────────────────────────────────
    def remember(self, text: str, user_id: str = "u_001",
                 topic: str | None = None, ts: float | None = None) -> None:
        """Add one episodic memory for this user.

        One memory == one chunk. Chunking happens on write, not on read, because
        the caller is the only one that knows where a turn actually ended
        (ARCHITECTURE.md §Decision 1).
        """
        if not text.strip():
            raise ValueError("refusing to store an empty memory")
        mem = Memory(self._next_id, user_id, text.strip(), topic, ts or time.time())
        self._next_id += 1

        vec = np.asarray(next(self.embedder.embed([mem.text])), dtype=np.float32)
        self.client.upsert(
            collection_name=COLLECTION,
            points=[models.PointStruct(
                id=mem.mem_id,
                vector=vec.tolist(),
                payload={"user_id": user_id, "topic": topic,
                         "ts": mem.ts, "text": mem.text},
            )],
        )
        self._memories[mem.mem_id] = mem
        self._by_user.setdefault(user_id, []).append(mem.mem_id)
        self._bm25.pop(user_id, None)      # invalidate; rebuilt on next recall

    # ── read path ───────────────────────────────────────────────────────
    def _bm25_for(self, user_id: str) -> tuple[BM25Okapi, list[int]] | None:
        ids = self._by_user.get(user_id, [])
        if not ids:
            return None
        if user_id not in self._bm25:
            corpus = [tokenize(self._memories[i].text) for i in ids]
            self._bm25[user_id] = (BM25Okapi(corpus), list(ids))
        return self._bm25[user_id]

    # A memory that matched only on "đọc" / "về" is not a lexical hit, it is
    # noise -- and with whitespace tokenisation and no Vietnamese stopword list,
    # score > 0 catches almost the entire corpus. Keep only what clears a share
    # of the best score, so `lex` in the provenance column means something.
    LEX_FLOOR = 0.25

    def _lexical(self, user_id: str, query: str, depth: int) -> list[int]:
        built = self._bm25_for(user_id)
        if built is None:
            return []
        bm25, ids = built
        scores = bm25.get_scores(tokenize(query))
        best = float(scores.max()) if len(scores) else 0.0
        if best <= 0:
            return []
        floor = best * self.LEX_FLOOR
        order = sorted(range(len(ids)), key=lambda i: -scores[i])[:depth]
        return [ids[i] for i in order if scores[i] >= floor]

    def _semantic(self, user_id: str, query: str, depth: int) -> list[int]:
        qv = np.asarray(next(self.embedder.embed([query])), dtype=np.float32)
        pts = self.client.query_points(
            collection_name=COLLECTION,
            query=qv.tolist(),
            query_filter=models.Filter(must=[models.FieldCondition(
                key="user_id", match=models.MatchValue(value=user_id))]),
            limit=depth,
        ).points
        return [int(p.id) for p in pts]

    def _affinity(self, user_id: str, profile: dict) -> list[int]:
        """Third RRF ranker: the user's own topic affinity, most recent first.

        This is the feature store paying rent. Without it, "gợi ý đọc tiếp" has
        nothing to rank by — the question itself names no topic at all.
        """
        aff = profile.get("topic_affinity")
        if not aff:
            return []
        ids = [i for i in self._by_user.get(user_id, [])
               if self._memories[i].topic == aff]
        return sorted(ids, key=lambda i: -self._memories[i].ts)

    def retrieve(self, query: str, user_id: str = "u_001",
                 profile: dict | None = None) -> tuple[list[Memory], dict[int, list[str]]]:
        """Weighted RRF over three rankers: lexical, semantic, affinity.

        Returns the top-K memories plus, for each, which rankers put it there —
        provenance is what makes a fusion score debuggable instead of magic.
        """
        profile = self.profile(user_id) if profile is None else profile
        rankers: dict[str, tuple[list[int], float]] = {
            "lex": (self._lexical(user_id, query, self.depth), 1.0),
            "sem": (self._semantic(user_id, query, self.depth), 1.0),
            "aff": (self._affinity(user_id, profile)[: self.depth], self.affinity_weight),
        }
        fused: dict[int, float] = {}
        sources: dict[int, list[str]] = {}
        for name, (ids, weight) in rankers.items():
            for rank, mem_id in enumerate(ids, start=1):
                fused[mem_id] = fused.get(mem_id, 0.0) + weight / (self.rrf_k + rank)
                sources.setdefault(mem_id, []).append(name)

        ordered = [i for i, _ in sorted(fused.items(), key=lambda kv: -kv[1])][: self.top_k]
        return [self._memories[i] for i in ordered], {i: sources[i] for i in ordered}

    def recall(self, query: str, user_id: str = "u_001") -> str:
        """Retrieve top-K memories + profile → the assembled LLM context."""
        profile = self.profile(user_id)
        memories, sources = self.retrieve(query, user_id, profile)
        return render(RecalledContext(user_id, query, profile, memories,
                                      sources, self.degraded))


def render(ctx: RecalledContext) -> str:
    """Profile first, then evidence, then the question.

    The order is a decision, not formatting. The profile is short and stable, so
    it sits at the front where a prompt-cache prefix can cover it; the retrieved
    memories change every turn and belong after it.
    """
    lines = [f"### Người dùng: {ctx.user_id}"]
    if ctx.profile:
        lines += [
            f"- ngôn ngữ ưu tiên : {ctx.profile.get('preferred_language')}",
            f"- chủ đề quan tâm  : {ctx.profile.get('topic_affinity')}",
            f"- tốc độ đọc       : {ctx.profile.get('reading_speed_wpm')} wpm",
            f"- hoạt động gần đây: {ctx.profile.get('queries_last_hour')} truy vấn/giờ, "
            f"{ctx.profile.get('distinct_topics_24h')} chủ đề/24h",
        ]
    else:
        lines.append(f"- (không có profile — {ctx.degraded or 'chưa materialize'})")

    lines.append("")
    lines.append("### Ký ức liên quan")
    if not ctx.memories:
        lines.append("- (chưa có ký ức nào khớp)")
    for i, m in enumerate(ctx.memories, 1):
        tag = f"[{m.topic}] " if m.topic else ""
        via = "+".join(ctx.sources.get(m.mem_id, []))
        lines.append(f"{i}. ({via:<11}) {tag}{m.text}")

    lines += ["", "### Câu hỏi", ctx.query]
    return "\n".join(lines)
