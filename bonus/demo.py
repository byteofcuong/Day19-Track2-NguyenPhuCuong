"""Five-query demo of HybridMemoryAgent — `python bonus/demo.py`.

Each query is chosen to stress a *different* retrieval path, so the output
shows which store did the work rather than just that something came back:

  1  vector only          literal recall of a stored memory
  2  profile-driven       question names no topic; only `topic_affinity` can rank
  3  fresh activity       `queries_last_hour` from the streaming-cadence view
  4  paraphrase           no shared words with the memory — embeddings or nothing
  5  mixed                lexical + semantic + profile all contribute

Prerequisite: NB4 must have run (`feast apply` + `materialize-incremental`), so
`app/feast_repo/online_store.db` holds a profile for u_001. Without it the agent
still runs and prints the degraded banner instead of profile lines.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bonus.agent import HybridMemoryAgent  # noqa: E402

USER = "u_001"          # NB4 materialises this user with topic_affinity=cloud
OTHER = "u_042"         # second tenant, to show isolation is not decorative

# (text, topic) — topic mirrors what a classifier would tag a session with.
SEED_MEMORIES: list[tuple[str, str]] = [
    ("Đọc bài về Kubernetes HPA: scale pod theo CPU và custom metric", "cloud"),
    ("Ghi chú: dùng Karpenter thay Cluster Autoscaler cho node group co giãn", "cloud"),
    ("Đọc về chi phí egress giữa các region trên AWS, tối ưu bằng VPC endpoint", "cloud"),
    ("Bài về pgvector: đánh index HNSW cho bảng embedding trong Postgres", "database"),
    ("Ghi chú: partition bảng log theo ngày để query 7 ngày gần nhất nhanh hơn", "database"),
    ("Đọc về bảo mật OAuth2 PKCE cho ứng dụng mobile, tránh lộ client secret", "security"),
    ("Ghi chú bảo mật: rotate JWT signing key mỗi 90 ngày, giữ 2 key song song", "security"),
    ("Bài về RRF: hợp nhất BM25 và vector bằng 1/(60+rank)", "ai_ml"),
    ("Đọc về quantise embedding xuống int8, giảm 4 lần bộ nhớ index", "ai_ml"),
    ("Ghi chú: CI chạy pytest song song 4 worker, giảm build từ 12 xuống 4 phút", "devops"),
]

QUERIES: list[tuple[str, str]] = [
    ("vector only",
     "tôi đã đọc gì về Kubernetes?"),
    ("profile-driven",
     "gợi ý cho tôi đọc tiếp cái gì?"),
    ("fresh activity",
     "dạo này tôi đang tập trung vào việc gì?"),
    ("paraphrase",
     "tài liệu nào nói về việc mở rộng hạ tầng theo nhu cầu?"),
    ("mixed",
     "tóm tắt giúp tôi phần bảo mật trên đám mây"),
]


def main() -> int:
    agent = HybridMemoryAgent()

    for text, topic in SEED_MEMORIES:
        agent.remember(text, user_id=USER, topic=topic)
    # A second user with an overlapping memory: if isolation were done with a
    # post-filter, this is the row that would leak into u_001's context.
    agent.remember("Bí mật nội bộ u_042: cụm Kubernetes production tên là atlas-prod",
                   user_id=OTHER, topic="cloud")

    print(f"seeded {len(SEED_MEMORIES)} memories for {USER}, 1 for {OTHER}\n")

    for i, (label, q) in enumerate(QUERIES, 1):
        print("=" * 72)
        print(f"[{i}/5] {label}")
        print("=" * 72)
        print(agent.recall(q, user_id=USER))
        print()

    # Isolation check — asserted, not merely printed, so a regression fails loud.
    ctx = agent.recall("cụm Kubernetes production tên là gì?", user_id=USER)
    assert "atlas-prod" not in ctx, "TENANT LEAK: u_042 memory reached u_001"
    print("=" * 72)
    print("isolation: u_042's memory never entered u_001's context — OK")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
