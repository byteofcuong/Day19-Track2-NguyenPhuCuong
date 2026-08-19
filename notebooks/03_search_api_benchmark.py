# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # NB3 — FastAPI `/search` Endpoint + Latency Benchmark
#
# **Stack:** FastAPI + uvicorn + httpx (client). Searcher từ `app/search.py`.
# Maps to slide §7 (Production Patterns) + deliverable bullets 1, 4.
#
# > Mục tiêu: bọc `Searcher` thành REST API, đo P50/P95/P99 latency, đảm bảo
# > P99 < 50 ms cho hybrid mode (rubric threshold).

# %%
import _setup  # noqa: F401
import statistics
import subprocess
import time
from pathlib import Path

import httpx

# %% [markdown]
# ## 1. Khởi động API server (background)
#
# Trong production thực tế, bạn sẽ chạy `make api` ở terminal riêng. Notebook
# này khởi động uvicorn ở background subprocess và đợi `/healthz` trả ready.

# %%
ROOT = Path(_setup.__file__).resolve().parent.parent
proc = subprocess.Popen(
    ["uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000",
     "--log-level", "warning"],
    cwd=str(ROOT),
)

# Địa chỉ literal, KHÔNG dùng "localhost". Trên Windows `localhost` phân giải
# ra ::1 trước; uvicorn chỉ bind IPv4 nên mỗi request tốn ~2 s chờ IPv6 fail
# rồi mới retry 127.0.0.1 — đủ để biến bench 300 call thành 12 phút và bơm
# wall-clock lên gấp 40 lần server-side.
URL = "http://127.0.0.1:8000"

# Một Client dùng chung => keep-alive. Mỗi `httpx.get(...)` rời rạc mở TCP
# connection mới (~150 ms handshake trên Windows loopback); đo cái đó là đo
# connection setup, không phải đo search.
client = httpx.Client(base_url=URL, timeout=30.0)

# Đợi server up + warm (Searcher.from_corpus loads embeddings + indexes 1000 docs)
for _ in range(180):
    try:
        r = client.get("/healthz", timeout=2.0)
        if r.status_code == 200 and r.json().get("ready"):
            break
    except httpx.HTTPError:
        pass
    time.sleep(1)
else:
    raise RuntimeError("API didn't become ready within 180s")

print(client.get("/healthz").json())

# %% [markdown]
# ## 2. Single query — kiểm tra response shape

# %%
r = client.get("/search", params={"q": "cloud computing tự động mở rộng", "mode": "hybrid"})
r.raise_for_status()
body = r.json()
print(f"latency_ms: {body['latency_ms']:.1f}")
print(f"top-3 hits:")
for h in body["hits"][:3]:
    print(f"  {h['doc_id']:>14}  score={h['score']:.4f}  {h['title']}")

# %% [markdown]
# ## 3. TODO — Latency benchmark (100 queries × 3 modes)
#
# Dùng 50 golden queries × 2 reps = 100 calls/mode. Ghi nhận latency từ
# `body["latency_ms"]` (server-side, đã trừ network) HOẶC từ wall-clock httpx
# (bao gồm network) — note: rubric assert P99 < 50ms áp dụng cho server-side.
#
# Output: bảng P50/P95/P99 cho 3 mode.

# %%
import json

DATA = ROOT / "data"
golden = [json.loads(l) for l in (DATA / "golden_set.jsonl").open(encoding="utf-8")]


def percentile(values: list[float], p: float) -> float:
    n = len(values)
    if n == 0:
        return 0.0
    return sorted(values)[min(int(n * p), n - 1)]


def benchmark_mode(mode: str, reps: int = 2, warmup: int = 10) -> dict[str, float]:
    # Warm-up KHÔNG được tính vào phân phối: request đầu của mỗi mode trả tiền
    # cho tokenizer/ONNX graph lazy-init. Trộn nó vào 100 mẫu thì nó *thành*
    # P99 và bảng đo cold start chứ không đo steady state.
    for q in golden[:warmup]:
        client.get("/search", params={"q": q["query"], "mode": mode})

    server_latencies: list[float] = []
    wall_latencies: list[float] = []
    for _ in range(reps):
        for q in golden:
            t0 = time.perf_counter()
            r = client.get("/search", params={"q": q["query"], "mode": mode})
            wall_latencies.append((time.perf_counter() - t0) * 1000)
            server_latencies.append(r.json()["latency_ms"])
    return {
        "p50_server": percentile(server_latencies, 0.50),
        "p95_server": percentile(server_latencies, 0.95),
        "p99_server": percentile(server_latencies, 0.99),
        "p99_wall":   percentile(wall_latencies, 0.99),
    }


print(f"  {'mode':10}  {'P50':>7}  {'P95':>7}  {'P99':>7}  {'P99(wall)':>9}")
results = {}
for mode in ("keyword", "semantic", "hybrid"):
    res = benchmark_mode(mode)
    results[mode] = res
    print(f"  {mode:10}  {res['p50_server']:>5.1f}ms  {res['p95_server']:>5.1f}ms  "
          f"{res['p99_server']:>5.1f}ms  {res['p99_wall']:>7.1f}ms")

# %% [markdown]
# ## 4. Rubric assertion — hybrid P99 server-side < 50ms

# %%
hybrid_p99 = results["hybrid"]["p99_server"]
print(f"Hybrid P99 server-side: {hybrid_p99:.1f}ms")
if hybrid_p99 < 50:
    print(f"PASS — hybrid P99 < 50ms ({hybrid_p99:.1f}ms)")
else:
    print(f"WARN — hybrid P99 >= 50ms ({hybrid_p99:.1f}ms)")
    print("  Warm-up đã chạy (10 query/mode) nên đây KHÔNG phải cold start.")
    print("  Xem §4b: phân rã latency để biết thời gian thật sự đi đâu.")

# %% [markdown]
# ## 4b. Ngân sách latency đi đâu? (chẩn đoán, không phải đoán)
#
# Một con số P99 trần trụi không nói được gì. Trước khi "tối ưu", đo xem mỗi
# tầng tốn bao nhiêu. Ở đây gọi thẳng `Searcher` trong-process nên loại bỏ
# HTTP/serialization khỏi phép đo — cái còn lại đúng là công việc retrieval.

# %%
from app.search import Searcher

searcher = Searcher.from_corpus(ROOT / "data" / "corpus_vn.jsonl")
probe = [g["query"] for g in golden]

for q in probe[:10]:                       # warm the ONNX graph
    searcher.search(q, mode="hybrid")


def stage_p50(fn) -> float:
    t = []
    for q in probe:
        t0 = time.perf_counter()
        fn(q)
        t.append((time.perf_counter() - t0) * 1000)
    return statistics.median(t)


embed_ms = stage_p50(lambda q: next(searcher.embedder.embed([q])))
bm25_ms = stage_p50(lambda q: searcher._search_keyword(q, 50))
ann_ms = stage_p50(lambda q: searcher._search_semantic(q, 50)) - embed_ms
total_ms = stage_p50(lambda q: searcher.search(q, mode="hybrid"))

print(f"  {'stage':26} {'P50':>8}   share")
for label, ms in [("query embedding (ONNX)", embed_ms),
                  ("BM25 scan (1000 docs)", bm25_ms),
                  ("Qdrant ANN lookup", max(ann_ms, 0.0)),
                  ("RRF fusion + overhead",
                   max(total_ms - embed_ms - bm25_ms - max(ann_ms, 0.0), 0.0))]:
    print(f"  {label:26} {ms:>6.1f}ms   {ms / total_ms:>5.0%}")
print(f"  {'-' * 26} {'-' * 8}")
print(f"  {'hybrid total':26} {total_ms:>6.1f}ms")

# %% [markdown]
# ### Đọc bảng trên — và cái bẫy suýt mắc phải
#
# Ở trạng thái hiện tại ngân sách chia tương đối đều: embedding ~46%, ANN ~32%,
# BM25 ~11%. Không tầng nào một mình quyết định, nên tối ưu thêm ở đây là công
# việc lợi ít — P99 đã ở 12,8 ms, dưới ngưỡng 50 ms gần bốn lần.
#
# **Nhưng lần chạy đầu tiên của notebook này báo P99 = 80,8 ms, với 95% ngân
# sách nằm ở tầng embedding (53 ms/query).** Kết luận lúc đó là "CPU/onnxruntime
# của máy này chậm, chấp nhận thôi" — vặn `intra_op_num_threads` (1/4/8) và
# `graph_optimization_level` đều không đổi bậc độ lớn, nên nó *nghe* rất hợp lý.
#
# Nó sai. Đo `onnxruntime.InferenceSession` trần trên chính file
# `model_optimized.onnx`, chuỗi 18 token, rồi **bisect theo phiên bản**:
#
# | onnxruntime | 1 forward pass (P50) |
# |---|---|
# | 1.22.1 | 11,1 ms |
# | 1.25.1 | 4,4 ms |
# | 1.26.0 | 4,2 ms |
# | 1.27.0 | 4,3 ms |
# | 1.28.0 | **4,0 ms** |
# | 1.29.0 | **45,9 ms** ← hồi quy ~10× |
#
# `fastembed` kéo `onnxruntime` vào theo kiểu transitive và **không chặn trần
# phiên bản**, nên `pip install` mặc định lấy 1.29.0. Đó không phải giới hạn
# phần cứng — đó là một bản hồi quy hiệu năng, và nó một mình quyết định
# notebook này đạt hay trượt ngưỡng 50 ms. `requirements.txt` giờ pin
# `onnxruntime>=1.22,<1.29`, và bảng P50/P95/P99 ở §3 là số đo **sau** khi
# pin: hybrid P99 đi từ **80,8 ms xuống 12,8 ms**.
#
# > Bài học đắt hơn cả con số P99: **"đã đo rồi" chưa phải là "đã tìm ra
# > nguyên nhân"**. Phân rã tầng chỉ ra *ở đâu* tốn thời gian, nó không nói
# > *tại sao*. Dừng lại ở "phần cứng chậm" là một lời giải thích tự vỗ về —
# > nó khớp với mọi dữ liệu đang có, và vẫn sai. Câu hỏi cứu được 10 điểm là
# > "4 ms là con số *hợp lý* cho bge-small ở 18 token; 46 ms thì không —
# > vậy giả định nào của mình đang sai?"
#
# Ba đòn bẩy còn lại, nếu tầng embedding vẫn là nút thắt sau khi đã pin đúng:
#
# | Đòn bẩy | Ảnh hưởng | Đánh đổi |
# |---|---|---|
# | Cache embedding của query (LRU / semantic cache NB7) | bỏ hẳn chi phí cho query lặp | chỉ cứu được traffic có phần đuôi lặp lại |
# | Model nhỏ hơn (MiniLM-L6, 6 lớp) | ~2× nhanh | tụt chất lượng — mà NB2 cho thấy bge-small **đã** yếu với paraphrase tiếng Việt |
# | Batch nhiều query / GPU | rẻ hơn nhiều mỗi doc khi batch lớn | chỉ hợp offline hoặc khi server có micro-batching |

# %% [markdown]
# ## 5. Cleanup — stop the API server

# %%
client.close()
proc.terminate()
proc.wait(timeout=5)
print("API server stopped")

# %% [markdown]
# ## Deliverable evidence
#
# 1. Output cell 2: 1 single hybrid query response with `top-3 hits`.
# 2. Output cell 3: latency table P50/P95/P99 for keyword/semantic/hybrid.
# 3. Output cell 4: hybrid P99 < 50ms PASS.
#
# ---
#
# ## Vibe-coding callout
#
# **Delegate freely:** the FastAPI scaffolding (route definition, Pydantic
# response model, lifespan handler). AI generates this perfectly given the
# spec "GET /search?q=str&mode=Literal[...] returning SearchResponse with
# latency_ms field". `app/main.py` is exactly that pattern — review the diff,
# don't write it from scratch.
#
# **Think hard yourself:** *what to measure*. Server-side latency vs wall-clock
# vs client-side. P50 vs P95 vs P99. Cold vs warm. Single user vs concurrent.
# These are *judgement* decisions: nếu rubric chỉ check P99, optimization sẽ
# hướng vào tail latency, không phải mean. Đừng nhờ AI quyết định metric —
# chỉ nhờ implement metric đã chọn.
