# Reflection - Lab 19

**Tên:** Nguyễn Phú Cường
**Cohort:** A20-K3
**Path đã chạy:** lite (Qdrant in-memory + SQLite Feast + fastembed/bge-small 384d)

---

## Câu hỏi (≤ 200 chữ)

> Trên golden set 50 queries, mode nào thắng ở loại query nào (`exact` /
> `paraphrase` / `mixed`), và tại sao? Khi nào bạn **không** dùng hybrid
> (i.e. khi nào pure BM25 hoặc pure vector là lựa chọn đúng)?

| type | n | kw | sem | **hyb** |
|---|---|---|---|---|
| exact | 15 | 96,7% | 88,7% | **96,7%** |
| paraphrase | 15 | **33,3%** | 24,0% | 32,0% |
| mixed | 20 | 97,0% | 98,5% | **100,0%** |
| **trung bình** | 50 | 77,8% | 73,2% | **78,6%** |

`exact`: BM25 thắng - query trùng nguyên văn tiêu đề, khớp token là tín hiệu
mạnh nhất, hybrid chỉ hoà chứ không vượt được. `mixed`: hybrid thắng tuyệt đối
(100%) vì hai retriever sai khác chỗ nhau, RRF `1/(60+rank)` cộng dồn được
đồng thuận. `paraphrase`: **semantic thua cả BM25** (24,0% vs 33,3%) - ngược
trực giác, và nguyên nhân là `bge-small-en-v1.5` là mô hình tiếng Anh đang phải
nhúng câu hỏi tiếng Việt. Đây không phải khuyết điểm của vector search mà là
chọn sai mô hình; cách chữa là `EMBEDDING_BACKEND=bge-m3` rồi index lại

**Khi nào không dùng hybrid:** khi truy vấn là định danh - mã đơn, SKU, error
code, tên biến. Ở đó chỉ khớp chính xác mới đúng, và một láng giềng ngữ nghĩa
lọt vào là kết quả *sai*, không phải kết quả kém. Dùng BM25 thuần, và nó còn rẻ
hơn 9 lần: NB3 đo keyword P99 = 1,4 ms so với hybrid 12,1 ms. Chiều ngược lại,
vector thuần hợp lý khi corpus đa ngữ và người dùng không bao giờ gõ đúng thuật
ngữ trong tài liệu - lúc đó nhánh BM25 chỉ thêm nhiễu


---

## Điều ngạc nhiên nhất khi làm lab này

**Đã đo rồi không có nghĩa là đã tìm ra nguyên nhân**

NB3 lần chạy đầu báo hybrid P99 = 80,8 ms, trượt ngưỡng 50 ms. Phân rã ngân
sách cho thấy **95% nằm ở một forward pass ONNX của bge-small** (53 ms/query),
còn BM25 + ANN cộng lại chưa tới 5%. Việc đó đã cứu tôi khỏi sai lầm hiển
nhiên: phản xạ đầu tiên là hạ `depth` của RRF, mà làm vậy chỉ giảm chất lượng
fusion chứ gần như không giảm latency

Nhưng rồi tôi dừng ở kết luận "CPU/onnxruntime máy này chậm, chấp nhận thôi".
Nó khớp với mọi dữ liệu tôi có - vặn `intra_op_num_threads` (1/4/8) và
`graph_optimization_level` đều không đổi bậc độ lớn - và vẫn **sai**. Câu hỏi
tôi lẽ ra phải hỏi sớm hơn: 4 ms là con số hợp lý cho bge-small ở 18 token;
46 ms thì vô lý - vậy *giả định nào của mình đang sai?*

Bisect theo phiên bản `onnxruntime` trên chính file model đó:

| 1.22.1 | 1.25.1 | 1.26.0 | 1.27.0 | 1.28.0 | 1.29.0 |
|---|---|---|---|---|---|
| 11,1 ms | 4,4 ms | 4,2 ms | 4,3 ms | **4,0 ms** | **45,9 ms** |

Hồi quy ~10× chỉ ở 1.29.0. `fastembed` kéo `onnxruntime` vào transitive và
không chặn trần phiên bản, nên `pip install` mặc định lấy đúng bản hỏng. Pin
`onnxruntime>=1.22,<1.29`: hybrid P99 **80,8 → 12,8 ms**, index 1000 doc
**196 s → 21 s**, `make test` **26 s → 3,6 s**. Không một dòng code retrieval
nào thay đổi

Điều ngạc nhiên thứ hai, ở NB5: post-filter tụt recall về **0,00** khi filter
còn chọn 3,8% corpus, trong khi filtered-ANN giữ **1,00**. Over-fetch phải kéo
`fetch_k` lên 500 (50% corpus) mới cứu được recall - tức là "lấy dư rồi lọc"
không phải một cách chỉnh, mà là bỏ hẳn index

---

## Bonus challenge

- [x] Đã làm bonus (xem [`bonus/`](../bonus/)) - `HybridMemoryAgent`: episodic
      memory trên Qdrant + hồ sơ ổn định trên Feast, weighted RRF 3 nhánh
      (lexical / semantic / affinity), cách ly người dùng bằng filtered-ANN có
      `assert` kiểm chứng. `python bonus/demo.py` chạy 5 query và exit 0.
      Kiến trúc + 3 tradeoff + phương án đã bỏ: [`bonus/ARCHITECTURE.md`](../bonus/ARCHITECTURE.md).
- [ ] Pair work với: _(làm cá nhân)_

---

