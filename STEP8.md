# Báo Cáo Phân Tích Kết Quả Benchmark & Kiến Trúc Hệ Thống Memory (Lab 17)

**Học viên:** Nguyễn Mạnh Cường  
**Mã số sinh viên (MSSV):** 2A202602650  
**Lớp / Cohort:** K4 — Track 3: AI Agent Memory Systems  
**Repo:** [K4-DAY17-NguyenManhCuong-2A202602650](https://github.com/cuong-cpu21/K4-DAY17-NguyenManhCuong-2A202602650)

---

## 1. Kết Quả Đo Lường Thực Tế (Benchmark Results)

Toàn bộ benchmark được thực hiện độc lập, tất định (deterministic) và có thể tái lập 100% bằng lệnh `python src/benchmark.py`.

### 1.1. Bảng 1: Standard Benchmark (`data/conversations.json`)
*Tập dữ liệu gồm 10 hội thoại (~10 lượt/hội thoại) của người dùng `dungct`, kiểm tra khả năng nhớ thông tin qua các phiên chat thông thường.*

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | 2,035 | 16,353 | **0.0%** | 0.50 | 0 | 0 |
| **Advanced** | 4,526 | 27,943 | **100.0%** | 1.00 | 289 | 3 |

---

### 1.2. Bảng 2: Long-Context Stress Benchmark (`data/advanced_long_context.json`)
*Tập dữ liệu gồm 1 hội thoại kéo dài 16 lượt với ngữ cảnh rất dài (chứa nhiều tin tức kỹ thuật NASA Artemis III, X-59, WMO, BC Energy) của người dùng `dungct_stress`, được thiết kế để ép lớp compact kích hoạt liên tục.*

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | 323 | 22,700 | **0.0%** | 0.50 | 0 | 0 |
| **Advanced** | 1,265 | **9,273** | **100.0%** | 1.00 | 217 | **26** |

---

## 2. Trả Lời 4 Câu Hỏi Trọng Tâm (Bước 8 - Guide.md)

### Câu hỏi 1: Vì sao Advanced có recall tốt hơn Baseline?
* **Bằng chứng số liệu:**
  * Ở cả hai bảng Standard và Stress, **Cross-session recall** của Advanced đạt **100.0%**, trong khi Baseline hoàn toàn là **0.0%**.
  * Memory growth của Baseline là `0 bytes`, còn Advanced ghi nhận tăng trưởng bền vững (`289 bytes` ở Standard và `217 bytes` ở Stress).
* **Cơ chế trong mã nguồn:**
  * **Baseline Agent** (`agent_baseline.py`) lưu trạng thái phiên làm việc (`SessionState`) gắn chặt theo `thread_id` (`self.sessions[thread_id]`). Khi câu hỏi recall được gửi ở một `recall_thread` mới (như quy ước của benchmark), Baseline đối mặt với một session hoàn toàn trắng và buộc phải trả lời không biết.
  * **Advanced Agent** (`agent_advanced.py`) sở hữu tầng **Persistent Memory** thông qua `UserProfileStore` (`memory_store.py`). Mỗi lượt hội thoại, hàm `extract_profile_updates()` chủ động bóc tách các fact ổn định (tên, nơi ở, nghề nghiệp, đồ uống, phong cách trả lời...) và ghi vào file markdown bền vững `state/profiles/<user_id>/User.md`. Khi bước sang bất kỳ thread mới nào, hàm `_offline_response()` luôn nạp lại hồ sơ này để trả lời đầy đủ, chính xác.

---

### Câu hỏi 2: Vì sao Advanced có thể tốn hơn ở hội thoại ngắn?
* **Bằng chứng số liệu:**
  * Tại **Standard Benchmark** (hội thoại ngắn ~10 lượt), tổng **Prompt tokens processed** của Advanced là **27,943 tokens**, cao hơn đáng kể so với **16,353 tokens** của Baseline (+70.8%).
  * **Agent tokens only** của Advanced cũng cao hơn (4,526 so với 2,035).
* **Cơ chế trong mã nguồn:**
  * Trong mỗi lượt hội thoại, Advanced Agent phải nạp ngữ cảnh tổng hợp:
    $$\text{Prompt Context} = \text{User.md tokens} + \text{Summary tokens} + \text{Recent Messages tokens}$$
  * Ở các hội thoại ngắn (dưới ngưỡng compact `compact_threshold_tokens = 500`), cơ chế nén hầu như chưa phát huy tác dụng (chỉ có 3 lần nén nhẹ), trong khi chi phí "cõng" hồ sơ `User.md` và tóm tắt lặp đi lặp lại qua từng lượt khiến tổng lượng prompt token tích lũy của Advanced cao hơn Baseline.
  * Về phần sinh câu trả lời, Advanced trả lời đầy đủ, có cấu trúc 3 bullet rõ ràng theo đúng sở thích người dùng nên số lượng token sinh ra (`Agent tokens only`) cũng lớn hơn câu phản hồi mặc định ngắn của Baseline.

---

### Câu hỏi 3: Vì sao compact memory có lợi thế ở hội thoại dài?
* **Bằng chứng số liệu:**
  * Tại **Long-Context Stress Benchmark**, tổng **Prompt tokens processed** của Baseline phình to đến **22,700 tokens**, trong khi Advanced chỉ tiêu tốn **9,273 tokens** (giảm tới **59.15%**!).
  * Cột **Compactions** của Advanced ghi nhận **26 lần nén**, trong khi Baseline là 0.
* **Cơ chế trong mã nguồn:**
  * Baseline Agent không có cơ chế nén, do đó qua 16 lượt dài dằng dặc, toàn bộ các đoạn văn bản tin tức dài hàng nghìn ký tự của các lượt trước đều bị nhồi nguyên văn vào prompt của lượt kế tiếp, khiến prompt tokens tăng trưởng tuyến tính theo cấp số cộng: $\mathcal{O}(N^2)$.
  * Ngược lại, `CompactMemoryManager` của Advanced Agent liên tục giám sát ngưỡng token (`threshold_tokens`). Khi vượt ngưỡng, các message cũ lập tức được cô đọng qua `summarize_messages()`, và chỉ giữ lại `keep_messages = 4` tin nhắn gần nhất nguyên văn. Nhờ đó, kích thước prompt context ở mỗi lượt bị chặn trên (bounded), bảo vệ mô hình khỏi hiện tượng tràn ngữ cảnh và cắt giảm phần lớn chi phí token đầu vào.

---

### Câu hỏi 4: File memory tăng trưởng ra sao và rủi ro gì đi kèm?
* **Đặc điểm tăng trưởng:**
  * Dung lượng `User.md` tăng từ 0 lên `289 bytes` (Standard) và `217 bytes` (Stress).
  * Nhờ áp dụng cơ chế **Upsert Fact theo Key** (`- **key**: value`), file chỉ tăng khi có thuộc tính (attribute) mới xuất hiện. Khi người dùng đính chính (correction) thông tin cũ, dung lượng file duy trì ổn định thay vì phình to vô hạn.
* **Rủi ro đi kèm trong hệ thống thực tế:**
  1. **Lưu sai thông tin do nhiễu (Hallucinated/Noisy Facts):** Nếu người dùng chỉ nói đùa, ví dụ "chắc chuyển sang làm Product Manager" hoặc nhắc địa danh tạm thời "vừa bay ra Hà Nội họp 2 ngày", một bộ trích xuất ngây thơ sẽ ghi đè lên fact thật, làm hỏng độ chính xác lâu dài.
  2. **Xung đột thời gian (Temporal Conflict / Stale Memory):** Khi người dùng thay đổi thông tin nhiều lần (Đà Nẵng $\rightarrow$ Huế $\rightarrow$ Đà Nẵng), nếu không có cơ chế versioning hoặc timestamp, agent có thể nhầm lẫn giữa thông tin lịch sử và trạng thái hiện tại.
  3. **Rủi ro bảo mật & Quyền riêng tư (Privacy & PII):** Việc lưu trữ vĩnh viễn dữ liệu người dùng dưới dạng văn bản thuần (`User.md`) đòi hỏi cơ chế mã hóa, phân quyền và tuân thủ các quy định như GDPR (Right to be forgotten).

---

## 3. Các Phần Mở Rộng Kỹ Thuật (Bonus - Mốc 90-100 Điểm)

Để đạt điểm tối đa theo [Rubric.md](Rubric.md), dự án đã tích hợp 4 kỹ thuật nâng cao sau:

### 3.1. Conflict Handling & Correction Strategy (Xử lý xung đột và đính chính)
* **Vấn đề giải quyết:** Người dùng thường xuyên thay đổi thông tin theo thời gian (ví dụ: ban đầu ở Huế, sau đó chuyển vào Đà Nẵng; ban đầu làm Backend, sau đó chuyển sang MLOps). Nếu chỉ nối thêm văn bản (append-only), `User.md` sẽ chứa cả hai thông tin đối lập nhau và agent sẽ trả lời mâu thuẫn.
* **Giải pháp trong code:**
  * Hàm `UserProfileStore.upsert_facts()` phân tích cú pháp cấu trúc markdown thành key-value.
  * Khi có đính chính mới, hệ thống tự động ghi đè giá trị mới nhất vào đúng khóa đó (hoặc sử dụng `edit_text()`), đảm bảo `User.md` luôn là nguồn sự thật duy nhất (Single Source of Truth).
* **Kết quả:** Ở lượt hỏi "nơi ở hiện tại", Advanced trả lời chính xác `Huế` cho `dungct` và `Đà Nẵng` cho `dungct_stress`.
* **Rủi ro phát sinh:** Nếu người dùng chỉ nêu giả định ("nếu mình chuyển sang Đà Nẵng thì sao?"), hệ thống có thể vô tình ghi đè mất thông tin thật nếu không có bước phân tích ngữ cảnh sâu.

### 3.2. Guardrails Lọc Câu Hỏi và Chống Nhiễu Hội Thoại
* **Vấn đề giải quyết:**
  * Trong tập test, người dùng có các lượt chỉ đặt câu hỏi: *"Bạn biết DũngCT là ai không?"*, *"Hiện tại mình đang ở đâu?"*. Nếu không có guardrail, regex trích xuất có thể hiểu nhầm câu hỏi là thông tin khai báo.
  * Tạp âm hội thoại: *"Hà Nội chỉ là nơi mình vừa bay ra họp hai ngày"*, *"đùa với đồng nghiệp hay là chuyển sang product manager"*.
* **Giải pháp trong code:**
  * Hàm `extract_profile_updates()` có bộ lọc câu hỏi `is_question`: nhận diện các mẫu câu hỏi truy vấn và kết thúc bằng dấu `?` mà không có mệnh đề khai báo để bỏ qua.
  * Tích hợp quy tắc lọc phủ định/nhiễu: bỏ qua "Hà Nội" nếu đi kèm ngữ cảnh "họp/bay ra/không phải nơi ở"; bỏ qua "product manager" nếu đi kèm "câu đùa/đùa".
* **Kết quả:** `User.md` hoàn toàn sạch, không bị vấy bẩn bởi các fact giả mạo.
* **Rủi ro phát sinh:** Có thể lọc sót nếu người dùng dùng cấu trúc câu quá phức tạp hoặc tiếng lóng không nằm trong tập mẫu nhận diện.

### 3.3. Cấu Trúc Hóa Dữ Liệu Entity Markdown (Structured Entity Storage)
* **Vấn đề giải quyết:** Lưu trữ văn bản tự do khiến việc tra cứu và chỉnh sửa tốn kém token và dễ sai lệch.
* **Giải pháp trong code:** Chuẩn hóa format `User.md` dưới dạng Markdown Key-Value:
  ```markdown
  # User Profile
  - **name**: DũngCT
  - **location**: Huế
  - **profession**: MLOps engineer
  - **favorite_drink**: cà phê sữa đá
  - **favorite_food**: mì Quảng
  - **pet**: corgi tên Bơ
  - **response_style**: 3 bullet ngắn, có ví dụ thực chiến, nhấn trade-off
  ```
  Giúp các hàm `facts()`, `upsert_facts()`, `file_size()` hoạt động với độ phức tạp $\mathcal{O}(1)$ dựa trên regex parse.

---

## 4. Chuỗi Logic Tổng Kết (Storyline Alignment)

Toàn bộ hệ thống phản ánh trung thực chuỗi 5 mắt xích theo tiêu chuẩn của Rubric:
1. **Baseline không nhớ dài hạn:** Phụ thuộc hoàn toàn vào `thread_id`, sang phiên mới recall = 0%.
2. **Advanced thêm `User.md` nên recall tăng:** Đạt recall 100% qua mọi session mới nhờ persistent storage.
3. **Hội thoại dài làm prompt cost tăng mạnh:** Baseline tích lũy 22,700 tokens ở hội thoại 16 lượt do mang vác toàn bộ lịch sử thô.
4. **Compact memory kéo chi phí ngữ cảnh xuống:** Advanced giảm hơn 59% prompt tokens (xuống 9,273 tokens) nhờ 26 lần nén lịch sử tự động.
5. **Hệ thống mạnh hơn nhưng phức tạp hơn:** Đòi hỏi các tầng guardrail chống nhiễu, xử lý conflict, quản lý tệp tin và cân bằng giữa chi phí ghi đĩa với token inference.
