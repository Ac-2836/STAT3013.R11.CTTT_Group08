# Rec.md: Biên bản các quyết định đã chốt

> **Mục đích:** Ghi lại toàn bộ những gì nhóm đã chốt cho đồ án. Khi tài liệu khác (README, Go.md, WBS.md, Introduction_of_dataset.md) mâu thuẫn với file này, **file này là căn cứ đúng**, và tài liệu kia phải được sửa lại.
>
> **Quy tắc:** Chỉ Leader (Khải Hưng) cập nhật file này, sau khi nhóm thống nhất. Không xóa quyết định cũ. Nếu thay đổi thì ghi "Thay đổi ngày …: … (lý do)".

---

## 1. Thông tin nhóm

| Mục | Giá trị | Ngày chốt |
|---|---|---|
| Lớp | **STAT3013.R11.CTTT** | 26/10/2026 |
| Nhóm | **Group 08** (tên nhóm: 4AnhTai) | 26/10/2026 |
| Tên repo GitHub khi nộp | **`STAT3013.R11.CTTT_Group08`** (theo quy ước `STAT3013.QXX_GroupYY` của đề cương) | 26/10/2026 |
| Thành viên | Khải Hưng (Leader, DL), Quốc Đạt (thống kê truyền thống), Tiến Hưng (thống kê truyền thống), Minh Vũ (ML) | 24/10/2026 |
| Ngôn ngữ | Nội dung bài làm (report, paper, slides, notebook, code, README) bằng **tiếng Anh**. File quản lý và giao tiếp nội bộ (Go.md, WBS.md, Rec.md, Work_log.md) bằng **tiếng Việt**. | 24/10/2026 |
| Đề tài | Thầy đã duyệt: chuyển thành bài toán nghiên cứu để rút ra insight về tuổi thọ pin cho doanh nghiệp | 23/10/2026 |

---

## 2. Dataset và tiền xử lý (chốt ngày 25/10/2026)

**Dataset:** NASA PCoE Li-ion Battery Aging Dataset, 34 cell 18650 (2,0 Ah), 9 nhóm G1–G9. Mô tả đầy đủ ở [Introduction_of_dataset.md](Introduction_of_dataset.md).

| # | Quyết định | Giá trị đã chốt | Lý do chính |
|---|---|---|---|
| 1 | Biến dung lượng chuẩn | **`capacity_int_Ah`**: tích phân dòng điện tới điện áp cắt riêng của pin | Cột `Capacity` gốc là dung lượng tới 2,7 V, sai lệch lớn ở 4°C (40/41 chu kỳ lệch hơn 50% đều ở 4°C). Ở nhiệt độ thường hai cột gần như trùng nhau. |
| 2 | Dung lượng lúc mới C₀ | **Max của trung vị trượt 3 chu kỳ trong 10 chu kỳ hợp lệ đầu tiên** của segment | Chu kỳ đầu, trung bình hay trung vị 3 chu kỳ đầu đều hỏng với B0033 (giai đoạn khởi động). Max 10 chu kỳ bắt nhầm giá trị nhảy vọt. Dung lượng danh định 2,0 Ah không khớp với pin nào. |
| 3 | SOH | `soh` = `capacity_int_Ah` / C₀, chỉ cho pin có **1 segment**. `soh_segment` cho mọi pin. | Pin đổi điều kiện giữa chừng thì SOH chung trộn lẫn tuổi pin với điều kiện đo |
| 4 | EOL | **SOH = 0,8** | Ngưỡng phổ biến của pin Li-ion (Khải Hưng đã research) |
| 5 | Pin bị loại | **Toàn bộ G8 (B0049–B0052)** | Crash phần mềm; dung lượng phi vật lý (tới 3,03 Ah); B0052 không có dòng xả; trở kháng vô lý. G9 có cùng điều kiện nên không mất thông tin. |
| 6 | Dòng bị loại | Không có lần sạc trước; điện áp thấp nhất > điện áp cắt + 0,1 V; dung lượng ngoài **[0,05; 2,1] Ah** | Lần xả không đại diện, bị cắt ngang, hoặc phi vật lý |
| 7 | Trở kháng | Chỉ giữ Re, Rct trong **(0; 0,5) Ω**; giá trị loại bỏ thay bằng giá trị hợp lệ gần nhất **trước đó** của cùng pin | Không lấy dữ liệu tương lai; không điền trung bình toàn cục |
| 8 | Cờ `cap_outlier` (giữ, không xóa) | Hampel (cửa sổ 7, k = 3, tối thiểu 0,05 Ah) **hoặc** sụt > 25% so với max 10 chu kỳ trước | Kết quả ổn định khi đổi tham số trong khoảng hợp lý |
| 9 | Cờ `long_rest` | Nghỉ trước khi xả > **1 giờ** | Tách nghỉ theo quy trình (~1 phút hoặc ~37 phút) khỏi nghỉ ngoài dự kiến; dung lượng tăng tạm thời sau nghỉ dài là hiện tượng thật |
| 10 | Cờ `break_in` | Trong 10 chu kỳ đầu và < **90%** C₀ | Giai đoạn khởi động của pin |
| 11 | Phân tầng pin | **A**: 1 segment, ≥ 60 chu kỳ, SOH thấp nhất ≤ 0,85 · **N**: > 10% outlier · **C**: nhiều segment · **B**: còn lại · **X**: G8 | Các ngưỡng nằm trong khoảng trống tự nhiên của dữ liệu |
| 12 | Isolation Forest | Chỉ dùng làm **công cụ kiểm tra chéo** các cờ outlier ở bước tiền xử lý, không tính là thuật toán modeling | — |

**Trạng thái (26/10/2026):** các script và file CSV đã xóa để làm lại từ đầu từng bước. Các quyết định trên vẫn giữ nguyên.

---

## 3. Nội dung Full Report (chốt ngày 26/10/2026)

| Chương | Trang | Nội dung |
|---|---|---|
| **1. Introduction** | 3 | Bối cảnh (pin chiếm phần lớn giá trị xe điện; rủi ro cho đội xe và nhà bảo hành); vấn đề; mục tiêu và các RQ; đóng góp; giải thích vì sao đề tài thuộc mảng business analytics |
| **2. Literature Review** | 4 | (a) Cơ chế và yếu tố gây lão hóa pin; (b) các phương pháp ước lượng SOH/RUL (thống kê, ML, DL), nhất là trên dataset NASA PCoE; (c) ứng dụng kinh doanh: bảo trì dự đoán, bảo hành, quản lý đội xe; (d) khoảng trống nghiên cứu |
| **3. Research Methodology** | 6 | Dataset và tiền xử lý; thống kê mô tả; thống kê suy diễn (đơn vị là pin); công thức và giả định của 4 thuật toán và SEM; giao thức đánh giá chung; công cụ |
| **4. Implementation & Experiments** | 7 | Workflow; code chính có giải thích; kiểm tra giả định; các hình bắt buộc (histogram, boxplot, scatter, SEM path diagram, forecasting plots) |
| **5. Results & Discussion** | 6 | Theo thứ tự Descriptive → Inferential → Regression → SEM → Forecasting → ML/DL; bảng so sánh mô hình; ý nghĩa thực tế |
| **6. Conclusion & Recommendations** | 2 | Tóm tắt; khuyến nghị cho doanh nghiệp (bảo hành, đội xe EV/drone) và người dùng; hạn chế; hướng phát triển **Explainable AI và Ethical AI** |
| **7. References** | 1–2 | Chuẩn IEEE, 15–25 nguồn |
| **8. Appendix** | – | Link GitHub và video, ảnh Turnitin ≤ 10%, ảnh kiểm tra AI |

**Ba điểm đóng góp chính:**
1. Chỉ ra cột `Capacity` gốc là dung lượng tới 2,7 V, sai ở 4°C, và giải thích hiện tượng "low capacity" mà README gốc ghi là chưa rõ nguyên nhân.
2. Phân tích trên 30 pin và nhiều mức điều kiện vận hành, không chỉ 4 pin G1.
3. Đưa kết quả về quyết định kinh doanh: bảo hành, vận hành ở trời lạnh, ước lượng SOH ngay khi sạc.

> ⚠ Nhận định "nhiều nghiên cứu chỉ dùng 4 pin G1" phải được kiểm chứng bằng tài liệu thật trước khi đưa vào report.

---

## 4. Bốn thuật toán modeling (chốt ngày 26/10/2026)

| # | Loại | Thuật toán | Người làm | Review | Vai trò |
|---|---|---|---|---|---|
| 1 | Thống kê truyền thống | **Hồi quy bội log-linear có tương tác** (Chap 3) | Quốc Đạt | Tiến Hưng | **Giải thích**: mỗi yếu tố làm SOH giảm thêm bao nhiêu % mỗi chu kỳ |
| 2 | Thống kê truyền thống | **ARIMA / ARIMAX** (Chap 5) | Tiến Hưng | Quốc Đạt | Dự báo chuỗi thời gian từ lịch sử của chính pin đó |
| 3 | Machine Learning | **SVR** (Chap 6) | Minh Vũ | Khải Hưng | Dự báo SOH chu kỳ kế tiếp từ các đặc trưng; permutation importance để biết đặc trưng nào quan trọng |
| 4 | Deep Learning | **LSTM** (Chap 6) | Khải Hưng | Minh Vũ | Dự báo SOH từ chuỗi đặc trưng |

**Các phương án bị loại:**

| Phương án | Lý do loại |
|---|---|
| Random Forest | Slide Chap 6 chỉ nhắc tên, trong khi SVR được dạy kỹ; slide ghi SVR phù hợp dữ liệu nhỏ–vừa; tầm quan trọng của đặc trưng vẫn có được qua permutation importance |
| GRU | LSTM được dùng nhiều nhất trong các nghiên cứu SOH trên dataset NASA, dễ so sánh ở Literature Review, và được dạy chi tiết hơn. Bù nhược điểm nhiều tham số bằng mô hình nhỏ (1 lớp, 32 đơn vị). |
| Hồi quy logistic | Đề cương không bắt buộc; chiếm thời gian của phần SEM (bắt buộc). Phần "sensitivity" được lấp theo mục 6. |

---

## 5. Phần bắt buộc ngoài 4 thuật toán (chốt ngày 26/10/2026)

| Phần | Người phụ trách |
|---|---|
| Thống kê mô tả | Minh Vũ (lead), cả nhóm cùng làm |
| Thống kê suy diễn | Tiến Hưng (lead), cả nhóm cùng làm |
| **SEM / path analysis** (sau khi học Chap 7) | Quốc Đạt (chính), Tiến Hưng (hỗ trợ) |
| Tiền xử lý, giao thức đánh giá chung, so sánh mô hình | Khải Hưng |

---

## 6. Giao thức đánh giá chung (chốt ngày 26/10/2026)

| Mục | Giá trị |
|---|---|
| Dữ liệu | Pin tier A, bỏ các dòng `cap_outlier` và `break_in` |
| Biến mục tiêu | `soh` |
| Pin test | **B0007** (24°C/2 A), **B0034** (24°C/4 A), **B0047** (4°C/1 A), **B0055** (4°C/2 A) |
| Pin train | B0005, B0006, B0018, B0033, B0045, B0046, B0048, B0056 |
| Chọn siêu tham số | `GroupKFold` theo pin trên tập train, k = 4 |
| Kịch bản E1 | Dự báo 1 bước trên 40% số chu kỳ cuối của mỗi pin test |
| Kịch bản E2 | Từ mốc 60% chuỗi, dự báo đến hết chuỗi; so sánh thời điểm chạm SOH 0,8 với thực tế (sai số RUL) |
| Metric | MAE, RMSE, MAPE, R²; sai số RUL |
| Baseline bắt buộc | Naive: SOH chu kỳ sau = SOH chu kỳ trước |
| Định dạng dự báo | `reports/predictions/<model>.csv` với các cột `model, battery_id, cycle_valid, scenario, y_true, y_pred, y_lower, y_upper` |

**Sensitivity (tạm chốt):** từ dự báo của mỗi mô hình, suy ra cảnh báo "pin chạm EOL trong 20 chu kỳ tới (có/không)", rồi tính sensitivity, specificity và confusion matrix theo Chap 4. Không fit thêm mô hình thứ 5. Người làm: Khải Hưng, trong giai đoạn so sánh mô hình.

---

## 7. Demo video và Streamlit (chốt ngày 26/10/2026)

- **Làm cả hai**, nhưng **ưu tiên thấp**: chỉ bắt đầu khi đã xong các phần chính.
- **Demo video 8–10 phút, mỗi thành viên tự trình bày phần của mình:**

| Phút | Nội dung | Người trình bày |
|---|---|---|
| 0:00–1:00 | Bài toán và ý nghĩa kinh doanh | Khải Hưng |
| 1:00–2:30 | Dữ liệu gốc (file `.mat`, vấn đề cột `Capacity` 2,7 V); chạy pipeline tiền xử lý | Khải Hưng |
| 2:30–3:30 | Các hình mô tả chính và kết quả kiểm định | Minh Vũ, Tiến Hưng |
| 3:30–6:30 | 4 mô hình, mỗi mô hình khoảng 45 giây | Từng người phụ trách |
| 6:30–7:30 | SEM | Quốc Đạt |
| 7:30–8:30 | Bảng so sánh mô hình | Khải Hưng |
| 8:30–9:30 | Khuyến nghị | Cả nhóm |

- **Streamlit app:** chọn một pin, xem SOH thực tế và dự báo, cảnh báo sắp chạm EOL.
