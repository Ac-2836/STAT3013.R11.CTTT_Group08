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
| Tên repo GitHub | **`STAT3013.R11.CTTT_Group08`** (theo quy ước `STAT3013.QXX_GroupYY` của đề cương). ✅ Đã đổi trên GitHub ngày 26/10/2026. | 26/10/2026 |
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
| 8 | Cờ `cap_outlier` (giữ, không xóa) | Hampel (cửa sổ 7, k = 3, tối thiểu 0,05 Ah) **hoặc** sụt > 25% so với max 10 chu kỳ trước | Kết quả ổn định khi đổi tham số trong khoảng hợp lý. *Cập nhật ngày 26/10/2026 (đo lại bằng pipeline mới):* số dòng gắn cờ là 57; cửa sổ 5 / 9 cho 56 / 57; k = 2,5 / 3,5 cho 62 / 53; mức sàn 0,03 / 0,08 Ah cho 73 / 52; ngưỡng sụt 20% / 30% cho 86 / 51. Như vậy kết quả **không nhạy với cửa sổ**, nhạy vừa với k, và **nhạy hơn với mức sàn và ngưỡng sụt**; báo cáo phải trình bày bảng độ nhạy này thay vì chỉ ghi "ổn định". |
| 9 | Cờ `long_rest` | Nghỉ trước khi xả > **1 giờ** | Tách nghỉ theo quy trình (~1 phút hoặc ~37 phút) khỏi nghỉ ngoài dự kiến; dung lượng tăng tạm thời sau nghỉ dài là hiện tượng thật |
| 10 | Cờ `break_in` | Trong 10 chu kỳ đầu và < **90%** C₀ | Giai đoạn khởi động của pin |
| 11 | Phân tầng pin | **A**: 1 segment, ≥ 60 chu kỳ, SOH thấp nhất ≤ 0,85 · **N**: > 10% outlier · **C**: nhiều segment · **B**: còn lại · **X**: G8 | Các ngưỡng nằm trong khoảng trống tự nhiên của dữ liệu |
| 12 | Isolation Forest | Chỉ dùng làm **công cụ kiểm tra chéo** các cờ outlier ở bước tiền xử lý, không tính là thuật toán modeling | — |

**Trạng thái (26/10/2026):** các script và file CSV đã xóa để làm lại từ đầu từng bước. Các quyết định trên vẫn giữ nguyên.

**Cập nhật ngày 26/10/2026:** đã viết lại `src/build_dataset.py`, `src/clean_dataset.py` và phần `battery_fade.csv` của `src/eda_dataset.py`. Chạy lại ra đúng mọi con số đối chiếu ở mục 2.1 (các script tự kiểm tra bằng `assert`). Chi tiết quá trình ở [Work_log.md](Work_log.md). Script ở commit `5121e3f` (branch `khai-hung`), chưa qua review của nhóm (WBS C1.4).

**Cập nhật ngày 21/11/2026:** pipeline đã được merge vào `main`. Chạy lại từ 34 file `.mat` trong một bản sao sạch cho ra 8 file CSV giống từng byte với bản đã commit; mọi con số đối chiếu ở mục 2.1 và 2.2 vẫn đúng.

### 2.1 Chi tiết thực thi (chốt ngày 26/10/2026)

Bảng trên chốt **tham số**. Bảng dưới chốt **cách cài đặt** của từng quyết định. Các quy tắc này giữ đúng như bản `clean_dataset.py` cũ, để bản làm lại cho ra đúng các con số đã chốt.

| # | Mục | Cách cài đặt đã chốt |
|---|---|---|
| 1 | `capacity_int_Ah` | Tích phân **`Current_measured`** (không dùng `Current_load`). Chỉ tính các mẫu có tải (I < −0,1 A): đổi dấu thành dương, các mẫu khác đặt bằng 0. Tích phân **hết chuỗi** bằng công thức hình thang theo `Time`: `Ah = Σ (y[k] + y[k−1]) / 2 · (t[k] − t[k−1]) / 3600`. Khi chạm cutoff dòng tải tự về khoảng 0, nên phần sau không đóng góp. |
| 1b | `capacity_to_2p7V_Ah` (cột kiểm chứng) | Cùng công thức, nhưng **dừng** ở mẫu đầu tiên có `Voltage_measured < 2,7` (tính cả mẫu đó). |
| 2 | "Không có lần sạc trước" | Lần xả bị loại nếu **không có lần sạc nào** có `op_index` nhỏ hơn hẳn trong cùng pin. Không yêu cầu lần sạc nằm sau lần xả trước. Ghép bằng `merge_asof(direction="backward", allow_exact_matches=False)` theo `op_index`, trong từng pin; EIS chen giữa không ảnh hưởng. Kiểm tra trên dữ liệu gốc: 23 lần xả không có sạc trước, 4 thuộc G8, còn **19**. |
| 2b | Ngoại lệ đã biết của quy tắc 2 | Lần xả thứ 90 của B0005, B0006, B0007 nằm liền sau một lần xả khác, không có sạc ở giữa. Quy tắc chặt hơn sẽ loại thêm 3 dòng này. **Giữ quy tắc đã chốt** (không loại). Các cột `chg_*` và `rest_before_h` của 3 dòng này mô tả lần sạc trước lần xả trước đó. |
| 3 | Segment | `I_load_mean_A` = trung bình \|I\| trên các mẫu có tải (I < −0,1 A). `I_level_A = round(I_load_mean_A)`. Với xung vuông, giá trị này là biên độ: B0025–27 ra 4 A, **B0028 ra 2 A**. Nhiệt độ lấy từ `ambient_temperature` của **chính thao tác xả** (`ambient_C`), không lấy từ EIS hay lần sạc. Khóa segment = (`ambient_C`, `I_level_A`); **22°C và 24°C là hai mức khác nhau** (không pin nào có cả hai, nên không ảnh hưởng kết quả). Segment mới bắt đầu khi khóa khác dòng hợp lệ liền trước của cùng pin; đánh số từ 1. |
| 4 | C₀ | Tính riêng cho từng (pin, segment), trên các dòng **đã qua quy tắc loại bỏ**, **chưa** bỏ `cap_outlier` (cờ này tính sau C₀). `r = s.head(10).rolling(3, min_periods=1).median()` (trung vị trượt **về sau**, không căn giữa). C₀ = `r.iloc[2:].max()`; nếu segment có ít hơn 3 dòng thì C₀ = `r.max()`. |
| 5 | Hampel | Trong từng (pin, segment), cửa sổ 7 **căn giữa**, `min_periods=3`: `med = x.rolling(7, center=True, min_periods=3).median()`; `mad = 1,4826 · |x − med|.rolling(7, center=True, min_periods=3).median()` (trung vị trượt của độ lệch so với chuỗi trung vị trượt). Gắn cờ khi `|x − med| > max(3·mad; 0,05)`. |
| 6 | Sụt > 25% | Trong từng (pin, segment): `prev_max = x.shift().rolling(10, min_periods=3).max()`; gắn cờ khi `x < 0,75 · prev_max`. `cap_outlier` = Hampel **hoặc** sụt 25%. |
| 7 | Thứ tự xét tier | **X → C → N → A → B** (thứ tự điều kiện trong `np.select`). Vì vậy B0054 (1 segment, 103 lần xả) rơi vào **N**, không vào A. |
| 8 | Số dùng để xếp tier | Tính trên các dòng hợp lệ của pin, **đã bỏ `break_in`**: `n_valid` = số dòng (có tính `cap_outlier`), dùng cho "≥ 60 chu kỳ"; tỉ lệ outlier = `n_cap_outlier / n_valid`; `soh_min` bỏ thêm `cap_outlier`. Ví dụ: B0054 có `n_valid` = 94 (103 − 2 dòng bị loại − 7 break-in), 17 outlier = 18,1%; B0033 có `n_valid` = 190. |
| 9 | Các cột còn thiếu định nghĩa | Xem từ điển dữ liệu ở [Introduction_of_dataset.md](Introduction_of_dataset.md) mục 6: `op_index`, `chg_op_index`, `cycle_valid`, `soh_last`, `cycles_to_soh80`, `fade_pct_per_100cyc`, `fade25_pct_per_100cyc`, `T_max_med`, `temp`. |

**Con số đối chiếu bắt buộc** (bản làm lại phải ra đúng, nếu lệch thì phải tìm nguyên nhân trước khi dùng):

| Mục | Giá trị |
|---|---|
| Dòng bị loại | G8: 100 · không có sạc trước: 19 · không chạm cutoff: 16 · dung lượng ngoài khoảng: 1 |
| Dòng hợp lệ | **2.658** |
| Cờ | `cap_outlier` 57 · `break_in` 17 · `long_rest` 167 |
| Tier | A 12 · B 10 · C 7 · N 1 · X 4 |
| C₀ mẫu (Ah) | B0005 1,852 · B0033 1,797 · B0054 1,483 |

### 2.2 Chi tiết bổ sung khi làm lại pipeline (chốt ngày 26/10/2026)

Các điểm dưới đây docs cũ chưa định nghĩa. Mỗi điểm được kiểm tra trên dữ liệu thật trước khi chốt; chi tiết và con số kiểm chứng ở [Work_log.md](Work_log.md).

| # | Bước | Quyết định | Lý do / kiểm chứng |
|---|---|---|---|
| 1 | Đường dẫn dữ liệu | File `.mat`, `metadata.csv` và README gốc nằm trong `data/raw/`. **Không bao giờ commit dữ liệu gốc**; chỉ `data/raw/README.md` (link Google Drive) và `.gitkeep` lên repo. README gốc của dataset đổi tên thành `data/raw/README_dataset.md` để không đè file của repo. | Khớp README và Go.md; bản sao lưu nằm trên Google Drive |
| 2 | Xả (bước 2) | `duration_s` = `Time[-1]`; `t_to_cutoff_s` = `Time` của mẫu có tải cuối cùng; `energy_Wh` = tích phân hình thang của V × dòng tải (chỉ mẫu có tải) / 3.600; `T_start_C` = nhiệt độ mẫu đầu; `T_max_C`, `T_mean_C` = max, mean trên toàn chuỗi | Không cần metadata nên tính được ngay khi parse |
| 3 | Sạc (bước 3) | `chg_duration_s` = `Time[-1]`; `chg_end_time` = `start_time` + `chg_duration_s`; `chg_charge_Ah` = hình thang của I chỉ trên mẫu I > 0,1 A; `chg_cc_time_s` = `Time` của mẫu đầu tiên có `Voltage_measured` ≥ 4,2 V (NaN nếu không chạm: 5 lần sạc); `chg_T_max_C` = max nhiệt độ | `long_rest` ra đúng 167; tương quan `chg_cc_time_s`–SOH ở G1 = 0,859 trên các dòng đã bỏ `cap_outlier` (0,844 nếu tính mọi dòng; khớp F7). Phương án "mẫu cuối có dòng 1,4–1,6 A" bị loại vì ngưỡng tự đặt, không có trong quy trình. |
| 4 | EIS (bước 4) | Re/Rct là số phức có phần ảo ≠ 0 thì đặt NaN (9 lần đo, đều thuộc G8). `impedance_ops.csv` giữ giá trị thô; lọc (0; 0,5) Ω làm ở bước làm sạch. | Kết quả fit hỏng không nên giữ một nửa |
| 5 | Ghép (bước 5) | Thêm cột `n_chg_since_prev_dis`. Từ metadata chỉ lấy `group`, `discharge_cutoff_V` và `profile` (suy từ `discharge_protocol`). `build_dataset.py` tự kiểm tra bằng `assert`. | Trên dòng hợp lệ: 0 → 3 dòng, 1 → 2.639, 2 → 16. Cutoff trong metadata khớp dữ liệu ở cả 30 pin; chỉ G2 là xung vuông. |
| 6 | Loại dòng (bước 6) | 136 dòng bị loại ghi vào `cycles_dropped.csv` kèm `drop_reason`; `cycles_clean.csv` không có cột này | Phục vụ review của Tiến Hưng (Go.md 1.6) |
| 7 | `break_in` (bước 9) | "10 chu kỳ đầu" là **`cycle_valid` ≤ 10** (theo pin), không phải 10 dòng đầu mỗi segment | Theo pin ra 17 (khớp số đã chốt); theo segment ra 36, gắn cờ nhầm hiệu ứng đổi điều kiện |
| 8 | Tier (bước 10) | `cycles_clean.csv` có cột `tier`. `battery_summary.csv` gồm cả 34 pin (pin X để trống cột thống kê). | Mỗi người lọc bằng `tier == "A"`, không chép tay danh sách pin |
| 9 | `battery_fade.csv` (bước 11) | Thêm `n_used`, `n_used25` và các cột điều kiện để file tự đủ cho kiểm định | `fade25` của B0026 / B0033 / B0045 / B0054 chỉ dùng 22 / 19 / 23 / 15 điểm |

Danh sách cột đầy đủ của mọi file đầu ra ở [Introduction_of_dataset.md](Introduction_of_dataset.md) mục 5–6.

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

### 6.1 Chi tiết thực thi giao thức (chốt ngày 02/11/2026)

Giao thức ở bảng trên đã được cài trong `src/protocol.py` (`load_split()`, `naive()`, `evaluate()`). Bảng dưới chốt các điểm bảng trên chưa định nghĩa.

| # | Mục | Quyết định | Lý do / kiểm chứng |
|---|---|---|---|
| 1 | Mốc 60% | Tính **theo số dòng** của từng pin test, làm tròn xuống: `pos >= int(n · 0,6)` là dòng đánh giá (cột `is_eval`). E1 và E2 dự báo đúng cùng các dòng này. | `cycle_valid` có thể nhảy số do đã bỏ outlier. Ra 213 dòng đánh giá (B0007 67, B0034 78, B0047 28, B0055 40). |
| 2 | Chạm EOL | Chu kỳ đầu tiên có SOH **< 0,8** (nhỏ hơn hẳn) | Giống cách tính `cycles_to_soh80` trong `eda_dataset.py` |
| 3 | Pin tính sai số RUL | **B0007, B0034, B0055**. B0047 chỉ dùng cho E1. | B0047 đã xuống dưới 0,8 trước mốc 60%, nên không còn RUL để dự báo. B0034 (min 0,799) và B0055 (min 0,791) chạm sát ngưỡng; báo cáo phải ghi chú. |
| 4 | Biến mục tiêu của mọi mô hình | `soh`, **kể cả ARIMA** (trước đây Go.md 5b.1 dùng `capacity_int_Ah`) | Tier A chỉ có 1 segment nên `soh = capacity_int_Ah / C₀` với C₀ cố định: chỉ là đổi thang, Box-Jenkins giữ nguyên. `evaluate()` từ chối file có `y_true` không phải `soh`. |
| 5 | Biến ngoại sinh ở E2 | Giữ nguyên **giá trị đo cuối cùng trước mốc 60%** cho mọi bước dự báo; chỉ lag SOH được cập nhật bằng giá trị dự báo. Áp dụng chung cho SVR, LSTM và ARIMAX. | Không dùng dữ liệu tương lai; dùng được ngoài thực tế |
| 6 | Re/Rct bị thiếu | Bỏ các cửa sổ/dòng huấn luyện có NaN, **không điền** | NaN chỉ có ở 19 chu kỳ đầu của B0005, B0006, B0007 (chưa có lần đo EIS nào); các dòng đánh giá không thiếu. Điền bằng lần đo đầu tiên là lấy dữ liệu tương lai, trái mục 2 #7. |
| 7 | Validation của LSTM | `GroupKFold` theo pin, k = 4, như SVR. Early stopping dùng fold validation. | Đúng bảng trên; Go.md 5d.3 (giữ 1 pin train) đã sửa cho khớp |
| 8 | Mục tiêu học của LSTM | Mạng dự báo **ΔSOH** = SOH_t − SOH_{t−1}, rồi cộng vào SOH_{t−1} để ra SOH. File dự báo vẫn ghi SOH. | Tránh việc LSTM học cách chép lại giá trị trước (chỉ ngang naive). Không đổi định dạng dự báo chung. |

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
