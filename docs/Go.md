# Go.md: Mô phỏng triển khai đồ án STAT3013 từng bước

> **Đề tài:** Statistical and Machine Learning-Based Modeling of Lithium-Ion Battery Degradation and State-of-Health Prognostics under Variable Operating Conditions
> **Dataset:** NASA PCoE Li-ion Battery Aging Dataset. Xem [Introduction_of_dataset.md](Introduction_of_dataset.md).
> **Phân công chi tiết:** [WBS.md](WBS.md) · **Các quyết định đã chốt:** [Rec.md](Rec.md) (khi mâu thuẫn, Rec.md là căn cứ đúng)
> **Lớp STAT3013.R11.CTTT · Group 08 (4AnhTai):**

| Thành viên | Vai trò |
|---|---|
| Khải Hưng | Leader, Deep Learning (LSTM), tiền xử lý, giao thức đánh giá, so sánh mô hình |
| Quốc Đạt | Mô hình thống kê truyền thống: hồi quy bội log-linear và SEM |
| Tiến Hưng | Thống kê truyền thống: thống kê suy diễn và ARIMA/ARIMAX |
| Minh Vũ | Machine Learning (SVR), thống kê mô tả |

> **Thời gian:** môn học kéo dài 07/09–28/11/2026. Tài liệu được lập ngày 23/10/2026, khi đã học tới Chap 6. Chap 7 (SEM) chưa học.

---

## 0. Cách đọc tài liệu này

Mỗi bước được viết theo cùng một khuôn:

| Mục | Ý nghĩa |
|---|---|
| **Ai** | Người chịu trách nhiệm chính (R) và người review chéo (Rv) |
| **Input** | Cần có gì trước khi làm (file, bảng, kết quả của bước trước) |
| **Làm thế nào** | Thao tác cụ thể, kèm code mẫu khi cần |
| **Vì sao** | Căn cứ: slide của thầy (chương, trang), yêu cầu đồ án, hoặc phát hiện từ dữ liệu |
| **Output** | File, bảng hoặc hình tạo ra, và nơi lưu |
| **Xong khi** | Tiêu chí để coi bước đó là hoàn thành (Definition of Done) |

Ký hiệu trạng thái: ✅ đã xong · 🔄 đang làm · ⏳ chưa bắt đầu.

---

## 1. Bức tranh tổng thể

### 1.1 Khung 4 loại analytics (Chap 1, tr. 8–10)
Toàn bộ đồ án đi theo mạch dưới đây. Mạch này cũng khớp với thứ tự "Descriptive → Inferential → Regression → SEM → Forecasting → ML" ở mục 5 của đề cương.

| Tầng | Câu hỏi | Phương pháp | Chương |
|---|---|---|---|
| Descriptive | Pin suy giảm như thế nào dưới từng điều kiện? | Thống kê mô tả, histogram, boxplot, đường SOH | Chap 1 |
| Diagnostic | Yếu tố nào làm suy giảm nhanh hơn, và nhanh hơn bao nhiêu? | t-test, F-test, Levene, ANOVA, Tukey, chi-square, hồi quy, SEM | Chap 2, 3, 7 |
| Predictive | SOH ở các chu kỳ tới là bao nhiêu? Khi nào chạm EOL? | ARIMA/ARIMAX, SVR, LSTM; sensitivity của cảnh báo EOL suy ra từ dự báo | Chap 4, 5, 6 |
| Prescriptive | Doanh nghiệp và người dùng nên làm gì? | Chuyển kết quả thành khuyến nghị | Discussion |

### 1.2 Câu hỏi nghiên cứu (Research Questions)

| RQ | Câu hỏi | Phương pháp | Người phụ trách |
|---|---|---|---|
| RQ1 | Tốc độ suy giảm SOH có khác nhau có ý nghĩa thống kê giữa các mức nhiệt độ, dòng xả và profile xả không? | Chap 2 | Cả nhóm (lead: Tiến Hưng) |
| RQ2 | Mỗi yếu tố làm SOH thay đổi bao nhiêu % sau mỗi chu kỳ? | Hồi quy bội log-linear (Chap 3) | Quốc Đạt |
| RQ3 | Dự báo được dung lượng ngắn hạn tốt đến đâu chỉ từ lịch sử của chính pin đó? Thêm biến ngoại sinh có giúp không? | ARIMA/ARIMAX (Chap 5) | Tiến Hưng |
| RQ4 | Mô hình ML với đặc trưng sạc và trở kháng dự báo SOH chu kỳ kế tiếp tốt đến đâu? Đặc trưng nào quan trọng nhất? | SVR (Chap 6) | Minh Vũ |
| RQ5 | Mô hình DL trên chuỗi đặc trưng có hơn ML trên dữ liệu nhỏ không? | LSTM (Chap 6) | Khải Hưng |
| RQ6 | Có phải điều kiện vận hành tác động lên SOH thông qua trở kháng không? | SEM / path analysis (Chap 7) | Quốc Đạt và Tiến Hưng |
| RQ7 | Mô hình kết hợp đường cong suy giảm chung, lịch sử riêng của pin và quan sát gần nhất có dự báo EOL tốt hơn và hỗ trợ quyết định thay pin không? *(thêm ngày 21/11/2026)* | Hybrid Bayes (Rec.md mục 8, `70_hybrid.ipynb`) | Khải Hưng |

### 1.3 Lịch tổng

| Giai đoạn | Nội dung | Thời gian | Người phụ trách chính | Trạng thái |
|---|---|---|---|---|
| GĐ0 | Chốt đề tài, setup repo và môi trường | trước 23/10 | Khải Hưng | ✅ |
| GĐ1 | Hiểu dữ liệu, pipeline parse và làm sạch | 23/10–27/10 | Khải Hưng (cả nhóm review) | ✅ |
| GĐ2 | Thống kê mô tả | 26/10–29/10 | Cả nhóm (lead: Minh Vũ) | ✅ `01_descriptive.ipynb` |
| GĐ3 | Thống kê suy diễn | 29/10–02/11 | Cả nhóm (lead: Tiến Hưng) | ✅ `02_inferential.ipynb` |
| GĐ4 | Chốt giao thức mô hình chung (split, metric, baseline) | 02/11–03/11 | Khải Hưng | ✅ `src/protocol.py` |
| GĐ5 | Modeling song song, 4 thuật toán | 03/11–15/11 | Mỗi người một mô hình | ✅ 5 file trong `reports/predictions/` |
| GĐ6 | SEM (sau khi học Chap 7) | 16/11–19/11 | Quốc Đạt, Tiến Hưng | ✅ `50_sem.ipynb` |
| GĐ7 | So sánh mô hình và tổng hợp kết quả | 16/11–18/11 | Khải Hưng | ✅ `60_compare.ipynb` |
| GĐ8 | Insight và khuyến nghị kinh doanh | 18/11–20/11 | Khải Hưng (cả nhóm góp) | ⏳ |
| GĐ9 | Viết các deliverable | 26/10–25/11 (Literature Review làm từ sớm) | Xem WBS | ⏳ |
| GĐ10 | Kiểm tra đạo văn và AI, tập dượt, nộp bài | 24/11–28/11 | Cả nhóm | ⏳ |

> Hạn nộp chính xác phải xem trên Moodle. Thầy chỉ nhận bài qua Moodle, không nhận qua Zalo hay email (Chap 0, tr. 5).

---

## GĐ0. Chốt đề tài và setup ✅

**0.1 Đề tài:** thầy đã duyệt hướng *"chuyển thành bài toán nghiên cứu để rút ra insight về tuổi thọ pin cho doanh nghiệp"*.

**0.2 Repo:**
- Repo hiện tại: `Ac-2836/STAT3013.R11.CTTT_Group08` (đã đổi tên ngày 26/10/2026; tên cũ `4AnhTai---Statistical-Analytics-`).
- ✅ Tên repo đã theo đúng quy ước `STAT3013.QXX_GroupYY` của đề cương (Phần C).

**0.3 Cấu trúc thư mục thống nhất:**
```
data/raw/              dữ liệu gốc .mat (KHÔNG commit, ~230 MB)
data/processed/        CSV sinh ra từ pipeline (CÓ commit; chỉ sinh bằng script, không sửa tay)
src/                   pipeline dùng chung: build, clean, eda
notebooks/             01_descriptive.ipynb, 02_inferential.ipynb, 10_regression.ipynb,
                       20_arima.ipynb, 30_ml.ipynb, 40_dl.ipynb, 50_sem.ipynb, 60_compare.ipynb
reports/figures/       hình dùng trong báo cáo (commit)
reports/predictions/   dự báo của từng mô hình theo một định dạng chung (xem GĐ4)
docs/                  tài liệu dự án
```

**0.4 Quy ước làm việc:**
- **Python 3.11**, thư viện trong `requirements.txt`. Ai cần thư viện mới (ví dụ `scikit-learn`, `torch`, `semopy`) thì báo Leader để thêm vào file.
- **Random seed** cố định là 42 ở mọi nơi có yếu tố ngẫu nhiên.
- **Git:** mỗi người làm trên branch mang tên mình (`khai-hung`, `quoc-dat`, `tien-hung`, `minh-vu`). Merge vào `main` qua Pull Request, và cần ít nhất một người review. Quy trình chi tiết và quy tắc commit message ở README mục 9–10.
- **Notebook:** phải chạy được từ đầu đến cuối (Restart & Run All) trước khi commit. Không đọc dữ liệu bằng đường dẫn tuyệt đối trên máy cá nhân.
- **Họp nhóm:** 2 lần mỗi tuần, khoảng 30 phút. Mỗi người trình bày lại phần của **một người khác**, vì đề cương yêu cầu "All members must understand the entire project".

---

## GĐ1. Hiểu và chuẩn bị dữ liệu ✅

> Chi tiết đầy đủ ở [Introduction_of_dataset.md](Introduction_of_dataset.md). Mục này mô tả lại **quá trình** đã làm và việc mỗi người cần kiểm tra lại.

### 1.1 Đọc tài liệu gốc và lập bảng manifest
- **Ai:** Khải Hưng (R), cả nhóm (Rv).
- **Làm thế nào:** đọc 9 file `README_*.txt` và `metadata.csv`, sau đó đối chiếu với giá trị **đo được** trong từng chu kỳ: nhiệt độ môi trường và dòng xả trung bình.
- **Vì sao:** metadata có thể sai, và thực tế đã sai ở 3 chỗ:
  - B0028 xả với biên độ 2 A, không phải 4 A.
  - G6 có 3 pha, với 22°C ở pha đầu.
  - G5 có 4 pha.
- **Output:** Mục 2 của Introduction_of_dataset.md.

### 1.2 Phân loại dữ liệu (Chap 1, tr. 13–14, 22–32)

| Câu hỏi | Kết luận |
|---|---|
| Dữ liệu thuộc loại cấu trúc nào? | Semi-structured (struct MATLAB lồng nhau), phải làm phẳng thành bảng |
| Dạng dữ liệu theo thời gian? | **Panel data**: 34 pin × chuỗi chu kỳ |
| Nhiệt độ thuộc thang đo nào? | Interval. Được tính mean/SD, nhưng không so sánh theo tỉ lệ. |
| Dung lượng, dòng điện, trở kháng thuộc thang đo nào? | Ratio, liên tục |
| Chỉ số chu kỳ? | Ratio, rời rạc |
| Mã pin, mã nhóm, profile? | Nominal |

### 1.3 Parse và gộp dữ liệu (Chap 1, tr. 15–16)
```bash
python src/build_dataset.py      # data/raw/*.mat  ->  data/processed/*_ops.csv, cycles.csv
```
- **Format transformation:** datevec chuyển sang datetime, số phức chuyển sang số thực.
- **Aggregation:** hàng trăm điểm đo mỗi lần xả được gộp thành một dòng gồm `capacity_int_Ah`, `T_max_C`, `V_min`, `I_load_mean_A`, `energy_Wh`, …
- **Integration:** mỗi lần xả được ghép với lần sạc ngay trước nó và lần EIS gần nhất **trước** nó, dùng `merge_asof` theo `op_index`.
- **Kiểm chứng then chốt:** tích phân dòng điện tới 2,7 V khớp với cột `Capacity` gốc ở 98,5% số chu kỳ. Từ đó phát hiện F1: cột gốc là dung lượng đến 2,7 V, nên **dùng `capacity_int_Ah`**.

### 1.4 Làm sạch dữ liệu (Chap 1, tr. 17; Chap 6, tr. 69–79)
```bash
python src/clean_dataset.py      # -> cycles_clean.csv, battery_summary.csv
```

| Quy tắc | Số dòng bị loại | Căn cứ |
|---|---|---|
| Loại các pin của G8 (B0049–B0052) | 100 | Crash phần mềm, dòng tải bằng 0, dung lượng vượt 2,0 Ah danh định |
| Lần xả không có lần sạc trước | 19 | Pin chưa được sạc đầy theo quy trình |
| Lần xả không chạm tới cutoff | 16 | Lần xả bị cắt ngang |
| Dung lượng ngoài khoảng [0,05; 2,1] Ah | 1 | Giá trị phi vật lý |

- **Kết quả:** 2.794 → **2.658** dòng hợp lệ.
- **Gắn cờ nhưng không xóa:** `cap_outlier` (57), `break_in` (17), `long_rest` (167).
- **Chia segment** theo (nhiệt độ, mức dòng). Chỉ tính SOH cho pin có điều kiện không đổi.
- **Phân tầng pin:** A (12 pin), B (10), C (7), N (1), X (4).

### 1.5 Isolation Forest để kiểm tra chéo outlier (Chap 6 §5): *bài tập cho Minh Vũ*
- **Input:** `cycles_clean.csv`, chỉ lấy tier A và B.
- **Làm thế nào:** chạy `IsolationForest(contamination='auto', random_state=42)` trên các cột (`soh`, ΔSOH so với chu kỳ trước, `T_max_C`, `V_min`, `rest_before_h`), fit riêng cho từng nhóm. Lấy điểm `score_samples` và chuyển sang thang s(x, n) như công thức trên slide tr. 78.
- **Vì sao:** vừa đối chiếu cờ `cap_outlier` (tạo bằng quy tắc Hampel) với một phương pháp ML của môn học, vừa là nội dung "anomaly detection" để đưa vào báo cáo.
- **Output:** bảng so khớp "Hampel và iForest cùng gắn cờ / chỉ Hampel / chỉ iForest", và một hình scatter.
- **Xong khi:** giải thích được vì sao các điểm hồi phục sau thời gian nghỉ (F5) **không nên** bị coi là outlier.

### 1.6 Review dữ liệu: mỗi thành viên kiểm tra một phần

| Thành viên | Việc cần kiểm tra | Cách làm |
|---|---|---|
| Quốc Đạt | Định nghĩa SOH và C₀ | Mở `cycles_clean.csv`, vẽ SOH của B0033 và B0054, xem cách tính C₀ có hợp lý không |
| Tiến Hưng | Quy tắc loại bỏ | Lấy ngẫu nhiên 5 dòng bị loại trong `cycles.csv`, kiểm tra lý do có đúng không |
| Minh Vũ | Các cờ outlier | So sánh với Isolation Forest (bước 1.5) |
| Khải Hưng | Join với dữ liệu sạc và EIS | Kiểm tra `chg_op_index` < `op_index` và Re/Rct không bao giờ lấy từ lần đo sau |

**Xong khi:** cả 4 người đồng ý với các quy tắc. Nếu có thay đổi thì Leader sửa `clean_dataset.py` và **ghi lại lý do** trong Introduction_of_dataset.md.

---

## GĐ2. Thống kê mô tả (Chap 1 §3–4) ✅

- **Ai:** lead là Minh Vũ. Cả nhóm cùng làm notebook `01_descriptive.ipynb`.
- **Input:** `cycles_clean.csv`, `battery_summary.csv`, `battery_fade.csv` (chạy `python src/eda_dataset.py` để tạo các hình nền).

### 2.1 Bảng thống kê mô tả
- **Chỉ số:** n, mean, median, Q1, Q3, SD, SE, min, max, range, skewness, kurtosis. Đây đúng là danh sách tóm tắt ở slide Chap 1, tr. 57.
- **Hai mức tính:**
  - Mức chu kỳ: `capacity_int_Ah`, `soh`, `T_max_C`, `Re_ohm`, `Rct_ohm`, `rest_before_h`.
  - Mức pin: `C0_Ah`, `soh_last`, `fade_pct_per_100cyc`, `cycles_to_soh80`.
- **Tách theo nhóm điều kiện** (nhiệt độ × dòng xả). Lý do: `T_max_C` có kurtosis 1,95 và nhiều đỉnh khi gộp ba mức nhiệt.
- ⚠ **Kurtosis:** pandas trả về *excess kurtosis*. Phải cộng thêm 3 để đúng quy ước trên slide (phân phối chuẩn = 3).

### 2.2 Biểu đồ (Chap 1, tr. 59–63)

| Hình | Nội dung | Đọc như thế nào |
|---|---|---|
| Histogram SOH | Phân phối SOH ở mức chu kỳ | Lệch trái (skewness −0,64). Phần lớn chu kỳ nằm ở giai đoạn SOH còn cao. |
| Histogram `rest_before_h` (thang log) | Phân phối thời gian nghỉ | Lệch phải cực mạnh (skewness 10,6), nên phải dùng log hoặc chia nhóm (Chap 1, tr. 17). |
| Boxplot `T_max_C` theo nhiệt độ môi trường | Mức tự phát nhiệt khi xả | Trung vị 15,0 / 40,6 / 61,8°C ở 4 / 24 / 43°C |
| Boxplot tốc độ suy giảm theo điều kiện | So sánh trực quan để chuẩn bị cho GĐ3 | Xem độ dài whisker để đánh giá phương sai trước khi chạy Levene |
| Đường SOH theo chu kỳ (`fig1`) | Hình chủ đạo của báo cáo | Tô màu theo nhiệt độ, chia panel theo mức dòng |
| Scatter Re/Rct theo SOH (`fig5`) | Chuẩn bị cho hồi quy và SEM | r = −0,78 với Re và −0,81 với Rct (G1) |

### 2.3 Viết nhận xét theo mẫu slide Chap 1, tr. 60
Mạch viết: phát hiện, rồi con số, rồi hàm ý. Ví dụ:
> "Ở 43°C, nhiệt độ cell khi xả đạt trung vị 61,8°C, so với 40,6°C ở 24°C. Xét trên cùng cửa sổ 25 chu kỳ đầu, pin ở 43°C mất khoảng 27,8 điểm % SOH sau mỗi 100 chu kỳ, so với 20,1 điểm % ở 24°C cùng dòng 4 A. Như vậy hệ thống làm mát cho pack là yêu cầu bắt buộc với xe vận hành ở khí hậu nóng."

**Output:** `notebooks/01_descriptive.ipynb` và hình trong `reports/figures/`.
**Xong khi:** mỗi bảng và mỗi hình có 2–3 câu diễn giải, và mỗi người giải thích được một hình bất kỳ.

---

## GĐ3. Thống kê suy diễn (Chap 2) ✅

- **Ai:** lead là Tiến Hưng. Cả nhóm làm notebook `02_inferential.ipynb`.
- **Input:** `data/processed/battery_fade.csv`, mỗi dòng là **một pin** (23 pin có điều kiện không đổi).

> **Nguyên tắc vàng: đơn vị quan sát là PIN.**
> - ANOVA giả định các quan sát độc lập (Chap 2, tr. 75).
> - Các chu kỳ trong cùng một pin có tự tương quan bậc 1 tới 0,93 (F9).
> - Nếu dùng khoảng 2.000 chu kỳ làm mẫu, p-value sẽ nhỏ một cách giả tạo (pseudo-replication).

**Biến phụ thuộc:** `fade_pct_per_100cyc` (cả chuỗi) và `fade25_pct_per_100cyc` (25 chu kỳ đầu, cửa sổ chung cho mọi pin).
- **Chạy mọi kiểm định trên cả hai biến.** Nếu kết luận khác nhau thì phải báo cáo rõ, vì đó chính là kiểm tra độ vững (robustness).

### 3.1 Khoảng tin cậy (Chap 2, tr. 4–19)
- **CI 95% cho tốc độ suy giảm trung bình của mỗi nhóm điều kiện.**
  - n < 30 và chưa biết σ, nên dùng t với df = n − 1 (tr. 8).
  - Ví dụ: G3 có n = 4, t(3; 0,025) = 3,182, nên khoảng tin cậy sẽ rộng. Đây là bằng chứng định lượng cho phần hạn chế về số lượng pin.
- **CI cho tỉ lệ (tr. 16–19):** tỉ lệ các lần xả có dung lượng tăng lên so với chu kỳ trước, trên tier A/B. Mẫu lớn nên dùng z.
```python
from scipy import stats
x = fade.loc[fade.group == "G3", "fade_pct_per_100cyc"]
ci = stats.t.interval(0.95, df=len(x) - 1, loc=x.mean(), scale=stats.sem(x))
```

### 3.2 So sánh hai nhóm, chỉ khác nhau một yếu tố (tr. 38–53)

| # | So sánh | H0 / Ha | Loại kiểm định | Ghi chú |
|---|---|---|---|---|
| T1 | **Nhiệt độ ở 4 A:** G3 (43°C, n = 4) với B0033/B0034 (24°C, n = 2) | H0: μ43 = μ24; Ha: μ43 > μ24 (một phía) | t-test (Issue 3) | Giả thuyết một phía phải lấy từ tài liệu tham khảo (nhiệt cao đẩy nhanh phản ứng phụ), **nêu trước khi nhìn số liệu** |
| T2 | **Nhiệt độ ở 2 A:** G1 (24°C, n = 4) với G9 (4°C, n = 3, không gồm B0054) | H0: μ24 = μ4; Ha: μ24 ≠ μ4 | t-test hai phía (Issue 1) | |
| T3 | **Dòng xả ở 4°C:** G7 (1 A, n = 4) với G9 (2 A, n = 3) | H0: μ1A = μ2A; Ha: ≠ | t-test hai phía | Kết quả dự kiến ngược trực giác: 1 A suy giảm nhanh hơn. Gợi ý giải thích qua nhiệt độ cell (xem 3.6). |
| T4 | **Profile ở 24°C/4 A:** G2 xung vuông (B0025–27) với CC (B0033/B0034) | H0: μsq = μcc | t-test hai phía | |

**Quy trình cho mỗi cặp:**
1. **F-test** (tr. 51) hoặc **Levene** (tr. 64) để kiểm tra phương sai bằng nhau.
2. Nếu phương sai bằng nhau thì dùng t-test pooled, df = n₁ + n₂ − 2 (tr. 40). Nếu không thì dùng Welch t-test, và ghi chú rằng Welch là mở rộng ngoài slide.
3. Báo cáo đầy đủ: thống kê T, df, giá trị tới hạn, p-value, kết luận, ý nghĩa thực tế.
```python
stats.levene(a, b, center="median")
stats.ttest_ind(a, b, equal_var=True, alternative="greater")   # T1
```

### 3.3 One-way ANOVA và Tukey HSD (tr. 72–89)
- **Nhân tố:** "nhóm điều kiện", gồm 6 mức: 24°C/2A, 24°C/4A, 24°C/xung vuông, 43°C/4A, 4°C/1A, 4°C/2A.
- **Trình tự:** Levene, rồi `f_oneway`. Nếu p < 0,05 thì chạy `pairwise_tukeyhsd` để biết cặp nào khác nhau.
- **Kiểm tra phân phối chuẩn:** n quá nhỏ, nên dùng QQ-plot của phần dư. Nếu vi phạm, chạy thêm Kruskal-Wallis như phương án thay thế và ghi rõ nằm ngoài slide.
- ⚠ Khi diễn giải phải gọi là **"nhóm điều kiện"**, không gọi là "nhiệt độ", vì nhiệt độ bị lẫn với dòng xả.

### 3.4 Two-way ANOVA: viết rõ vì sao KHÔNG làm được (tr. 90)
Bảng số pin theo từng ô (nhiệt độ × dòng xả, chỉ tính CC):

| | 1 A | 2 A | 4 A |
|---|---|---|---|
| 4°C | 4 | 3–4 | 0 |
| 24°C | 0 | 5 | 2 |
| 43°C | 0 | 0 | 4 |

- Slide yêu cầu cỡ mẫu bằng nhau ở mọi ô. Ở đây có nhiều ô bằng 0, nên **không ước lượng được hiệu ứng tương tác**.
- Đoạn giải thích này nên đưa nguyên vào báo cáo, vì nó thể hiện hiểu biết về giả định của phương pháp.

### 3.5 Chi-square (tr. 56–63)
- **Bảng chéo ở mức pin:** nhóm nhiệt độ (4°C, 24°C, 43°C) × "đã xuống dưới SOH 0,9 trong 40 chu kỳ đầu (có/không)".
- Mốc 40 chu kỳ được chọn vì đây là số chu kỳ ngắn nhất của G3, nên mọi pin đều được quan sát đủ.
- Tính bảng tần số kỳ vọng E_ij như ví dụ ở tr. 59–60. Nếu E_ij < 5 thì kết quả không tin cậy; ghi chú Fisher exact là phương án thay thế.

### 3.6 Bổ sung: tương quan giữa tốc độ suy giảm và nhiệt độ cell
- Tương quan Pearson giữa `fade_pct_per_100cyc` và `T_max_med` ở mức pin, kèm scatter.
- **Mục đích:** kiểm tra giả thuyết "nhiệt độ của chính cell, chứ không phải nhiệt độ môi trường, mới quyết định tốc độ suy giảm" (từ kết quả T3).

**Output:** bảng tổng hợp gồm các cột: Kiểm định | H0 | Thống kê | df | p | Kết luận | Ý nghĩa kinh doanh.
**Xong khi:** mọi kiểm định đều ghi rõ giả định đã kiểm tra, và mỗi người tự trả lời được câu hỏi "vì sao dùng t mà không dùng z".

---

## GĐ4. Giao thức mô hình chung ✅

- **Ai:** Khải Hưng (R), cả nhóm thống nhất.
- **Vì sao:** đề cương yêu cầu "Compare models (accuracy, p-value, RMSEA, R², sensitivity)". Muốn so sánh được thì 4 mô hình phải dùng **cùng dữ liệu, cùng cách chia và cùng metric**.

### 4.1 Dữ liệu
- Lấy tier A trong `cycles_clean.csv`, bỏ các dòng `cap_outlier` và `break_in`.
- Biến mục tiêu: `soh`.
- Trục thời gian: `cycle_valid`.

### 4.2 Chia train/test theo pin
| Tập | Pin |
|---|---|
| **Test** (mỗi điều kiện một pin) | B0007 (24°C/2A), B0034 (24°C/4A), B0047 (4°C/1A), B0055 (4°C/2A) |
| **Train** | B0005, B0006, B0018, B0033, B0045, B0046, B0048, B0056 |
| Validation (để chọn siêu tham số) | `GroupKFold` theo `battery_id` **trên tập train**, k = 4 |

*Vì sao không chia ngẫu nhiên theo chu kỳ:* hai chu kỳ kề nhau gần như giống hệt nhau. Chia ngẫu nhiên sẽ cho R² khoảng 0,99, nhưng đó là R² ảo do rò rỉ dữ liệu.

### 4.3 Hai kịch bản đánh giá

| Kịch bản | Mô tả | Áp dụng cho |
|---|---|---|
| **E1: dự báo 1 bước** | Với **40% số chu kỳ cuối** của mỗi pin test, dự báo SOH tại chu kỳ t khi đã biết thông tin đến t−1 | Cả 4 mô hình. ARIMA được fit trên 60% đầu của chính pin đó, rồi dự báo cuốn chiếu từng bước. |
| **E2: dự báo nhiều bước / RUL** | Đứng ở mốc 60% chuỗi, dự báo đến hết chuỗi. So sánh chu kỳ dự báo chạm SOH 0,8 với chu kỳ thực tế. Biến ngoại sinh giữ nguyên giá trị đo cuối cùng trước mốc (Rec.md 6.1 #5). | Sai số RUL chỉ tính cho B0007, B0034, B0055; B0047 đã dưới 0,8 trước mốc 60% (Rec.md 6.1 #3) |

### 4.4 Metric (Chap 3, tr. 81–86)
- MAE, RMSE, MAPE, R² trên cả E1 và E2 (`protocol.evaluate()`).
- Sai số RUL (số chu kỳ) trên E2.

### 4.5 Baseline bắt buộc
- **Naive persistence:** SOH_t = SOH_{t−1}.
- Mọi mô hình **phải được so với baseline này**. Nếu không thắng baseline thì phải nói thẳng ra trong báo cáo.

### 4.6 Danh sách biến được phép và bị cấm làm biến dự báo

| Được phép dùng | Bị cấm (rò rỉ cùng chu kỳ) |
|---|---|
| `soh` và `capacity_int_Ah` ở các chu kỳ t−1, t−2, … | `capacity_*`, `energy_Wh`, `duration_s`, `t_to_cutoff_s` của **chính chu kỳ t** |
| `chg_cc_time_s`, `chg_charge_Ah`, `chg_T_max_C` của lần sạc ngay trước chu kỳ t | `T_max_C`, `V_min` của chính chu kỳ t. Chỉ được dùng giá trị của t−1. |
| `Re_ohm`, `Rct_ohm` (đã được lấy từ lần đo trước) | `C0_Ah` khi so sánh giữa các pin (vì đây là thông tin tương lai của giai đoạn break-in) |
| `ambient_C`, `I_level_A`, `discharge_cutoff_V`, `profile`, `rest_before_h`, `cycle_valid` | |

### 4.7 Định dạng file dự báo chung
Mỗi mô hình xuất một file CSV vào `reports/predictions/<model>.csv` với các cột:
```
model, battery_id, cycle_valid, scenario (E1|E2), y_true, y_pred, y_lower, y_upper
```
- `y_lower` và `y_upper` là khoảng dự báo 95%, để trống nếu mô hình không cung cấp.
- Notebook `60_compare.ipynb` (Leader phụ trách) đọc toàn bộ thư mục này và tự sinh bảng so sánh.

**Xong khi:** có file `src/protocol.py` chứa hàm `load_split()` và `evaluate(pred_df)` dùng chung. Leader viết, cả nhóm review.

---

## GĐ5. Modeling song song ✅

### GĐ5a. Quốc Đạt: Hồi quy bội log-linear có tương tác (Chap 3)

**Notebook:** `notebooks/10_regression.ipynb` · **Review:** Tiến Hưng

| Bước | Làm thế nào | Vì sao (slide) |
|---|---|---|
| 5a.1 Tương quan | Ma trận tương quan của `soh`, `cycle_valid`, `Re_ohm`, `Rct_ohm`, `T_max_C`, `chg_cc_time_s`. **Re và Rct có r = 0,97 trong G1, nhưng chỉ 0,43 trên tier A** (mọi dòng; tính bằng pipeline ngày 26/10/2026). Chỉ trên 8 pin train thì r = 0,51, và G1 train là 0,95 (`Regression_Report.md`). Report phải ghi rõ dùng tập nào. Quyết định có đưa cả hai vào hay không phải dựa trên VIF thật của tập train. | Chap 3, tr. 8, tr. 41 (\|r\| > 0,7) |
| 5a.2 Hồi quy đơn | `soh ~ cycle_valid` cho G1. Đọc R², SE, F-test, t-test và CI 95% của hệ số dốc. | tr. 22–30 |
| 5a.3 Residual plot | Vẽ phần dư theo giá trị dự báo. Nếu thấy hình chữ U hoặc đường cong thì chuyển sang mô hình phi tuyến. | tr. 74, tr. 76 |
| 5a.4 Log-linear | `ln(soh) ~ cycle_valid`, khi đó hệ số b₁·100 là % SOH thay đổi mỗi chu kỳ. So sánh với polynomial bậc 2 bằng adjusted R². | tr. 67–68, 73–74, 85 |
| 5a.5 Hồi quy bội + dummy + tương tác | `ln(soh) ~ cycle_valid*C(temp) + cycle_valid*I_level_A + discharge_cutoff_V + profile` | tr. 52–60 (dummy k−1; tương tác giống ví dụ Age×MBA) |
| 5a.6 Chọn mô hình | Bỏ dần biến có p lớn nhất vượt α = 0,05, theo dõi adjusted R². Kiểm tra VIF; nếu VIF > 10 thì không đưa Re và Rct vào cùng nhau. | tr. 39, 41 |
| 5a.7 Kiểm tra giả định | Vẽ QQ-plot. Phần dư tự tương quan làm SE bị đánh giá thấp, nên **làm thêm hồi quy 2 bước**: `fade_pct_per_100cyc ~ temp + I_level_A + cutoff` trên `battery_fade.csv` (n = 23, các quan sát độc lập). | tr. 36; F9 |
| 5a.8 Đánh giá | Dự báo trên tập test theo GĐ4 (E1, E2), xuất file vào `reports/predictions/regression.csv` | tr. 81–86 |
| 5a.9 Diễn giải | Viết thành câu, ví dụ: "Ở 43°C, mỗi chu kỳ SOH giảm thêm x% so với 24°C, giữ các yếu tố khác không đổi" | tr. 73 |

```python
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
m = smf.ols("np.log(soh) ~ cycle_valid * C(temp) + cycle_valid * I_level_A + discharge_cutoff_V",
            data=train).fit()
print(m.summary())       # R², adj R², F-test, t-test và CI của từng hệ số
```

**Xong khi:** có bảng hệ số kèm CI, residual plot, bảng VIF, file dự báo, và một đoạn diễn giải có hàm ý kinh doanh.

---

### GĐ5b. Tiến Hưng: ARIMA / ARIMAX (Chap 5)

**Notebook:** `notebooks/20_arima.ipynb` · **Review:** Quốc Đạt

| Bước | Làm thế nào | Vì sao (slide) |
|---|---|---|
| 5b.1 Chọn chuỗi | Chuỗi `soh` theo `cycle_valid` (không dùng `capacity_int_Ah`, xem Rec.md 6.1 #4) của từng pin tier A. Dùng số thứ tự chu kỳ làm trục thời gian, **không dùng ngày**, vì khoảng nghỉ giữa các chu kỳ không đều. | tr. 4 |
| 5b.2 Vẽ chuỗi | Nhận xét xu hướng giảm, các "răng cưa" sau những lần nghỉ dài, và việc không có tính mùa vụ | tr. 5–6 |
| 5b.3 Tính dừng | ADF trên chuỗi gốc: không dừng ở 10/12 pin tier A (p = 0,11–0,97); B0045 và B0055 cho p < 0,002 do nhiều bước nhảy hồi phục. ACF chuỗi gốc giảm chậm. **Sai phân bậc 1** dừng ở cả 12 pin (p < 0,001). Nhóm chọn d = 1 cho mọi pin (kết quả thật ở `20_arima.ipynb` mục 2). *Sửa ngày 21/11/2026: bản cũ ghi p = 0,53–0,89, không tái lập được.* | tr. 11–14, tr. 71 |
| 5b.4 Nhận dạng mô hình | Vẽ ACF/PACF của chuỗi sai phân. PACF cắt sau lag p thì gợi ý AR(p); ACF cắt sau lag q thì gợi ý MA(q). Thử p, q ∈ {0, 1, 2}, chọn theo AIC. | tr. 47, 65–67 |
| 5b.5 Ước lượng | `ARIMA(y_train, order=(p, 1, q)).fit()` | tr. 48 |
| 5b.6 Chẩn đoán | ACF của phần dư không còn đỉnh nào nổi bật. Kiểm định Ljung-Box (mở rộng ngoài slide). Nếu chưa đạt thì quay lại bước 5b.4. | tr. 49 |
| 5b.7 Dự báo | Kịch bản E1: dự báo cuốn chiếu 1 bước trên 40% cuối. Kịch bản E2: dự báo một mạch từ mốc 60%, kèm khoảng tin cậy. | tr. 49 |
| 5b.8 ARIMAX | Thêm biến ngoại sinh `log1p(rest_before_h)` (giải thích hiện tượng hồi phục F5) và `Rct_ohm`. So sánh AIC và RMSE với ARIMA. | tr. 52–53 |
| 5b.9 SARIMA | Xem ACF ở các lag mùa vụ. Không có chu kỳ cố định, nên **kết luận không cần SARIMA**. Ghi điều này vào báo cáo để chứng tỏ đã cân nhắc. | tr. 60–62 |

```python
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
fit = ARIMA(y_tr, exog=x_tr, order=(1, 1, 1)).fit()
acorr_ljungbox(fit.resid, lags=[10])
fc = fit.get_forecast(steps=len(y_te), exog=x_te)
fc.summary_frame()      # giá trị dự báo kèm khoảng tin cậy 95%
```

**Insight cần chỉ ra:**
- ARIMA là mô hình tuyến tính (tr. 51), nên với chuỗi d = 1 thì dự báo dài hạn sẽ đi theo đường drift. Mô hình không bắt được "đầu gối" suy giảm, và khoảng tin cậy nở rất rộng.
- **Kết luận:** ARIMA phù hợp cho dự báo ngắn hạn.

**Xong khi:** có bảng mô hình theo từng pin (p, d, q, AIC), hình dự báo kèm khoảng tin cậy, kết quả Ljung-Box, và file `reports/predictions/arima.csv`.

---

### GĐ5c. Minh Vũ: Machine Learning, SVR (Chap 6)

**Notebook:** `notebooks/30_ml.ipynb` · **Review:** Khải Hưng

Làm theo đúng 8 bước SVR ở slide Chap 6, tr. 33:

| Bước | Làm thế nào |
|---|---|
| 1. Import | `sklearn.svm.SVR`, `GroupKFold`, `GridSearchCV`, `sklearn.inspection.permutation_importance` |
| 2. Load dữ liệu | Dùng `load_split()` của GĐ4 |
| 3. Tạo đặc trưng và scale | Lag SOH từ 1 đến 5, trung bình trượt 5 chu kỳ, `chg_cc_time_s`, `Re_ohm`, `Rct_ohm`, `T_max_C` của chu kỳ trước, `log1p(rest_before_h)`, dummy nhiệt độ, `I_level_A`, cutoff. Dùng `StandardScaler` **đặt trong Pipeline**, để scaler chỉ fit trên phần train của mỗi fold. |
| 4. Định nghĩa mô hình | SVR kernel RBF. Lưới siêu tham số ban đầu: C ∈ {0,1; 1; 10; 100}, ε ∈ {0,001; 0,005; 0,01}, gamma ∈ {'scale'; 0,01; 0,1}. Giải thích ý nghĩa của C và ε theo tr. 31. *Bản thật đã mở rộng lưới (tham số tốt nhất rơi vào biên) và chọn bằng quy tắc một sai số chuẩn; lý do ở `30_ml.ipynb` bước 4.* |
| 5. Dự báo | Chạy E1 trên tập test. E2 làm bằng dự báo đệ quy: dùng giá trị vừa dự báo làm lag cho bước sau. |
| 6. Đánh giá | MAE, RMSE, MAPE, R², so với baseline naive |
| 7. Vẽ train/test | Vẽ giá trị thực và giá trị dự báo cho từng pin test |
| 8. Vẽ dự báo | Đường dự báo E2 và mốc SOH 0,8 |

```python
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
pipe = make_pipeline(StandardScaler(), SVR(kernel="rbf"))
gs = GridSearchCV(pipe, {"svr__C": [0.1, 1, 10, 100], "svr__epsilon": [0.001, 0.005, 0.01]},
                  cv=GroupKFold(4), scoring="neg_root_mean_squared_error")
gs.fit(X_tr, y_tr, groups=g_tr)
```

**Đặc trưng nào quan trọng (trả lời RQ4):**
- Tính **permutation importance** của SVR trên tập test: xáo trộn từng đặc trưng và đo RMSE tăng bao nhiêu.
- **Đối chiếu với các hệ số hồi quy của Quốc Đạt.** Nếu hai cách cùng chỉ ra Rct và nhiệt độ là quan trọng thì kết luận vững hơn.

**Isolation Forest:** chỉ là công cụ kiểm tra chéo ở bước tiền xử lý (bước 1.5 ở GĐ1), không phải thuật toán modeling.

**Xong khi:** có bảng siêu tham số tốt nhất, bảng metric, hình permutation importance, và file `reports/predictions/svr.csv`.

---

### GĐ5d. Khải Hưng: Deep Learning, LSTM (Chap 6)

**Notebook:** `notebooks/40_dl.ipynb` · **Review:** Minh Vũ

| Bước | Làm thế nào | Vì sao (slide) |
|---|---|---|
| 5d.1 Tạo chuỗi | Cửa sổ trượt dài 10 chu kỳ, dùng cùng bộ đặc trưng với GĐ5c, để dự báo SOH chu kỳ kế tiếp. **Không cho cửa sổ vượt qua ranh giới giữa hai pin.** Bỏ cửa sổ có Re/Rct thiếu. Mạng học ΔSOH rồi cộng vào SOH_{t−1} (Rec.md 6.1 #6, #8). | tr. 36–38 (sequence modeling) |
| 5d.2 Kiến trúc | LSTM 1 lớp, 32 đơn vị, dropout 0,2, Dense(1). Mô hình nhỏ để bù việc LSTM nhiều tham số trên dữ liệu ít. | tr. 57–62 (3 cổng forget/input/output, forward pass) |
| 5d.3 Huấn luyện | Adam, loss MSE, early stopping với patience 20. Validation bằng `GroupKFold` theo pin, k = 4, như SVR (Rec.md 6.1 #7). Dùng seed 42; chạy 5 seed khác nhau và báo cáo trung bình ± độ lệch chuẩn. | tr. 62 (BPTT) |
| 5d.4 Đánh giá | Kịch bản E1 và E2 (E2 dự báo đệ quy), so với baseline naive, SVR và ARIMA | GĐ4 |
| 5d.5 Diễn giải trung thực | Chỉ có khoảng 1.500 dòng tier A. Bảng so sánh ở tr. 13 nói DL cần nhiều dữ liệu. Nếu LSTM không thắng SVR thì **đó vẫn là một kết luận có giá trị**. | tr. 13–14 |

```python
import torch.nn as nn
class LSTMReg(nn.Module):
    def __init__(self, n_feat, h=32):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, h, batch_first=True)
        self.drop = nn.Dropout(0.2)
        self.out = nn.Linear(h, 1)
    def forward(self, x):                  # x: (batch, 10, n_feat)
        _, (hT, _) = self.lstm(x)          # LSTM returns (hidden state, cell state)
        return self.out(self.drop(hT[-1])).squeeze(-1)
```

**Xong khi:** có learning curve, bảng metric trung bình trên 5 seed, và file `reports/predictions/lstm.csv`.

---

## GĐ6. SEM / Path analysis (Chap 7) ✅

- **Ai:** Quốc Đạt (R), Tiến Hưng (hỗ trợ).
- **Notebook:** `notebooks/50_sem.ipynb`.

**Việc có thể làm ngay:**
- Vẽ **sơ đồ giả thuyết**: Điều kiện vận hành (`ambient_C`, `I_level_A`, `discharge_cutoff_V`) tác động lên Trở kháng (`Re_ohm`, `Rct_ohm`), rồi Trở kháng tác động lên `soh`. Thêm các đường tác động trực tiếp từ điều kiện vận hành lên `soh`, và `cycle_valid` làm biến kiểm soát.
- Chuẩn bị dữ liệu: chỉ lấy các chu kỳ có lần đo EIS mới (Re/Rct thay đổi so với dòng trước), trên tier A/B.

**Sau khi học Chap 7:**
- Chạy path analysis bằng `semopy` (Python) hoặc `lavaan` (R).
- Báo cáo các chỉ số phù hợp mô hình (RMSEA, CFI, TLI), hệ số đường dẫn và hiệu ứng gián tiếp (mediation).
- Dữ liệu chỉ có 2–3 chỉ báo cho mỗi khái niệm, nên **dùng biến quan sát** thay vì biến tiềm ẩn.
- Ghi rõ hạn chế: các quan sát trong cùng một pin phụ thuộc nhau. Có thể chạy thêm trên dữ liệu ở mức pin như kiểm tra độ vững.

**Xong khi:** có sơ đồ đường dẫn kèm hệ số, bảng chỉ số phù hợp, và đoạn diễn giải "x% tác động của nhiệt độ lên SOH đi qua Rct".

---

## GĐ7. So sánh mô hình và tổng hợp ✅

- **Ai:** Khải Hưng (R), cả nhóm.
- **Notebook:** `notebooks/60_compare.ipynb`. Notebook này đọc toàn bộ `reports/predictions/*.csv`.

**Bảng so sánh chính** (đưa vào cả báo cáo và paper):

| Mô hình | MAE | RMSE | MAPE | R² | Sai số RUL (chu kỳ) | Giải thích được? | Khoảng dự báo? | Thời gian train |
|---|---|---|---|---|---|---|---|---|
| Naive | | | | | | – | Không | 0 |
| Hồi quy log-linear | | | | | | Cao | Không | |
| ARIMA / ARIMAX | | | | | | Trung bình | Có (E1 và E2) | |
| SVR | | | | | | Trung bình | Chỉ E1 (conformal) | |
| LSTM | | | | | | Thấp | Không | |
| Hybrid Bayes (thêm 21/11/2026) | | | | | | Cao (tham số đường cong) | Có (E1 và E2) | |

**Thêm:**
- Sai số tách theo từng pin test (tức theo từng điều kiện vận hành), để biết mô hình nào yếu ở điều kiện nào.
- **Sensitivity của cảnh báo EOL (Chap 4):** từ dự báo của mỗi mô hình, suy ra cảnh báo "pin chạm SOH 0,8 trong 20 chu kỳ tới (có/không)". Tính sensitivity, specificity và confusion matrix cho từng mô hình. Không fit thêm mô hình thứ 5. Đây là phần đáp ứng mục "sensitivity" trong yêu cầu so sánh mô hình của đề cương. *Thay đổi ngày 21/11/2026: có thêm mô hình thứ 5 (Rec.md mục 8).*

**Kết luận theo từng RQ:** mỗi RQ trả lời bằng 2–3 câu, mỗi câu đều có con số kèm theo.

**Kết quả thật** (bảng số, RUL, cảnh báo EOL và chú thích về cách fit của ARIMA) ở `notebooks/60_compare.ipynb` và `reports/tables/compare_*.csv`. Cột "Khoảng dự báo?" ở bảng trên đã sửa ngày 21/11/2026 theo các file dự báo thật.

---

## GĐ8. Chuyển kết quả thành khuyến nghị ⏳

- **Ai:** Khải Hưng (R), mỗi người đóng góp phần insight từ mô hình của mình.
- **Quy tắc:** **mỗi khuyến nghị phải truy được về một bảng hoặc một hình**. Trình bày theo khung:

| Insight | Bằng chứng (bảng/hình) | Khuyến nghị | Bên liên quan | Mức tin cậy |
|---|---|---|---|---|
| Ở 43°C/4 A pin suy giảm nhanh hơn ở 24°C/4 A trên 25 chu kỳ đầu (27,8 so với 20,1 điểm %/100 chu kỳ), nhưng **chưa có ý nghĩa thống kê** (p = 0,105). Không dùng tỉ số trên toàn chuỗi vì bị thiên lệch: G3 chỉ chạy 39 chu kỳ, B0033/B0034 chạy khoảng 190 chu kỳ | `02_inferential` T1, `fig1` | Làm mát pack, dựa trên cơ chế đã biết hơn là trên dữ liệu này | Hãng xe, đơn vị bảo hành | Thấp (n = 4 so với 2) |
| Ở 4°C với tải 4 A, pin gần như không xả được (cutoff ≥ 2,5 V) | F3, `fig4` | Sưởi ấm pin trước khi vận hành tải nặng trong trời lạnh; giảm tải khi pin lạnh | Đội xe, drone, người dùng | Cao (hiệu ứng rất lớn) |
| Thời gian pha CC khi sạc tương quan 0,86 với SOH ở G1; nhưng khi đã biết SOH gần đây thì đặc trưng sạc gần như không giúp dự báo chu kỳ kế tiếp | F7, `30_ml` (RQ4) | BMS ước lượng **gần đúng** SOH khi sạc, dùng khi không có phép xả kiểm tra; không thay được lịch sử đo dung lượng | Nhà vận hành đội xe | Trung bình (chỉ G1) |
| Dung lượng hồi lại tạm thời sau khi nghỉ dài | F5, `fig3` | Không kết luận "pin đã hồi phục" chỉ từ một lần đo sau khi nghỉ; đo lại SOH sau vài chu kỳ | Bộ phận bảo hành, kiểm định pin cũ | Cao |
| Thời điểm chạm SOH 0,8 dự báo được kèm khoảng tin cậy | GĐ5 E2 | Lập lịch thay pin chủ động; kiểm tra top 10% pin rủi ro trước (lift) | Đội xe | Tùy kết quả mô hình |

**Chính sách bảo hành:**
- Lấy `cycles_to_soh80` theo từng điều kiện, rồi lấy phân vị thấp (gọi là "B10 life") làm mức bảo hành an toàn.
- Quy ra tiền bằng **giả định có trích nguồn**, ví dụ giá pack $/kWh từ báo cáo ngành. Làm phân tích độ nhạy ±30%.

**Người dùng cuối:**
- Chỉ khuyến nghị những gì dữ liệu cho phép: nhiệt độ vận hành, mức tải, độ sâu xả (qua cutoff), thời gian nghỉ.
- **Không đưa ra khuyến nghị về sạc nhanh**, vì dataset sạc cố định CC 1,5 A.

---

## GĐ9. Viết các deliverable ⏳

Phân công chi tiết theo chương nằm ở [WBS.md](WBS.md). Phần dưới đây mô tả **cách viết**.

### 9.1 Full Report (~30 trang), theo 8 mục của đề cương

| Chương | Số trang | Nội dung chính | Lấy từ |
|---|---|---|---|
| 1. Introduction | 3 | Bối cảnh xe điện và lưu trữ năng lượng, chi phí pin; mục tiêu; RQ1–RQ6; đóng góp | Go.md §1 |
| 2. Literature Review | 4 | SOH/RUL prognostics; các nghiên cứu dùng dataset NASA PCoE; yếu tố gây lão hóa (nhiệt độ, dòng xả, độ sâu xả); analytics cho bảo hành và đội xe; khoảng trống nghiên cứu | 15–25 nguồn |
| 3. Research Methodology | 6 | Mô tả dataset và tiền xử lý (Introduction_of_dataset.md); công thức và giả định của từng phương pháp | GĐ1, GĐ4 |
| 4. Implementation & Experiments | 7 | Workflow, code có giải thích, kiểm định giả định, các hình (histogram, boxplot, scatter, SEM path, dự báo) | GĐ2–GĐ6 |
| 5. Results & Discussion | 6 | Kết quả theo thứ tự Descriptive → Inferential → Regression → SEM → Forecasting → ML/DL; bảng so sánh; ý nghĩa thực tế | GĐ7, GĐ8 |
| 6. Conclusion & Recommendations | 2 | Tóm tắt phát hiện, khuyến nghị, hạn chế, hướng phát triển (Explainable AI, Ethical AI) | GĐ8; Mục 10 của Introduction_of_dataset.md |
| 7. References | 1–2 | Định dạng IEEE hoặc APA | Zotero chung của nhóm |
| 8. Appendix | – | Link GitHub và video demo, ảnh Turnitin, ảnh kiểm tra AI | GĐ10 |

**Quy tắc viết:**
- Mọi con số trong báo cáo phải lấy từ notebook hoặc script, **không gõ tay**.
- Công thức lấy theo slide và trích dẫn các sách tham khảo ở Chap 0, tr. 7 (Groebner; Evans; Bruce & Gedeck; Nielsen).
- Mỗi hình có caption gồm: hình cho thấy điều gì, và rút ra kết luận gì.

### 9.2 IEEE Paper (7–10 trang)
- Dùng template IEEE Conference trên Overleaf, định dạng hai cột.
- Rút gọn từ report: Abstract (khoảng 200 từ), Keywords, Introduction, Related Work, Methodology, Results, Discussion, Conclusion, References.
- Chỉ giữ những gì mạnh nhất: 1 bảng kiểm định, 1 bảng hệ số hồi quy, 1 bảng so sánh mô hình, và 4 hình (`fig1`, `fig2`, hình dự báo, hình SEM).

### 9.3 Slides (15–20 trang)
- Kể theo mạch Descriptive → Prescriptive. Mỗi slide một ý, tiêu đề là chính kết luận.
- Ví dụ tiêu đề: "Lạnh và tải nặng làm pin gần như không xả được". Con số trong tiêu đề phải lấy từ kết quả kiểm định cuối cùng.

### 9.4 Demo video và Streamlit app (đã chốt ở Rec.md mục 7)
- **Làm cả hai, nhưng ưu tiên thấp:** chỉ bắt đầu khi đã xong các phần chính.
- **Demo video 8–10 phút, mỗi thành viên tự trình bày phần của mình:**

| Phút | Nội dung | Người trình bày |
|---|---|---|
| 0:00–1:00 | Bài toán và ý nghĩa kinh doanh | Khải Hưng |
| 1:00–2:30 | Dữ liệu gốc (file `.mat`, vấn đề cột `Capacity` 2,7 V); chạy pipeline tiền xử lý | Khải Hưng |
| 2:30–3:30 | Các hình mô tả chính và kết quả kiểm định | Minh Vũ, Tiến Hưng |
| 3:30–6:30 | 4 mô hình, mỗi mô hình khoảng 45 giây: chạy notebook, hình dự báo, một con số chính | Từng người phụ trách |
| 6:30–7:30 | SEM: sơ đồ đường dẫn | Quốc Đạt |
| 7:30–8:30 | Bảng so sánh mô hình | Khải Hưng |
| 8:30–9:30 | Khuyến nghị cho doanh nghiệp và người dùng | Cả nhóm |

- **Streamlit app:** chọn một pin, xem SOH thực tế và dự báo, cảnh báo sắp chạm EOL.

### 9.5 GitHub
- README có phần chạy lại từ đầu: `pip install -r requirements.txt`, tải dataset vào `data/raw/`, chạy 3 script và các notebook theo thứ tự số.
- Không commit dữ liệu.
- Đổi tên repo thành `STAT3013.R11.CTTT_Group08`.

---

## GĐ10. Kiểm tra, tập dượt và nộp bài ⏳

### 10.1 Turnitin ≤ 10% và kiểm tra AI
- Mỗi người tự viết phần của mình. Diễn giải lại các định nghĩa thay vì chép. Trích dẫn đầy đủ.
- Nộp thử bản nháp lên Turnitin **trước hạn ít nhất 3 ngày**.
- **Kiểm tra AI:** chụp ảnh kết quả từ Grammarly, Quillbot hoặc công cụ tương tự.
  - Viết một trang giải trình: phần nào có dùng AI hỗ trợ (ví dụ debug code, sửa ngữ pháp) và đã kiểm chứng như thế nào.
  - Đề cương ghi rõ: "AI-generated content must be checked and explained".

### 10.2 Tập dượt bảo vệ
Mỗi người phải trả lời được mọi câu dưới đây, không chỉ câu về phần mình.

| Câu hỏi | Ý chính của câu trả lời |
|---|---|
| Vì sao dùng t mà không dùng z? | Mỗi nhóm có n < 30 pin và σ chưa biết (Chap 2, tr. 8) |
| Vì sao đơn vị quan sát là pin? | Các chu kỳ trong một pin tự tương quan tới 0,93, vi phạm giả định độc lập |
| Vì sao không chia train/test ngẫu nhiên? | Gây rò rỉ dữ liệu, R² ảo khoảng 0,99 |
| Vì sao không làm two-way ANOVA? | Nhiều ô không có pin nào, cỡ mẫu không bằng nhau |
| Cột `Capacity` gốc có vấn đề gì? | Là dung lượng đến 2,7 V; ở 4°C/4 A gây ra giá trị thấp giả tạo |
| Vì sao chọn d = 1? | ADF: chuỗi gốc p > 0,5, chuỗi sai phân p < 0,001 |
| VIF là gì, vì sao không đưa Re và Rct cùng lúc? | VIF đo mức một biến được giải thích bởi các biến khác (Chap 3, tr. 41). r(Re, Rct) = 0,97 trong G1 nhưng 0,43 trên tier A, nên chỉ bỏ một biến khi VIF trên tập train > 10 |
| Vì sao LSTM không thắng? (nếu đúng như vậy) | Dữ liệu nhỏ khoảng 1.500 dòng (Chap 6, tr. 13) |
| Khuyến nghị về sạc pin dựa trên gì? | Dataset không có dữ liệu sạc nhanh; chỉ nói về nhiệt độ, tải và độ sâu xả |

### 10.3 Checklist nộp bài
- [ ] Full Report (PDF), nộp trên Moodle
- [ ] IEEE Paper (PDF), nộp trên Moodle
- [ ] Slides, nộp trên Moodle
- [ ] Link GitHub (đã đổi tên repo `STAT3013.R11.CTTT_Group08`)
- [ ] Link demo video
- [ ] Link Streamlit app (nếu kịp)
- [ ] Ảnh Turnitin ≤ 10%
- [ ] Ảnh kiểm tra AI và trang giải trình

---

## 11. Rủi ro và cách xử lý

| Rủi ro | Khả năng | Ảnh hưởng | Cách xử lý |
|---|---|---|---|
| Kiểm định không có ý nghĩa do quá ít pin | Cao | Trung bình | Báo cáo khoảng tin cậy và effect size; nêu rõ trong phần Limitations; không nói quá kết luận |
| Chap 7 (SEM) được dạy muộn | Trung bình | Trung bình | Chuẩn bị sẵn dữ liệu và sơ đồ; chạy ngay sau buổi học |
| Mô hình DL hoặc ML không thắng baseline | Trung bình | Thấp | Đây vẫn là kết quả hợp lệ; giải thích bằng kích thước dữ liệu |
| Bốn mô hình đánh giá khác nhau nên không so sánh được | Trung bình | Cao | Bắt buộc dùng chung giao thức GĐ4 và `src/protocol.py` |
| Turnitin > 10% | Thấp | Cao | Viết sớm, nộp thử sớm, trích dẫn đúng |
| Một thành viên chậm tiến độ | Trung bình | Cao | Họp 2 lần mỗi tuần; Leader theo dõi WBS; người review chéo hỗ trợ |
