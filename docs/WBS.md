# WBS: Phân công công việc đồ án STAT3013

> **Đề tài:** Statistical and Machine Learning-Based Modeling of Lithium-Ion Battery Degradation and State-of-Health Prognostics under Variable Operating Conditions
> **Lớp:** STAT3013.R11.CTTT · **Nhóm:** Group 08 (4AnhTai) · **Dataset:** NASA PCoE Battery Aging. Xem [Introduction_of_dataset.md](Introduction_of_dataset.md).
> **Kế hoạch chi tiết từng bước:** [Go.md](Go.md). Các mã GĐx trong tài liệu này trỏ tới giai đoạn tương ứng trong Go.md.
> **Các quyết định đã chốt:** [Rec.md](Rec.md). Khi mâu thuẫn, Rec.md là căn cứ đúng.
> **Thời gian:** 23/10/2026 – 28/11/2026. Hạn nộp chính xác xem trên Moodle.

---

## 1. Nguyên tắc phân công

1. **Giai đoạn chung (GĐ1–GĐ4): cả 4 người cùng làm.**
   - Gồm hiểu dữ liệu, thống kê mô tả, thống kê suy diễn và chốt giao thức mô hình.
   - Mỗi mục có một người *lead*, chịu trách nhiệm tổng hợp và đảm bảo đúng hạn, nhưng cả nhóm cùng làm.
2. **Giai đoạn modeling (GĐ5): mỗi người một thuật toán.**
   - 2 người làm thống kê truyền thống, 1 người làm ML, 1 người làm DL.
   - Cả 4 người dùng **chung dữ liệu, chung cách chia train/test, chung metric** (GĐ4 trong Go.md).
3. **Review chéo bắt buộc.**
   - Mỗi kết quả phải có một người khác kiểm tra trước khi merge vào `main`.
   - Cặp review được xếp sao cho mỗi người hiểu ít nhất một mô hình ngoài mô hình của mình.
4. **Ai cũng phải hiểu toàn bộ dự án.** Đề cương ghi "All members must understand the entire project". Vì vậy ở mỗi buổi họp, mỗi người trình bày lại phần của **một người khác**.

---

## 2. Bảng phân công vai trò

| Thành viên | Vai trò | Thuật toán phụ trách (GĐ5) | Chương lý thuyết chính | Lead giai đoạn chung | Review chéo cho | Lead deliverable |
|---|---|---|---|---|---|---|
| **Khải Hưng** | **Leader**, Deep Learning | **LSTM** | Chap 4 (metric phân loại), Chap 6 (§3–4) | GĐ1 (tiền xử lý), GĐ4 (giao thức), GĐ7 (so sánh, sensitivity), GĐ8 (insight) | Minh Vũ (ML) | Full Report (biên tập tổng), GitHub, nộp bài |
| **Quốc Đạt** | Mô hình thống kê truyền thống | **Hồi quy bội log-linear có tương tác** và **SEM** | Chap 3, Chap 7 | Literature Review | Tiến Hưng (ARIMA) | IEEE Paper |
| **Tiến Hưng** | Thống kê truyền thống | **ARIMA / ARIMAX** | Chap 2, Chap 5 | GĐ3 (thống kê suy diễn), kiểm tra Turnitin/AI | Quốc Đạt (hồi quy) | Kiểm tra chất lượng (QA) và bằng chứng Turnitin/AI |
| **Minh Vũ** | Machine Learning | **SVR** (Isolation Forest chỉ dùng để kiểm tra chéo ở bước tiền xử lý) | Chap 1, Chap 6 (§1–2, §5) | GĐ2 (thống kê mô tả) | Khải Hưng (DL) | Slides, Demo video, Streamlit app |

**Trách nhiệm riêng của Leader (Khải Hưng):**
- Giữ lịch WBS và cập nhật trạng thái sau mỗi buổi họp.
- Duy trì pipeline `src/` và `src/protocol.py`.
- Merge các Pull Request vào `main`.
- Ghép các phần thành báo cáo cuối.
- Đổi tên repo thành `STAT3013.R11.CTTT_Group08` và nộp bài trên Moodle.

---

## 3. Ma trận RACI theo deliverable

Ký hiệu: **R** = người làm chính · **A** = người chịu trách nhiệm cuối và duyệt · **C** = người đóng góp nội dung · **I** = người được thông báo.

| Deliverable | Khải Hưng | Quốc Đạt | Tiến Hưng | Minh Vũ |
|---|---|---|---|---|
| Pipeline dữ liệu (`src/`) | **R/A** | C | C | C |
| Thống kê mô tả | C | C | C | **R** (A: Khải Hưng) |
| Thống kê suy diễn | C | C | **R** (A: Khải Hưng) | C |
| Hồi quy bội log-linear | I | **R** | C (review) | I |
| ARIMA / ARIMAX | I | C (review) | **R** | I |
| SVR (và Isolation Forest kiểm tra dữ liệu) | C (review) | I | I | **R** |
| LSTM | **R** | I | I | C (review) |
| SEM | I | **R** | C | I |
| So sánh mô hình và insight | **R/A** | C | C | C |
| Full Report (~30 trang) | **R/A** (biên tập) | C | C | C |
| IEEE Paper (7–10 trang) | A | **R** | C | C |
| Slides | A | C | C | **R** |
| Demo video | C | I | I | **R** (A: Khải Hưng) |
| Streamlit app (ưu tiên thấp) | C | I | I | **R** (A: Khải Hưng) |
| GitHub (README, cấu trúc repo) | **R/A** | C | C | C |
| Turnitin ≤ 10% và kiểm tra AI | A | C | **R** | C |

---

## 4. WBS theo chương lý thuyết của môn học

Mỗi chương của môn học được chuyển thành các gói công việc cụ thể trên dataset. Cột "Output" ghi rõ file cần tạo.

### Chap 1: Overview of Data Analytics & Descriptive Statistics

| Mã | Công việc | R | Hỗ trợ | Bắt đầu | Kết thúc | Output | Xong khi |
|---|---|---|---|---|---|---|---|
| C1.1 | Phân loại dữ liệu: cấu trúc, panel data, thang đo của từng biến | Minh Vũ | Cả nhóm | 26/10 | 27/10 | Bảng trong Report §3 | Mỗi biến trong từ điển dữ liệu có thang đo kèm lý do |
| C1.2 | Parse `.mat` thành bảng, gộp và ghép dữ liệu ⏳ (làm lại từ đầu) | Khải Hưng | – | 23/10 | 25/10 | `src/build_dataset.py`, `cycles.csv` | Chạy được, kết quả ghi vào Work_log |
| C1.3 | Làm sạch dữ liệu, chia segment, tính SOH, phân tầng pin ⏳ (làm lại từ đầu, theo quyết định ở Rec.md mục 2) | Khải Hưng | – | 23/10 | 25/10 | `src/clean_dataset.py`, `cycles_clean.csv` | Chạy được, đúng các tham số đã chốt, chờ review |
| C1.4 | Review quy tắc làm sạch (mỗi người một phần, Go.md bước 1.6) | Cả nhóm | – | 26/10 | 27/10 | Biên bản review trong issue GitHub | Cả 4 người đồng ý |
| C1.5 | Bảng thống kê mô tả ở mức chu kỳ và mức pin, tách theo điều kiện | Minh Vũ | Tiến Hưng | 27/10 | 29/10 | `notebooks/01_descriptive.ipynb` | Có đủ n, mean, median, SD, SE, skewness, kurtosis (đã +3) |
| C1.6 | Histogram, boxplot, đường SOH, scatter trở kháng | Minh Vũ | Khải Hưng | 27/10 | 29/10 | `reports/figures/` | Mỗi hình có 2–3 câu diễn giải |
| C1.7 | Viết insight mô tả theo mẫu slide tr. 60 | Minh Vũ | Cả nhóm | 29/10 | 29/10 | Nháp Report §4 | Mỗi insight có con số và hàm ý |

### Chap 2: Analysis and Interpretation of Data (thống kê suy diễn)

| Mã | Công việc | R | Hỗ trợ | Bắt đầu | Kết thúc | Output | Xong khi |
|---|---|---|---|---|---|---|---|
| C2.1 | CI 95% cho tốc độ suy giảm theo nhóm (t, df = n−1) và CI cho tỉ lệ hồi phục (z) | Tiến Hưng | Quốc Đạt | 29/10 | 30/10 | `notebooks/02_inferential.ipynb` | Có bảng CI kèm diễn giải |
| C2.2 | F-test và Levene cho từng cặp so sánh | Tiến Hưng | Minh Vũ | 30/10 | 31/10 | Như trên | Đã chọn được t-test pooled hoặc Welch cho từng cặp |
| C2.3 | t-test T1–T4 (nhiệt độ ở 4 A, nhiệt độ ở 2 A, dòng xả ở 4°C, profile) | Tiến Hưng | Cả nhóm | 30/10 | 01/11 | Như trên | Mỗi kiểm định có H0/Ha, T, df, p, kết luận, chạy trên **cả hai** biến tốc độ suy giảm |
| C2.4 | One-way ANOVA theo nhóm điều kiện và Tukey HSD | Quốc Đạt | Tiến Hưng | 31/10 | 01/11 | Như trên | Có bảng ANOVA dạng như slide tr. 83 |
| C2.5 | Đoạn giải thích vì sao không làm được two-way ANOVA, kèm bảng số pin theo ô | Tiến Hưng | – | 01/11 | 01/11 | Nháp Report §4 | Có bảng ô và lý do |
| C2.6 | Chi-square: nhóm nhiệt độ × "SOH < 0,9 trong 40 chu kỳ" | Minh Vũ | Tiến Hưng | 01/11 | 02/11 | Như trên | Có bảng O_ij, E_ij và cảnh báo nếu E < 5 |
| C2.7 | Tương quan giữa tốc độ suy giảm và nhiệt độ cell (mức pin) | Khải Hưng | Tiến Hưng | 01/11 | 02/11 | Như trên | Có scatter và r kèm p |
| C2.8 | Bảng tổng hợp các kiểm định và ý nghĩa kinh doanh | Tiến Hưng | Cả nhóm | 02/11 | 02/11 | Report §5.2 | Đã qua review của Leader |

### Chap 3: Analysis of Regression (Quốc Đạt)

| Mã | Công việc | R | Review | Bắt đầu | Kết thúc | Output | Xong khi |
|---|---|---|---|---|---|---|---|
| C3.1 | Ma trận tương quan và VIF; Re–Rct có r = 0,97 trong G1 nhưng 0,43 trên tier A, phải kiểm tra lại trên tập train | Quốc Đạt | Tiến Hưng | 03/11 | 04/11 | `notebooks/10_regression.ipynb` | Có bảng VIF |
| C3.2 | Hồi quy đơn `soh ~ cycle` (G1) và residual plot | Quốc Đạt | Tiến Hưng | 04/11 | 05/11 | Như trên | Có R², F-test, t-test, CI của hệ số dốc |
| C3.3 | Mô hình log-linear và polynomial bậc 2, so sánh bằng adjusted R² | Quốc Đạt | Tiến Hưng | 05/11 | 06/11 | Như trên | Đã chọn được dạng mô hình kèm lý do |
| C3.4 | Hồi quy bội với dummy và tương tác cycle × điều kiện | Quốc Đạt | Tiến Hưng | 06/11 | 09/11 | Như trên | Có bảng hệ số kèm CI và diễn giải dạng "x% mỗi chu kỳ" |
| C3.5 | Chọn biến theo quy trình 4 bước (tr. 39) | Quốc Đạt | Tiến Hưng | 09/11 | 10/11 | Như trên | Bảng adjusted R² sau từng lần bỏ biến |
| C3.6 | Hồi quy 2 bước trên `battery_fade.csv` (n = 23) | Quốc Đạt | Tiến Hưng | 10/11 | 11/11 | Như trên | So sánh kết quả với C3.4 |
| C3.7 | Đánh giá theo giao thức chung E1/E2, xuất file dự báo | Quốc Đạt | Khải Hưng | 11/11 | 13/11 | `reports/predictions/regression.csv` | Notebook so sánh đọc được file |
| C3.8 | Viết phần Phương pháp và Kết quả cho hồi quy | Quốc Đạt | Tiến Hưng | 13/11 | 15/11 | Report §3.3, §5.3 | Có công thức, giả định, kết quả |

### Chap 4: Metric phân loại cho cảnh báo EOL (Khải Hưng)

> Không làm mô hình logistic (đã loại, xem Rec.md mục 4). Chap 4 chỉ được dùng cho phần **đánh giá**: suy ra cảnh báo EOL từ dự báo của 4 mô hình, để đáp ứng mục "sensitivity" trong yêu cầu so sánh mô hình.

| Mã | Công việc | R | Review | Bắt đầu | Kết thúc | Output | Xong khi |
|---|---|---|---|---|---|---|---|
| C4.1 | Từ dự báo của mỗi mô hình, suy ra cảnh báo "chạm SOH 0,8 trong 20 chu kỳ tới (có/không)"; tính sensitivity, specificity và confusion matrix cho cả 4 mô hình | Khải Hưng | Quốc Đạt | 16/11 | 18/11 | `notebooks/60_compare.ipynb` | Có bảng sensitivity/specificity của 4 mô hình, cùng định nghĩa nhãn |

### Chap 5: Forecasting Model of Time Series (Tiến Hưng)

| Mã | Công việc | R | Review | Bắt đầu | Kết thúc | Output | Xong khi |
|---|---|---|---|---|---|---|---|
| C5.1 | Vẽ chuỗi dung lượng theo chu kỳ của từng pin tier A | Tiến Hưng | Quốc Đạt | 03/11 | 04/11 | `notebooks/20_arima.ipynb` | Có nhận xét trend, regeneration, seasonality |
| C5.2 | ADF trên chuỗi gốc và sai phân; ACF/PACF | Tiến Hưng | Quốc Đạt | 04/11 | 05/11 | Như trên | Xác nhận d = 1 |
| C5.3 | Nhận dạng (p, q) bằng ACF/PACF và lưới AIC với p, q ∈ {0, 1, 2} | Tiến Hưng | Quốc Đạt | 05/11 | 07/11 | Như trên | Có bảng (p, d, q, AIC) theo từng pin |
| C5.4 | Chẩn đoán phần dư: ACF và Ljung-Box | Tiến Hưng | Quốc Đạt | 07/11 | 08/11 | Như trên | Phần dư không còn tự tương quan đáng kể |
| C5.5 | ARIMAX với `log1p(rest_before_h)` và `Rct_ohm` | Tiến Hưng | Quốc Đạt | 08/11 | 11/11 | Như trên | Có bảng so sánh AIC/RMSE giữa ARIMA và ARIMAX |
| C5.6 | Kiểm tra seasonality và kết luận về SARIMA | Tiến Hưng | – | 11/11 | 11/11 | Như trên | Có một đoạn giải thích |
| C5.7 | Dự báo theo E1 (cuốn chiếu) và E2 (kèm khoảng tin cậy), xuất file | Tiến Hưng | Khải Hưng | 11/11 | 13/11 | `reports/predictions/arima.csv` | Notebook so sánh đọc được file |
| C5.8 | Viết phần Phương pháp và Kết quả cho ARIMA | Tiến Hưng | Quốc Đạt | 13/11 | 15/11 | Report §3.4, §5.4 | Có các bước Box-Jenkins kèm hình |

### Chap 6: Machine Learning, Deep Learning & Forecasting Issues

**ML: Minh Vũ**

| Mã | Công việc | R | Review | Bắt đầu | Kết thúc | Output | Xong khi |
|---|---|---|---|---|---|---|---|
| C6.1 | Isolation Forest kiểm tra chéo các cờ outlier ở bước tiền xử lý (Go.md bước 1.5; không phải thuật toán modeling) | Minh Vũ | Khải Hưng | 27/10 | 30/10 | `notebooks/31_iforest.ipynb` | Có bảng so khớp Hampel với iForest |
| C6.2 | Tạo đặc trưng lag và rolling; Pipeline gồm scaler và mô hình | Minh Vũ | Khải Hưng | 03/11 | 05/11 | `notebooks/30_ml.ipynb` | Không có đặc trưng nào nằm trong danh sách cấm |
| C6.3 | SVR: grid search với GroupKFold theo pin | Minh Vũ | Khải Hưng | 05/11 | 08/11 | Như trên | Có bảng siêu tham số tốt nhất |
| C6.4 | Permutation importance cho SVR trên tập test | Minh Vũ | Khải Hưng | 08/11 | 10/11 | Như trên | Có hình importance, đối chiếu với hệ số của C3.4 |
| C6.5 | Đánh giá E1/E2, so với baseline naive, xuất file | Minh Vũ | Khải Hưng | 10/11 | 13/11 | `reports/predictions/svr.csv` | Notebook so sánh đọc được file |
| C6.6 | Viết phần Phương pháp và Kết quả cho ML | Minh Vũ | Khải Hưng | 13/11 | 15/11 | Report §3.5, §5.5 | Có 8 bước SVR và kết quả |

**DL: Khải Hưng**

| Mã | Công việc | R | Review | Bắt đầu | Kết thúc | Output | Xong khi |
|---|---|---|---|---|---|---|---|
| C6.7 | Tạo cửa sổ chuỗi dài 10, không cắt qua ranh giới giữa hai pin | Khải Hưng | Minh Vũ | 04/11 | 05/11 | `notebooks/40_dl.ipynb` | Có hàm kiểm tra (assert) về ranh giới pin |
| C6.8 | LSTM (1 lớp, 32 đơn vị), early stopping, chạy với 5 seed | Khải Hưng | Minh Vũ | 05/11 | 10/11 | Như trên | Có learning curve, kết quả mean ± SD |
| C6.9 | Đánh giá E1/E2 (E2 đệ quy), xuất file | Khải Hưng | Minh Vũ | 10/11 | 13/11 | `reports/predictions/lstm.csv` | Notebook so sánh đọc được file |
| C6.10 | Viết phần Phương pháp và Kết quả cho DL | Khải Hưng | Minh Vũ | 13/11 | 15/11 | Report §3.6, §5.6 | Có kiến trúc, quá trình train và diễn giải trung thực |

### Chap 7: Structural Equation Modeling (Quốc Đạt và Tiến Hưng, sau khi học Chap 7)

| Mã | Công việc | R | Hỗ trợ | Bắt đầu | Kết thúc | Output | Xong khi |
|---|---|---|---|---|---|---|---|
| C7.1 | Vẽ sơ đồ giả thuyết: điều kiện → Re/Rct → SOH (làm trước được) | Quốc Đạt | Tiến Hưng | 09/11 | 10/11 | Hình sơ đồ | Cả nhóm đồng ý với các đường dẫn |
| C7.2 | Chuẩn bị dữ liệu: các chu kỳ có lần đo EIS mới | Tiến Hưng | Quốc Đạt | 10/11 | 11/11 | `notebooks/50_sem.ipynb` | Có bảng dữ liệu đầu vào |
| C7.3 | Chạy path analysis (semopy hoặc lavaan) | Quốc Đạt | Tiến Hưng | sau buổi học Chap 7 (dự kiến 16/11) | 19/11 | Như trên | Có RMSEA, CFI, TLI và hệ số đường dẫn |
| C7.4 | Hiệu ứng gián tiếp (mediation) và diễn giải | Quốc Đạt | Tiến Hưng | 18/11 | 19/11 | Report §5.7 | Có câu dạng "x% tác động đi qua Rct" |

---

## 5. WBS theo chương của Full Report

| Chương report | Mục | Người viết | Người review | Nguồn nội dung | Hạn bản nháp | Hạn bản cuối |
|---|---|---|---|---|---|---|
| **1. Introduction** (3 tr.) | Bối cảnh xe điện và lưu trữ năng lượng, chi phí pin | Khải Hưng | Quốc Đạt | Báo cáo ngành | 08/11 | 22/11 |
| | Mục tiêu, RQ1–RQ6, đóng góp | Khải Hưng | Tiến Hưng | Go.md §1 | 08/11 | 22/11 |
| **2. Literature Review** (4 tr.) | Tổng quan về SOH/RUL prognostics | Quốc Đạt | Khải Hưng | Paper | 08/11 | 22/11 |
| | Yếu tố gây lão hóa: nhiệt độ, dòng xả, độ sâu xả | Tiến Hưng | Quốc Đạt | Paper | 08/11 | 22/11 |
| | Các nghiên cứu ML/DL trên dataset NASA PCoE | Minh Vũ | Khải Hưng | Paper | 08/11 | 22/11 |
| | Analytics cho bảo hành và đội xe; khoảng trống nghiên cứu | Khải Hưng | Minh Vũ | Paper, báo cáo ngành | 08/11 | 22/11 |
| **3. Research Methodology** (6 tr.) | §3.1 Dataset và tiền xử lý | Khải Hưng | Minh Vũ | Introduction_of_dataset.md | 06/11 | 22/11 |
| | §3.2 Thống kê mô tả và suy diễn | Tiến Hưng | Minh Vũ | GĐ2–GĐ3 | 06/11 | 22/11 |
| | §3.3 Hồi quy bội log-linear | Quốc Đạt | Tiến Hưng | C3 | 15/11 | 22/11 |
| | §3.4 ARIMA / ARIMAX | Tiến Hưng | Quốc Đạt | C5 | 15/11 | 22/11 |
| | §3.5 SVR | Minh Vũ | Khải Hưng | C6.2–C6.6 | 15/11 | 22/11 |
| | §3.6 LSTM | Khải Hưng | Minh Vũ | C6.7–C6.10 | 15/11 | 22/11 |
| | §3.7 SEM | Quốc Đạt | Tiến Hưng | C7 | 19/11 | 22/11 |
| **4. Implementation & Experiments** (7 tr.) | Workflow, cấu trúc code, kiểm tra giả định, hình | Mỗi người viết phần mô hình của mình; Khải Hưng ghép | Chéo | Notebook | 15/11 | 23/11 |
| **5. Results & Discussion** (6 tr.) | §5.1 Mô tả · §5.2 Suy diễn | Minh Vũ · Tiến Hưng | Chéo | GĐ2–GĐ3 | 08/11 | 23/11 |
| | §5.3–§5.7 Kết quả từng mô hình | Người phụ trách mô hình đó | Chéo | GĐ5–GĐ6 | 15/11 (SEM: 19/11) | 23/11 |
| | §5.8 So sánh mô hình (kèm sensitivity của cảnh báo EOL) | Khải Hưng | Cả nhóm | GĐ7, C4.1 | 18/11 | 23/11 |
| | §5.9 Ý nghĩa thực tế | Khải Hưng | Cả nhóm | GĐ8 | 20/11 | 23/11 |
| **6. Conclusion & Recommendations** (2 tr.) | Tóm tắt, khuyến nghị, hạn chế, hướng phát triển | Khải Hưng | Quốc Đạt | GĐ8 | 20/11 | 24/11 |
| **7. References** | Quản lý trích dẫn bằng Zotero, định dạng IEEE | Quốc Đạt | Tiến Hưng | – | liên tục | 24/11 |
| **8. Appendix** | Link GitHub và video, ảnh Turnitin, ảnh kiểm tra AI | Tiến Hưng | Khải Hưng | GĐ10 | – | 26/11 |

---

## 6. WBS các deliverable khác

| Mã | Deliverable | R | Hỗ trợ | Bắt đầu | Kết thúc | Xong khi |
|---|---|---|---|---|---|---|
| D1 | IEEE Paper 7–10 trang (Overleaf) | Quốc Đạt | Cả nhóm | 19/11 | 25/11 | Đúng template 2 cột, có 4 bảng/hình chính, không quá 10 trang |
| D2 | Slides 15–20 trang | Minh Vũ | Cả nhóm | 20/11 | 25/11 | Mỗi slide một ý, tiêu đề là kết luận |
| D3 | Demo video 8–10 phút, mỗi thành viên tự trình bày phần của mình (ưu tiên thấp) | Minh Vũ | Khải Hưng | 22/11 | 26/11 | Theo kịch bản ở Rec.md mục 7, có link xem được |
| D3b | Streamlit app: chọn pin, xem SOH thực tế và dự báo, cảnh báo EOL (ưu tiên thấp, chỉ làm khi xong phần chính) | Minh Vũ | Khải Hưng | 22/11 | 26/11 | Chạy được bằng `streamlit run`, có link hoặc hướng dẫn chạy |
| D4 | GitHub: README, đổi tên repo, dọn code | Khải Hưng | Cả nhóm | 20/11 | 26/11 | Người ngoài clone về chạy lại được từ đầu |
| D5 | Turnitin ≤ 10% | Tiến Hưng | Cả nhóm | 23/11 | 26/11 | Có ảnh chụp kết quả ≤ 10% |
| D6 | Kiểm tra AI và trang giải trình | Tiến Hưng | Cả nhóm | 23/11 | 26/11 | Có ảnh chụp và một trang giải trình |
| D7 | Tập dượt bảo vệ (bộ câu hỏi ở Go.md §10.2) | Khải Hưng | Cả nhóm | 25/11 | 27/11 | Mỗi người trả lời được mọi câu |
| D8 | Nộp bài trên Moodle | Khải Hưng | – | 27/11 | 28/11 | Đã nộp đủ các mục trong checklist ở Go.md §10.3 |

---

## 7. Lịch theo tuần (Gantt)

| Hạng mục | T1 (26/10–01/11) | T2 (02/11–08/11) | T3 (09/11–15/11) | T4 (16/11–22/11) | T5 (23/11–28/11) |
|---|---|---|---|---|---|
| Review dữ liệu (C1.4) | ■ | | | | |
| Thống kê mô tả (C1.5–C1.7) | ■ | | | | |
| Thống kê suy diễn (C2) | ■ | ■ | | | |
| Giao thức chung (GĐ4) | | ■ | | | |
| Hồi quy log-linear: Quốc Đạt | | ■ | ■ | | |
| ARIMA/ARIMAX: Tiến Hưng | | ■ | ■ | | |
| SVR: Minh Vũ (iForest kiểm tra dữ liệu từ T1) | ■ | ■ | ■ | | |
| LSTM: Khải Hưng | | ■ | ■ | | |
| SEM: Quốc Đạt và Tiến Hưng | | | ■ (chuẩn bị) | ■ | |
| So sánh mô hình, sensitivity và insight | | | | ■ | |
| Literature Review | ■ | ■ | | | |
| Viết report | | ■ (nháp) | ■ (nháp) | ■ | ■ |
| IEEE paper, slides, video, Streamlit | | | | ■ | ■ |
| Turnitin, kiểm tra AI, tập dượt, nộp bài | | | | | ■ |

**Các mốc kiểm soát (milestone):**
- **M1 (29/10):** chốt dữ liệu sạch và xong thống kê mô tả.
- **M2 (03/11):** xong thống kê suy diễn, có `src/protocol.py`.
- **M3 (13/11):** cả 4 mô hình đã xuất file dự báo vào `reports/predictions/`.
- **M4 (20/11):** xong SEM, so sánh mô hình và khuyến nghị.
- **M5 (25/11):** có bản đầy đủ của report, paper và slides.
- **M6 (28/11):** nộp bài.

---

## 8. Tổng hợp khối lượng theo từng người

| Thành viên | Phần phân tích | Phần viết | Deliverable lead | Review |
|---|---|---|---|---|
| Khải Hưng | Pipeline dữ liệu, giao thức chung, LSTM, so sánh mô hình, sensitivity của cảnh báo EOL, insight | Report Ch1, §3.1, §3.6, §5.8–5.9, Ch6; ghép toàn bộ report | Full Report, GitHub, nộp bài | ML (Minh Vũ) |
| Quốc Đạt | ANOVA/Tukey, hồi quy bội log-linear, SEM | Literature Review §2.1, §3.3, §3.7, §5.3, §5.7, References | IEEE Paper | ARIMA (Tiến Hưng) |
| Tiến Hưng | CI, F-test, Levene, t-test, two-way ANOVA (lý do không làm), ARIMA/ARIMAX, hỗ trợ SEM | Literature Review §2.2, §3.2, §3.4, §5.2, §5.4, Appendix | Turnitin và kiểm tra AI | Hồi quy (Quốc Đạt) |
| Minh Vũ | Phân loại dữ liệu, thống kê mô tả, chi-square, Isolation Forest (kiểm tra dữ liệu), SVR | Literature Review §2.3, §3.5, §5.1, §5.5 | Slides, Demo video, Streamlit app | DL (Khải Hưng) |

---

## 9. Quy trình làm việc và theo dõi tiến độ

- **Họp:** thứ Hai và thứ Năm hằng tuần, khoảng 30 phút.
  - Mỗi người nêu đã làm gì, sẽ làm gì, đang vướng gì.
  - Sau đó mỗi người **trình bày lại phần của người mình review**.
- **Theo dõi công việc:** dùng GitHub Issues, mỗi mã công việc (C1.1, C3.4, D2, …) là một issue. Leader cập nhật bảng trạng thái trong file này sau mỗi buổi họp.
- **Quy trình code:**
  1. Làm trên branch mang tên mình (`khai-hung`, `quoc-dat`, `tien-hung`, `minh-vu`), theo quy tắc ở README mục 9–10.
  2. Mở Pull Request.
  3. Người review chéo duyệt.
  4. Leader merge vào `main`.
- **Khi trễ hạn:** báo trong nhóm **trước hạn ít nhất 24 giờ**. Người review chéo hỗ trợ trước, sau đó Leader điều chỉnh lịch.
- **Quy tắc số liệu:** mọi con số trong report, paper và slides phải lấy từ notebook hoặc script đã commit. Không gõ tay.
