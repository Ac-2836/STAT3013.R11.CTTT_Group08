# Work Log: Nhật ký quy trình làm việc của nhóm 4AnhTai

> **Mục đích:** Mỗi thành viên ghi lại chi tiết quá trình làm phần việc của mình trên branch riêng, để cả nhóm đọc được cách làm của nhau, và để làm nguồn tư liệu chính khi tổng hợp viết Full Report, IEEE Paper, slides và chuẩn bị thuyết trình.

---

> [!IMPORTANT]
> **NOTICE: YÊU CẦU VỀ ĐỘ CHI TIẾT (dành cho người viết, đặc biệt là AI)**
>
> Đây là nguồn tư liệu gốc để viết các file báo cáo. Phần nào ghi thiếu ở đây thì sẽ thiếu trong báo cáo. Vì vậy mỗi mục ghi phải đạt các yêu cầu sau:
>
> 1. **Đủ để người không có mặt hiểu, tái lập và viết lại.** Một thành viên khác đọc xong phải:
>    - hiểu *vì sao* làm như vậy;
>    - chạy lại ra *đúng* kết quả;
>    - viết thẳng thành đoạn Methodology/Results mà không cần hỏi lại.
> 2. **Ghi đầy đủ theo trình tự thực tế**, không chỉ ghi kết quả cuối:
>    - vấn đề phát hiện được, kèm bằng chứng;
>    - các phương án đã cân nhắc, phương án được chọn và lý do loại các phương án khác;
>    - cách làm cụ thể;
>    - cách kiểm chứng;
>    - kết quả;
>    - quyết định đã chốt, và ai chốt.
> 3. **Mọi con số phải lấy từ kết quả chạy thực tế** và ghi rõ nguồn: script, notebook, lệnh, hoặc commit hash. Không ước lượng, không làm tròn mất ý nghĩa, không bịa.
> 4. **Ghi rõ mọi tham số, ngưỡng, công thức:** giá trị cụ thể, đơn vị, và lý do chọn giá trị đó. Nếu đã thử nhiều giá trị thì ghi cả kết quả của từng giá trị.
> 5. **Ghi cả hướng đã thử nhưng thất bại hoặc bị loại.** Đây là nội dung có giá trị cho phần Discussion và cho phần hỏi đáp khi bảo vệ.
> 6. **Ghi đầu ra cụ thể:** tên file, đường dẫn, số dòng/cột, một dòng tương ứng với cái gì, và ý nghĩa của từng cột quan trọng. Hình thì ghi đường dẫn và nội dung hình thể hiện.
> 7. **Liên hệ kiến thức môn học khi có:** chương và trang slide (ví dụ: Chap 2, tr. 75).
> 8. **Viết như biên bản kỹ thuật**, không viết tóm tắt chung chung kiểu "đã làm sạch dữ liệu". Thuật ngữ chuyên môn, tên biến, tên file giữ nguyên tiếng Anh.
> 9. **Quy tắc ghi:**
>    - Mỗi thành viên chỉ ghi vào phần của mình, để tránh conflict khi merge.
>    - Chỉ ghi thêm, không xóa hay sửa mục cũ của người khác. Nếu cần đính chính thì ghi thêm dòng "Cập nhật ngày …: …".
>    - Mỗi mục ghi có đủ: **ngày, thành viên, branch, giai đoạn**.

---

## Tiến Hưng

### [11/11/2026] Tiến Hưng · branch `tien-hung` · GĐ5b ARIMA / ARIMAX (WBS C5.1–C5.7)

**Nguồn của mọi con số:** `notebooks/20_arima.ipynb` (chạy Restart & Run All, statsmodels 0.15.0, pandas 3.1.1, numpy 2.5.4, Python 3.15). Dữ liệu và chấm điểm lấy từ `src/protocol.py` (`load_split()`, `evaluate()`), không sửa protocol.

**1. Dữ liệu và quy ước**
- Chuỗi: `soh` của từng pin tier A, theo thứ tự dòng (`cycle_valid` tăng dần), đã bỏ `cap_outlier` và `break_in` (Rec.md 6.1 #4: mục tiêu là `soh`, không phải `capacity_int_Ah`). Train 947 dòng / 8 pin, test 528 dòng / 4 pin, 213 dòng đánh giá.
- Trục thời gian là số thứ tự dòng, không dùng ngày (Chap 5, tr. 4). `cycle_valid` có chỗ nhảy số do đã bỏ outlier; mô hình coi các dòng liền nhau là các bước liền nhau, giống `is_eval` của protocol.
- Phân tích Box-Jenkins (ADF, ACF/PACF, lưới AIC, Ljung-Box) làm trên 8 pin **train** (cả chuỗi). Pin test chỉ được nhìn phần trước mốc 60% khi chọn mô hình, để không rò rỉ dữ liệu đánh giá.

**2. Tính dừng (C5.2, Chap 5 tr. 11–14)**
- ADF (`autolag="AIC"`) trên chuỗi gốc, dạng `c` và `ct`, và trên `Δsoh` dạng `c`. `Δsoh` dừng ở cả 12 chuỗi (p_diff < 0,0001).
- Chuỗi gốc: không dừng ở 10/12 chuỗi theo ít nhất một dạng. B0045 (p_c = 0,0005; p_ct = 0,0027) và B0055 trước mốc (p_c = 0,0013; p_ct = 0,0030) bác bỏ H0. Đây là hai chuỗi ngắn (67 và 59 điểm) có nhiều bước nhảy hồi phục lớn.
- Khác với Go.md 5b.3 (ghi "p = 0,53–0,89"): trên dữ liệu tier A hiện tại, p của ADF dạng `c` trên chuỗi gốc dao động 0,0005–0,97. Vẫn **chốt d = 1 cho mọi pin**, vì xu hướng giảm thấy rõ trên mọi chuỗi, ACF chuỗi gốc giảm rất chậm (hình `arima_02`) và `Δsoh` đã dừng nên không cần d = 2.

**3. Nhận dạng và ước lượng (C5.3, tr. 47–48, 65–67)**
- Lưới p, q ∈ {0, 1, 2}, d = 1, `trend` ∈ {`n`, `t`} (`t` = drift khi d = 1), tổng 18 mô hình mỗi chuỗi, MLE bằng `statsmodels.tsa.arima.model.ARIMA`. Chỉ nhận mô hình hội tụ (`mle_retvals["converged"]`). Ví dụ: ở B0005, mô hình AIC nhỏ nhất (0,1,2)+t **không hội tụ** nên bị loại và chọn (2,1,0)+t.
- Lý do thêm drift: không có drift thì dự báo nhiều bước của ARIMA d = 1 gần như đi ngang. AIC chọn `t` ở 7/8 pin train.
- Mô hình chọn trên train: B0005 (2,1,0)t, B0006 (1,1,1)t, B0018 (1,1,2)t, B0033 (2,1,1)t, B0045 (0,1,0)t, B0046 (0,1,2)t, B0048 (0,1,0)t, B0056 (1,1,2)n.

**4. Chẩn đoán (C5.4, tr. 49)**
- Ljung-Box ở lag 10 và 20, `model_df = p + q`, bỏ phần dư đầu tiên (giá trị khởi tạo khi d = 1).
- Quy tắc chọn cuối cùng: đi theo AIC tăng dần, lấy mô hình đầu tiên hội tụ **và** có LB p > 0,05 ở cả hai lag. Nếu không có mô hình nào đạt thì giữ mô hình AIC nhỏ nhất và đánh dấu `LB_pass = False`.
- Kết quả: 7/8 pin train đạt ngay ở mô hình AIC nhỏ nhất. **B0018 không đạt với mô hình nào trong lưới** (mô hình AIC tốt nhất có LB p = 0,0408 ở lag 10 và 0,0003 ở lag 20).
- Nguyên nhân B0018 (bảng bước nhảy ở mục 5 notebook, ngưỡng Δsoh > 0,015): B0018 có 9 bước nhảy, 8 bước đứng ngay sau một lần đo EIS mới, 0 bước sau nghỉ dài (B0018 không có `long_rest` nào, `rest_before_h` ở các bước nhảy chỉ 0,02 h). Vì vậy `rest_before_h` không giải thích được hiện tượng hồi phục ở pin này.
- **Đề xuất cho nhóm (chưa áp dụng):** thêm biến cờ "vừa đo EIS" cho các mô hình. Không đưa vào ARIMAX vì danh sách biến đã chốt ở Go.md 5b.8. Lưu ý B0005–B0007 đo EIS gần như mỗi chu kỳ (139/167 dòng) nên cờ này không có tác dụng ở các pin đó.

**5. Seasonality / SARIMA (C5.6, tr. 60–62)**
- ACF của `Δsoh` đến lag 40 trên 8 pin train: không lag nào vượt dải ±1,96/√n ở quá 2/8 pin. Các lag vượt dải nằm rải rác, đúng mức kỳ vọng do ngẫu nhiên.
- Khoảng cách (số dòng) giữa các lần nghỉ dài không cố định, ví dụ B0005: [5, 29, 25, 17, 13, 17, 17]; B0033: [21, 23, 67, 32].
- Các bước nhảy gắn với thời gian nghỉ: Δsoh trung bình ngay sau nghỉ dài dương (0,0024 đến 0,0639), còn ở các chu kỳ khác âm (−0,0021 đến −0,0068). corr(Δsoh, log1p(rest_before_h)) dương ở 7/8 pin (0,28–0,78); B0018 là −0,11, xem mục 4.
- **Kết luận: không dùng SARIMA.** Biến động có quy luật được mô hình hoá bằng biến ngoại sinh trong ARIMAX.

**6. Mô hình trên pin test (C5.5)**
- Ba biến thể: `arima` (không có exog), `arimax_rest` (`log1p(rest_before_h)`), `arimax` (`log1p(rest_before_h)` + `Rct_ohm`, đúng Go.md 5b.8). Mỗi pin, mỗi biến thể chọn (p, q, trend) bằng lưới AIC + Ljung-Box như mục 4, chỉ trên 60% dòng đầu (`is_eval == False`). Cả 12 mô hình đều có `LB_pass = True`.
- `Rct_ohm` thiếu ở 19 dòng đầu của B0007. Theo Rec.md 6.1 #6, các dòng này bị bỏ, không điền. Để AIC so được, **cả ba biến thể** của B0007 đều fit trên cùng 81 dòng (từ dòng thứ 20). Ba pin test còn lại không thiếu giá trị nào.
- Exog được truyền vào dưới dạng DataFrame có tên cột. Lỗi đã gặp và đã sửa: khi truyền numpy và có `trend="t"`, statsmodels đặt tên drift là `x1`, nên lần chạy đầu đọc nhầm drift thành hệ số `log_rest` (ra âm). Sau khi sửa, hệ số `log_rest` trong `arimax_rest` dương ở cả 4 pin: B0007 0,0014 (p = 0,031), B0034 0,0118, B0047 0,0038, B0055 0,0097 (ba pin sau p < 0,0001).
- Với d = 1, exog nằm trong phương trình mức (regression with ARIMA errors). Hệ số dương nghĩa là chu kỳ có thời gian nghỉ dài hơn thì SOH cao hơn, và khi thời gian nghỉ trở lại bình thường thì SOH rơi lại.
- Hệ số `Rct_ohm` không ổn định giữa các pin: B0007 +1,18 (p < 0,0001), B0034 +2,52 (p = 0,109), B0047 −0,45 (p = 0,275), B0055 −1,72 (p = 0,064).
- AIC: thêm `log_rest` giảm AIC mạnh ở B0034 (−607,6 → −673,2), B0047 (−227,7 → −236,4) và B0055 (−255,3 → −262,3); không giúp ở B0007 (−622,2 → −621,9).

**7. Dự báo theo protocol (C5.7, tr. 49)**
- **E1:** tham số ước lượng một lần trên 60% đầu, rồi chạy bộ lọc qua cả chuỗi bằng `res.apply(y_all, exog=x_all, refit=False)`. Dự báo một bước bằng `get_prediction(start=n_pre)`, nên dự báo tại t chỉ dùng `soh` đến t−1. Không fit lại theo từng bước, để cùng điều kiện với SVR/LSTM (fit một lần).
- **E2:** `get_forecast(steps=n_eval)` từ mốc. Exog giữ nguyên giá trị cuối cùng trước mốc (Rec.md 6.1 #5).
- Cả hai kịch bản ghi khoảng dự báo 95% vào `y_lower`, `y_upper`.

| Mô hình | RMSE E1 | MAE E1 | R² E1 | RMSE E2 | MAE E2 | R² E2 |
|---|---|---|---|---|---|---|
| naive | 0,0226 | 0,0119 | 0,500 | 0,0343 | 0,0249 | −0,155 |
| arima | 0,0199 | 0,0102 | 0,612 | 0,0589 | 0,0458 | −2,400 |
| arimax_rest | 0,0168 | 0,0095 | 0,724 | 0,0570 | 0,0442 | −2,187 |
| arimax | 0,0170 | 0,0098 | 0,716 | 0,0577 | 0,0398 | −2,260 |

- **E1:** ARIMA thắng naive về RMSE gộp. Thêm `log_rest` giúp nhiều nhất (RMSE giảm khoảng 26% so với naive); thêm `Rct_ohm` không giúp thêm. Theo pin, naive vẫn tốt hơn ở B0007 (0,0037 so với 0,0045 của arima), vì chuỗi này trơn và ít răng cưa. Độ phủ của khoảng 95% ở E1 là 94,4% (arima) và 92,5% (arimax).
- **E2:** cả ba biến thể ARIMA đều **kém naive**. Với d = 1, dự báo dài hạn là đường thẳng theo drift của 60% đầu. B0034 và B0047 suy giảm nhanh lúc đầu rồi chững lại, nên drift đi xuống quá dốc (RMSE E2 của arima: 0,0689 và 0,0932). Ở B0034, `arimax` được AIC chọn `trend = n` nên RMSE E2 chỉ 0,0185. Kết quả E2 vì vậy phụ thuộc mạnh vào việc có drift hay không. Độ phủ của khoảng 95% ở E2: arima 74,2%, arimax 93,4%.
- **RUL** (E2, B0007/B0034/B0055; giá trị âm là báo sớm):

  | Mô hình | B0007 | B0034 | B0055 |
  |---|---|---|---|
  | arima | −11 | −49 | không chạm 0,8 |
  | arimax | −10 | không chạm 0,8 | +6 |
  | naive | không chạm 0,8 | không chạm 0,8 | không chạm 0,8 |

**8. Đầu ra**
- `reports/predictions/arima.csv`: 852 dòng = 2 mô hình (`arima`, `arimax`) × 2 kịch bản (E1, E2) × 213 dòng đánh giá. Cột: `PRED_COLS`. Đã qua `evaluate()`. `arimax_rest` chỉ dùng phân tích trong notebook, không xuất ra. Cả hai mô hình định trước đều được xuất, không chọn mô hình theo kết quả trên tập test.
- Hình trong `reports/figures/`:
  - `arima_01_series.png`: chuỗi SOH 12 pin, tách theo nhiệt độ, đánh dấu các chu kỳ sau nghỉ dài.
  - `arima_02_acf_level_vs_diff.png`: B0005, ACF của chuỗi gốc so với `Δsoh`.
  - `arima_03_acf_pacf_diff.png`: ACF/PACF của `Δsoh` cho 8 pin train.
  - `arima_04_resid_acf_train.png`: ACF phần dư của mô hình đã chọn.
  - `arima_05_forecast_arima.png`, `arima_05_forecast_arimax.png`: dự báo E1/E2 kèm khoảng 95% trên 4 pin test.

**9. Kết luận cho RQ3 (để viết Report §3.4, §5.4)**
- Lịch sử của chính pin đủ để dự báo **một chu kỳ tới** tốt hơn naive, nhất là khi có thời gian nghỉ làm biến ngoại sinh.
- ARIMA không phù hợp cho dự báo dài hạn hay RUL, vì là mô hình tuyến tính (Chap 5, tr. 51) và không bắt được "đầu gối" hay đoạn chững lại của đường suy giảm.

**Việc còn lại:**
- C5.8: viết Report §3.4, §5.4.
- Gửi Quốc Đạt review notebook.
- Báo leader về biến cờ EIS và việc Go.md 5b.3 ghi khoảng p-value của ADF chưa khớp dữ liệu hiện tại.

### [11/11/2026] Tiến Hưng · branch `tien-hung` · GĐ3 Thống kê suy diễn (WBS C2.1–C2.8)

**Nguồn của mọi con số:** `notebooks/02_inferential.ipynb` (Restart & Run All; scipy 1.18.1, statsmodels 0.15.0). Input: `data/processed/battery_fade.csv` (đơn vị quan sát là pin) và `cycles_clean.csv` (cho CI tỉ lệ và chi-square). C2.4 (Quốc Đạt), C2.6 (Minh Vũ) và C2.7 (Khải Hưng) cũng đã làm trong notebook này để GĐ3 có đủ một chỗ; các thành viên phụ trách vui lòng review phần của mình.

**1. Mẫu**
- 23 pin trong `battery_fade.csv`. Loại B0054 (tier N, 18,1% outlier; Go.md 3.2 đã ghi G9 "không gồm B0054") và B0028 (xung vuông biên độ 2 A, là pin duy nhất của điều kiện này nên không có phương sai trong nhóm). Còn **21 pin trong 6 nhóm điều kiện**.
- Nhóm theo điều kiện (nhiệt độ môi trường / dạng tải / mức dòng), không theo nhóm G: 24C/CC/2A = G1 + B0036 (n = 5), 24C/CC/4A = B0033, B0034 (2), 24C/SQ/4A = B0025–27 (3), 43C/CC/4A = G3 (4), 4C/CC/1A = G7 (4), 4C/CC/2A = B0053, B0055, B0056 (3). B0036 được gộp vào vì cùng điều kiện với G1, giống bảng ở Introduction_of_dataset.md mục 8.
- Mọi kiểm định chạy trên cả `fade_pct_per_100cyc` (fade) và `fade25_pct_per_100cyc` (fade25). α = 0,05.

**2. CI (C2.1)**
- CI t, df = n − 1, cho trung bình từng nhóm (bảng `reports/tables/inference_ci.csv`). Ví dụ fade: 43C/CC/4A 28,37 (20,99–35,75); 4C/CC/1A 29,93 (25,11–34,76); 24C/CC/4A (n = 2, t = 12,706) 8,27 (−9,95 đến 26,50).
- CI tỉ lệ các lần xả có SOH tăng so với lần trước, trên 22 pin tier A/B (bỏ `cap_outlier`, `break_in`): 562/1.966 = 28,6%, CI z 26,6–30,6%. Vì các lần xả trong một pin không độc lập (F9), tính thêm CI dựa trên đơn vị pin (trung bình tỉ lệ của từng pin, t với df = 21): 26,7% (22,3–31,1%), rộng hơn khoảng 2,2 lần.

**3. T1–T4 (C2.2–C2.3)**
- Quy trình định trước: F-test hai phía (tr. 51) quyết định pooled hay Welch; Levene (`center="median"`) chỉ để đối chiếu. T1 là một phía (Ha: μ43 > μ24), nêu trước dựa trên cơ chế Arrhenius/SEI; các cặp còn lại hai phía.
- Ở T2 và T3 (fade), F-test bác bỏ phương sai bằng nhau (p = 0,0064 và 0,036) còn Levene thì không (p = 0,29 và 0,43), vì nhóm 4C/CC/2A có sd rất nhỏ (0,41). Theo quy trình, các cặp này dùng Welch.

| Cặp | Biến | Loại t | T | df | p | Chênh lệch (CI 95%) | Kết luận |
|---|---|---|---|---|---|---|---|
| T1 43 °C vs 24 °C, 4 A (một phía) | fade | pooled | 5,603 | 4 | 0,0025 | 20,10 (10,14–30,06) | bác bỏ H0 |
| | fade25 | pooled | 1,495 | 4 | 0,1046 | 7,74 (−6,64–22,13) | không bác bỏ |
| T2 24 °C vs 4 °C, 2 A | fade | Welch | 2,440 | 4,04 | 0,0706 | 7,94 (−1,06–16,94) | không bác bỏ |
| | fade25 | pooled | −1,485 | 6 | 0,1881 | −13,51 (−35,78–8,75) | không bác bỏ |
| T2b chỉ G1 vs 4 °C, 2 A | fade | Welch | 6,886 | 3,14 | 0,0054 | 10,95 (6,01–15,89) | bác bỏ H0 |
| | fade25 | pooled | −1,135 | 5 | 0,3080 | −10,99 (−35,90–13,91) | không bác bỏ |
| T3 1 A vs 2 A, 4 °C | fade | Welch | 13,017 | 3,15 | 0,0008 | 19,98 (15,22–24,74) | bác bỏ H0 |
| | fade25 | pooled | 3,773 | 5 | 0,0130 | 36,84 (11,74–61,95) | bác bỏ H0 |
| T4 xung vuông vs CC, 24 °C/4 A | fade | pooled | 0,695 | 3 | 0,5369 | 2,67 (−9,54–14,87) | không bác bỏ |
| | fade25 | pooled | −1,833 | 3 | 0,1642 | −10,38 (−28,40–7,64) | không bác bỏ |

- Chỉ **T3** vững qua cả hai biến. T1 có ý nghĩa ở fade nhưng không có ở fade25: G3 chỉ có 39 chu kỳ, còn B0033/B0034 có khoảng 180–194 chu kỳ với đoạn chững lại kéo độ dốc cả chuỗi xuống, nên fade so sánh không công bằng. T2 phụ thuộc việc có gộp B0036 (fade 5,86) hay không.

**4. ANOVA (C2.4)**
- 6 nhóm, 21 pin. fade: F(5,15) = 12,686, p = 0,0001. fade25: F(5,15) = 12,668, p = 0,0001. Bảng ANOVA tự tính (SS, df, MS, F, F_crit = 2,9013) đã được `assert` khớp với `scipy.stats.f_oneway`.
- Giả định: Levene p = 0,763 (fade) và 0,686 (fade25). Shapiro-Wilk trên phần dư p = 0,042 (fade) và 0,612 (fade25). Vì fade vi phạm giả định chuẩn, chạy thêm Kruskal-Wallis: H = 15,55 (p = 0,0082) và 13,12 (p = 0,0223). Kết luận không đổi.
- Tukey HSD: fade có 7 cặp khác nhau, fade25 có 5 cặp. 4C/CC/1A khác 24C/CC/2A, 24C/CC/4A, 24C/SQ/4A và 4C/CC/2A ở cả hai biến. Riêng 43C/CC/4A khác 24C/CC/4A, 24C/SQ/4A và 4C/CC/2A ở fade, nhưng ở fade25 chỉ khác 4C/CC/1A.

**5. Two-way ANOVA (C2.5)**
- Bảng số pin theo ô (CC, mẫu chính), 4/9 ô trống:
  - 4 °C: 1 A = 4, 2 A = 3, 4 A = 0
  - 24 °C: 1 A = 0, 2 A = 5, 4 A = 2
  - 43 °C: 1 A = 0, 2 A = 0, 4 A = 4
- Không ước lượng được tương tác, và hiệu ứng nhiệt độ bị lẫn với dòng xả. Đoạn giải thích đầy đủ nằm ở mục 5 của notebook.

**6. Chi-square (C2.6)**
- **Phát hiện:** lý do chọn W = 40 trong Go.md 3.5 ("số chu kỳ ngắn nhất của G3") không khớp dữ liệu. G3 chỉ có `cycle_valid` 1–39, còn G2 (B0025–27) chỉ tới 28. Với W = 40, các pin này được quan sát thiếu.
- **Bản chính W = 25 (21 pin):** quan sát "có / không" theo nhiệt độ là 4 °C 6/1, 24 °C 1/9, 43 °C 0/4. χ² = 13,093, df = 2, p = 0,0014; p hoán vị (20.000 lần, seed 42) = 0,0014. 5/6 ô có E_ij < 5.
- **Bản kiểm tra W = 39 (18 pin, bỏ G2):** 4 °C 6/1, 24 °C 3/4, 43 °C 3/1. χ² = 3,054, p = 0,217; p hoán vị = 0,319.
- **Kết luận:** pin ở 4 °C mất dung lượng sớm hơn, nhưng kết luận không vững khi đổi cửa sổ. Hoán vị được dùng vì E_ij < 5 và Fisher exact chỉ áp dụng trực tiếp cho bảng 2×2.

**7. Tương quan với nhiệt độ cell (C2.7)**
- `T_max_med` với fade: r = −0,019 (p = 0,937), Spearman ρ = −0,107 (p = 0,646).
- Với fade25: r = −0,529 (p = 0,014), ρ = −0,379 (p = 0,090).
- Riêng nhóm 4 °C (n = 7): r = −0,586 (p = 0,167) với fade và −0,518 (p = 0,234) với fade25.
- Scatter (`inf_03`) cho thấy quan hệ hình chữ U: cell lạnh nhất (4 °C/1 A) và cell nóng nhất (43 °C) đều suy giảm nhanh. Vì vậy r tuyến tính không đo đúng quan hệ này. Giả thuyết "nhiệt độ cell quan trọng hơn nhiệt độ môi trường" chỉ được ủng hộ về hướng.

**8. Đầu ra**
- Bảng trong `reports/tables/`:
  - `inference_summary.csv`: bảng tổng hợp, mỗi dòng một kiểm định; cột test, variable, H0, method, statistic, df, p, decision.
  - `inference_ttests.csv`: chi tiết T1–T4 gồm F, Levene, T, df, t_crit, p, chênh lệch và CI.
  - `inference_ci.csv`: CI theo nhóm.
- Hình trong `reports/figures/`:
  - `inf_01_ci_by_condition.png`: CI 95% theo nhóm, cho hai biến.
  - `inf_02_anova_qq.png`: QQ-plot phần dư ANOVA.
  - `inf_03_fade_vs_cell_temp.png`: scatter tốc độ suy giảm theo `T_max_med`.
- Bảng kết luận và ý nghĩa thực tế (C2.8) ở mục 9 của notebook.

**Việc còn lại:**
- Leader review C2.8.
- Báo nhóm sửa Go.md 3.5 (mốc 40 chu kỳ).
- Viết Report §3.2 và §5.2.
