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
