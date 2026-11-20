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

### [13/11/2026] Tiến Hưng · branch `tien-hung` · GĐ3 + GĐ5b Sửa theo review của Leader (WBS C2.8, C5.1–C5.7)

**Căn cứ:** `docs/reviews/Member_Review_TienHung.md` mục 4.1, 4.2, 5.1. **Nguồn của mọi con số:** `notebooks/20_arima.ipynb`, `notebooks/02_inferential.ipynb` (Restart & Run All; Python 3.15, numpy 2.5.4, scipy 1.18.1, statsmodels 0.15.0, pandas 3.1.1).

**1. ARIMA chạy lại ra số khác (mục 4.1): đã tìm ra nguyên nhân và sửa**
- Trên cùng một máy, notebook cũ chạy lại cho kết quả giống hệt; lệch chỉ xuất hiện khi đổi phiên bản numpy/scipy. Để mô phỏng việc đổi máy, tôi nhân `soh` với (1 + 1e-12·nhiễu): với cách fit cũ, 4/12 mô hình trên pin test **đổi bậc** và dự báo E1 lệch tới 0,018. Nguyên nhân: trên thang SOH, phương sai nhiễu khoảng 1e-5 nên bài toán MLE bị lệch thang đo; L-BFGS (mặc định 50 vòng) dừng ở các điểm hơi khác nhau và cờ `converged` lật qua lại (ví dụ AIC cùng một cấu hình lệch tới 3,5 khi đổi điểm khởi đầu 0,1%).
- Chỉ tăng `maxiter = 1000` (đề xuất trong review) **không đủ**: số cấu hình không ổn định gần như không đổi. Siết dung sai (`pgtol`, `factr`) hay thêm bước BFGS còn làm cờ hội tụ lật nhiều hơn.
- **Cách sửa đã áp dụng:** fit MLE trên SOH tính theo điểm phần trăm (`100·soh`) với `maxiter = 1000`, rồi đổi tham số về đơn vị SOH (AR/MA giữ nguyên, drift và hệ số exog ÷ 100, σ² ÷ 100²) và chạy Kalman smoother trên chuỗi `soh` gốc. AIC, phần dư, sai số chuẩn và dự báo vì vậy vẫn ở đơn vị SOH; cờ `converged` lấy từ lần tối ưu. Đổi thang `y` chỉ cộng cùng một hằng số vào log-likelihood của mọi mô hình trên cùng tập dòng, nên **không đổi thứ hạng AIC** và không đổi quy tắc chọn bậc. Đã kiểm tra: dự báo của kết quả đổi thang khớp bản fit gốc tới 3e-16.
- Kiểm tra sau khi sửa: nhiễu 1e-12 và 1e-9 (10 lần rút ngẫu nhiên) **không làm đổi bậc nào**; dự báo lệch tối đa khoảng 2e-4 (trước là 0,018). Chạy notebook hai lần liên tiếp: `arima.csv` và `arima_orders.csv` giống hệt từng byte.
- Lưu ý kỹ thuật: statsmodels sửa trực tiếp dict `method_kwargs` khi fit, nên phải truyền dict mới mỗi lần gọi.
- **Mới:** `reports/tables/arima_orders.csv` (20 dòng: 8 mô hình train + 4 pin test × 3 biến thể; cột set, battery_id, model, n_fit, p, d, q, trend, AIC, converged, LB_p_lag10, LB_p_lag20, LB_pass, aic_rank). Cả 20 mô hình cuối đều hội tụ.

**2. Số liệu ARIMA đã đổi (thay cho các số trong mục [11/11/2026] GĐ5b bên dưới)**
- Bậc trên train: B0005 giờ là (0,1,2)t (trước bị loại vì "không hội tụ", đúng như review nghi ngờ), B0045 (2,1,2)t, B0046 (0,1,2)t, B0048 (0,1,0)t, B0006/B0018/B0033/B0056 (1,1,1)t. Ljung-Box: 7/8 train (B0018 vẫn không đạt: p = 0,061 ở lag 10, 0,0003 ở lag 20), 12/12 mô hình test.
- Bậc trên test: B0007 (1,1,1)t cả ba biến thể; B0034 (2,1,2)t, (2,1,2)t, arimax (2,1,2)n; B0047 (0,1,0)t cả ba; B0055 arima (0,1,1)n, arimax_rest (1,1,0)n, arimax (1,1,0)t.
- Hệ số `log_rest` trong `arimax_rest` vẫn dương ở cả 4 pin: B0007 0,0014 (p = 0,019), B0034 0,0119, B0047 0,0038, B0055 0,0097 (ba pin sau p < 0,0001).

| Mô hình | RMSE E1 | MAE E1 | R² E1 | RMSE E2 | MAE E2 | R² E2 |
|---|---|---|---|---|---|---|
| naive | 0,0226 | 0,0119 | 0,500 | 0,0343 | 0,0249 | −0,155 |
| arima | 0,0203 | 0,0114 | 0,598 | 0,0594 | 0,0468 | −2,456 |
| arimax_rest | 0,0172 | 0,0108 | 0,709 | 0,0579 | 0,0456 | −2,287 |
| arimax | 0,0171 | 0,0098 | 0,715 | 0,0583 | 0,0407 | −2,326 |

- Độ phủ khoảng 95%: E1 88,7–92,0%; **E2 chỉ 68,1–70,9% ở cả ba biến thể** (trước ghi arimax 93,4%, số đó không còn đúng).
- RUL (E2): arima −12 / −49 / không chạm 0,8 (B0007 / B0034 / B0055); arimax −11 / không chạm / +6.
- Kết luận RQ3 không đổi: E1 thắng naive, E2 kém naive do drift tuyến tính quyết định kết quả.

**3. Dịch sang tiếng Anh (mục 4.2)**
- Toàn bộ markdown, comment, chuỗi in ra của hai notebook; nội dung các bảng `inference_*.csv` (`decision`, `method`, `H0`, `comparison`, tên nhóm); tiêu đề, nhãn trục, chú giải của các hình `inf_0*.png`, `arima_0*.png`. Dấu thập phân là dấu chấm.
- Đã so `inference_*.csv` trước và sau khi dịch: mọi giá trị số giữ nguyên (lệch 0).

**4. Suy diễn: bổ sung theo mục 5.1**
- Cột `p_holm`, `decision_holm` trong `inference_summary.csv` (Holm trên cả 18 kiểm định). 10/18 bác bỏ H0 khi chưa hiệu chỉnh, còn 5/18 sau Holm (T1 fade, T3 fade, hai ANOVA, chi-square W = 25). **T3 với fade25 không còn ý nghĩa sau Holm** (p_holm = 0,143); Kruskal-Wallis cũng không (0,099 và 0,200). Quyết định chính vẫn dùng p chưa hiệu chỉnh vì mỗi kiểm định trả lời một câu hỏi riêng đã nêu trước; đã ghi rõ trong notebook.
- Cột `p_Welch`, `reject_H0_Welch` trong `inference_ttests.csv`: kiểm tra độ nhạy cho trường hợp n = 2 (F-test/Levene gần như không có lực). Welch cho cùng quyết định ở cả 10 dòng.
- Kết luận notebook đã thêm: "kết luận về nhiệt độ phụ thuộc cách đo tốc độ suy giảm; kết luận về dòng xả ở 4 °C thì vững"; T2 đổi chiều giữa fade và fade25; cảnh báo không diễn giải r âm thành "cell nóng hơn suy giảm chậm hơn"; chênh lệch mẫu 21 pin (notebook này) và 23 pin (hồi quy của Quốc Đạt).
- Chi-square: ghi rõ việc đổi W là **đề xuất chờ Leader chốt** và report phải trình bày cả W = 25 và W = 39.

**Việc còn lại:**
- Nhắn Leader chốt W của chi-square (ghi vào Rec.md, sửa Go.md 3.5).
- Thống nhất với Quốc Đạt mẫu chính (21 hay 23 pin) cho report.
- C5.8: viết Report §3.4, §5.4 theo số mới ở mục 2.

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

## Quốc Đạt

### [01/11/2026] Quốc Đạt · branch `quoc-dat` · Hoàn thành hồi quy log-linear, đánh giá E1/E2 và kiểm tra

**Mục tiêu.** Hoàn thiện phân tích hồi quy được giao trong notebooks/10_regression.ipynb, giữ lại các cell EDA và baseline đã có, sau đó chạy notebook từ đầu đến cuối theo giao thức chung.

**Dữ liệu.** Dùng data/processed/cycles_clean.csv; lọc tier A, bỏ cap_outlier và break_in theo README dữ liệu đã xử lý. Tập hồi quy có 1.475 dòng từ 12 pin. profile chỉ có một mức constant_current nên không đưa vào mô hình. T_max_C không dùng làm predictor độc lập vì là phản ứng nhiệt có tương quan với điều kiện vận hành. Condition table cho thấy thiết kế không có đủ mọi tổ hợp ambient/current/cutoff.

**So sánh dạng hàm.** Kiểm tra toàn bộ soh dương trước khi tạo ln_soh. So sánh soh ~ cycle_valid, ln(soh) ~ cycle_valid và soh ~ cycle_valid + I(cycle_valid ** 2). R² lần lượt là 0,4693; 0,4552; 0,5248. RMSE trên thang SOH lần lượt là 0,06241; 0,06178 (log-linear dùng smearing correction); 0,05906. AIC/BIC của mô hình tuyến tính SOH là -3993,45/-3982,86; log-linear là -3383,46/-3372,87; quadratic là -4154,44/-4138,55. AIC/BIC giữa SOH và ln(SOH) không so sánh trực tiếp vì khác thang đáp ứng. Hệ số cycle của log-linear đơn là -0,001412, tương ứng giảm xấp xỉ 0,1411% SOH mỗi chu kỳ trong mô hình đơn, không phải tác động nhân quả.

**Mô hình bội và lựa chọn.** Tạo cycle_c, current_c và cutoff_c bằng cách trừ trung bình mẫu; centering giảm đa cộng tuyến không cần thiết và làm main effects ứng với mức chu kỳ/dòng/cutoff trung bình. Mô hình ứng viên có tương tác cycle × ambient, cycle × current và cycle × cutoff. Backward selection dùng kiểm định Wald theo nhóm main effect + interaction, giữ cycle main effect và phân cấp tương tác; chỉ số tham số được xác định qua fit.params.index với restriction matrix tường minh. Mô hình được chọn:

```text
ln_soh ~ cycle_c + C(ambient_C) + cycle_c:C(ambient_C)
       + current_c + cycle_c:current_c
       + cutoff_c + cycle_c:cutoff_c
```

Mô hình có R² = 0,7467; adjusted R² = 0,7455; AIC = -4501,13; BIC = -4458,76. Sai số chuẩn gom cụm theo battery_id cho 12 cụm chỉ là phân tích độ nhạy. Ba tương tác có p-value gom cụm lần lượt 0,0811 (ambient 24°C so với mức tham chiếu), 0,0744 (mỗi A chênh lệch dòng), và 0,3229 (mỗi V chênh lệch cutoff); các CI 95% đều chứa 0. Dấu và độ lớn OLS thường không đủ để kết luận tương tác khi xét phụ thuộc trong pin. Hệ số cycle_c là -0,001210 (p gom cụm 0,0214; CI [-0,002203; -0,000216]) tại nhiệt độ tham chiếu và mức current/cutoff trung bình. Main effects trong mô hình tương tác là có điều kiện.

**Chẩn đoán.** VIF lớn nhất là 13,48 cho cycle_c và 12,53 cho cycle_c × ambient; các VIF còn lại tối đa 2,58. Breusch–Pagan LM = 367,79, p = 1,91×10⁻⁷⁵, cho thấy phương sai phần dư không đồng nhất. Có 87 điểm vượt ngưỡng Cook’s distance 4/n; leverage lớn nhất là 0,0180. Durbin–Watson theo pin rất thấp ở nhiều chuỗi (ví dụ B0007 = 0,0063), chỉ ra tự tương quan phần dư mạnh. Vì vậy 1.475 chu kỳ không thể xem là các đơn vị độc lập; cluster inference với 12 pin cũng có độ bất định đáng kể.

**Hồi quy mức pin.** data/processed/battery_fade.csv có 23 pin. Với fade_pct_per_100cyc ~ ambient_C + I_level_A + discharge_cutoff_V, R² = 0,4349 và adjusted R² = 0,3456. HC3: ambient_C = 0,6915 (SE 0,1727; p = 0,000062; CI [0,3531; 1,0299]); I_level_A = -8,6521 (SE 2,2081; p = 0,000089; CI [-12,9799; -4,3244]); cutoff = -12,1916 (SE 7,8670; p = 0,1212; CI [-27,6106; 3,2275]). Với fade25_pct_per_100cyc, R² = 0,3579, adjusted R² = 0,2566; không predictor nào có p < 0,05. Số pin nhỏ và điều kiện thí nghiệm không factorial đầy đủ nên không diễn giải như các tác động độc lập chắc chắn.

**E1/E2.** Áp dụng đúng train/test cells trong Rec.md; bước chọn predictor và centering cho đánh giá chỉ dùng 8 pin train. Test gồm B0007, B0034, B0047, B0055. Chuyển dự đoán ln(SOH) về thang SOH bằng exp trước khi tính metric. Trung bình theo pin: regression E1/E2 có MAE 0,08376, RMSE 0,08902, MAPE 10,516%, R² -21,507; persistence E1 có MAE 0,01412, RMSE 0,02014, MAPE 1,698%, R² 0,0447; persistence E2 có MAE 0,02382, RMSE 0,02997, MAPE 2,987%, R² -0,8717. E1 và E2 của regression trùng nhau vì mô hình trực tiếp dùng cycle/điều kiện, không dùng dự đoán SOH bước trước làm feature. EOL error E2 là 24 chu kỳ cho B0007 và 27 cho B0047; dự báo không chạm 0,8 trong phạm vi dự báo cho B0034 và B0055. y_lower/y_upper để trống vì mô hình điểm không xuất khoảng dự đoán.

**Đầu ra và kiểm chứng.** Notebook có 37 cell sau khi thêm 14 cell mới; nguồn của 23 cell đầu được giữ nguyên ngoài sửa đường dẫn PROJECT_ROOT ở cell 2 để lệnh chạy từ repository root hoạt động (vẫn tự nhận thư mục notebooks nếu chạy từ đó). Chạy lệnh nbconvert theo yêu cầu với PowerShell alias py; notebook hoàn thành, 19 cell code đã chạy, không có error output. Đã kiểm tra source 23 cell đầu giữa notebook gốc và executed copy. Tệp dự báo reports/predictions/regression.csv có 426 dòng (213 E1, 213 E2). Hình được tạo: reports/figures/regression_functional_forms.png, regression_simple_diagnostics.png, regression_preferred_diagnostics.png. Bản notebook có output: notebooks/10_regression.ipynb.

**Giới hạn.** Mô hình log-linear yếu hơn persistence trên các pin test và có R² âm mạnh; không nên dùng kết quả như dự báo tổng quát hóa tốt. Condition structure thiếu tổ hợp chéo, VIF tương tác còn vượt 10, phần dư heteroskedastic và tự tương quan. Các kết quả là liên hệ có điều kiện, không chứng minh nguyên nhân.

**Tệp thay đổi.** notebooks/10_regression.ipynb; reports/figures/regression_functional_forms.png; reports/figures/regression_simple_diagnostics.png; reports/figures/regression_preferred_diagnostics.png; reports/predictions/regression.csv; docs/Work_log.md.

### [01/11/2026] Quốc Đạt · branch `quoc-dat` · SEM / Path Analysis

**Objective.** Completed the SEM/path-analysis work in `notebooks/50_sem.ipynb` to examine associations among operating conditions, EIS resistance measurements, and SOH. Results are interpreted as statistical associations rather than causal effects.

**Data.** Used new EIS observations from Tier A/B batteries. After complete-case filtering for `ambient_C`, `I_level_A`, `discharge_cutoff_V`, `Re_ohm`, `Rct_ohm`, `soh`, and `cycle_valid`, the SEM dataset contained 812 observations from 22 batteries with no missing values in the analysis variables. New-EIS filtering avoids treating carried-forward EIS values as fresh measurements.

**Model.** The observed-variable path model includes paths from operating conditions to `Re_ohm` and `Rct_ohm`. `Re_ohm`, `Rct_ohm`, operating conditions, and `cycle_valid` are then used to explain `ln_soh`. semopy used MLW with SLSQP optimization and converged successfully after 113 iterations.

**Path results.** Notable standardized associations were `Re_ohm -> ln_soh` = 0.1045 (p = 0.0041), `Rct_ohm -> ln_soh` = -0.3489 (p < 0.001), `ambient_C -> ln_soh` = -0.2645 (p < 0.001), `I_level_A -> ln_soh` = 0.1722 (p < 0.001), and `cycle_valid -> ln_soh` = -0.6559 (p < 0.001). The direct `discharge_cutoff_V -> ln_soh` path was not significant at the 5% level (p = 0.1822).

**Model fit.** Fit statistics were chi-square = 479.29 with 13 degrees of freedom, CFI = 0.9235, GFI = 0.9217, NFI = 0.9217, TLI = 0.8529, and RMSEA = 0.2103. The high RMSEA and relatively low TLI indicate that overall model fit is not strong, so the model should not be treated as a complete representation of the data-generating structure.

**Direct, indirect, and total associations.** Point estimates of the total associations with `ln_soh` were approximately 0.00008 for ambient temperature, 0.02574 for discharge current, and 0.03262 for cutoff voltage. Battery-block bootstrap completed 200 successful replicates with zero failures. The 95% bootstrap intervals for the direct, indirect, and total associations of all three operating-condition predictors contained zero, indicating substantial uncertainty between batteries.

**Limitations.** Repeated EIS observations within the same battery are dependent, and the operating-condition design does not contain all factorial combinations. The SEM coefficients therefore represent conditional associations rather than causal effects. Battery-block bootstrap results are used to assess uncertainty more conservatively.

**Output and verification.** `notebooks/50_sem.ipynb` contains 12 cells, including 6 code cells. All 6 code cells were executed successfully with no error outputs. The path diagram was saved to `reports/figures/sem_path_diagram.png`.



### [10/11/2026] Quốc Đạt · branch `quoc-dat` · Audit Regression và SEM (bắt đầu 09/11)

**Bối cảnh và bảo toàn.** Yêu cầu audit trực tiếp Regression/SEM, không commit/push/merge. Đã kiểm tra status/branch/log;
branch đúng `quoc-dat`, HEAD `73452fd`. Working copy có SEM, regression CSV thay đổi và regression.test chưa track.
Đã sao lưu các bản này cùng notebook/hình cũ vào `reports/audit/regression_sem_before/`, giữ `preexisting.patch` và bản HEAD riêng.
Không sửa pipeline/dataset/giao thức của Leader hay mô hình thành viên khác. Chỉ thêm nội dung vào phần Quốc Đạt này.

**Nguồn số liệu và phương pháp.** Đọc README/Rec/Go/WBS/Work_log/Introduction, requirements, hai notebook,
cycles_clean/battery_fade/prediction CSV và ba module build/clean/eda. Ưu tiên Rec; seed 42, test/train giữ nguyên.
Python 3.11.16; version package và SHA256 từng input/output ở `regression_sem_execution.json`;
dependency lock riêng để không thay requirements chung. Script runner dùng fresh kernels từ root và notebooks.

**Regression.** Sửa simple thành G1 train (n=465, 3 pin), không fit test B0007;
R²=0.861688, adjusted R²=0.861390, F=2884.511506. Xuất conventional/cluster SE, p/CI, residual/standardized QQ/BP/JB/DW.
Thêm Re/Rct/charge duration correlations và design VIF; r Re–Rct train=0.510722, G1 train=0.953088.
Full research log-linear giữ dummy/interaction; backward kiểm định term có hierarchy, covariance cluster, theo dõi adjusted R² từng bước,
OLS là sensitivity. Chưa có slide Chap3 p39 để xác nhận nguyên văn bốn bước. Smearing train-only; AIC log model đổi về SOH density bằng Jacobian.
Xác minh cả full/25-cycle slopes của 23 battery_fade; HC3 với t inference, n nhỏ và slope uncertainty được nêu rõ.
GroupKFold4 train-only thử đúng 5 candidate có cơ sở: full/selected/quadratic log-linear và 2 anchored variants.
CV chọn E1=anchored_loglinear, E2=anchored_quadratic; không thay lựa chọn sau khi nhìn test.

| model | scenario | MAE | RMSE | MAPE_pct | R2 |
| --- | --- | --- | --- | --- | --- |
| persistence | E1 | 0.0141225 | 0.0201418 | 1.69765 | 0.0447127 |
| persistence | E2 | 0.0238145 | 0.0299687 | 2.9871 | -0.871726 |
| preaudit_working_copy | E1 | 0.0837614 | 0.0890195 | 10.516 | -21.5069 |
| preaudit_working_copy | E2 | 0.0837614 | 0.0890195 | 10.516 | -21.5069 |
| regression | E1 | 0.0137394 | 0.0202341 | 1.64797 | 0.0446581 |
| regression | E2 | 0.0335355 | 0.0420071 | 4.17526 | -2.75718 |
| research_loglinear | E1 | 0.0839132 | 0.0891713 | 10.5348 | -21.5502 |
| research_loglinear | E2 | 0.0839132 | 0.0891713 | 10.5348 | -21.5502 |

E1 chỉ thắng nhẹ persistence MAE, không thắng RMSE; E2 vẫn kém persistence. Global uncalibrated model chưa giải quyết heterogeneity.
Giữ kết quả cũ và toàn bộ candidate. CSV chung 426 dòng/8 cột, 213 dòng mỗi scenario, intervals NaN.
Xuất cả micro/macro/per-battery; forecasts plot có threshold .8. EOL dùng first SOH≤.8; B0047 đã event cycle30 trước origin40,
bỏ RUL/error không hợp lệ, giữ đủ mọi battery và explicit no-crossing status, không extrapolate horizon.
E1 là next retained clean observation; centered Hampel/lifetime Tier là upstream retrospective constraints, không tự sửa protocol.

**SEM.** Join từng EIS operation thực, không đếm ffill thành measurement; 812 điểm/22 pin, 14 dòng có quality flag được giữ
trong primary để đối chiếu mẫu gốc; sensitivity bỏ flag còn 798. 137 measurement của discharge4°C có EIS ambient24°C.
Sửa root path; kiểm tra convergence/rank/variance. Đếm tường minh 10 nuisance exogenous moments: primary25 parameter/28 moment, df3.
Internal z scaling và đổi coefficient/SE về đơn vị gốc; test đổi A thành mA kiểm tra scale. Bản raw-unit solver có thể success ở objective kém;
không dùng nó thay historical output. Primary theoretical graph không đổi; sensitivity age→mediators và shared Re/Rct residual có lý do vật lý.
Model thêm cả hai là saturated df0, không coi fit đẹp là xác nhận theory.

| DoF | chi2 | CFI | TLI | RMSEA |
| --- | --- | --- | --- | --- |
| 13 | 479.29 | 0.923489 | 0.852863 | 0.210303 |

Primary mới: χ²=479.289894, df=3, CFI=0.851816, TLI=-0.037285, RMSEA=0.442450.
Khác biệt indices có sửa cả df/baseline; fit vẫn kém, repeated rows khiến IID fit tests chỉ descriptive.
Bootstrap sample theo pin, có duplicate block multiplicity, seed42, target200 valid/max500; thực tế 200/200
hợp lệ và 0 failure. Xuất attempts/solver checks/SE/CI95 cho từng đường, direct/Re/Rct/combined/total;
tất cả condition-effect CI chứa0, không kết luận mediation significant/nhân quả, không tính unstable mediation% khi total≈0.
Sơ đồ 320dpi + vector PDF: từng predictor thật, mũi tên và standardized coefficient khớp CSV model, không hard-code estimates.

**Đầu ra và kiểm chứng.** Hai notebook lưu outputs fresh, [{"notebook": "10_regression.ipynb", "code_cells": 13, "error_outputs": 0}, {"notebook": "50_sem.ipynb", "code_cells": 4, "error_outputs": 0}]; exact numerical reproduction
root vs notebooks=True; input unchanged=True.
Validation: {"date": "2026-11-10", "branch": "quoc-dat", "tests": 17, "failures": 0, "errors": 0, "syntax_checked_files": 6, "notebooks": [{"name": "10_regression.ipynb", "executed_code_cells": 13, "error_outputs": 0}, {"name": "50_sem.ipynb", "executed_code_cells": 4, "error_outputs": 0}], "protected_group_files_unchanged": true, "preexisting_test_notebook_unchanged": true, "processed_input_hashes_unchanged": true, "numerical_output_hashes_match": true, "exact_reproduction_across_working_directories": true}. Đã inspect trực quan forecast/path/residual figures và kiểm tra dependency compatibility.
Lệnh tái lập và toàn bộ findings/WBS/review checklist: `docs/Regression_SEM_Audit.md`;
methods/results + IEEE-ready text: `docs/Regression_Report.md`, `docs/SEM_Report.md`.
Các file mới `src/regression_analysis.py`, `sem_analysis.py`, `regression_sem_utils.py`, `run_regression_sem.py`,
`write_regression_sem_reports.py`, `tests/test_regression_sem.py`, lock, tables và figures đều chỉ phục vụ hai phần này.

**Giới hạn và review còn lại.** Các thay đổi là phiên bản đề xuất để Quốc Đạt/Tiến Hưng/Leader review, không phải quyết định nhóm đã duyệt.
C3.1–C3.8, C7.2–C7.4 đã có code/output/nội dung báo cáo; C3.5 còn đối chiếu slide, C7.1 còn nhóm đồng ý theory paths,
và còn tích hợp deliverable, review chéo. Không thể làm E2 thắng baseline hoặc SEM fit tốt bằng sửa code hợp lệ trong phạm vi này.
Giữ trung thực các hạn chế n nhỏ, confounding, serial dependence, retrospective cleaning và chưa có prediction intervals.

### [10/11/2026] Quốc Đạt · Final stabilization Regression/SEM · `quoc-dat`

Đã tách historical archive thành comparison tùy chọn; giữ fit/CV/current metrics/prediction export độc lập.
SEM vẫn chạy đủ 6 specifications, bootstrap 200 valid, bảng và sơ đồ khi thiếu notebook cũ.
Kiểm tra toàn chuỗi 12 pin Tier A: cả ambient/current/cutoff đều cố định (36/36 checks).
E2 lấy condition values tại origin, từ chối điều kiện thay đổi hoặc không hữu hạn; test synthetic xác nhận fail rõ ràng.
CSV chung giữ 426 dòng/8 cột, `model=regression`; metadata ghi E1 anchored_loglinear và E2 anchored_quadratic,
hai bộ hệ số khác nhau, anchor chỉ dùng SOH được phép của chính pin test. Giữ research/baseline riêng.

28 tests PASS, 0 failure/error; compile 6 Python files và notebook cells; 98 dependency constraints không vi phạm
trong .venv hiện có (không có pip, dùng importlib.metadata/packaging). Hai notebook chạy đủ từ root và notebooks/,
13+4 code cells, 0 errors, CSV/JSON số học khớp tuyệt đối. Repo copy không archive chạy đủ hai notebook và sinh báo cáo;
57 current CSV khớp từng byte. 35 protected file hashes không đổi, gồm dữ liệu/pipeline/Rec/archives/test notebook.
Đã xem forecast/path figures; git diff --check đạt. Không commit/push/merge/PR.

Regression macro E1 MAE/RMSE=0.0137394/0.0202341 (persistence 0.0141225/0.0201418);
E2=0.0335355/0.0420071 (persistence 0.0238145/0.0299687). E2 vẫn kém baseline.
Primary SEM χ²=479.289894, df3, CFI0.851816, TLI−0.037285, RMSEA0.442450; fit vẫn kém.
Bootstrap 200/200 valid, 0 failures; toàn bộ 15 condition-effect CI chứa0. Ưu tiên block CI, không dựa riêng IID p-values;
812 observations chỉ là 22 independent batteries, không kết luận causal mediation.

Đối chiếu file bảo toàn: prediction max delta6.85e−14, bootstrap CI endpoint max delta6.13e−7.
Source chưa sửa trong bản đối chứng tái tạo đúng outputs mới. Raw-unit sensitivity thay objective≈0.590259 thành6.242571
do nghiệm nhạy scale; source đối chứng cũng tái tạo kết quả đó, không đổi primary estimator hay protocol để cải thiện metric.
Current tables bỏ historical rows; historical metrics/status lưu riêng. Chi tiết và file list:
`docs/Regression_SEM_Audit.md`, `reports/audit/regression_sem_final_validation.json` và `regression_sem_final_tests.log`.
Nhóm còn review scenario-specific strategy/anchoring, slide p39, theory paths và tích hợp Model Comparison;
các giới hạn retrospective cleaning, ít pin, confounding, thiếu prediction intervals và E2 yếu vẫn giữ nguyên.

### [21/11/2026] Khải Hưng (Leader) · branch `quoc-dat` · Dọn phần không tái lập được trước khi merge

Cập nhật theo review ngày 13/11 (Leader sửa trực tiếp trên nhánh để kịp hạn):
- Bỏ so sánh "historical/legacy": các bảng này cần thư mục `reports/audit/` không có trong repo nên chạy lại cho 0 dòng hoặc số khác. Đã xóa `regression_historical_*`, `sem_historical_*`, `sem_legacy_raw_units_reexecution_coefficients.csv`, dòng `legacy_raw_units_reexecution` trong `sem_fit_comparison.csv`, và code sinh ra chúng. So sánh bậc tự do 13 → 3 vẫn giữ qua dòng `original_scaled_implicit_exog`.
- Bỏ `src/write_regression_sem_reports.py` và `docs/Regression_SEM_Audit.md` (nhật ký audit, phụ thuộc `reports/audit/`). `docs/Regression_Report.md` và `docs/SEM_Report.md` giờ sửa tay; đã bỏ các link tới `reports/audit/` và `.venv`, thay bằng lệnh chạy lại.
- Tách test: `python -m unittest discover -s tests` chạy nhanh (25 test đạt, 2 bỏ qua, khoảng 15 giây); đặt `RUN_SLOW=1` để chạy thêm 2 test chạy lại toàn bộ.
- Các entry ở trên có nhắc `reports/audit/` và `.venv`: các file đó chỉ có trên máy của Đạt, không có trong repo.
