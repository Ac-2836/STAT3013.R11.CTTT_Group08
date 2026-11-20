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

## Minh Vũ

### 05/11/2026 · Minh Vũ · branch `minh-vu` · GĐ5c: Machine Learning (SVR) · WBS C6.2–C6.5

> **Cập nhật ngày 14/11/2026:** mục này là bản trước review. Sau review ngày 13/11, Re/Rct thiếu được bỏ dòng thay vì điền median (Rec.md 6.1 #6), nên số dòng train, tham số được chọn và các con số ở mục 2, 4.6, 5, 6, 7 đã thay đổi. **Số cuối cùng xem mục 14/11/2026 bên dưới.** Mục 1 và 3 vẫn đúng. Mục 4.1–4.5 và 4.7 là lịch sử thăm dò (cũng chạy khi còn điền median), giữ nguyên để ghi lại quá trình ra quyết định.

**Trả lời:** RQ4. Đặc trưng sạc và trở kháng dự báo SOH chu kỳ kế tiếp tốt đến đâu, và đặc trưng nào quan trọng nhất?
**Kế hoạch gốc:** [Go.md](Go.md) GĐ5c, 8 bước SVR theo slide Chap 6, tr. 33.
**Code:** `src/svr_model.py` (commit `998c86a`) · `notebooks/30_ml.ipynb` và `reports/predictions/svr.csv` (commit `393eaaf`) · 3 hình trong `reports/figures/` (commit `3f85a07`).
**Trạng thái:** xong phần phân tích. Các quyết định ở mục 4 và 5 do Minh Vũ đề xuất, **chưa qua review** của Khải Hưng (người review theo WBS).

#### 1. Dữ liệu và cách chia

- Đọc qua `load_split()` của `src/protocol.py` (commit `1afc383`), không đọc trực tiếp `cycles_clean.csv`. Như vậy SVR dùng đúng dữ liệu, cách chia và metric chung với 3 mô hình còn lại.
- Dữ liệu: tier A, bỏ dòng `cap_outlier` và `break_in`. Biến mục tiêu `soh`, trục thời gian `cycle_valid`.
- Train: 8 pin, **947 dòng**. Test: 4 pin, **528 dòng**, trong đó **213 dòng `is_eval`** (40% chu kỳ cuối mỗi pin test) là các dòng phải dự báo. Notebook kiểm tra 3 con số này bằng `assert`, và kiểm tra không pin nào nằm ở cả hai tập.

| Tập | Pin | Số dòng | Nhiệt độ / dòng xả / cutoff | SOH thấp nhất |
|---|---|---|---|---|
| train | B0005 | 167 | 24°C / 2 A / 2,7 V | 0,6967 |
| train | B0006 | 166 | 24°C / 2 A / 2,5 V | 0,5756 |
| train | B0018 | 132 | 24°C / 2 A / 2,5 V | 0,7327 |
| train | B0033 | 178 | 24°C / 4 A / 2,0 V | 0,7831 |
| train | B0045 | 67 | 4°C / 1 A / 2,0 V | 0,7240 |
| train | B0046 | 68 | 4°C / 1 A / 2,2 V | 0,7766 |
| train | B0048 | 68 | 4°C / 1 A / 2,7 V | 0,7728 |
| train | B0056 | 101 | 4°C / 2 A / 2,7 V | 0,8183 |
| test | B0007 | 167 | 24°C / 2 A / 2,2 V | 0,7464 |
| test | B0034 | 194 | 24°C / 4 A / 2,2 V | 0,7988 |
| test | B0047 | 68 | 4°C / 1 A / 2,5 V | 0,7444 |
| test | B0055 | 99 | 4°C / 2 A / 2,5 V | 0,7907 |

- **Nhận xét:** tier A chỉ có hai mức nhiệt độ (4°C, 24°C) và đều là `constant_current`. Vì vậy chỉ cần 1 dummy nhiệt độ, và bỏ biến `profile` (hằng số). Hệ quả: **mô hình không được dùng để dự báo cho pin ở 43°C hay xung vuông** (ngoài miền dữ liệu train).

#### 2. Đặc trưng (WBS C6.2)

Với lần xả ở chu kỳ *t*, chỉ dùng thông tin đã biết **trước** lần xả đó, theo danh sách được phép ở Go.md 4.6. Tạo trong class `FeatureBuilder` (lag tính riêng trong từng pin, trên các dòng của protocol).

| Đặc trưng | Định nghĩa | Lý do được phép |
|---|---|---|
| `soh_lag1` … `soh_lag5` | `soh.shift(k)` trong từng pin, k = 1..5 | giá trị quá khứ |
| `soh_roll5` | trung bình của 5 lag | giá trị quá khứ |
| `chg_cc_time_s` | thời gian pha CC của lần sạc ngay trước (F7) | đo trước khi xả |
| `Re_ohm`, `Rct_ohm` | trở kháng lần đo EIS gần nhất trước chu kỳ *t* | đã ghép ngược thời gian ở tiền xử lý |
| `T_max_prev_C` | `T_max_C.shift(1)`: nhiệt độ cell cao nhất của lần xả *t−1* | giá trị *t−1*, không phải *t* |
| `log_rest_h` | `log1p(rest_before_h)`; dùng log vì `rest_before_h` có skewness 10,6 (Introduction_of_dataset mục 8) | biết khi bắt đầu xả |
| `is_4C` | 1 nếu `ambient_C < 10`, ngược lại 0 | điều kiện thiết kế |
| `I_level_A`, `cutoff_V` | mức dòng xả, `discharge_cutoff_V` | điều kiện thiết kế |

- **Tổng: 14 đặc trưng.** Biến bị cấm (`capacity_*`, `energy_Wh`, `duration_s`, `t_to_cutoff_s`, `T_max_C` và `V_min` của chính chu kỳ *t*, `C0_Ah`, `soh_segment`, `soh`) được kiểm tra bằng `FeatureBuilder.check_no_leakage()`: chương trình dừng nếu có biến cấm lọt vào.
- **Dòng train bị bỏ:** 5 chu kỳ đầu mỗi pin chưa đủ 5 lag, nên bỏ 40 dòng. Còn **907 / 947 dòng train**.
- **Giá trị thiếu:** 57 dòng thiếu `Re_ohm`/`Rct_ohm` (19 chu kỳ đầu của B0005, B0006, B0007, trước lần EIS đầu tiên). Trong 907 dòng train còn 28 dòng thiếu. Điền bằng **median** (`SimpleImputer`), đặt **trong Pipeline** nên median chỉ tính trên phần train của mỗi fold. Không dòng `is_eval` nào thiếu (đã `assert`).
- **Chuẩn hóa:** `StandardScaler` (z-score), cũng trong Pipeline. Lý do: SVR với kernel RBF dựa trên khoảng cách giữa các điểm; `chg_cc_time_s` có SD ≈ 918 s trong khi `Re_ohm` có SD ≈ 0,015 Ω, không chuẩn hóa thì kernel chỉ "thấy" thời gian sạc.
- **Pipeline:** `SimpleImputer(median) → StandardScaler → SVR(kernel="rbf")` (`SVRTuner.make_pipeline()`).

#### 3. Mô hình

SVR kernel RBF (Chap 6, tr. 31). Ba siêu tham số:
- **C:** trọng số phạt khi sai số vượt quá ε. C lớn thì bám sát dữ liệu train (dễ overfit), C nhỏ thì hàm mượt hơn.
- **ε:** nửa độ rộng "ống" dung sai, đơn vị SOH. Sai số nhỏ hơn ε không bị phạt.
- **γ:** độ cục bộ của kernel RBF, K(x, x′) = exp(−γ‖x − x′‖²). γ nhỏ thì hàm gần tuyến tính, γ lớn thì hàm uốn lượn.

#### 4. Chọn siêu tham số (WBS C6.3): ghi theo trình tự thực tế

Tiêu chí chung: `GroupKFold(4)` theo `battery_id` **trên tập train** (Rec.md mục 6). Mỗi fold giữ lại nguyên 2 pin train. Điểm = RMSE dự báo 1 bước. Naive trên cùng các fold: **CV RMSE = 0,01393**.

**4.1. Lưới gốc của Go.md** (C ∈ {0,1; 1; 10; 100}, ε ∈ {0,001; 0,005; 0,01}, γ ∈ {'scale'; 0,01; 0,1}, 36 tổ hợp):
- Tốt nhất: C = 0,1, ε = 0,001, γ = 0,01, **CV RMSE = 0,01800**, tức **kém hơn naive**.
- Cả 3 tham số đều rơi đúng biên nhỏ nhất của lưới, nghĩa là điểm tối ưu có thể nằm ngoài lưới. Vì vậy quyết định mở rộng lưới.

**4.2. Mở rộng lưới lần 1** (thăm dò bằng script ngoài repo; C ∈ {0,003; 0,01; 0,03; 0,1; 0,3; 1}, ε ∈ {0,0003; 0,001; 0,003}, γ ∈ {0,0003; 0,001; 0,003; 0,01; 0,03}):
- Tốt nhất: C = 1, ε = 0,0003, γ = 0,001, **CV RMSE = 0,01305**, lần đầu tiên thắng naive. Theo fold: 0,0119 / 0,0094 / 0,0130 / 0,0178, so với naive 0,0123 / 0,0111 / 0,0123 / 0,0200.
- C = 1 lại nằm ở biên trên.

**4.3. Mở rộng lần 2** (C tới 100, γ xuống 0,0003):
- Tốt nhất: C = 3, ε = 0,001, γ = 0,0003, **CV RMSE = 0,01281**.
- γ = 0,0003 nằm ở biên dưới.

**4.4. Lưới cuối** (dùng trong notebook): C ∈ {0,01; 0,03; 0,1; 0,3; 1; 3; 10; 30; 100}, ε ∈ {0,0001; 0,0003; 0,001; 0,005; 0,01}, γ ∈ {'scale'; 0,00003; 0,0001; 0,0003; 0,001; 0,003; 0,01; 0,1}, tổng **9 × 5 × 8 = 360 tổ hợp**, mỗi tổ hợp 4 fold. Lưới này chứa toàn bộ giá trị của Go.md.
- Tốt nhất theo RMSE 1 bước: **C = 100, ε = 0,005, γ = 0,0001, CV RMSE = 0,01237**. C = 100 là giá trị lớn nhất của Go.md, nên không mở rộng C thêm.
- 10 tổ hợp đầu có CV RMSE từ 0,01237 đến 0,01281, chênh nhau ít hơn độ lệch chuẩn giữa các fold (khoảng 0,003). Tức là bảng xếp hạng **rất phẳng** theo hướng "C lớn, γ nhỏ".

**4.5. Phát hiện: RMSE 1 bước không phân biệt được, RMSE cuốn chiếu thì có.**
- ⚠ **Ghi trung thực:** trong lúc thử, đã chạy cấu hình C = 100 trên tập test. RMSE E2 = **0,1951**, so với 0,0721 của cấu hình C = 3, ε = 0,001, γ = 0,0003 chạy trước đó. Như vậy vấn đề được phát hiện **sau khi đã nhìn tập test**.
- Để không chọn tham số bằng tập test, đã đo **RMSE cuốn chiếu trên chính các fold train**: mỗi pin bị giữ lại được dự báo từ mốc 60% chuỗi đến hết, giống E2. Kết quả (script thăm dò):

| C | ε | γ | RMSE 1 bước (CV) | RMSE cuốn chiếu (CV) |
|---|---|---|---|---|
| 100 | 0,005 | 0,0001 | 0,01237 | 0,09459 |
| 30 | 0,005 | 0,0001 | 0,01259 | 0,09444 |
| 10 | 0,001 | 0,0001 | 0,01266 | 0,04754 |
| 3 | 0,001 | 0,0003 | 0,01281 | 0,04646 |
| 1 | 0,0003 | 0,001 | 0,01305 | 0,04873 |
| 1 | 0,001 | 0,001 | 0,01317 | 0,05222 |
| 0,3 | 0,001 | 0,003 | 0,01453 | 0,05132 |
| 0,1 | 0,001 | 0,01 | 0,01800 | 0,04589 |
| 1 | 0,001 | 0,01 | 0,02273 | 0,08280 |
| 10 | 0,001 | 0,01 | 0,03573 | 0,13880 |
| 1 | 0,001 | 'scale' | 0,03836 | 0,08390 |
| naive | – | – | 0,01393 | **0,04503** |

- Kết luận: trên train, các cấu hình C lớn cũng có RMSE cuốn chiếu **gấp đôi**. Vấn đề phát hiện được mà không cần tập test.
- **Không cấu hình SVR nào thắng naive ở RMSE cuốn chiếu**, kể cả trên train.

**4.6. Quy tắc chọn cuối cùng: one-standard-error (1-SE) + RMSE cuốn chiếu** (`SVRTuner.select_one_se()`, chỉ dùng train):
1. SE của cấu hình tốt nhất = SD giữa 4 fold / √4 = 0,00306 / 2 = **0,00153**. Giữ các cấu hình có CV RMSE 1 bước ≤ 0,01237 + 0,00153: được **41 cấu hình** ("ngang điểm").
2. Với 41 cấu hình này, tính RMSE cuốn chiếu bằng GroupKFold (như 4.5); chọn cấu hình nhỏ nhất.

10 cấu hình đầu theo RMSE cuốn chiếu:

| C | ε | γ | RMSE 1 bước | RMSE cuốn chiếu |
|---|---|---|---|---|
| **3** | **0,005** | **0,0003** | **0,01295** | **0,04594** |
| 30 | 0,005 | 0,00003 | 0,01282 | 0,04604 |
| 30 | 0,001 | 0,00003 | 0,01267 | 0,04607 |
| 3 | 0,001 | 0,0003 | 0,01281 | 0,04646 |
| 10 | 0,001 | 0,0001 | 0,01266 | 0,04754 |
| 1 | 0,0001 | 0,001 | 0,01308 | 0,04791 |
| 3 | 0,0001 | 0,0003 | 0,01287 | 0,04800 |
| 3 | 0,0003 | 0,0003 | 0,01288 | 0,04805 |
| 1 | 0,0003 | 0,001 | 0,01305 | 0,04873 |
| 30 | 0,0003 | 0,00003 | 0,01273 | 0,04877 |

- **Được chọn: C = 3, ε = 0,005, γ = 0,0003.** Cả C và γ đều nằm hẳn bên trong lưới (`SVRTuner.edge_check()`).
- Cấu hình thứ 2 và 3 chỉ kém 0,0001, nên đây là **đại diện của một vùng tham số tốt**, không phải giá trị duy nhất đúng.
- So với naive theo fold (cấu hình C = 100 của 4.4): 0,01168 / 0,00911 / 0,01130 / 0,01739 so với naive 0,01232 / 0,01112 / 0,01231 / 0,01995. SVR thắng ở cả 4 fold.
- Hình: `reports/figures/svr_tuning_heatmap.png`. Trục ngang là C, trục dọc là γ; ô trái là RMSE 1 bước, ô phải là RMSE cuốn chiếu của 41 cấu hình. Hình cho thấy vùng tốt là **một dải chéo** (γ nhỏ thì cần C lớn), lưới gốc Go.md nằm ngoài dải này, và trong dải RMSE cuốn chiếu khác nhau tới khoảng 4 lần.

**Diễn giải các giá trị được chọn:**
- **ε = 0,005** = 0,5 điểm % SOH. Mức này xấp xỉ 2–3 chu kỳ suy giảm bình thường ở 24°C/2 A (khoảng 17,9 điểm % / 100 chu kỳ theo Introduction_of_dataset mục 8), và nhỏ hơn mức nhiễu (RMSE naive ≈ 0,014). Kết quả không nhạy với ε: trong 41 cấu hình, ε từ 0,0001 đến 0,005 cho RMSE gần nhau.
- **γ = 0,0003** rất nhỏ so với `'scale'` (≈ 1/14 ≈ 0,07). Với 14 đặc trưng đã chuẩn hóa, ‖x − x′‖² điển hình ≈ 28, nên γ‖x − x′‖² ≈ 0,008 và kernel ≈ 0,99: hàm gần tuyến tính. **Quan hệ giữa đặc trưng và SOH chu kỳ sau là gần tuyến tính**; `'scale'` cho CV RMSE ≥ 0,0368.
- **C = 3** là mức phạt vừa phải trong dải chéo. C lớn hơn (30, 100) ở γ = 0,0003 cho RMSE cuốn chiếu 0,165–0,199.

**4.7. Phương án đã thử và không dùng: dự báo ΔSOH thay vì SOH.**
- Thử trong script thăm dò trên lưới gốc Go.md: mục tiêu là ΔSOH = SOH(t) − SOH(t−1), rồi cộng lại với `soh_lag1`.
- CV RMSE 1 bước = 0,01444, tốt hơn dự báo SOH trực tiếp (0,01800) trên cùng lưới.
- ⚠ Đã chạy trên tập test: E1 RMSE 0,0181, nhưng **E2 RMSE 0,1704**, tức dự báo cuốn chiếu bị trôi mạnh.
- **Không dùng**, vì Go.md quy định biến mục tiêu là `soh`, và vì E2 bị trôi. Lý do thứ hai có được từ tập test, nên chỉ ghi lại để tham khảo, không dùng làm căn cứ chọn.

#### 5. Dự báo trên tập test (WBS C6.5)

Model cuối là cấu hình đã chọn, train lại trên toàn bộ 907 dòng train (`SVRForecaster.fit()`).
- **E1 (dự báo 1 bước):** với mỗi dòng `is_eval`, lag là SOH **thật** của các chu kỳ trước.
- **E2 (dự báo cuốn chiếu):**
  - Mốc là dòng `is_eval` đầu tiên của mỗi pin test. Mỗi giá trị vừa dự báo trở thành `soh_lag1` của bước sau (`RecursiveForecaster.forecast()`).
  - **Quyết định:** các số đo chưa biết sau mốc (`chg_cc_time_s`, `Re_ohm`, `Rct_ohm`, `log_rest_h`, `T_max_prev_C`) được **giữ ở giá trị cuối cùng trước mốc**. Lý do: protocol chỉ cho dùng thông tin trước mốc. Điều kiện vận hành (`is_4C`, `I_level_A`, `cutoff_V`) cố định theo thiết kế nên vẫn dùng.
- **Khoảng dự báo 95% cho E1 (split-conformal):**
  - Lấy phần dư out-of-fold của model cuối trên 907 dòng train (`cross_val_predict` với cùng GroupKFold(4)). q = phân vị 95% của |sai số| = **0,0275 SOH**. Khoảng = ŷ ± 0,0275.
  - **Tỉ lệ bao phủ thực tế trên test = 88,7%**, thấp hơn mục tiêu 95%: pin test có biến động lớn hơn pin train, nhất là B0055.
  - Không tính khoảng cho E2 (sai số tăng theo số bước); `y_lower`/`y_upper` để trống như định dạng chung cho phép.

#### 6. Kết quả (Chap 3, tr. 81–86)

Tính bằng `evaluate()` của protocol, trên 213 dòng `is_eval`.

| Kịch bản | Mô hình | MAE | RMSE | MAPE (%) | R² |
|---|---|---|---|---|---|
| E1 | naive | 0,0119 | 0,0226 | 1,4288 | 0,5001 |
| E1 | **SVR** | **0,0100** | **0,0172** | **1,2000** | **0,7108** |
| E2 | naive | 0,0249 | 0,0343 | 3,1271 | −0,1552 |
| E2 | SVR | 0,0544 | 0,0637 | 6,6974 | −2,9763 |

RMSE theo từng pin test:

| Kịch bản | Pin | naive | SVR | SVR tốt hơn? |
|---|---|---|---|---|
| E1 | B0007 | 0,0037 | 0,0055 | Không |
| E1 | B0034 | 0,0135 | 0,0107 | Có |
| E1 | B0047 | 0,0172 | 0,0117 | Có |
| E1 | B0055 | 0,0462 | 0,0347 | Có |
| E2 | B0007 | 0,0515 | 0,0559 | Không |
| E2 | B0034 | 0,0179 | 0,0817 | Không |
| E2 | B0047 | 0,0199 | 0,0341 | Không |
| E2 | B0055 | 0,0305 | 0,0505 | Không |

Sai số RUL ở E2 (B0047 không tính vì đã xuống dưới 0,8 trước mốc; `rul_error` = chu kỳ dự báo chạm 0,8 − chu kỳ thật, âm là cảnh báo sớm):

| Pin | Chạm 0,8 thật | SVR dự báo | `rul_error` | naive |
|---|---|---|---|---|
| B0007 | 126 | 110 | −16 | không bao giờ chạm |
| B0034 | 186 | 129 | −57 | không bao giờ chạm |
| B0055 | 65 | 80 | +15 | không bao giờ chạm |

**Diễn giải:**
1. **E1: SVR thắng naive**, RMSE giảm 24% (0,0226 → 0,0172) và thắng ở 3/4 pin.
   - B0007 là ngoại lệ: đường suy giảm rất mượt, naive đã gần như đúng (RMSE 0,0037).
2. **E2: SVR thua naive về sai số ở cả 4 pin.**
   - Nguyên nhân: dự báo cuốn chiếu đi dốc hơn thực tế (hình `svr_forecasts.png`, đường cam), và sai số cộng dồn qua từng bước.
   - Nhất quán với kết quả trên train (mục 4.5: không cấu hình nào thắng naive ở RMSE cuốn chiếu).
3. **SVR vẫn ước lượng được thời điểm chạm EOL**, naive thì không (giữ nguyên một giá trị).
   - 2/3 pin được cảnh báo sớm (an toàn nhưng tốn chi phí thay pin sớm), 1 pin cảnh báo muộn 15 chu kỳ.
   - Sai số lớn nhất là B0034 (−57), pin có SOH dao động quanh 0,8 rất lâu trước khi chạm.

#### 7. Đặc trưng nào quan trọng (WBS C6.4, RQ4)

Permutation importance trên 213 dòng E1 của tập test, 30 lần lặp, `random_state = 42`, điểm `neg_root_mean_squared_error` (`ImportanceAnalyzer.run()`). RMSE gốc = 0,01718. Vì 5 lag và `soh_roll5` chứa gần như cùng thông tin, xáo riêng một cột thì model vẫn đọc được từ các cột còn lại. Vì vậy đã xáo thêm **cả nhóm 6 cột lịch sử SOH** bằng cùng một hoán vị dòng.

| Đặc trưng | RMSE tăng khi xáo | SD |
|---|---|---|
| [nhóm] lịch sử SOH (lag 1–5 + roll5) | 0,02422 | 0,00126 |
| `soh_lag1` | 0,00980 | 0,00091 |
| `log_rest_h` | 0,00289 | 0,00025 |
| `I_level_A` | 0,00254 | 0,00038 |
| `soh_lag2` | 0,00102 | 0,00031 |
| `T_max_prev_C` | 0,00058 | 0,00021 |
| `soh_roll5` | 0,00057 | 0,00023 |
| `Re_ohm` | 0,00019 | 0,00011 |
| `soh_lag5` | 0,00009 | 0,00009 |
| `chg_cc_time_s` | 0,00008 | 0,00006 |
| `is_4C` | 0,00004 | 0,00008 |
| `Rct_ohm` | 0,00000 | 0,00005 |
| `cutoff_V` | −0,00001 | 0,00005 |
| `soh_lag4` | −0,00003 | 0,00002 |
| `soh_lag3` | −0,00006 | 0,00010 |

**Trả lời RQ4:**
- Thông tin quan trọng nhất là **lịch sử SOH gần đây**, chủ yếu `soh_lag1`.
- Ngoài lịch sử SOH, chỉ **thời gian nghỉ trước khi xả** (`log_rest_h`, giải thích hiện tượng hồi phục dung lượng F5) và **mức dòng xả** (`I_level_A`) đóng góp rõ.
- **Đặc trưng sạc (`chg_cc_time_s`) và trở kháng (`Re_ohm`, `Rct_ohm`) gần như không thêm thông tin** khi đã biết SOH gần đây. Điều này không mâu thuẫn với F7 (tương quan `chg_cc_time_s`–SOH = 0,86 ở G1): biến này mang cùng thông tin với SOH, nên khi đã có lag SOH thì không còn gì mới.
- Các giá trị âm rất nhỏ (`cutoff_V`, `soh_lag3`, `soh_lag4`) nằm trong mức nhiễu, đọc là "không quan trọng".
- **Việc còn lại:** đối chiếu với hệ số hồi quy của Quốc Đạt (WBS C6.4) khi có kết quả.

#### 8. Kiểm chứng

- `assert` về số dòng (947 / 528 / 213), không trùng pin giữa train và test, không dùng biến cấm, mọi dòng `is_eval` đủ 5 lag.
- `svr.csv` qua được `evaluate()` của protocol (đúng 213 dòng `is_eval` ở mỗi kịch bản, `y_true` đúng là `soh`). Notebook đọc lại file đã lưu và gọi `evaluate()` lần nữa.
- **Tái lập:** chạy lại toàn bộ notebook (Restart & Run All) cho ra đúng các số trên, mất khoảng 40 giây.
- Khi viết lại code từ notebook sang các class trong `src/svr_model.py`, đã so sánh `svr.csv` của hai phiên bản: 426 dòng trùng khớp hoàn toàn (sai lệch lớn nhất của `y_pred`, `y_lower`, `y_upper` = 0).
- **Môi trường chạy ra các số trên:** Python 3.10, scikit-learn 1.7.2, pandas 2.3.3, numpy 2.2.6.

#### 9. Đầu ra

| File | Nội dung |
|---|---|
| `src/svr_model.py` | Toàn bộ code, 8 class: `SVRConfig`, `FeatureBuilder`, `RecursiveForecaster`, `SVRTuner`, `SVRForecaster`, `ImportanceAnalyzer`, `SVRPlotter`, `SVRExperiment`. Chạy cả quy trình: `python src/svr_model.py` |
| `notebooks/30_ml.ipynb` | Gọi từng class theo thứ tự, hiển thị bảng trung gian, giải thích bằng tiếng Việt |
| `reports/predictions/svr.csv` | 426 dòng = 213 dòng `is_eval` × 2 kịch bản. Cột: `model` (= svr), `battery_id`, `cycle_valid`, `scenario` (E1/E2), `y_true` (SOH thật), `y_pred`, `y_lower`, `y_upper` (khoảng 95%, chỉ có ở E1) |
| `reports/figures/svr_tuning_heatmap.png` | Heatmap chọn tham số (mục 4.6) |
| `reports/figures/svr_forecasts.png` | 4 ô, mỗi ô một pin test: SOH thật, dự báo E1 kèm khoảng 95%, dự báo E2, mốc 60% và đường EOL 0,8 |
| `reports/figures/svr_permutation_importance.png` | Biểu đồ thanh ngang của bảng mục 7; thanh màu cam là nhóm lịch sử SOH |

#### 10. Hạn chế và việc còn lại

- **Hạn chế cần đưa vào báo cáo:**
  - SVR thua naive ở E2.
  - Khoảng E1 chỉ bao phủ 88,7%.
  - Chỉ 8 pin train, 2 mức nhiệt độ; không ngoại suy ra 43°C hay xung vuông.
  - Quy tắc 1-SE (mục 4.6) được đặt ra sau khi đã thấy kết quả test một lần (mục 4.5). Việc chọn vẫn chỉ dùng train, nhưng phải nêu rõ.
- **Chờ Khải Hưng review:**
  - mở rộng lưới tham số ra ngoài Go.md;
  - quy tắc 1-SE với RMSE cuốn chiếu;
  - giữ các số đo ở giá trị cuối cùng trong E2;
  - khoảng conformal cho E1.
- **Trước khi mở Pull Request vào `main`:** notebook và comment trong code đang bằng tiếng Việt, trong khi README quy định nội dung trên `main` bằng tiếng Anh. Cần hỏi Leader.
- **Việc tiếp theo:** đối chiếu importance với hồi quy (C6.4); viết Report §3.5 và §5.5 (C6.6).

---

### 14/11/2026 · Minh Vũ · branch `minh-vu` · GĐ5c SVR, cập nhật sau review · WBS C6.2–C6.5

**Lý do cập nhật:** review của Khải Hưng ngày 13/11/2026 (`Member_Review_MinhVu.md`, commit được review `3f85a07`) yêu cầu 4 việc bắt buộc trước khi mở PR: (4.1) bỏ dòng thiếu Re/Rct thay vì điền median, (4.2) script lỗi `UnicodeEncodeError` trên console Windows, (4.3) code và notebook phải bằng tiếng Anh, (4.4) ghi Work_log. Mục này ghi lại các thay đổi và **thay thế các con số cuối cùng** của mục 05/11 (mục 05/11 giữ nguyên làm lịch sử, xem ghi chú ở đầu mục đó).
**Code:** `src/svr_model.py`, `notebooks/30_ml.ipynb`, `reports/predictions/svr.csv`, 3 hình `reports/figures/svr_*.png` (commit sau 14/11, xem lịch sử nhánh `minh-vu`).
**Trạng thái:** đã sửa đủ 4.1–4.3; chờ Khải Hưng review lại.

#### 1. Sửa 4.1: bỏ dòng thiếu Re/Rct, không điền (Rec.md 6.1 #6)

- **Vấn đề:** bản 05/11 điền median cho Re/Rct bị thiếu (`SimpleImputer` trong Pipeline). Quy tắc đã chốt ngày 02/11/2026 (Rec.md 6.1 #6, hiện nằm trên nhánh `khai-hung`) là bỏ dòng có NaN, không điền.
- **Cách sửa:**
  1. `FeatureBuilder.training_matrix()`: điều kiện lọc thêm `Re_ohm`, `Rct_ohm` (`has_history = X[lag_cols + ["T_max_prev_C", "Re_ohm", "Rct_ohm"]].notna().all(axis=1)`).
  2. `SVRTuner.make_pipeline()`: bỏ `SimpleImputer`, Pipeline còn `StandardScaler → SVR(kernel="rbf")`. Bỏ import `SimpleImputer`.
  3. `SVRForecaster.predict_e1()`: thêm `assert` để dừng nếu bất kỳ dòng `is_eval` nào thiếu đặc trưng.
- **Quy mô:** 947 dòng train → bỏ 40 dòng thiếu lag → bỏ thêm **28 dòng thiếu Re/Rct** (B0005: 14, B0006: 14) → còn **879 dòng** (trước đây 907). Khớp với con số Leader tính trong review. Không dòng `is_eval` nào thiếu (assert qua).
- **E2 không bị ảnh hưởng về quy tắc:** mốc 60% của các pin test nằm sau chu kỳ 19, nên giá trị Re/Rct cuối cùng trước mốc luôn có.

#### 2. Kết quả chọn tham số sau khi sửa (chạy lại toàn bộ, chỉ dùng train)

- **Naive trên cùng các fold:** RMSE 1 bước = **0,01407**; RMSE cuốn chiếu = **0,04503**.
- **Lưới gốc Go.md:** tốt nhất C = 0,1, ε = 0,001, γ = 0,01, CV RMSE = **0,01789**, vẫn kém naive. Lưới cuối giữ nguyên như mục 05/11 (360 tổ hợp).
- **Tốt nhất theo RMSE 1 bước:** C = 30, ε = 0,0003, γ = 0,00003, CV RMSE = **0,01305**. Theo fold: 0,01314 / 0,00975 / 0,01100 / 0,01833, so với naive 0,01232 / 0,01146 / 0,01256 / 0,01995 (SVR thắng 3/4 fold).
- **Quy tắc 1-SE:** SE = 0,00328 / √4 = **0,00164**; **46 cấu hình** nằm trong 1 SE (trước đây 41).
- 10 cấu hình đầu theo RMSE cuốn chiếu (CV):

| C | ε | γ | RMSE 1 bước | RMSE cuốn chiếu |
|---|---|---|---|---|
| **10** | **0,005** | **0,00003** | **0,01416** | **0,04913** |
| 1 | 0,005 | 0,0003 | 0,01426 | 0,04934 |
| 3 | 0,005 | 0,0001 | 0,01423 | 0,04961 |
| 3 | 0,01 | 0,0001 | 0,01460 | 0,05222 |
| 10 | 0,01 | 0,00003 | 0,01457 | 0,05265 |
| 10 | 0,0003 | 0,00003 | 0,01400 | 0,05370 |
| 0,3 | 0,001 | 0,001 | 0,01459 | 0,05377 |
| 10 | 0,0001 | 0,00003 | 0,01404 | 0,05444 |
| 1 | 0,0003 | 0,0003 | 0,01412 | 0,05508 |
| 10 | 0,001 | 0,00003 | 0,01413 | 0,05509 |

- **Được chọn: C = 10, ε = 0,005, γ = 0,00003** (trước đây C = 3, ε = 0,005, γ = 0,0003).
- **Ngay trong CV, SVR đã không thắng naive ở dự báo cuốn chiếu:** 0,04913 so với 0,04503.
- **Diễn giải:** ba cấu hình đầu (C = 10, γ = 3e-5), (C = 1, γ = 3e-4), (C = 3, γ = 1e-4) nằm trên cùng một **dải chéo** C × γ và chỉ chênh nhau dưới 0,0005. Với γ nhỏ, kernel RBF gần tuyến tính, nên γ nhỏ hơn cần C lớn hơn để khớp như nhau. Cấu hình cũ (C = 3, γ = 3e-4) và cấu hình mới đều thuộc dải này; việc bỏ 28 dòng chỉ làm thứ tự trong dải thay đổi.

#### 3. Kiểm tra biên của γ (mới)

- `SVRTuner.edge_check()`: C = 10 nằm trong lưới, nhưng **γ = 0,00003 là giá trị nhỏ nhất của lưới**.
- **Kiểm tra thêm, chỉ dùng train** (`SVRTuner.ridge_check()`, notebook bước 4c): thử γ = 0,00001 với C ∈ {10, 30, 100}, ε = 0,005.

| C | γ | RMSE 1 bước (CV) | RMSE cuốn chiếu (CV) | so với cấu hình chọn |
|---|---|---|---|---|
| 10 | 0,00001 | 0,01642 | 0,06632 | +0,01719 |
| 10 | 0,00003 | 0,01416 | 0,04913 | 0 (được chọn) |
| 30 | 0,00001 | 0,01411 | 0,04843 | −0,00070 (−1,4%) |
| 30 | 0,00003 | 0,01349 | 0,07000 | +0,02087 |
| 100 | 0,00001 | 0,01380 | 0,08211 | +0,03298 |
| 100 | 0,00003 | 0,01438 | 0,14061 | +0,09148 |

- **Quyết định: giữ nguyên lưới, không mở rộng thêm.** Lý do:
  1. Cấu hình tốt nhất ngoài lưới (C = 30, γ = 0,00001) cũng nằm trên dải chéo và chỉ tốt hơn 1,4%, vẫn kém naive (0,04843 so với 0,04503).
  2. RMSE cuốn chiếu rất nhạy khi rời khỏi dải chéo (ví dụ C = 30, γ = 0,00003 cho 0,07000), tức là tiêu chí này nhiễu; chênh 1% không được coi là cải thiện thật.
  3. Mở rộng lưới lần thứ 3 chỉ để đổi lấy 1% sẽ dễ thành dò tham số quá mức (over-tuning) trên 8 pin train.

#### 4. Kết quả trên tập test sau khi sửa

Tính bằng `evaluate()` của protocol trên 213 dòng `is_eval`. `svr.csv` qua `check_pred()`.

| Kịch bản | Mô hình | MAE | RMSE | MAPE (%) | R² |
|---|---|---|---|---|---|
| E1 | naive | 0,0119 | 0,0226 | 1,4288 | 0,5001 |
| E1 | **SVR** | **0,0102** | **0,0172** | **1,2225** | **0,7091** |
| E2 | naive | 0,0249 | 0,0343 | 3,1271 | −0,1552 |
| E2 | SVR | 0,0476 | 0,0588 | 5,8139 | −2,3864 |

RMSE theo từng pin test:

| Kịch bản | Pin | naive | SVR | SVR tốt hơn? |
|---|---|---|---|---|
| E1 | B0007 | 0,0037 | 0,0055 | Không |
| E1 | B0034 | 0,0135 | 0,0125 | Có |
| E1 | B0047 | 0,0172 | 0,0138 | Có |
| E1 | B0055 | 0,0462 | 0,0331 | Có |
| E2 | B0007 | 0,0515 | 0,0314 | Có |
| E2 | B0034 | 0,0179 | 0,0847 | Không |
| E2 | B0047 | 0,0199 | 0,0360 | Không |
| E2 | B0055 | 0,0305 | 0,0433 | Không |

Sai số RUL ở E2 (âm = cảnh báo sớm):

| Pin | Chạm 0,8 thật | SVR dự báo | `rul_error` | SOH thấp nhất thật | naive |
|---|---|---|---|---|---|
| B0007 | 126 | 112 | −14 | 0,7464 | không chạm |
| B0034 | 186 | 128 | −58 | **0,7988** | không chạm |
| B0055 | 65 | 87 | +22 | 0,7907 | không chạm |

- **Khoảng conformal E1:** q = **0,0300 SOH**, tỉ lệ bao phủ thực tế **91,1%**, vẫn thấp hơn mức danh nghĩa 95%.

**Diễn giải (dùng cho báo cáo C6.6):**
1. **E1 tốt:** RMSE 0,0172 so với naive 0,0226; thắng 3/4 pin. Thua ở B0007 (0,0055 so với 0,0037) vì B0007 là chuỗi trơn, naive gần như hoàn hảo.
2. **E2 là điểm yếu:** MAE 0,0476, gấp khoảng 1,9 lần naive (0,0249); thua naive ở 3/4 pin, chỉ thắng ở B0007. So với bản 05/11, E2 tốt hơn (MAE 0,0544 → 0,0476) nhưng kết luận không đổi. Giải thích khả dĩ: sai số tích lũy khi cuốn chiếu, và mô hình học từ lag nên dự báo trôi về mức suy giảm trung bình của tập train. Phải viết thẳng trong báo cáo.
3. **RUL:** cảnh báo sớm 14 và 58 chu kỳ ở B0007 và B0034, muộn 22 chu kỳ ở B0055. B0034 chỉ chạm sát 0,8 (SOH thấp nhất 0,7988), nên sai số RUL của pin này rất nhạy, phải ghi chú như Rec.md 6.1 #3.
4. **Khoảng conformal phủ 91,1% < 95%:** phần dư out-of-fold trên pin train nhỏ hơn sai số thật trên pin test (pin test khác điều kiện). Không gọi là "hiệu chỉnh tốt".
5. **E2 không có khoảng dự báo:** hợp lệ theo định dạng chung, nhưng là hạn chế. Hướng mở rộng (chưa làm): mô phỏng cuốn chiếu nhiều lần, mỗi bước cộng nhiễu lấy mẫu từ phần dư out-of-fold, rồi lấy phân vị 2,5% và 97,5%.

#### 5. Permutation importance sau khi sửa (RQ4)

30 lần lặp, `random_state = 42`, trên 213 dòng E1 của test. RMSE gốc = **0,01723**.

| Đặc trưng | RMSE tăng khi xáo | SD |
|---|---|---|
| [nhóm] lịch sử SOH (lag 1–5 + roll5) | 0,02358 | 0,00123 |
| `soh_lag1` | 0,00540 | 0,00068 |
| `I_level_A` | 0,00341 | 0,00043 |
| `log_rest_h` | 0,00263 | 0,00024 |
| `soh_lag2` | 0,00171 | 0,00039 |
| `is_4C` | 0,00070 | 0,00024 |
| `soh_roll5` | 0,00057 | 0,00024 |
| `Re_ohm` | 0,00037 | 0,00019 |
| `chg_cc_time_s` | 0,00027 | 0,00012 |
| `T_max_prev_C` | 0,00017 | 0,00014 |
| `soh_lag5` | 0,00014 | 0,00013 |
| `soh_lag3` | 0,00011 | 0,00018 |
| `Rct_ohm` | 0,00001 | 0,00004 |
| `cutoff_V` | −0,00001 | 0,00005 |
| `soh_lag4` | −0,00008 | 0,00012 |

**Diễn giải:**
- Lịch sử SOH vẫn áp đảo. Sau đó là `I_level_A` và `log_rest_h` (thứ tự hai biến này đổi so với bản 05/11, nhưng cả hai vẫn là nhóm thứ hai).
- **Lưu ý về `I_level_A`** (theo review): biến này là hằng số trong từng pin. Xáo nó trên 4 pin test thực chất là gán điều kiện của pin này cho pin khác, nên con số đo "điều kiện vận hành khác nhau giữa các pin", không phải hiệu ứng của dòng xả trong một pin. Cùng lưu ý cho `is_4C`.
- **Trả lời RQ4:** đặc trưng sạc và trở kháng (`chg_cc_time_s` 0,00027, `Re_ohm` 0,00037, `Rct_ohm` 0,00001) gần như không đóng góp; lớn nhất trong ba biến chỉ bằng **1,6%** của nhóm lịch sử SOH. Kết luận: khi đã biết SOH gần đây, đặc trưng sạc và trở kháng dự báo SOH chu kỳ kế tiếp rất kém.
- **Đối chiếu với hồi quy của Quốc Đạt (WBS C6.4)**, theo tóm tắt trong review: hồi quy cho thấy nhiệt độ, dòng xả và cutoff có liên hệ với **tốc độ** suy giảm giữa các pin; SVR cho thấy khi đã có lịch sử SOH thì các biến này thêm rất ít cho dự báo **1 bước**. Hai kết quả không mâu thuẫn: điều kiện vận hành quyết định độ dốc của cả đường cong, còn chu kỳ kế tiếp chủ yếu là chu kỳ trước cộng một bước nhỏ. **Việc còn lại:** cập nhật bằng hệ số cụ thể khi notebook hồi quy vào `main`.

#### 6. Sửa 4.2 và 4.3: tiếng Anh và chạy được trên Windows

- `src/svr_model.py`: dịch toàn bộ docstring, comment, thông báo `assert`, chuỗi `print` và nhãn hình sang tiếng Anh. Thêm hàm `SVRTuner.ridge_check()` (mục 3).
- **Kiểm chứng logic không đổi khi dịch:** so sánh cây cú pháp (AST) của file trước và sau khi dịch, sau khi bỏ docstring và thay mọi chuỗi bằng cùng một giá trị. Chỉ 4 hàm khác nhau, đúng là các chỗ sửa có chủ đích: `FeatureBuilder.training_matrix`, `SVRTuner.make_pipeline`, `SVRForecaster.predict_e1`, `SVRTuner.ridge_check` (mới); `SVRConfig` giữ nguyên.
- **Chạy trên console kiểu Windows:** `PYTHONIOENCODING=cp1252 python src/svr_model.py` chạy hết, **mã thoát 0**. Mọi chuỗi `print` giờ chỉ dùng ký tự ASCII.
- `notebooks/30_ml.ipynb`: viết lại toàn bộ bằng tiếng Anh, 35 ô; Restart & Run All không ô nào lỗi, mọi ô code có output. Thêm: bước 4c (kiểm tra biên γ), bảng thắng/thua naive theo pin, SOH thấp nhất của từng pin test, phần đọc tỉ lệ bao phủ, hai lưu ý về permutation importance, câu trả lời RQ4 và đoạn đối chiếu với hồi quy.
- `reports/figures/svr_tuning_heatmap.png`: tiêu đề và nhãn trục bằng tiếng Anh.

#### 7. Kiểm chứng

- Lệnh kiểm tra ở mục 8 của review: `check_pred(p, te)` và `evaluate(p)` chạy không lỗi.
- `assert`: 947 / 528 / 213 dòng; không trùng pin; không dùng biến cấm; mọi dòng `is_eval` đủ 5 lag và không thiếu đặc trưng nào.
- Môi trường: Python 3.10, scikit-learn 1.7.2, pandas 2.3.3.

#### 8. Việc còn lại

- Chờ Leader merge Rec.md 6.1 vào `main`, rồi merge `origin/main` vào `minh-vu` trước khi mở PR.
- Viết Report §3.5 và §5.5 (C6.6, hạn 15/11) từ mục 2–5 ở trên.
- Cập nhật đoạn đối chiếu với hồi quy khi có hệ số cụ thể.

---

### 14/11/2026 · Minh Vũ · branch `minh-vu` · GĐ2: Thống kê mô tả · WBS C1.1, C1.5–C1.7

**Trả lời:** mô tả dữ liệu trước khi kiểm định và mô hình hóa (Go.md GĐ2, slide Chap 1 §3–4).
**Code:** `src/descriptive_stats.py` (7 class: `DescConfig`, `DescriptiveData`, `MeasurementScales`, `SummaryStats`, `DescriptivePlots`, `InsightWriter`, `DescriptiveAnalysis`), `notebooks/01_descriptive.ipynb` (43 ô), 6 hình `reports/figures/desc_*.png`. Chạy cả quy trình: `python src/descriptive_stats.py`.
**Trạng thái:** làm từ 05/11 đến 10/11 nhưng chưa commit nên chưa có trong lần review 13/11 (review ghi C1.1, C1.5, C1.7 là "không thấy"). Ngày 14/11 đã chuyển code, notebook và hình sang tiếng Anh, bổ sung theo review, chạy lại toàn bộ. **Chưa qua review.**

#### 1. Dữ liệu

- **Mức chu kỳ:** `cycles_clean.csv` (2.658 dòng) → chỉ tier A, B, N (pin có điều kiện không đổi): 2.121 dòng → bỏ dòng `cap_outlier` và `break_in`: **2.065 dòng, 23 pin**. Cách lọc giống `Introduction_of_dataset.md` mục 8; notebook kiểm tra bằng `assert len == 2065` và `23 pin`.
- **Mức pin:** `battery_fade.csv` (23 pin) ghép `C0_Ah` từ `battery_summary.csv`.
- **Nhãn thêm:** `temp` ("4°C" nếu `ambient_C` < 10, "24°C" nếu < 30, còn lại "43°C") và `condition` = nhiệt độ × dòng xả × kiểu xả (ví dụ "24°C, 2 A, CC"), 7 nhóm điều kiện.
- Lý do có hai mức: các chu kỳ trong một pin phụ thuộc nhau (F9), nên so sánh giữa các điều kiện phải dùng **pin** làm đơn vị.

#### 2. Phân loại dữ liệu (C1.1, Chap 1 tr. 13–14, 22–32)

- Cấu trúc: dữ liệu gốc semi-structured (struct MATLAB lồng nhau), đã làm phẳng thành bảng structured ở bước tiền xử lý.
- Kiểu dữ liệu: **panel data** (đơn vị chéo = pin, chuỗi thời gian = `cycle_valid`).
- **Thang đo cho mọi cột** của 3 bảng dùng ở đây, kèm lý do: class `MeasurementScales`, gọi bằng `DescriptiveAnalysis.scale_table()`. Tổng **57 cột**: 9 Nominal, 1 Ordinal (`segment`), 8 Interval, 15 Ratio rời rạc (đếm, vị trí), 24 Ratio. Code dừng (`assert`) nếu một cột chưa được gán thang đo, nên bảng luôn đủ khi thêm cột.
- Theo từ điển dữ liệu (`Introduction_of_dataset.md` mục 6), chỉ khác một chỗ có chủ đích: `chg_end_time` là mốc thời gian nên xếp **Interval** (như `start_time`), không phải Ratio như dòng gộp trong từ điển.
- Hệ quả dùng về sau: nhiệt độ (Interval) chỉ so sánh bằng hiệu, không nói "43°C nóng gấp đôi 22°C"; SOH, dung lượng, điện trở, thời gian nghỉ (Ratio) được so sánh bằng tỉ số.

#### 3. Bảng thống kê mô tả (C1.5, Chap 1 tr. 57)

Chỉ số: n, số thiếu, mean, median, Q1, Q3, SD, SE = SD/√n, min, max, range, skewness (pandas, có hiệu chỉnh mẫu), **kurtosis Pearson = excess + 3** (chuẩn = 3, theo Go.md 2.1).

Mức chu kỳ (n = 2.065):

| Biến | n | mean | median | SD | SE | min | max | skewness | kurtosis |
|---|---|---|---|---|---|---|---|---|---|
| `capacity_int_Ah` | 2065 | 1,5064 | 1,5121 | 0,2543 | 0,0056 | 0,7461 | 2,0467 | −0,52 | 2,92 |
| `soh` | 2065 | 0,8771 | 0,8859 | 0,0868 | 0,0019 | 0,5756 | 1,0413 | −0,64 | 3,07 |
| `T_max_C` | 2065 | 36,84 | 38,73 | 15,66 | 0,34 | 10,20 | 63,02 | −0,13 | 1,95 |
| `Re_ohm` | 2008 | 0,0816 | 0,0789 | 0,0244 | 0,0005 | 0,0276 | 0,1383 | 0,07 | 2,30 |
| `Rct_ohm` | 2008 | 0,1275 | 0,1228 | 0,0489 | 0,0011 | 0,0395 | 0,2524 | 0,44 | 2,28 |
| `rest_before_h` | 2065 | 3,1873 | 0,0240 | 19,83 | 0,44 | 0,0177 | 303,93 | 10,60 | 136,95 |

- `Re_ohm`, `Rct_ohm` thiếu 57 giá trị: 19 chu kỳ đầu của B0005, B0006, B0007 (trước lần đo EIS đầu tiên).
- **Tách theo nhiệt độ môi trường** (vì `T_max_C` gộp có kurtosis 1,95 < 3, trộn 3 nhóm): median `T_max_C` 4°C = 15,04 (Q1–Q3: 13,23–18,65), 24°C = 40,58 (38,04–52,79), 43°C = 61,82 (60,32–62,43).
- ⚠ Bảng SOH theo nhiệt độ **không** được đọc là "nóng thì pin khỏe": nhóm 43°C có SOH trung bình cao vì chỉ chạy tối đa 39 chu kỳ. Notebook in số chu kỳ dài nhất của từng nhóm ngay dưới bảng.
- **Mới thêm theo review (tách theo điều kiện):** `cycle_tables_by_condition()` cho `soh`, `T_max_C`, `rest_before_h` theo 7 nhóm điều kiện (mức chu kỳ, chỉ để mô tả); `battery_table_by_condition()` cho `C0_Ah` và `soh_last` theo nhóm điều kiện (mức pin, không tính B0054).

Mức pin (23 pin): `C0_Ah` mean 1,695 Ah (min 1,031, max 2,037); `soh_last` mean 0,836; `fade_pct_per_100cyc` mean 18,5; `fade25_pct_per_100cyc` mean 28,3; `cycles_to_soh80` chỉ có ở **12 / 23 pin** (median 63).

Tốc độ suy giảm theo nhóm điều kiện (điểm % SOH / 100 chu kỳ, không tính pin nhiễu B0054):

| Nhóm | n | Cả chuỗi (mean) | 25 chu kỳ đầu (mean) |
|---|---|---|---|
| 4°C, 1 A, CC | 4 | 29,9 | 64,8 |
| 4°C, 2 A, CC | 3 | 10,0 | 27,9 |
| 24°C, 2 A, CC | 5 | 17,9 | 14,4 |
| 24°C, 2 A, xung vuông | 1 | 14,6 | 13,3 |
| 24°C, 4 A, CC | 2 | 8,3 | 20,1 |
| 24°C, 4 A, xung vuông | 3 | 10,9 | 9,7 |
| 43°C, 4 A, CC | 4 | 28,4 | 27,8 |

#### 4. Sáu hình (C1.6, Go.md 2.2)

Mỗi hình có khung "How to read" hoặc chú thích trực tiếp trên hình, và một đoạn đọc hình 2–3 câu trong notebook. Các hình được làm lại ngày 10/11 cho dễ đọc (trục thời gian nghỉ ghi bằng phút/giờ/ngày, số trung bình ghi cạnh vạch, đánh dấu điểm chạm EOL, chú thích gai hồi phục và pin nhiễu), rồi dịch sang tiếng Anh ngày 14/11.

| File | Nội dung |
|---|---|
| `desc_soh_hist.png` | Histogram SOH mức chu kỳ, đường mean, median, EOL 0,8 |
| `desc_rest_hist.png` | Histogram thời gian nghỉ, trục log; cột > 1 giờ tô đỏ; bảng n/min/median/mean/max/skewness/kurtosis trong hình |
| `desc_tmax_box.png` | Boxplot `T_max_C` theo nhiệt độ môi trường, vạch đứt = nhiệt độ môi trường, median và Q1–Q3 ghi trên hình |
| `desc_fade_by_condition.png` | Tốc độ suy giảm theo nhóm điều kiện, mỗi chấm 1 pin (không dùng boxplot vì mỗi nhóm 1–5 pin); ô trái cả chuỗi kèm số chu kỳ đã chạy và SOH cuối, ô phải 25 chu kỳ đầu |
| `desc_soh_curves.png` | Đường SOH của 23 pin, 3 ô theo dòng xả; ✕ = lần đầu chạm EOL; chú thích 4 lần nghỉ khoảng 3 ngày (67–73 giờ) ở nhóm 1 A và pin nhiễu B0054 |
| `desc_impedance_soh.png` | Re và Rct theo SOH ở G1, màu + hình dạng theo pin, r gộp và r trong từng pin; chú thích ngoại lệ B0018 |

#### 5. Nhận xét (C1.7, mẫu Chap 1 tr. 60: phát hiện → con số → hàm ý)

8 nhận xét, mọi con số được tính trong code (`InsightWriter`), không gõ tay:
1. **SOH lệch trái:** mean 0,877 < median 0,886, skewness −0,64, kurtosis 3,07. Hàm ý: ít dữ liệu ở giai đoạn cuối đời, dự báo gần EOL bất định nhất đúng lúc cần quyết định thay pin.
2. **Thời gian nghỉ lệch phải cực mạnh:** median 1,4 phút, mean 3,19 giờ, max 303,9 giờ; chỉ 122 / 2.065 chu kỳ (5,9%) nghỉ > 1 giờ. Hàm ý: không dùng mean; vào mô hình phải lấy log (SVR dùng `log1p`) hoặc chia nhóm (cờ `long_rest`).
3. **Dung lượng hồi phục sau nghỉ dài** (mới thêm 14/11): sau nghỉ > 1 giờ, 81,1% chu kỳ có SOH tăng (median ΔSOH +0,0106), so với median −0,0031 khi nghỉ ngắn; thấy ở 18 pin (tier A/B/N). Hàm ý: giữ các điểm này; khi kiểm tra bảo hành, một lần đo ngay sau khi xe để lâu sẽ đánh giá pin khỏe hơn thực tế.
4. **Cell tự nóng khi xả:** median `T_max_C` cao hơn môi trường 11,0°C (4°C), 16,6°C (24°C), 18,8°C (43°C). Hàm ý: xe chạy ở khí hậu nóng bắt buộc có hệ thống làm mát pack.
5. **Tốc độ suy giảm khác nhau giữa điều kiện (giả thuyết cho GĐ3):** cùng 4 A, 25 chu kỳ đầu, 43°C mất 27,8 so với 20,1 ở 24°C; ở 4°C, 1 A (29,9) suy giảm nhanh hơn 2 A (10,0) trên cả chuỗi, ngược trực giác; thứ hạng đổi theo cách đo. Hàm ý (chờ GĐ3 xác nhận): nhiệt độ có vẻ là đòn bẩy mạnh hơn dòng xả.
6. **Trở kháng tăng khi pin già:** G1, r gộp SOH–Re = −0,78, SOH–Rct = −0,81; trong từng pin r với Rct từ −0,99 đến −0,24, ngoại lệ **B0018** (−0,24). Hàm ý: so trở kháng với mức ban đầu của chính pin; một ngưỡng chung cho mọi pack sẽ đánh giá sai một số pack. **Cần báo** phát hiện B0018 cho Quốc Đạt (liên quan SEM).
7. **Pin khác nhau ngay từ đầu:** C0 của 4 pin G1 từ 1,852 đến 2,037 Ah (chênh 10,0%), 3 / 4 pin dưới 2,0 Ah danh định; B0045 chỉ 1,031 Ah so với 1,501–1,559 Ah của các pin cùng nhóm G7. Hàm ý: chuẩn hóa SOH theo C0 của chính pin; nên sàng lọc cell đầu vào.
8. **Dữ liệu bị kiểm duyệt (censored)** (bổ sung 14/11): chỉ 12 / 23 pin chạm EOL; 3 pin 24°C/4 A xung vuông dừng sau 25–28 chu kỳ và 4 pin 43°C/4 A dừng cùng ở 39 chu kỳ, tức thí nghiệm bị kết thúc theo nhóm, không phải pin hỏng (dữ liệu không ghi lý do). Hàm ý: không lấy trung bình `cycles_to_soh80` làm tuổi thọ chung; nhóm có ít pin chạm EOL không có nghĩa là điều kiện "nhẹ" hơn.

#### 6. Kiểm chứng

- `assert`: 2.065 dòng, 23 pin; mọi cột có thang đo.
- Tái lập các con số của `Introduction_of_dataset.md` mục 8 (SOH mean 0,877, skewness thời gian nghỉ 10,6, 12 / 23 pin chạm EOL).
- Notebook Restart & Run All: 43 ô, không ô nào lỗi, mọi ô code có output. `python src/descriptive_stats.py` chạy với console cp1252 (giả lập Windows), mã thoát 0.

#### 7. Hạn chế và việc còn lại

- Mọi so sánh giữa điều kiện chỉ là mô tả; kiểm định ở GĐ3 (Tiến Hưng lead).
- Tỉ lệ hồi phục ở đây (tier A/B/N) khác nhẹ với notebook Isolation Forest (tier A/B); cả hai đều tính từ dữ liệu.
- Review notebook chi-square `02_inferential.ipynb` của Tiến Hưng (C2.6), theo phân công mới trong review.

---

### 14/11/2026 · Minh Vũ · branch `minh-vu` · GĐ1 bước 1.5: Isolation Forest kiểm tra chéo cờ outlier · WBS C6.1

**Mục đích:** kiểm tra lại cờ `cap_outlier` (Hampel filter hoặc sụt > 25%) bằng một phương pháp ML độc lập. **Không phải thuật toán modeling** (Rec.md mục 2 dòng 12), **không thay đổi dữ liệu hay cờ đã chốt**.
**Code:** `src/iforest_check.py` (6 class), `notebooks/31_iforest.ipynb` (28 ô), 2 hình `reports/figures/iforest_rest_scatter.png`, `reports/figures/iforest_examples.png`. Chạy: `python src/iforest_check.py`.
**Trạng thái:** làm ngày 05/11, chưa commit nên chưa có trong lần review 13/11. Ngày 14/11 dịch sang tiếng Anh; so sánh đầu ra bản tiếng Việt và tiếng Anh, mọi bảng số trùng khớp. **Chưa qua review.**

#### 1. Thiết lập (Go.md 1.5, slide Chap 6 tr. 69–79)

- Dữ liệu: `cycles_clean.csv`, tier A và B, **giữ** dòng `cap_outlier` và `break_in` (vì đang kiểm tra chính các cờ này): **2.020 dòng**, 6 nhóm thí nghiệm. 25 dòng `cap_outlier`, 130 dòng `long_rest`.
- Đặc trưng: `soh`, `d_soh` (SOH trừ SOH chu kỳ hợp lệ liền trước, cùng pin; chu kỳ đầu = 0), `T_max_C`, `V_min`, `rest_before_h`.
- `IsolationForest(n_estimators=100, contamination='auto', random_state=42)`, **fit riêng từng nhóm thí nghiệm** (T_max_C của nhóm 43°C bình thường trong nhóm của nó). Không chuẩn hóa (mỗi nhát cắt chỉ xét một đặc trưng).
- Điểm s(x, n) = 2^(−E[h(x)]/c(n)) = −`score_samples()`. Đã `assert` rằng cờ của sklearn với `contamination='auto'` đúng bằng s > 0,5.

#### 2. Kết quả

Bảng so khớp:

| | iForest: không | iForest: có | Tổng |
|---|---|---|---|
| Hampel: không | 1.786 | 209 | 1.995 |
| Hampel: có | 2 | 23 | 25 |
| Tổng | 1.788 | 232 | 2.020 |

- iForest xác nhận **23 / 25** cờ Hampel (92%), nhưng gắn cờ tổng cộng 232 dòng (11,5%), gấp 9,3 lần Hampel.
- 209 điểm chỉ iForest gắn cờ: **98 (46,9%) ngay sau nghỉ dài**, 55 (26,3%) nằm ở 10% chu kỳ đầu hoặc cuối đời pin, 56 (26,8%) còn lại (chủ yếu bước nhảy ΔSOH vừa phải, chưa vượt ngưỡng Hampel).
- **Chu kỳ sau nghỉ dài (> 1 giờ, 130 dòng):** median ΔSOH +0,0107, 79,2% có SOH tăng; iForest gắn cờ 84,6%, Hampel 9,2%. Khi nghỉ ngắn (1.890 dòng): median ΔSOH −0,0031. Hiện tượng hồi phục xuất hiện ở 17 pin thuộc 5 nhóm.
- **So sánh ở cùng tỉ lệ gắn cờ** (buộc iForest gắn cờ 1,24% như Hampel): iForest 28 cờ, trùng Hampel 12; **16 / 16** điểm chỉ iForest là chu kỳ sau nghỉ dài.
- **2 điểm chỉ Hampel gắn cờ:** B0055 chu kỳ 69 (ΔSOH +0,050) và 90 (ΔSOH −0,049), s = 0,48, sát ngưỡng; lệch khỏi xu hướng của chính pin nhưng không đủ hiếm trong nhóm G9 vốn dao động mạnh.
- **Độ ổn định theo seed** (0–4): số cờ 215–258, Jaccard với seed 42 từ 0,77 đến 0,84; số cờ Hampel bắt được 23–25.

#### 3. Kết luận và quyết định

1. Các cờ `cap_outlier` được một phương pháp độc lập xác nhận.
2. Nhóm điểm lớn nhất iForest gắn cờ thêm là **dung lượng hồi phục sau nghỉ dài**: hiện tượng điện hóa thật (F5), có nguyên nhân, lặp lại ở nhiều pin và có chiều xác định (SOH tăng). **Giữ lại**, dùng cờ `long_rest` (và `log_rest_h` trong SVR) làm biến giải thích. "Bất thường về thống kê" không có nghĩa là "dữ liệu sai".
3. **Không đề xuất thay đổi** quy tắc làm sạch hay các cờ đã chốt.

#### 4. Kiểm chứng

- `assert` cờ sklearn = s > 0,5.
- Đầu ra bản tiếng Việt (05/11) và bản tiếng Anh (14/11) trùng khớp ở mọi bảng (232 cờ, 23 trùng, 98 / 55 / 56, seed 215–258, Jaccard 0,77–0,84, cùng tỉ lệ 28 / 12 / 16).
- Notebook Restart & Run All: 28 ô, không ô nào lỗi. `python src/iforest_check.py` chạy với console cp1252, mã thoát 0.

---

## Khải Hưng

### [21/11/2026] Khải Hưng · branch `khai-hung` · So sánh mô hình, cảnh báo EOL, vẽ lại hình 1–6, rà soát tính nhất quán

**So sánh mô hình (`src/compare_models.py`, `notebooks/60_compare.ipynb`, PR #5).**
- Đọc 6 file trong `reports/predictions/` (naive, regression, arima, arimax, svr, lstm), chấm bằng `protocol.evaluate()`. Kết quả gộp trên 213 dòng: E1 MAE thấp nhất là LSTM và ARIMAX (0,0098), naive 0,0119; E2 không mô hình nào thắng naive (0,0249), gần nhất là LSTM (0,0303).
- Cảnh báo "SOH < 0,8 trong 20 chu kỳ tới": suy từ quỹ đạo E2. Dòng quyết định là mọi chu kỳ đánh giá của B0007, B0034, B0055 trước lần chạm thật; bỏ các dòng mà cửa sổ vượt quá chu kỳ cuối mà chưa thấy chạm (không biết nhãn). Cùng 96 dòng cho mọi mô hình. Naive: sensitivity 0; ARIMA/SVR/LSTM: sensitivity 0,89, specificity 0,00–0,22; regression/ARIMAX: sensitivity 0,44–0,56, specificity 0,92. Thử thêm cửa sổ 10 và 30 chu kỳ: cùng xu hướng.
- Hạn chế đã ghi trong notebook: 96 dòng chỉ từ 3 pin và phụ thuộc nhau; cảnh báo dùng quỹ đạo cố định từ mốc 60%; ARIMA/ARIMAX fit trên 60% đầu của chính pin test, các mô hình khác học từ 8 pin train.

**Hình 1–6 (`src/make_report_figures.py`).** Các hình cũ (23/10) có trước khi làm lại pipeline và không có code trong repo. Đã viết script vẽ lại từ `data/processed/`. Kết luận không đổi; dung lượng theo điều kiện của B0038/B0042 lệch tối đa 0,01 Ah so với bản cũ. Hình 1 bỏ dòng outlier, break-in và pin tier N. Hình 3 dùng tier A/B, bỏ `cap_outlier`, ΔC trong cùng segment, cho đúng bảng F5: n = 1.159 / 696 / 61 / 57, trung vị −0,0073 / −0,0052 / +0,0081 / +0,0294 Ah.

**Rà soát tính nhất quán toàn repo.** Chạy lại pipeline từ 34 file `.mat` trong một bản sao sạch: 8 file CSV giống từng byte. Mọi con số đối chiếu ở Rec.md mục 2.1, 2.2, 6.1 đều đúng. Đã sửa trong tài liệu của Leader:
- ADF (Introduction_of_dataset.md F9, Go.md 5b.3): bản cũ ghi p = 0,53–0,89 cho mọi pin; theo `20_arima` thì B0045 và B0055 dừng (p < 0,002), 10 pin còn lại p = 0,11–0,97.
- Bảng F5 đã tái lập được (xem trên), bỏ cảnh báo "chưa tái tạo được".
- F3 (B0041 chỉ hồi lại 1,02 Ah), F7 (r = 0,86 là khi bỏ outlier; 0,84 trên mọi dòng; thêm lưu ý từ kết quả RQ4 của SVR), F10 (đợt sụt của B0033 là chu kỳ 140–147).
- ARIMA dự báo `soh`, không phải dung lượng; tier B không dùng làm pin test; `eda_dataset.py` chỉ tạo `battery_fade.csv`.
- Go.md: bảng "khoảng dự báo" của các mô hình; hai khuyến nghị ở mục 8 (nhiệt độ cao, ước lượng SOH khi sạc) viết lại theo kết quả kiểm định T1 và RQ4; trạng thái các giai đoạn đã xong.
- README: cây thư mục, bảng file, các bước tái lập.
