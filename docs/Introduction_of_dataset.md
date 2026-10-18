# Giới thiệu & Tóm tắt Dữ liệu Nghiên cứu (Dataset Introduction)

> **Đề tài:** Statistical and Machine Learning-Based Modeling of Lithium-Ion Battery Degradation and State-Of-Health Prognostics under Variable Operating Conditions  
> **Môn học:** Phân tích Thống kê (Statistical Analytics)  
> **Nhóm thực hiện:** 4AnhTai

---

## 1. Tổng quan hai bộ dữ liệu

Thư mục `data/` của dự án bao gồm hai bộ dữ liệu thực nghiệm tiêu chuẩn trong lĩnh vực Dự đoán & Quản lý Tình trạng Pin (Battery Prognostics and Health Management - PHM). 

Điểm nổi bật nhất của cả hai bộ dữ liệu này là **tập trung vào quá trình suy thoái của pin Lithium-ion dưới điều kiện vận hành biến thiên ngẫu nhiên (Variable / Dynamic Operating Conditions)** thay vì chỉ thử nghiệm sạc/xả ở dòng tĩnh (Constant Current - CC) như các tập dữ liệu truyền thống. Đây là tiền đề thực nghiệm lý tưởng cho đề tài nghiên cứu.

| Tiêu chí | Dataset 1: NASA Random Walk Battery Dataset | Dataset 2: Accelerated Life Testing (ALT) Dataset |
| :--- | :--- | :--- |
| **Thư mục lưu trữ** | `data/Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post/` | `data/battery_alt_dataset/` |
| **Nguồn gốc / Tác giả** | NASA Ames Research Center (Brian Bole, Chetan Kulkarni, Matthew Daigle - 2014) | PML-UCF Research Group & NASA ARC (Kajetan Fricke, Renato Nascimento, Felipe Viana - 2023) |
| **Đối tượng thử nghiệm** | 4 cell pin đơn Li-ion định dạng 18650 (`RW9`, `RW10`, `RW11`, `RW12`) | Hơn 25 khối pin ghép (battery packs 2S, điện áp sạc đầy ~8.3V – 8.4V) |
| **Định dạng dữ liệu** | `.mat` (MATLAB structure) và `.Rda` (R DataFrame) | File chuẩn `.csv` (thuận tiện cho Python/Pandas) |
| **Chế độ tải trọng** | Phân bố đều ngẫu nhiên (Uniform Random Walk) với 12 mức dòng sạc/xả (-4.5A đến +4.5A) | 3 nhóm: Tải không đổi, Tải biến thiên theo đoạn, Đổi nhiệm vụ (Recommissioned), Pin vòng đời 2 (Second-life) |
| **Chu trình đo chuẩn SOH** | Reference cycles thực hiện định kỳ sau mỗi 1500 bước RW (~5 ngày) | Reference discharge định kỳ (`mission_type = 0`, dòng cố định 2.5A) |

---

## 2. Chi tiết từng bộ dữ liệu

### 2.1. Dataset 1: NASA Room Temperature Random Walk Dataset
* **Đường dẫn thư mục:** `data/Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post/`
* **Ấn phẩm tham chiếu:** 
  > *Brian Bole, Chetan Kulkarni, and Matthew Daigle, "Adaptation of an Electrochemistry-based Li-Ion Battery Model to Account for Deterioration Observed Under Randomized Use", Annual Conference of the Prognostics and Health Management Society, 2014.*

#### Quy trình thí nghiệm:
1. **Chế độ vận hành ngẫu nhiên (Random Walk - RW):**
   * Các cell pin 18650 hoạt động liên tục ở nhiệt độ phòng.
   * Dòng điện áp đặt được chọn ngẫu nhiên từ tập 12 giá trị phân bố đều:
     $$\mathcal{I} = \{-4.5\text{A}, -3.75\text{A}, -3.0\text{A}, -2.25\text{A}, -1.5\text{A}, -0.75\text{A}, +0.75\text{A}, +1.5\text{A}, +2.25\text{A}, +3.0\text{A}, +3.75\text{A}, +4.5\text{A}\}$$
     *(Trong đó: giá trị âm biểu thị dòng nạp sạc, giá trị dương biểu thị dòng xả tải).*
   * Mỗi mức dòng duy trì tối đa 5 phút hoặc tự động ngắt khi điện áp chạm ngưỡng giới hạn an toàn ($3.2\text{V} - 4.2\text{V}$). Sau mỗi bước có khoảng nghỉ ngắn $< 1\text{s}$ trước khi đổi sang mức dòng kế tiếp.

2. **Chu trình kiểm chuẩn định kỳ (Reference Cycles):**
   Cứ sau mỗi 1500 bước RW (~5 ngày vận hành thực tế), chuỗi kiểm tra chuẩn được thực hiện nhằm định chuẩn tình trạng sức khỏe pin:
   * **Low current discharge (0.04A):** Xả dòng rất nhỏ để quan sát đường đặc tuyến điện áp hở mạch OCV (Open-Circuit Voltage) theo trạng thái tích điện SOC.
   * **Reference charge & discharge (CC-CV sạc 2A đến 4.2V, xả 2A về 3.2V):** Dùng để tính toán dung lượng xả thực tế $C_{\text{measured}}$ qua tích phân dòng điện (Coulomb counting) làm nhãn chuẩn cho SOH:
     $$C_{\text{measured}} = \int_{0}^{t_{\text{end}}} I(t) \, dt$$
   * **Pulsed current charge/discharge:** Xung dòng 1A trong 10 phút, nghỉ 20 phút (thực hiện sau mỗi 3000 bước RW) để đo đáp ứng động học quá độ (transient dynamics) và sự gia tăng nội trở của pin.

#### Cấu trúc và trường dữ liệu:
* Dữ liệu được lưu trữ trong 2 thư mục: `data/Matlab/` (`.mat`) và `data/R/` (`.Rda`) cho 4 viên pin: `RW9`, `RW10`, `RW11`, `RW12`.
* Mỗi file chứa cấu trúc `data` gồm:
  * `procedure`: Tên quy trình thí nghiệm.
  * `description`: Mô tả chi tiết thí nghiệm.
  * `step`: Mảng các bước hoạt động (dataframe/struct), mỗi bước bao gồm:
    * `comment`: Mô tả bước (ví dụ: `"reference discharge"`, `"discharge (random walk)"`, `"low current discharge at 0.04A"`).
    * `type`: Ký hiệu loại bước (`'C'` = Sạc, `'D'` = Xả, `'R'` = Nghỉ).
    * `relativeTime`: Vector thời gian tương đối (tính bằng giây từ đầu bước).
    * `time`: Vector thời gian tích lũy từ khi bắt đầu thí nghiệm.
    * `voltage`: Vector điện áp pin (V).
    * `current`: Vector dòng điện (A).
    * `temperature`: Vector nhiệt độ pin (°C).
    * `date`: Thời điểm bắt đầu bước (`dd-Mon-yyyy HH:MM:SS`).

---

### 2.2. Dataset 2: Accelerated Life Testing (ALT) Dataset
* **Đường dẫn thư mục:** `data/battery_alt_dataset/`
* **Ấn phẩm tham chiếu:**
  > *Kajetan Fricke, Renato G. Nascimento, Matthew Corbetta, Chetan S. Kulkarni, Felipe A. C. Viana, "An accelerated Life Testing Dataset for Lithium-Ion Batteries with Constant and Variable Loading Conditions", NASA / PML-UCF, 2023.*

#### Phân loại các nhóm pin trong thí nghiệm:
Bộ dữ liệu gồm 3 thư mục đại diện cho các kịch bản vòng đời đa dạng:

1. **`regular_alt_batteries/` (16 khối pin):**
   * *Nhóm Constant Current (Tải tĩnh):* Chịu tải không đổi trong suốt vòng đời ở các mức dòng xả khác nhau: $9.3\text{A}$ (pin 0.1, 1.1), $12.9\text{A}$ (pin 3.1, 2.2), $14.3\text{A}$ (pin 2.3, 5.2), $16.0\text{A}$ (pin 0.0, 1.0), $19.0\text{A}$ (pin 2.0, 3.0, 2.1).
   * *Nhóm Variable Current (Tải biến thiên):* Dòng xả thay đổi liên tục giữa các phân đoạn thời gian ngẫu nhiên (40s, 60s hoặc 80s) với biên độ ngẫu nhiên: dải trung bình $14.3\text{A}$ ($12.9\text{A} - 16\text{A}$) và dải trung bình $17.0\text{A}$ ($16\text{A} - 19\text{A}$).
2. **`recommissioned_batteries/` (8 khối pin):**
   * Mô phỏng pin đổi nhiệm vụ / thay đổi mức tải ở các giai đoạn vòng đời khác nhau (multi-stage loading):
     * *2 giai đoạn:* Chạy ở mức dòng 1 cho đến khi dung lượng giảm về khoảng $2.2\text{Ah}$, sau đó đổi sang mức dòng 2 cho đến cuối vòng đời (EOL - End of Life).
     * *3 giai đoạn:* Đổi mức tải tại các mốc dung lượng $2.2\text{Ah}$ và $2.0\text{Ah}$.
3. **`second_life_batteries/` (3 khối pin):**
   * Mô phỏng pin tái sử dụng (Second-life): Các cell còn hoạt động tốt sau khi pack vòng đời 1 hư hỏng được gom lại và đóng gói thành pack mới, tiếp tục chạy dưới mức tải khác cho đến EOL.

#### Cấu trúc trường dữ liệu (.csv):
Mỗi khối pin được ghi nhận trong một file `.csv` độc lập với tần số lấy mẫu cao:
* **Thông tin vận hành xuyên suốt:**
  * `start_time`: Thời điểm bắt đầu chu kỳ ngày (`yyyy-mm-dd hh:mm:ss`).
  * `time`: Thời gian tích lũy liên tục từ đầu vòng đời đến khi hỏng hoàn toàn (giây).
  * `mode`: Chế độ hoạt động (`-1` = Xả, `0` = Nghỉ, `1` = Sạc).
  * `voltage_charger`: Điện áp đo tại cổng sạc (V).
  * `temperature_battery`: Nhiệt độ đo trực tiếp trên bề mặt điện cực của cell pin (°C).
* **Thông tin khi kết nối tải xả (Load board):**
  * `voltage_load`: Điện áp khối pin đo tại bo mạch tải (V).
  * `current_load`: Dòng xả đo qua điện trở shunt cảm biến dòng (A).
  * `temperature_mosfet`: Nhiệt độ linh kiện MOSFET trên bo mạch tải (°C).
  * `temperature_resistor`: Nhiệt độ điện trở cảm biến (°C).
  * `mission_type`: **`0` = Reference discharge** (xả chuẩn dòng $2.5\text{A}$ để đo dung lượng chuẩn), **`1` = Regular mission** (chu trình vận hành làm việc thông thường).

---

## 3. Định hướng nghiên cứu cho Đề tài môn Phân tích Thống kê

Nhờ vào tính chất tải biến thiên ngẫu nhiên và việc có các mốc đo chuẩn định kỳ, hai bộ dữ liệu này đáp ứng trọn vẹn các yêu cầu học thuật của đồ án:

### 3.1. Thống kê mô tả (Descriptive Analytics) & Khám phá dữ liệu (EDA)
* **Phân tích phân phối:** Khảo sát hàm mật độ xác suất (KDE), trung bình, phương sai, độ lệch (skewness) và độ nhọn (kurtosis) của dòng tải, điện áp và nhiệt độ hoạt động.
* **Đặc tính suy thoái (Degradation Trajectory):** Xây dựng biểu đồ đường cong suy giảm dung lượng thực nghiệm theo thời gian và số chu kỳ.
* **So sánh thống kê (Hypothesis Testing):** Kiểm định sự khác biệt có ý nghĩa thống kê về tốc độ suy thoái giữa:
  * Chế độ tải cố định (Constant) vs Tải biến thiên ngẫu nhiên (Variable/Random Walk).
  * Các mức C-rate cao ($19\text{A}$) vs C-rate trung bình ($9.3\text{A}$).
  * Pin nguyên bản vs Pin tái vận hành (Recommissioned) / Pin vòng đời 2 (Second-life).

### 3.2. Rút trích đặc trưng y tế (Health Indicator Extraction)
* Tính toán dung lượng thực tế $C_k$ tại từng chu kỳ chuẩn để thiết lập Ground-truth $\text{SOH}_k$:
  $$\text{SOH}_k = \frac{C_k}{C_{\text{nominal}}} \times 100\%$$
* Rút trích các đặc trưng gián tiếp từ đường cong sạc/xả thông thường:
  * Thời gian sạc trong dải điện áp cố định (ví dụ $3.8\text{V} \to 4.1\text{V}$).
  * Tốc độ biến thiên điện áp theo thời gian ($dV/dt$).
  * Nội trở thuần ước lượng qua độ sụt áp tức thời tại thời điểm ngắt dòng: $R_{\text{int}} \approx \frac{\Delta V}{\Delta I}$.
  * Nhiệt độ cực đại và diện tích tích lũy nhiệt độ trong mỗi chu kỳ.

### 3.3. Mô hình hóa Thống kê & Học máy (Statistical & Machine Learning Modeling)
* **Mô hình Thống kê / Chuỗi thời gian:**
  * **Gaussian Process Regression (GPR):** Dự báo suy thoái dung lượng kèm khoảng tin cậy thống kê (95% Confidence Interval) để định lượng độ bất định (uncertainty quantification).
  * **ARIMA / State-space Models:** Dự báo xu hướng chuỗi thời gian suy thoái dung lượng.
* **Mô hình Học máy (Machine Learning):**
  * Random Forest, XGBoost, LightGBM: Hồi quy ước lượng SOH dựa trên tập đặc trưng gián tiếp (phục vụ trường hợp thực tế không thể thực hiện chu trình đo chuẩn).
* **Mô hình Học sâu (Deep Learning) cho bài toán RUL:**
  * LSTM / GRU / Transformer: Dự báo Remaining Useful Life (RUL) — số chu kỳ còn lại trước khi pin chạm ngưỡng suy thoái quy ước (thường là $80\%$ hoặc $70\%$ SOH).
