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

## Khải Hưng

### [26/10/2026] Khải Hưng · branch `khai-hung` · GĐ1 Tiền xử lý dữ liệu: làm lại pipeline từ đầu (bước 1–11)

**Bối cảnh.** Các quyết định tiền xử lý đã chốt ngày 25/10/2026 (Rec.md mục 2). Sau đó các script và file CSV cũ bị xóa để làm lại từng bước. Ngày 26/10 làm lại toàn bộ theo 11 bước. Ở mỗi bước: giải thích, chạy thử trên dữ liệu thật, chốt các điểm còn thiếu định nghĩa, sau đó mới viết code. Người chốt mọi quyết định trong mục này: Khải Hưng (Leader). Chưa qua review của nhóm (WBS C1.4).

**Nguồn số liệu.** Mọi con số dưới đây lấy từ 3 script, chạy trên 34 file `.mat` trong `data/raw/` (bản gốc trên Google Drive, link ở `data/raw/README.md`). Môi trường: Python 3.11.9, pandas 3.0.5, numpy 2.4.6, scipy 1.17.1. Các script nằm ở commit `5121e3f` trên branch `khai-hung`. Lệnh chạy lại, từ thư mục gốc repo:
```bash
python src/build_dataset.py && python src/clean_dataset.py && python src/eda_dataset.py
```
Cả 3 script tự kiểm tra bằng `assert`. Nếu một con số đối chiếu bị lệch, script dừng và báo lỗi. Tổng thời gian chạy cả 3 script khoảng 5,5 giây.

**Đầu ra** (trong `data/processed/`, không commit):

| File | Số dòng | Một dòng là |
|---|---|---|
| `discharge_ops.csv` | 2.794 | Một lần xả |
| `charge_ops.csv` | 2.815 | Một lần sạc |
| `impedance_ops.csv` | 1.956 | Một lần EIS (Re, Rct thô) |
| `cycles.csv` | 2.794 | Một lần xả, đã ghép sạc, EIS, metadata; chưa làm sạch (32 cột) |
| `cycles_clean.csv` | 2.658 | Một lần xả hợp lệ (40 cột): bảng dùng chung |
| `cycles_dropped.csv` | 136 | Một lần xả bị loại, kèm `drop_reason` |
| `battery_summary.csv` | 34 | Một pin, kèm tier |
| `battery_fade.csv` | 23 | Một pin tier A/B/N: tốc độ suy giảm |

*Cập nhật ngày 26/10/2026:* theo quyết định của Khải Hưng, các file trong `data/processed/` **có commit** lên repo (commit `75bfca2`, sửa `.gitignore`), để cả nhóm dùng chung một bộ bảng mà không cần chạy lại pipeline. Dữ liệu gốc trong `data/raw/` vẫn không bao giờ commit (bản sao lưu trên Google Drive).

Định nghĩa từng cột: Introduction_of_dataset.md mục 5–6. Các quyết định: Rec.md mục 2, 2.1, 2.2.

---

#### Bước 1. Đọc file `.mat` (`build_dataset.py`; Chap 1 tr. 13, 15–16)

- **Vấn đề.** Dữ liệu là struct MATLAB lồng nhau (semi-structured). Đọc mặc định bằng `scipy.io.loadmat` thì mỗi tầng bị bọc trong mảng 1×1, rất dễ sai chỉ số.
- **Cách làm.** `loadmat(f, simplify_cells=True)[battery_id]["cycle"]` cho ra list, mỗi phần tử là một **thao tác**. Vị trí trong list là `op_index` (đếm từ 0), dùng làm trục thứ tự duy nhất. Datevec chuyển sang datetime bằng `datetime(Y, M, D, h, m) + timedelta(seconds=s)` để giữ phần lẻ của giây.
- **Bằng chứng cần `op_index`.** B0005 có 616 thao tác nhưng chỉ 168 lần xả. Thứ tự thao tác lúc đầu là `c d c d`, sau đó thành `i c i d`, nên không được giả định "thao tác ngay trước lần xả là lần sạc".
- **Kiểm chứng.** 2.815 sạc / 2.794 xả / 1.956 EIS, khớp docs. Thời gian 02/04/2008 → 30/09/2010. 7.282.946 mốc thời gian (khớp "khoảng 7,3 triệu điểm"). Trung vị số mẫu: sạc 2.178, xả 263.
- **Phát hiện.** README gốc ghi sai tên trường của lần xả: file thật dùng `Current_load`/`Voltage_load`, và trường trở kháng tên `Rectified_Impedance`. Có 25 lần xả có `Capacity` rỗng (B0052: 21, B0050: 4). B0053 có 56 lần xả nhưng chỉ 55 lần sạc.
- **Quyết định.** Dữ liệu gốc chuyển từ `data/nasa_pcoe/` sang `data/raw/` cho khớp README. README gốc của dataset đổi tên thành `README_dataset.md` để không đè `data/raw/README.md` của repo. **Không bao giờ commit dữ liệu gốc** (`.gitignore` dòng 5 đã chặn, đã kiểm bằng `git check-ignore`).

#### Bước 2. Gộp mỗi lần xả thành 1 dòng (`build_dataset.py`)

- **Mẫu có tải** là mẫu có `Current_measured < −0,1 A`. Ví dụ B0005, lần xả 1: 178/197 mẫu có tải; các mẫu đầu và cuối chỉ khoảng −0,005 A.
- **Dung lượng `capacity_int_Ah`**: hình thang của dòng tải (đã đổi dấu) theo `Time` thực tế, chia 3.600, tích phân **hết chuỗi**. Bắt buộc dùng `Time` thực vì bước lấy mẫu không đều (0 → 16,8 → 35,7 s). `capacity_to_2p7V_Ah` dừng ở mẫu đầu tiên có V < 2,7.
- **`I_level_A = round(I_load_mean_A)`**. Trên dòng hợp lệ, giá trị thực tế nằm gọn trong 3 cụm: 0,994–1,008 (454 dòng), 1,987–2,013 (1.354 dòng), 3,885–4,030 (850 dòng), cách xa ranh giới làm tròn.
- **5 cột trước đây chưa có định nghĩa, chốt 26/10:** `duration_s` = `Time[-1]`; `t_to_cutoff_s` = `Time` của mẫu có tải cuối cùng (không cần metadata); `energy_Wh` = hình thang của V × dòng tải / 3.600; `T_start_C` = mẫu đầu; `T_max_C`, `T_mean_C` trên toàn chuỗi.
- **Kiểm chứng F1.** `capacity_to_2p7V_Ah` lệch `Capacity` dưới 0,01 Ah ở 2.752/2.794 lần xả = **98,50%** (mẫu số gồm cả 25 dòng `Capacity` rỗng). Trên 2.658 dòng hợp lệ: khớp 100%. Có 41 dòng hợp lệ mà `Capacity` gốc lệch > 50% so với `capacity_int_Ah`, trong đó 40 dòng ở 4°C. Ví dụ B0042, lần xả 42–44 (4°C/4 A): `Capacity` 0,071–0,084 Ah, trong khi `capacity_int_Ah` 1,09–1,16 Ah.
- **Lưu ý code.** 25 lần xả không có mẫu có tải (đều ở G8) cho `I_load_mean_A` = NaN. Code phải chạy qua các dòng này mà không lỗi.

#### Bước 3. Gộp mỗi lần sạc thành 1 dòng (`build_dataset.py`)

- **Nhiễu khởi động.** Ví dụ B0005, thao tác 0: mẫu thứ 2 có I = −4,03 A. Vì vậy `chg_charge_Ah` chỉ tích phân các mẫu có I > 0,1 A.
- **`chg_cc_time_s`** = `Time` của mẫu đầu tiên có `Voltage_measured` ≥ 4,2 V, tức lúc hết pha CC theo đúng quy trình sạc. NaN ở 5 lần sạc không chạm 4,2 V (B0050 thao tác 9; B0025–28 thao tác 79). **Phương án bị loại:** "mẫu cuối cùng có dòng 1,4–1,6 A". Hai cách lệch nhau trung vị 99 s; cách này cho tương quan với SOH thấp hơn (0,838 so với 0,844 trước khi bỏ cờ), và ngưỡng 1,4–1,6 A do tự đặt, không có trong quy trình.
- **Các cột khác:** `chg_duration_s` = `Time[-1]`; `chg_end_time` = giờ bắt đầu + thời lượng; `chg_T_max_C` = max nhiệt độ.
- **Kiểm chứng.** `rest_before_h` = (giờ bắt đầu xả − `chg_end_time`) / 3.600. Trên 2.658 dòng hợp lệ: 167 dòng > 1 giờ (khớp `long_rest` đã chốt), 0 giá trị âm, ngắn nhất 63,7 s. F7: tương quan `chg_cc_time_s`–SOH ở G1 = **0,859** (sau khi bỏ `cap_outlier`, `break_in`), khớp con số 0,86; trước khi bỏ cờ là 0,844.
- **Đính chính trong quá trình làm.** Lần đầu tôi gán 16 dòng "sạc nhỏ" cho G1/G4/G5/G2 bằng cách lọc lần sạc < 0,5 Ah. **Cách đó sai.** Đếm trực tiếp cho thấy 16 dòng có 2 lần sạc kể từ lần xả trước đều thuộc G1 và G2, và chỉ 6 trong số đó là sạc bù thật (0,007–0,139 Ah). Các lần sạc nhỏ ở G6 (143 dòng), B0033 lần 141–148, B0039 lần 13 và B0040 lần 47 là hiện tượng thật, vì lần xả trước đó cũng nhỏ.

#### Bước 4. Gộp mỗi lần EIS thành 1 dòng (`build_dataset.py`)

- **Chỉ dùng `Re`, `Rct`.** File không lưu vector tần số, nên 48 điểm `Battery_impedance` và 39 điểm `Rectified_Impedance` không dùng được.
- **Giá trị ngoài (0; 0,5) Ω:** 14 lần đo, **đều ở G8** (B0052: 10, B0050: 4), tới ±10¹⁵ Ω. Mỗi lần đo sai thì sai cả Re lẫn Rct.
- **Số phức:** 9 lần đo (B0049, B0051), Re và Rct là liên hợp của nhau, tức phép fit hỏng. **Quyết định:** phần ảo khác 0 thì đặt NaN. `impedance_ops.csv` giữ giá trị thô; việc lọc làm ở `clean_dataset.py`.
- **Nhiệt độ ghi ở EIS** luôn là 24°C (riêng G3 là 43°C), kể cả ở nhóm chạy 4°C. Vì vậy nhiệt độ chỉ lấy từ thao tác xả.
- **Phát hiện ảnh hưởng tới hồi quy.** Docs ghi "Re và Rct có r = 0,97". Thực tế chỉ đúng trong G1 (0,968). Các nhóm khác: G2 0,995, G3 0,996, G4 0,983, G5 0,915, G6 0,242, G7 0,250, G9 0,578; **tier A 0,43**; toàn bộ dòng hợp lệ 0,38. Đã sửa ở Go.md 5a.1 và 10.2, WBS C3.1, Introduction 9.2.
- **Kiểm chứng.** 57 dòng hợp lệ không có Re/Rct, khớp docs. G1: SOH–Re −0,775, SOH–Rct −0,810, khớp −0,78 và −0,81.

#### Bước 5. Ghép xả với sạc, EIS, metadata → `cycles.csv` (`build_dataset.py`; Chap 1 tr. 16)

- **Cách ghép.** `pd.merge_asof(by="battery_id", direction="backward", allow_exact_matches=False)` theo `op_index`. **Lỗi đã gặp:** `merge_asof` yêu cầu khóa sắp xếp trên **toàn bảng**. Sắp theo `["battery_id", "op_index"]` sẽ báo `left keys must be sorted`; phải sắp chỉ theo `op_index`.
- **Kết quả.** 23 dòng không có sạc trước (4 dòng ở G8), 76 dòng không có EIS trước.
- **Metadata: chỉ lấy 3 cột.**
  - `group`.
  - `discharge_cutoff_V`: ở cả 30 pin ngoài G8, trung vị `V_min` đều thấp hơn cutoff 0,005–0,27 V, tức cutoff đúng.
  - `profile`: suy từ `discharge_protocol`. Đếm số lần tải bật trong mỗi lần xả: chỉ G2 có trung vị 157,5, các nhóm khác đều 1, khớp metadata.
  - **Không lấy** `ambient_temperature_C` và mức dòng trong metadata, vì sai ở G6 và B0028.
- **Cột mới `n_chg_since_prev_dis`** (chốt 26/10). Trên dòng hợp lệ: 0 → 3 dòng, 1 → 2.639, 2 → 16.
- **`assert` trong code:** `chg_op_index < op_index`; `eis_op_index < op_index`; không có thời gian nghỉ âm; đúng 23 dòng thiếu sạc và 76 dòng thiếu EIS. Đây chính là việc kiểm tra mà Go.md 1.6 giao cho Khải Hưng.
- **Lỗi khi chạy lần đầu.** `print` đường dẫn tuyệt đối có chữ "Khải Hưng" làm console cp1252 báo `UnicodeEncodeError`. Đã sửa bằng cách in đường dẫn tương đối. Thành viên nào có đường dẫn chứa ký tự tiếng Việt cũng sẽ gặp lỗi này.

#### Bước 6. Loại dòng (`clean_dataset.py`; Chap 1 tr. 17, Chap 6 tr. 69)

- **Thứ tự cố định** bằng `np.select`: G8 → không có sạc trước → `V_min > cutoff + 0,1` → `capacity_int_Ah` ngoài [0,05; 2,1]. Kết quả **100 / 19 / 16 / 1**, còn **2.658** dòng.
- **Vì sao thứ tự quan trọng.** Đếm độc lập ra 100 / 23 / 19 / 32. Phần trùng: G8 với không có sạc 4; G8 với không chạm cutoff 3; G8 với dung lượng ngoài khoảng 29; không chạm cutoff với dung lượng ngoài khoảng 2. Tổng dòng bị loại luôn là 136, nhưng số dòng gán cho từng lý do thay đổi theo thứ tự.
- **Chi tiết các dòng bị loại:**
  - 19 dòng không có sạc trước: đều là lần xả đầu tiên của 19 pin.
  - 16 dòng không chạm cutoff do 2 nguyên nhân. (a) Sự cố thiết bị: nhiều pin bị cắt cùng một mốc thời gian (B0042–44 lần 6 lúc 04/06/2010 17:26; G7 lần 20 lúc 29/07/2010 02:14, lần 54 lúc 07/08/2010 11:26, lần 66 lúc 13/08/2010 11:16). (b) Lần xả cuối của B0053 và B0054 (30/09/2010 15:32, dài 23 s, 0,004 Ah): thí nghiệm kết thúc giữa chừng.
  - 1 dòng dung lượng ngoài khoảng: B0036 lần 114, 2,444 Ah, kéo dài 4.424 s, trong khi các lần lân cận khoảng 1,67–1,69 Ah và 3.024–3.067 s.
- **Khoảng trống tự nhiên của các ngưỡng:**
  - `V_min − cutoff`: lớn nhất ở phía giữ là 0,000 V, nhỏ nhất ở phía loại là 0,378 V.
  - Dung lượng: lớn nhất được giữ 2,047 Ah, dòng bị loại 2,444 Ah; nhỏ nhất được giữ 0,0543 Ah (G6, F3), dòng thấp hơn là 0,004 Ah (đã bị loại ở quy tắc cutoff).
  - Cận dưới 0,05 Ah chỉ có tác dụng ở G8.
- **Quyết định (26/10):** ghi `cycles_dropped.csv` kèm `drop_reason` để Tiến Hưng review (Go.md 1.6).

#### Bước 7. Lọc trở kháng, chia segment, đánh số `cycle_valid` (`clean_dataset.py`)

- **Lọc trở kháng.** Giá trị ngoài (0; 0,5) Ω đặt NaN, sau đó `groupby("battery_id").ffill()`; tuyệt đối không dùng `bfill`. Trên dòng hợp lệ, bộ lọc **không thay đổi giá trị nào**. Vẫn còn 57 ô NaN (B0005, B0006, B0007, mỗi pin 19 dòng, trước lần EIS đầu tiên ở thao tác 40).
- **Segment.** Khóa (`ambient_C`, `I_level_A`), đánh số bằng `(khóa != khóa.shift()).cumsum()` trong từng pin. Kết quả 46 segment:
  - B0038–40: 4 segment (11 / 27 / 6 / 2 dòng).
  - B0041: 2 segment (41 / 25).
  - B0042–44: 3 segment (40 / 46 / 25).
  - 23 pin còn lại: 1 segment.
  - Khớp các pha ở Introduction mục 2. Không pin nào quay lại điều kiện cũ.
- **`cycle_valid`.** `groupby("battery_id").cumcount() + 1`, đếm theo pin, không đếm lại từ đầu mỗi segment. Lớn nhất 196 (B0033, B0034). Docs ghi "tối đa 197 điểm", là số lần xả gốc; đã sửa ở Introduction 1.3.

#### Bước 8. C₀ và SOH (`clean_dataset.py`; Chap 1 tr. 41)

- **C₀** theo từng (pin, segment): `r = s.head(10).rolling(3, min_periods=1).median()`, rồi `r.iloc[2:].max()` (nếu segment ít hơn 3 dòng thì lấy `r.max()`). Bỏ 2 vị trí đầu vì chúng không phải trung vị thật của 3 điểm.
- **Ví dụ B0033.** 10 giá trị đầu: 0,716 → 1,194 → 1,301 → 1,351 → 1,381 → 1,344 → 1,842 → 1,797 → 1,773 → 1,759. Trung vị trượt lớn nhất là 1,797, tức đỉnh đơn lẻ 1,842 bị loại.
- **So sánh các định nghĩa C₀ khác** (B0033 / B0054):
  - Lần đầu: 0,716 / 1,290.
  - Trung bình 3 lần đầu: 1,070 / 1,338.
  - Trung vị 3 lần đầu: 1,194 / 1,290.
  - Max 10 lần: 1,842 / 1,508.
  - **Định nghĩa đã chốt: 1,797 / 1,483.**
  - Với pin bình thường (B0005, B0036), mọi cách gần như trùng nhau.
- **Kết quả.**
  - C₀: B0005 **1,852**, B0033 **1,797**, B0054 **1,483** (khớp số đã chốt).
  - G1: 1,852–2,037, trung bình 1,913 (khớp F8). B0045: 1,031.
  - `soh` có giá trị ở 2.121 dòng (chỉ pin 1 segment), khoảng 0,124–1,104 trước khi bỏ cờ. Có 36 dòng > 1 (hồi phục sau nghỉ hoặc chu kỳ đầu); không cắt về 1.
  - Ở segment 4°C/4 A của G6, C₀ chỉ 0,074–0,087 Ah, nên `soh_segment` ở đó chỉ mang nghĩa tương đối.

#### Bước 9. Gắn cờ (`clean_dataset.py`; Chap 6 tr. 69)

- **`cap_outlier`.** Hampel (cửa sổ 7 căn giữa, `min_periods` 3, MAD × 1,4826, ngưỡng `max(3·MAD; 0,05 Ah)`) **hoặc** `x < 0,75 × max(10 dòng trước)`; tính trong từng (pin, segment). Kết quả **57** dòng: chỉ Hampel 30, chỉ quy tắc sụt 24, cả hai 3. Theo pin: B0054 18, B0033 13 (lần 2, 46, 114, 139–147, 156), B0042 13, B0026 3, B0006/B0034/B0055 mỗi pin 2, B0005/B0007/B0036/B0041 mỗi pin 1.
- **Độ nhạy:**

  | Thay đổi tham số | Số dòng `cap_outlier` |
  |---|---|
  | Cửa sổ 5 / 9 | 56 / 57 |
  | k = 2,5 / 3,5 | 62 / 53 |
  | Mức sàn 0,03 / 0,08 Ah | 73 / 52 |
  | Ngưỡng sụt 20% / 30% | 86 / 51 |

  Câu "ổn định" trong Rec.md mục 2 đã được bổ sung bảng này.
- **Điểm cần kiểm tra chéo.** 14/57 dòng `cap_outlier` là dòng `long_rest`; có thể là hiện tượng hồi phục thật (F5). Giao cho bước kiểm tra bằng Isolation Forest (Go.md 1.5, Minh Vũ).
- **`break_in`.** Chốt 26/10: dùng `cycle_valid` ≤ 10 (theo pin) **và** dung lượng < 0,9 × C₀. Kết quả **17** dòng: B0033 lần 1–6; B0054 lần 1, 2, 4, 6, 8, 9, 10; B0045 lần 9, 10; B0026 lần 6; B0041 lần 9. **Phương án bị loại:** 10 dòng đầu mỗi segment cho 36 dòng, gắn cờ nhầm các hiệu ứng đổi điều kiện. 4 dòng vừa là `break_in` vừa là `cap_outlier`.
- **`long_rest`.** `rest_before_h > 1`: **167** dòng.

#### Bước 10. Tier, `cycles_clean.csv`, `battery_summary.csv` (`clean_dataset.py`)

- **Các con số dùng để xếp tier**, tính trên dòng hợp lệ đã bỏ `break_in`: `n_valid` (có tính outlier), `outlier_share`, `soh_min` (bỏ thêm outlier). Thứ tự xét **X → C → N → A → B**.
- **Thứ tự này ảnh hưởng tới 2 pin.** B0054 (18,1% outlier) sẽ thành A nếu xét A trước N. B0042 (11,7% outlier) sẽ thành N nếu xét N trước C.
- **Kết quả.** Tier A 12 / B 10 / C 7 / N 1 / X 4 pin; số dòng 1.503 / 517 / 537 / 101. Khớp số đã chốt.
- **Khoảng trống tự nhiên của các ngưỡng tier:**
  - Tỉ lệ outlier: 7,4% (B0026) → 18,1% (B0054).
  - `n_valid`: 54 (B0053) → 67 (B0045).
  - `soh_min`: 0,818 (B0056) → 0,864 (B0036).
- **Lưu ý cho giao thức đánh giá (GĐ4).** B0034 (pin test) có `soh_min` 0,799, tức chỉ vừa chạm dưới 0,8, nên kịch bản E2 nhạy với pin này. B0056 (pin train) chưa bao giờ chạm 0,8 (0,818).
- **Quyết định (26/10):** thêm cột `tier` vào `cycles_clean.csv`; `battery_summary.csv` gồm cả 34 pin, các cột đếm dùng kiểu số nguyên cho phép trống (`Int64`).

#### Bước 11. `battery_fade.csv` (`eda_dataset.py`; Chap 2 tr. 75)

- **Phạm vi.** 23 pin tier A, B, N; bỏ `cap_outlier` và `break_in`. `fade` = −slope × 10⁴ của `np.polyfit(cycle_valid, soh, 1)`; `fade25` tính trên `cycle_valid` ≤ 25.
- **Đối chiếu Introduction mục 8** (trung bình theo điều kiện, toàn chuỗi / 25 chu kỳ đầu):
  - 24°C CC 2 A: 17,9 / 14,4.
  - 24°C CC 4 A: 8,3 / 20,1.
  - Xung vuông 4 A / 2 A: 10,9 / 14,6 và 9,7 / 13,3.
  - 43°C CC 4 A: 28,4 / **27,8**.
  - 4°C CC 1 A: 29,9 / 64,8.
  - 4°C CC 2 A (không gồm B0054): 10,0 / 27,9.
  - Docs ghi 27,9 cho ô 43°C; đã sửa thành 27,8 (giá trị chưa làm tròn là 27,85) ở Introduction mục 8 và Go.md 2.3.
- **`T_max_med`:** G7 10,8–16,2°C; G9 (không gồm B0054) 14,5–20,9°C. Khớp docs.
- **`cycles_to_soh80`:** NaN ở toàn bộ 10 pin tier B và B0056. Hệ quả: phần khuyến nghị "B10 life" không có số liệu cho 43°C và xung vuông.
- **Ghi chú cho GĐ3.** `fade25` dao động mạnh ngay trong G1 (4,2 / 25,5 / 5,2 / 32,8). Nhóm "xung vuông" trong ANOVA (Go.md 3.3) đang gộp B0028 (biên độ 2 A) với B0025–27 (4 A): Tiến Hưng cần quyết định cách xếp nhóm cho B0028.

---

**Việc còn mở sau mục này:**
1. Review của nhóm theo Go.md 1.6 (WBS C1.4).
2. Bảng F5 (thay đổi dung lượng theo thời gian nghỉ) chưa tái tạo được (thử ra n = 1.167 / 701 / 64 / 66, trung vị nhóm > 1 ngày +0,0374 so với docs +0,0294). Đã ghi cảnh báo ở Introduction F5; kiểm tra lại khi viết phần EDA.
3. `eda_dataset.py` mới có phần `battery_fade.csv`; còn thống kê mô tả, ADF và các hình.
4. Push branch `khai-hung` và mở Pull Request vào `main` sau khi nhóm review (chưa làm).
