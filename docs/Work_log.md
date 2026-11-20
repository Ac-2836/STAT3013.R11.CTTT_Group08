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

