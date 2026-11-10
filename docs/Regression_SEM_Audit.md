# Audit Regression và SEM — 09–10/11/2026

Thành viên: Quốc Đạt · branch `quoc-dat`. Đây là tài liệu review nội bộ. Nội dung chèn báo cáo tiếng Anh:
[Regression_Report.md](Regression_Report.md), [SEM_Report.md](SEM_Report.md).

## Phạm vi và bảo toàn

Đã kiểm tra `git status`, branch và 5 commit trước khi sửa. HEAD lúc bắt đầu: `73452fd`.
Thay đổi có sẵn: `notebooks/50_sem.ipynb`, `reports/predictions/regression.csv`, notebook chưa track `notebooks/10_regression.test.ipynb`.
Đã sao lưu nguyên trạng cả ba, notebook Regression và các hình cũ vào `reports/audit/regression_sem_before/`;
patch ban đầu ở `preexisting.patch`. Bản Git HEAD của SEM và dự báo được lưu riêng để phân biệt với working copy.
Không sửa notebook thử nghiệm; không commit/push/merge/rebase/PR. Không sửa dữ liệu gốc, processed tables,
pipeline của Leader, giao thức trong Rec/Go/WBS, hay mô hình thành viên khác.

## Lỗi logic/code và phần đã sửa

| Phát hiện | Cách xử lý và bằng chứng |
| --- | --- |
| Simple regression dùng toàn Tier A thay vì G1 | Dùng G1 train, loại pin test B0007; đủ F/t/SE/CI và chẩn đoán |
| Correlation thiếu Re/Rct và thời gian sạc | Xuất ma trận train và G1 train; VIF trên đúng design matrix |
| Centering/selection phần suy diễn dùng cả pin test | C3 fit trên train; full-sample EDA giữ vai trò mô tả; battery-level 23-cell là sensitivity suy diễn riêng |
| Backward selection bỏ cả nhóm main+interaction | Kiểm định từng term đủ điều kiện, giữ hierarchy, xuất tất cả bước, thêm OLS sensitivity |
| Chưa có GroupKFold chọn biến thể | K=4 theo pin; mọi centering, selection, smearing refit từng fold; freeze winner riêng E1/E2 |
| Forecast exp(log-SOH) thiếu retransformation correction | Train-only Duan cho direct model; hệ số chung triệt tiêu trong anchored ratio |
| Global curve lệch mức giữa pin | Thử anchored log-linear và quadratic bằng lịch sử được phép; giữ research baseline và mọi candidate |
| RUL B0047 tính sau khi pin đã EOL | First SOH≤0.8 tại chu kỳ 30, origin 40; đánh dấu pre-origin, không tính RUL/error |
| Pin không dự báo crossing dễ bị loại khỏi thống kê | Giữ đủ bốn pin, ghi rõ trạng thái censored/no crossing, không extrapolate |
| SEM root phụ thuộc working directory | Tìm repository theo cây thư mục; đã kiểm tra từ root và notebooks |
| EIS index mới có thể mang Re/Rct ffill | Join lại raw operation, kiểm tra (0,0.5) ohm và timestamp; dữ liệu thật không có mismatch |
| SEM bootstrap chỉ kiểm tra exception | Kiểm tra solver success, rank, giá trị hữu hạn, variance/covariance hợp lệ; tiếp tục đến 200 fit hợp lệ |
| SEM df không đếm moment ngoại sinh cố định theo mẫu | Khai báo tự do 10 nuisance moments; primary df=3, không dùng df=13 để kết luận fit |
| Raw-unit optimizer có thể báo success ở nghiệm kém | Internal scaling; đối chiếu scaled original và unit-conversion test; giữ raw reexecution như sensitivity |
| Path diagram thiếu hệ số/biến thực | Vẽ đúng predictor, hệ số standardized lấy từ model; PNG 320 dpi + PDF |

## Kết quả Regression

| model | scenario | MAE | RMSE | MAPE_pct | R2 |
| --- | --- | --- | --- | --- | --- |
| persistence | E1 | 0.0141225 | 0.0201418 | 1.69765 | 0.0447127 |
| persistence | E2 | 0.0238145 | 0.0299687 | 2.9871 | -0.871726 |
| regression | E1 | 0.0137394 | 0.0202341 | 1.64797 | 0.0446581 |
| regression | E2 | 0.0335355 | 0.0420071 | 4.17526 | -2.75718 |
| research_loglinear | E1 | 0.0839132 | 0.0891713 | 10.5348 | -21.5502 |
| research_loglinear | E2 | 0.0839132 | 0.0891713 | 10.5348 | -21.5502 |

E1 chọn `anchored_loglinear`, E2 chọn `anchored_quadratic` bằng CV train.
Không đổi sang candidate khác sau khi nhìn test. E1 chỉ cải thiện nhỏ MAE so với persistence;
RMSE không thắng. E2 vẫn kém persistence. CSV chung 426 dòng (213 mỗi scenario), đúng schema 8 cột, interval trống.
Chi tiết từng pin, micro/macro và toàn bộ candidate nằm trong `reports/tables/`.

| model | battery_id | origin_cycle | observed_EOL_cycle | predicted_EOL_cycle | status | absolute_RUL_error |
| --- | --- | --- | --- | --- | --- | --- |
| persistence | B0007 | 101 | 126 | — | prediction_no_crossing_in_horizon | — |
| persistence | B0034 | 118 | 186 | — | prediction_no_crossing_in_horizon | — |
| persistence | B0047 | 40 | 30 | 41 | already_at_or_before_origin | — |
| persistence | B0055 | 59 | 65 | — | prediction_no_crossing_in_horizon | — |
| regression | B0007 | 101 | 126 | 112 | both_cross_within_horizon | 14 |
| regression | B0034 | 118 | 186 | — | prediction_no_crossing_in_horizon | — |
| regression | B0047 | 40 | 30 | 41 | already_at_or_before_origin | — |
| regression | B0055 | 59 | 65 | 91 | both_cross_within_horizon | 26 |

## Kết quả SEM

Kết quả gốc lấy từ output đã commit, không ghi lại con số từ trí nhớ:

| DoF | chi2 | CFI | TLI | RMSEA |
| --- | --- | --- | --- | --- |
| 13 | 479.29 | 0.923489 | 0.852863 | 0.210303 |

| model | n | DoF | chi2 | CFI | TLI | RMSEA |
| --- | --- | --- | --- | --- | --- | --- |
| legacy_raw_units_reexecution | 812 | 13 | 5068.97 | 0.490055 | 0.0193372 | 0.6925 |
| original_scaled_implicit_exog | 812 | 13 | 479.29 | 0.854747 | 0.720667 | 0.210303 |
| primary_explicit_exog | 812 | 3 | 479.29 | 0.851816 | -0.037285 | 0.44245 |
| age_control_sensitivity | 812 | 1 | 341.741 | 0.893988 | -1.22624 | 0.648189 |
| age_correlated_residuals_SATURATED | 812 | 0 | 3.90642e-08 | — | — | — |
| exclude_quality_flags_sensitivity | 798 | 3 | 460.583 | 0.867736 | 0.0741516 | 0.437467 |

Các thay đổi CFI/TLI có cả phần sửa baseline và df; không diễn giải là so sánh hai theory tốt/xấu đơn thuần.
Mô hình chính fit kém; mô hình có age→mediators và residual covariance bão hòa không có giá trị kiểm định overall fit.
Bootstrap: 200/200 hợp lệ, 0 thất bại, seed 42;
CI 95% cho direct/indirect/total của cả ba điều kiện đều chứa 0. Hệ số đầy đủ và CI từng đường ở `sem_primary_paths_with_bootstrap.csv`.

## Hạn chế dữ liệu và thống kê vẫn còn

- Dữ liệu ít pin độc lập; thí nghiệm không factorial đầy đủ; Tier A không có 43°C/profile thay đổi.
- Tier và cờ Hampel được quyết định hồi cứu theo giao thức nhóm; không thể tuyên bố benchmark hoàn toàn prospective hoặc không có upstream look-ahead.
- SOH chuẩn hóa bằng C0 giai đoạn đầu; C0 không làm predictor, đã có trước forecast origin. Sau lọc có khoảng trống chu kỳ; E1 là next retained observation.
- Phần dư tự tương quan/heteroskedastic, VIF tương tác cao, chỉ 8 cụm train và 3 cụm G1; post-selection inference không mang nghĩa confirmatory.
- Battery fade là slope ước lượng; HC3 không truyền uncertainty của bước 1. Độ dài chuỗi khác nhau ảnh hưởng so sánh.
- EIS ambient ghi 24°C cho discharge 4°C; mediator đồng thời phụ thuộc age/chemistry; quan sát không xác lập nhân quả.
- Bootstrap không chữa misspecification/confounding. 200 replicate là mức tối thiểu, chưa cho endpoint CI chính xác cao.
- Không có prediction intervals đã được hiệu chỉnh; E2 chưa thắng baseline; không thêm mô hình ngoài Regression/SEM.

## Kiểm chứng và lệnh tái lập

Môi trường riêng Python 3.11.16; tất cả version và SHA256 input/output ở
`reports/tables/regression_sem_execution.json`. Lock của 60 dependency cho hai notebook:
`requirements-regression-sem.lock.txt`; `requirements.txt` chung giữ nguyên (torch không cần cho phần này).

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-regression-sem.lock.txt
.venv/Scripts/python src/run_regression_sem.py
.venv/Scripts/python src/run_regression_sem.py --cwd notebooks --verify-repro
.venv/Scripts/python -m unittest discover -s tests -p test_regression_sem.py -v
.venv/Scripts/python src/write_regression_sem_reports.py
```

Execution: [{"notebook": "10_regression.ipynb", "code_cells": 13, "error_outputs": 0}, {"notebook": "50_sem.ipynb", "code_cells": 4, "error_outputs": 0}].
So sánh chính xác CSV/JSON số học giữa hai working directory: **True**.
Input CSV unchanged: **True**.
Kiểm thử cuối: **28 tests PASS, 0 failures/errors**; syntax 6 file Python; xem `reports/audit/regression_sem_final_validation.json`.
`uv pip check` xác nhận các package đã cài tương thích. Không cài torch/jupyter UI vì không được hai notebook sử dụng;
nbclient/ipykernel/nbformat thực thi fresh kernel và lưu đầy đủ output. Các lỗi chạy đầu trong quá trình phát triển
(chuỗi newline của notebook builder, API metadata statsmodels 0.15) đã được sửa trước lần kiểm chứng cuối.

## WBS và việc cần review

| Mã | Tình trạng trong phạm vi Quốc Đạt |
| --- | --- |
| C3.1 | Hoàn thành correlation và design VIF, Re/Rct train |
| C3.2 | Hoàn thành G1 simple, F/t/CI, residual/QQ |
| C3.3 | Hoàn thành ba dạng hàm, SOH-scale RMSE, likelihood Jacobian và smearing |
| C3.4 | Hoàn thành full log-linear dummy/interactions, OLS/cluster CI và conditional slopes |
| C3.5 | Đã chạy four-step backward có hierarchy; **chưa đối chiếu nguyên văn slide p.39 vì file không có trong repo** |
| C3.6 | Hoàn thành verify battery_fade, 23-cell HC3 full/25-cycle và phân tích hạn chế |
| C3.7 | Hoàn thành E1/E2 CSV/metrics/plots/EOL statuses; E2 chưa thắng persistence là kết quả thực tế |
| C3.8 | Có nội dung English report/paper riêng; chờ nhóm tích hợp và review |
| C7.1 | Có sơ đồ thực với hệ số; **chờ nhóm đồng ý đường dẫn**, không tự coi là đã duyệt |
| C7.2 | Hoàn thành dữ liệu new EIS, raw join/missingness/temp audit |
| C7.3 | Hoàn thành estimation, identification, fit audit và sensitivities; fit chính vẫn kém |
| C7.4 | Hoàn thành products + battery bootstrap; không báo % mediation thiếu ổn định hoặc khẳng định significance |

Tiến Hưng review phương pháp, df correction, assumption và slide selection; Leader review ý nghĩa label `regression`
cho strategy có history anchoring và scenario-specific selection trước ghép comparison.
Nhóm cần xác nhận giới hạn retrospective benchmark, treatment của flagged rows trong SEM, EOL pre-origin của B0047,
và chỉ tích hợp những kết luận có CI/metric hỗ trợ. Việc xác nhận đã học Chap 7, duyệt giả thuyết, tích hợp deliverable,
review chéo và kiểm tra Turnitin/AI là phần nhóm thực hiện; notebook chạy thành công không thay thế các việc đó.
Không cần nhóm đổi giao thức để đọc các đầu ra hiện tại; pipeline chung `src/protocol.py` vẫn thuộc Leader.

## File và git diff

Thay đổi tracked: hai notebook, bốn hình cũ liên quan, CSV regression và `.gitignore` (bỏ qua bytecode).
Thêm: bốn module analysis/runner/utils, module sinh report này, tests, lock dependency, report Markdown,
tables, forecast/selected diagnostic figures, diagram PDF và bản sao bảo toàn trước audit.
Notebook thử nghiệm chưa track là thay đổi có sẵn, không phải kết quả mới của audit.
`git diff --stat` chỉ đếm tracked files; file mới chưa stage vẫn được liệt kê bằng status:

```text
.gitignore                                         |    3 +
 docs/Work_log.md                                   |   70 +
 notebooks/10_regression.ipynb                      | 4498 ++++++++++++--------
 notebooks/50_sem.ipynb                             | 1491 ++++---
 reports/figures/regression_functional_forms.png    |  Bin 207110 -> 144521 bytes
 .../figures/regression_preferred_diagnostics.png   |  Bin 265016 -> 336978 bytes
 reports/figures/regression_simple_diagnostics.png  |  Bin 188622 -> 233101 bytes
 reports/figures/sem_path_diagram.png               |  Bin 67964 -> 428545 bytes
 reports/predictions/regression.csv                 |  852 ++--
 9 files changed, 4066 insertions(+), 2848 deletions(-)
```

```text
M .gitignore
 M docs/Work_log.md
 M notebooks/10_regression.ipynb
 M notebooks/50_sem.ipynb
 M reports/figures/regression_functional_forms.png
 M reports/figures/regression_preferred_diagnostics.png
 M reports/figures/regression_simple_diagnostics.png
 M reports/figures/sem_path_diagram.png
 M reports/predictions/regression.csv
?? docs/Regression_Report.md
?? docs/Regression_SEM_Audit.md
?? docs/SEM_Report.md
?? notebooks/10_regression.test.ipynb
?? reports/audit/
?? reports/figures/regression_selected_diagnostics.png
?? reports/figures/regression_test_forecasts.png
?? reports/figures/sem_path_diagram.pdf
?? reports/tables/
?? requirements-regression-sem.lock.txt
?? src/regression_analysis.py
?? src/regression_sem_utils.py
?? src/run_regression_sem.py
?? src/sem_analysis.py
?? src/write_regression_sem_reports.py
?? tests/
```

## Final stabilization — 10/11/2026

Branch `quoc-dat`, HEAD `73452fd`; đã đọc branch/status/log trước khi sửa. Bảo toàn 109 file hiện có trong
`.cache/final_stabilization_before/`, không reset/clean/stash và không sửa dữ liệu, Rec.md, pipeline hay source thành viên khác.
Đã kiểm tra SHA256 35 file được bảo vệ, kể cả archive thật và notebook thử nghiệm có sẵn: không thay đổi.

- **P0 archive:** fit/CV/test/export Regression độc lập với archive; historical metrics chuyển sang bảng riêng.
  SEM chạy primary, đủ sensitivity/bootstrap/tables/diagram trước khi đọc historical comparison tùy chọn.
  Archive thiếu/sai định dạng được bỏ qua với thông báo và status JSON; bảng historical rỗng, không tạo kết quả giả.
- **P1 feature availability:** cả 12 Tier A battery, gồm mọi dòng trước khi bỏ quality flags, đều có ba điều kiện cố định;
  36/36 battery-feature checks đạt, mỗi biến có đúng một giá trị hữu hạn. Lưu `regression_operating_conditions.csv`.
  E2 dựng predictors từ giá trị tại origin; nếu điều kiện thay đổi/NaN/Inf thì dừng rõ ràng. Test giả lập sửa từng điều kiện
  sau origin xác nhận không âm thầm dùng future features. Tọa độ cycle tương lai là lịch đánh giá retrospective đã chốt.
- **P2 nhãn:** giữ CSV 426 dòng/8 cột và `model=regression`. Mapping E1=`anchored_loglinear`,
  E2=`anchored_quadratic`, công thức và quy tắc anchor ở `regression_selected.json`. Đây là strategy theo scenario,
  có hai bộ hệ số riêng; SOH anchor thuộc chính pin test, chỉ dùng preceding observation (E1) hoặc origin cố định (E2).
- **SEM inference:** giữ nguyên specification và code ước lượng. 812 dòng chỉ thuộc 22 pin độc lập;
  IID SE/p-values chỉ tham khảo, ưu tiên battery-block CI. Cả 15 CI cho direct/Re-indirect/Rct-indirect/combined/total
  của ba điều kiện đều chứa 0; không khẳng định causal mediation. Primary df=3, rank=25/25, RMSEA=0.442450 vẫn cao.

**Kiểm chứng đã thực thi:** 28/28 tests đạt; 0 failure/error; bao gồm train/test disjoint, GroupKFold theo pin,
selection không đổi khi sửa test outcomes, E1/E2 SOH và feature availability, archive thiếu/hỏng,
schema/finite predictions, censoring EOL/RUL, bootstrap whole-block/multiplicity/rejection/exhaustion,
effects–CI–diagram consistency và deterministic sampling. Sáu file Python và code cells của hai notebook compile thành công.
Hai notebook chạy đủ từ root và notebooks/: 13 + 4 code cells, 0 error outputs; CSV/JSON số học khớp tuyệt đối giữa hai cwd.
Bản sao repo riêng không có hai archive chạy đủ cả hai notebook, sinh báo cáo thành công;
57 CSV kết quả hiện tại khớp từng byte với bản có archive. Đã xem hình forecast và path diagram.
Môi trường `.venv` hiện có không có pip; kiểm tra 98 dependency constraints bằng importlib.metadata/packaging: 0 vi phạm,
không cài hay thay dependency hệ thống. `git diff --check` đạt.

**Sai khác số học được giữ trung thực:** dự báo so với file bảo toàn lệch tối đa 6.85e-14 SOH;
bootstrap CI endpoint lệch tối đa 6.13e-7. Chạy đối chứng bằng source trước sửa tái tạo chính xác CSV dự báo và SEM mới,
cho thấy sai khác do tái thực thi số học, không do đổi seed/specification/protocol. Riêng sensitivity raw-unit có objective
từ khoảng 0.590259 trong bảng có sẵn sang 6.242571; source đối chứng cũng cho đúng kết quả này.
Đó là độ nhạy solver với scale ở bản không chuẩn hóa; success flag không chứng minh nghiệm tốt. Primary scaled SEM không đổi estimator.
Không dùng lại `sem_historical_raw_units_coefficients.csv` có sẵn để thay kết quả reexecution.
Historical rows không còn trong current metrics/EOL; archive metrics được tách riêng.

**Kết quả chính (macro theo pin):** Regression E1 MAE=0.0137394/RMSE=0.0202341 so với persistence
0.0141225/0.0201418; E2 MAE=0.0335355/RMSE=0.0420071 so với 0.0238145/0.0299687.
E1 chỉ thắng nhẹ MAE; E2 vẫn kém persistence. SEM χ²=479.289894, df=3, CFI=0.851816,
TLI=-0.037285, RMSEA=0.442450; bootstrap 200 hợp lệ/200 lần, 0 thất bại, seed 42.

**File trong lượt ổn định cuối:** `src/regression_analysis.py`, `src/sem_analysis.py`,
`src/write_regression_sem_reports.py`, `tests/test_regression_sem.py`, hai notebook, bốn tài liệu được yêu cầu;
outputs liên quan được tái sinh và thêm status/availability/historical tables, validation JSON/log.
Không sửa `.gitignore`, requirements chung/lock hay các module utils/runner trong lượt này.
Diff so với HEAD còn bao gồm thay đổi chưa commit từ audit trước; file mới vẫn chưa stage.
Chưa commit/push/merge/PR. Bằng chứng chi tiết:
[validation JSON](../reports/audit/regression_sem_final_validation.json), [test log](../reports/audit/regression_sem_final_tests.log).

**Nhóm cần review:** chấp thuận nhãn chung cho strategy khác nhau theo scenario và history anchoring;
đối chiếu slide p.39; xác nhận theory paths; tích hợp báo cáo và Model Comparison.
Giới hạn còn nguyên: chỉ 4 pin test, 22 block SEM, confounding/incomplete crossing, retrospective preprocessing,
chưa có prediction intervals, primary SEM fit kém và E2 chưa thắng persistence. Dữ liệu time-varying cần giao thức mới trước khi hỗ trợ.
