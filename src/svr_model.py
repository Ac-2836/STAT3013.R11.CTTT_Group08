"""Mô hình SVR dự báo SOH chu kỳ kế tiếp (Minh Vũ, nhánh minh-vu).

Trả lời RQ4: đặc trưng sạc và trở kháng dự báo SOH chu kỳ kế tiếp tốt đến đâu,
và đặc trưng nào quan trọng nhất. Kế hoạch: docs/Go.md GĐ5c, WBS C6.2 đến C6.5.

Code được chia thành các class, mỗi class lo đúng một việc:

    SVRConfig            Toàn bộ tham số đặt ở một chỗ (số lag, lưới tham số, seed...).
    FeatureBuilder       Tạo đặc trưng cho từng chu kỳ và kiểm tra không rò rỉ dữ liệu.
    RecursiveForecaster  Dự báo cuốn chiếu nhiều bước (kịch bản E2).
    SVRTuner             Chọn C, epsilon, gamma bằng GroupKFold theo pin + quy tắc 1-SE.
    SVRForecaster        Train model cuối, dự báo E1 và E2, tạo file dự báo chung.
    ImportanceAnalyzer   Permutation importance: đặc trưng nào quan trọng (RQ4).
    SVRPlotter           Vẽ hình dự báo và hình importance.
    SVRExperiment        Gọi các class trên theo đúng thứ tự, từ đọc dữ liệu tới lưu file.

Luồng chạy (SVRExperiment.run):
    load_split() -> FeatureBuilder -> SVRTuner -> SVRForecaster -> evaluate() -> ImportanceAnalyzer -> SVRPlotter

Dữ liệu, cách chia train/test và metric đều lấy từ src/protocol.py (dùng chung cả nhóm),
nên kết quả so sánh được với 3 mô hình còn lại.

Chạy từ thư mục gốc của repo:
    python src/svr_model.py
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from protocol import EOL_SOH, KEYS, ORIGIN_FRACTION, PRED_COLS, evaluate, load_split, naive

ROOT = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------------------------
# 1. Cấu hình
# ----------------------------------------------------------------------------------------
@dataclass
class SVRConfig:
    """Mọi con số "tự chọn" của mô hình nằm ở đây, để dễ tìm và dễ ghi vào báo cáo."""

    n_lags: int = 5                 # số chu kỳ trước được dùng làm đặc trưng (soh_lag1..soh_lag5)
    n_splits: int = 4               # GroupKFold k = 4 theo pin (docs/Rec.md mục 6)
    seed: int = 42
    n_repeats: int = 30             # số lần xáo trộn trong permutation importance
    interval_level: float = 0.95    # mức của khoảng dự báo E1

    # Lưới tham số cuối cùng = toàn bộ giá trị của Go.md + phần mở rộng.
    # Lý do mở rộng: với lưới gốc, giá trị tốt nhất rơi đúng biên của cả 3 tham số.
    param_grid: dict = field(default_factory=lambda: {
        "svr__C":       [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100],
        "svr__epsilon": [0.0001, 0.0003, 0.001, 0.005, 0.01],
        "svr__gamma":   ["scale", 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.1],
    })
    # Lưới gốc trong docs/Go.md GĐ5c, giữ lại để báo cáo kết quả trên lưới này
    original_grid: dict = field(default_factory=lambda: {
        "svr__C":       [0.1, 1, 10, 100],
        "svr__epsilon": [0.001, 0.005, 0.01],
        "svr__gamma":   ["scale", 0.01, 0.1],
    })

    # Đặc trưng đo được (không biết trước sau mốc E2), bị giữ nguyên ở giá trị cuối cùng đã biết
    held_features: tuple = ("chg_cc_time_s", "Re_ohm", "Rct_ohm", "log_rest_h")

    # Biến bị cấm làm đặc trưng vì rò rỉ thông tin của chính chu kỳ đang dự báo (Go.md 4.6)
    forbidden: frozenset = frozenset({
        "soh", "capacity_Ah", "capacity_to_2p7V_Ah", "capacity_int_Ah", "energy_Wh", "duration_s",
        "t_to_cutoff_s", "T_max_C", "V_min", "C0_Ah", "soh_segment"})

    # Nơi lưu kết quả
    fig_dir: Path = ROOT / "reports" / "figures"
    pred_path: Path = ROOT / "reports" / "predictions" / "svr.csv"


# ----------------------------------------------------------------------------------------
# 2. Tạo đặc trưng
# ----------------------------------------------------------------------------------------
class FeatureBuilder:
    """Biến bảng chu kỳ thành ma trận đặc trưng X.

    Nguyên tắc: với lần xả ở chu kỳ t, chỉ dùng thông tin đã biết TRƯỚC lần xả đó.
    - soh_lag1..5, soh_roll5: SOH của 5 chu kỳ trước (cùng pin) và trung bình của chúng
    - chg_cc_time_s: thời gian pha CC của lần sạc ngay trước (đo trước khi xả)
    - Re_ohm, Rct_ohm: trở kháng đo lần gần nhất TRƯỚC chu kỳ t (đã ghép ở bước tiền xử lý)
    - T_max_prev_C: nhiệt độ cao nhất của lần xả TRƯỚC (t-1), không phải của t
    - log_rest_h: log(1 + thời gian nghỉ), vì thời gian nghỉ lệch phải rất mạnh
    - is_4C, I_level_A, cutoff_V: điều kiện vận hành, cố định theo thiết kế thí nghiệm
    """

    def __init__(self, config: SVRConfig):
        self.cfg = config
        self.lag_cols = [f"soh_lag{k}" for k in range(1, config.n_lags + 1)]

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trả về X, cùng index với df. Lag được tính riêng trong từng pin."""
        g = df.groupby("battery_id")
        X = pd.DataFrame(index=df.index)
        for k in range(1, self.cfg.n_lags + 1):
            X[f"soh_lag{k}"] = g.soh.shift(k)          # shift(k): giá trị của k chu kỳ trước
        X["soh_roll5"] = X[self.lag_cols].mean(axis=1)
        X["chg_cc_time_s"] = df.chg_cc_time_s
        X["Re_ohm"] = df.Re_ohm
        X["Rct_ohm"] = df.Rct_ohm
        X["T_max_prev_C"] = g.T_max_C.shift(1)
        X["log_rest_h"] = np.log1p(df.rest_before_h)
        X["is_4C"] = (df.ambient_C < 10).astype(float)  # tier A chỉ có 4°C và 24°C nên 1 dummy là đủ
        X["I_level_A"] = df.I_level_A
        X["cutoff_V"] = df.discharge_cutoff_V
        return X

    @property
    def feature_names(self) -> list:
        return self.lag_cols + ["soh_roll5", "chg_cc_time_s", "Re_ohm", "Rct_ohm", "T_max_prev_C",
                                "log_rest_h", "is_4C", "I_level_A", "cutoff_V"]

    def check_no_leakage(self):
        """Dừng chương trình nếu có biến bị cấm lọt vào danh sách đặc trưng."""
        leaked = self.cfg.forbidden & set(self.feature_names)
        assert not leaked, f"biến bị cấm được dùng làm đặc trưng: {leaked}"

    def training_matrix(self, train: pd.DataFrame):
        """X, y, groups cho tập train.

        5 chu kỳ đầu mỗi pin chưa đủ lịch sử (lag bị NaN) nên bị bỏ.
        groups = battery_id, dùng cho GroupKFold để mỗi fold giữ lại nguyên pin.
        """
        X = self.build(train)
        has_history = X[self.lag_cols + ["T_max_prev_C"]].notna().all(axis=1)
        return X[has_history], train.soh[has_history], train.battery_id[has_history]


# ----------------------------------------------------------------------------------------
# 3. Dự báo cuốn chiếu (E2)
# ----------------------------------------------------------------------------------------
class RecursiveForecaster:
    """Đứng ở một mốc (origin), dự báo liên tục đến hết chuỗi của một pin.

    Mỗi giá trị vừa dự báo được đẩy vào "lịch sử" và trở thành soh_lag1 của bước sau.
    Các số đo chưa biết sau mốc (cfg.held_features, T_max_prev_C) được giữ ở giá trị cuối cùng
    trước mốc. Điều kiện vận hành (is_4C, I_level_A, cutoff_V) cố định nên vẫn dùng bình thường.
    Class này được dùng ở 2 chỗ: chọn tham số (SVRTuner) và dự báo E2 trên tập test (SVRForecaster).
    """

    def __init__(self, builder: FeatureBuilder):
        self.builder = builder
        self.held = list(builder.cfg.held_features)

    def forecast(self, model, battery: pd.DataFrame, origin: int) -> np.ndarray:
        """Dự báo các dòng origin..cuối của MỘT pin. Chỉ dùng thông tin trước dòng `origin`."""
        b = battery.reset_index(drop=True)
        Xb = self.builder.build(b)
        history = list(b.soh[:origin])                 # SOH thật trước mốc
        last_known = Xb.iloc[origin - 1]
        t_max_last = b.T_max_C.iloc[origin - 1]        # nhiệt độ max của lần xả cuối cùng đã biết
        out = []
        for i in range(origin, len(b)):
            x = Xb.iloc[[i]].copy()
            for k in range(1, self.builder.cfg.n_lags + 1):
                x[f"soh_lag{k}"] = history[-k]         # lag lấy từ lịch sử (thật + đã dự báo)
            x["soh_roll5"] = np.mean(history[-self.builder.cfg.n_lags:])
            x[self.held] = last_known[self.held].values
            x["T_max_prev_C"] = t_max_last
            y_hat = model.predict(x[self.builder.feature_names])[0]
            history.append(y_hat)
            out.append(y_hat)
        return np.array(out)


# ----------------------------------------------------------------------------------------
# 4. Chọn tham số
# ----------------------------------------------------------------------------------------
class SVRTuner:
    """Chọn C, epsilon, gamma chỉ bằng dữ liệu train (không nhìn tập test).

    Bước 1: GridSearchCV với GroupKFold(4) theo pin, điểm = RMSE dự báo 1 bước.
    Bước 2 (quy tắc 1-SE): nhiều cấu hình có RMSE 1 bước gần như bằng nhau, nhưng khi dự báo
            cuốn chiếu thì khác nhau rất nhiều. Giữ các cấu hình có RMSE trong vòng 1 sai số chuẩn
            so với cấu hình tốt nhất, rồi chọn cấu hình có RMSE cuốn chiếu (kiểu E2) nhỏ nhất.

    Ý nghĩa tham số (Chap 6, tr. 31):
    - C: phạt nặng hay nhẹ khi sai quá epsilon. C lớn -> bám sát dữ liệu train (dễ học vẹt).
    - epsilon: sai số nhỏ hơn mức này coi như đúng (đơn vị SOH).
    - gamma: độ "cục bộ" của kernel RBF. Nhỏ -> hàm mượt, gần tuyến tính; lớn -> hàm uốn lượn.
    """

    def __init__(self, config: SVRConfig, builder: FeatureBuilder, forecaster: RecursiveForecaster):
        self.cfg = config
        self.builder = builder
        self.forecaster = forecaster
        self.cv = GroupKFold(n_splits=config.n_splits)

    @staticmethod
    def make_pipeline():
        """Imputer -> Scaler -> SVR nằm chung một Pipeline.

        Nhờ vậy trong mỗi fold, median để điền Re/Rct bị thiếu và mean/SD để chuẩn hóa
        chỉ được tính trên phần train của fold đó (không rò rỉ sang phần kiểm tra).
        """
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), SVR(kernel="rbf"))

    # ---- Bước 1 -------------------------------------------------------------------------
    def grid_search(self, X, y, groups) -> pd.DataFrame:
        gs = GridSearchCV(self.make_pipeline(), self.cfg.param_grid, cv=self.cv,
                          scoring="neg_root_mean_squared_error", n_jobs=-1)
        gs.fit(X, y, groups=groups)
        self.grid_ = gs
        res = pd.DataFrame(gs.cv_results_).sort_values("rank_test_score")
        res["cv_RMSE"] = -res.mean_test_score
        res["cv_RMSE_sd"] = res.std_test_score
        self.cv_results_ = res
        return res

    def best_on_original_grid(self) -> pd.Series:
        """Cấu hình tốt nhất nếu chỉ dùng lưới gốc của Go.md (để ghi vào báo cáo)."""
        r, og = self.cv_results_, self.cfg.original_grid
        mask = (r.param_svr__C.isin(og["svr__C"]) & r.param_svr__epsilon.isin(og["svr__epsilon"])
                & r.param_svr__gamma.astype(str).isin([str(v) for v in og["svr__gamma"]]))
        return r[mask].iloc[0]

    def compare_with_naive(self, X, y, groups) -> pd.DataFrame:
        """RMSE từng fold của cấu hình tốt nhất (bước 1) so với naive SOH(t) = SOH(t-1)."""
        best_i = self.grid_.best_index_
        rows = []
        for i, (_, va) in enumerate(self.cv.split(X, y, groups)):
            rows.append({"fold": i, "held_out": ", ".join(sorted(groups.iloc[va].unique())),
                         "SVR_RMSE": -self.grid_.cv_results_[f"split{i}_test_score"][best_i],
                         "naive_RMSE": np.sqrt(((y.iloc[va] - X.soh_lag1.iloc[va]) ** 2).mean())})
        out = pd.DataFrame(rows)
        out.loc["mean"] = ["", "", out.SVR_RMSE.mean(), out.naive_RMSE.mean()]
        return out

    # ---- Bước 2 -------------------------------------------------------------------------
    def recursive_cv_rmse(self, params, train, X, y, groups) -> float:
        """RMSE cuốn chiếu: trong mỗi fold, dự báo các pin bị giữ lại từ mốc 60% đến hết."""
        fold_rmse = []
        for tr_i, va_i in self.cv.split(X, y, groups):
            fitted = clone(self.make_pipeline()).set_params(**params).fit(X.iloc[tr_i], y.iloc[tr_i])
            err = []
            for bid in groups.iloc[va_i].unique():
                b = train[train.battery_id == bid]
                origin = int(len(b) * ORIGIN_FRACTION)
                err.extend(self.forecaster.forecast(fitted, b, origin) - b.soh.values[origin:])
            fold_rmse.append(np.sqrt(np.mean(np.square(err))))
        return float(np.mean(fold_rmse))

    def naive_recursive_cv_rmse(self, train, X, y, groups) -> float:
        """Naive cho E2: giữ nguyên SOH cuối cùng trước mốc. Dùng làm mốc so sánh."""
        fold_rmse = []
        for _, va_i in self.cv.split(X, y, groups):
            err = []
            for bid in groups.iloc[va_i].unique():
                s = train.soh[train.battery_id == bid].values
                origin = int(len(s) * ORIGIN_FRACTION)
                err.extend(s[origin - 1] - s[origin:])
            fold_rmse.append(np.sqrt(np.mean(np.square(err))))
        return float(np.mean(fold_rmse))

    def select_one_se(self, train, X, y, groups) -> dict:
        """Quy tắc 1-SE + RMSE cuốn chiếu. Trả về bộ tham số được chọn."""
        res = self.cv_results_
        self.se_best_ = res.iloc[0].cv_RMSE_sd / np.sqrt(self.cfg.n_splits)   # sai số chuẩn = SD / căn(k)
        cand = res[res.cv_RMSE <= res.iloc[0].cv_RMSE + self.se_best_].copy()
        cand["recursive_cv_RMSE"] = [self.recursive_cv_rmse(p, train, X, y, groups) for p in cand.params]
        self.candidates_ = cand.sort_values("recursive_cv_RMSE")
        self.naive_recursive_ = self.naive_recursive_cv_rmse(train, X, y, groups)
        self.naive_one_step_ = float(np.mean([np.sqrt(((y.iloc[va] - X.soh_lag1.iloc[va]) ** 2).mean())
                                              for _, va in self.cv.split(X, y, groups)]))
        self.chosen_ = self.candidates_.iloc[0].params
        return self.chosen_

    def edge_check(self, params: dict) -> pd.DataFrame:
        """Tham số có nằm hẳn bên trong lưới không? Nằm ở biên nghĩa là có thể cần mở rộng lưới."""
        rows = []
        for p in ["svr__C", "svr__gamma"]:
            numeric = [v for v in self.cfg.param_grid[p] if v != "scale"]
            inside = params[p] != "scale" and min(numeric) < params[p] < max(numeric)
            rows.append({"param": p, "value": params[p], "inside_grid": inside})
        return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------
# 5. Train model cuối và dự báo trên tập test
# ----------------------------------------------------------------------------------------
class SVRForecaster:
    """Train model cuối trên toàn bộ train, rồi dự báo 213 dòng is_eval của 4 pin test.

    E1: dự báo 1 bước, lag là SOH THẬT của các chu kỳ trước. Có khoảng dự báo 95% (conformal).
    E2: đứng ở dòng is_eval đầu tiên (mốc 60%), dự báo cuốn chiếu đến hết chuỗi.
    """

    def __init__(self, config: SVRConfig, builder: FeatureBuilder, forecaster: RecursiveForecaster):
        self.cfg = config
        self.builder = builder
        self.forecaster = forecaster

    def fit(self, X, y, groups, params: dict):
        self.model_ = clone(SVRTuner.make_pipeline()).set_params(**params).fit(X, y)
        # Khoảng dự báo conformal: lấy phần dư out-of-fold trên các pin train (cùng GroupKFold),
        # q = phân vị 95% của |sai số|. Khoảng dự báo = y_hat ± q.
        oof = cross_val_predict(self.model_, X, y, groups=groups, cv=GroupKFold(self.cfg.n_splits))
        self.q_ = float(np.quantile(np.abs(y - oof), self.cfg.interval_level))
        return self

    def predict_e1(self, test: pd.DataFrame) -> pd.DataFrame:
        X_te = self.builder.build(test)
        ev = test.is_eval.values
        assert X_te.loc[ev, self.builder.lag_cols].notna().all().all(), "dòng cần dự báo phải có đủ 5 lag"
        y_hat = self.model_.predict(X_te[ev][self.builder.feature_names])
        rows = test.loc[ev, KEYS + ["soh"]].reset_index(drop=True)
        return rows.assign(model="svr", scenario="E1", y_true=rows.soh, y_pred=y_hat,
                           y_lower=y_hat - self.q_, y_upper=y_hat + self.q_)[PRED_COLS]

    def predict_e2(self, test: pd.DataFrame) -> pd.DataFrame:
        parts = []
        for bid, b in test.groupby("battery_id", sort=True):
            b = b.reset_index(drop=True)
            origin = int(np.flatnonzero(b.is_eval)[0])
            y_hat = self.forecaster.forecast(self.model_, b, origin)
            parts.append(b.loc[origin:, KEYS + ["soh"]].assign(y_pred=y_hat))
        rows = pd.concat(parts, ignore_index=True)
        # Khoảng dự báo không tính cho E2 (sai số tăng theo số bước), để trống như định dạng cho phép
        return rows.assign(model="svr", scenario="E2", y_true=rows.soh,
                           y_lower=np.nan, y_upper=np.nan)[PRED_COLS]

    def predict(self, test: pd.DataFrame) -> pd.DataFrame:
        """Gộp E1 và E2 theo đúng định dạng chung reports/predictions/<model>.csv."""
        return pd.concat([self.predict_e1(test), self.predict_e2(test)], ignore_index=True)

    @staticmethod
    def interval_coverage(pred: pd.DataFrame) -> float:
        """Tỉ lệ giá trị thật nằm trong khoảng dự báo E1 (lý tưởng khoảng 95%)."""
        e1 = pred[pred.scenario == "E1"]
        return float(((e1.y_true >= e1.y_lower) & (e1.y_true <= e1.y_upper)).mean())


# ----------------------------------------------------------------------------------------
# 6. Đặc trưng nào quan trọng (RQ4)
# ----------------------------------------------------------------------------------------
class ImportanceAnalyzer:
    """Permutation importance trên các dòng E1 của tập test.

    Xáo trộn một đặc trưng (cắt đứt quan hệ của nó với SOH) rồi đo RMSE tăng bao nhiêu.
    Tăng nhiều = model dựa nhiều vào đặc trưng đó.
    Vì 5 lag và soh_roll5 gần như cùng một thông tin, xáo từng cái riêng lẻ thì model vẫn đọc
    được từ các cái còn lại. Nên ta xáo thêm CẢ NHÓM lịch sử SOH cùng lúc.
    """

    def __init__(self, config: SVRConfig, builder: FeatureBuilder):
        self.cfg = config
        self.builder = builder

    def run(self, model, test: pd.DataFrame) -> pd.DataFrame:
        ev = test.is_eval.values
        X_ev = self.builder.build(test)[ev][self.builder.feature_names]
        y_ev = test.soh[ev]

        # Từng đặc trưng riêng lẻ (hàm có sẵn của sklearn)
        pi = permutation_importance(model, X_ev, y_ev, scoring="neg_root_mean_squared_error",
                                    n_repeats=self.cfg.n_repeats, random_state=self.cfg.seed)
        imp = pd.DataFrame({"feature": self.builder.feature_names,
                            "RMSE_increase": pi.importances_mean, "sd": pi.importances_std})

        # Cả nhóm lịch sử SOH: dùng CÙNG một hoán vị dòng cho tất cả các cột trong nhóm
        rng = np.random.default_rng(self.cfg.seed)
        self.base_rmse_ = float(np.sqrt(((model.predict(X_ev) - y_ev) ** 2).mean()))
        hist_cols = self.builder.lag_cols + ["soh_roll5"]
        inc = []
        for _ in range(self.cfg.n_repeats):
            Xp = X_ev.copy()
            Xp[hist_cols] = X_ev[hist_cols].values[rng.permutation(len(X_ev))]
            inc.append(np.sqrt(((model.predict(Xp) - y_ev) ** 2).mean()) - self.base_rmse_)
        group = pd.DataFrame([{"feature": "[group] SOH history (lags 1-5 + roll5)",
                               "RMSE_increase": np.mean(inc), "sd": np.std(inc)}])
        return pd.concat([imp, group]).sort_values("RMSE_increase", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------------------------
# 7. Vẽ hình
# ----------------------------------------------------------------------------------------
class SVRPlotter:
    """Ba hình cho báo cáo, lưu vào reports/figures/."""

    def __init__(self, config: SVRConfig):
        self.cfg = config

    def forecasts(self, test: pd.DataFrame, pred: pd.DataFrame):
        """4 ô, mỗi ô một pin test: SOH thật, dự báo E1 (kèm khoảng 95%), dự báo E2, mốc EOL."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
        for ax, (bid, b) in zip(axes.ravel(), test.groupby("battery_id", sort=True)):
            p1 = pred[(pred.battery_id == bid) & (pred.scenario == "E1")]
            p2 = pred[(pred.battery_id == bid) & (pred.scenario == "E2")]
            ax.plot(b.cycle_valid, b.soh, color="0.3", lw=1.2, label="true SOH")
            ax.fill_between(p1.cycle_valid, p1.y_lower, p1.y_upper, color="tab:blue", alpha=0.15,
                            label="E1 95% interval")
            ax.plot(p1.cycle_valid, p1.y_pred, color="tab:blue", lw=1.2, label="E1 one step ahead")
            ax.plot(p2.cycle_valid, p2.y_pred, color="tab:orange", lw=1.8, ls="--", label="E2 from the origin")
            ax.axvline(p1.cycle_valid.min() - 0.5, color="0.6", ls=":", lw=1)    # mốc 60%
            ax.axhline(EOL_SOH, color="tab:red", lw=0.8, ls="--")                # SOH = 0.8
            c = b.iloc[0]
            ax.set_title(f"{bid}: {c.ambient_C:.0f}°C, {c.I_level_A:.0f} A, cutoff {c.discharge_cutoff_V} V")
            ax.set_xlabel("cycle_valid")
            ax.set_ylabel("SOH")
        axes[0, 0].legend(fontsize=8, loc="lower left")
        fig.suptitle("SVR on the test batteries (dotted line = origin at 60%, red line = EOL at SOH 0.8)")
        fig.tight_layout()
        fig.savefig(self.cfg.fig_dir / "svr_forecasts.png", dpi=150)
        return fig

    def tuning_heatmap(self, tuner):
        """Bản đồ nhiệt kết quả chọn tham số: trục ngang C, trục dọc gamma.

        Mỗi ô lấy epsilon tốt nhất của cặp (C, gamma) đó.
        Ô trái: RMSE dự báo 1 bước (vòng 2). Viền đen = nhóm ngang điểm trong 1 SE.
        Ô phải: RMSE dự báo cuốn chiếu, chỉ tính cho nhóm ngang điểm (vòng 3).
        Viền vàng = bộ tham số được chọn. Ô có dấu chấm = giá trị thuộc lưới gốc Go.md.
        """
        Cs = self.cfg.param_grid["svr__C"]
        # 'scale' của sklearn = 1 / (số đặc trưng × phương sai), sau chuẩn hóa ≈ 1/14 ≈ 0,07,
        # nên xếp nó vào đúng vị trí giữa 0,01 và 0,1 để trục gamma tăng dần từ trên xuống
        n_feat = len(self.cfg.param_grid) and len(tuner.builder.feature_names)
        order = sorted(self.cfg.param_grid["svr__gamma"], key=lambda g: 1 / n_feat if g == "scale" else g)
        gammas = order
        g_labels = [str(g) for g in gammas]
        g_ticks = [f"scale (≈{1 / n_feat:.2f})" if g == "scale" else str(g) for g in gammas]
        res = tuner.cv_results_.assign(g=tuner.cv_results_.param_svr__gamma.astype(str))
        one_step = res.groupby(["g", "param_svr__C"]).cv_RMSE.min().unstack().reindex(index=g_labels, columns=Cs)
        cand = tuner.candidates_.assign(g=tuner.candidates_.param_svr__gamma.astype(str))
        recursive = cand.groupby(["g", "param_svr__C"]).recursive_cv_RMSE.min().unstack().reindex(index=g_labels, columns=Cs)
        in_1se = recursive.notna()
        chosen = (g_labels.index(str(tuner.chosen_["svr__gamma"])), Cs.index(tuner.chosen_["svr__C"]))
        og = self.cfg.original_grid

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        panels = [(one_step, "Vòng 2: RMSE dự báo 1 bước (GroupKFold)", 0.03, "Blues_r"),
                  (recursive, "Vòng 3: RMSE dự báo cuốn chiếu (chỉ nhóm trong 1 SE)", None, "Oranges_r")]
        for ax, (tab, title, vmax, cmap) in zip(axes, panels):
            cm = plt.get_cmap(cmap).copy()
            cm.set_bad("#EEEEEE")                                   # ô không có giá trị
            im = ax.imshow(tab.values, cmap=cm, vmax=vmax, aspect="auto")
            cb = fig.colorbar(im, ax=ax, shrink=0.85)
            cb.set_label("RMSE (thấp hơn = tốt hơn)" + (f", màu cắt ở {vmax}" if vmax else ""))
            for i in range(len(g_labels)):
                for j in range(len(Cs)):
                    v = tab.values[i, j]
                    if np.isfinite(v):
                        ax.text(j, i, f"{v * 1000:.1f}", ha="center", va="center", fontsize=7.5,
                                color="white" if (vmax and v < 0.016) or (not vmax and v < 0.06) else "0.15")
                    if in_1se.values[i, j]:                         # nhóm ngang điểm: viền đen
                        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="black", lw=1.6))
                    if Cs[j] in og["svr__C"] and gammas[i] in og["svr__gamma"]:
                        ax.plot(j, i - 0.30, "o", ms=3, color="0.2")            # thuộc lưới gốc Go.md (chấm phía trên số)
            # ô được chọn: viền vàng dày bao quanh đúng ô đó
            ax.add_patch(plt.Rectangle((chosen[1] - 0.5, chosen[0] - 0.5), 1, 1, fill=False, ec="gold", lw=4, zorder=5))
            ax.set_xticks(range(len(Cs)), [str(c) for c in Cs])
            ax.set_yticks(range(len(g_labels)), g_ticks)
            ax.set_xlabel("C (mức phạt)")
            ax.set_ylabel("gamma (độ uốn của kernel RBF)")
            ax.set_title(title, fontsize=11)
        c = tuner.chosen_
        fig.suptitle(f"Chọn tham số SVR. Viền vàng = được chọn: C = {c['svr__C']}, epsilon = {c['svr__epsilon']}, "
                     f"gamma = {c['svr__gamma']}\n"
                     "Số trong ô = RMSE × 1000 (epsilon tốt nhất của ô) · viền đen = nhóm ngang điểm (trong 1 SE) · "
                     f"chấm = lưới gốc Go.md · naive: 1 bước {tuner.naive_one_step_ * 1000:.1f}, "
                     f"cuốn chiếu {tuner.naive_recursive_ * 1000:.1f}", fontsize=10.5)
        fig.tight_layout()
        fig.savefig(self.cfg.fig_dir / "svr_tuning_heatmap.png", dpi=150)
        return fig

    def importance(self, imp: pd.DataFrame):
        """Biểu đồ thanh ngang; thanh màu cam là nhóm lịch sử SOH."""
        fig, ax = plt.subplots(figsize=(8, 5))
        d = imp.iloc[::-1]
        colors = ["tab:orange" if f.startswith("[group]") else "tab:blue" for f in d.feature]
        ax.barh(d.feature, d.RMSE_increase, xerr=d.sd, color=colors)
        ax.axvline(0, color="0.5", lw=0.8)
        ax.set_xlabel("increase in test RMSE when the feature is shuffled (E1)")
        ax.set_title(f"SVR permutation importance ({self.cfg.n_repeats} repeats)")
        fig.tight_layout()
        fig.savefig(self.cfg.fig_dir / "svr_permutation_importance.png", dpi=150)
        return fig


# ----------------------------------------------------------------------------------------
# 8. Chạy toàn bộ
# ----------------------------------------------------------------------------------------
class SVRExperiment:
    """Gọi các class trên theo thứ tự. Mỗi bước là một hàm riêng để notebook gọi từng bước
    và hiển thị kết quả trung gian; run() chạy tất cả một lần."""

    def __init__(self, config: SVRConfig = None):
        self.cfg = config or SVRConfig()
        self.builder = FeatureBuilder(self.cfg)
        self.recursive = RecursiveForecaster(self.builder)
        self.tuner = SVRTuner(self.cfg, self.builder, self.recursive)
        self.forecaster = SVRForecaster(self.cfg, self.builder, self.recursive)
        self.importance = ImportanceAnalyzer(self.cfg, self.builder)
        self.plotter = SVRPlotter(self.cfg)

    def load_data(self):
        """Bước 2: dữ liệu và cách chia dùng chung của nhóm."""
        self.train, self.test = load_split()
        self.builder.check_no_leakage()
        self.X_tr, self.y_tr, self.g_tr = self.builder.training_matrix(self.train)
        return self.train, self.test

    def tune(self) -> dict:
        """Bước 4: chọn tham số (chỉ dùng train)."""
        self.tuner.grid_search(self.X_tr, self.y_tr, self.g_tr)
        self.params = self.tuner.select_one_se(self.train, self.X_tr, self.y_tr, self.g_tr)
        return self.params

    def fit_predict(self) -> pd.DataFrame:
        """Bước 5: train model cuối, dự báo E1/E2."""
        self.forecaster.fit(self.X_tr, self.y_tr, self.g_tr, self.params)
        self.pred = self.forecaster.predict(self.test)
        return self.pred

    def evaluate(self):
        """Bước 6: chấm điểm bằng evaluate() của nhóm, kèm baseline naive."""
        self.metrics, self.rul = evaluate(pd.concat([self.pred, naive(self.test)], ignore_index=True))
        return self.metrics, self.rul

    def pooled_metrics(self) -> pd.DataFrame:
        m = self.metrics[self.metrics.battery_id == "all"].drop(columns="battery_id")
        return m.set_index(["scenario", "model"]).sort_index()

    def per_battery_rmse(self) -> pd.DataFrame:
        m = self.metrics[self.metrics.battery_id != "all"]
        t = m.pivot_table(index=["scenario", "battery_id"], columns="model", values="RMSE")
        t["svr_better"] = t.svr < t.naive
        return t

    def explain(self) -> pd.DataFrame:
        """RQ4: permutation importance."""
        self.imp = self.importance.run(self.forecaster.model_, self.test)
        return self.imp

    def save(self) -> Path:
        """Lưu file dự báo và kiểm tra lại file đã lưu bằng evaluate() của nhóm."""
        self.pred.to_csv(self.cfg.pred_path, index=False)
        evaluate(pd.read_csv(self.cfg.pred_path))
        return self.cfg.pred_path

    def run(self):
        self.load_data()
        self.tune()
        self.fit_predict()
        self.evaluate()
        self.explain()
        self.plotter.tuning_heatmap(self.tuner)
        self.plotter.forecasts(self.test, self.pred)
        self.plotter.importance(self.imp)
        self.save()
        return self


if __name__ == "__main__":
    pd.set_option("display.width", 140)
    exp = SVRExperiment().run()
    print("tham số được chọn:", exp.params)
    print(exp.pooled_metrics().round(4))
    print(exp.rul.to_string(index=False))
    print(exp.imp.head(6).round(5).to_string(index=False))
    print("đã lưu", exp.cfg.pred_path.relative_to(ROOT).as_posix())
