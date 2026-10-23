# Lithium-Ion Battery Degradation and SOH Prognostics under Variable Operating Conditions

**Course project, STAT3013 (Intermediate Statistical Analysis), Group 4AnhTai**
*Statistical and Machine Learning-Based Modeling of Lithium-Ion Battery Degradation and State-of-Health Prognostics under Variable Operating Conditions*

## 1. Project overview

Battery lifetime is a central business problem for electric-vehicle (EV) and drone fleets and for warranty providers: a pack that degrades faster than expected costs money, and a pack replaced too early wastes it. This project studies how the **capacity of lithium-ion cells fades** and which **operating conditions drive that fade** (ambient temperature, discharge current, discharge depth, rest time, internal resistance), then **predicts State of Health (SOH)** and the point where a cell reaches end of life. The modeling results are turned into practical recommendations for companies (warranty policy, fleet maintenance) and for users (safe operating habits).

SOH is defined as `SOH = C / C0`, the current discharge capacity divided by the cell's own capacity when new. End of life (EOL) is taken at SOH = 0.8 (20% capacity loss).

### Research questions

| RQ | Question | Method |
|---|---|---|
| RQ1 | Does the fade rate differ significantly across temperature, current and discharge profile? | Inferential statistics |
| RQ2 | How much SOH does each factor cost per cycle? | Multiple regression |
| RQ3 | How well can capacity be forecast from a cell's own history? | ARIMA / ARIMAX |
| RQ4 | How well do charge and impedance features predict the next-cycle SOH? | SVR / Random Forest |
| RQ5 | Does a sequence model beat classical ML on this small dataset? | GRU / LSTM |
| RQ6 | Do operating conditions act on SOH through internal resistance? | SEM / path analysis |

## 2. Expected outputs

- **Full report** (about 30 pages) and **IEEE conference paper** (7 to 10 pages).
- **Presentation slides** and a **demo video**.
- **This repository**: reproducible pipeline from raw files to figures and model predictions.
- **Plagiarism (Turnitin <= 10%) and AI-content check evidence.**
- Findings and a model comparison table, translated into recommendations for warranty design, fleet management and users.

## 3. Dataset

**NASA PCoE Li-ion Battery Aging Dataset** (B. Saha and K. Goebel, 2007, NASA Ames Prognostics Data Repository).

| Item | Value |
|---|---|
| Cells | 34 commercial 18650 cells, rated 2.0 Ah, in 9 experiment groups (G1 to G9) |
| Period | April 2008 to September 2010 |
| Raw size | 2,815 charge, 2,794 discharge and 1,956 impedance (EIS) operations, about 7.3 million measurements |
| Conditions | Ambient 4, 22 to 24 and 43 to 44 C; discharge current 1, 2 and 4 A; constant-current and square-wave loads; cutoff voltage 2.0 to 2.7 V |
| Target | Discharge capacity of every cycle, so SOH is measured directly |
| After cleaning | 2,658 valid discharge cycles from 30 cells (4 cells excluded as corrupted) |

Raw files are not stored in git. Download them from the [project Google Drive](https://drive.google.com/drive/folders/1w9p1saNKsAFjhIGZ5JKSlOro6wuLO4OP?usp=sharing) into `data/raw/`. The full dataset description, data quality findings and cleaning rules are in [`docs/Introduction_of_dataset.md`](docs/Introduction_of_dataset.md).

## 4. Research methodology

The workflow follows the four analytics levels taught in the course:

1. **Descriptive**: distributions, boxplots and SOH curves by operating condition.
2. **Diagnostic / inferential**: confidence intervals, t-test, F-test, Levene, one-way ANOVA with Tukey HSD, chi-square. The unit of observation is the **cell**, not the cycle, because cycles of one cell are strongly autocorrelated.
3. **Predictive** (one algorithm per team member, same data split and metrics):
   - Multiple log-linear regression with interaction terms (optional extension: logistic regression for "SOH below 0.8 within 20 cycles").
   - ARIMA / ARIMAX.
   - SVR / Random Forest, with Isolation Forest for anomaly checks.
   - GRU / LSTM.
4. **Structural**: SEM / path analysis, operating conditions -> impedance (Re, Rct) -> SOH.
5. **Prescriptive**: recommendations for warranty, fleet maintenance and users.

**Evaluation protocol:** train/test split by cell (never by cycle), one test cell per operating condition, MAE / RMSE / MAPE / R^2 against a naive persistence baseline, plus RUL error. Details are in [`docs/Go.md`](docs/Go.md).

## 5. Team

| Member | Role |
|---|---|
| Khai Hung | Leader, Deep Learning (GRU / LSTM) |
| Quoc Dat | Traditional statistical models (regression, SEM) |
| Tien Hung | Traditional statistical models (inference, ARIMA / ARIMAX) |
| Minh Vu | Machine Learning (SVR / Random Forest) |

## 6. Project structure

`[planned]` marks files and folders that do not exist yet. Folders in the tree are created empty.

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── build_dataset.py
│   ├── clean_dataset.py
│   ├── eda_dataset.py
│   └── protocol.py                [planned]
├── notebooks/
├── reports/
│   ├── figures/
│   └── predictions/
├── docs/
│   ├── Introduction_of_dataset.md
│   ├── Go.md
│   ├── WBS.md
│   ├── requirements_text.txt
│   └── [Project] STAT3013_Project_Requirements (2026) (1).pdf
└── deliverables/
    ├── report/
    ├── paper/
    ├── slides/
    ├── demo/
    └── evidence/
```

| Path | What it is |
|---|---|
| `README.md` | This file: project summary and structure. |
| `requirements.txt` | Python dependencies (`pip install -r requirements.txt`). |
| `.gitignore` | Excludes `data/` so large data files are never pushed. |
| `data/` | Local data, **git-ignored**, never committed. |
| `data/raw/` | Local copy of the original dataset files (34 `.mat` files plus the dataset's own `metadata.csv` and READMEs). Download link: see Section 3. |
| `data/processed/` | CSV tables generated by `src/`, safe to delete and regenerate. |
| `src/` | Shared data pipeline used by the whole team. Run in order: build, clean, eda. |
| `src/build_dataset.py` | Parses the `.mat` files into one row per charge, discharge and impedance operation, then joins each discharge with its preceding charge and latest earlier impedance (`cycles.csv`). |
| `src/clean_dataset.py` | Drops invalid rows, splits batteries into operating-condition segments, computes C0 and SOH, flags outliers, assigns each cell a usage tier. Produces `cycles_clean.csv` and `battery_summary.csv`. |
| `src/eda_dataset.py` | Descriptive statistics, per-cell fade rates (`battery_fade.csv`), ADF stationarity tests and the report figures. |
| `src/protocol.py` | `[planned]` Shared train/test split and evaluation function so all models are compared fairly. |
| `notebooks/` | One notebook per analysis step, runnable top to bottom: descriptive, inferential, regression, ARIMA, ML, DL, SEM, model comparison. |
| `reports/figures/` | Figures used in the report, paper and slides. |
| `reports/predictions/` | One prediction CSV per model in a common format, read by the comparison notebook. |
| `docs/` | Project documentation. |
| `docs/Introduction_of_dataset.md` | Full dataset description, quality findings, cleaning rules and data dictionary. |
| `docs/Go.md` | Step-by-step implementation plan (Vietnamese, team use). |
| `docs/WBS.md` | Work breakdown, roles and schedule (Vietnamese, team use). |
| `docs/[Project] STAT3013_Project_Requirements (2026) (1).pdf` | Official course project requirements. |
| `docs/requirements_text.txt` | Plain-text copy of the requirements PDF. |
| `deliverables/` | Final submission material. |
| `deliverables/report/` | Full report (about 30 pages). |
| `deliverables/paper/` | IEEE conference paper (7 to 10 pages). |
| `deliverables/slides/` | Presentation slides. |
| `deliverables/demo/` | Demo video (or its link). |
| `deliverables/evidence/` | Turnitin and AI-content check screenshots, AI usage statement. |

### Processed tables (`data/processed/`)

| File | One row is | Use |
|---|---|---|
| `discharge_ops.csv`, `charge_ops.csv`, `impedance_ops.csv` | One operation | Intermediate tables |
| `cycles.csv` | One discharge, joined, **not cleaned** | Input of the cleaning step |
| `cycles_clean.csv` | One valid discharge with SOH, segment and quality flags | **Main table for analysis and modeling** |
| `battery_summary.csv` | One cell | Conditions, C0, final SOH, usage tier |
| `battery_fade.csv` | One cell (23 cells with constant conditions) | Unit of observation for inferential tests |

## 7. How to reproduce

```bash
pip install -r requirements.txt
# download the raw .mat files from the Drive link in Section 3 into data/raw/
python src/build_dataset.py
python src/clean_dataset.py
python src/eda_dataset.py
```

## 8. Reference

B. Saha and K. Goebel, "Battery Data Set", NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA, 2007.
