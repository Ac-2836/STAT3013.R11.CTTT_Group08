# Lithium-Ion Battery Degradation and SOH Prognostics under Variable Operating Conditions

**Course project, STAT3013 (Intermediate Statistical Analysis), class STAT3013.R11.CTTT, Group 08 (4AnhTai)**
*Statistical and Machine Learning-Based Modeling of Lithium-Ion Battery Degradation and State-of-Health Prognostics under Variable Operating Conditions*

## 1. Project overview

Battery lifetime is a central business problem for electric-vehicle (EV) and drone fleets and for warranty providers: a pack that degrades faster than expected costs money, and a pack replaced too early wastes it. This project studies how the **capacity of lithium-ion cells fades** and which **operating conditions drive that fade** (ambient temperature, discharge current, discharge depth, rest time, internal resistance), then **predicts State of Health (SOH)** and the point where a cell reaches end of life. The modeling results are turned into practical recommendations for companies (warranty policy, fleet maintenance) and for users (safe operating habits).

SOH is defined as `SOH = C / C0`, the current discharge capacity divided by the cell's own capacity when new. End of life (EOL) is taken at SOH = 0.8 (20% capacity loss).

### Research questions

| RQ | Question | Method |
|---|---|---|
| RQ1 | Does the fade rate differ significantly across temperature, current and discharge profile? | Inferential statistics |
| RQ2 | How much SOH does each factor cost per cycle? | Multiple regression |
| RQ3 | How well can SOH be forecast from a cell's own history? | ARIMA / ARIMAX |
| RQ4 | How well do charge and impedance features predict the next-cycle SOH? | SVR |
| RQ5 | Does a sequence model beat classical ML on this small dataset? | LSTM |
| RQ6 | Do operating conditions act on SOH through internal resistance? | SEM / path analysis |

## 2. Expected outputs

- **Full report** (about 30 pages) and **IEEE conference paper** (7 to 10 pages).
- **Presentation slides** and a **demo video**, plus a small **Streamlit app** (low priority, built after the core work).
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

0. **Preprocessing**: parse, join and clean the raw files, compute SOH, flag outliers (Isolation Forest is used only as a cross-check of the outlier flags).
1. **Descriptive**: distributions, boxplots and SOH curves by operating condition.
2. **Diagnostic / inferential**: confidence intervals, t-test, F-test, Levene, one-way ANOVA with Tukey HSD, chi-square. The unit of observation is the **cell**, not the cycle, because cycles of one cell are strongly autocorrelated.
3. **Predictive** (four algorithms, one per team member, same data split and metrics):

   | Type | Algorithm | Owner |
   |---|---|---|
   | Traditional statistics | Multiple log-linear regression with interaction terms | Quoc Dat |
   | Traditional statistics | ARIMA / ARIMAX | Tien Hung |
   | Machine Learning | SVR (Support Vector Regression) | Minh Vu |
   | Deep Learning | LSTM | Khai Hung |

4. **Structural**: SEM / path analysis, operating conditions -> impedance (Re, Rct) -> SOH.
5. **Prescriptive**: recommendations for warranty, fleet maintenance and users.

**Evaluation protocol:** train/test split by cell (never by cycle), one test cell per operating condition, MAE / RMSE / MAPE / R^2 against a naive persistence baseline, plus RUL error. The sensitivity and specificity of an "EOL within 20 cycles" warning are derived from each model's forecasts. Details are in [`docs/Rec.md`](docs/Rec.md) and [`docs/Go.md`](docs/Go.md).

## 5. Team

| Member | Role |
|---|---|
| Khai Hung | Leader, Deep Learning (LSTM), preprocessing, evaluation protocol, model comparison |
| Quoc Dat | Traditional statistical models (multiple regression, SEM) |
| Tien Hung | Traditional statistical models (inferential statistics, ARIMA / ARIMAX) |
| Minh Vu | Machine Learning (SVR), descriptive statistics |

## 6. Project structure

```
.
├── README.md
├── requirements.txt                    # all Python dependencies
├── requirements-regression-sem.lock.txt # exact versions used for regression and SEM
├── .gitignore
├── data/
│   ├── raw/                            # git-ignored, download from Drive
│   └── processed/                      # committed CSV tables
├── src/
│   ├── build_dataset.py   clean_dataset.py   eda_dataset.py      # preprocessing
│   ├── make_report_figures.py                                    # fig1-fig6
│   ├── protocol.py                                               # shared split and scoring
│   ├── descriptive_stats.py   iforest_check.py                   # descriptive, outlier cross-check
│   ├── regression_analysis.py   sem_analysis.py   regression_sem_utils.py   run_regression_sem.py
│   ├── svr_model.py   lstm_model.py
│   └── compare_models.py                                         # model comparison, EOL alarm
├── notebooks/
│   ├── 01_descriptive.ipynb   02_inferential.ipynb
│   ├── 10_regression.ipynb   20_arima.ipynb   30_ml.ipynb   31_iforest.ipynb
│   ├── 40_dl.ipynb   50_sem.ipynb   60_compare.ipynb
├── tests/
│   └── test_regression_sem.py
├── reports/
│   ├── figures/
│   ├── tables/
│   └── predictions/
├── docs/
│   ├── Introduction_of_dataset.md   Rec.md   Go.md   WBS.md   Work_log.md
│   └── Regression_Report.md   SEM_Report.md   LSTM_Report.md
└── deliverables/
    ├── report/   paper/   slides/   demo/   evidence/
```

| Path | What it is |
|---|---|
| `README.md` | This file: project summary and structure. |
| `requirements.txt` | Python dependencies (`pip install -r requirements.txt`). |
| `.gitignore` | Excludes the raw data in `data/raw/` (only its README is kept); processed tables are tracked. |
| `data/` | Data folder. `data/raw/` is **git-ignored** and never committed; `data/processed/` **is committed**. |
| `data/raw/` | Local copy of the original dataset files (34 `.mat` files plus the dataset's own `metadata.csv` and READMEs). Download link: see Section 3. |
| `data/processed/` | CSV tables generated by `src/`, committed so every member uses the same tables. Regenerate them with the scripts, never edit them by hand. |
| `src/` | Shared pipeline and model code. Preprocessing runs in order: build, clean, eda. |
| `src/build_dataset.py` | Parses the `.mat` files into one row per charge, discharge and impedance operation, then joins each discharge with its preceding charge and latest earlier impedance (`cycles.csv`). |
| `src/clean_dataset.py` | Drops invalid rows, splits batteries into operating-condition segments, computes C0 and SOH, flags outliers, assigns each cell a usage tier. Produces `cycles_clean.csv` and `battery_summary.csv`. |
| `src/eda_dataset.py` | Per-cell fade rates (`battery_fade.csv`). |
| `src/make_report_figures.py` | Dataset figures `fig1`–`fig6` from `data/processed/`. |
| `src/protocol.py` | Shared train/test split (`load_split`), naive baseline and scoring (`check_pred`, `evaluate`) so all models are compared fairly. |
| `src/descriptive_stats.py`, `src/iforest_check.py` | Descriptive statistics; Isolation Forest cross-check of the outlier flags. |
| `src/regression_analysis.py`, `src/sem_analysis.py`, `src/regression_sem_utils.py`, `src/run_regression_sem.py` | Log-linear regression and SEM; `run_regression_sem.py` executes both notebooks. |
| `src/svr_model.py`, `src/lstm_model.py` | SVR and LSTM models. |
| `src/compare_models.py` | Scores every prediction file, RUL errors and the EOL-alarm sensitivity/specificity. |
| `tests/` | Automated checks for regression and SEM (`python -m unittest discover -s tests`). |
| `reports/tables/` | CSV tables produced by the notebooks and scripts. |
| `notebooks/` | One notebook per analysis step, runnable top to bottom: `01` descriptive, `02` inferential, `10` regression, `20` ARIMA, `30` SVR, `31` Isolation Forest, `40` LSTM, `50` SEM, `60` model comparison. |
| `reports/figures/` | Figures used in the report, paper and slides. |
| `reports/predictions/` | One prediction CSV per model in a common format, read by the comparison notebook. |
| `docs/` | Project documentation. |
| `docs/Introduction_of_dataset.md` | Full dataset description, quality findings, cleaning rules and data dictionary. |
| `docs/Rec.md` | Record of all locked team decisions (Vietnamese). The reference when other documents disagree. |
| `docs/Work_log.md` | Detailed log of each member's work, the main source for writing the report and paper (Vietnamese). |
| `docs/Go.md` | Step-by-step implementation plan (Vietnamese, team use). |
| `docs/Regression_Report.md`, `docs/SEM_Report.md`, `docs/LSTM_Report.md` | Draft methods and results text of these models, for the report and paper. |
| `docs/WBS.md` | Work breakdown, roles and schedule (Vietnamese, team use). |
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
python src/build_dataset.py          # 1. preprocessing: data/processed/*.csv
python src/clean_dataset.py
python src/eda_dataset.py
python src/make_report_figures.py    # 2. dataset figures fig1-fig6
python src/run_regression_sem.py     # 3. regression and SEM notebooks
python src/compare_models.py         # 4. model comparison tables and figures
python -m unittest discover -s tests # 5. checks (set RUN_SLOW=1 for the slow end-to-end tests)
```

The other notebooks (`01`, `02`, `20`, `30`, `31`, `40`, `60`) run top to bottom from the `notebooks/` folder. Steps 2-5 only need `data/processed/`, which is already in the repository.

## 8. Reference

B. Saha and K. Goebel, "Battery Data Set", NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA, 2007.

## 9. Getting started for team members

Each member works on their **own branch**. `main` only receives work through reviewed Pull Requests.

| Member | Branch |
|---|---|
| Khai Hung | `khai-hung` |
| Minh Vu | `minh-vu` |
| Quoc Dat | `quoc-dat` |
| Tien Hung | `tien-hung` |

### 9.1 Clone the repository

```bash
git clone https://github.com/Ac-2836/STAT3013.R11.CTTT_Group08.git
cd STAT3013.R11.CTTT_Group08
```

### 9.2 Check the available branches

```bash
git fetch origin
git branch -a
```

The list should contain `remotes/origin/khai-hung`, `remotes/origin/minh-vu`, `remotes/origin/quoc-dat` and `remotes/origin/tien-hung`. If one is missing, run `git fetch origin` again or contact the leader.

### 9.3 Switch to your own branch

Replace `minh-vu` with your branch name from the table above:

```bash
git switch minh-vu
```

Git creates the local branch and links it to `origin/minh-vu` automatically.

### 9.4 Verify you are on the right branch

```bash
git branch --show-current
git status
```

- The first command must print **your** branch name.
- `git status` must say `Your branch is up to date with 'origin/<your-branch>'`.

Check this every time before you start working. Committing on the wrong branch is the most common mistake.

### 9.5 Set up the environment and data

```bash
pip install -r requirements.txt
```

Download the raw dataset from the Google Drive link in Section 3 (also in `data/raw/README.md`) and put the files in `data/raw/`. Raw files in `data/raw/` are git-ignored, so they are never committed. The processed tables in `data/processed/` are already in the repository, so you only need the raw files to rerun the pipeline.

### 9.6 Daily workflow

1. Get the latest changes of your branch before you start:
   ```bash
   git pull
   ```
2. Bring in new work merged into `main` by others:
   ```bash
   git fetch origin
   git merge origin/main
   ```
3. Do your work. Record what you did, why, and the results in **your own section** of [`docs/Work_log.md`](docs/Work_log.md), following the NOTICE in that file.
4. Stage only the files you changed, by name. Avoid `git add .`:
   ```bash
   git add path/to/file1 path/to/file2
   git commit -m "short description of the change"
   git push
   ```
5. When a piece of work is finished, open a **Pull Request** from your branch into `main` on GitHub and ask your review partner (see `docs/WBS.md`) to review it. The leader merges after approval.

### 9.7 Rules

- Never commit directly to `main`. (Exception, 21/11/2026: the leader pushed the regenerated `fig1`–`fig6` directly, commit `9036006`.)
- Never commit raw data in `data/raw/` (only its `README.md`). Commit `data/processed/` only as produced by the scripts in `src/`.
- Only edit your own section of `docs/Work_log.md`, to avoid merge conflicts.
- Every number in the report must come from code or a notebook committed in this repository.

## 10. Commit rules

These rules are **mandatory** for every member. They keep the history readable, avoid merge conflicts, and make it possible to trace every number in the report back to the code that produced it.

### 10.1 Required workflow for every change

1. **Be on your own branch.** Run `git branch --show-current` and check the name (Section 9.4).
2. **Sync first.** `git pull`, then `git fetch origin` and `git merge origin/main`.
3. **Make one logical change.** One commit = one purpose (for example "add ADF test", not "ADF test + fix figure + update README").
4. **Run it.** Scripts must run without errors; notebooks must run top to bottom (Restart & Run All).
5. **Log it.** Add an entry to your section of `docs/Work_log.md` for any analysis step, decision or result.
6. **Stage by name.** `git add path/to/file`. Never `git add .` or `git add -A`. Check with `git status` before committing.
7. **Write the message** using the format and table below.
8. **Push** to your own branch: `git push`.
9. **Open a Pull Request** into `main` when the piece of work is complete. Your review partner reviews it; the leader merges it.
10. **After a merge into `main`**, every member runs `git fetch origin` and `git merge origin/main` on their branch.

### 10.2 Commit message format

```
<type>(<scope>): <summary>
```

- **type**: one value from the table in 10.3.
- **scope**: the part of the project the change touches (see 10.4).
- **summary**: in English, imperative mood ("add", not "added"), lowercase, no full stop at the end, at most 72 characters.
- Optional body after one blank line: explain **why**, not what.
- Do **not** add AI attribution lines (for example `Co-Authored-By: <AI tool>`).

### 10.3 Commit types

| Type | Use when you… | Example |
|---|---|---|
| `feat` | add a new analysis, model, test or script | `feat(arima): add ARIMAX with rest-time regressor` |
| `fix` | fix a bug or a wrong result | `fix(preprocess): use preceding charge only in join` |
| `data` | change data processing rules or thresholds (code plus the regenerated `data/processed/` tables, never raw files) | `data(preprocess): exclude G8 cells` |
| `fig` | add or update figures | `fig(eda): add SOH curves by ambient temperature` |
| `report` | write or edit report, paper or slide content | `report(paper): draft methodology section` |
| `docs` | update documentation: README, Work log, WBS, plans | `docs(worklog): add preprocessing entry` |
| `refactor` | restructure code without changing results | `refactor(ml): move feature building into a function` |
| `test` | add or fix checks and assertions | `test(preprocess): assert SOH within plausible range` |
| `chore` | change setup: dependencies, `.gitignore`, folders | `chore(repo): add scikit-learn to requirements` |

### 10.4 Scopes

| Scope | Covers |
|---|---|
| `preprocess` | Parsing, joining, cleaning, SOH computation |
| `eda` | Descriptive statistics and exploratory figures |
| `inference` | Confidence intervals, t-test, ANOVA, chi-square |
| `regression` | Multiple log-linear regression |
| `arima` | ARIMA / ARIMAX |
| `ml` | SVR (and Isolation Forest data checks) |
| `dl` | LSTM |
| `sem` | SEM / path analysis |
| `compare` | Model comparison |
| `report`, `paper`, `slides` | The three written deliverables |
| `worklog`, `wbs`, `readme` | Team documents |
| `repo` | Repository setup |

### 10.5 Never

- Commit raw data files (anything in `data/raw/` except its `README.md`), or edit the `data/processed/` tables by hand.
- Commit directly to `main`, or push to another member's branch.
- Rewrite shared history (`git push --force`, `git commit --amend` on pushed commits) without the leader's agreement.
- Commit numbers, tables or figures that were typed by hand instead of produced by committed code.
