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
git clone https://github.com/Ac-2836/4AnhTai---Statistical-Analytics-.git
cd 4AnhTai---Statistical-Analytics-
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

Download the raw dataset from the Google Drive link in Section 3 (also in `data/raw/README.md`) and put the files in `data/raw/`. The `data/` folder is git-ignored, so data files are never committed.

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

- Never commit directly to `main`.
- Never commit anything inside `data/`.
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
| `data` | change data processing rules or thresholds (code only, never data files) | `data(preprocess): exclude G8 cells` |
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
| `regression` | Linear and logistic regression |
| `arima` | ARIMA / ARIMAX |
| `ml` | SVR, Random Forest, Isolation Forest |
| `dl` | GRU / LSTM |
| `sem` | SEM / path analysis |
| `compare` | Model comparison |
| `report`, `paper`, `slides` | The three written deliverables |
| `worklog`, `wbs`, `readme` | Team documents |
| `repo` | Repository setup |

### 10.5 Never

- Commit files inside `data/` or any raw or processed dataset.
- Commit directly to `main`, or push to another member's branch.
- Rewrite shared history (`git push --force`, `git commit --amend` on pushed commits) without the leader's agreement.
- Commit numbers, tables or figures that were typed by hand instead of produced by committed code.
