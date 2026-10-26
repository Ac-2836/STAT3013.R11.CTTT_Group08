# Processed data tables

These CSV files are produced from the 34 raw NASA battery files (`data/raw/*.mat`) by the scripts in `src/`. **Do not edit them by hand.** To rebuild them, put the raw files in `data/raw/` and run, from the repository root:

```bash
python src/build_dataset.py && python src/clean_dataset.py && python src/eda_dataset.py
```

The files are described below **in the order they are created**. Full rules and the reasons behind them: [`docs/Rec.md`](../../docs/Rec.md) section 2 and [`docs/Introduction_of_dataset.md`](../../docs/Introduction_of_dataset.md).

## Basic terms

| Term | Meaning |
|---|---|
| Cell | One rechargeable lithium-ion battery (type 18650, rated 2.0 Ah). There are 34 cells, `B0005` to `B0056`. |
| Group | Cells tested together under the same plan, `G1` to `G9`. G8 (`B0049`–`B0052`) is corrupted and excluded. |
| Operation | One action on a cell: a **charge**, a **discharge**, or an **EIS** measurement (see below). |
| Discharge / cycle | The cell powers a load until its voltage falls to a set limit (the **cutoff voltage**). In this project one discharge = one cycle. |
| Ah (ampere-hour) | Amount of charge. 1 Ah = a current of 1 A flowing for 1 hour. |
| Capacity | How many Ah the cell delivers in one discharge. It shrinks as the cell ages. |
| EIS | A measurement of the cell's internal resistance. Resistance grows as the cell ages. |
| SOH (State of Health) | Capacity now ÷ capacity when new. 1.0 = like new; **0.8 = end of life (EOL)**. |
| Empty cell | Value not available (`NaN`). True/False columns are flags. Times are ISO date-times. |

## Pipeline order

| Step | Script | Files written |
|---|---|---|
| 1. Read the raw files and summarise each operation | `build_dataset.py` | `discharge_ops.csv`, `charge_ops.csv`, `impedance_ops.csv` |
| 2. Attach to each discharge its earlier charge, EIS and cell information | `build_dataset.py` | `cycles.csv` |
| 3. Remove invalid discharges | `clean_dataset.py` | `cycles_dropped.csv` |
| 4. Compute SOH, add quality flags | `clean_dataset.py` | `cycles_clean.csv` |
| 5. Summarise and rank each cell | `clean_dataset.py` | `battery_summary.csv` |
| 6. Measure how fast each cell ages | `eda_dataset.py` | `battery_fade.csv` |

---

## 1. `discharge_ops.csv`: one row per discharge (2,794 rows)

Every discharge in the raw data, nothing removed yet. A sample is "loaded" when the measured current is below −0.1 A (current is negative while discharging).

| Column | Unit | Meaning |
|---|---|---|
| `battery_id` | – | Cell ID, e.g. `B0005`. |
| `op_index` | – | Position of this operation in the cell's raw file (0 = first). It orders all charges, discharges and EIS of a cell in time. |
| `ambient_C` | °C | Temperature of the test chamber during this discharge (4, 22, 24, 43 or 44). |
| `start_time` | date-time | When the discharge started. |
| `cycle` | – | Discharge number of the cell in the raw data, counting from 1. |
| `n_samples` | – | Number of measurement points recorded. |
| `capacity_Ah` | Ah | Capacity supplied by NASA. It counts only the charge delivered **until the voltage first drops below 2.7 V**, so it is far too low for cold, high-current tests. Kept for comparison only. Empty for 25 G8 discharges. |
| `capacity_to_2p7V_Ah` | Ah | Our recomputation of the same 2.7 V quantity. It matches `capacity_Ah` within 0.01 Ah for 2,752 of 2,794 discharges, which proves what `capacity_Ah` measures. |
| **`capacity_int_Ah`** | Ah | **Main capacity measure:** total charge the cell delivered down to its own cutoff voltage (current added up over time, loaded samples only). |
| `I_load_mean_A` | A | Average discharge current while the load was on. |
| `I_level_A` | A | `I_load_mean_A` rounded to 1, 2 or 4 A. For pulsed loads it is the pulse height. |
| `V_min` | V | Lowest voltage reached. |
| `duration_s` | s | Length of the recording. |
| `t_to_cutoff_s` | s | Time from the start until the load switched off (last loaded sample). |
| `energy_Wh` | Wh | Energy delivered (voltage × current added up over time, loaded samples only). |
| `T_start_C`, `T_max_C`, `T_mean_C` | °C | Temperature of the cell itself at the start, maximum and average. The cell heats up while it discharges. |

## 2. `charge_ops.csv`: one row per charge (2,815 rows)

Every charge follows the same recipe: a constant 1.5 A until the voltage reaches 4.2 V (the **CC phase**), then 4.2 V held while the current falls to 0.02 A.

| Column | Unit | Meaning |
|---|---|---|
| `battery_id`, `op_index`, `ambient_C`, `start_time` | | As in `discharge_ops.csv`, for this charge. |
| `chg_duration_s` | s | Length of the charge. |
| `chg_end_time` | date-time | `start_time` + `chg_duration_s`. |
| `chg_charge_Ah` | Ah | Charge put into the cell (only samples with current above 0.1 A, which skips start-up noise). |
| `chg_cc_time_s` | s | Length of the CC phase: time until the voltage first reaches 4.2 V. It gets shorter as the cell ages. Empty for 5 charges that never reached 4.2 V. |
| `chg_T_max_C` | °C | Highest cell temperature during the charge. |

## 3. `impedance_ops.csv`: one row per EIS measurement (1,956 rows)

| Column | Unit | Meaning |
|---|---|---|
| `battery_id`, `op_index`, `start_time` | | As above, for this measurement. |
| `ambient_C` | °C | Temperature recorded by NASA for the measurement. It reads 24 (43 for G3) even for cells tested at 4°C, so **do not use it**. |
| `Re_ohm` | Ω | Electrolyte resistance (resistance of the liquid inside the cell). |
| `Rct_ohm` | Ω | Charge-transfer resistance (resistance of the reactions at the electrodes). |

Values are **raw**. They are empty for 9 measurements where NASA's calculation failed (all G8). 14 impossible values (G8 only) are still present here; they are removed in `cycles_clean.csv`.

## 4. `cycles.csv`: one row per discharge, joined (2,794 rows)

All columns of `discharge_ops.csv`, plus the information below. **Not cleaned yet.**

| Column | Unit | Meaning |
|---|---|---|
| `group` | – | Experiment group, `G1` to `G9`. |
| `discharge_cutoff_V` | V | Voltage at which this cell's discharges stop (2.0–2.7 V). Lower = deeper discharge. |
| `profile` | – | `constant_current`, or `square_wave_0.05Hz` (load on 10 s, off 10 s; group G2 only). |
| `chg_op_index` | – | `op_index` of the most recent charge **before** this discharge. Empty for 23 discharges that have no earlier charge. |
| `chg_duration_s` … `chg_T_max_C` | | Values of that charge (see `charge_ops.csv`). |
| `rest_before_h` | h | Hours between the end of that charge and the start of this discharge. |
| `n_chg_since_prev_dis` | – | Number of charges between the previous discharge and this one. Normally 1. 2 = the matched charge may be a short top-up of a full cell. 0 = the cell was discharged twice in a row, so the matched charge is older. |
| `eis_op_index`, `Re_ohm`, `Rct_ohm` | –, Ω | The most recent EIS **before** this discharge (raw values). Empty for 76 discharges with no earlier EIS. |

Only earlier measurements are attached, never later ones, so no future information leaks into a row.

## 5. `cycles_dropped.csv`: the removed discharges (136 rows)

Same columns as `cycles.csv`, plus `drop_reason`. Rules are checked in this order and the first match is recorded:

| `drop_reason` | Rows | Why the discharge is not a valid capacity measurement |
|---|---|---|
| `excluded_battery` | 100 | Group G8: control software crashed, impossible values. |
| `no_preceding_charge` | 19 | The cell was not charged first (always the cell's first discharge). |
| `cutoff_not_reached` | 16 | Stopped early: `V_min` stayed more than 0.1 V above the cutoff. |
| `capacity_out_of_range` | 1 | `capacity_int_Ah` outside 0.05–2.1 Ah (B0036, cycle 114: 2.444 Ah). |

## 6. `cycles_clean.csv`: one row per valid discharge (2,658 rows) — **main table**

All columns of `cycles.csv` except `n_samples`, plus the columns below. `Re_ohm` and `Rct_ohm` are cleaned: values outside 0–0.5 Ω are replaced by the cell's last valid earlier value (no row changed in practice). They are empty for 57 rows (first 19 discharges of B0005, B0006, B0007, before any EIS).

| Column | Unit | Meaning |
|---|---|---|
| `cycle_valid` | – | Discharge number among the cell's valid discharges, counting from 1, with no gaps. **Use this as the time axis.** |
| `segment` | – | Stretch of consecutive discharges with the same `ambient_C` and `I_level_A`, numbered from 1. 23 cells have one segment; 7 cells (`B0038`–`B0044`) changed conditions and have 2–4. |
| `C0_Ah` | Ah | Capacity of the cell when new, for this segment: the highest 3-discharge rolling median within the segment's first 10 valid discharges. |
| `soh_segment` | – | `capacity_int_Ah ÷ C0_Ah`, for every cell. |
| **`soh`** | – | **State of Health**, same formula, only for cells with one segment (empty for `B0038`–`B0044`, whose capacity mixes ageing with changing test conditions). It can exceed 1 for a short time (36 rows). |
| `cap_outlier` | True/False | Suspicious capacity, judged within the same segment: it differs from the median of the 7 surrounding discharges by more than about 3 standard deviations and more than 0.05 Ah, **or** it is more than 25% below the best of the 10 previous discharges. 57 rows. Kept, not deleted. |
| `break_in` | True/False | One of the cell's first 10 valid discharges with capacity below 90% of `C0_Ah` (a new cell still settling). 17 rows. |
| `long_rest` | True/False | `rest_before_h` above 1 hour. 167 rows. Capacity is temporarily higher after a long rest. |
| `tier` | – | Usage group of the cell (see `battery_summary.csv`). |

The shared modelling data set is `tier == "A"` without `cap_outlier` and `break_in` rows ([`docs/Rec.md`](../../docs/Rec.md) section 6).

## 7. `battery_summary.csv`: one row per cell (34 rows)

| Column | Unit | Meaning |
|---|---|---|
| `battery_id`, `group`, `profile`, `discharge_cutoff_V` | | As above. |
| `ambient_C`, `I_level_A` | °C, A | Test conditions. Empty for cells with more than one segment. |
| `n_discharges_raw` | – | Discharges in the raw data. |
| `n_dropped` | – | Discharges removed (total 136). |
| `n_segments` | – | Number of segments. |
| `n_valid` | – | Valid discharges, not counting `break_in` rows. |
| `n_cap_outlier` | – | `cap_outlier` rows among them. |
| `outlier_share` | – | `n_cap_outlier ÷ n_valid`. |
| `n_break_in` | – | `break_in` rows. |
| `C0_Ah` | Ah | Capacity when new. Empty for cells with more than one segment. |
| `soh_min` | – | Lowest SOH, ignoring `break_in` and `cap_outlier` rows. Empty for cells with more than one segment. |
| `soh_last` | – | SOH of the last discharge, ignoring `break_in` rows (outliers **included**). Empty for cells with more than one segment. |
| **`tier`** | – | Usage group, decided in the order X → C → N → A → B (first match wins). |

| Tier | Rule | Cells | Use |
|---|---|---|---|
| X | Group G8 | 4 | Excluded (statistics columns empty) |
| C | More than one segment | 7 | Effect of changing temperature or current |
| N | More than 10% outliers | 1 (`B0054`) | Robustness checks only |
| A | ≥ 60 valid discharges and `soh_min` ≤ 0.85 | 12 | Forecasting models |
| B | All other cells | 10 | Comparing ageing speed between conditions |

## 8. `battery_fade.csv`: ageing speed per cell (23 rows)

Cells of tiers A, B and N (constant conditions). Calculated without `cap_outlier` and `break_in` rows. Statistical tests use these rows, because discharges of the same cell are not independent.

| Column | Unit | Meaning |
|---|---|---|
| `battery_id`, `group`, `tier`, `ambient_C`, `I_level_A`, `profile`, `discharge_cutoff_V` | | As above. |
| `n_used` | – | Number of rows used for `fade_pct_per_100cyc`. |
| `fade_pct_per_100cyc` | SOH points per 100 cycles | Ageing speed over the whole test: slope of a straight line fitted to `soh` against `cycle_valid`. 20 means the cell loses 0.20 SOH (e.g. 1.00 → 0.80) every 100 discharges. |
| `n_used25` | – | Number of rows used for `fade25_pct_per_100cyc` (can be below 25 because flagged rows are removed). |
| `fade25_pct_per_100cyc` | SOH points per 100 cycles | Same, using only `cycle_valid` ≤ 25, a window every cell has. |
| `cycles_to_soh80` | – | First `cycle_valid` with `soh` below 0.8 (end of life). Empty if the cell never got there (all tier B cells and `B0056`). |
| `soh_last` | – | SOH of the last discharge, ignoring outliers too (differs from `soh_last` in `battery_summary.csv`). |
| `T_max_med` | °C | Median of `T_max_C`: how hot the cell typically got during discharge. |
