# Catching Fraud in the Chat Box

A continuously-updated intent-classification and fraud-review pipeline for
Banking77. DSAN 6000 (Big Data and Cloud Computing), Georgetown University,
Fall 2026.

**Team:** Yixuan He, Lanbin Fan, Zizheng Wang, Ken Qin

## Problem

Banking77 gives us 13,083 labelled customer-chat messages across 77 intents,
but a bank's real message stream is mostly *unreviewed*: `label = NA`. This
project builds an AWS prototype that ranks unreviewed messages for a fraud
reviewer's attention and folds completed reviews back into the next round of
intent training.

## Repository layout

| Path | Contents |
|---|---|
| `scripts/generate_log.py` | Builds the simulated production log from Banking77 |
| `notebooks/` | EDA and querying (Polars, DuckDB, Athena) |
| `report/` | Quarto manuscript for the final report |
| `figures/` | Figures exported by the notebooks |

## The simulated log

Banking77's raw files are about 1 MB, so the production message log is
simulated: the 13,083 messages are replayed with light perturbation across
three branches (Washington DC, Addis Ababa, Yuzhno-Sakhalinsk), each with its
own local business-hours traffic curve.

```bash
python scripts/generate_log.py --rows 50000000 --days 90 --out log
```

Output is Hive-partitioned Parquet, `date=YYYY-MM-DD/branch=<name>/`:

| Column | Type | Notes |
|---|---|---|
| `msg_id` | string | Stable ID, so a later review can be matched back |
| `label` | int32 | Intent code 0-76; **null** when not yet reviewed |
| `text` | string | Message body |
| `received` | timestamp | UTC |
| `injected_type` | string | Ground truth for evaluation only — null for ordinary rows |

About 5% of rows carry a label. `injected_type` marks the three patterns
planted for evaluation: `token_flooding` (messages over 999 characters),
`identity_fail_transfer` (identity-check failures followed by a transfer
request), and `drift_vocab` (a vocabulary shift on the
`supported_cards_and_currencies` intent that ramps in over the final third of
the window).

Data is never committed to this repository; see `.gitignore`.
