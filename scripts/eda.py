"""EDA over the simulated Banking77 production log, read straight from S3."""
import json
import polars as pl

URI = "s3://dsan6000-banking77-team/raw/**/*.parquet"
SO = {"aws_region": "us-east-1"}

lf = pl.scan_parquet(URI, hive_partitioning=True, storage_options=SO)
out = {}

# scale
out["rows"] = lf.select(pl.len()).collect().item()
out["cols"] = len(lf.collect_schema())

# label coverage
cov = lf.select(
    labelled=pl.col("label").is_not_null().sum(),
    total=pl.len(),
).collect()
out["labelled"] = cov["labelled"][0]
out["labelled_pct"] = round(100 * cov["labelled"][0] / cov["total"][0], 2)

# per-branch volume
out["by_branch"] = lf.group_by("branch").agg(
    n=pl.len(), labelled=pl.col("label").is_not_null().sum()
).sort("branch").collect().to_dicts()

# daily volume (for the time-series figure)
daily = lf.group_by("date", "branch").agg(n=pl.len()).sort("date", "branch").collect()
daily.write_parquet("fig_daily.parquet")
out["days"] = daily["date"].n_unique()

# message length
ln = lf.select(L=pl.col("text").str.len_chars()).collect()["L"]
out["len"] = {k: float(v) for k, v in
              zip(["min", "p50", "p90", "p99", "max", "mean"],
                  [ln.min(), ln.median(), ln.quantile(.9), ln.quantile(.99), ln.max(), ln.mean()])}
out["over_999"] = int((ln > 999).sum())
ln.to_frame("len_chars").sample(300_000, seed=1).write_parquet("fig_len.parquet")

# injected ground truth
out["injected"] = lf.group_by("injected_type").agg(n=pl.len()).sort("n", descending=True).collect().to_dicts()

# intent imbalance on the labelled subset
intents = pl.read_csv("intents.csv")
top = (lf.filter(pl.col("label").is_not_null()).group_by("label").agg(n=pl.len())
         .collect().join(intents, on="label").sort("n", descending=True))
top.write_parquet("fig_intents.parquet")
out["n_intents_seen"] = top.height
out["intent_top5"] = top.head(5).to_dicts()
out["intent_bottom5"] = top.tail(5).to_dicts()
out["imbalance_ratio"] = round(top["n"][0] / top["n"][-1], 2)

# drift: share of intent-54 rows carrying the new vocabulary, by week
drift = (lf.filter(pl.col("label") == 54)
           .with_columns(week=pl.col("date").dt.truncate("1w"))
           .group_by("week").agg(
               n=pl.len(),
               drifted=(pl.col("injected_type") == "drift_vocab").sum())
           .sort("week").collect()
           .with_columns(pct=(100 * pl.col("drifted") / pl.col("n")).round(2)))
drift.write_parquet("fig_drift.parquet")
out["drift_by_week"] = drift.to_dicts()

print(json.dumps(out, indent=1, default=str))
