"""Second pass: the extracts the first report was missing."""
import json
import polars as pl

URI = "s3://dsan6000-banking77-team/raw/**/*.parquet"
SO = {"aws_region": "us-east-1"}
lf = pl.scan_parquet(URI, hive_partitioning=True, storage_options=SO)
out = json.load(open("eda_out.json"))

# hour-of-day curve, in each branch's own local time
OFF = {"washington_dc": -5, "addis_ababa": 3, "yuzhno_sakhalinsk": 11}
hour = (lf.with_columns(utc_h=pl.col("received").dt.hour())
          .group_by("branch", "utc_h").agg(n=pl.len()).collect()
          .with_columns(local_h=(pl.col("utc_h") + pl.col("branch").replace_strict(OFF)) % 24)
          .sort("branch", "local_h"))
hour.write_parquet("fig_hour.parquet")

# duplication
uniq = lf.select(pl.col("text").n_unique()).collect().item()
out["unique_text"] = uniq
out["dup_factor"] = round(out["rows"] / uniq, 1)

# partition skew
part = lf.group_by("date", "branch").agg(n=pl.len()).collect()
part.write_parquet("fig_part.parquet")
out["part"] = {"n": part.height, "min": int(part["n"].min()),
               "median": int(part["n"].median()), "max": int(part["n"].max())}
out["part_skew"] = round(part["n"].max() / part["n"].min(), 1)

# vocabulary
toks = (lf.select(pl.col("text").str.to_lowercase().str.split(" ").alias("w"))
          .explode("w").filter(pl.col("w").str.len_chars() > 2)
          .group_by("w").agg(n=pl.len()).sort("n", descending=True).limit(15).collect())
toks.write_parquet("fig_tokens.parquet")
out["vocab_top"] = toks.to_dicts()[:6]

json.dump(out, open("eda_out.json", "w"), indent=1, default=str)
print("unique text:", format(uniq, ","), "| dup factor:", out["dup_factor"])
print("partition skew:", out["part_skew"], "x", out["part"])
