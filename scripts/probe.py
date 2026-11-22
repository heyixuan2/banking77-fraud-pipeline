import polars as pl
URI = "s3://dsan6000-banking77-team/raw/**/*.parquet"
SO = {"aws_region": "us-east-1"}
lf = pl.scan_parquet(URI, hive_partitioning=True, storage_options=SO)

uniq = lf.select(pl.col("text").n_unique()).collect().item()
tot = lf.select(pl.len()).collect().item()
print("唯一文本数 :", format(uniq, ","))
print("总行数     :", format(tot, ","))
print("平均重复   :", round(tot / uniq, 1), "次")

d = lf.group_by("date", "branch").agg(n=pl.len()).collect()
print("分区行数   : min", format(d["n"].min(), ","),
      "| median", format(int(d["n"].median()), ","),
      "| max", format(d["n"].max(), ","))

h = lf.with_columns(utc_h=pl.col("received").dt.hour()).group_by("branch", "utc_h").agg(n=pl.len()).collect()
for b in ["washington_dc", "addis_ababa", "yuzhno_sakhalinsk"]:
    s = h.filter(pl.col("branch") == b).sort("n", descending=True)
    print("  %-20s UTC 峰值 %2d:00" % (b, s["utc_h"][0]))
