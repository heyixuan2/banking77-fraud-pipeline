"""Run one logical query three ways and record cost."""
import json, subprocess, time
import duckdb, polars as pl

S3 = "s3://dsan6000-banking77-team/raw"
SO = {"aws_region": "us-east-1"}
LO, HI = "2026-11-01", "2026-11-21"
OUT = "s3://dsan6000-yh973/athena-results/"
rows = []


def athena(sql, label):
    t = time.time()
    qid = subprocess.run(["aws", "athena", "start-query-execution", "--query-string", sql,
                          "--result-configuration", f"OutputLocation={OUT}",
                          "--query-execution-context", "Database=dsan6000",
                          "--query", "QueryExecutionId", "--output", "text"],
                         capture_output=True, text=True).stdout.strip()
    while True:
        r = subprocess.run(["aws", "athena", "get-query-execution", "--query-execution-id", qid,
                            "--query", "QueryExecution.[Status.State,Statistics.DataScannedInBytes]",
                            "--output", "text"], capture_output=True, text=True).stdout.split()
        if r and r[0] in ("SUCCEEDED", "FAILED", "CANCELLED"):
            break
        time.sleep(1)
    el = time.time() - t
    scanned = int(r[1]) / 1e6 if len(r) > 1 and r[1] != "None" else None
    rows.append({"engine": "Athena", "query": label, "seconds": round(el, 2), "scanned_mb": scanned})
    print(f"  Athena  {label:<14} {el:6.2f}s  scanned {scanned:8.1f} MB")


def polars_q(label, pruned):
    t = time.time()
    lf = pl.scan_parquet(f"{S3}/**/*.parquet", hive_partitioning=True, storage_options=SO)
    if pruned:
        lf = lf.filter(pl.col("date").is_between(pl.lit(LO).str.to_date(), pl.lit(HI).str.to_date()))
    lf.group_by("branch").agg(n=pl.len(), reviewed=pl.col("label").is_not_null().sum()).collect()
    el = time.time() - t
    rows.append({"engine": "Polars", "query": label, "seconds": round(el, 2), "scanned_mb": None})
    print(f"  Polars  {label:<14} {el:6.2f}s")


def duck_q(label, pruned):
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("CREATE OR REPLACE SECRET s (TYPE s3, PROVIDER credential_chain, REGION 'us-east-1');")
    where = f"WHERE date BETWEEN DATE '{LO}' AND DATE '{HI}'" if pruned else ""
    t = time.time()
    con.execute(f"""SELECT branch, count(*) n, count(label) reviewed
                    FROM read_parquet('{S3}/**/*.parquet', hive_partitioning=true)
                    {where} GROUP BY branch""").fetchall()
    el = time.time() - t
    rows.append({"engine": "DuckDB", "query": label, "seconds": round(el, 2), "scanned_mb": None})
    print(f"  DuckDB  {label:<14} {el:6.2f}s")


AQ_FULL = "SELECT branch, count(*) n, count(label) reviewed FROM banking77_log GROUP BY branch"
AQ_PRUNED = (f"SELECT branch, count(*) n, count(label) reviewed FROM banking77_log "
             f"WHERE date BETWEEN DATE '{LO}' AND DATE '{HI}' GROUP BY branch")

print("全量扫描 (90 天):")
polars_q("full scan", False); duck_q("full scan", False); athena(AQ_FULL, "full scan")
print("\n分区裁剪 (21 天):")
polars_q("3-week slice", True); duck_q("3-week slice", True); athena(AQ_PRUNED, "3-week slice")

json.dump(rows, open("bench.json", "w"), indent=1)
print("\n写入 bench.json")
