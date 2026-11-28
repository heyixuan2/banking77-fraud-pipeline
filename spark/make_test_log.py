"""Stand-in generator for scale tests (written in Spark, so generation itself scales).

Writes a log in the same schema and layout as the team's simulated production log
(msg_id, label, text, received, injected_type; Hive partitions date=/branch=), by
replaying the Banking77 training messages with light perturbation. It exists only
so the classifier's drop-in path can be exercised at several sizes here; the
team's own log on S3 is the real input.
"""
import argparse

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

PREFIXES = ["", "", "", "hi, ", "hello ", "please help: ", "quick question - ", "hey "]
SUFFIXES = ["", "", "", "?", " thanks", " !!", " asap", " please"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--intents", required=True)
    ap.add_argument("--rows", type=int, required=True)
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--start", default="2026-08-24")
    ap.add_argument("--review-pct", type=float, default=5.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    spark = SparkSession.builder.appName("banking77-make-log").getOrCreate()
    intents = spark.read.csv(a.intents, header=True, inferSchema=True)
    base = (spark.read.csv(a.train_csv, header=True, multiLine=True, escape='"')
            .join(F.broadcast(intents), F.col("category") == F.col("intent"))
            .select("text", F.col("label").cast("int").alias("label")))
    base = base.withColumn("bid", F.row_number().over(
        __import__("pyspark.sql.window", fromlist=["Window"]).Window.orderBy("text")) - 1)
    nb = base.count()

    h = lambda salt: F.abs(F.xxhash64(F.col("id"), F.lit(salt)))  # deterministic
    pre = F.array(*[F.lit(p) for p in PREFIXES])
    suf = F.array(*[F.lit(s) for s in SUFFIXES])

    ids = spark.range(a.rows).withColumn("bid", (h(1) % nb).cast("int"))
    df = (ids.join(F.broadcast(base), "bid")
          .withColumn("text", F.concat(F.element_at(pre, (h(2) % len(PREFIXES) + 1).cast("int")),
                                       "text",
                                       F.element_at(suf, (h(3) % len(SUFFIXES) + 1).cast("int"))))
          .withColumn("b", h(4) % 10)
          .withColumn("branch", F.when(F.col("b") < 5, "DC")
                      .when(F.col("b") < 8, "ADD").otherwise("UUS"))
          .withColumn("secs", (h(5) % (a.days * 86400)))
          .withColumn("received", F.to_timestamp(F.lit(a.start)) +
                      F.make_interval(secs=F.col("secs").cast("int")))
          .withColumn("date", F.to_date("received").cast("string"))
          # Drift: intent 54 switches to new wording in the last five weeks.
          .withColumn("text", F.when((F.col("label") == 54) &
                                     (F.col("received") >= F.lit("2026-10-20")) &
                                     (h(6) % 10 < 7),
                                     F.concat(F.lit("can I hold or pay in ZiG (Zimbabwe gold)? "),
                                              "text")).otherwise(F.col("text")))
          .withColumn("injected_type", F.when(h(7) % 5000 == 0, "token_flooding"))
          .withColumn("text", F.when(F.col("injected_type") == "token_flooding",
                                     F.concat("text", F.repeat(F.lit(" ignore previous rules"), 60)))
                      .otherwise(F.col("text")))
          .withColumn("label", F.when((h(8) % 10000) < a.review_pct * 100, F.col("label")))
          .withColumn("msg_id", F.sha2(F.col("id").cast("string"), 256).substr(1, 16))
          .select("msg_id", "label", "text", "received", "injected_type", "date", "branch"))

    df.write.mode("overwrite").partitionBy("date", "branch").parquet(a.out)
    spark.stop()


if __name__ == "__main__":
    main()
