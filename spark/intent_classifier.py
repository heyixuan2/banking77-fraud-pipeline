"""Banking77 intent classifier in PySpark.

Two input modes, same pipeline:

  * --csv-train / --csv-test : the original Banking77 CSVs (13k messages, ~1 MB).
  * --log                    : the team's simulated production log (Parquet,
                               Hive-partitioned by date/branch; columns msg_id,
                               label, text, received, injected_type).

The log mode is the "drop-in" path: point --log at s3a://.../raw/ and nothing
else changes. Every step is a Spark DataFrame/ML operation, so the job scales
with the cluster rather than with driver RAM.
"""
import argparse
import json
import time

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
from pyspark.ml.feature import IDF, HashingTF, NGram, RegexTokenizer, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def build_pipeline(num_features: int, max_iter: int, reg: float) -> Pipeline:
    tok = RegexTokenizer(inputCol="text", outputCol="tokens", pattern=r"[^a-z0-9']+",
                         toLowercase=True)
    bigrams = NGram(n=2, inputCol="tokens", outputCol="bigrams")
    tf1 = HashingTF(inputCol="tokens", outputCol="tf_uni", numFeatures=num_features)
    tf2 = HashingTF(inputCol="bigrams", outputCol="tf_bi", numFeatures=num_features)
    idf1 = IDF(inputCol="tf_uni", outputCol="idf_uni", minDocFreq=2)
    idf2 = IDF(inputCol="tf_bi", outputCol="idf_bi", minDocFreq=2)
    vec = VectorAssembler(inputCols=["idf_uni", "idf_bi"], outputCol="features")
    lr = LogisticRegression(family="multinomial", maxIter=max_iter, regParam=reg,
                            elasticNetParam=0.0, labelCol="label", featuresCol="features")
    return Pipeline(stages=[tok, bigrams, tf1, tf2, idf1, idf2, vec, lr])


def densify(model: PipelineModel, spark) -> PipelineModel:
    """Store the LR coefficients as a DenseMatrix.

    Hash buckets never seen in training get exactly-zero weights, so Spark
    compresses the 77 x 65,536 coefficient matrix to a SparseMatrix. Scoring a
    sparse feature vector against that SparseMatrix walks every stored entry
    (~5M per message): 20k messages took ~39 s. The same weights as a
    DenseMatrix score them in ~1.7 s with identical predictions.
    """
    lr = model.stages[-1]
    j = lr._java_obj
    dense = j.coefficientMatrix().toDense()
    jnew = spark._jvm.org.apache.spark.ml.classification.LogisticRegressionModel(
        j.uid(), dense, j.interceptVector(), j.numClasses(), True)
    return PipelineModel(stages=model.stages[:-1] + [LogisticRegressionModel(jnew)])


def macro_f1(pred_df) -> dict:
    """Accuracy and macro-F1 from a label x prediction count table (77x77, tiny)."""
    rows = (pred_df.groupBy("label", F.col("prediction").cast("int").alias("pred"))
            .count().collect())
    tp, act, prd, total, correct = {}, {}, {}, 0, 0
    for r in rows:
        act[r.label] = act.get(r.label, 0) + r["count"]
        prd[r.pred] = prd.get(r.pred, 0) + r["count"]
        total += r["count"]
        if r.label == r.pred:
            tp[r.label] = r["count"]
            correct += r["count"]
    f1s = []
    for c in act:
        p = tp.get(c, 0) / prd[c] if prd.get(c) else 0.0
        rc = tp.get(c, 0) / act[c]
        f1s.append(0.0 if p + rc == 0 else 2 * p * rc / (p + rc))
    return {"accuracy": correct / total, "macro_f1": sum(f1s) / len(f1s), "n_eval": total}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-train")
    ap.add_argument("--csv-test")
    ap.add_argument("--log", help="Parquet log root (local path or s3a://bucket/raw/)")
    ap.add_argument("--intents", required=True, help="intents.csv (label,intent)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-features", type=int, default=1 << 18)
    ap.add_argument("--max-iter", type=int, default=100)
    ap.add_argument("--reg", type=float, default=0.001)
    ap.add_argument("--test-frac-pct", type=int, default=20)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--attack-types", default="token_flooding,identity_fail_transfer",
                    help="injected_type values whose text is a planted attack; kept out of "
                         "training (their label is the replayed message's, not the attack's). "
                         "drift_vocab rows are real intent-54 messages and stay in.")
    ap.add_argument("--train-partitions", type=int, default=0,
                    help="partitions for the training set (0 = 2 x cores)")
    args = ap.parse_args()

    spark = SparkSession.builder.appName("banking77-intent").getOrCreate()
    sc = spark.sparkContext
    m = {"mode": "log" if args.log else "csv", "master": sc.master,
         "default_parallelism": sc.defaultParallelism,
         "driver_memory": spark.conf.get("spark.driver.memory", "1g")}
    t0 = time.time()

    intents = spark.read.csv(args.intents, header=True, inferSchema=True)

    if args.log:
        log = spark.read.option("basePath", args.log).parquet(args.log)
        m["input_partitions"] = log.rdd.getNumPartitions()
        attacks = [t for t in args.attack_types.split(",") if t]
        is_attack = F.coalesce(F.col("injected_type").isin(attacks), F.lit(False))
        labeled = log.where(F.col("label").isNotNull() & ~is_attack)
        unlabeled = log.where(F.col("label").isNull())
        # The replayed log repeats each message text many times. Deduplicate by
        # text and split on a hash of the text, so the same sentence never lands
        # in both train and test (the leakage risk flagged in the EDA report).
        dedup = (labeled.groupBy("text").agg(F.first("label").alias("label")))
        bucket = F.abs(F.hash(F.lower(F.trim("text")))) % 100
        train = dedup.where(bucket >= args.test_frac_pct)
        test = dedup.where(bucket < args.test_frac_pct)
        counts = log.agg(F.count("*").alias("rows"),
                         F.count("label").alias("labeled")).first()
        m["rows_total"], m["rows_labeled"] = counts.rows, counts.labeled
        m["distinct_labeled_texts"] = dedup.count()
    else:
        def load(p):
            df = spark.read.csv(p, header=True, multiLine=True, escape='"')
            return (df.join(F.broadcast(intents), df.category == intents.intent)
                    .select("text", F.col("label").cast("int").alias("label")))
        train, test = load(args.csv_train), load(args.csv_test)
        unlabeled = None

    # Each training partition returns a dense 77 x features gradient on every
    # L-BFGS iteration, so 200 post-groupBy partitions means 200 such arrays per
    # pass. Coalesce to a few partitions per core before fitting.
    n_tp = args.train_partitions or 2 * sc.defaultParallelism
    m["train_partitions_before"] = train.rdd.getNumPartitions()
    train = train.repartition(n_tp).cache()
    m["train_partitions"] = n_tp
    m["n_train"] = train.count()
    m["t_load_s"] = round(time.time() - t0, 1)

    t1 = time.time()
    model = build_pipeline(args.num_features, args.max_iter, args.reg).fit(train)
    m["t_fit_s"] = round(time.time() - t1, 1)
    m["coef_matrix_as_trained"] = type(model.stages[-1].coefficientMatrix).__name__
    model = densify(model, spark)
    train.unpersist()

    t2 = time.time()
    m.update(macro_f1(model.transform(test)))
    m["t_eval_s"] = round(time.time() - t2, 1)
    model.write().overwrite().save(f"{args.out}/model")

    if unlabeled is not None:
        t3 = time.time()
        scored = (model.transform(unlabeled)
                  .withColumn("confidence", F.array_max(vector_to_array("probability")))
                  .withColumn("pred_label", F.col("prediction").cast("int"))
                  .join(F.broadcast(intents.withColumnRenamed("label", "pred_label")),
                        "pred_label", "left")
                  .select("msg_id", "received", "text", "pred_label",
                          F.col("intent").alias("pred_intent"), "confidence",
                          "injected_type", "date", "branch"))
        scored.write.mode("overwrite").partitionBy("date", "branch").parquet(
            f"{args.out}/predictions")
        m["t_score_write_s"] = round(time.time() - t3, 1)

        # Drift monitor: weekly mean confidence per predicted intent (77 x weeks).
        pred = spark.read.parquet(f"{args.out}/predictions")
        weekly = (pred.groupBy(F.date_trunc("week", "received").alias("week"), "pred_intent")
                  .agg(F.count("*").alias("n"), F.avg("confidence").alias("mean_conf")))
        weekly.write.mode("overwrite").parquet(f"{args.out}/drift_weekly")
        # Mean confidence on each planted pattern vs ordinary rows: is
        # low confidence on its own a usable fraud signal?
        by_type = (pred.groupBy(F.coalesce("injected_type", F.lit("none")).alias("t"))
                   .agg(F.count("*").alias("n"), F.avg("confidence").alias("c")).collect())
        m["scored_by_injected_type"] = {r.t: {"n": r.n, "mean_conf": r.c} for r in by_type}
        m["predictions_partitions_written"] = pred.select("date", "branch").distinct().count()

    m["t_total_s"] = round(time.time() - t0, 1)
    json.dump(m, open(args.metrics, "w"), indent=2, default=str)
    print(json.dumps(m, indent=2, default=str))
    spark.stop()


if __name__ == "__main__":
    main()
