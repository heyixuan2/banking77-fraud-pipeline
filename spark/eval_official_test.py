"""Score saved models on Banking77's official 3,080-message test split.

Only meaningful for models whose training log never replayed the test split
(make_test_log.py replays train.csv only; scripts/generate_log.py currently
replays train + test).

usage: spark-submit spark/eval_official_test.py <test.csv> <intents.csv> <model_dir> [...]
"""
import json
import sys

from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from intent_classifier import macro_f1  # noqa: E402

spark = SparkSession.builder.appName("banking77-official-test").getOrCreate()
test_csv, intents_csv, models = sys.argv[1], sys.argv[2], sys.argv[3:]
intents = spark.read.csv(intents_csv, header=True, inferSchema=True)
t = spark.read.csv(test_csv, header=True, multiLine=True, escape='"')
t = (t.join(F.broadcast(intents), t.category == intents.intent)
     .select("text", F.col("label").cast("int").alias("label")))
print(json.dumps({m: macro_f1(PipelineModel.load(m).transform(t)) for m in models}, indent=1))
spark.stop()
