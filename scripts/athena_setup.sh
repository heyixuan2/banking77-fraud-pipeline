#!/bin/bash
set -e
OUT=s3://dsan6000-yh973/athena-results/
DB=dsan6000

run() {
  QID=$(aws athena start-query-execution --query-string "$1" \
        --result-configuration "OutputLocation=$OUT" \
        --query-execution-context "Database=$2" \
        --query QueryExecutionId --output text)
  for i in $(seq 1 60); do
    S=$(aws athena get-query-execution --query-execution-id "$QID" \
        --query 'QueryExecution.Status.State' --output text)
    case "$S" in SUCCEEDED) echo "$QID"; return 0;;
      FAILED|CANCELLED)
        aws athena get-query-execution --query-execution-id "$QID" \
          --query 'QueryExecution.Status.StateChangeReason' --output text >&2
        return 1;;
    esac
    sleep 2
  done
  echo "timeout" >&2; return 1
}

echo "--- database ---"
run "CREATE DATABASE IF NOT EXISTS $DB" default > /dev/null && echo ok

echo "--- drop old table if any ---"
run "DROP TABLE IF EXISTS banking77_log" $DB > /dev/null && echo ok

echo "--- create external table ---"
run "CREATE EXTERNAL TABLE banking77_log (
  msg_id string,
  label int,
  text string,
  received timestamp,
  injected_type string
)
PARTITIONED BY (\`date\` date, branch string)
STORED AS PARQUET
LOCATION 's3://dsan6000-banking77-team/raw/'" $DB > /dev/null && echo ok

echo "--- load partitions ---"
run "MSCK REPAIR TABLE banking77_log" $DB > /dev/null && echo ok

echo "--- sanity count ---"
QID=$(run "SELECT count(*) FROM banking77_log" $DB)
aws athena get-query-results --query-execution-id "$QID" \
  --query 'ResultSet.Rows[1].Data[0].VarCharValue' --output text
