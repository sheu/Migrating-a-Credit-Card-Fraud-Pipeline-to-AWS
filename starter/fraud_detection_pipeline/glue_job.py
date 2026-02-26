import os
import sys
import shutil
from urllib.parse import urlparse

import boto3
from awsglue.context import GlueContext
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType
from pyspark.sql.functions import col, when

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer


# ----------------------------
# Helpers: S3 URI + S3 upload
# ----------------------------
def parse_s3_uri(uri: str) -> tuple[str, str]:
    p = urlparse(uri)
    if p.scheme != "s3" or not p.netloc:
        raise ValueError(f"Invalid S3 URI: {uri}")
    return p.netloc, p.path.lstrip("/").rstrip("/")


def s3_delete_prefix(bucket: str, prefix: str) -> None:
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    batch = []

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            batch.append({"Key": obj["Key"]})
            if len(batch) == 1000:
                s3.delete_objects(Bucket=bucket, Delete={"Objects": batch})
                batch = []

    if batch:
        s3.delete_objects(Bucket=bucket, Delete={"Objects": batch})


def s3_upload_dir(local_dir: str, bucket: str, prefix: str) -> dict:
    if not os.path.isdir(local_dir):
        raise FileNotFoundError(f"Local upload directory does not exist: {local_dir}")

    s3 = boto3.client("s3")
    normalized_prefix = prefix.rstrip("/")

    uploaded_local = []
    uploaded_local_total_bytes = 0

    for root, _, files in os.walk(local_dir):
        for f in files:
            full_path = os.path.join(root, f)
            rel_path = os.path.relpath(full_path, local_dir)
            key = f"{normalized_prefix}/{rel_path}".replace("\\", "/")
            size = os.path.getsize(full_path)
            s3.upload_file(full_path, bucket, key)

            uploaded_local.append({"Key": key, "Size": size})
            uploaded_local_total_bytes += size

    if not uploaded_local:
        raise RuntimeError(f"No files found to upload under {local_dir}")

    # Verify what exists in S3 after upload
    s3_objects = []
    paginator = s3.get_paginator("list_objects_v2")
    verify_prefix = f"{normalized_prefix}/"
    for page in paginator.paginate(Bucket=bucket, Prefix=verify_prefix):
        for obj in page.get("Contents", []):
            s3_objects.append({"Key": obj["Key"], "Size": obj["Size"]})

    return {
        "local_files_uploaded": len(uploaded_local),
        "local_total_bytes_uploaded": uploaded_local_total_bytes,
        "s3_object_count": len(s3_objects),
        "s3_total_bytes": sum(obj["Size"] for obj in s3_objects),
        "sample_s3_objects": s3_objects[:10],
    }


def s3_prefix_report(bucket: str, prefix: str) -> dict:
    s3 = boto3.client("s3")
    normalized_prefix = prefix.rstrip("/")
    verify_prefix = f"{normalized_prefix}/"

    s3_objects = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=verify_prefix):
        for obj in page.get("Contents", []):
            s3_objects.append({"Key": obj["Key"], "Size": obj["Size"]})

    return {
        "s3_object_count": len(s3_objects),
        "s3_total_bytes": sum(obj["Size"] for obj in s3_objects),
        "sample_s3_objects": s3_objects[:10],
    }


# ----------------------------
# Model training logic
# ----------------------------
def add_class_weights(df):
    counts = df.groupBy("Class").count().collect()
    total = sum(r["count"] for r in counts)
    weights = {r["Class"]: (total / r["count"]) for r in counts}

    # Handles Class as int or string
    df = df.withColumn(
        "weight",
        when((col("Class") == 0) | (col("Class") == "0"), float(weights.get(0, weights.get("0", 1.0))))
        .otherwise(float(weights.get(1, weights.get("1", 1.0))))
    )
    return df


def build_pipeline(feature_cols):
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="numericFeatures")
    scaler = StandardScaler(
        inputCol="numericFeatures",
        outputCol="features",
        withStd=True,
        withMean=True,
    )
    indexer = StringIndexer(inputCol="Class", outputCol="label")
    rf = RandomForestClassifier(
        labelCol="label",
        featuresCol="features",
        numTrees=100,
        maxDepth=10,
        weightCol="weight",
    )
    return Pipeline(stages=[assembler, scaler, indexer, rf])


def train_model(df, test_fraction=0.2, seed=42):
    df = add_class_weights(df)

    feature_cols = [c for c in df.columns if c not in ("Class", "weight")]

    train_df, test_df = df.randomSplit([1 - test_fraction, test_fraction], seed=seed)

    pipeline = build_pipeline(feature_cols)
    model = pipeline.fit(train_df)

    return model, train_df, test_df


def model_debug_summary(model) -> dict:
    stage_types = [type(stage).__name__ for stage in getattr(model, "stages", [])]
    summary = {
        "model_type": type(model).__name__,
        "stage_count": len(stage_types),
        "stage_types": stage_types,
    }

    rf_stage = None
    for stage in getattr(model, "stages", []):
        if type(stage).__name__ == "RandomForestClassificationModel":
            rf_stage = stage
            break

    if rf_stage is not None:
        num_trees = rf_stage.getNumTrees if not callable(rf_stage.getNumTrees) else rf_stage.getNumTrees()
        total_nodes = rf_stage.totalNumNodes if not callable(rf_stage.totalNumNodes) else rf_stage.totalNumNodes()
        summary["rf_num_trees"] = num_trees
        summary["rf_total_nodes"] = total_nodes

    return summary


def build_input_schema():
    # Credit card dataset columns: Time, V1..V28, Amount, Class
    return StructType(
        [StructField("Time", DoubleType(), True)]
        + [StructField(f"V{i}", DoubleType(), True) for i in range(1, 29)]
        + [StructField("Amount", DoubleType(), True)]
        + [StructField("Class", IntegerType(), True)]
    )


def get_optional_arg(argv, name: str):
    # name should be like "--TRIGGER_OBJECT"
    for i, a in enumerate(argv):
        if a == name and i + 1 < len(argv):
            return argv[i + 1]
    return None


def count_files_recursive(path: str) -> int:
    if not os.path.isdir(path):
        return 0
    return sum(len(files) for _, _, files in os.walk(path))


# ----------------------------
# Main Glue job
# ----------------------------
def main():
    args = getResolvedOptions(sys.argv, ["JOB_NAME", "INPUT_S3", "PROCESSED_S3", "MODEL_S3"])

    trigger_object = get_optional_arg(sys.argv, "--TRIGGER_OBJECT")
    input_s3 = trigger_object or args["INPUT_S3"]
    processed_s3 = args["PROCESSED_S3"].rstrip("/")
    model_s3 = args["MODEL_S3"].rstrip("/")

    sc = SparkContext.getOrCreate()
    glue_context = GlueContext(sc)
    spark = glue_context.spark_session

    # S3A committer settings (safe; do NOT set spark.speculation here)
    spark.conf.set(
        "spark.hadoop.mapreduce.outputcommitter.factory.scheme.s3a",
        "org.apache.hadoop.fs.s3a.commit.S3ACommitterFactory",
    )
    spark.conf.set("spark.hadoop.fs.s3a.committer.name", "directory")
    spark.conf.set("spark.hadoop.fs.s3a.committer.staging.conflict-mode", "replace")
    spark.conf.set("spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version", "2")
    spark.conf.set("spark.hadoop.mapreduce.fileoutputcommitter.cleanup-failures.ignored", "true")

    # ---- Force a valid committer (Glue runtime lacks DirectOutputCommitter) ----
    hconf = spark.sparkContext._jsc.hadoopConfiguration()
    hconf.set("mapreduce.outputcommitter.class", "org.apache.hadoop.mapreduce.lib.output.FileOutputCommitter")
    hconf.set("mapred.output.committer.class", "org.apache.hadoop.mapred.FileOutputCommitter")
    hconf.set("mapreduce.fileoutputcommitter.algorithm.version", "2")

    print(f"INPUT_S3 used: {input_s3}")
    print(f"PROCESSED_S3 : {processed_s3}")
    print(f"MODEL_S3     : {model_s3}")

    schema = build_input_schema()

    df = (
        spark.read
        .option("header", True)
        .option("mode", "DROPMALFORMED")
        .schema(schema)
        .csv(input_s3)
    )

    # Basic sanity check
    row_count = df.count()
    print(f"Loaded rows: {row_count}")
    if row_count == 0:
        raise RuntimeError(f"No rows loaded from {input_s3}. Check S3 path and file contents.")

    model, train_df, test_df = train_model(df)
    model_info = model_debug_summary(model)
    print(
        "Model object created -> "
        f"type: {model_info['model_type']}, "
        f"stages: {model_info['stage_count']}, "
        f"stage types: {model_info['stage_types']}"
    )
    if "rf_num_trees" in model_info:
        print(
            "RandomForest stage -> "
            f"num trees: {model_info['rf_num_trees']}, "
            f"total nodes: {model_info['rf_total_nodes']}"
        )

    # Write processed datasets
    train_df.write.mode("overwrite").parquet(f"{processed_s3}/train/")
    test_df.write.mode("overwrite").parquet(f"{processed_s3}/test/")

    # Save model LOCALLY then upload to S3 (avoids DirectOutputCommitter issues)
    bucket, prefix = parse_s3_uri(model_s3)

    local_model_dir = "/tmp/fraud_detection_model_latest"

    if os.path.exists(local_model_dir):
        shutil.rmtree(local_model_dir)

    # Glue/Spark runtimes can vary on local path handling. Try plain path first,
    # then fall back to file URI if needed.
    save_attempts = [local_model_dir, f"file:{local_model_dir}"]
    save_used = None
    save_errors = []

    for save_target in save_attempts:
        if os.path.exists(local_model_dir):
            shutil.rmtree(local_model_dir)

        try:
            model.write().overwrite().save(save_target)
            local_model_files = count_files_recursive(local_model_dir)
            if local_model_files > 0:
                save_used = save_target
                break
            save_errors.append(f"save target '{save_target}' produced 0 files under {local_model_dir}")
        except Exception as e:
            save_errors.append(f"save target '{save_target}' failed: {e}")

    local_model_files = count_files_recursive(local_model_dir)
    if local_model_files == 0:
        top_entries = []
        if os.path.isdir("/tmp"):
            try:
                top_entries = sorted(os.listdir("/tmp"))[:50]
            except Exception:
                top_entries = ["<unable to list /tmp>"]

        print(
            "Local model save produced no files; falling back to direct S3 save. "
            f"Attempts: {save_errors}. "
            f"Checked local model dir: {local_model_dir}. "
            f"/tmp sample entries: {top_entries}"
        )

        # Fallback: write model directly to S3 and verify prefix contents.
        model.write().overwrite().save(model_s3)
        direct_report = s3_prefix_report(bucket, prefix)
        if direct_report["s3_object_count"] == 0:
            raise RuntimeError(
                "Direct S3 model save completed but no objects found at "
                f"s3://{bucket}/{prefix}"
            )

        print("Training complete")
        print(f"Processed train -> {processed_s3}/train/")
        print(f"Processed test  -> {processed_s3}/test/")
        print(f"Model uploaded (direct) -> s3://{bucket}/{prefix}")
        print(
            "Direct upload summary -> "
            f"s3 object count: {direct_report['s3_object_count']}, "
            f"s3 total bytes: {direct_report['s3_total_bytes']}"
        )
        print(f"Sample uploaded objects (up to 10): {direct_report['sample_s3_objects']}")
        return

    print(f"Model saved locally using target: {save_used}")

    # Replace existing model prefix
    s3_delete_prefix(bucket, prefix)
    upload_report = s3_upload_dir(local_model_dir, bucket, prefix)

    print("Training complete")
    print(f"Processed train -> {processed_s3}/train/")
    print(f"Processed test  -> {processed_s3}/test/")
    print(f"Model uploaded  -> s3://{bucket}/{prefix}")
    print(
        "Upload summary -> "
        f"local files uploaded: {upload_report['local_files_uploaded']}, "
        f"local total bytes: {upload_report['local_total_bytes_uploaded']}, "
        f"s3 object count: {upload_report['s3_object_count']}, "
        f"s3 total bytes: {upload_report['s3_total_bytes']}"
    )
    print(f"Sample uploaded objects (up to 10): {upload_report['sample_s3_objects']}")


if __name__ == "__main__":
    main()