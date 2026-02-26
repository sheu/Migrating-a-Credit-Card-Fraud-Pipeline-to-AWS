from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, when
from pyspark.sql.types import DoubleType


def build_pipeline(feature_cols: list[str]) -> Pipeline:
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


def cast_features_to_double(df):
    feature_cols = [c for c in df.columns if c != "Class"]
    for c in feature_cols:
        df = df.withColumn(c, col(c).cast(DoubleType()))
    return df


def add_class_weights(df):
    class_counts = df.groupBy("Class").count().collect()
    total = sum(r["count"] for r in class_counts)
    weights = {r["Class"]: (total / r["count"]) for r in class_counts}

    # Default-safe if class labels come as strings
    # (Glue inferSchema can give Class as int/long/string depending on CSV)
    df = df.withColumn(
        "weight",
        when(col("Class") == 0, weights.get(0, 1.0))
        .when(col("Class") == "0", weights.get("0", weights.get(0, 1.0)))
        .otherwise(weights.get(1, weights.get("1", 1.0))),
    )
    return df


def train_model(df, label_col="Class", test_fraction=0.2, seed=42):
    if label_col not in df.columns:
        raise ValueError(f"Missing label column '{label_col}'")

    df = cast_features_to_double(df)
    df = add_class_weights(df)

    feature_cols = [c for c in df.columns if c not in (label_col, "weight")]

    train_df, test_df = df.randomSplit([1 - test_fraction, test_fraction], seed=seed)

    pipeline = build_pipeline(feature_cols)
    model = pipeline.fit(train_df)

    return model, train_df, test_df, feature_cols