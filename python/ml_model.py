from pyspark.sql import SparkSession


spark = SparkSession.builder.getOrCreate()
df = spark.read.json('input/train_set')
df.select('features').show(truncate=False)