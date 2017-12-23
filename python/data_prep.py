from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.sql.functions import udf, col, size, lit
from pyspark.sql.types import *

import re


def replaceDigits(text):
	"""
	Takes a string of text and preprocesses it. The preprocessing includes integer replacement with '#', remove LaTex
	math and macros, tokenizing, stopword removal and encoding to ascii.

	Argument:
		text: A string.
	Returns:
		filtered_tokens: A list of the tokens.
	"""
	return re.sub('\d+','#',text)

def removePunctuation(text):
	"""
	"""
	return re.sub('[^a-zA-Z# ]','',text)




data_path = 'data/train.csv'
spark = SparkSession.builder.getOrCreate()

df = spark.read.csv(data_path,
	header=True)

replaceDigitsUdf = udf(replaceDigits, StringType())
df = df.withColumn('nodigits_title', replaceDigitsUdf('product_title'))
df = df.withColumn('nodigits_sterm', replaceDigitsUdf('search_term'))

removePunctuationUdf = udf(removePunctuation, StringType())
df = df.withColumn('processed_title', removePunctuationUdf('nodigits_title'))
df = df.withColumn('processed_sterm', removePunctuationUdf('nodigits_sterm'))

tokenizer = Tokenizer(inputCol="processed_description",
	outputCol="tokens")
df = tokenizer.transform(df)
df = tokenizer.transform(df)
df.show()

remover = StopWordsRemover(inputCol="tokens",
	outputCol="filtered_tokens")
df = remover.transform(df)

hashingTF = HashingTF(inputCol="filtered_tokens",
	outputCol="rawFeatures",
	numFeatures=32768)
df = hashingTF.transform(df)

idf = IDF(inputCol="rawFeatures",
	outputCol="features")

idfModel = idf.fit(df)
df = idfModel.transform(df)

df.show()
print df.take(1)