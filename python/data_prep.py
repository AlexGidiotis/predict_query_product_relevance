from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, VectorAssembler	
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




data_path = '../input/train.csv'
spark = SparkSession.builder.getOrCreate()

df = spark.read.csv(data_path,
	header=True)

df = df.withColumn('new_relevance',df['relevance'].cast(FloatType()))

replaceDigitsUdf = udf(replaceDigits, StringType())
df = df.withColumn('nodigits_title', replaceDigitsUdf('product_title'))
df = df.withColumn('nodigits_sterm', replaceDigitsUdf('search_term'))

removePunctuationUdf = udf(removePunctuation, StringType())
df = df.withColumn('processed_title', removePunctuationUdf('nodigits_title'))
df = df.withColumn('processed_sterm', removePunctuationUdf('nodigits_sterm'))

title_tokenizer = Tokenizer(inputCol="processed_title",
	outputCol="title_tokens")
sterm_tokenizer = Tokenizer(inputCol="processed_sterm",
	outputCol="sterm_tokens")

df = title_tokenizer.transform(df)
df = sterm_tokenizer.transform(df)

title_remover = StopWordsRemover(inputCol="title_tokens",
	outputCol="filtered_title_tokens")
sterm_remover = StopWordsRemover(inputCol="sterm_tokens",
	outputCol="filtered_sterm_tokens")
df = title_remover.transform(df)
df = sterm_remover.transform(df)

title_hashingTF = HashingTF(inputCol="filtered_title_tokens",
	outputCol="title_rawFeatures",
	numFeatures=32768)
sterm_hashingTF = HashingTF(inputCol="filtered_sterm_tokens",
	outputCol="sterm_rawFeatures",
	numFeatures=32768)

df = title_hashingTF.transform(df)
df = sterm_hashingTF.transform(df)

title_idf = IDF(inputCol="title_rawFeatures",
	outputCol="title_features")
sterm_idf = IDF(inputCol="sterm_rawFeatures",
	outputCol="sterm_features")

title_idfModel = title_idf.fit(df)
sterm_idfModel = sterm_idf.fit(df)

df = title_idfModel.transform(df)
df = sterm_idfModel.transform(df)

df.select('new_relevance','title_features','sterm_features').show()
assembler = VectorAssembler(
    inputCols=['title_features','sterm_features'],
    outputCol='features')

df = assembler.transform(df)
df.select('title_features','sterm_features','features','new_relevance').show()

train_df, test_df = df.randomSplit([0.8, 0.2], seed=45)
train_size = train_df.count()
test_size =  test_df.count()

train_df.select('title_features','sterm_features','features','new_relevance').write.json(path="input/train_set", mode='overwrite')
test_df.select('title_features','sterm_features','features','new_relevance').write.json(path="input/test_set", mode='overwrite')



spark.stop()
