from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, VectorAssembler, CountVectorizer
from pyspark.sql.functions import udf, col, size, lit, concat
from pyspark.sql.types import *
from pyspark.ml.regression import RandomForestRegressor, LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

from itertools import chain
import re
import json
import time

from keras.preprocessing.text import Tokenizer as Keras_Tokenizer


keras_tokenizer = Keras_Tokenizer(filters='')

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


def convert_to_int(wrd2id):
	"""
	Converts word tokens from string to int using the word2id dictionary. 
	"""
	def convert_to_int_(col):
		keras_tokenizer.word_index = wrd2id
		token_seq_int = keras_tokenizer.texts_to_sequences(col)
		return [token for sublist in token_seq_int for token in sublist]
	return udf(convert_to_int_,ArrayType(IntegerType(),False))


def process_data(data_path='../input/train.csv',
	output_name='train'):
	"""
	"""

	spark = SparkSession.builder.getOrCreate()

	min_word_TF = 10
	min_word_DF = 2
	vocabulary_size = 30000

	start_time = time.time()

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

	def concat(type):
	    def concat_(*args):
	        return list(chain(*args))
	    return udf(concat_, ArrayType(type))

	concat_string_arrays = concat(StringType())

	df = df.withColumn('joined_tokens',concat_string_arrays(col('filtered_title_tokens'),col('filtered_sterm_tokens')))

	print('Creating the dictionary...')
	cv = CountVectorizer(inputCol='joined_tokens',
		minTF=min_word_TF,
		minDF=min_word_DF,
		vocabSize=vocabulary_size).fit(df)

	df.select('filtered_title_tokens','filtered_sterm_tokens','relevance').show()

	vocab = cv.vocabulary
	wrd2id = dict((w,i+1) for i,w in enumerate(vocab))
	print('Found %s unique tokens' % len(wrd2id))
	df = df.withColumn('int_title_tokens', convert_to_int(wrd2id)('filtered_title_tokens'))
	df = df.withColumn('int_sterm_tokens', convert_to_int(wrd2id)('filtered_sterm_tokens'))

	df.select('filtered_title_tokens','filtered_sterm_tokens','int_title_tokens','int_sterm_tokens','relevance').show()

	print 'writting indexes...'
	with open('data/word_index.json', 'w') as fp:
		json.dump(wrd2id, fp)

	train_df, test_df = df.randomSplit([0.8, 0.2], seed=45)
	train_size = train_df.count()
	test_size =  test_df.count()

	print train_size,test_size

	train_df.select('int_title_tokens','int_sterm_tokens','product_uid').write.json(path="data/train_set", mode='overwrite')
	test_df.select('int_title_tokens','int_sterm_tokens','product_uid').write.json(path="data/test_set", mode='overwrite')
	train_df.select('relevance').write.json(path="data/train_set_labels", mode='overwrite')
	test_df.select('relevance').write.json(path="data/test_set_labels", mode='overwrite')

	spark.stop()

	end_time = time.time()
	print "--- Run time: %s seconds ---" % (end_time - start_time)


def process_new_data(data_path='../input/test.csv',
	output_name='test'):
	"""
	"""

	spark = SparkSession.builder.getOrCreate()


	start_time = time.time()

	df = spark.read.csv(data_path,
		header=True)

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

	def concat(type):
	    def concat_(*args):
	        return list(chain(*args))
	    return udf(concat_, ArrayType(type))

	concat_string_arrays = concat(StringType())

	df = df.withColumn('joined_tokens',concat_string_arrays(col('filtered_title_tokens'),col('filtered_sterm_tokens')))

	df.select('filtered_title_tokens','filtered_sterm_tokens').show()

	print('Loading the dictionary...')
	
	json_file = open('data/word_index.json')
	json_string = json_file.read()
	wrd2id = json.loads(json_string)

	print('Found %s unique tokens' % len(wrd2id))

	df = df.withColumn('int_title_tokens', convert_to_int(wrd2id)('filtered_title_tokens'))
	df = df.withColumn('int_sterm_tokens', convert_to_int(wrd2id)('filtered_sterm_tokens'))

	df.select('filtered_title_tokens','filtered_sterm_tokens','int_title_tokens','int_sterm_tokens').show()

	data_size = df.count()

	print data_size

	df.select('int_title_tokens','int_sterm_tokens','product_uid').write.json(path="data/new_set", mode='overwrite')

	df.select('id').write.json(path="data/new_set_ids", mode='overwrite')

	spark.stop()

	end_time = time.time()
	print "--- Run time: %s seconds ---" % (end_time - start_time)


if __name__ == '__main__':
    process_new_data(data_path='../input/test.csv')
