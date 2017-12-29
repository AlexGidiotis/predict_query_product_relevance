from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, VectorAssembler	
from pyspark.sql.functions import udf, col, size, lit, concat, concat_ws, collect_list
from pyspark.sql.types import *
from pyspark.ml.regression import RandomForestRegressor, LinearRegression, GeneralizedLinearRegression, DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, NaiveBayes
from pyspark.mllib.classification import SVMWithSGD, LabeledPoint
from pyspark.sql.types import Row
from pyspark.ml.linalg import DenseVector

from itertools import chain
import re

from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

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


def createClassLabel(relevance):

	return int(relevance)


def stemming(tokens):
	return [ps.stem(token) for token in tokens]


data_path = '../input/train.csv'
description_path = '../input/product_descriptions.csv'
attributes_path = '../input/attributes.csv'

spark = SparkSession.builder.getOrCreate()

df = spark.read.csv(data_path,
	header=True)

#descr_df = spark.read.csv(description_path,
#	header=True)

attr_df = spark.read.csv(attributes_path,
	header=True)

#df = df.join(descr_df, ['product_uid'])
#attr_df = attr_df.groupBy('product_uid').agg(f.concat_ws(" ", f.collect_list(attr_df.name))).show()

attr_df = attr_df.select('product_uid','value').groupBy('product_uid').agg(concat_ws(' ',collect_list('value')).alias('attribute_names'))

df = df.join(attr_df, ['product_uid'])

df.show()

df = df.withColumn('new_relevance',df['relevance'].cast(FloatType()))

replaceDigitsUdf = udf(replaceDigits, StringType())

createClassLabelUdf = udf(createClassLabel, IntegerType())

df = df.withColumn('label',createClassLabelUdf('new_relevance'))

df = df.withColumn('nodigits_title', replaceDigitsUdf('product_title'))
df = df.withColumn('nodigits_sterm', replaceDigitsUdf('search_term'))
df = df.withColumn('nodigits_attr', replaceDigitsUdf('attribute_names'))
#df = df.withColumn('nodigits_product_description', replaceDigitsUdf('product_description'))

removePunctuationUdf = udf(removePunctuation, StringType())
df = df.withColumn('processed_title', removePunctuationUdf('nodigits_title'))
df = df.withColumn('processed_sterm', removePunctuationUdf('nodigits_sterm'))
df = df.withColumn('processed_attr', removePunctuationUdf('nodigits_attr'))
#df = df.withColumn('processed_product_description', removePunctuationUdf('nodigits_product_description'))


title_tokenizer = Tokenizer(inputCol="processed_title",
	outputCol="title_tokens")
sterm_tokenizer = Tokenizer(inputCol="processed_sterm",
	outputCol="sterm_tokens")
attr_tokenizer = Tokenizer(inputCol="processed_attr",
	outputCol="attr_tokens")
#product_description_tokenizer = Tokenizer(inputCol="processed_product_description",
#	outputCol="product_description_tokens")

df = title_tokenizer.transform(df)
df = sterm_tokenizer.transform(df)
df = attr_tokenizer.transform(df)
#df = product_description_tokenizer.transform(df)

title_remover = StopWordsRemover(inputCol="title_tokens",
	outputCol="filtered_title_tokens")
sterm_remover = StopWordsRemover(inputCol="sterm_tokens",
	outputCol="filtered_sterm_tokens")
attr_remover = StopWordsRemover(inputCol="attr_tokens",
	outputCol="filtered_attr_tokens")
#product_description_remover = StopWordsRemover(inputCol="product_description_tokens",
#	outputCol="filtered_product_description_tokens")
df = title_remover.transform(df)
df = sterm_remover.transform(df)
df = attr_remover.transform(df)
#df = product_description_remover.transform(df)

def concat(type):
    def concat_(*args):
        return list(chain(*args))
    return udf(concat_, ArrayType(type))

concat_string_arrays = concat(StringType())

df = df.withColumn('joined_tokens',concat_string_arrays(col('filtered_title_tokens'),col('filtered_sterm_tokens'),col('filtered_attr_tokens')))

stemmingUdf = udf(stemming, ArrayType(StringType()))
df = df.withColumn('stemmed_tokens', stemmingUdf('joined_tokens'))

joined_hashingTF = HashingTF(inputCol="stemmed_tokens",
	outputCol="joined_rawFeatures",
	numFeatures=16384)

df = joined_hashingTF.transform(df)

joined_idf = IDF(inputCol="joined_rawFeatures",
	outputCol="features")

joined_idfModel = joined_idf.fit(df)

df = joined_idfModel.transform(df)

'''
assembler = VectorAssembler(
	inputCols=['title_features','sterm_features','product_description_features'],
	outputCol='features')

df = assembler.transform(df)
'''

df.select('features','label').show()

train_df, test_df = df.randomSplit([0.8, 0.2], seed=45)
train_size = train_df.count()
test_size =  test_df.count()

print train_size,test_size

'''
#========================================== Classification =====================================================
# Naive Bayess

nb = NaiveBayes(labelCol='label',
	smoothing=1.0,
	modelType="multinomial")

model = nb.fit(train_df)




predictions = model.transform(test_df)

# Select example rows to display.
predictions.select("prediction", "label", "new_relevance").show()

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(labelCol="label",
    predictionCol="prediction",
    metricName="f1")

accuracy = evaluator.evaluate(predictions)
print("Test Error = %g " % (1.0 - accuracy))

evaluator_2 = RegressionEvaluator(labelCol='label',
	predictionCol='prediction',
	metricName='rmse')

rmse = evaluator_2.evaluate(predictions)
print('Root Mean Squared Error (RMSE) on test data = %g' % rmse)

'''
#============================================= Regression =======================================================

dt = DecisionTreeRegressor(labelCol='new_relevance')

lr = LinearRegression(labelCol='new_relevance',
	maxIter=25,
	regParam=0.5)

# Fit the model
model = lr.fit(train_df)


# Print the coefficients and intercept for linear regression
#print("Coefficients: %s" % str(model.coefficients))
#print("Intercept: %s" % str(model.intercept))

# Summarize the model over the training set and print out some metrics
trainingSummary = model.summary
print("numIterations: %d" % trainingSummary.totalIterations)
print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
trainingSummary.residuals.show()
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)


evaluator = RegressionEvaluator(labelCol='new_relevance',
	predictionCol='prediction',
	metricName='rmse')

predictions = model.transform(test_df)
predictions.select('prediction','new_relevance').show()
rmse = evaluator.evaluate(predictions)
print('Root Mean Squared Error (RMSE) on test data = %g' % rmse)

spark.stop()
