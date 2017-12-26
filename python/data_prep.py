from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, VectorAssembler	
from pyspark.sql.functions import udf, col, size, lit
from pyspark.sql.types import *
from pyspark.ml.regression import RandomForestRegressor, LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, NaiveBayes
from pyspark.mllib.classification import SVMWithSGD, LabeledPoint
from pyspark.sql.types import Row
from pyspark.ml.linalg import DenseVector


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


def createClassLabel(relevance):

	return int(relevance)




data_path = '../input/train.csv'
spark = SparkSession.builder.getOrCreate()

df = spark.read.csv(data_path,
	header=True)

df = df.withColumn('new_relevance',df['relevance'].cast(FloatType()))

replaceDigitsUdf = udf(replaceDigits, StringType())

createClassLabelUdf = udf(createClassLabel, IntegerType())

df = df.withColumn('label',createClassLabelUdf('new_relevance'))

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

df.select('label','title_features','sterm_features').show()
assembler = VectorAssembler(
	inputCols=['title_features','sterm_features'],
	outputCol='features')

df = assembler.transform(df)
df.select('title_features','sterm_features','features','label').show()

train_df, test_df = df.randomSplit([0.8, 0.2], seed=45)
train_size = train_df.count()
test_size =  test_df.count()

print train_size,test_size


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
lr = LinearRegression(labelCol='new_relevance',
	maxIter=100,
	regParam=0.2,
	elasticNetParam=0.1)

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
'''
spark.stop()
