import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.{SaveMode, SparkSession}
import preprocessing.{FeatureExtractor, Preprocessor}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.sql.functions.{collect_list, concat, concat_ws, lit}

object RelevancePredictorMain {

  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)

  val currentDir = System.getProperty("user.dir")
  val attributesPath = "file://" + currentDir + "/input/attributes.csv"
  val productDescriptionsPath = "file://" + currentDir + "/input/product_descriptions.csv"
  val trainPath = "file://" + currentDir + "/input/train.csv"
  val testPath = "file://" + currentDir + "/input/test.csv"
  val outputPath = "file://" + currentDir + "/output/"

  def main(args: Array[String]) {

    // Setting up Spark Session
    val conf = new SparkConf().setAppName("Relevance Predictor App").setMaster("local[4]").set("spark.executor.memory", "1g")
    val spark = SparkSession.builder().config(conf).getOrCreate()

    // Initializing preprocessor for removing stopwords, punctuations etc.
    val preprocessor = new Preprocessor()

    // Initializing featureExtractor for tf-idf
    val featureExtractor = new FeatureExtractor()

    // Reading files to DataFrames
    val attributesDF = spark.read.option("header", "true").option("delimiter", ",").csv(attributesPath)
    val productDescriptionsDF = spark.read.option("header", "true").option("delimiter", ",").csv(productDescriptionsPath)
    val trainDF = spark.read.option("header", "true").option("delimiter", ",").csv(trainPath)
    val testDF = spark.read.option("header", "true").option("delimiter", ",").csv(testPath)

    // Preparing a joined dataframe to work on
    import spark.implicits._

    val attributesGrouppedAndAggregatedDF  = attributesDF
      .withColumn("attributes", concat($"name", lit(" "), $"value"))
      .select("product_uid","attributes")
      .groupBy("product_uid")
      .agg(concat_ws(" ", collect_list("attributes")).as("attributes"))

    val workingDF = trainDF
      .join(productDescriptionsDF, "product_uid")
      .join(attributesGrouppedAndAggregatedDF, "product_uid")

    val testingDF_1 = testDF
      .join(productDescriptionsDF, "product_uid")
      .join(attributesGrouppedAndAggregatedDF, "product_uid")

    // preprocessing
    val preprocessedTrainDF = preprocessor
      .preprocess(workingDF, Array("product_title", "search_term", "product_description", "attributes"))
    val testingDF_2 = preprocessor
      .preprocess(testingDF_1, Array("product_title", "search_term", "product_description", "attributes"))

    // extracting features with TF-IDF
    val extractedFeaturesTrainDF = featureExtractor
      .extractFeature(preprocessedTrainDF, Array( "product_title",
                                                  "search_term",
                                                  "product_description",
                                                  "attributes"))
    val testingDF_3 = featureExtractor
      .extractFeature(testingDF_2, Array( "product_title",
                                          "search_term",
                                          "product_description",
                                          "attributes"))

    // keeping only specific columns
    val selectedTrainDF = extractedFeaturesTrainDF
      .select("relevance",
        "product_title_NoStopWords_TF_IDF",
        "search_term_NoStopWords_TF_IDF",
        "product_description_NoStopWords_TF_IDF",
        "attributes_NoStopWords_TF_IDF")
    val testingDF_4 = testingDF_3
      .select("id",
        "product_title_NoStopWords_TF_IDF",
        "search_term_NoStopWords_TF_IDF",
        "product_description_NoStopWords_TF_IDF",
        "attributes_NoStopWords_TF_IDF")

    // assembling collumns to one feature column
    val featuresDF = new VectorAssembler()
      .setInputCols(Array("product_title_NoStopWords_TF_IDF",
        "search_term_NoStopWords_TF_IDF",
        "product_description_NoStopWords_TF_IDF",
        "attributes_NoStopWords_TF_IDF"))
      .setOutputCol("features")
      .transform(selectedTrainDF)
      .selectExpr("cast(relevance as float) relevance", "features")

    val testingDF = new VectorAssembler()
      .setInputCols(Array("product_title_NoStopWords_TF_IDF",
        "search_term_NoStopWords_TF_IDF",
        "product_description_NoStopWords_TF_IDF",
        "attributes_NoStopWords_TF_IDF"))
      .setOutputCol("features")
      .transform(testingDF_4)
      .selectExpr("id", "features")

    // preparing linear regression
    val lr = new LinearRegression()
      .setMaxIter(10)
      .setRegParam(0.2)
      .setElasticNetParam(0)
      .setLabelCol("relevance")
      .setFeaturesCol("features")

    // Fit the model
    val Array(featuresTrainDF, featuresTestDF) = featuresDF.randomSplit(Array(0.8, 0.2), 45)
    val lrModel = lr.fit(featuresTrainDF)

    // Print the coefficients and intercept for linear regression
    println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

    // Summarize the model over the training set and print out some metrics
    val trainingSummary = lrModel.summary
    println(s"numIterations: ${trainingSummary.totalIterations}")
    println(s"objectiveHistory: [${trainingSummary.objectiveHistory.mkString(",")}]")
    trainingSummary.residuals.show()
    println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
    println(s"r2: ${trainingSummary.r2}")

    val predictions = lrModel.transform(featuresTestDF)
    predictions.select("prediction", "relevance").show(50, false)

    val evaluator = new RegressionEvaluator()
      .setLabelCol("relevance")
      .setPredictionCol("prediction")
      .setMetricName("rmse")

    val rmse = evaluator.evaluate(predictions)
    println("Root Mean Squared Error (RMSE) on test data = " + rmse)


    val result = lrModel.transform(testingDF)
    result.select("id","prediction").coalesce(1).write.mode(SaveMode.Overwrite).csv(outputPath)

  }

}