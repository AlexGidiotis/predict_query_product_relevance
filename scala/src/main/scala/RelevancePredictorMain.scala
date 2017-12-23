import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession
import preprocessing.{FeatureExtractor, Preprocessor}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.types.IntegerType

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

    // Initializing preprocessor
    val preprocessor = new Preprocessor()

    // Initializing featureExtractor
    val featureExtractor = new FeatureExtractor()

    // Reading files to DataFrames
    val attributesDF = spark.read.option("header", "true").option("delimiter", ",").csv(attributesPath)
    val productDescriptionsDF = spark.read.option("header", "true").option("delimiter", ",").csv(productDescriptionsPath)
    val trainDF = spark.read.option("header", "true").option("delimiter", ",").csv(trainPath)
    val testDF = spark.read.option("header", "true").option("delimiter", ",").csv(testPath)

    val preprocessedDF = preprocessor.preprocess(trainDF, Array("product_title", "search_term"))
    val extractedFeaturesDF = featureExtractor.extractFeature(preprocessedDF, Array("product_title", "search_term"))

    val selectedDF = extractedFeaturesDF.select("relevance", "product_title_NoStopWords_TF_IDF", "search_term_NoStopWords_TF_IDF")

    val featuresDF = new VectorAssembler()
      .setInputCols(Array("product_title_NoStopWords_TF_IDF", "search_term_NoStopWords_TF_IDF"))
      .setOutputCol("features")
      .transform(selectedDF)
      .selectExpr("cast(relevance as float) relevance", "features")

    val lr = new LinearRegression()
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)
      .setLabelCol("relevance")
      .setFeaturesCol("features")

    // Fit the model
    val lrModel = lr.fit(featuresDF)

    // Print the coefficients and intercept for linear regression
    println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

    // Summarize the model over the training set and print out some metrics
    val trainingSummary = lrModel.summary
    println(s"numIterations: ${trainingSummary.totalIterations}")
    println(s"objectiveHistory: [${trainingSummary.objectiveHistory.mkString(",")}]")
    trainingSummary.residuals.show()
    println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
    println(s"r2: ${trainingSummary.r2}")

  }

}