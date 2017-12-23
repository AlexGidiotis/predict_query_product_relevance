import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import preprocessing.{FeatureExtractor, Preprocessor}

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

    val preprocessedDF = preprocessor.preprocess(productDescriptionsDF, "product_description")
    val extractedFeatures = featureExtractor.extractFeature(preprocessedDF, "product_description" + "Preprocessed")

    extractedFeatures.show(1, false)
  }

}