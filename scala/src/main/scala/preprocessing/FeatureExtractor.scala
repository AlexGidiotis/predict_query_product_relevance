package preprocessing

import org.apache.spark.ml.feature.{HashingTF, IDF}
import org.apache.spark.sql.DataFrame

class FeatureExtractor {

  val vocabularySize = 32768

  def extractFeature(df: DataFrame, inputCol: String) : DataFrame = {
    val inputColumn = inputCol + "_NoStopWords"
    val outputColTF = inputColumn + "_TF"
    val outputColIDF = inputColumn + "_TF_IDF"

    return IDFer(outputColTF, outputColIDF, hashTF(inputColumn, outputColTF, df))
  }

  def extractFeature(df: DataFrame, inputCols: Array[String]) : DataFrame = {
    var dataFrame = df
    inputCols.foreach(inputCol => {
      val inputColumn = inputCol + "_NoStopWords"
      val outputColTF = inputColumn + "_TF"
      val outputColIDF = inputColumn + "_TF_IDF"
      dataFrame = IDFer(outputColTF, outputColIDF, hashTF(inputColumn, outputColTF, dataFrame))
    })

    return dataFrame
  }

  private def hashTF(inputCol: String, outputCol: String, df: DataFrame): DataFrame = {
    val tf_hasher = new HashingTF().setInputCol(inputCol).setOutputCol(outputCol).setNumFeatures(vocabularySize)
    return tf_hasher.transform(df)
  }

  private def IDFer(inputCol: String, outputCol: String, df: DataFrame): DataFrame = {
    val idf = new IDF().setInputCol(inputCol).setOutputCol(outputCol)
    val idfModel = idf.fit(df)
    return idfModel.transform(df)
  }
}
