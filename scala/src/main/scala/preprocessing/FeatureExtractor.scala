package preprocessing

import org.apache.spark.ml.feature.{HashingTF, IDF}
import org.apache.spark.sql.DataFrame

class FeatureExtractor {

  val vocabularySize = 32768

  def extractFeature(df: DataFrame, inputCol: String) : DataFrame = {
    val outputColTF = inputCol + "TF"
    val outputColIDF = inputCol + "IDF"

    return IDFer(outputColTF, outputColIDF,
      hashTF(inputCol, outputColTF, df))
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
