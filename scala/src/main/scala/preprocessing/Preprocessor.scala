package preprocessing

import org.apache.spark.ml.feature.{StopWordsRemover, Tokenizer}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{col, udf}

class Preprocessor {

  def preprocess(df: DataFrame, inputCol: String) : DataFrame = {
    val outputColNoDigits = inputCol + "NoDigits"
    val outputColTokenized = inputCol + "Tokenized"
    val outputColNoPunctuation = inputCol + "NoPunctuation"
    val outputColNoStopWrods = inputCol + "Preprocessed"

    return removeStopWords(outputColTokenized, outputColNoStopWrods,
      tokenize(outputColNoPunctuation, outputColTokenized,
        noPunctuation(outputColNoDigits, outputColNoPunctuation,
          replaceDigits(inputCol, outputColNoDigits, df))))

  }

  def preprocess(df: DataFrame, inputCols: Array[String]) : DataFrame = {
    var dataFrame = df
    inputCols.foreach(inputCol => {
      val outputColNoDigits = inputCol + "_NoDigits"
      val outputColTokenized = inputCol + "_Tokenized"
      val outputColNoPunctuation = inputCol + "_NoPunctuation"
      val outputColNoStopWrods = inputCol + "_NoStopWords"
      dataFrame = removeStopWords(outputColTokenized, outputColNoStopWrods,
        tokenize(outputColNoPunctuation, outputColTokenized,
          noPunctuation(outputColNoDigits, outputColNoPunctuation,
            replaceDigits(inputCol, outputColNoDigits, dataFrame))))
    })
    return dataFrame
  }

  private def replaceDigits(inputCol: String, outputCol: String, df: DataFrame): DataFrame = {
    val replaceDigitsWithHashes: String => String = _.replaceAll("[0-9]+", "#")
    val replaceDigitsWithHashesUDF = udf(replaceDigitsWithHashes)
    return df.withColumn(outputCol, replaceDigitsWithHashesUDF(col(inputCol)))
  }

  private def noPunctuation(inputCol: String, outputCol: String, df: DataFrame): DataFrame = {
    val removePunctuation: String => String = _.replaceAll("[^a-zA-Z# ]", "")
    val removePunctuationUDF = udf(removePunctuation)
    return df.withColumn(outputCol, removePunctuationUDF(col(inputCol)))
  }

  private def tokenize(inputCol: String, outputCol: String, df: DataFrame): DataFrame = {
    val tokenizer = new Tokenizer().setInputCol(inputCol).setOutputCol(outputCol)
    return tokenizer.transform(df)
  }

  private def removeStopWords(inputCol: String, outputCol: String, df: DataFrame): DataFrame = {
    val stopWordsRemover = new StopWordsRemover().setInputCol(inputCol).setOutputCol(outputCol)
    return stopWordsRemover.transform(df)
  }
}
