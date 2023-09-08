import sys
from pyspark.sql import SparkSession, functions, types as func

import string
import re

spark = SparkSession.builder.appName('reddit averages').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

assert sys.version_info >= (3, 8) # make sure we have Python 3.8+
assert spark.version >= '3.2' # make sure we have Spark 3.2+


def main(in_directory, out_directory):
    spark = SparkSession.builder.appName("WordCount").getOrCreate()

    lines_data = spark.read.text(in_directory)
    # lines_data.show()

    # Split the lines into words with the provided regular expression
    wordbreak = r'[%s\s]+' % (re.escape(string.punctuation),)
    words_data = lines_data.select(functions.explode(func.split(func.col("value"), wordbreak)).alias("word"))

    # Normalize all strings to lower-case
    words_data = words_data.withColumn("word", func.lower(func.col("word")))

    word_counts = words_data.groupBy("word").count()

    # Sort by decreasing count and alphabetically
    word_counts = word_counts.orderBy(["count", "word"], ascending=[False, True])
    word_counts = word_counts.filter("word != ''") # Remove empty

    word_counts.write.csv(out_directory, header=["word", "count"])

if __name__ == "__main__":
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]
    main(in_directory, out_directory)


