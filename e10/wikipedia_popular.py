import sys
import re
from pathlib import Path
from pyspark.sql import SparkSession, functions, types

spark = SparkSession.builder.appName('reddit averages').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

assert sys.version_info >= (3, 8) # make sure we have Python 3.8+
assert spark.version >= '3.2' # make sure we have Spark 3.2+


pages_schema = types.StructType([
    types.StructField('language', types.StringType()),
    types.StructField('title', types.StringType()),
    types.StructField('frequency', types.IntegerType()),
    types.StructField('bytes', types.IntegerType()),
])

def get_time_label(filename):
    path = Path(filename)
    pathname = path.stem
    pattern = r"\d{8}-\d{2}"
    match = re.search(pattern, pathname)
    return match.group()


def main(in_directory, out_directory):
    pages = spark.read.csv(in_directory, schema=pages_schema, sep=' ').withColumn('filename', functions.input_file_name())
    # pages.show()

    path_to_hour = functions.udf(get_time_label, returnType=types.StringType())
    pages = pages.withColumn('timelabel', path_to_hour(pages['filename']))
    pages = pages.drop('filename')
    pages = pages.filter(pages['language'] == 'en')
    pages = pages.filter(pages['title'] != 'Main_Page')
    pages = pages.filter(pages['title'].startswith('Special:') == False)

    pages = pages.cache() 

    hour_pages = pages.groupBy('timelabel')
    hour_pages = hour_pages.agg(functions.max(pages['frequency']))
    hour_pages = hour_pages.withColumnRenamed('max(frequency)', 'frequency')

    max_frequency_pages = pages.join(hour_pages, ['timelabel', 'frequency']).select('timelabel', 'title', 'frequency')
    max_frequency_pages = max_frequency_pages.sort('timelabel', 'title')

    # max_frequency_pages.show()

    max_frequency_pages.write.csv(out_directory, mode='overwrite')


if __name__=='__main__':
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]
    main(in_directory, out_directory)
