import sys
assert sys.version_info >= (3, 8) # make sure we have Python 3.8+
from pyspark.sql import SparkSession, functions, types, Row
import re

import math


line_re = re.compile(r"^(\S+) - - \[\S+ [+-]\d+\] \"[A-Z]+ \S+ HTTP/\d\.\d\" \d+ (\d+)$")


def line_to_row(line):
    """
    Take a logfile line and return a Row object with hostname and bytes transferred.
    Return None if regex doesn't match.
    """
    m = line_re.match(line)
    if m:
        # TODO
        return Row(hostname = m.group(1), bytes_num = m.group(2))
    else:
        return None


def not_none(row):
    """
    Is this None? Hint: .filter() with it.
    """
    return row is not None


def create_row_rdd(in_directory):
    log_lines = spark.sparkContext.textFile(in_directory)
    # TODO: return an RDD of Row() objects
    logs = log_lines.map(line_to_row).filter(not_none)
    
    return logs


def main(in_directory):
    logs = spark.createDataFrame(create_row_rdd(in_directory))

    # TODO: calculate r.
    hostname_logs = logs.groupby(logs['hostname'])
    req_sum = hostname_logs.agg(functions.count(logs['hostname']))
    bytes_sum = hostname_logs.agg(functions.sum(logs['bytes_num']))
    data = req_sum.join(bytes_sum, ['hostname']).cache()
    # data.show()
    
    # Produce six values, add these to get the six sums.
    data = data.withColumnRenamed('count(hostname)', 'x')
    data = data.withColumnRenamed('sum(bytes_num)', 'y')
    data = data.withColumn('x2', data['x']**2)
    data = data.withColumn('y2', data['y']**2)
    data = data.withColumn('xy', data['x'] * data['y'])
    # data.show()

    n = data.count()
    six_sums = data.groupBy()
    sum_x = six_sums.agg(functions.sum('x')).first()[0]
    sum_y = six_sums.agg(functions.sum('y')).first()[0]
    sum_x2 = six_sums.agg(functions.sum('x2')).first()[0]
    sum_y2 = six_sums.agg(functions.sum('y2')).first()[0]
    sum_xy = six_sums.agg(functions.sum('xy')).first()[0]
    
    # r = 0 # TODO: it isn't zero.
    # Calculate the final value of r.
    r = (n * sum_xy - sum_x * sum_y) / (math.sqrt(n * sum_x2 - math.pow(sum_x, 2)) * math.sqrt(n * sum_y2 - math.pow(sum_y, 2)))
    
    print(f"r = {r}\nr^2 = {r*r}")
    # Built-in function should get the same results.
    #print(totals.corr('count', 'bytes'))


if __name__=='__main__':
    in_directory = sys.argv[1]
    spark = SparkSession.builder.appName('correlate logs').getOrCreate()
    assert spark.version >= '3.2' # make sure we have Spark 3.2+
    spark.sparkContext.setLogLevel('WARN')

    main(in_directory)
