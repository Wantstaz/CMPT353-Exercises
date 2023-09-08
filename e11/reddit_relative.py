import sys
assert sys.version_info >= (3, 8) # make sure we have Python 3.8+
from pyspark.sql import SparkSession, functions, types

comments_schema = types.StructType([
    types.StructField('archived', types.BooleanType()),
    types.StructField('author', types.StringType()),
    types.StructField('author_flair_css_class', types.StringType()),
    types.StructField('author_flair_text', types.StringType()),
    types.StructField('body', types.StringType()),
    types.StructField('controversiality', types.LongType()),
    types.StructField('created_utc', types.StringType()),
    types.StructField('distinguished', types.StringType()),
    types.StructField('downs', types.LongType()),
    types.StructField('edited', types.StringType()),
    types.StructField('gilded', types.LongType()),
    types.StructField('id', types.StringType()),
    types.StructField('link_id', types.StringType()),
    types.StructField('name', types.StringType()),
    types.StructField('parent_id', types.StringType()),
    types.StructField('retrieved_on', types.LongType()),
    types.StructField('score', types.LongType()),
    types.StructField('score_hidden', types.BooleanType()),
    types.StructField('subreddit', types.StringType()),
    types.StructField('subreddit_id', types.StringType()),
    types.StructField('ups', types.LongType()),
    #types.StructField('year', types.IntegerType()),
    #types.StructField('month', types.IntegerType()),
])


def main(in_directory, out_directory):
    comments = spark.read.json(in_directory, schema=comments_schema)

    # TODO
    # Calculate the average score for each subreddit, as before.
    # Exclude any subreddits with average score ≤0.
    avg = comments.groupBy('subreddit').agg(functions.avg('score'))
    avg = avg.filter(avg['avg(score)'] > 0)
    
    avg = avg.cache()
    
    # avg = avg.hint('broadcast')
    
    # Join the average score to the collection of all comments. Divide to get the relative score.
    cmt = comments.join(avg, on='subreddit')
    cmt = cmt.withColumn('rel_score', cmt['score'] / cmt['avg(score)'])
    
    # Determine the max relative score for each subreddit.
    max_rel_scores = cmt.groupBy('subreddit').agg(functions.max('rel_score').alias('rel_score'))
    # max_rel_scores.cache()

    # Join again to get the best comment on each subreddit: we need this step to get the author.
    best_author = cmt.join(max_rel_scores, ['subreddit', 'rel_score'])
    best_author = best_author.select('subreddit', 'author', 'rel_score')
    
    best_author.write.json(out_directory, mode='overwrite')


if __name__=='__main__':
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]
    spark = SparkSession.builder.appName('Reddit Relative Scores').getOrCreate()
    assert spark.version >= '3.2' # make sure we have Spark 3.2+
    spark.sparkContext.setLogLevel('WARN')

    main(in_directory, out_directory)
