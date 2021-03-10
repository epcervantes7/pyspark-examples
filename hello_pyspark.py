# to run in local node execute
# pyspark --deploy-mode client --master local[16] --driver-memory 4G --name GStest_1 < hello_pypark.py 


import pandas as pd
from sklearn import datasets
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.linalg import Vectors
from pyspark.ml.clustering import KMeans, KMeansModel

#load iris dataset
iris = datasets.load_iris()

#reorganizing data
X = iris.data 
target = iris.target 
names = iris.target_names
n_names=['sepal_length','sepal_width','petal_length','petal_width']
original_df = pd.DataFrame(X, columns=n_names)
original_df['species'] = iris.target
original_df['species'] = original_df['species'].replace(to_replace= [0, 1, 2], value = ['setosa', 'versicolor', 'virginica'])

print(original_df.head())


#create a pyspark dataframe
spark = SparkSession.builder.getOrCreate()
df = spark.createDataFrame(original_df)

df.show(5,True)
df.printSchema()



#transforms data into vectors
def transData(data):
    return data.rdd.map(lambda r: [Vectors.dense(r[:-1])]).toDF(['features'])
transformed= transData(df)
transformed.show(5, False)

# Automatically identify categorical features, and index them.
# We specify maxCategories so features with > 4 distinct values are treated as continuous.
featureIndexer = VectorIndexer(inputCol="features", \
                               outputCol="indexedFeatures",\
                               maxCategories=4).fit(transformed)

data = featureIndexer.transform(transformed)

#create a kmeans stage
kmeans = KMeans() \
          .setK(3) \
          .setFeaturesCol("indexedFeatures")\
          .setPredictionCol("cluster")

# Chain indexer and kmeans in a Pipeline
pipeline = Pipeline(stages=[featureIndexer, kmeans])

#fit pipeline
model = pipeline.fit(transformed)

#transform data
cluster = model.transform(transformed)

#show results
cluster.show(10)

#cluster sizes
distribution=cluster.select('cluster').groupBy('cluster').count().orderBy('count',ascending=False)
distribution.show()