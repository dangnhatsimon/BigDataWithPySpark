# Import the pyspark.sql.types library
from pyspark.sql.types import *
from pyspark.sql import SparkSession 
from pyspark import SparkConf
from pyspark import SparkContext
import pyspark.sql.functions as F
import time
import seaborn as sns
from matplotlib import pyplot as plt
from pyspark.sql.functions import datediff, to_date, lit
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import RandomForestRegressionModel

sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))
spark = SparkSession(sc)


def train_test_split_date(df, split_col, test_days=45):
    """Calculate the date to split test and training sets"""
    # Find how many days our data spans
    max_date = df.agg({split_col: "max"}).collect()[0][0]
    min_date = df.agg({split_col: "min"}).collect()[0][0]
    # Subtract an integer number of days from the last date in dataset
    split_date = max_date - timedelta(days=test_days)
    return split_date

# Find the date to use in spitting test and train
split_date = train_test_split_date(df, "OFFMKTDATE")

# Create Sequential Test and Training Sets
train_df = df.where(df["OFFMKTDATE"] < split_date) 
test_df = df.where(df["OFFMKTDATE"] >= split_date).where(df["LISTDATE"] <= split_date)




split_date = to_date(lit('2017-12-10'))
# Create Sequential Test set
test_df = df.where(df["OFFMKTDATE"] >= split_date).where(df["LISTDATE"] <= split_date)

# Create a copy of DAYSONMARKET to review later
test_df = test_df.withColumn('DAYSONMARKET_Original', test_df['DAYSONMARKET'])

# Recalculate DAYSONMARKET from what we know on our split date
test_df = test_df.withColumn("DAYSONMARKET", datediff(split_date, "LISTDATE"))

# Review the difference
test_df[['LISTDATE', 'OFFMKTDATE', 'DAYSONMARKET_Original', 'DAYSONMARKET']].show()


obs_threshold = 30
cols_to_remove = list()
# Inspect first 10 binary columns in list
for col in binary_cols[0:10]:
  # Count the number of 1 values in the binary column
  obs_count = df.agg({col: "sum"}).collect()[0][0]
  # If less than our observation threshold, remove
  if obs_count < obs_threshold:
    cols_to_remove.append(col)
    
# Drop columns and print starting and ending dataframe shapes
new_df = df.drop(*cols_to_remove)

print('Rows: ' + str(df.count()) + ' Columns: ' + str(len(df.columns)))
print('Rows: ' + str(new_df.count()) + ' Columns: ' + str(len(new_df.columns)))




# Replace missing values
df = df.fillna(-1, subset=["WALKSCORE", "BIKESCORE"])

# Create list of StringIndexers using list comprehension
indexers = [StringIndexer(inputCol=col, outputCol=col+"_IDX")\
            .setHandleInvalid("keep") for col in categorical_cols]
# Create pipeline of indexers
indexer_pipeline = Pipeline(stages=indexers)
# Fit and Transform the pipeline to the original data
df_indexed = indexer_pipeline.fit(df).transform(df)

# Clean up redundant columns
df_indexed = df_indexed.drop(*categorical_cols)
# Inspect data transformations
print(df_indexed.dtypes)





# Train a Gradient Boosted Trees (GBT) model.
gbt = GBTRegressor(featuresCol="features",
                           labelCol="SALESCLOSEPRICE",
                           predictionCol="Prediction_Price",
                           seed=42
                           )

# Train model.
model = gbt.fit(train_df)





# Select columns to compute test error
evaluator = RegressionEvaluator(labelCol="SALESCLOSEPRICE", 
                                predictionCol="Prediction_Price")
# Dictionary of model predictions to loop over
models = {'Gradient Boosted Trees': gbt_predictions, 'Random Forest Regression': rfr_predictions}
for key, preds in models.items():
    # Create evaluation metrics
    rmse = evaluator.evaluate(preds, {evaluator.metricName: "rmse"})
    r2 = evaluator.evaluate(preds, {evaluator.metricName: "r2"})
    
    # Print Model Metrics
    print(key + ' RMSE: ' + str(rmse))
    print(key + ' R^2: ' + str(r2))



# Convert feature importances to a pandas column
fi_df = pd.DataFrame(importances, columns=["importance"])

# Convert list of feature names to pandas column
fi_df['feature'] = pd.Series(feature_cols)

# Sort the data based on feature importance
fi_df.sort_values(by=["importance"], ascending=False, inplace=True)

# Inspect Results
fi_df.head(10)





# Save model
model.save("rfr_no_listprice")

# Load model
loaded_model = RandomForestRegressionModel.load("rfr_no_listprice")