# Import the pyspark.sql.types library
from pyspark.sql.types import *
from pyspark.sql import SparkSession 
from pyspark import SparkConf
from pyspark import SparkContext
import pyspark.sql.functions as F
from pyspark.sql.functions import mean, stddev
from pyspark.sql.functions import log
import time
import seaborn as sns
from matplotlib import pyplot as plt

sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))
spark = SparkSession(sc)


# Show top 30 records
df.show(30)

# List of columns to remove from dataset
cols_to_drop = ["STREETNUMBERNUMERIC", "LOTSIZEDIMENSIONS"]

# Drop columns in list
df = df.drop(*cols_to_drop)


# Inspect unique values in the column 'ASSUMABLEMORTGAGE'
df.select(["ASSUMABLEMORTGAGE"]).distinct().show()

# List of possible values containing 'yes'
yes_values = ["Yes w/ Qualifying", "Yes w/No Qualifying"]

# Filter the text values out of df but keep null values
text_filter = ~df['ASSUMABLEMORTGAGE'].isin(yes_values) | df['ASSUMABLEMORTGAGE'].isNull()
df = df.where(text_filter)

# Print count of remaining records
print(df.count())



# Calculate values used for outlier filtering
mean_val = df.agg({"log_SalesClosePrice": "mean"}).collect()[0][0]
stddev_val = df.agg({"log_SalesClosePrice": "stddev"}).collect()[0][0]

# Create three standard deviation (μ ± 3σ) lower and upper bounds for data
low_bound = mean_val - (3 * stddev_val)
hi_bound = mean_val + (3 * stddev_val)

# Filter the data to fit between the lower and upper bounds
df = df.where((df["log_SalesClosePrice"] < hi_bound) & (df["log_SalesClosePrice"] > low_bound))



# Define max and min values and collect them
max_days = df.agg({"DAYSONMARKET": "max"}).collect()[0][0]
min_days = df.agg({"DAYSONMARKET": "min"}).collect()[0][0]

# Create a new column based off the scaled data
df = df.withColumn("percentage_scaled_days", 
                  round((df["DAYSONMARKET"] - min_days) / (max_days - min_days)) * 100)

# Calc max and min for new column
print(df.agg({"DAYSONMARKET": "max"}).collect())
print(df.agg({"DAYSONMARKET": "min"}).collect())



def min_max_scaler(df, cols_to_scale):
    # Takes a dataframe and list of columns to minmax scale. Returns a dataframe.
    for col in cols_to_scale:
        # Define min and max values and collect them
        max_days = df.agg({col: 'max'}).collect()[0][0]
        min_days = df.agg({col: 'min'}).collect()[0][0]
        new_column_name = 'scaled_' + col
        # Create a new column based off the scaled data
        df = df.withColumn(new_column_name,
                           (df[col] - min_days) / (max_days - min_days))
    return df
  
df = min_max_scaler(df, cols_to_scale)
# Show that our data is now between 0 and 1
df[['DAYSONMARKET', 'scaled_DAYSONMARKET']].show()



# Compute the skewness
print(df.agg({"YEARBUILT": "skewness"}).collect())

# Calculate the max year
max_year = df.agg({"YEARBUILT": "max"}).collect()[0][0]

# Create a new column of reflected data
df = df.withColumn("Reflect_YearBuilt", (max_year + 1) - df["YEARBUILT"])

# Create a new column based reflected data
df = df.withColumn("adj_yearbuilt", 1 / log(df["Reflect_YearBuilt"]))



# Sample the dataframe and convert to Pandas
sample_df = df.select(columns).sample(False, 0.1, 42)
pandas_df = sample_df.toPandas()

# Convert all values to T/F
tf_df = pandas_df.isnull()

# Plot it
sns.heatmap(data=tf_df)
plt.xticks(rotation=30, fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.show()

# Set the answer to the column with the most missing data
answer = 'BACKONMARKETDATE'


# Count missing rows
missing = df.where(df["PDOM"].isNull()).count()

# Calculate the mean value
col_mean = df.agg({"PDOM": "mean"}).collect()[0][0]

# Replacing with the mean value for that column
df.fillna(col_mean, subset=["PDOM"])



def column_dropper(df, threshold):
    # Takes a dataframe and threshold for missing values. Returns a dataframe.
    total_records = df.count()
    for col in df.columns:
        # Calculate the percentage of missing values
        missing = df.where(df[col].isNull()).count()
        missing_percent = missing / total_records
        # Drop column if percent of missing is more than threshold
        if missing_percent > threshold:
            df = df.drop(col)
    return df

# Drop columns that are more than 60% missing
df = column_dropper(df, 0.6)





# Cast data types
walk_df = walk_df.withColumn('longitude', walk_df['longitude'].cast('double'))
walk_df = walk_df.withColumn('latitude', walk_df['latitude'].cast('double'))

# Round precision
df = df.withColumn('longitude', round('longitude', 5))
df = df.withColumn('latitude', round('latitude', 5))

# Create join condition
condition = [walk_df['latitude'] == df['latitude'], walk_df['longitude'] == df['longitude']]

# Join the dataframes together
join_df = df.join(walk_df, on=condition, how='left')
# Count non-null records from new field
print(join_df.where(~join_df['walkscore'].isNull()).count())



# Register dataframes as tables
df.createOrReplaceTempView("df")
walk_df.createOrReplaceTempView("walk_df")

# SQL to join dataframes
join_sql = 	"""
			SELECT 
				*
			FROM df
			LEFT JOIN walk_df
			ON df.longitude = walk_df.longitude
			AND df.latitude = walk_df.latitude
			"""
# Perform sql join
joined_df = spark.sql(join_sql)

# Join on mismatched keys precision 
wrong_prec_cond = [df_orig["longitude"] == walk_df["longitude"], df_orig["latitude"] == walk_df["latitude"]]
wrong_prec_df = df_orig.join(walk_df, on=wrong_prec_cond, how='left')

# Compare bad join to the correct one
print(wrong_prec_df.where(wrong_prec_df['walkscore'].isNull()).count())
print(correct_join_df.where(correct_join_df['walkscore'].isNull()).count())

# Create a join on too few keys
few_keys_cond = [df["longitude"] == walk_df["longitude"]]
few_keys_df = df.join(walk_df, on=few_keys_cond, how='left')

# Compare bad join to the correct one
print("Record Count of the Too Few Keys Join Example: " + str(few_keys_df.count()))
print("Record Count of the Correct Join Example: " + str(correct_join_df.count()))

