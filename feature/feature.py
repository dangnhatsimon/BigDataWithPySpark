# Import the pyspark.sql.types library
from pyspark.sql.types import *
from pyspark.sql import SparkSession 
from pyspark import SparkConf
from pyspark import SparkContext
import pyspark.sql.functions as F
# Import needed functions
from pyspark.sql.functions import to_date, dayofweek
from pyspark.sql.functions import year
# Import needed functions
from pyspark.sql.functions import when
# Import needed functions
from pyspark.sql.functions import split, explode
from pyspark.sql.functions import coalesce, first
# Import transformer
from pyspark.ml.feature import Binarizer
from pyspark.ml.feature import Bucketizer
from pyspark.ml.feature import OneHotEncoder, StringIndexer
import time
import seaborn as sns
from matplotlib import pyplot as plt

sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))
spark = SparkSession(sc)

# Lot size in square feet
acres_to_sqfeet = 43560
df = df.withColumn("LOT_SIZE_SQFT", df["ACRES"] * acres_to_sqfeet)

# Create new column YARD_SIZE
df = df.withColumn("YARD_SIZE", df["LOT_SIZE_SQFT"] - df["FOUNDATIONSIZE"])

# Corr of ACRES vs SALESCLOSEPRICE
print("Corr of ACRES vs SALESCLOSEPRICE: " + str(df.corr("ACRES", "SALESCLOSEPRICE")))
# Corr of FOUNDATIONSIZE vs SALESCLOSEPRICE
print("Corr of FOUNDATIONSIZE vs SALESCLOSEPRICE: " + str(df.corr("FOUNDATIONSIZE", "SALESCLOSEPRICE")))
# Corr of YARD_SIZE vs SALESCLOSEPRICE
print("Corr of YARD_SIZE vs SALESCLOSEPRICE: " + str(df.corr("YARD_SIZE", "SALESCLOSEPRICE")))


# ASSESSED_TO_LIST
df = df.withColumn('ASSESSED_TO_LIST', df['ASSESSEDVALUATION'] / df['LISTPRICE'])
df[['ASSESSEDVALUATION', 'LISTPRICE', 'ASSESSED_TO_LIST']].show(5)
# TAX_TO_LIST
df = df.withColumn('TAX_TO_LIST', df['TAXES'] / df['LISTPRICE'])
df[['TAX_TO_LIST', 'TAXES', 'LISTPRICE']].show(5)
# BED_TO_BATHS
df = df.withColumn('BED_TO_BATHS', df['BEDROOMS'] / df['BATHSTOTAL'])
df[['BED_TO_BATHS', 'BEDROOMS', 'BATHSTOTAL']].show(5)



# Convert to date type
df = df.withColumn("LISTDATE", to_date("LISTDATE"))

# Get the day of the week
df = df.withColumn("List_Day_of_Week", dayofweek("LISTDATE"))

# Sample and convert to pandas dataframe
sample_df = df.sample(False, 0.5, 42).toPandas()

# Plot count plot of of day of week
sns.countplot(x="List_Day_of_Week", data=sample_df)
plt.show()





# Initialize dataframes
df = real_estate_df
price_df = median_prices_df

# Create year column
df = df.withColumn("list_year", year("LISTDATE"))

# Adjust year to match
df = df.withColumn("report_year", (df["list_year"] - 1))

# Create join condition
condition = [df['CITY'] == price_df['City'], df['report_year'] == price_df['Year']]

# Join the dataframes together
df = df.join(price_df, on=condition, how="left")
# Inspect that new columns are available
df[['MedianHomeValue']].show()





# Create boolean conditions for string matches
has_attached_garage = df["GARAGEDESCRIPTION"].like("%Attached Garage%")
has_detached_garage = df["GARAGEDESCRIPTION"].like("%Detached Garage%")

# Conditional value assignment 
df = df.withColumn("has_attached_garage", (when(has_attached_garage, 1)
                                          .when(has_detached_garage, 0)
                                          .otherwise(None)))

# Inspect results
df[['GARAGEDESCRIPTION', 'has_attached_garage']].show(truncate=100)


# Convert string to list-like array
df = df.withColumn("garage_list", split("GARAGEDESCRIPTION", ", "))

# Explode the values into new records
ex_df = df.withColumn("ex_garage_list", explode("garage_list"))

# Inspect the values
ex_df[['ex_garage_list']].distinct().show(100, truncate=50)


# Pivot 
piv_df = ex_df.groupBy("NO").pivot("ex_garage_list").agg(coalesce(first("constant_val")))

# Join the dataframes together and fill null
joined_df = df.join(piv_df, on='NO', how="left")

# Columns to zero fill
zfill_cols = piv_df.columns

# Zero fill the pivoted values
zfilled_df = joined_df.fillna(0, subset=zfill_cols)




# Create the transformer
binarizer = Binarizer(threshold=5.0, inputCol='List_Day_of_Week', outputCol='Listed_On_Weekend')

# Apply the transformation to df
df = binarizer.transform(df)

# Verify transformation
df[['List_Day_of_Week', 'Listed_On_Weekend']].show()


# Plot distribution of sample_df
sns.distplot(sample_df, axlabel='BEDROOMS')
plt.show()

# Create the bucket splits and bucketizer
splits = [0, 1, 2, 3, 4, 5, float('Inf')]
buck = Bucketizer(splits=splits, inputCol="BEDROOMS", outputCol="bedrooms")

# Apply the transformation to df: df_bucket
df_bucket = buck.transform(df)

# Display results
df_bucket[['BEDROOMS', 'bedrooms']].show()


# Map strings to numbers with string indexer
string_indexer = StringIndexer(inputCol="SCHOOLDISTRICTNUMBER", outputCol="School_Index")
indexed_df = string_indexer.fit(df).transform(df)

# Onehot encode indexed values
encoder = OneHotEncoder(inputCol="School_Index", outputCol="School_Vec")
encoded_df = encoder.transform(indexed_df)

# Inspect the transformation steps
encoded_df[['SCHOOLDISTRICTNUMBER', 'School_Index', 'School_Vec']].show(truncate=100)



