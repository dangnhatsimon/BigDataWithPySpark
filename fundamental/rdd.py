from pyspark.sql import SparkSession 
from pyspark import SparkConf
from pyspark import SparkContext
sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))
spark = SparkSession(sc)


file_path = '/usr/local/share/datasets/README.md'
fileRDD = ['[![buildstatus](https://travis-ci.org/holdenk/learning-spark-examples.svg?branch=master)](https://travis-ci.org/holdenk/learning-spark-examples)',
 'Examples for Learning Spark',
 '===============',
 'Examples for the Learning Spark book. These examples require a number of libraries and as such have long build files. We have also added a stand alone example with minimal dependencies and a small build file',
 'in the mini-complete-example directory.',
 '',
 '',
 'These examples have been updated to run against Spark 1.3 so they may',
 'be slightly different than the versions in your copy of "Learning Spark".',
 '',
 'Requirements',
 '==',
 '* JDK 1.7 or higher',
 '* Scala 2.10.3',
 '- scala-lang.org',
 '* Spark 1.3',
 '* Protobuf compiler',
 '- On debian you can install with sudo apt-get install protobuf-compiler',
 '* R & the CRAN package Imap are required for the ChapterSixExample',
 '* The Python examples require urllib3',
 '',
 'Python examples',
 '===',
 '',
 'From spark just run ./bin/pyspark ./src/python/[example]',
 '',
 'Spark Submit',
 '===',
 '',
 'You can also create an assembly jar with all of the dependencies for running either the java or scala',
 'versions of the code and run the job with the spark-submit script',
 '',
 './sbt/sbt assembly OR mvn package',
 'cd $SPARK_HOME; ./bin/spark-submit   --class com.oreilly.learningsparkexamples.[lang].[example] ../learning-spark-examples/target/scala-2.10/learning-spark-examples-assembly-0.0.1.jar',
 '',
 '[![Learning Spark](http://akamaicovers.oreilly.com/images/0636920028512/cat.gif)](http://www.jdoqocy.com/click-7645222-11260198?url=http%3A%2F%2Fshop.oreilly.com%2Fproduct%2F0636920028512.do%3Fcmp%3Daf-strata-books-videos-product_cj_9781449358600_%2525zp&cjsku=0636920028512)']

# Create an RDD from a list of words
RDD = sc.parallelize(["Spark", "is", "a", "framework", "for", "Big Data processing"])

# Print out the type of the created object
print("The type of RDD is", type(RDD))

# Print the file_path
print("The file_path is", file_path)

# Create a fileRDD from file_path
fileRDD = sc.textFile(file_path)

# Check the type of fileRDD
print("The file type of fileRDD is", type(fileRDD))

# Check the number of partitions in fileRDD
print("Number of partitions in fileRDD is", fileRDD.getNumPartitions())

# Create a fileRDD_part from file_path with 5 partitions
fileRDD_part = sc.textFile(file_path, minPartitions=5)

# Check the number of partitions in fileRDD_part
print("Number of partitions in fileRDD_part is", fileRDD_part.getNumPartitions())


numbRDD = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# Create map() transformation to cube numbers
cubedRDD = numbRDD.map(lambda x: x*x*x)

# Collect the results
numbers_all = cubedRDD.collect()

# Print the numbers from numbers_all
for numb in numbers_all:
    print(numb)

# Filter the fileRDD to select lines with Spark keyword
fileRDD_filter = fileRDD.filter(lambda line: 'Spark' in line)

# How many lines are there in fileRDD?
print("The total number of lines with the keyword Spark is", fileRDD_filter.count())

# Print the first four lines of fileRDD
for line in fileRDD_filter.take(4):
    print(line)

# Create PairRDD Rdd with key value pairs
Rdd = sc.parallelize([(1, 2), (3, 4), (3, 6), (4, 5)])

# Apply reduceByKey() operation on Rdd
Rdd_Reduced = Rdd.reduceByKey(lambda x, y: x+y)

# Iterate over the result and print the output
for num in Rdd_Reduced.collect(): 
    print("Key {} has {} Counts".format(num[0], num[1]))


# Sort the reduced RDD with the key by descending order
Rdd_Reduced_Sort = Rdd_Reduced.sortByKey(ascending=False)

# Iterate over the result and retrieve all the elements of the RDD
for num in Rdd_Reduced_Sort.collect():
    print("Key {} has {} Counts".format(num[0], num[1]))


# Count the unique keys
total = Rdd.countByKey()

# What is the type of total?
print("The type of total is", type(total))

# Iterate over the total and print the output
for k, v in total.items():
    print("key", k, "has", v, "counts")


# Create a baseRDD from the file path
baseRDD = sc.textFile(file_path)

# Split the lines of baseRDD into words
splitRDD = baseRDD.flatMap(lambda x: x.split())

# Count the total number of words
print("Total number of words in splitRDD:", splitRDD.count())

# Convert the words in lower case and remove stop words from the stop_words curated list
splitRDD_no_stop = splitRDD.filter(lambda x: x.lower() not in stop_words)

# Create a tuple of the word and 1 
splitRDD_no_stop_words = splitRDD_no_stop.map(lambda w: (w, 1))

# Count of the number of occurences of each word
resultRDD = splitRDD_no_stop_words.reduceByKey(lambda x, y: x + y)


# Display the first 10 words and their frequencies from the input RDD
for word in resultRDD.take(10):
    print(word)

# Swap the keys and values from the input RDD
resultRDD_swap = resultRDD.map(lambda x: (x[1], x[0]))

# Sort the keys in descending order
resultRDD_swap_sort = resultRDD_swap.sortByKey(ascending=False)

# Show the top 10 most frequent words and their frequencies from the sorted RDD
for word in resultRDD_swap_sort.take(10):
    print("{},{}". format(word[1], word[0]))
 
