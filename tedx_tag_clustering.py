###### TEDx-Load-Aggregate-Model
######

import sys
import json
import pyspark
from pyspark.sql.functions import col, collect_list, array_join, collect_set

from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job


##### FROM FILES
tedx_dataset_path = "s3://unibg-2023-dati-tedx-fl/tedx_dataset.csv"

###### READ PARAMETERS
args = getResolvedOptions(sys.argv, ['JOB_NAME'])

##### START JOB CONTEXT AND JOB
sc = SparkContext()

glueContext = GlueContext(sc)
spark = glueContext.spark_session
    
job = Job(glueContext)
job.init(args['JOB_NAME'], args)


## TAGS DATASET
tags_dataset_path = "s3://unibg-2023-dati-tedx-fl/tags_dataset.csv"

# NLP function for pre process tags
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

def preprocess_tags(tags):
    # Tokenize the tags
    tokens = [nltk.word_tokenize(tag.lower()) for tag in tags]

    # Remove stop words and punctuation
    stop_words = set(stopwords.words('english'))
    punctuation = set(string.punctuation)
    filtered_tokens = [[word for word in tag if word not in stop_words and word not in punctuation] for tag in tokens]

    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [[lemmatizer.lemmatize(word) for word in tag] for tag in filtered_tokens]

    return lemmatized_tokens

### Category clustering 
import pandas as pd
import gensim
from gensim.corpora import Dictionary
from sklearn.cluster import KMeans


# Load the TEDx tags into a dataframe
tedx_tags_df = pd.read_csv(tags_dataset_path)

# Preprocess the tags using NLP techniques
tag_corpus = preprocess_tags(tedx_tags_df["tag"])

# Create a dictionary from the corpus
dictionary = Dictionary(tag_corpus)

# Perform topic modeling on the tag corpus
num_topics = 10
lda_model = gensim.models.LdaModel(tag_corpus, num_topics=num_topics, id2word=dictionary, passes=10)

# Get the topic distribution for each tag
tag_topics = lda_model.get_document_topics(tag_corpus)

# Cluster the tags based on their topic distribution
num_clusters = 5
kmeans_model = KMeans(n_clusters=num_clusters, random_state=42)
tag_clusters = kmeans_model.fit_predict(tag_topics)

# Identify similar categories based on the clusters
cluster_to_category = {
    0: "Science and Technology",
    1: "Arts and Entertainment",
    2: "Education",
    3: "Social Issues",
    4: "Business and Economics"
}

# Map the tags to their corresponding categories
tedx_tags_df["category"] = [cluster_to_category[cluster] for cluster in tag_clusters]

final_df = tedx_tags_df.columns("tag", "category")

# Write data in Mongo collection
mongo_uri = "mongodb+srv://flazzari2:ZYRMTEd7ByIPQGaw@expresscluster.ohodxd3.mongodb.net"
print(mongo_uri)

write_mongo_options = {
    "uri": mongo_uri,
    "database": "unibg_tedx_2023",
    "collection": "tags_to_macro_category",
    "ssl": "true",
    "ssl.domain_match": "false"}

from awsglue.dynamicframe import DynamicFrame
tedx_dataset_dynamic_frame = DynamicFrame.fromDF(final_df, glueContext, "nested")

glueContext.write_dynamic_frame.from_options(tedx_dataset_dynamic_frame, connection_type="mongodb", connection_options=write_mongo_options)