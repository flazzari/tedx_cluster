"""NLP Module"""
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
"""Modules for ML and .csv handling"""
import pandas as pd
"""test"""
import gensim
from gensim.corpora import Dictionary
from sklearn.cluster import KMeans

def preprocess_tags(tags):
    ''' Tag preprocessing for input of kMeans '''
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

# dataset path
TAGS_DATASET = "tedx_dataset.csv"

# Load the TEDx tags into a dataframe
tedx_tags_df = pd.read_csv(TAGS_DATASET)

# Preprocess the tags using NLP techniques
tag_corpus = preprocess_tags(tedx_tags_df["tag"])

# Create a dictionary from the corpus
dictionary = Dictionary(tag_corpus)

# Perform topic modeling on the tag corpus
NUM_TOPICS = 10
lda_model = gensim.models.LdaModel(tag_corpus, num_topics=NUM_TOPICS, id2word=dictionary, passes=10)

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