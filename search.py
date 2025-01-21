import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pymongo import MongoClient

db_uri = st.secrets["MONGO_URI"]

def store_vectors_in_mongo(chunks, vectors, db_uri):
    client = MongoClient(db_uri)
    db = client['document_database']
    collection = db['vectors']
    
    for i, chunk in enumerate(chunks):
        document = {"chunk": chunk, "vector": vectors[i].tolist()}
        collection.insert_one(document)

def search_similar_chunks(query, vectorized_db, vectorizer):
    query_vector = vectorizer.transform([query]).toarray()
    similarities = cosine_similarity(query_vector, vectorized_db)
    
    return similarities
