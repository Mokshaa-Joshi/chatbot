from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def vectorize_text(text, chunk_size=500):
    
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
    
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(chunks).toarray()
    print(len(vectors))  

    return chunks, vectors
