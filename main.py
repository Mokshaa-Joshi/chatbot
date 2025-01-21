import streamlit as st
from app.data_processing import extract_text_from_pdf
from app.vectorization import vectorize_text
from app.search import store_vectors_in_mongo, search_similar_chunks
from app.response_generator import generate_response
from sklearn.feature_extraction.text import TfidfVectorizer
from pinecone import Pinecone
import numpy as np

# Initialize Pinecone
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENVIRONMENT = st.secrets["PINECONE_ENVIRONMENT"]

pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

# Ensure index exists
if 'project1' not in pc.list_indexes().names():
    pc.create_index(
        name='project1',
        dimension=254,  
        metric='cosine',  
    )

index = pc.Index('project1')

def main():
    st.title("Project1 - PDF Query App")
    
    pdf_file = st.file_uploader("Upload PDF", type=["pdf"])
    if pdf_file:
        # Extract and vectorize text
        text = extract_text_from_pdf(pdf_file)
        chunks, vectors = vectorize_text(text)
        
        # Store vectors in MongoDB
        mongo_uri = st.secrets["MONGO_URI"]
        store_vectors_in_mongo(chunks, vectors, mongo_uri)
        
        # Upload vectors to Pinecone
        for i, vector in enumerate(vectors):
            index.upsert([(str(i), vector)])
        
        query = st.text_input("Enter your query")
        if query:
            # Vectorize query and pad to match dimension
            query_vector = vectorize_text(query)[1][0]
            query_vector = np.pad(query_vector, (0, 254 - len(query_vector)), mode='constant')
            
            # Query Pinecone for similar chunks
            search_results = index.query(vector=query_vector.tolist(), top_k=5)
            similar_chunks = [chunks[int(res.id)] for res in search_results.matches]
            
            # Display similar chunks
            st.write("Similar Chunks:")
            for chunk in similar_chunks:
                st.write(chunk)
            
            # Generate and display response
            response = generate_response(query)
            st.write("Generated Response:")
            st.write(response)

if __name__ == "__main__":
    main()
