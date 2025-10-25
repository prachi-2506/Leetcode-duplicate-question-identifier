import streamlit as st
import torch
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load model and vectorizer
@st.cache_resource
def load_model():
    model = torch.load('leetcodemodel.pt', map_location=torch.device('cpu'))
    model.eval()
    vectorizer = joblib.load('tfidfvectorizer.pkl')
    return model, vectorizer

model, vectorizer = load_model()

# Load problem titles
import pandas as pd
df = pd.read_csv('leetcodecleaned.csv')
titles = df['cleantitle'].tolist()

# Function to get similar problems
def find_similar(query, top_n=5):
    query_vec = torch.FloatTensor(vectorizer.transform([query]).toarray())
    query_embedding = model(query_vec).detach().numpy()
    
    # Get all embeddings
    all_vecs = torch.FloatTensor(vectorizer.transform(titles).toarray())
    all_embeddings = model(all_vecs).detach().numpy()
    
    # Compute similarities
    similarities = cosine_similarity(query_embedding, all_embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_n]
    
    results = [(titles[i], similarities[i]) for i in top_indices]
    return results

# Streamlit UI
st.title("🔍 LeetCode Problem Similarity Finder")
st.write("Find similar LeetCode problems based on your query")

query = st.text_input("Enter a problem title or description:")

if query:
    with st.spinner("Finding similar problems..."):
        results = find_similar(query)
    
    st.subheader("Similar Problems:")
    for i, (title, score) in enumerate(results, 1):
        st.write(f"{i}. **{title}** (Similarity: {score:.2f})")
