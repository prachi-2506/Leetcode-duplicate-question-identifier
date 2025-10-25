import streamlit as st
import torch
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn as nn

# --- Define the same model architecture ---
class TextEmbedder(nn.Module):
    def __init__(self, input_dim, embed_dim=128):
        super(TextEmbedder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, embed_dim)
        )

    def forward(self, x):
        return self.model(x)

# --- Load model and vectorizer ---
@st.cache_resource
def load_model():
    # Load vectorizer first
    vectorizer = joblib.load('tfidf_vectorizer.pkl')

    # Get input dimension (must match training)
    input_dim = vectorizer.transform(["example"]).shape[1]

    # Rebuild model
    model = TextEmbedder(input_dim=input_dim, embed_dim=128)

    # Load weights (state dict)
    state_dict = torch.load('leetcode_model.pt', map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()

    return model, vectorizer

model, vectorizer = load_model()

# --- Load problem titles ---
df = pd.read_csv('leetcode_cleaned.csv')

# NOTE: Fix column name if necessary
if 'clean_title' in df.columns:
    titles = df['clean_title'].tolist()
else:
    titles = df['cleantitle'].tolist()  # fallback if typo

# --- Function to find similar problems ---
def find_similar(query, top_n=5):
    query_vec = torch.FloatTensor(vectorizer.transform([query]).toarray())
    query_embedding = model(query_vec).detach().numpy()

    # Get all embeddings
    all_vecs = torch.FloatTensor(vectorizer.transform(titles).toarray())
    all_embeddings = model(all_vecs).detach().numpy()

    # Compute cosine similarities
    similarities = cosine_similarity(query_embedding, all_embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_n]

    results = [(titles[i], similarities[i]) for i in top_indices]
    return results

# --- Streamlit UI ---
st.title("🔍 LeetCode Problem Similarity Finder")
st.write("Find similar LeetCode problems based on your query")

query = st.text_input("Enter a problem title or description:")

if query:
    with st.spinner("Finding similar problems..."):
        results = find_similar(query)
    
    st.subheader("Similar Problems:")
    for i, (title, score) in enumerate(results, 1):
        st.write(f"{i}. **{title}** (Similarity: {score:.2f})")
