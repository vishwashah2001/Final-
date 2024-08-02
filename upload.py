import streamlit as st
from pinecone import Pinecone, ServerlessSpecstrem
from transformers import DistilBertTokenizer, DistilBertModel
import torch

# Initialize Pinecone client
api_key = "1e768c11-49a6-4d79-b238-10ca5a6c92e9"
pc = Pinecone(api_key=api_key)

# Define the index name
index_name = 'restaurant-index'

# Check if the index exists before creating it
if index_name not in pc.list_indexes().names():
    try:
        pc.create_index(
            name=index_name,
            dimension=768,  # Ensure this matches the model embeddings
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
    except Exception as e:
        st.error(f"Failed to create index: {e}")

# Connect to the Pinecone index
index = pc.Index(index_name)

# Initialize Hugging Face model and tokenizer
model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertModel.from_pretrained(model_name)

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

def get_restaurant_metadata(description):
    embedding = get_embedding(description)
    response = index.query(
        vector=embedding,
        top_k=10,  # Adjust this to retrieve more matches
        include_metadata=True
    )
    return response['matches'] if response['matches'] else []

# Streamlit UI
st.title('Restaurant Finder')
st.write('Enter a description of the restaurant to find matches:')

description = st.text_input('Description', '')

if st.button('Find Restaurants'):
    if description:
        matches = get_restaurant_metadata(description)
        if matches:
            st.write(f"**Restaurants Matching Description:**")
            for match in matches:
                metadata = match['metadata']
                st.write(f"**Name:** {metadata.get('name', 'N/A')}")
                st.write(f"**Average Rating:** {metadata.get('averageRating', 'N/A')}")
                st.write(f"**City:** {metadata.get('city', 'N/A')}")
                st.write(f"**Description:** {metadata.get('description', 'N/A')}")
                st.write(f"**Display Address:** {metadata.get('displayAddress', 'N/A')}")
                st.write(f"**Image URL:** {metadata.get('imageUrl', 'N/A')}")
                st.write(f"**Latitude:** {metadata.get('latitude', 'N/A')}")
                st.write(f"**Longitude:** {metadata.get('longitude', 'N/A')}")
                st.write(f"**Market:** {metadata.get('market', 'N/A')}")
                st.write(f"**Price Range:** {metadata.get('priceRange', 'N/A')}")
                st.write(f"**Rating Count:** {metadata.get('ratingCount', 'N/A')}")
                st.write(f"**State:** {metadata.get('state', 'N/A')}")
                st.write(f"**Street:** {metadata.get('street', 'N/A')}")
                st.write(f"**Timezone:** {metadata.get('timezone', 'N/A')}")
                st.write(f"**Zip Code:** {metadata.get('zipCode', 'N/A')}")
                st.write("---")
        else:
            st.write("No matches found.")
    else:
        st.write("Please enter a description.")