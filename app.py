import os
import streamlit as st
from transformers import AutoTokenizer, AutoModel, DistilBertTokenizer, DistilBertModel
from pinecone import Pinecone, ServerlessSpec
import torch
from openai import OpenAI

# Hardcoded API keys for Pinecone and OpenAI
PINECONE_API_KEY = "1e768c11-49a6-4d79-b238-10ca5a6c92e9"
OPENAI_API_KEY = "sk-proj-CrPimQN5ZrukDH5Eq2DqT3BlbkFJ6dm2syewi71Oh4mvHGa4"

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Define the index names
recipe_index_name = 'recipe-index'
restaurant_index_name = 'restaurant-index'

# Create the Pinecone indices if they do not exist
def setup_pinecone(index_name, dimension):
    if index_name not in pc.list_indexes().names():
        try:
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )
        except Exception as e:
            st.error(f"Failed to create index: {e}")
    return pc.Index(index_name)

# Initialize Hugging Face models and tokenizers
recipe_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
recipe_model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")

restaurant_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
restaurant_model = DistilBertModel.from_pretrained("distilbert-base-uncased")

# Initialize OpenAI client
def setup_openai():
    return OpenAI(api_key=OPENAI_API_KEY)

# Functions for embedding generation
def get_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

def convert_text_to_embedding(text):
    return get_embedding(text, recipe_tokenizer, recipe_model)

def get_restaurant_embedding(text):
    return get_embedding(text, restaurant_tokenizer, restaurant_model)

# Initialize Pinecone indices
recipe_index = setup_pinecone(recipe_index_name, 768)
restaurant_index = setup_pinecone(restaurant_index_name, 768)

# Initialize the OpenAI client
client = setup_openai()

# Function to query Pinecone for recipes
def fetch_recipe_metadata(user_query):
    query_vector = convert_text_to_embedding(user_query)
    response = recipe_index.query(vector=query_vector, top_k=10, include_metadata=True)
    
    matches = response.get('matches', [])
    filtered_recipes = [match.get('metadata', {}) for match in matches if all(ingredient.lower() in match.get('metadata', {}).get('ingredients', '').lower() for ingredient in user_query.split(', '))]
    
    return filtered_recipes

# Function to generate a new recipe using OpenAI
def create_new_recipe(ingredients):
    prompt = f"Create a detailed recipe using the following ingredients: {', '.join(ingredients)}. Provide a name, description, and steps to make it."
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    recipe_text = response.choices[0].message.content.strip()
    parts = recipe_text.split('\n', 2)
    name = parts[0]
    description = parts[1]
    steps = parts[2] if len(parts) > 2 else ""
    return {'name': name, 'description': description, 'steps': steps}

# Function to query Pinecone for restaurants
def get_restaurant_metadata(description):
    embedding = get_restaurant_embedding(description)
    response = restaurant_index.query(
        vector=embedding,
        top_k=10,
        include_metadata=True
    )
    return response['matches'] if response['matches'] else []

# Function to process chat messages
def process_message():
    if st.session_state.chat_input.strip():
        user_message = {"role": "user", "content": st.session_state.chat_input.strip()}
        st.session_state.messages.append(user_message)
        response = fetch_model_response(st.session_state.messages)
        assistant_message = {"role": "assistant", "content": response}
        st.session_state.messages.append(assistant_message)
        st.session_state.chat_input = ""

# Function to fetch model response from OpenAI
def fetch_model_response(messages):
    if messages:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        return completion.choices[0].message.content.strip()
    else:
        return "No message to process."

# Main function for the Streamlit app
def main():
    st.sidebar.title("Menu")  # Set the title for the sidebar
    menu_option = st.sidebar.selectbox("Select an Option", ["Find Recipes", "Find Restaurants", "Chat"])
    
    if menu_option == "Find Recipes":
        user_query = st.sidebar.text_input("Search for a recipe:", "")
        if user_query and (user_query != st.session_state.get('last_query', '')):
            st.session_state.last_query = user_query
            ingredients = user_query.split(', ')
            recipes = fetch_recipe_metadata(user_query)
            if recipes:
                st.session_state.recommended_recipes = [
                    recipe for recipe in recipes
                ][:3]
            else:
                st.session_state.recommended_recipes = [create_new_recipe(ingredients) for _ in range(3)]
            
            st.title("Recommended Recipes")
            if st.session_state.get('recommended_recipes'):
                for recipe in st.session_state.recommended_recipes:
                    st.write("**Name:** " + recipe['name'])
                    st.write("**Description:** " + recipe['description'])
                    st.write("**Steps:** " + recipe['steps'])
                    st.markdown("---")
            else:
                st.write("No recommended recipes found.")

    elif menu_option == "Find Restaurants":
        st.title('Restaurant Finder')
        description = st.text_input('Enter a description of the restaurant to find matches:')
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

    elif menu_option == "Chat":
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "Hello, I'm RecipeBot! Ask me anything about recipes!"}]
        
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        st.text_input("Ask about recipes here:", value="", key="chat_input", on_change=process_message)

# Run the main function when the script is executed
if __name__ == "__main__":
    main()