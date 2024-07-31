# streamlit_app.py
import streamlit as st
import requests
from agent_setup import agent_executor
from PIL import Image
import re

# Load the placeholder image
placeholder_image = Image.open("pokedex.png")

def get_sprite_url(name, category):
    base_url = "https://pokeapi.co/api/v2"
    try:
        response = requests.get(f"{base_url}/{category}/{name.lower()}")
        data = response.json()
        if category == "pokemon":
            return data['sprites']['front_default']
        elif category == "item":
            return data['sprites']['default']
    except:
        return None

def display_sprite(name, category):
    sprite_url = get_sprite_url(name, category)
    if sprite_url:
        st.image(sprite_url, caption=f"{name.capitalize()} sprite", width=100)
    else:
        st.image(placeholder_image, caption=f"No sprite found for {name}", width=100)

def extract_pokemon_and_items(text):
    # Simple regex to match Pokemon and item names (adjust as needed)
    pattern = r'\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\b'
    return re.findall(pattern, text)

def answer_pokemon_question(question):
    try:
        response = agent_executor.invoke({"input": question})
        answer = response['output']
        
        # Extract potential Pokemon and item names
        names = extract_pokemon_and_items(answer)
        
        # Display sprites for extracted names
        for name in names:
            if get_sprite_url(name, "pokemon"):
                display_sprite(name, "pokemon")
            elif get_sprite_url(name, "item"):
                display_sprite(name, "item")
        
        return answer
    except Exception as e:
        return f"I apologize, I encountered an error while processing your question: {str(e)}"

st.title("Pokémon Assistant")

user_question = st.text_input("Ask me anything about Pokémon:")

if user_question:
    with st.spinner("Thinking..."):
        answer = answer_pokemon_question(user_question)
    st.write(answer)