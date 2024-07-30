# streamlit_app.py
import streamlit as st
from agent_setup import agent_executor

def answer_pokemon_question(question):
    try:
        response = agent_executor.invoke({"input": question})
        return response['output']
    except Exception as e:
        return f"I apologize, I encountered an error while processing your question: {str(e)}"

st.title("Pokémon Assistant")

user_question = st.text_input("Ask me anything about Pokémon:")

if user_question:
    with st.spinner("Thinking..."):
        answer = answer_pokemon_question(user_question)
    st.write(answer)