# main.py
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from agent_setup import get_agent_executor
from collections import deque
import os

# Initialize a new LLM for relevance checking
relevance_llm = ChatOpenAI(
    temperature=0,
    model_name="gpt-4o-mini",  # Use the same model as the main agent or a smaller one if needed
    max_tokens=100,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

relevance_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI assistant specialized in determining if questions are related to Pokémon. "
               "Respond with 'Yes' if the question is about Pokémon games, creatures, or the Pokémon world. "
               "Respond with 'No' if the question is unrelated to Pokémon. "
               "If you're unsure, respond with 'Uncertain'."),
    ("human", "{question}")
])

def is_pokemon_related(question):
    response = relevance_llm.invoke(relevance_prompt.format_messages(question=question))
    return response.content.strip().lower()

def detect_loop(actions, window_size=3):
    if len(actions) < window_size * 2:
        return False
    recent_actions = deque(actions[-window_size:], maxlen=window_size)
    previous_actions = deque(actions[-2*window_size:-window_size], maxlen=window_size)
    return recent_actions == previous_actions

def answer_pokemon_question(question):
    relevance = is_pokemon_related(question)
    
    if relevance == "no":
        return "I'm sorry, but I'm specifically designed to answer questions about Pokémon. Your question doesn't seem to be related to Pokémon. Could you please ask something about Pokémon games, creatures, or the Pokémon world?"
    
    if relevance == "uncertain":
        return "I'm not sure if your question is related to Pokémon. Could you please clarify or rephrase your question to be more specifically about Pokémon?"

    try:
        agent_executor = get_agent_executor()  # Get a new agent executor for each question
        response = agent_executor.invoke({"input": question})
        actions = [step[0].tool for step in response.get('intermediate_steps', [])]
        
        if detect_loop(actions):
            last_tool_output = response['intermediate_steps'][-1][1]
            return f"I got stuck trying to answer that. Based on my last attempt: {last_tool_output}"
        
        if 'intermediate_steps' in response and response['intermediate_steps']:
            last_tool_output = response['intermediate_steps'][-1][1]
            
            if isinstance(last_tool_output, dict) and 'final_answer' in last_tool_output:
                output = last_tool_output['final_answer']
                sources = last_tool_output.get('sources', [])
                return f"{output}\n\nSources: {', '.join(sources)}"
            
            if 'Agent stopped due to iteration limit or time limit' in response['output']:
                return f"I'm not sure how to answer that question. My last attempt gave me this information: {last_tool_output}"
        
        output = response['output'].strip()
        if not output:
            return "I'm sorry, I couldn't find a specific answer to that question. It might be outside my current knowledge base about Pokémon."
        
        return output
    except Exception as e:
        return f"I apologize, I encountered an error while processing your question: {str(e)}"

if __name__ == "__main__":
    print("Welcome to the Pokemon Assistant! Ask me anything about Pokemon.")
    print("Type 'quit' to exit.")
    
    while True:
        user_question = input("\nYour question: ")
        if user_question.lower() == 'quit':
            print("Thank you for using the Pokemon Assistant. Goodbye!")
            break
        
        answer = answer_pokemon_question(user_question)
        print(f"\nAssistant: {answer}")