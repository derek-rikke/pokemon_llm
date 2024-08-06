# main.py
from agent_setup import get_agent_executor
from collections import deque

def detect_loop(actions, window_size=3):
    if len(actions) < window_size * 2:
        return False
    recent_actions = deque(actions[-window_size:], maxlen=window_size)
    previous_actions = deque(actions[-2*window_size:-window_size], maxlen=window_size)
    return recent_actions == previous_actions

def answer_pokemon_question(question):
    try:
        agent_executor = get_agent_executor()
        response = agent_executor.invoke({"input": question})
        actions = [step[0].tool for step in response.get('intermediate_steps', [])]
        
        if detect_loop(actions):
            last_tool_output = response['intermediate_steps'][-1][1]
            return f"The agent got stuck in a loop. Based on the last tool output: {last_tool_output}"
        
        # Check if the response contains a tool's output
        if 'intermediate_steps' in response and response['intermediate_steps']:
            last_tool_output = response['intermediate_steps'][-1][1]
            
            # If the agent didn't provide a coherent final answer, use the tool's output
            if 'Agent stopped due to iteration limit or time limit' in response['output']:
                return f"Based on the tool's output: {last_tool_output}"
        
        return response['output']
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