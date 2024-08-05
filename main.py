# main.py
from agent_setup import agent_executor

def answer_pokemon_question(question):
    try:
        response = agent_executor.invoke({"input": question})
        if 'final_answer' in response:
            return response['final_answer']
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
