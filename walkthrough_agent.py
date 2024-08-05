# walkthrough_agent.py

from langchain.agents import Tool
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

class WalkthroughAgent:
    def __init__(self):
        # Load the vectorstore
        embeddings = OpenAIEmbeddings()
        self.vectorstore = FAISS.load_local("full_vectorstore", embeddings, allow_dangerous_deserialization=True)
        
        # Create a ChatOpenAI instance
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
        
        # Create a prompt template
        prompt_template = """You are a Pokémon game expert. Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Provide a concise answer without mentioning the sources.

Context: {context}

Question: {question}
Answer: """
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        
        # Create a RetrievalQA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=False,
            chain_type_kwargs={"prompt": PROMPT}
        )

    def run(self, query: str) -> dict:
        try:
            result = self.qa_chain({"query": query})
            answer = result['result'].strip()
            
            if answer.lower().startswith("answer:"):
                answer = answer[7:].strip()  # Remove "Answer:" prefix if present
            
            final_answer = f"Final Answer: {answer}"
            return {"final_answer": final_answer}
        except Exception as e:
            return {"final_answer": f"An error occurred while processing your question: {str(e)}"}

def get_walkthrough_tool():
    walkthrough_agent = WalkthroughAgent()
    return Tool(
        name="WalkthroughTool",
        func=walkthrough_agent.run,
        description="""
        This tool passes a question to a retrieval agent who will search through all of the
        Pokemon video game walkthroughs and return a natural language answer. The walkthroughs
        contain step by step instructions for correctly progressing through each game. They also
        contain information such as which trainers are on which routes and what their Pokemon are
        as well as what Pokemon can be found in each location. This tool should be used to answer
        most complex questions that are not related to pokemon types and stats.
        """
    )
