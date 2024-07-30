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

    def run(self, query: str) -> str:
        try:
            result = self.qa_chain({"query": query})
            answer = result['result'].strip()
            
            if answer.lower().startswith("answer:"):
                answer = answer[7:].strip()  # Remove "Answer:" prefix if present
            
            return answer
        except Exception as e:
            return f"An error occurred while processing your question: {str(e)}"

def get_walkthrough_tool():
    walkthrough_agent = WalkthroughAgent()
    return Tool(
        name="WalkthroughTool",
        func=walkthrough_agent.run,
        description="Use this tool for questions about game walkthroughs, in-game events, locations, and strategies in various Pokémon games. Input should be a question about a specific Pokémon game."
    )