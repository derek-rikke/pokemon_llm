# walkthrough_agent.py
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain.tools import Tool
import os

class WalkthroughAgent:
    def __init__(self):
        # Load the vectorstore
        embeddings = OpenAIEmbeddings()
        self.vectorstore = FAISS.load_local("full_vectorstore", embeddings, allow_dangerous_deserialization=True)
        
        # Create a ChatOpenAI instance
        self.llm = ChatOpenAI(
            temperature=0,
            model_name="gpt-4o-mini",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Create a prompt template
        self.qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert on Pokémon game walkthroughs. Always strive for accuracy and specificity in your answers. If asked about a specific gym, mention the city, the gym leader, and their Pokémon."),
            ("human", "Question: {question}\n\nRelevant walkthrough section:\n{context}"),
            ("human", "Based on the above walkthrough section, please answer the question accurately and concisely. If the walkthrough doesn't contain the exact information needed, say so instead of guessing.")
        ])
        
        # Create a RetrievalQA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": self.qa_prompt},
            return_source_documents=True
        )

    def run(self, query: str) -> dict:
        try:
            result = self.qa_chain({"question": query})
            answer = result['result']
            sources = [doc.metadata.get('source', 'Unknown') for doc in result['source_documents']]
            
            return {
                "final_answer": answer,
                "sources": sources
            }
        except Exception as e:
            return {
                "final_answer": f"An error occurred while processing your question: {str(e)}",
                "sources": []
            }

def get_walkthrough_tool():
    walkthrough_agent = WalkthroughAgent()
    return Tool(
        name="WalkthroughTool",
        func=walkthrough_agent.run,
        description="""Use for Pokémon game walkthrough info.
        Provides step-by-step instructions for game progression,
        trainer and Pokémon locations, and general game guidance."""
    )