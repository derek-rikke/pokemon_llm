# walkthrough_agent.py
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.llms.openai import OpenAI
from langchain.tools import Tool
import os
from dotenv import load_dotenv
from langchain.tools import StructuredTool
from pydantic import BaseModel


load_dotenv()

class WalkthroughAgent:
    def __init__(self):
        # Load the index
        storage_context = StorageContext.from_defaults(persist_dir="./storage")
        self.index = load_index_from_storage(storage_context)
        
        # Create a retriever
        self.retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=3  # Retrieve top 3 most relevant nodes
        )

        # Create an LLM
        self.llm = OpenAI(temperature=0, model="gpt-4")

        # Create a response synthesizer
        self.response_synthesizer = get_response_synthesizer(
            response_mode="compact",
            use_async=True
        )

        # Create a query engine
        self.query_engine = RetrieverQueryEngine(
            retriever=self.retriever,
            response_synthesizer=self.response_synthesizer,
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)],
        )

    async def run(self, query: str) -> dict:
        try:
            # Get the response from the query engine
            response = await self.query_engine.aquery(query)
            
            # Extract source nodes and their metadata
            source_nodes = response.source_nodes
            sources = []
            for node in source_nodes:
                sources.append({
                    'node_id': node.node.node_id,
                    'metadata': node.node.metadata,
                    'score': node.score,
                })
            
            return {
                "final_answer": response.response,
                "sources": [source['metadata'].get('file_name', 'Unknown') for source in sources]
            }
        except Exception as e:
            return {
                "final_answer": f"An error occurred while processing your question: {str(e)}",
                "sources": []
            }

class WalkthroughInput(BaseModel):
    query: str

def get_walkthrough_tool():
    walkthrough_agent = WalkthroughAgent()
    return StructuredTool(
        name="WalkthroughTool",
        func=walkthrough_agent.run,
        description="""Use for Pokémon game walkthrough info.
        Provides step-by-step instructions for game progression,
        trainer and Pokémon locations, and general game guidance.""",
        args_schema=WalkthroughInput
    )