import os
from typing import List, Dict
from llama_index.core import Document, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding

# Custom document loader
class WalkthroughLoader(SimpleDirectoryReader):
    def __init__(self, input_dir: str):
        super().__init__(input_dir)

    def load_data(self) -> List[Document]:
        documents = super().load_data()
        for doc in documents:
            metadata = self.extract_metadata(doc.text)
            doc.metadata.update(metadata)
        return documents

    @staticmethod
    def extract_metadata(content: str) -> Dict[str, str]:
        metadata = {}
        lines = content.split('\n')
        for line in lines[:5]:  # Check first 5 lines for metadata
            if line.startswith('Game:'):
                metadata['game'] = line.split(':')[1].strip()
            elif line.startswith('Part:'):
                metadata['part'] = line.split(':')[1].strip()
            elif line.startswith('Keywords:'):
                metadata['keywords'] = [kw.strip() for kw in line.split(':')[1].split(',')]
        return metadata

# Custom node parser
class WalkthroughNodeParser(SimpleNodeParser):
    def get_nodes_from_documents(self, documents: List[Document]) -> List[TextNode]:
        nodes = super().get_nodes_from_documents(documents)
        for node in nodes:
            self.add_relationships(node)
        return nodes

    @staticmethod
    def add_relationships(node: TextNode):
        game = node.metadata.get('game')
        part = node.metadata.get('part')
        
        if game and part:
            part_num = int(part)
            if part_num > 1:
                node.relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(
                    node_id=f"{game} - Part {part_num - 1}",
                    metadata={"game": game, "part": str(part_num - 1)}
                )
            node.relationships[NodeRelationship.NEXT] = RelatedNodeInfo(
                node_id=f"{game} - Part {part_num + 1}",
                metadata={"game": game, "part": str(part_num + 1)}
            )

# Main pipeline function
def process_walkthrough(input_dir: str, output_dir: str):
    # Load documents
    loader = WalkthroughLoader(input_dir)
    documents = loader.load_data()

    # Parse documents into nodes
    parser = WalkthroughNodeParser()
    nodes = parser.get_nodes_from_documents(documents)

    # Create embedding model
    embed_model = OpenAIEmbedding()

    # Create vector store
    vector_store = FaissVectorStore(dim=1536)  # OpenAI embeddings are 1536-dimensional
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Create and save index
    index = VectorStoreIndex(nodes, storage_context=storage_context, embed_model=embed_model)
    index.storage_context.persist(persist_dir=output_dir)

    return index

def query_index(index, query: str, n_results: int = 5):
    query_engine = index.as_query_engine()
    response = query_engine.query(query)
    
    print(f"Query: {query}")
    print(f"Response: {response}")
    print("\nSource Nodes:")
    for node in response.source_nodes[:n_results]:
        print(f"- {node.node.metadata.get('game')} Part {node.node.metadata.get('part')}: {node.node.text[:100]}...")

# Example usage
if __name__ == "__main__":
    input_directory = "./walkthrough_rewrites/Black_and_White"
    output_directory = "./walkthrough_index"
    
    index = process_walkthrough(input_directory, output_directory)
    print(f"Index created and saved to {output_directory}")

    # Test query
    query_index(index, "How do I start my journey in Pokemon Black and White?")