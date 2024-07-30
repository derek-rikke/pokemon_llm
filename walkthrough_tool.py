# walkthrough_tool.py
import os
import re
from typing import Dict, List, Tuple
from langchain.tools import BaseTool
from langchain_community.vectorstores import FAISS

def extract_keywords(walkthrough_dir: str) -> Dict[str, Dict[str, List[str]]]:
    keyword_map = {}
    for root, dirs, files in os.walk(walkthrough_dir):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    game_match = re.search(r'Game: (.+)', content)
                    keywords_match = re.search(r'Keywords: (.+)', content)
                    if game_match and keywords_match:
                        game = game_match.group(1).strip()
                        keywords = keywords_match.group(1).strip().split(', ')
                        if game not in keyword_map:
                            keyword_map[game] = {}
                        for keyword in keywords:
                            if keyword not in keyword_map[game]:
                                keyword_map[game][keyword] = []
                            keyword_map[game][keyword].append(file_path)
    return keyword_map

class WalkthroughTool(BaseTool):
    name = "WalkthroughTool"
    description = "Use this tool for questions about game walkthroughs, in-game events, and strategies in Pokémon games."
    keyword_map: Dict[str, Dict[str, List[str]]]
    vectorstore: FAISS

    def _run(self, query: str) -> str:
        # First, try to identify the game
        game = None
        for potential_game in self.keyword_map.keys():
            if potential_game.lower() in query.lower():
                game = potential_game
                break
        
        if not game:
            return "Could not identify a specific Pokémon game in the query. Please specify which game you're asking about."

        # Now search for other keywords within the identified game
        relevant_files = set()
        for keyword, file_paths in self.keyword_map[game].items():
            if keyword.lower() in query.lower():
                relevant_files.update(file_paths)

        if not relevant_files:
            return f"No relevant information found for {game} in the walkthroughs."

        results = []
        for file_path in relevant_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            results.append(f"From {game} walkthrough:\n{content}")

        combined_results = "\n\n".join(results)
        
        # Use the vectorstore for more precise retrieval
        docs = self.vectorstore.similarity_search(query, k=3)
        vectorstore_results = "\n\n".join([doc.page_content for doc in docs])

        return f"Keyword-based results for {game}:\n{combined_results}\n\nVectorstore results:\n{vectorstore_results}"

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("This tool does not support async")