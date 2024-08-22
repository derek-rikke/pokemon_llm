# pokemon_stats_sql_agent.py
from langchain.tools import StructuredTool
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain.agents.agent_types import AgentType
import os
from dotenv import load_dotenv

load_dotenv()

# Create SQLDatabase instance
db = SQLDatabase.from_uri("sqlite:///data/db.sqlite3")

# Create OpenAI language model instance
llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")

# Create SQL agent
sql_agent = create_sql_agent(
    llm=llm,
    db=db,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    verbose=True
)

class PokemonStatsInput(BaseModel):
    query: str = Field(..., description="The natural language query about Pokemon data")

def run_sql_agent(query: str):
    response = sql_agent.run(query)
    return f"Final Answer: {response}"

pokemon_stats_sql_tool = StructuredTool(
    name="PokemonStatsSQL",
    func=run_sql_agent,
    description="""Use for querying comprehensive Pokémon data. This tool can access information on:
    1. Pokémon species: Basic info, stats, abilities, types, evolution chains, etc.
    2. Moves: Power, accuracy, effects, damage class, etc.
    3. Items: Categories, effects, attributes, etc.
    4. Locations: Areas, encounter rates, etc.
    5. Games: Versions, generation info, etc.
    6. Berries, contests, and more.
    Provide natural language queries about any Pokémon-related topic, and the tool will return relevant information from the database.""",
    args_schema=PokemonStatsInput
)