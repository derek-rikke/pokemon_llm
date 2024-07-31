# pokemon_stats_sql_agent.py
import sqlite3
import pandas as pd
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain.tools import Tool

# Load CSV into SQLite database
df = pd.read_csv('pokemon_types_stats.csv')
conn = sqlite3.connect('pokemon_stats.db')
df.to_sql('pokemon_stats', conn, if_exists='replace', index=False)
conn.close()

# Create SQLDatabase instance
db = SQLDatabase.from_uri("sqlite:///pokemon_stats.db")

# Create OpenAI language model instance
llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")

# Create SQL agent
sql_agent = create_sql_agent(
    llm=llm,
    db=db,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    verbose=True
)

# Create a tool that uses the SQL agent
pokemon_stats_sql_tool = Tool(
    name="PokemonStatsSQL",
    func=sql_agent.run,
    description="Useful for when you need to answer questions about Pokémon stats. Input should be a natural language question about Pokémon stats."
)