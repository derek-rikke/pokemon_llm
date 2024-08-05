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

def run_sql_agent(query):
    response = sql_agent.run(query)
    final_answer = f"Final Answer: {response}"
    return {"final_answer": final_answer}

# Create a tool that uses the SQL agent
pokemon_stats_sql_tool = Tool(
    name="PokemonStatsSQL",
    func=run_sql_agent,
    description="""
    This tool takes a natural language question, performs data analysis queries on a Pokemon database to answer the question, and returns a natural language answer.
    The database contains the following columns: "ID","Name","Form","Type1","Type2","Total","HP","Attack",
    "Defense","Sp. Atk","Sp. Def","Speed","Generation". Sp. is short for "Special".
    This tool is perfect for comparing stats and types and can handle any question that is related to the database.
    """
)
