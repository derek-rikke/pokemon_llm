# agent_setup.py

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from pokeapi_tool import PokeAPITool
from walkthrough_agent import get_walkthrough_tool
from pokemon_stats_sql_agent import pokemon_stats_sql_tool
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatOpenAI(
    temperature=0,
    model_name="gpt-3.5-turbo-16k",
    max_tokens=1000,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Initialize tools
pokeapi_tool = PokeAPITool()
walkthrough_tool = get_walkthrough_tool()

# Create the toolbox with available tools
tools = [pokeapi_tool, pokemon_stats_sql_tool, walkthrough_tool]

# Initialize conversation memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# In agent_setup.py

system_template = """Greetings, Trainer! I'm your friendly Pokédex Assistant, ready to help you on your Pokémon journey! I have access to the following tools:

{tools}

To assist you effectively, I'll use this format:

Question: the Pokémon-related question you're asking
Thought: I'll ponder on how to best answer your question
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: what I learn from the action
... (I might need to think, act, and observe a few times)
Thought: Now I know the answer to your question!
Final Answer: the complete answer to your original question

When providing the Final Answer, please format it as follows:
Final Answer: [Your answer here]

Here are examples of how I use each tool:

1. PokeAPI Tool: Use for general Pokémon information, abilities, types, and basic stats.
2. PokemonStatsSQL Tool: Use for detailed stat queries, comparisons, and finding Pokémon with specific stat characteristics.
3. WalkthroughTool: Use for questions about game progression, events, locations, and strategies in various Pokémon games.

For walkthrough-related questions, always use the WalkthroughTool and be sure to specify the game in your query.

I'm here to provide accurate and helpful information about all things Pokémon, including game walkthroughs. If I'm not sure about something, I'll let you know or ask for more details.

Remember to always provide a Final Answer after using the necessary tools. If you've used a tool and received the information you need, proceed to give the Final Answer without unnecessary repetition.

Let's begin our Pokémon adventure!

Question: {input}
Thought: Let's consider which tool would be best for this query.
{agent_scratchpad}"""

# Create a ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_template),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template("{input}"),
])

# Create the agent
agent = create_react_agent(llm, tools, prompt)

# Initialize the agent executor
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    max_iterations=5,
    handle_parsing_errors=True
)