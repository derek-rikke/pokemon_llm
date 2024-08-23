# agent_setup.py

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.prompts.few_shot import FewShotChatMessagePromptTemplate
from langchain.tools.render import format_tool_to_openai_function
from decomposer_agent import DecomposerAgent
from walkthrough_agent_llamaindex import get_walkthrough_tool
from pokemon_stats_sql_agent import pokemon_stats_sql_tool
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatOpenAI(
    temperature=0,
    model_name="gpt-4o-mini",
    max_tokens=1000,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Initialize tools
walkthrough_tool = get_walkthrough_tool()

# Create the toolbox with available tools
tools = [pokemon_stats_sql_tool, walkthrough_tool]

# Get tool names
tool_names = ", ".join([tool.name for tool in tools])

# Initialize conversation memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def load_examples(filename):
    with open(filename, 'r') as file:
        return file.read().strip()

def get_tool_examples(tool_name):
    example_files = {
        "PokemonStatsSQL": "pokemon_stats_sql_examples.txt",
        "WalkthroughTool": "walkthrough_examples.txt"
    }
    return load_examples(example_files.get(tool_name, ""))

# Create example messages for each tool
def create_example_messages(tool_name):
    examples_text = get_tool_examples(tool_name)
    examples = examples_text.split("\n\n")
    
    messages = []
    for example in examples:
        lines = example.split("\n")
        human_message = ""
        ai_message = ""
        for line in lines:
            if line.startswith("Human:"):
                human_message = line[7:].strip()
            elif line.startswith("AI:") or line.startswith("Final Answer:"):
                ai_message += line.split(":", 1)[1].strip() + "\n"
        messages.append({"human": human_message, "ai": ai_message.strip()})
    
    return messages

# Create few-shot templates for each tool
pokemon_stats_few_shot = FewShotChatMessagePromptTemplate(
    examples=create_example_messages("PokemonStatsSQL"),
    example_prompt=ChatPromptTemplate.from_messages([
        ("human", "{human}"),
        ("ai", "{ai}"),
    ]),
)

walkthrough_few_shot = FewShotChatMessagePromptTemplate(
    examples=create_example_messages("WalkthroughTool"),
    example_prompt=ChatPromptTemplate.from_messages([
        ("human", "{human}"),
        ("ai", "{ai}"),
    ]),
)

# System prompt
system_template = """Greetings, Trainer! I'm here to provide accurate and helpful information about all things Pokémon, including game walkthroughs. I have access to the following tools:

{tools}

To assist you effectively, I'll use this format:

Question: the Pokémon-related question you're asking
Thought: I'll ponder on how to best answer your question
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: what I learn from the action
Thought: I'll reflect on the information I've gathered
Final Answer: the complete answer to your original question

Remember:
1. If I have enough information to answer the question without using a tool, I provide a direct Final Answer.
2. If I need to use a tool, I always provide a Final Answer that directly answers the user's question after using the tool.
3. I incorporate the tool's output into my answer.
4. When using tools, I keep Action Inputs concise and specific for best results.
5. I refer to the examples for the specific tool you're using to guide your approach.

IMPORTANT: I always provide a Final Answer after using a tool. I do not repeat the same action multiple times. If I find myself repeating an action, I stop and provide the best answer I can based on the information I have.

Let's begin our Pokémon adventure!"""

# Create a ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_template),
    MessagesPlaceholder(variable_name="chat_history"),
    pokemon_stats_few_shot,
    walkthrough_few_shot,
    HumanMessagePromptTemplate.from_template("{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Create the agent
agent = create_openai_functions_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

# Initialize the agent executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    max_iterations=5,
    handle_parsing_errors=True,
    early_stopping_method="force",
)

# This function will be called from main.py
def get_agent_executor():
    # Format the tools as a string
    tools_str = "\n".join([f"- {tool.name}: {tool.description}" for tool in tools])
    
    # Get tool names
    tool_names = ", ".join([tool.name for tool in tools])
    
    # Create a new prompt with the formatted tools and tool names
    formatted_prompt = prompt.partial(tools=tools_str, tool_names=tool_names)
    
    # Create a new agent with the formatted prompt
    new_agent = create_openai_functions_agent(
        llm=llm,
        prompt=formatted_prompt,
        tools=tools
    )
    
    # Create a new agent executor with the new agent
    new_agent_executor = AgentExecutor(
        agent=new_agent,
        tools=tools,
        memory=memory,
        verbose=True,
        max_iterations=5,
        handle_parsing_errors=True,
        early_stopping_method="force",
    )
    
    return new_agent_executor