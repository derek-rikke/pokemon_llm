# test_walkthrough_agent.py

import pytest
import asyncio
from walkthrough_agent_llamaindex import WalkthroughAgent, get_walkthrough_tool

@pytest.fixture
def walkthrough_tool():
    return get_walkthrough_tool()

async def run_walkthrough_query(tool, query):
    print(f"\nTesting query: {query}")
    try:
        result = await tool.func(query)
        print(f"Retrieved result: {result}")
        return result
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return None

@pytest.mark.asyncio
async def test_bike_in_pokemon_gold(walkthrough_tool):
    query = "How do I get a bike in Pokemon Gold?"
    result = await run_walkthrough_query(walkthrough_tool, query)
    assert result is not None
    assert "goldenrod city" in result['final_answer'].lower()
    assert "bike shop" in result['final_answer'].lower()

@pytest.mark.asyncio
async def test_eevee_evolution(walkthrough_tool):
    query = "How do I evolve Eevee into Sylveon in Pokémon Sword and Shield?"
    result = await run_walkthrough_query(walkthrough_tool, query)
    assert result is not None
    assert "sylveon" in result['final_answer'].lower()
    assert "friendship" in result['final_answer'].lower()

@pytest.mark.asyncio
async def test_hm_surf_location(walkthrough_tool):
    query = "Where can I find the HM for Surf in Pokémon FireRed/LeafGreen?"
    result = await run_walkthrough_query(walkthrough_tool, query)
    assert result is not None
    assert "safari zone" in result['final_answer'].lower()

@pytest.mark.asyncio
async def test_platinum_team_composition(walkthrough_tool):
    query = "What is the best in-game team for Pokémon Platinum?"
    result = await run_walkthrough_query(walkthrough_tool, query)
    assert result is not None
    assert len(result['final_answer'].split(',')) >= 4  # Expecting at least 4 Pokémon suggestions

@pytest.mark.asyncio
async def test_regi_trio_unlock(walkthrough_tool):
    query = "What are the steps to unlock the Regi trio in Pokémon Emerald?"
    result = await run_walkthrough_query(walkthrough_tool, query)
    assert result is not None
    assert "braille" in result['final_answer'].lower()

@pytest.mark.asyncio
async def test_pokemon_go_transfer(walkthrough_tool):
    query = "How can I transfer Pokémon from Pokémon GO to Pokémon HOME?"
    result = await run_walkthrough_query(walkthrough_tool, query)
    assert result is not None
    assert "pokémon home" in result['final_answer'].lower()

if __name__ == "__main__":
    pytest.main(["-v", "test_walkthrough_agent.py"])