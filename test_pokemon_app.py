# test_pokemon_app.py
import pytest
from main import answer_pokemon_question, is_pokemon_related
from agent_setup import PokeAPITool, pokemon_stats_sql_tool
from walkthrough_agent import WalkthroughAgent
import sqlite3

# Unit Tests
def test_is_pokemon_related():
    assert is_pokemon_related("What type is Pikachu?") == "yes"
    assert is_pokemon_related("What is the capital of France?") == "no"
    assert is_pokemon_related("Is Pokémon related to world economics?") == "uncertain"

def test_invalid_question():
    question = "What is the capital of France?"
    answer = answer_pokemon_question(question)
    assert "I'm sorry" in answer and "Pokémon" in answer

def test_uncertain_question():
    question = "How do Pokémon relate to world politics?"
    answer = answer_pokemon_question(question)
    assert "clarify" in answer or "rephrase" in answer

def test_pokemon_keyword_in_unrelated_question():
    question = "What's the Pokémon exchange rate for US dollars?"
    answer = answer_pokemon_question(question)
    assert "clarify" in answer or "rephrase" in answer

def test_obviously_pokemon_question():
    question = "What are the starter Pokémon in the Kanto region?"
    answer = answer_pokemon_question(question)
    assert "Bulbasaur" in answer and "Charmander" in answer and "Squirtle" in answer

def test_tangentially_related_question():
    question = "How has Pokémon influenced popular culture?"
    answer = answer_pokemon_question(question)
    assert "Pokémon" in answer and len(answer) > 50  # Ensure a substantive answer
    
def test_stats_question():
    question = "What are Charizard's base stats?"
    answer = answer_pokemon_question(question)
    assert "hp" in answer.lower() and "attack" in answer.lower()

def test_game_walkthrough_question():
    question = "How do I beat the first gym in Pokemon Red?"
    answer = answer_pokemon_question(question)
    assert "brock" in answer.lower()
    assert "pewter city" in answer.lower()
    assert "rock" in answer.lower() or "ground" in answer.lower()
    assert "geodude" in answer.lower() and "onix" in answer.lower()
    assert "Red_and_Blue - Part 3 - Viridian Forest, Pewter City, Pewter Gym.txt" in answer

# Integration Tests
def test_pokeapi_tool_integration():
    tool = PokeAPITool()
    result = tool._run("pokemon pikachu")
    assert "electric" in result.lower()

def test_stats_sql_tool_integration():
    result = pokemon_stats_sql_tool.run("What are Bulbasaur's base stats?")
    assert "hp" in result.lower() and "attack" in result.lower()

def test_walkthrough_agent_integration():
    agent = WalkthroughAgent()
    result = agent.run("How do I get to Vermilion City?")
    assert "vermilion" in result['final_answer'].lower()

# Edge Case Tests
def test_misspelled_pokemon_name():
    question = "What type is Pikachoo?"
    answer = answer_pokemon_question(question)
    assert "electric" in answer.lower() or "did you mean pikachu" in answer.lower()

def test_non_existent_pokemon():
    question = "What type is Notapokemon?"
    answer = answer_pokemon_question(question)
    assert "doesn't exist" in answer.lower() or "couldn't find information" in answer.lower()

def test_very_long_question():
    question = "What " + "very " * 100 + "long question about Pikachu?"
    answer = answer_pokemon_question(question)
    assert len(answer) > 0  # Ensure we get some kind of response

# Performance Test
def test_response_time():
    import time
    start_time = time.time()
    answer_pokemon_question("What type is Pikachu?")
    end_time = time.time()
    assert end_time - start_time < 5  # Ensure response time is under 5 seconds

# Mocking external API calls
@pytest.fixture
def mock_pokeapi(monkeypatch):
    def mock_get(*args, **kwargs):
        class MockResponse:
            def __init__(self):
                self.status_code = 200
            def json(self):
                return {"name": "pikachu", "types": [{"type": {"name": "electric"}}]}
        return MockResponse()
    
    import requests
    monkeypatch.setattr(requests, "get", mock_get)

def test_pokeapi_with_mock(mock_pokeapi):
    tool = PokeAPITool()
    result = tool._run("pokemon pikachu")
    assert "electric" in result.lower()

# Database Test
@pytest.fixture
def test_db():
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE pokemon_stats
                      (Name TEXT, Type1 TEXT, HP INTEGER, Attack INTEGER)''')
    cursor.execute("INSERT INTO pokemon_stats VALUES (?, ?, ?, ?)", 
                   ("Bulbasaur", "Grass", 45, 49))
    conn.commit()
    yield conn
    conn.close()

def test_database_query(test_db):
    cursor = test_db.cursor()
    cursor.execute("SELECT * FROM pokemon_stats WHERE Name='Bulbasaur'")
    result = cursor.fetchone()
    assert result == ("Bulbasaur", "Grass", 45, 49)

# Run tests
if __name__ == "__main__":
    pytest.main(["-v", "test_pokemon_app.py"])