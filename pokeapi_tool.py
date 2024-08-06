# pokeapi_tool.py
import requests
from langchain.tools import BaseTool

BASE_URL = "https://pokeapi.co/api/v2/"

class PokeAPITool(BaseTool):
    name = "PokeAPI"
    description = """Use for general Pokémon info (not stats or walkthroughs).
    Input format: 'endpoint name_or_id [subresource]'
    Endpoints: pokemon, ability, move, item, type, etc."""

    def _run(self, query: str) -> str:
        try:
            parts = query.split(maxsplit=2)
            if len(parts) < 2:
                return "Error: Input should be in the format 'endpoint name_or_id [subresource]'"
            
            endpoint, name_or_id = parts[:2]
            subresource = parts[2] if len(parts) > 2 else None

            endpoint = endpoint.lower().replace('_', '-')
            name_or_id = name_or_id.lower()

            valid_endpoints = [
                "berry", "berry-firmness", "berry-flavor", "contest-type", "contest-effect", "super-contest-effect",
                "encounter-method", "encounter-condition", "encounter-condition-value", "evolution-chain", "evolution-trigger",
                "generation", "pokedex", "version", "version-group", "item", "item-attribute", "item-category", "item-fling-effect",
                "item-pocket", "location", "location-area", "pal-park-area", "region", "machine", "move", "move-ailment", "move-battle-style",
                "move-category", "move-damage-class", "move-learn-method", "move-target", "ability", "characteristic", "egg-group", "gender",
                "growth-rate", "nature", "pokeathlon-stat", "pokemon", "pokemon-color", "pokemon-form", "pokemon-habitat", "pokemon-shape",
                "pokemon-species", "stat", "type"
            ]

            if endpoint not in valid_endpoints:
                return f"Error: Invalid endpoint. Valid endpoints are: {', '.join(valid_endpoints)}"

            url = f"{BASE_URL}{endpoint}/{name_or_id}"
            if subresource:
                url += f"/{subresource}"

            response = requests.get(url)
            if response.status_code != 200:
                return f"Error: Could not find {name_or_id} in {endpoint}" + (f" with subresource {subresource}" if subresource else "")

            data = response.json()
            
            # Handle specific endpoints with custom formatting
            if endpoint == "pokemon":
                return self._format_pokemon(data, subresource)
            elif endpoint == "ability":
                return self._format_ability(data)
            elif endpoint == "move":
                return self._format_move(data)
            elif endpoint == "item":
                return self._format_item(data)
            elif endpoint == "type":
                return self._format_type(data)
            elif endpoint == "pokemon-species":
                return self._format_pokemon_species(data)
            else:
                # For other endpoints, return a summary of the data
                return f"Data for {endpoint} {name_or_id}" + (f" (subresource: {subresource})" if subresource else "") + f":\n{self._summarize_data(data)}"

        except Exception as e:
            return f"Error: {str(e)}"

    def _format_pokemon(self, data, subresource):
    # Basic information
        formatted_data = f"""Name: {data['name']}
        Height: {data['height']}
        Weight: {data['weight']}
        Types: {', '.join([t['type']['name'] for t in data['types']])}
        Abilities: {', '.join([a['ability']['name'] for a in data['abilities']])}
        Base Experience: {data['base_experience']}

        Stats:
        {self._format_stats(data['stats'])}

        Moves:
        {self._format_moves(data['moves'])}

        Location Area Encounters: {data['location_area_encounters']}
        """
        return formatted_data

    def _format_stats(self, stats):
        return "\n".join([f"{stat['stat']['name'].capitalize()}: {stat['base_stat']}" for stat in stats])

    def _format_moves(self, moves):
        level_up_moves = []
        for move in moves:
            move_name = move['move']['name']
            for version in move['version_group_details']:
                if version['move_learn_method']['name'] == "level-up":
                    level = version['level_learned_at']
                    level_up_moves.append(f"{move_name} (level {level})")
                    break
        level_up_moves.sort(key=lambda x: int(x.split('(level ')[1].split(')')[0]))
        return "\n".join(level_up_moves)

    def _format_ability(self, data):
        effect = next((e['effect'] for e in data['effect_entries'] if e['language']['name'] == 'en'), "No English description available")
        return f"""Name: {data['name']}
        Effect: {effect}
        Pokemon with this ability: {', '.join([p['pokemon']['name'] for p in data['pokemon']][:5])}"""

    def _format_move(self, data):
        effect = next((e['effect'] for e in data['effect_entries'] if e['language']['name'] == 'en'), "No English description available")
        return f"""Name: {data['name']}
        Type: {data['type']['name']}
        Power: {data['power']}
        Accuracy: {data['accuracy']}
        PP: {data['pp']}
        Effect: {effect}"""

    def _format_item(self, data):
        effect = next((e['effect'] for e in data['effect_entries'] if e['language']['name'] == 'en'), "No English description available")
        return f"""Name: {data['name']}
        Category: {data['category']['name']}
        Effect: {effect}"""

    def _format_type(self, data):
        return f"""Name: {data['name']}
        Double damage to: {', '.join([t['name'] for t in data['damage_relations']['double_damage_to']])}
        Half damage from: {', '.join([t['name'] for t in data['damage_relations']['half_damage_from']])}"""

    def _format_pokemon_species(self, data):
        return f"""Name: {data['name']}
        Generation: {data['generation']['name']}
        Capture rate: {data['capture_rate']}
        Base happiness: {data['base_happiness']}
        Growth rate: {data['growth_rate']['name']}"""

    def _summarize_data(self, data, max_items=5):
        summary = []
        for key, value in data.items():
            if isinstance(value, list) and len(value) > max_items:
                summary.append(f"{key}: {len(value)} items")
            elif isinstance(value, dict):
                summary.append(f"{key}: {self._summarize_data(value, max_items)}")
            else:
                summary.append(f"{key}: {value}")
        return ", ".join(summary[:max_items]) + ("..." if len(summary) > max_items else "")

    async def _arun(self, query: str) -> str:
        # This tool does not support async operations
        raise NotImplementedError("This tool does not support async operations")