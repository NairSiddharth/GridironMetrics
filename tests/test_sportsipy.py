"""Test using sportsipy to get weather data"""

from sportsipy.nfl.boxscore import Boxscore

# Test with a 2000 game - need the boxscore ID
# Format from PFR URL: /boxscores/200009030nwe.htm
# ID is: 200009030nwe

boxscore_id = "200009030nwe"

print(f"Testing sportsipy with boxscore ID: {boxscore_id}")

try:
    game = Boxscore(boxscore_id)

    print(f"\nGame: {game.away_name} @ {game.home_name}")
    print(f"Date: {game.date}")
    print(f"Stadium: {game.stadium}")

    # Check if weather attribute exists
    if hasattr(game, 'weather'):
        print(f"Weather: {game.weather}")
    else:
        print("No weather attribute found")

    # Print all attributes
    print("\nAll available attributes:")
    attrs = [attr for attr in dir(game) if not attr.startswith('_')]
    for attr in sorted(attrs):
        try:
            val = getattr(game, attr)
            if not callable(val):
                print(f"  {attr}: {val}")
        except:
            pass

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
