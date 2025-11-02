"""Analyze the weather data files"""
import polars as pl

# Read weather data
weather = pl.read_csv('games_weather.csv', ignore_errors=True)
games = pl.read_csv('games.csv', ignore_errors=True)

print("=== GAMES_WEATHER.CSV ===")
print(f"Shape: {weather.shape}")
print(f"Columns: {weather.columns}")
print("\nSample rows:")
for row in weather.head(3).iter_rows(named=True):
    print(row)

print("\n\n=== GAMES.CSV ===")
print(f"Shape: {games.shape}")
print(f"Columns: {games.columns}")
print("\nSample rows:")
for row in games.head(3).iter_rows(named=True):
    print(row)

# Check year coverage
print("\n\n=== COVERAGE ===")
if 'schedule_season' in games.columns:
    years = games['schedule_season'].unique().sort()
    print(f"Years covered: {years.to_list()}")
elif 'season' in games.columns:
    years = games['season'].unique().sort()
    print(f"Years covered: {years.to_list()}")

# Check how many games have weather
print(f"\nTotal games: {len(games)}")
print(f"Games with weather data: {len(weather)}")
