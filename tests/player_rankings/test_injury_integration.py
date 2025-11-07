from modules.injury_cache_builder import (
    get_player_gsis_id,
    calculate_injury_adjusted_games,
    count_games_missed_due_to_injury
)

# Test CMC
gsis_id = get_player_gsis_id('C.McCaffrey', 'SF', 'RB', 2024)
print(f'CMC GSIS ID: {gsis_id}')

for year in [2024, 2023, 2022]:
    result = count_games_missed_due_to_injury(gsis_id, year)
    print(f'{year}: {result["games_played"]} played, {result["injury_missed"]} injured')

eff = calculate_injury_adjusted_games(gsis_id, 2024, 4, 17)
print(f'\nEffective games: {eff:.2f} (from 4 actual games)')

# Test Derrick Henry (reliable)
henry_id = get_player_gsis_id('D.Henry', 'BAL', 'RB', 2024)
print(f'\nDerrick Henry GSIS ID: {henry_id}')

for year in [2024, 2023, 2022]:
    result = count_games_missed_due_to_injury(henry_id, year)
    print(f'{year}: {result["games_played"]} played, {result["injury_missed"]} injured')

henry_eff = calculate_injury_adjusted_games(henry_id, 2024, 17, 17)
print(f'\nEffective games: {henry_eff:.2f} (from 17 actual games)')
