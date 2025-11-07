import polars as pl

df_2025 = pl.read_parquet('cache/pbp/pbp_2025.parquet')
df_2024 = pl.read_parquet('cache/pbp/pbp_2024.parquet')

print('=== 2025 NEGATIVE PLAY DATA ===')
sacks_2025 = df_2025.filter(pl.col('sack') == 1)
print(f'Total sacks: {len(sacks_2025)}')
qb_sacks = df_2025.filter((pl.col('sack') == 1) & (pl.col('passer_player_name').is_not_null()))
print(f'Sacks with QB identified: {len(qb_sacks)}')

ints_2025 = df_2025.filter(pl.col('interception') == 1)
print(f'Total interceptions: {len(ints_2025)}')
qb_ints = df_2025.filter((pl.col('interception') == 1) & (pl.col('passer_player_name').is_not_null()))
print(f'INTs with QB identified: {len(qb_ints)}')

fumbles_2025 = df_2025.filter(pl.col('fumble_lost') == 1)
print(f'Total fumbles lost: {len(fumbles_2025)}')

print('\n=== 2024 NEGATIVE PLAY DATA ===')
sacks_2024 = df_2024.filter(pl.col('sack') == 1)
print(f'Total sacks: {len(sacks_2024)}')
ints_2024 = df_2024.filter(pl.col('interception') == 1)
print(f'Total interceptions: {len(ints_2024)}')
fumbles_2024 = df_2024.filter(pl.col('fumble_lost') == 1)
print(f'Total fumbles lost: {len(fumbles_2024)}')

print('\n=== Sample 2025 sack with situation ===')
if len(qb_sacks) > 0:
    sample = qb_sacks.select([
        'passer_player_name', 'down', 'ydstogo', 
        'score_differential', 'game_seconds_remaining', 'yardline_100'
    ]).head(3)
    print(sample)

print('\n=== Sample 2025 INT with situation ===')
if len(qb_ints) > 0:
    sample = qb_ints.select([
        'passer_player_name', 'down', 'ydstogo', 'air_yards',
        'score_differential', 'game_seconds_remaining', 'yardline_100'
    ]).head(3)
    print(sample)
