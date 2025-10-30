import nflreadpy as nfl

ftn = nfl.load_ftn_charting(seasons=2024)

print("FTN Charting Data Overview")
print("=" * 60)
print(f"Total plays: {len(ftn)}")
print()

# Check each boolean column
bool_cols = [
    'is_play_action',
    'is_rpo', 
    'is_screen_pass',
    'is_qb_out_of_pocket',
    'is_interception_worthy',
    'is_qb_fault_sack',
    'is_contested_ball',
    'is_drop',
    'is_created_reception',
    'is_catchable_ball'
]

print("Boolean Flags Distribution:")
print("-" * 60)
for col in bool_cols:
    true_count = ftn[col].sum()
    false_count = len(ftn) - true_count
    pct = (true_count / len(ftn)) * 100
    print(f"{col:30s}: True={true_count:5d} ({pct:5.2f}%), False={false_count:5d}")

print()
print("Numeric columns:")
print("-" * 60)
print(f"n_blitzers - Min: {ftn['n_blitzers'].min()}, Max: {ftn['n_blitzers'].max()}, Avg: {ftn['n_blitzers'].mean():.2f}")
print(f"n_pass_rushers - Min: {ftn['n_pass_rushers'].min()}, Max: {ftn['n_pass_rushers'].max()}, Avg: {ftn['n_pass_rushers'].mean():.2f}")
print(f"n_defense_box - Min: {ftn['n_defense_box'].min()}, Max: {ftn['n_defense_box'].max()}, Avg: {ftn['n_defense_box'].mean():.2f}")

print()
print("Sample play with interception_worthy:")
print("-" * 60)
int_worthy = ftn.filter(ftn['is_interception_worthy'] == True).head(1)
print(int_worthy)

print()
print("Sample play with contested_ball:")
print("-" * 60)
contested = ftn.filter(ftn['is_contested_ball'] == True).head(1)
print(contested)

print()
print("Sample play with qb_fault_sack:")
print("-" * 60)
qb_fault = ftn.filter(ftn['is_qb_fault_sack'] == True).head(1)
print(qb_fault)
