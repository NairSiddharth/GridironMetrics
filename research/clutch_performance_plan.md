# Clutch Performance Adjustment Implementation Plan

## Objective
To implement a clutch performance adjustment metric that evaluates players based on:
1. **Strength of Opponent**: Adjust based on the quality of the opposing team.
2. **Margin of Loss**: Reflects how close the game was.
3. **Time Left**: Accounts for the difficulty of the situation based on the remaining time during a game-winning drive (GWD).

---

## Integration Plan

### 1. Code Integration

#### Module Placement
- Add a new module `clutch_metrics.py` under the `modules/` directory to encapsulate all logic related to clutch performance.
- Update `main.py` to integrate this module into the appropriate phase of the pipeline.

#### Naming Conventions
- Follow the existing naming conventions (e.g., `positional_cache_builder.py`).
- Functions will be prefixed with `calculate_` or `get_` for clarity.

#### Pipeline Integration
- Integrate the clutch adjustment calculation into the existing pipeline in `main.py`:
  - **Phase 1**: Fetch and cache required data.
  - **Phase 2**: Calculate clutch metrics using the new module.
  - **Phase 3**: Integrate the clutch metrics into the final player evaluation.

---

### 2. Data Requirements

#### From `sportsipy` API:
- **Strength of Opponent**:
  - Use team statistics (e.g., win-loss record, defensive/offensive rankings) to determine the strength of the opposing team.
  - Cache this data in `cache/team_stats/`.

#### From `nflreadpy`:
- **Time Left**:
  - Extract play-by-play (PBP) data to determine the time left when a GWD attempt begins.
  - Use `nflreadpy` to fetch game data and cross-reference with PBP data for time left.
  - Cache this data in `cache/pbp/`.

- **Margin of Loss**:
  - Extract game results to calculate the margin of loss for each GWD attempt.
  - Cache this data in `cache/game_results/`.

#### Player Matching
- Match players using:
  - `gsis_id` (preferred for precision).
  - If unavailable, fallback to `team`, `game_id`, and `F.Lastname` matching.

---

### 3. Caching Strategy

#### Cache Structure
- Use the existing `cache/` directory structure.
- Add subdirectories for new data:
  - `cache/team_stats/`
  - `cache/pbp/`
  - `cache/game_results/`

#### Cache Validation
- Implement a timestamp-based validation to ensure cached data is refreshed periodically (e.g., weekly).

#### Avoid Redundant API Calls
- Check the cache before making any API calls.
- If data is missing or outdated, fetch and update the cache.

---

## Implementation Steps

### 1. Create `clutch_metrics.py`

#### Functions:
- `fetch_team_stats()`: Fetch and cache team statistics from `sportsipy`.
- `fetch_pbp_data()`: Fetch and cache play-by-play data using `nflreadpy`.
- `calculate_clutch_score(player_id, game_id)`: Calculate the clutch score for a player based on:
  - Strength of opponent.
  - Margin of loss.
  - Time left.

### 2. Update `main.py`
- Add a new phase to integrate clutch metrics:
  - Fetch required data (team stats, PBP, game results).
  - Calculate clutch scores for relevant players.
  - Integrate clutch scores into the final player evaluation.

### 3. Player Matching
- Use `gsis_id` for precise matching.
- If `gsis_id` is unavailable:
  - Match using `team`, `game_id`, and `F.Lastname`.

---

## Testing Plan

### Unit Tests

#### Module: `clutch_metrics.py`
- Test `fetch_team_stats()` with mock API responses.
- Test `fetch_pbp_data()` with mock `nflreadpy` responses.
- Test `calculate_clutch_score()` with various scenarios:
  - Different time buckets (<1 min, 1-2 min, >2 min).
  - Different margins of loss.
  - Different opponent strengths.

### Integration Tests

#### Pipeline: `main.py`
- Test the full pipeline with mock data to ensure clutch metrics are calculated and integrated correctly.

### End-to-End Tests
- Run the entire pipeline with real data to validate:
  - Correct caching of data.
  - Accurate player matching.
  - Reasonable clutch score adjustments.

---

## Next Steps
1. Implement `clutch_metrics.py` with the outlined functions.
2. Update `main.py` to integrate the new module.
3. Write and execute unit tests for `clutch_metrics.py`.
4. Validate the implementation with integration and end-to-end tests.