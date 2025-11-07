"""
Unit tests for FTN cache builder functionality.

Tests cache creation, data integrity, and joining with PBP data.
"""

import pytest
from pathlib import Path
import polars as pl
from modules.ftn_cache_builder import (
    build_ftn_cache_for_year,
    load_ftn_cache,
    ftn_cache_exists,
    FTN_START_YEAR
)


class TestFTNCacheBuilder:
    """Test FTN cache builder functions"""
    
    def test_ftn_cache_exists_for_valid_years(self):
        """FTN cache should exist for 2022-2025"""
        for year in range(2022, 2026):
            assert ftn_cache_exists(year), f"FTN cache missing for {year}"
    
    def test_ftn_cache_not_exists_for_pre_2022(self):
        """FTN cache should not exist (and should return False) for pre-2022 years"""
        # build_ftn_cache_for_year returns False for pre-2022
        result = build_ftn_cache_for_year(2021)
        assert result is False, "Should return False for years before FTN_START_YEAR"
    
    def test_load_ftn_cache_returns_dataframe(self):
        """load_ftn_cache should return polars DataFrame for valid years"""
        ftn = load_ftn_cache(2024)
        assert ftn is not None, "FTN cache should load for 2024"
        assert isinstance(ftn, pl.DataFrame), "Should return polars DataFrame"
        assert len(ftn) > 0, "FTN cache should have data"
    
    def test_load_ftn_cache_pre_2022_returns_none(self):
        """load_ftn_cache should return None for pre-2022 years"""
        ftn = load_ftn_cache(2021)
        assert ftn is None, "Should return None for years before 2022"
    
    def test_ftn_cache_has_required_columns(self):
        """FTN cache should have all required columns"""
        ftn = load_ftn_cache(2024)
        
        required_cols = [
            'nflverse_game_id',
            'nflverse_play_id',
            'is_play_action',
            'is_qb_out_of_pocket',
            'n_blitzers',
            'is_contested_ball',
            'is_drop',
            'is_rpo',
            'is_screen_pass',
            'n_defense_box'
        ]
        
        for col in required_cols:
            assert col in ftn.columns, f"Missing required column: {col}"
    
    def test_ftn_flags_are_reasonable(self):
        """FTN flags should be within expected percentage ranges"""
        ftn = load_ftn_cache(2024)
        total_plays = len(ftn)
        
        # Play action: 5-15% of plays
        pa_pct = (ftn['is_play_action'].sum() / total_plays) * 100
        assert 5 < pa_pct < 15, f"Play action {pa_pct:.1f}% outside expected range (5-15%)"
        
        # Out of pocket: 7-12% of plays
        oop_pct = (ftn['is_qb_out_of_pocket'].sum() / total_plays) * 100
        assert 7 < oop_pct < 12, f"Out of pocket {oop_pct:.1f}% outside expected range (7-12%)"
        
        # Contested balls: 3-10% of plays
        contested_pct = (ftn['is_contested_ball'].sum() / total_plays) * 100
        assert 3 < contested_pct < 10, f"Contested {contested_pct:.1f}% outside expected range (3-10%)"
        
        # Drops: 0.5-3% of plays
        drop_pct = (ftn['is_drop'].sum() / total_plays) * 100
        assert 0.5 < drop_pct < 3, f"Drops {drop_pct:.1f}% outside expected range (0.5-3%)"
        
        # RPO: 1-5% of plays
        rpo_pct = (ftn['is_rpo'].sum() / total_plays) * 100
        assert 1 < rpo_pct < 5, f"RPO {rpo_pct:.1f}% outside expected range (1-5%)"
        
        # Screen pass: 2-7% of plays
        screen_pct = (ftn['is_screen_pass'].sum() / total_plays) * 100
        assert 2 < screen_pct < 7, f"Screen {screen_pct:.1f}% outside expected range (2-7%)"


class TestFTNPBPJoin:
    """Test FTN data joining with PBP data"""
    
    def test_ftn_joins_with_pbp_correctly(self):
        """FTN data should join with PBP data without errors"""
        from modules.constants import CACHE_DIR
        
        # Load PBP data using absolute path
        pbp_path = Path(CACHE_DIR) / "pbp" / "pbp_2024.parquet"
        assert pbp_path.exists(), "PBP cache must exist for test"
        
        pbp = pl.read_parquet(pbp_path)
        ftn = load_ftn_cache(2024)
        
        # Join
        joined = pbp.join(
            ftn,
            left_on=['game_id', 'play_id'],
            right_on=['nflverse_game_id', 'nflverse_play_id'],
            how='left'
        )
        
        # Should have same number of rows as PBP
        assert len(joined) == len(pbp), "Join should not change row count"
        
        # Should have FTN columns
        assert 'is_play_action' in joined.columns, "Should have FTN columns after join"
        
        # Some plays should have FTN data (not all nulls)
        pa_count = joined['is_play_action'].sum()
        assert pa_count > 0, "Should have some play action flags after join"
    
    def test_ftn_coverage_percentage(self):
        """FTN should cover a reasonable percentage of PBP plays"""
        from modules.constants import CACHE_DIR
        
        pbp_path = Path(CACHE_DIR) / "pbp" / "pbp_2024.parquet"
        pbp = pl.read_parquet(pbp_path)
        ftn = load_ftn_cache(2024)
        
        # Join
        joined = pbp.join(
            ftn,
            left_on=['game_id', 'play_id'],
            right_on=['nflverse_game_id', 'nflverse_play_id'],
            how='left'
        )
        
        # Check coverage (should be high, >95%)
        non_null = joined['is_play_action'].is_not_null().sum()
        coverage_pct = (non_null / len(joined)) * 100
        
        # FTN should cover most plays (they chart all games)
        assert coverage_pct > 90, f"FTN coverage {coverage_pct:.1f}% is too low"


class TestFTNCacheConsistency:
    """Test FTN cache consistency across years"""
    
    def test_all_years_have_consistent_schema(self):
        """All FTN caches should have identical schema"""
        schemas = []
        for year in range(2022, 2026):
            ftn = load_ftn_cache(year)
            if ftn is not None:
                schemas.append((year, ftn.columns))
        
        # All schemas should be identical
        base_schema = schemas[0][1]
        for year, schema in schemas[1:]:
            assert schema == base_schema, f"{year} schema differs from 2022"
    
    def test_all_years_have_data(self):
        """All FTN cache years should have reasonable amount of data"""
        for year in range(2022, 2026):
            ftn = load_ftn_cache(year)
            assert ftn is not None, f"FTN cache should exist for {year}"
            
            # 2025 is partial season (week 8), expect ~20K plays
            # Full seasons have 40K-48K plays
            min_plays = 15000 if year == 2025 else 35000
            assert len(ftn) > min_plays, f"{year} has suspiciously low play count: {len(ftn)}"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
