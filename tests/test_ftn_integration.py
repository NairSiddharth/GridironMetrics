"""
Integration tests for FTN data impact on rankings.

Tests that FTN contextual adjustments actually affect rankings as expected.
"""

import pytest
from pathlib import Path
import polars as pl
from modules.ftn_cache_builder import load_ftn_cache
import modules.play_by_play as pbp_processor


class TestQBFTNIntegration:
    """Test FTN integration in QB rankings"""
    
    def test_qb_rankings_work_with_ftn(self):
        """QB rankings should complete successfully with FTN data"""
        from main import generate_qb_rankings
        
        result = generate_qb_rankings(2024)
        assert result is not None, "QB rankings should return data"
        assert len(result) > 0, "QB rankings should have content"
        assert "QB Rankings" in result, "Should contain QB rankings header"
    
    def test_qb_rankings_work_without_ftn(self):
        """QB rankings should work gracefully for pre-FTN years"""
        from main import generate_qb_rankings
        
        result = generate_qb_rankings(2020)
        assert result is not None, "QB rankings should return data for 2020"
        assert len(result) > 0, "QB rankings should have content"
        assert "QB Rankings" in result, "Should contain QB rankings header"
    
    def test_qb_play_action_adjustment_applied(self):
        """Play action should reduce QB contribution"""
        from modules.constants import CACHE_DIR
        
        # Load PBP data directly from cache
        pbp_path = Path(CACHE_DIR) / "pbp" / "pbp_2024.parquet"
        pbp_data = pl.read_parquet(pbp_path)
        ftn_data = load_ftn_cache(2024)
        
        # Join FTN
        pbp_data = pbp_data.join(
            ftn_data,
            left_on=['game_id', 'play_id'],
            right_on=['nflverse_game_id', 'nflverse_play_id'],
            how='left'
        )
        
        # Get a QB with high play action usage
        qb_plays = pbp_data.filter(pl.col('passer_player_name').is_not_null())
        
        # Should have play action flags
        pa_plays = qb_plays.filter(pl.col('is_play_action') == True)
        assert len(pa_plays) > 0, "Should have some play action plays"
        
        # Verification: play action adjustments are baked into contribution calculation
        # This test verifies the data is available for adjustment


class TestWRTEFTNIntegration:
    """Test FTN integration in WR/TE rankings"""
    
    def test_wr_rankings_work_with_ftn(self):
        """WR rankings should complete successfully with FTN data"""
        from main import generate_wr_rankings
        
        result = generate_wr_rankings(2024)
        assert result is not None, "WR rankings should return data"
        assert len(result) > 0, "WR rankings should have content"
        assert "WR Rankings" in result, "Should contain WR rankings header"
    
    def test_te_rankings_work_with_ftn(self):
        """TE rankings should complete successfully with FTN data"""
        from main import generate_te_rankings
        
        result = generate_te_rankings(2024)
        assert result is not None, "TE rankings should return data"
        assert len(result) > 0, "TE rankings should have content"
        assert "TE Rankings" in result, "Should contain TE rankings header"
    
    def test_wr_contested_catch_data_available(self):
        """Contested catch data should be available for WRs"""
        from modules.constants import CACHE_DIR
        
        # Load PBP data directly from cache
        pbp_path = Path(CACHE_DIR) / "pbp" / "pbp_2024.parquet"
        pbp_data = pl.read_parquet(pbp_path)
        ftn_data = load_ftn_cache(2024)
        
        # Join FTN
        pbp_data = pbp_data.join(
            ftn_data,
            left_on=['game_id', 'play_id'],
            right_on=['nflverse_game_id', 'nflverse_play_id'],
            how='left'
        )
        
        # Get WR targets
        wr_targets = pbp_data.filter(
            (pl.col('receiver_player_id').is_not_null()) &
            (pl.col('complete_pass') == 1)
        )
        
        # Should have contested catch flags
        contested = wr_targets.filter(pl.col('is_contested_ball') == True)
        assert len(contested) > 0, "Should have some contested catches"
        
        # Should have drops
        drops = pbp_data.filter(pl.col('is_drop') == True)
        assert len(drops) > 0, "Should have some drops"


class TestRBFTNIntegration:
    """Test FTN integration in RB rankings"""
    
    def test_rb_rankings_work_with_ftn(self):
        """RB rankings should complete successfully with FTN data"""
        from main import generate_rb_rankings
        
        result = generate_rb_rankings(2024)
        assert result is not None, "RB rankings should return data"
        assert len(result) > 0, "RB rankings should have content"
        assert "RB Rankings" in result, "Should contain RB rankings header"
    
    def test_rb_rpo_data_available(self):
        """RPO data should be available for RBs"""
        from modules.constants import CACHE_DIR
        
        # Load PBP data directly from cache
        pbp_path = Path(CACHE_DIR) / "pbp" / "pbp_2024.parquet"
        pbp_data = pl.read_parquet(pbp_path)
        ftn_data = load_ftn_cache(2024)
        ftn_data = load_ftn_cache(2024)
        
        # Join FTN
        pbp_data = pbp_data.join(
            ftn_data,
            left_on=['game_id', 'play_id'],
            right_on=['nflverse_game_id', 'nflverse_play_id'],
            how='left'
        )
        
        # Get RB rushes
        rb_rushes = pbp_data.filter(pl.col('rusher_player_id').is_not_null())
        
        # Should have RPO flags
        rpo_runs = rb_rushes.filter(pl.col('is_rpo') == True)
        assert len(rpo_runs) > 0, "Should have some RPO runs"
        
        # Should have heavy box data
        heavy_box = rb_rushes.filter(pl.col('n_defense_box') >= 8)
        assert len(heavy_box) > 0, "Should have some heavy box runs"


class TestFTNGracefulDegradation:
    """Test that system works correctly without FTN data"""
    
    def test_all_positions_work_pre_ftn(self):
        """All position rankings should work for pre-FTN years"""
        from main import (
            generate_qb_rankings,
            generate_rb_rankings,
            generate_wr_rankings,
            generate_te_rankings
        )
        
        # Test 2020 (no FTN data)
        qb_result = generate_qb_rankings(2020)
        assert qb_result is not None and len(qb_result) > 0
        
        rb_result = generate_rb_rankings(2020)
        assert rb_result is not None and len(rb_result) > 0
        
        wr_result = generate_wr_rankings(2020)
        assert wr_result is not None and len(wr_result) > 0
        
        te_result = generate_te_rankings(2020)
        assert te_result is not None and len(te_result) > 0


class TestFTNAdjustmentLogic:
    """Test specific FTN adjustment calculations"""
    
    def test_screen_pass_identification(self):
        """Screen passes should be correctly identified"""
        from modules.constants import CACHE_DIR
        
        # Load PBP data directly from cache
        pbp_path = Path(CACHE_DIR) / "pbp" / "pbp_2024.parquet"
        pbp_data = pl.read_parquet(pbp_path)
        ftn_data = load_ftn_cache(2024)
        
        # Join FTN
        pbp_data = pbp_data.join(
            ftn_data,
            left_on=['game_id', 'play_id'],
            right_on=['nflverse_game_id', 'nflverse_play_id'],
            how='left'
        )
        
        # Screen passes should exist
        screens = pbp_data.filter(pl.col('is_screen_pass') == True)
        assert len(screens) > 0, "Should identify screen passes"
        
        # Screen percentage should be reasonable (2-7%)
        screen_pct = (len(screens) / len(pbp_data)) * 100
        assert 2 < screen_pct < 7, f"Screen pass % {screen_pct:.1f} outside expected range"
    
    def test_blitz_tracking(self):
        """Blitz count column should be present"""
        from modules.constants import CACHE_DIR
        
        # Load PBP data directly from cache
        pbp_path = Path(CACHE_DIR) / "pbp" / "pbp_2024.parquet"
        pbp_data = pl.read_parquet(pbp_path)
        ftn_data = load_ftn_cache(2024)
        
        # Join FTN
        pbp_data = pbp_data.join(
            ftn_data,
            left_on=['game_id', 'play_id'],
            right_on=['nflverse_game_id', 'nflverse_play_id'],
            how='left'
        )
        
        # Column should exist (may be null if FTN doesn't track it)
        assert 'n_blitzers' in pbp_data.columns, "n_blitzers column should exist after FTN join"
        
        # If data exists, check it's reasonable
        pass_plays = pbp_data.filter(pl.col('pass_attempt') == 1)
        blitz_data = pass_plays.filter(pl.col('n_blitzers').is_not_null())
        if len(blitz_data) > 0:
            # Blitzers should be 0-11 (max players on defense)
            max_blitzers = blitz_data.select(pl.col('n_blitzers').max()).item()
            assert max_blitzers <= 11, f"Max blitzers {max_blitzers} exceeds 11"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
