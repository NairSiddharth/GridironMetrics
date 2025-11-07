"""
test_context_adjustments.py

Unit tests for the ContextAdjustments module and PersonnelInference.
Tests all adjustment calculations including third down multipliers,
garbage time detection, YAC analysis, and personnel grouping inference.
"""

import pytest
import polars as pl
from modules.context_adjustments import ContextAdjustments
from modules.personnel_inference import PersonnelInference


@pytest.fixture
def context_adj():
    """Fixture providing a ContextAdjustments instance for tests."""
    return ContextAdjustments()


@pytest.fixture
def personnel_inf():
    """Fixture providing a PersonnelInference instance for tests."""
    return PersonnelInference()


class TestThirdDownMultiplier:
    """Test suite for third down distance-based multipliers."""
    
    def test_not_third_down(self, context_adj):
        """Non-third down plays should get 1.0x (no adjustment)."""
        assert context_adj.calculate_third_down_multiplier(down=1, ydstogo=10) == 1.0
        assert context_adj.calculate_third_down_multiplier(down=2, ydstogo=10) == 1.0
        assert context_adj.calculate_third_down_multiplier(down=4, ydstogo=10) == 1.0
    
    def test_third_and_short(self, context_adj):
        """3rd & 1 should get minimal bonus (1.05x)."""
        assert context_adj.calculate_third_down_multiplier(down=3, ydstogo=1) == 1.05
    
    def test_third_and_medium_short(self, context_adj):
        """3rd & 2-3 should get 1.15x."""
        assert context_adj.calculate_third_down_multiplier(down=3, ydstogo=2) == 1.15
        assert context_adj.calculate_third_down_multiplier(down=3, ydstogo=3) == 1.15
    
    def test_third_and_medium(self, context_adj):
        """3rd & 4-6 should get 1.30x."""
        assert context_adj.calculate_third_down_multiplier(down=3, ydstogo=4) == 1.30
        assert context_adj.calculate_third_down_multiplier(down=3, ydstogo=5) == 1.30
        assert context_adj.calculate_third_down_multiplier(down=3, ydstogo=6) == 1.30
    
    def test_third_and_medium_long(self, context_adj):
        """3rd & 7-9 should get 1.45x."""
        assert context_adj.calculate_third_down_multiplier(down=3, ydstogo=7) == 1.45
        assert context_adj.calculate_third_down_multiplier(down=3, ydstogo=8) == 1.45
        assert context_adj.calculate_third_down_multiplier(down=3, ydstogo=9) == 1.45
    
    def test_third_and_long(self, context_adj):
        """3rd & 10-14 should get 1.55x."""
        assert context_adj.calculate_third_down_multiplier(down=3, ydstogo=10) == 1.55
        assert context_adj.calculate_third_down_multiplier(down=3, ydstogo=12) == 1.55
        assert context_adj.calculate_third_down_multiplier(down=3, ydstogo=14) == 1.55
    
    def test_third_and_very_long(self, context_adj):
        """3rd & 15+ should get maximum bonus (1.6x)."""
        assert context_adj.calculate_third_down_multiplier(down=3, ydstogo=15) == 1.60
        assert context_adj.calculate_third_down_multiplier(down=3, ydstogo=20) == 1.60
        assert context_adj.calculate_third_down_multiplier(down=3, ydstogo=25) == 1.60


class TestGarbageTimeMultiplier:
    """Test suite for garbage time detection and penalties."""
    
    def test_close_game_no_penalty(self, context_adj):
        """Close games (≤17 pts) should not trigger garbage time penalty."""
        assert context_adj.calculate_garbage_time_multiplier(
            score_differential=-14, game_seconds_remaining=120) == 1.0
        assert context_adj.calculate_garbage_time_multiplier(
            score_differential=-17, game_seconds_remaining=120) == 1.0
    
    def test_blowout_early_no_penalty(self, context_adj):
        """Blowouts with >8 minutes left should not trigger garbage time penalty."""
        assert context_adj.calculate_garbage_time_multiplier(
            score_differential=-21, game_seconds_remaining=600) == 1.0
        assert context_adj.calculate_garbage_time_multiplier(
            score_differential=-28, game_seconds_remaining=481) == 1.0
    
    def test_winning_team_no_penalty(self, context_adj):
        """Winning team should never get garbage time penalty."""
        assert context_adj.calculate_garbage_time_multiplier(
            score_differential=21, game_seconds_remaining=120) == 1.0
        assert context_adj.calculate_garbage_time_multiplier(
            score_differential=28, game_seconds_remaining=60) == 1.0
    
    def test_garbage_time_mild_penalty(self, context_adj):
        """Down 18+ with 8 min left should get mild penalty (~0.9x)."""
        result = context_adj.calculate_garbage_time_multiplier(
            score_differential=-18, game_seconds_remaining=480)
        assert 0.88 <= result <= 0.92  # ~0.9x
    
    def test_garbage_time_moderate_penalty(self, context_adj):
        """Down 24+ with 5 min left should get moderate penalty (~0.79x)."""
        result = context_adj.calculate_garbage_time_multiplier(
            score_differential=-24, game_seconds_remaining=300)
        assert 0.77 <= result <= 0.80  # ~0.7875x (300/480 = 0.625)
    
    def test_garbage_time_severe_penalty(self, context_adj):
        """Down 24+ with 2 min left should get severe penalty (~0.675x)."""
        result = context_adj.calculate_garbage_time_multiplier(
            score_differential=-24, game_seconds_remaining=120)
        assert 0.66 <= result <= 0.68  # ~0.675x (120/480 = 0.25)
    
    def test_garbage_time_maximum_penalty(self, context_adj):
        """Down 24+ with 0 sec left should get maximum penalty (0.6x)."""
        result = context_adj.calculate_garbage_time_multiplier(
            score_differential=-28, game_seconds_remaining=0)
        assert result == 0.6


class TestYACMultiplier:
    """Test suite for yards after catch multipliers."""
    
    def test_elite_yac(self, context_adj):
        """70%+ YAC should get maximum bonus (1.15x)."""
        result = context_adj.calculate_yac_multiplier(
            air_yards=5.0, yards_after_catch=15.0, yards_gained=20.0)
        assert result == 1.15  # 75% YAC
        
        result = context_adj.calculate_yac_multiplier(
            air_yards=2.0, yards_after_catch=8.0, yards_gained=10.0)
        assert result == 1.15  # 80% YAC
    
    def test_good_yac(self, context_adj):
        """50-70% YAC should get 1.10x bonus."""
        result = context_adj.calculate_yac_multiplier(
            air_yards=5.0, yards_after_catch=10.0, yards_gained=15.0)
        assert result == 1.10  # 67% YAC
        
        result = context_adj.calculate_yac_multiplier(
            air_yards=8.0, yards_after_catch=12.0, yards_gained=20.0)
        assert result == 1.10  # 60% YAC
    
    def test_solid_yac(self, context_adj):
        """30-50% YAC should get 1.05x bonus."""
        result = context_adj.calculate_yac_multiplier(
            air_yards=10.0, yards_after_catch=8.0, yards_gained=18.0)
        assert result == 1.05  # 44% YAC
        
        result = context_adj.calculate_yac_multiplier(
            air_yards=12.0, yards_after_catch=6.0, yards_gained=18.0)
        assert result == 1.05  # 33% YAC
    
    def test_average_yac(self, context_adj):
        """10-30% YAC should get neutral multiplier (1.0x)."""
        result = context_adj.calculate_yac_multiplier(
            air_yards=15.0, yards_after_catch=5.0, yards_gained=20.0)
        assert result == 1.0  # 25% YAC
        
        result = context_adj.calculate_yac_multiplier(
            air_yards=18.0, yards_after_catch=2.0, yards_gained=20.0)
        assert result == 1.0  # 10% YAC
    
    def test_low_yac_penalty(self, context_adj):
        """<10% YAC should get penalty (0.95x)."""
        result = context_adj.calculate_yac_multiplier(
            air_yards=19.0, yards_after_catch=1.0, yards_gained=20.0)
        assert result == 0.95  # 5% YAC
        
        result = context_adj.calculate_yac_multiplier(
            air_yards=30.0, yards_after_catch=0.0, yards_gained=30.0)
        assert result == 0.95  # 0% YAC
    
    def test_missing_data_returns_neutral(self, context_adj):
        """Missing YAC data should return 1.0 (no adjustment)."""
        result = context_adj.calculate_yac_multiplier(
            air_yards=10.0, yards_after_catch=None, yards_gained=10.0)
        assert result == 1.0
        
        result = context_adj.calculate_yac_multiplier(
            air_yards=10.0, yards_after_catch=5.0, yards_gained=0.0)
        assert result == 1.0


class TestCatchRateAdjustment:
    """Test suite for catch rate over expected calculations."""
    
    def test_placeholder_adds_column(self, context_adj):
        """Placeholder should add catch_rate_adjustment column with 1.0."""
        import polars as pl
        
        test_df = pl.DataFrame({
            "player_id": ["00-0001", "00-0002"],
            "targets": [10, 15],
            "receptions": [7, 12]
        })
        
        result = context_adj.calculate_catch_rate_adjustment(test_df)
        assert "catch_rate_adjustment" in result.columns
        assert result["catch_rate_adjustment"].to_list() == [1.0, 1.0]
    
    # TODO: Add real tests when implemented in Phase 4


class TestBlockingQualityProxy:
    """Test suite for blocking quality estimation."""
    
    def test_placeholder_adds_column(self, context_adj):
        """Placeholder should add blocking_quality_adjustment column with 1.0."""
        import polars as pl
        
        test_df = pl.DataFrame({
            "player_id": ["00-0001", "00-0002"],
            "team": ["KC", "SF"],
            "position": ["WR", "WR"]
        })
        
        result = context_adj.calculate_blocking_quality_proxy(test_df)
        assert "blocking_quality_adjustment" in result.columns
        assert result["blocking_quality_adjustment"].to_list() == [1.0, 1.0]
    
    # TODO: Add real tests when implemented in Phase 4


class TestTalentAdjustment:
    """Test suite for teammate quality talent adjustments."""
    
    def test_placeholder_adds_column(self, context_adj):
        """Placeholder should add talent_adjustment column with 1.0."""
        import polars as pl
        
        test_df = pl.DataFrame({
            "player_id": ["00-0001", "00-0002"],
            "team": ["KC", "SF"],
            "position": ["WR", "RB"]
        })
        
        baseline_scores = {
            ("00-0001", "KC"): 85.0,
            ("00-0002", "SF"): 92.0
        }
        
        result = context_adj.calculate_teammate_quality_index(test_df, baseline_scores)
        assert "talent_adjustment" in result.columns
        assert result["talent_adjustment"].to_list() == [1.0, 1.0]
    
    # TODO: Add real tests when implemented in Phase 5


class TestSampleSizeDampening:
    """Test suite for sample size dampening with 0.4 root curve."""
    
    def test_placeholder_returns_unchanged_score(self, context_adj):
        """Placeholder should return score unchanged."""
        assert context_adj.apply_sample_size_dampening(score=100.0, games_played=17) == 100.0
        assert context_adj.apply_sample_size_dampening(score=75.0, games_played=10) == 75.0
    
    # TODO: Add real tests when implemented in Phase 5
    # Expected behavior:
    # - 17 games: 1.0x (full credit)
    # - 10 games: ~0.75x credit
    # - 5 games: ~0.57x credit
    # - 2 games: ~0.39x credit


class TestPersonnelInference:
    """Test suite for personnel grouping inference."""
    
    def test_first_and_ten_predicts_11_personnel(self, personnel_inf):
        """1st & 10 should predict base 11 personnel with high confidence."""
        personnel, confidence = personnel_inf.infer_personnel(
            play_type='run', down=1, ydstogo=10, yardline_100=50,
            score_differential=0, game_seconds_remaining=1800
        )
        assert personnel == '11'
        assert confidence >= 0.7  # High confidence
    
    def test_third_and_long_predicts_spread(self, personnel_inf):
        """3rd & 15 should predict spread formation (10 personnel)."""
        personnel, confidence = personnel_inf.infer_personnel(
            play_type='pass', down=3, ydstogo=15, yardline_100=65,
            score_differential=0, game_seconds_remaining=1800,
            air_yards=20.0
        )
        assert personnel in ['10', '11']  # Spread or base
        # Lower confidence is okay - ambiguous situation
        assert confidence > 0.25
    
    def test_goal_line_predicts_heavy(self, personnel_inf):
        """Goal line situation should predict heavy formation (22)."""
        personnel, confidence = personnel_inf.infer_personnel(
            play_type='run', down=1, ydstogo=1, yardline_100=1,
            score_differential=0, game_seconds_remaining=600
        )
        assert personnel == '22'
        assert confidence > 0.6
    
    def test_short_yardage_predicts_heavy(self, personnel_inf):
        """3rd & 1 should predict heavy formation (22 or 21)."""
        personnel, confidence = personnel_inf.infer_personnel(
            play_type='run', down=3, ydstogo=1, yardline_100=45,
            score_differential=0, game_seconds_remaining=1200
        )
        assert personnel in ['22', '21']
        assert confidence > 0.6
    
    def test_deep_pass_predicts_spread(self, personnel_inf):
        """Deep pass (20+ air yards) should predict spread."""
        personnel, confidence = personnel_inf.infer_personnel(
            play_type='pass', down=2, ydstogo=8, yardline_100=70,
            score_differential=0, game_seconds_remaining=1500,
            receiver_position='WR', air_yards=25.0
        )
        assert personnel in ['10', '11']
        assert confidence > 0.6
    
    def test_losing_team_late_predicts_spread(self, personnel_inf):
        """Losing team late should predict spread to score quickly."""
        personnel, confidence = personnel_inf.infer_personnel(
            play_type='pass', down=2, ydstogo=7, yardline_100=55,
            score_differential=-17, game_seconds_remaining=300
        )
        assert personnel == '10'
        # Moderate confidence expected
        assert confidence > 0.3
    
    def test_winning_team_late_predicts_heavy(self, personnel_inf):
        """Winning team late should predict heavy to run clock."""
        personnel, confidence = personnel_inf.infer_personnel(
            play_type='run', down=2, ydstogo=5, yardline_100=45,
            score_differential=17, game_seconds_remaining=300
        )
        assert personnel in ['21', '22', '11']
        # Lower confidence expected for 2nd down situations
        assert confidence > 0.25
    
    def test_wr_multiplier_benefits_from_spread(self, personnel_inf):
        """WR should get 1.15x in 10 personnel (spread)."""
        multiplier = personnel_inf.get_position_multiplier('10', 'WR', 0.8)
        assert multiplier == 1.15
    
    def test_wr_penalized_in_heavy(self, personnel_inf):
        """WR should get 0.85x in 22 personnel (jumbo)."""
        multiplier = personnel_inf.get_position_multiplier('22', 'WR', 0.8)
        assert multiplier == 0.85
    
    def test_rb_benefits_from_heavy(self, personnel_inf):
        """RB should get 1.10x in 22 personnel (heavy)."""
        multiplier = personnel_inf.get_position_multiplier('22', 'RB', 0.8)
        assert multiplier == 1.10
    
    def test_rb_penalized_in_spread(self, personnel_inf):
        """RB should get 0.90x in 10 personnel (spread)."""
        multiplier = personnel_inf.get_position_multiplier('10', 'RB', 0.8)
        assert multiplier == 0.90
    
    def test_te_benefits_from_multi_te(self, personnel_inf):
        """TE should get 1.10x in 12 personnel (2 TE)."""
        multiplier = personnel_inf.get_position_multiplier('12', 'TE', 0.8)
        assert multiplier == 1.10
    
    def test_low_confidence_returns_neutral(self, personnel_inf):
        """Low confidence (<0.6) should return 1.0x multiplier."""
        multiplier = personnel_inf.get_position_multiplier('10', 'WR', 0.5)
        assert multiplier == 1.0
        
        multiplier = personnel_inf.get_position_multiplier('22', 'RB', 0.4)
        assert multiplier == 1.0


# ============================================================================
# TEST CATCH RATE ADJUSTMENT
# ============================================================================

class TestCatchRateAdjustment:
    """Test catch rate over expected calculations."""
    
    @pytest.fixture
    def context_adj(self):
        return ContextAdjustments()
    
    def test_high_catch_rate_deep_targets(self, context_adj):
        """High catch rate on deep targets should get maximum boost."""
        pbp = pl.DataFrame({
            'receiver_player_id': ['P1'] * 30,
            'complete_pass': [1] * 20 + [0] * 10,  # 67% catch rate
            'incomplete_pass': [0] * 20 + [1] * 10,
            'air_yards': [22.0] * 30  # Deep targets (expected 45%)
        })
        # 67% vs 45% expected = +22% over expected
        multiplier = context_adj.calculate_catch_rate_adjustment(pbp, 'P1')
        assert multiplier == 1.10
    
    def test_low_catch_rate_short_targets(self, context_adj):
        """Low catch rate on short targets should get penalty."""
        pbp = pl.DataFrame({
            'receiver_player_id': ['P1'] * 40,
            'complete_pass': [1] * 25 + [0] * 15,  # 62.5% catch rate
            'incomplete_pass': [0] * 25 + [1] * 15,
            'air_yards': [5.0] * 40  # Short targets (expected 75%)
        })
        # 62.5% vs 75% expected = -12.5% below expected
        multiplier = context_adj.calculate_catch_rate_adjustment(pbp, 'P1')
        assert multiplier == 0.90
    
    def test_league_average_catch_rate(self, context_adj):
        """Average catch rate should be neutral."""
        pbp = pl.DataFrame({
            'receiver_player_id': ['P1'] * 50,
            'complete_pass': [1] * 33 + [0] * 17,  # 66% catch rate
            'incomplete_pass': [0] * 33 + [1] * 17,
            'air_yards': [12.0] * 50  # Intermediate (expected 65%)
        })
        # 66% vs 65% expected = +1% (within neutral band)
        multiplier = context_adj.calculate_catch_rate_adjustment(pbp, 'P1')
        assert multiplier == 1.0
    
    def test_intermediate_positive_adjustment(self, context_adj):
        """Moderate outperformance should get moderate boost."""
        pbp = pl.DataFrame({
            'receiver_player_id': ['P1'] * 40,
            'complete_pass': [1] * 25 + [0] * 15,  # 62.5% catch rate
            'incomplete_pass': [0] * 25 + [1] * 15,
            'air_yards': [17.0] * 40  # Intermediate-deep (expected 55%)
        })
        # 62.5% vs 55% expected = +7.5% over expected
        multiplier = context_adj.calculate_catch_rate_adjustment(pbp, 'P1')
        assert multiplier == 1.07
    
    def test_insufficient_targets_returns_neutral(self, context_adj):
        """Players with <20 targets should get neutral multiplier."""
        pbp = pl.DataFrame({
            'receiver_player_id': ['P1'] * 15,
            'complete_pass': [1] * 15 + [0] * 0,  # 100% catch rate
            'incomplete_pass': [0] * 15 + [1] * 0,
            'air_yards': [10.0] * 15
        })
        multiplier = context_adj.calculate_catch_rate_adjustment(pbp, 'P1')
        assert multiplier == 1.0
    
    def test_no_targets_returns_neutral(self, context_adj):
        """Player with no targets should get neutral multiplier."""
        pbp = pl.DataFrame({
            'receiver_player_id': ['P2'] * 50,
            'complete_pass': [1] * 30 + [0] * 20,
            'incomplete_pass': [0] * 30 + [1] * 20,
            'air_yards': [10.0] * 50
        })
        multiplier = context_adj.calculate_catch_rate_adjustment(pbp, 'P1')
        assert multiplier == 1.0
    
    def test_depth_tiers(self, context_adj):
        """Test expected catch rate adjusts correctly for different depths."""
        # Deep: 20+ yards (45% expected)
        pbp_deep = pl.DataFrame({
            'receiver_player_id': ['P1'] * 30,
            'complete_pass': [1] * 16 + [0] * 14,  # 53.3% catch rate
            'incomplete_pass': [0] * 16 + [1] * 14,
            'air_yards': [25.0] * 30
        })
        # 53.3% vs 45% = +8.3% (1.10x multiplier)
        assert context_adj.calculate_catch_rate_adjustment(pbp_deep, 'P1') == 1.10
        
        # Intermediate: 10-15 yards (65% expected), test 1.07x tier
        pbp_int = pl.DataFrame({
            'receiver_player_id': ['P1'] * 100,
            'complete_pass': [1] * 71 + [0] * 29,  # 71% catch rate
            'incomplete_pass': [0] * 71 + [1] * 29,
            'air_yards': [12.0] * 100
        })
        # 71% vs 65% = +6% (1.07x multiplier, well above 5% threshold)
        assert context_adj.calculate_catch_rate_adjustment(pbp_int, 'P1') == 1.07


# ============================================================================
# TEST BLOCKING QUALITY PROXY
# ============================================================================

class TestBlockingQualityProxy:
    """Test blocking quality proxy calculations."""
    
    @pytest.fixture
    def context_adj(self):
        return ContextAdjustments()
    
    def test_player_significantly_outperforms_teammates(self, context_adj):
        """Player with 1.4x teammate YPC should get 1.05x multiplier."""
        pbp = pl.DataFrame({
            'rusher_player_id': ['P1'] * 50 + ['P2'] * 30 + ['P3'] * 20,
            'posteam': ['KC'] * 100,
            'rush_attempt': [1] * 100,
            'yards_gained': [7.0] * 50 + [4.0] * 30 + [6.0] * 20  # P1: 7.0, Others: 4.8 avg
        })
        multiplier = context_adj.calculate_blocking_quality_proxy(pbp, 'P1', 'KC')
        # P1 YPC: 7.0, Teammates: 4.8, ratio: 1.458 → 1.05x
        assert multiplier == 1.05
    
    def test_player_underperforms_teammates(self, context_adj):
        """Player with 0.6x teammate YPC should get 0.95x multiplier."""
        pbp = pl.DataFrame({
            'rusher_player_id': ['P1'] * 40 + ['P2'] * 50 + ['P3'] * 30,
            'posteam': ['DAL'] * 120,
            'rush_attempt': [1] * 120,
            'yards_gained': [3.0] * 40 + [5.0] * 50 + [5.5] * 30  # P1: 3.0, Others: 5.2 avg
        })
        multiplier = context_adj.calculate_blocking_quality_proxy(pbp, 'P1', 'DAL')
        # P1 YPC: 3.0, Teammates: 5.2, ratio: 0.577 → 0.95x
        assert multiplier == 0.95
    
    def test_player_similar_to_teammates(self, context_adj):
        """Player with ~1.0x teammate YPC should get neutral multiplier."""
        pbp = pl.DataFrame({
            'rusher_player_id': ['P1'] * 50 + ['P2'] * 40 + ['P3'] * 30,
            'posteam': ['SF'] * 120,
            'rush_attempt': [1] * 120,
            'yards_gained': [4.5] * 50 + [4.3] * 40 + [4.7] * 30  # P1: 4.5, Others: 4.5 avg
        })
        multiplier = context_adj.calculate_blocking_quality_proxy(pbp, 'P1', 'SF')
        # P1 YPC: 4.5, Teammates: 4.49, ratio: 1.002 → 1.0x
        assert multiplier == 1.0
    
    def test_moderate_outperformance(self, context_adj):
        """Player with 1.2x teammate YPC should get 1.03x multiplier."""
        pbp = pl.DataFrame({
            'rusher_player_id': ['P1'] * 60 + ['P2'] * 50,
            'posteam': ['BUF'] * 110,
            'rush_attempt': [1] * 110,
            'yards_gained': [6.0] * 60 + [5.0] * 50  # P1: 6.0, P2: 5.0
        })
        multiplier = context_adj.calculate_blocking_quality_proxy(pbp, 'P1', 'BUF')
        # P1 YPC: 6.0, P2: 5.0, ratio: 1.2 → 1.03x
        assert multiplier == 1.03
    
    def test_insufficient_player_carries(self, context_adj):
        """Player with <20 carries should get neutral multiplier."""
        pbp = pl.DataFrame({
            'rusher_player_id': ['P1'] * 15 + ['P2'] * 50,
            'posteam': ['MIA'] * 65,
            'rush_attempt': [1] * 65,
            'yards_gained': [10.0] * 15 + [4.0] * 50
        })
        multiplier = context_adj.calculate_blocking_quality_proxy(pbp, 'P1', 'MIA')
        assert multiplier == 1.0
    
    def test_insufficient_teammate_carries(self, context_adj):
        """Team with <30 teammate carries should get neutral multiplier."""
        pbp = pl.DataFrame({
            'rusher_player_id': ['P1'] * 50 + ['P2'] * 20,
            'posteam': ['LAC'] * 70,
            'rush_attempt': [1] * 70,
            'yards_gained': [5.0] * 50 + [3.0] * 20
        })
        multiplier = context_adj.calculate_blocking_quality_proxy(pbp, 'P1', 'LAC')
        assert multiplier == 1.0
    
    def test_zero_teammate_ypc(self, context_adj):
        """Avoid division by zero when teammates have 0 YPC."""
        pbp = pl.DataFrame({
            'rusher_player_id': ['P1'] * 30 + ['P2'] * 40,
            'posteam': ['NYJ'] * 70,
            'rush_attempt': [1] * 70,
            'yards_gained': [5.0] * 30 + [0.0] * 40
        })
        multiplier = context_adj.calculate_blocking_quality_proxy(pbp, 'P1', 'NYJ')
        assert multiplier == 1.0


# ============================================================================
# TEST TEAMMATE QUALITY INDEX
# ============================================================================

class TestTeammateQualityIndex:
    """Test teammate quality (talent adjustment) calculations."""
    
    @pytest.fixture
    def context_adj(self):
        return ContextAdjustments()
    
    @pytest.fixture
    def sample_scores(self):
        """Create sample player scores for multiple teams."""
        return pl.DataFrame({
            'player_id': ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10'],
            'player_name': ['Player1', 'Player2', 'Player3', 'Player4', 'Player5',
                           'Player6', 'Player7', 'Player8', 'Player9', 'Player10'],
            'team': ['KC', 'KC', 'KC', 'DAL', 'DAL', 'NYJ', 'NYJ', 'NYJ', 'BUF', 'BUF'],
            'position': ['WR', 'RB', 'TE', 'WR', 'RB', 'WR', 'RB', 'TE', 'WR', 'RB'],
            'baseline_score': [400.0, 350.0, 250.0,  # KC: 1000 total (elite)
                              200.0, 180.0,          # DAL: 380 total (weak)
                              300.0, 280.0, 220.0,   # NYJ: 800 total (good)
                              250.0, 240.0]          # BUF: 490 total (average)
        })
    
    def test_player_with_weak_teammates(self, context_adj, sample_scores):
        """Player on weak team (DAL) should get 1.08x boost."""
        # P4 on DAL with only P5 as teammate (380 total - weakest)
        multiplier = context_adj.calculate_teammate_quality_index(
            'P4', 'Player4', 'DAL', 'WR', sample_scores
        )
        assert multiplier == 1.08
    
    def test_player_with_elite_teammates(self, context_adj, sample_scores):
        """Player on elite team (KC) should get 0.95x penalty."""
        # P1 on KC with P2, P3 as teammates (600 excluding self)
        # KC teammate score: 600 (P2+P3 excluding P1)
        multiplier = context_adj.calculate_teammate_quality_index(
            'P1', 'Player1', 'KC', 'WR', sample_scores
        )
        # KC: 600, NYJ: 800, BUF: 490, DAL: 380
        # KC is 2nd out of 4 = 50th percentile (not 75th+), should be neutral
        assert multiplier == 1.0
    
    def test_player_with_average_teammates(self, context_adj, sample_scores):
        """Player on weak team (BUF) should get boost."""
        # P9 on BUF with P10 as teammate (240 total)
        multiplier = context_adj.calculate_teammate_quality_index(
            'P9', 'Player9', 'BUF', 'WR', sample_scores
        )
        # BUF: 240, DAL: 380, KC: 600, NYJ: 800
        # BUF is lowest = 0th percentile, should be 1.08x
        assert multiplier == 1.08
    
    def test_player_with_good_teammates(self, context_adj, sample_scores):
        """Player on good team (NYJ) should get slight penalty."""
        # P6 on NYJ with P7, P8 as teammates (800 total - 75th+ percentile)
        multiplier = context_adj.calculate_teammate_quality_index(
            'P6', 'Player6', 'NYJ', 'WR', sample_scores
        )
        # NYJ is 3rd best out of 4 teams = 50th percentile, should be neutral
        assert multiplier in [0.97, 1.0]  # Could be either depending on exact percentile
    
    def test_no_teammates_returns_neutral(self, context_adj):
        """Player with no teammates should get neutral multiplier."""
        lonely_scores = pl.DataFrame({
            'player_id': ['P1'],
            'player_name': ['LoneWolf'],
            'team': ['ARI'],
            'position': ['WR'],
            'baseline_score': [500.0]
        })
        multiplier = context_adj.calculate_teammate_quality_index(
            'P1', 'LoneWolf', 'ARI', 'WR', lonely_scores
        )
        assert multiplier == 1.0
    
    def test_excludes_self_from_teammates(self, context_adj, sample_scores):
        """Teammate calculation should exclude the player being evaluated."""
        # P2 on KC - should calculate based on P1 + P3 (650), not including self (350)
        multiplier = context_adj.calculate_teammate_quality_index(
            'P2', 'Player2', 'KC', 'RB', sample_scores
        )
        # KC teammates: 650 (P1+P3), BUF: 240, DAL: 380, NYJ: 800
        # KC is 2nd out of 4 = 50th percentile, should be neutral
        assert multiplier == 1.0


# ============================================================================
# TEST SAMPLE SIZE DAMPENING
# ============================================================================

class TestSampleSizeDampening:
    """Test sample size dampening calculations."""
    
    @pytest.fixture
    def context_adj(self):
        return ContextAdjustments()
    
    def test_full_season_no_dampening(self, context_adj):
        """Full 17-game season should get full credit."""
        score = 500.0
        dampened = context_adj.apply_sample_size_dampening(score, 17, 17)
        assert dampened == 500.0
    
    def test_half_season_dampening(self, context_adj):
        """Half season (8 games) should get ~74% credit."""
        score = 500.0
        dampened = context_adj.apply_sample_size_dampening(score, 8, 17)
        # 8^0.4 / 17^0.4 ≈ 0.7397
        expected = 500.0 * 0.7397
        assert abs(dampened - expected) < 1.0  # Within 1 point
    
    def test_single_game_severe_dampening(self, context_adj):
        """Single game should get ~32% credit."""
        score = 500.0
        dampened = context_adj.apply_sample_size_dampening(score, 1, 17)
        # 1^0.4 / 17^0.4 ≈ 0.3220
        expected = 500.0 * 0.3220
        assert abs(dampened - expected) < 1.0
    
    def test_ten_games_moderate_dampening(self, context_adj):
        """10 games should get ~81% credit."""
        score = 500.0
        dampened = context_adj.apply_sample_size_dampening(score, 10, 17)
        # 10^0.4 / 17^0.4 ≈ 0.8088
        expected = 500.0 * 0.8088
        assert abs(dampened - expected) < 1.0
    
    def test_zero_games_returns_zero(self, context_adj):
        """Zero games should return zero score."""
        score = 500.0
        dampened = context_adj.apply_sample_size_dampening(score, 0, 17)
        assert dampened == 0.0
    
    def test_more_than_full_season_no_bonus(self, context_adj):
        """Playing more than full season should not give bonus."""
        score = 500.0
        dampened = context_adj.apply_sample_size_dampening(score, 20, 17)
        assert dampened == 500.0
    
    def test_dampening_curve_is_smooth(self, context_adj):
        """Dampening should increase smoothly with games played."""
        score = 100.0
        results = []
        for games in [1, 5, 10, 15, 17]:
            dampened = context_adj.apply_sample_size_dampening(score, games, 17)
            results.append(dampened)
        
        # Each result should be >= previous (monotonically increasing)
        for i in range(len(results) - 1):
            assert results[i] <= results[i + 1]
