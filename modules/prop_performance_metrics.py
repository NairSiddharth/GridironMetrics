"""
Player Props Performance Metrics

Proper betting-focused metrics for evaluating projection system performance.

Key Metrics:
- Line Hit Rate: Percentage of bets where projection beat the betting line
- ROI: Return on investment assuming $100 bets on each value bet
- Edge Capture Rate: Win rate when system identifies high-edge bets
- MAE/RMSE: Projection accuracy metrics
- Confidence Grade Performance: Win rates by grade (A/B/C)

Replaces misleading "accuracy %" (projection/actual) with metrics that
actually matter for betting purposes.
"""

from typing import List, Dict, Tuple
from dataclasses import dataclass
import polars as pl
from pathlib import Path


@dataclass
class BetResult:
    """Single bet result for performance tracking"""
    player_id: str
    player_name: str
    position: str
    prop_type: str

    # Line and projections
    betting_line: float
    projection: float
    actual_outcome: float

    # Bet details
    recommendation: str  # 'OVER' or 'UNDER'
    confidence: str  # 'A', 'B', or 'C'
    edge_pct: float

    # Pricing (optional)
    over_price: int = -110
    under_price: int = -110

    @property
    def won_bet(self) -> bool:
        """Did the bet win? Returns False if actual_outcome is None."""
        if self.actual_outcome is None:
            return False

        if self.recommendation == 'OVER':
            return self.actual_outcome > self.betting_line
        else:  # UNDER
            return self.actual_outcome < self.betting_line

    @property
    def projection_error(self) -> float:
        """Absolute error of projection. Returns 0 if actual_outcome is None."""
        if self.actual_outcome is None:
            return 0.0
        return abs(self.projection - self.actual_outcome)

    @property
    def projection_squared_error(self) -> float:
        """Squared error for RMSE calculation. Returns 0 if actual_outcome is None."""
        if self.actual_outcome is None:
            return 0.0
        return (self.projection - self.actual_outcome) ** 2

    def calculate_payout(self, wager: float = 100.0) -> float:
        """
        Calculate payout for this bet.

        Args:
            wager: Amount wagered (default $100)

        Returns:
            Net profit/loss (positive = win, negative = loss)
        """
        if self.actual_outcome is None:
            # No result available - treat as loss
            return -wager

        if self.won_bet:
            # Win: payout based on odds
            price = self.over_price if self.recommendation == 'OVER' else self.under_price

            # Default to -110 if price is None
            if price is None:
                price = -110

            if price > 0:
                # Underdog odds (+150 = win $150 on $100)
                profit = wager * (price / 100)
            else:
                # Favorite odds (-110 = win $100 on $110)
                profit = wager * (100 / abs(price))

            return profit
        else:
            # Loss: lose the wager
            return -wager

    def calculate_profit(self, wager: float = 100.0) -> float:
        """Alias for calculate_payout for backwards compatibility."""
        return self.calculate_payout(wager)

    def hit_line(self) -> bool:
        """Alias for won_bet for backwards compatibility."""
        return self.won_bet


class PropPerformanceMetrics:
    """Calculate betting performance metrics for prop projections"""

    def __init__(self, wager_per_bet: float = 100.0):
        """
        Initialize metrics calculator.

        Args:
            wager_per_bet: Standard wager amount for ROI calculation
        """
        self.wager_per_bet = wager_per_bet

    def calculate_line_hit_rate(self, bets: List[BetResult]) -> Dict[str, float]:
        """
        Calculate line hit rate (primary betting metric).

        Args:
            bets: List of bet results

        Returns:
            Dict with:
            - overall: Overall win rate
            - over: Win rate on OVER bets
            - under: Win rate on UNDER bets
            - by_confidence: Win rates by grade (A/B/C)
        """
        if not bets:
            return {
                'overall': 0.0,
                'over': 0.0,
                'under': 0.0,
                'by_confidence': {'A': 0.0, 'B': 0.0, 'C': 0.0}
            }

        total_wins = sum(1 for b in bets if b.won_bet)
        total_bets = len(bets)

        # Overall hit rate
        overall_rate = (total_wins / total_bets) * 100 if total_bets > 0 else 0

        # By recommendation
        over_bets = [b for b in bets if b.recommendation == 'OVER']
        under_bets = [b for b in bets if b.recommendation == 'UNDER']

        over_wins = sum(1 for b in over_bets if b.won_bet)
        under_wins = sum(1 for b in under_bets if b.won_bet)

        over_rate = (over_wins / len(over_bets) * 100) if over_bets else 0
        under_rate = (under_wins / len(under_bets) * 100) if under_bets else 0

        # By confidence grade
        by_confidence = {}
        for grade in ['A', 'B', 'C']:
            grade_bets = [b for b in bets if b.confidence == grade]
            if grade_bets:
                grade_wins = sum(1 for b in grade_bets if b.won_bet)
                by_confidence[grade] = (grade_wins / len(grade_bets)) * 100
            else:
                by_confidence[grade] = 0.0

        # By confidence counts
        by_confidence_counts = {}
        for grade in ['A', 'B', 'C']:
            grade_bets = [b for b in bets if b.confidence == grade]
            grade_wins = sum(1 for b in grade_bets if b.won_bet)
            by_confidence_counts[grade] = {
                'wins': grade_wins,
                'total': len(grade_bets)
            }

        return {
            'overall_rate': overall_rate,
            'overall_wins': total_wins,
            'overall_total': total_bets,
            'over_rate': over_rate,
            'over_wins': over_wins,
            'over_total': len(over_bets),
            'under_rate': under_rate,
            'under_wins': under_wins,
            'under_total': len(under_bets),
            'by_confidence': by_confidence,
            'by_confidence_counts': by_confidence_counts
        }

    def calculate_roi(self, bets: List[BetResult]) -> Dict[str, float]:
        """
        Calculate return on investment.

        Args:
            bets: List of bet results

        Returns:
            Dict with:
            - overall_roi: Overall ROI percentage
            - total_wagered: Total amount wagered
            - total_profit: Net profit/loss
            - avg_bet_return: Average return per bet
            - over_roi: ROI on OVER bets
            - under_roi: ROI on UNDER bets
        """
        if not bets:
            return {
                'overall_roi': 0.0,
                'total_wagered': 0.0,
                'total_profit': 0.0,
                'avg_bet_return': 0.0,
                'over_roi': 0.0,
                'under_roi': 0.0
            }

        total_wagered = len(bets) * self.wager_per_bet
        total_profit = sum(b.calculate_payout(self.wager_per_bet) for b in bets)

        overall_roi = (total_profit / total_wagered) * 100
        avg_bet_return = total_profit / len(bets)

        # By recommendation
        over_bets = [b for b in bets if b.recommendation == 'OVER']
        under_bets = [b for b in bets if b.recommendation == 'UNDER']

        over_profit = sum(b.calculate_payout(self.wager_per_bet) for b in over_bets)
        over_wagered = len(over_bets) * self.wager_per_bet
        over_roi = (over_profit / over_wagered * 100) if over_wagered > 0 else 0

        under_profit = sum(b.calculate_payout(self.wager_per_bet) for b in under_bets)
        under_wagered = len(under_bets) * self.wager_per_bet
        under_roi = (under_profit / under_wagered * 100) if under_wagered > 0 else 0

        return {
            'overall_roi': overall_roi,
            'total_wagered': total_wagered,
            'total_profit': total_profit,
            'avg_bet_return': avg_bet_return,
            'over_roi': over_roi,
            'under_roi': under_roi
        }

    def calculate_edge_capture_rate(self, bets: List[BetResult]) -> Dict[str, Dict[str, float]]:
        """
        Calculate win rate by edge bucket.

        Args:
            bets: List of bet results

        Returns:
            Dict with win rates for different edge thresholds:
            - edge_8_12: 8-12% edge bets
            - edge_12_15: 12-15% edge bets
            - edge_15_plus: 15%+ edge bets
        """
        edge_buckets = {
            'edge_8_12': [b for b in bets if 0.08 <= abs(b.edge_pct) < 0.12],
            'edge_12_15': [b for b in bets if 0.12 <= abs(b.edge_pct) < 0.15],
            'edge_15_plus': [b for b in bets if abs(b.edge_pct) >= 0.15]
        }

        results = {}
        for bucket_name, bucket_bets in edge_buckets.items():
            if bucket_bets:
                wins = sum(1 for b in bucket_bets if b.won_bet)
                win_rate = (wins / len(bucket_bets)) * 100
                # Create cleaner key names
                clean_name = bucket_name.replace('edge_', '').replace('_', '-')
                results[f'{clean_name}_rate'] = win_rate
                results[f'{clean_name}_count'] = len(bucket_bets)
                results[f'{clean_name}_wins'] = wins
            else:
                clean_name = bucket_name.replace('edge_', '').replace('_', '-')
                results[f'{clean_name}_rate'] = 0.0
                results[f'{clean_name}_count'] = 0
                results[f'{clean_name}_wins'] = 0

        return results

    def calculate_projection_accuracy(self, bets: List[BetResult]) -> Dict[str, float]:
        """
        Calculate MAE and RMSE for projection accuracy.

        Args:
            bets: List of bet results

        Returns:
            Dict with:
            - mae: Mean Absolute Error
            - rmse: Root Mean Squared Error
            - mean_proj: Average projection
            - mean_actual: Average actual outcome
            - mape: Mean Absolute Percentage Error
        """
        if not bets:
            return {
                'mae': 0.0,
                'rmse': 0.0,
                'mean_proj': 0.0,
                'mean_actual': 0.0,
                'mape': 0.0
            }

        # Filter out bets with None actual_outcome for accuracy metrics
        valid_bets = [b for b in bets if b.actual_outcome is not None]

        if not valid_bets:
            return {
                'mae': 0.0,
                'rmse': 0.0,
                'mean_proj': 0.0,
                'mean_actual': 0.0,
                'mape': 0.0
            }

        mae = sum(b.projection_error for b in valid_bets) / len(valid_bets)
        rmse = (sum(b.projection_squared_error for b in valid_bets) / len(valid_bets)) ** 0.5

        mean_proj = sum(b.projection for b in valid_bets) / len(valid_bets)
        mean_actual = sum(b.actual_outcome for b in valid_bets) / len(valid_bets)

        # MAPE (Mean Absolute Percentage Error)
        mape = sum(
            abs(b.projection - b.actual_outcome) / b.actual_outcome * 100
            for b in valid_bets if b.actual_outcome != 0
        ) / len([b for b in valid_bets if b.actual_outcome != 0]) if any(b.actual_outcome != 0 for b in valid_bets) else 0

        return {
            'mae': mae,
            'rmse': rmse,
            'mean_proj': mean_proj,
            'mean_actual': mean_actual,
            'mape': mape
        }

    def calculate_confidence_performance(self, bets: List[BetResult]) -> Dict[str, Dict]:
        """
        Detailed performance breakdown by confidence grade.

        Args:
            bets: List of bet results

        Returns:
            Dict with grade -> {win_rate, count, roi, avg_edge}
        """
        results = {}

        for grade in ['A', 'B', 'C']:
            grade_bets = [b for b in bets if b.confidence == grade]

            if not grade_bets:
                results[grade] = {
                    'win_rate': 0.0,
                    'count': 0,
                    'roi_pct': 0.0,
                    'avg_edge': 0.0
                }
                continue

            # Win rate
            wins = sum(1 for b in grade_bets if b.won_bet)
            win_rate = (wins / len(grade_bets)) * 100

            # ROI
            total_profit = sum(b.calculate_payout(self.wager_per_bet) for b in grade_bets)
            total_wagered = len(grade_bets) * self.wager_per_bet
            roi_pct = (total_profit / total_wagered) * 100

            # Average edge
            avg_edge = (sum(abs(b.edge_pct) for b in grade_bets) / len(grade_bets)) * 100

            results[grade] = {
                'win_rate': win_rate,
                'count': len(grade_bets),
                'wins': wins,
                'roi_pct': roi_pct,
                'avg_edge': avg_edge
            }

        return results

    def generate_full_report(self, bets: List[BetResult]) -> Dict:
        """
        Generate complete performance report.

        Args:
            bets: List of bet results

        Returns:
            Comprehensive metrics dict
        """
        return {
            'summary': {
                'total_bets': len(bets),
                'time_period': 'Historical backtest' if bets else 'No bets'
            },
            'line_hit_rate': self.calculate_line_hit_rate(bets),
            'roi': self.calculate_roi(bets),
            'edge_capture': self.calculate_edge_capture_rate(bets),
            'projection_accuracy': self.calculate_projection_accuracy(bets),
            'confidence_performance': self.calculate_confidence_performance(bets)
        }

    def print_report(self, bets: List[BetResult]) -> None:
        """
        Print formatted performance report.

        Args:
            bets: List of bet results
        """
        report = self.generate_full_report(bets)

        print("\n" + "="*70)
        print("PLAYER PROPS PERFORMANCE REPORT")
        print("="*70)

        print(f"\nTotal Bets: {report['summary']['total_bets']}")

        # Line Hit Rate
        lhr = report['line_hit_rate']
        print(f"\n--- LINE HIT RATE (Primary Metric) ---")
        print(f"Overall: {lhr['overall']:.1f}% ({lhr['total_wins']}/{lhr['total_bets']})")
        print(f"  OVER bets: {lhr['over']:.1f}%")
        print(f"  UNDER bets: {lhr['under']:.1f}%")
        print(f"\nBy Confidence Grade:")
        for grade in ['A', 'B', 'C']:
            print(f"  Grade {grade}: {lhr['by_confidence'][grade]:.1f}%")

        # ROI
        roi = report['roi']
        print(f"\n--- RETURN ON INVESTMENT ---")
        print(f"ROI: {roi['roi_pct']:+.1f}%")
        print(f"Total Wagered: ${roi['total_wagered']:.2f}")
        print(f"Total Profit: ${roi['total_profit']:+.2f}")
        print(f"Avg Return/Bet: ${roi['avg_bet_return']:+.2f}")

        # Edge Capture
        ec = report['edge_capture']
        print(f"\n--- EDGE CAPTURE RATE ---")
        for bucket, stats in ec.items():
            bucket_display = bucket.replace('_', ' ').replace('edge ', '').upper()
            print(f"{bucket_display}: {stats['win_rate']:.1f}% ({stats['wins']}/{stats['count']} bets)")

        # Projection Accuracy
        pa = report['projection_accuracy']
        print(f"\n--- PROJECTION ACCURACY ---")
        print(f"MAE (Mean Absolute Error): {pa['mae']:.2f}")
        print(f"RMSE (Root Mean Squared Error): {pa['rmse']:.2f}")
        print(f"Mean Projection: {pa['mean_proj']:.2f}")
        print(f"Mean Actual: {pa['mean_actual']:.2f}")

        # Confidence Performance
        cp = report['confidence_performance']
        print(f"\n--- CONFIDENCE GRADE PERFORMANCE ---")
        for grade in ['A', 'B', 'C']:
            stats = cp[grade]
            print(f"Grade {grade}: {stats['win_rate']:.1f}% win rate, "
                  f"{stats['roi_pct']:+.1f}% ROI, "
                  f"{stats['avg_edge']:.1f}% avg edge "
                  f"({stats['count']} bets)")

        print("\n" + "="*70)


if __name__ == "__main__":
    # Test with sample data
    print("=== Testing prop_performance_metrics.py ===")
    print("\nModule loaded successfully")
    print("\nKey Metrics:")
    print("1. Line Hit Rate - Did projection beat the betting line?")
    print("2. ROI - Return on investment with standard wagers")
    print("3. Edge Capture Rate - Win rate by edge bucket (8-12%, 12-15%, 15%+)")
    print("4. MAE/RMSE - Projection accuracy measurements")
    print("5. Confidence Performance - Performance by grade (A/B/C)")

    print("\n\nTo use:")
    print("1. Create BetResult objects for each bet")
    print("2. Initialize PropPerformanceMetrics()")
    print("3. Call generate_full_report(bets) or print_report(bets)")

    # Sample bet
    sample_bet = BetResult(
        player_id="00-0033873",
        player_name="Patrick Mahomes",
        position="QB",
        prop_type="passing_yards",
        betting_line=275.5,
        projection=312.4,
        actual_outcome=298.0,
        recommendation="OVER",
        confidence="A",
        edge_pct=0.134,
        over_price=-110,
        under_price=-110
    )

    print(f"\n\nSample BetResult:")
    print(f"  Player: {sample_bet.player_name}")
    print(f"  Prop: {sample_bet.prop_type}")
    print(f"  Line: {sample_bet.betting_line}")
    print(f"  Projection: {sample_bet.projection}")
    print(f"  Actual: {sample_bet.actual_outcome}")
    print(f"  Recommendation: {sample_bet.recommendation}")
    print(f"  Won: {sample_bet.won_bet}")
    print(f"  Payout: ${sample_bet.calculate_payout():+.2f}")
