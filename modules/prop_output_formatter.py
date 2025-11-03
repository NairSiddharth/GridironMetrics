"""
Player Props Output Formatter

Generates markdown output files for prop projections and value bets.

Output Structure:
output/{year}/playerprops/week{N}/
├── value_bets.md - Top value bets ranked by edge
├── qb_props.md - All QB projections vs lines
├── rb_props.md - All RB projections vs lines
├── wr_props.md - All WR projections vs lines
└── te_props.md - All TE projections vs lines

Format:
- Summary statistics at top
- Tables with player, prop, line, projection, edge, confidence
- Historical performance context (L3, L5, season avg)
"""

from pathlib import Path
from typing import List, Dict
from datetime import datetime
from modules.logger import get_logger

logger = get_logger(__name__)


class PropOutputFormatter:
    """Formats prop projections and value bets to markdown."""

    def __init__(self, output_dir: Path):
        """
        Initialize formatter.

        Args:
            output_dir: Base output directory (e.g., output/2024/playerprops)
        """
        self.output_dir = Path(output_dir)

    def format_value_bets_table(
        self,
        value_bets: List[Dict],
        summary: Dict
    ) -> str:
        """
        Format value bets into markdown table.

        Args:
            value_bets: List of value bet dicts from PropEvaluator
            summary: Summary statistics dict

        Returns:
            Markdown string
        """
        lines = []

        # Header
        lines.append("# GridironMetrics Player Props - Value Bets")
        lines.append("")
        lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        lines.append("")

        # Summary
        lines.append("## Summary")
        lines.append("")
        lines.append(f"- **Total Props Evaluated:** {summary['total_props_evaluated']}")
        lines.append(f"- **Value Bets Found:** {summary['total_value_found']} ({summary['value_pct']:.1f}%)")
        lines.append(f"- **Average Edge:** {summary['avg_edge_pct']:.1f}%")
        lines.append("")
        lines.append("**Confidence Breakdown:**")
        lines.append(f"- Grade A: {summary['confidence_breakdown']['A']} bets")
        lines.append(f"- Grade B: {summary['confidence_breakdown']['B']} bets")
        lines.append(f"- Grade C: {summary['confidence_breakdown']['C']} bets")
        lines.append("")
        lines.append("**Recommendations:**")
        lines.append(f"- OVER: {summary['recommendation_breakdown']['OVER']} bets")
        lines.append(f"- UNDER: {summary['recommendation_breakdown']['UNDER']} bets")
        lines.append("")

        # Legend
        lines.append("## Legend")
        lines.append("")
        lines.append("**Confidence Grades:**")
        lines.append("- **A**: Low variance (<0.20) + strong sample (8+ games) + strong edge (>12%)")
        lines.append("- **B**: Medium variance (<0.30) + decent sample (5+ games) + moderate edge (>8%)")
        lines.append("- **C**: Meets minimum edge threshold (>8%) but higher variance or smaller sample")
        lines.append("")
        lines.append("**Edge:** (Projection - Line) / Line × 100%")
        lines.append("")

        if not value_bets:
            lines.append("## No Value Bets Found")
            lines.append("")
            lines.append("No props met the minimum edge threshold (8%).")
            return "\n".join(lines)

        # Value bets table
        lines.append("## Value Bets (Ranked by Edge)")
        lines.append("")
        lines.append("| Rank | Player | Pos | Prop | Opponent | Line | Projection | Edge | Rec | Conf | Games | Var |")
        lines.append("|------|--------|-----|------|----------|------|------------|------|-----|------|-------|-----|")

        for i, bet in enumerate(value_bets, 1):
            edge_str = f"{bet['edge_pct']*100:+.1f}%"
            lines.append(
                f"| {i} | {bet['player_name']} | {bet['position']} | "
                f"{bet['prop_display']} | {bet['opponent']} | {bet['betting_line']:.1f} | "
                f"{bet['projection']:.1f} | {edge_str} | {bet['recommendation']} | "
                f"{bet['confidence']} | {bet['games_played']} | {bet['variance']:.3f} |"
            )

        lines.append("")

        # Grade A bets section
        grade_a = [b for b in value_bets if b['confidence'] == 'A']
        if grade_a:
            lines.append("## Grade A Bets (Highest Confidence)")
            lines.append("")
            lines.append("| Player | Prop | Line | Projection | Edge | Rec | Adjustments |")
            lines.append("|--------|------|------|------------|------|-----|-------------|")

            for bet in grade_a:
                edge_str = f"{bet['edge_pct']*100:+.1f}%"
                adj_str = ", ".join(f"{k}={v:.2f}x" for k, v in bet['adjustments'].items())
                lines.append(
                    f"| {bet['player_name']} | {bet['prop_display']} | {bet['betting_line']:.1f} | "
                    f"{bet['projection']:.1f} | {edge_str} | {bet['recommendation']} | {adj_str} |"
                )

            lines.append("")

        return "\n".join(lines)

    def format_position_props_table(
        self,
        position: str,
        props_data: List[Dict]
    ) -> str:
        """
        Format props for a position into markdown table.

        Args:
            position: Position code (QB, RB, WR, TE)
            props_data: List of prop data dicts (from evaluator or projections)

        Returns:
            Markdown string
        """
        lines = []

        # Header
        lines.append(f"# GridironMetrics {position} Props")
        lines.append("")
        lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        lines.append("")

        if not props_data:
            lines.append(f"## No {position} Props Available")
            return "\n".join(lines)

        # Group by player
        players = {}
        for prop in props_data:
            player_key = (prop['player_id'], prop['player_name'])
            if player_key not in players:
                players[player_key] = []
            players[player_key].append(prop)

        # Sort players by total edge (absolute)
        player_list = sorted(
            players.items(),
            key=lambda x: sum(abs(p.get('edge_pct', 0)) for p in x[1]),
            reverse=True
        )

        lines.append(f"## {position} Props Summary")
        lines.append("")
        lines.append(f"**Total Players:** {len(players)}")
        lines.append(f"**Total Props:** {len(props_data)}")
        lines.append("")

        # Player tables
        for (player_id, player_name), player_props in player_list:
            lines.append(f"### {player_name}")
            lines.append("")

            # Get opponent and position from first prop
            opponent = player_props[0].get('opponent', 'N/A')
            pos = player_props[0].get('position', position)
            lines.append(f"**Position:** {pos} | **Opponent:** {opponent}")
            lines.append("")

            lines.append("| Prop | Line | Projection | Edge | Rec | Conf | L3 Avg | L5 Avg | Season Avg | Var |")
            lines.append("|------|------|------------|------|-----|------|--------|--------|------------|-----|")

            # Sort props by edge
            player_props_sorted = sorted(
                player_props,
                key=lambda x: abs(x.get('edge_pct', 0)),
                reverse=True
            )

            for prop in player_props_sorted:
                line = prop.get('betting_line', 0)
                proj = prop.get('projection', 0)
                edge_pct = prop.get('edge_pct', 0)
                edge_str = f"{edge_pct*100:+.1f}%" if edge_pct != 0 else "—"
                rec = prop.get('recommendation', '—')
                conf = prop.get('confidence', '—')

                # Get historical averages if available
                l3 = prop.get('stat_summary', {}).get('last_3_avg', 0) if 'stat_summary' in prop else 0
                l5 = prop.get('stat_summary', {}).get('last_5_avg', 0) if 'stat_summary' in prop else 0
                season = prop.get('stat_summary', {}).get('season_avg', 0) if 'stat_summary' in prop else 0
                var = prop.get('variance', 0)

                lines.append(
                    f"| {prop.get('prop_display', prop['prop_type'])} | {line:.1f} | "
                    f"{proj:.1f} | {edge_str} | {rec} | {conf} | {l3:.1f} | {l5:.1f} | "
                    f"{season:.1f} | {var:.3f} |"
                )

            lines.append("")

        return "\n".join(lines)

    def save_week_output(
        self,
        season: int,
        week: int,
        value_bets: List[Dict],
        summary: Dict,
        all_props_by_position: Dict[str, List[Dict]]
    ) -> None:
        """
        Save all output files for a week.

        Args:
            season: Season year
            week: Week number
            value_bets: List of value bets
            summary: Summary statistics
            all_props_by_position: Dict of {position: [props]}
        """
        # Create output directory
        week_dir = self.output_dir / str(season) / "playerprops" / f"week{week}"
        week_dir.mkdir(parents=True, exist_ok=True)

        # Save value bets
        value_bets_file = week_dir / "value_bets.md"
        value_bets_content = self.format_value_bets_table(value_bets, summary)
        value_bets_file.write_text(value_bets_content, encoding='utf-8')
        logger.info(f"Saved value bets to {value_bets_file}")

        # Save position-specific files
        for position, props in all_props_by_position.items():
            if props:
                position_file = week_dir / f"{position.lower()}_props.md"
                position_content = self.format_position_props_table(position, props)
                position_file.write_text(position_content, encoding='utf-8')
                logger.info(f"Saved {position} props to {position_file}")

        logger.info(
            f"Week {week} output complete: {week_dir} "
            f"({len(value_bets)} value bets, {sum(len(p) for p in all_props_by_position.values())} total props)"
        )


if __name__ == "__main__":
    # Test formatter
    print("=== Testing prop_output_formatter.py ===")
    print("Module loaded successfully")
    print("\nOutput Structure:")
    print("output/{year}/playerprops/week{N}/")
    print("├── value_bets.md")
    print("├── qb_props.md")
    print("├── rb_props.md")
    print("├── wr_props.md")
    print("└── te_props.md")
    print("\nTo test:")
    print("1. Run prop evaluation to get value_bets and summary")
    print("2. Initialize formatter: formatter = PropOutputFormatter(Path('output'))")
    print("3. Save output: formatter.save_week_output(2024, 11, value_bets, summary, props_by_position)")
