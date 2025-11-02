"""
Table Formatters Module

Functions for generating markdown tables and formatted output for:
- Weekly offensive share tables
- Season summaries
- Position-specific rankings (QB, RB, WR, TE)
- Top contributors
- FTN context tables
"""

from pathlib import Path
import polars as pl
from prettytable import PrettyTable, TableStyle
from modules.logger import get_logger
from modules.constants import (
    SKILL_POSITIONS, COMBINED_METRICS,
    SKILL_POSITION_METRICS, QB_METRICS
)
from modules.data_loaders import (
    load_team_weekly_stats, load_position_weekly_stats,
    is_season_complete, classify_player_profile
)
from modules.contribution_calculators import (
    calculate_offensive_shares,
    calculate_qb_contribution_from_pbp,
    calculate_average_difficulty,
    calculate_difficulty_context,
    calculate_defensive_stats,
    adjust_for_opponent
)
from modules.adjustment_pipeline import (
    apply_phase4_adjustments,
    apply_phase4_5_weather_adjustments,
    apply_phase5_adjustments
)
from modules.rankings_generator import generate_position_rankings

# Initialize logger
logger = get_logger(__name__)

def generate_weekly_tables(year: int) -> str:
    """Generate weekly offensive share tables for all teams."""
    logger.info(f"Generating weekly offensive share tables for {year}")

    # Load team stats
    team_stats = load_team_weekly_stats(year)
    if team_stats is None:
        return "No team data available."
        
    # Load player stats for each position
    position_stats = {}
    for pos in SKILL_POSITIONS:
        df = load_position_weekly_stats(year, pos)
        if df is not None:
            position_stats[pos] = df
    
    if not position_stats:
        return "No player data available."
    
    # Combine all position stats
    player_stats = pl.concat(list(position_stats.values()))
    
    # Generate tables for each week
    markdown = f"# Offensive Share Analysis - {year}\n\n"
    
    # Get unique teams
    teams = sorted(team_stats['team'].unique().to_list())
    weeks = sorted(team_stats['week'].unique().to_list())
    
    for team in teams:
        markdown += f"## {team}\n\n"
        
        for week in weeks:
            markdown += f"### Week {week}\n\n"
            
            # Create a table for each skill position metric
            for metric, metric_name in SKILL_POSITION_METRICS.items():
                try:
                    shares = calculate_offensive_shares(
                        team_stats.filter(pl.col('team') == team).filter(pl.col('week') == week),
                        player_stats.filter(pl.col('team') == team).filter(pl.col('week') == week),
                        metric,
                        opponent_stats=team_stats,
                        year=year,
                        week=week,
                        team=team
                    )
                    
                    if shares.height == 0:
                        continue
                    
                    table = PrettyTable()
                    table.title = f"{metric_name} Share"
                    table.field_names = ["Player", "Position", "Raw Share (%)", "Adjusted Share (%)"]
                    table.align = "l"
                    table.float_format = '.1'
                    table.set_style(TableStyle.MARKDOWN)
                    
                    # Get top 5 contributors by adjusted share
                    adj_metric = f"{metric}_adjusted" if f"{metric}_adjusted" in shares.columns else f"{metric}"
                    top_players = shares.sort(f'{adj_metric}_share', descending=True).head(5)
                    
                    for row in top_players.iter_rows(named=True):
                        player_name = row['player_name']
                        position = player_stats.filter(pl.col('player_name') == player_name)['position'].head(1)[0]
                        raw_share = row[f'{metric}_share']
                        adj_share = row[f'{adj_metric}_share']
                        table.add_row([player_name, position, raw_share, adj_share])
                    
                    return table.get_string()
                    
                    markdown += "\n"
                except Exception as e:
                    logger.error(f"Error generating table for {team} week {week} {metric}: {str(e)}")
                    continue
                    
        markdown += "---\n\n"
    
    return markdown

def generate_season_summary(year: int) -> str:
    """Generate season summary table showing top offensive share holder per team."""
    logger.info(f"Generating season summary for {year}")
    
    # Load team stats
    logger.info("Loading team stats...")
    team_stats = load_team_weekly_stats(year)
    if team_stats is None:
        return "No team data available."
    logger.info(f"Loaded {team_stats.height} team-week records")
        
    # Load player stats for each position
    logger.info("Loading player stats...")
    position_stats = {}
    for pos in SKILL_POSITIONS:
        df = load_position_weekly_stats(year, pos)
        if df is not None:
            position_stats[pos] = df
    
    if not position_stats:
        return "No player data available."
    logger.info(f"Loaded player stats for {len(position_stats)} positions")
    
    # Combine all position stats
    player_stats = pl.concat(list(position_stats.values()))
    logger.info(f"Combined {player_stats.height} player-week records")
    
    # Get unique teams
    teams = sorted(team_stats['team'].unique().to_list())
    logger.info(f"Processing {len(teams)} teams")
    
    markdown = f"# Season Summary - {year}\n\n"
    
    # Create a table for each skill position metric
    logger.info(f"Creating tables for {len(SKILL_POSITION_METRICS)} metrics")
    for metric, metric_name in SKILL_POSITION_METRICS.items():
        logger.info(f"Processing metric: {metric_name}")
            
        table = PrettyTable()
        table.title = f"{metric_name} Share Leaders"
        table.field_names = ["Team", "Player", "Position", "Season Share (%)"]
        table.align = "l"
        table.float_format = '.1'
        table.set_style(TableStyle.MARKDOWN)
        
        logger.info(f"Processing {len(teams)} teams for {metric_name}")
        for team_idx, team in enumerate(teams):
            if team_idx % 5 == 0:
                logger.info(f"  [{metric_name}] Processing team {team_idx+1}/{len(teams)}: {team}")
            try:
                # Get all weeks for this team
                team_season = team_stats.filter(pl.col('team') == team)
                player_season = player_stats.filter(pl.col('team') == team)
                
                # Calculate week-by-week shares with situational adjustments
                all_shares = []
                for week in team_season['week'].unique().to_list():
                    team_week = team_season.filter(pl.col('week') == week)
                    player_week = player_season.filter(pl.col('week') == week)
                    
                    week_shares = calculate_offensive_shares(team_week, player_week, metric, opponent_stats=team_stats, year=year, week=week, team=team)
                    if week_shares.height > 0:
                        all_shares.append(week_shares)
                
                if not all_shares:
                    continue
                
                # Combine all weeks
                combined_shares = pl.concat(all_shares)
                
                # Average the share percentages across all weeks
                player_avg_shares = (
                    combined_shares
                    .group_by('player_name')
                    .agg([
                        pl.col(f'{metric}_share').mean().alias('avg_share')
                    ])
                    .sort('avg_share', descending=True)
                )
                
                if player_avg_shares.height == 0:
                    continue
                
                # Get the top contributor
                top_player = player_avg_shares.head(1)
                player_name = top_player['player_name'][0]
                position = player_stats.filter(pl.col('player_name') == player_name)['position'].head(1)[0]
                share = top_player['avg_share'][0]
                
                table.add_row([team, player_name, position, share])
            except Exception as e:
                logger.error(f"Error generating summary for {team} {metric}: {str(e)}")
                continue
        
        markdown += table.get_string() + "\n\n"

    return markdown

# ============================================================================
# POSITION RANKINGS (Unified Implementation)
# ============================================================================

def generate_qb_rankings(year: int) -> str:
    """Generate QB rankings using unified rankings generator."""
    return generate_position_rankings(year, 'QB')

def generate_rb_rankings(year: int) -> str:
    """Generate RB rankings using unified rankings generator."""
    return generate_position_rankings(year, 'RB')

def generate_wr_rankings(year: int) -> str:
    """Generate WR rankings using unified rankings generator."""
    return generate_position_rankings(year, 'WR')

def generate_te_rankings(year: int) -> str:
    """Generate TE rankings using unified rankings generator."""
    return generate_position_rankings(year, 'TE')

def generate_ftn_context(year: int, player_stats: pl.DataFrame, season_contributions: pl.DataFrame) -> str:
    """Generate FTN charting context tables for QB/RB/WR/TE.
    
    Only generates content for years 2022+ (when FTN data is available).
    
    Args:
        year: Season year
        player_stats: Weekly player stats with FTN flags
        season_contributions: Season-level contribution scores
        
    Returns:
        Markdown string with FTN context tables
    """
    from modules.ftn_cache_builder import FTN_START_YEAR
    
    if year < FTN_START_YEAR:
        return ""
    
    # Load FTN cache
    ftn_path = Path("cache/ftn") / f"ftn_{year}.parquet"
    if not ftn_path.exists():
        logger.warning(f"FTN cache not found for {year}, skipping FTN context")
        return ""
    
    try:
        ftn_data = pl.read_parquet(ftn_path)
    except Exception as e:
        logger.error(f"Error loading FTN cache for {year}: {e}")
        return ""
    
    # Load PBP data to join FTN with player stats
    pbp_path = Path("cache/pbp") / f"pbp_{year}.parquet"
    if not pbp_path.exists():
        logger.warning(f"PBP cache not found for {year}, cannot generate FTN context")
        return ""
    
    try:
        pbp_data = pl.read_parquet(pbp_path)
    except Exception as e:
        logger.error(f"Error loading PBP cache for {year}: {e}")
        return ""
    
    # Join FTN with PBP - handle column name variations and type casting
    # PBP cache may have 'game_id'/'play_id' or 'nflverse_game_id'/'nflverse_play_id'
    try:
        # Step 1: Cast play_id to Int64 BEFORE renaming (handles Float64/Int32/Int64 sources)
        if 'play_id' in pbp_data.columns:
            pbp_data = pbp_data.with_columns([
                pl.col('play_id').cast(pl.Int64, strict=False)
            ])

        # Step 2: Ensure FTN's nflverse_play_id is also Int64 for consistent join
        if 'nflverse_play_id' in ftn_data.columns:
            ftn_data = ftn_data.with_columns([
                pl.col('nflverse_play_id').cast(pl.Int64, strict=False)
            ])

        # Step 3: Rename PBP columns to match FTN naming convention
        if 'game_id' in pbp_data.columns and 'nflverse_game_id' not in pbp_data.columns:
            pbp_data = pbp_data.rename({'game_id': 'nflverse_game_id'})
        if 'play_id' in pbp_data.columns and 'nflverse_play_id' not in pbp_data.columns:
            pbp_data = pbp_data.rename({'play_id': 'nflverse_play_id'})

        # Step 4: Join with consistent Int64 types on both sides
        pbp_with_ftn = pbp_data.join(
            ftn_data,
            on=['nflverse_game_id', 'nflverse_play_id'],
            how='left'
        )
    except Exception as e:
        logger.error(f"Error joining FTN with PBP for {year}: {e}")
        return ""
    
    md = "## FTN Context (2022+)\n\n"
    md += "*Human-charted play characteristics from Football Technology Network*\n\n"
    
    # QB Play Style Context
    qb_players = player_stats.filter(pl.col('position') == 'QB').select(['player_id', 'player_name', 'position']).unique()
    if len(qb_players) > 0:
        qb_pbp = pbp_with_ftn.filter(
            (pl.col('passer_player_id').is_in(qb_players['player_id'].to_list())) &
            (pl.col('play_type') == 'pass') &
            (pl.col('is_play_action').is_not_null())  # Ensure FTN data exists
        )
        
        if len(qb_pbp) > 0:
            qb_ftn = qb_pbp.group_by('passer_player_id').agg([
                pl.len().alias('pass_attempts'),
                pl.col('is_play_action').mean().alias('pa_rate'),
                pl.col('is_qb_out_of_pocket').mean().alias('oop_rate'),
                pl.col('is_screen_pass').mean().alias('screen_rate'),
                (pl.col('n_blitzers') >= 5).mean().alias('blitz_rate')
            ]).rename({'passer_player_id': 'player_id'})
            
            # Join with player names and season scores
            qb_ftn = qb_ftn.join(qb_players, on='player_id', how='left')
            qb_season = season_contributions.filter(pl.col('position') == 'QB').head(15)
            qb_with_ftn = qb_season.join(qb_ftn, on=['player_id', 'player_name'], how='left')
            
            # Filter to QBs with at least 100 pass attempts
            qb_with_ftn = qb_with_ftn.filter(pl.col('pass_attempts') >= 100)
            
            if len(qb_with_ftn) > 0:
                md += "### QB - Play Style Context\n\n"
                md += "*How often QBs face different play types*\n\n"
                md += "- **Play Action**: -10% adjustment (easier)\n"
                md += "- **Out of Pocket**: +3 pts/completion (harder)\n"
                md += "- **5+ Blitzers**: -2 pts/completion (harder)\n"
                md += "- **Screen Pass**: -15% adjustment (easier)\n\n"
                
                qb_table = PrettyTable()
                qb_table.field_names = ["Player", "PA%", "OOP%", "Screen%", "Blitz%", "Pass Att", "Net Adjustment"]
                qb_table.align = "l"
                qb_table.float_format = '.1'
                qb_table.set_style(TableStyle.MARKDOWN)
                
                for row in qb_with_ftn.sort('raw_score', descending=True).iter_rows(named=True):
                    player_name = row['player_name']
                    pa_pct = row['pa_rate'] * 100 if row['pa_rate'] is not None else 0
                    oop_pct = row['oop_rate'] * 100 if row['oop_rate'] is not None else 0
                    screen_pct = row['screen_rate'] * 100 if row['screen_rate'] is not None else 0
                    blitz_pct = row['blitz_rate'] * 100 if row['blitz_rate'] is not None else 0
                    pass_att = int(row['pass_attempts']) if row['pass_attempts'] is not None else 0
                    
                    # Calculate net adjustment impact (simplified)
                    # Positive = harder, Negative = easier
                    net_adj = (oop_pct * 0.03) + (blitz_pct * -0.02) + (pa_pct * -0.10) + (screen_pct * -0.15)
                    adj_display = f"+{net_adj:.1f}%" if net_adj > 0 else f"{net_adj:.1f}%"
                    
                    qb_table.add_row([player_name, f"{pa_pct:.1f}%", f"{oop_pct:.1f}%", 
                                     f"{screen_pct:.1f}%", f"{blitz_pct:.1f}%", pass_att, adj_display])
                
                md += qb_table.get_string() + "\n\n"
    
    # RB Scheme Context
    rb_players = player_stats.filter(pl.col('position') == 'RB').select(['player_id', 'player_name', 'position']).unique()
    if len(rb_players) > 0:
        rb_pbp = pbp_with_ftn.filter(
            (pl.col('rusher_player_id').is_in(rb_players['player_id'].to_list())) &
            (pl.col('play_type') == 'run') &
            (pl.col('is_rpo').is_not_null())  # Ensure FTN data exists
        )
        
        if len(rb_pbp) > 0:
            rb_ftn = rb_pbp.group_by('rusher_player_id').agg([
                pl.len().alias('rush_attempts'),
                pl.col('is_rpo').mean().alias('rpo_rate'),
                (pl.col('n_defense_box') >= 8).mean().alias('heavy_box_rate')
            ]).rename({'rusher_player_id': 'player_id'})
            
            # Join with player names and season scores
            rb_ftn = rb_ftn.join(rb_players, on='player_id', how='left')
            rb_season = season_contributions.filter(pl.col('position') == 'RB').head(15)
            rb_with_ftn = rb_season.join(rb_ftn, on=['player_id', 'player_name'], how='left')
            
            # Filter to RBs with at least 50 rush attempts
            rb_with_ftn = rb_with_ftn.filter(pl.col('rush_attempts') >= 50)
            
            if len(rb_with_ftn) > 0:
                md += "### RB - Scheme Context\n\n"
                md += "*How often RBs face different rushing situations*\n\n"
                md += "- **RPO**: -12% adjustment (easier due to defensive confusion)\n"
                md += "- **8+ Defenders in Box**: +15% adjustment (harder)\n\n"
                
                rb_table = PrettyTable()
                rb_table.field_names = ["Player", "RPO%", "Heavy Box%", "Rush Att", "Net Adjustment"]
                rb_table.align = "l"
                rb_table.float_format = '.1'
                rb_table.set_style(TableStyle.MARKDOWN)
                
                for row in rb_with_ftn.sort('raw_score', descending=True).iter_rows(named=True):
                    player_name = row['player_name']
                    rpo_pct = row['rpo_rate'] * 100 if row['rpo_rate'] is not None else 0
                    heavy_box_pct = row['heavy_box_rate'] * 100 if row['heavy_box_rate'] is not None else 0
                    rush_att = int(row['rush_attempts']) if row['rush_attempts'] is not None else 0
                    
                    # Calculate net adjustment impact
                    # Positive = harder, Negative = easier
                    net_adj = (rpo_pct * -0.12) + (heavy_box_pct * 0.15)
                    adj_display = f"+{net_adj:.1f}%" if net_adj > 0 else f"{net_adj:.1f}%"
                    
                    rb_table.add_row([player_name, f"{rpo_pct:.1f}%", f"{heavy_box_pct:.1f}%", 
                                     rush_att, adj_display])
                
                md += rb_table.get_string() + "\n\n"
    
    # WR/TE Reception Context
    wr_te_players = player_stats.filter(
        pl.col('position').is_in(['WR', 'TE'])
    ).select(['player_id', 'player_name', 'position']).unique()
    
    if len(wr_te_players) > 0:
        wr_te_pbp = pbp_with_ftn.filter(
            (pl.col('receiver_player_id').is_in(wr_te_players['player_id'].to_list())) &
            (pl.col('play_type') == 'pass') &
            (pl.col('is_contested_ball').is_not_null())  # Ensure FTN data exists
        )
        
        if len(wr_te_pbp) > 0:
            wr_te_ftn = wr_te_pbp.group_by('receiver_player_id').agg([
                pl.len().alias('targets'),
                pl.col('is_contested_ball').mean().alias('contested_rate'),
                pl.col('is_drop').sum().alias('drops'),
                pl.col('is_screen_pass').mean().alias('screen_rate')
            ]).rename({'receiver_player_id': 'player_id'})
            
            # Join with player names and season scores
            wr_te_ftn = wr_te_ftn.join(wr_te_players, on='player_id', how='left')
            
            for pos in ['WR', 'TE']:
                pos_season = season_contributions.filter(pl.col('position') == pos).head(15)
                pos_with_ftn = pos_season.join(wr_te_ftn, on=['player_id', 'player_name'], how='left')
                
                # Filter to players with at least 20 targets
                pos_with_ftn = pos_with_ftn.filter(pl.col('targets') >= 20)
                
                if len(pos_with_ftn) > 0:
                    md += f"### {pos} - Reception Context\n\n"
                    md += "*How often receivers face difficult catch situations*\n\n"
                    md += "- **Contested Catch**: +25% adjustment (harder)\n"
                    md += "- **Drop**: -8 pts penalty\n"
                    md += "- **Screen Pass**: -10% adjustment (easier)\n\n"
                    
                    pos_table = PrettyTable()
                    pos_table.field_names = ["Player", "Contested%", "Drops", "Screen%", "Targets", "Net Adjustment"]
                    pos_table.align = "l"
                    pos_table.float_format = '.1'
                    pos_table.set_style(TableStyle.MARKDOWN)
                    
                    for row in pos_with_ftn.sort('raw_score', descending=True).iter_rows(named=True):
                        player_name = row['player_name']
                        contested_pct = row['contested_rate'] * 100 if row['contested_rate'] is not None else 0
                        drops = int(row['drops']) if row['drops'] is not None else 0
                        screen_pct = row['screen_rate'] * 100 if row['screen_rate'] is not None else 0
                        targets = int(row['targets']) if row['targets'] is not None else 0
                        
                        # Calculate net adjustment impact
                        # Positive = harder, Negative = easier
                        net_adj = (contested_pct * 0.25) + (screen_pct * -0.10) + (drops * -0.08)
                        adj_display = f"+{net_adj:.1f}%" if net_adj > 0 else f"{net_adj:.1f}%"
                        
                        pos_table.add_row([player_name, f"{contested_pct:.1f}%", drops, 
                                          f"{screen_pct:.1f}%", targets, adj_display])
                    
                    md += pos_table.get_string() + "\n\n"
    
    md += "*Note: Net Adjustment shows cumulative impact of FTN flags on player scores. Positive = harder situations faced, Negative = easier situations faced.*\n\n"
    
    logger.info(f"Generated FTN context tables for {year}")
    return md


def generate_top_contributors(year: int) -> tuple[str, str]:
    """Generate a table of the top 10 offensive contributors for the season.
    
    Returns:
        tuple of (overview_markdown, deep_dive_markdown)
    """
    logger.info(f"Generating top contributors table for {year}")

    # Load team stats
    team_stats = load_team_weekly_stats(year)
    if team_stats is None:
        return "No team data available."

    # Load player stats for each position
    position_stats = {}
    for pos in SKILL_POSITIONS:
        df = load_position_weekly_stats(year, pos)
        if df is not None:
            position_stats[pos] = df
    
    if not position_stats:
        return "No player data available."
    
    # Combine all position stats
    player_stats = pl.concat(list(position_stats.values()))
    
    # Apply minimum activity thresholds (1978-now standards)
    # Calculate per-game rates for each player
    player_games = player_stats.group_by(['player_id', 'player_name', 'position']).agg([
        pl.col('week').count().alias('games'),
        pl.col('carries').sum().alias('total_carries'),
        pl.col('receptions').sum().alias('total_receptions')
    ])
    
    # Calculate per-game rates
    player_games = player_games.with_columns([
        (pl.col('total_carries') / pl.col('games')).alias('carries_per_game'),
        (pl.col('total_receptions') / pl.col('games')).alias('receptions_per_game')
    ])
    
    # Apply position-specific minimum thresholds
    # RB: 6.25 rushing attempts per game
    # WR/TE: 1.875 receptions per game
    # Note: QB thresholds (14 attempts/game) handled separately in QB rankings
    qualified_players = player_games.filter(
        ((pl.col('position') == 'RB') & (pl.col('carries_per_game') >= 6.25)) |
        ((pl.col('position').is_in(['WR', 'TE'])) & (pl.col('receptions_per_game') >= 1.875))
    )
    
    # Filter player_stats to only include qualified players
    qualified_ids = qualified_players.select(['player_id', 'player_name']).unique()
    player_stats = player_stats.join(qualified_ids, on=['player_id', 'player_name'], how='inner')
    
    logger.info(f"Filtered to {len(qualified_ids)} qualified players meeting minimum activity thresholds")
    
    # Calculate RAW scores first (without situational adjustments - year=None means multipliers = 1.0)
    raw_contributions = calculate_offensive_shares(team_stats, player_stats, 'overall_contribution', opponent_stats=team_stats, year=None)
    
    # Calculate ADJUSTED scores (with situational adjustments from PBP data)
    contributions = calculate_offensive_shares(team_stats, player_stats, 'overall_contribution', opponent_stats=team_stats, year=year)
    
    # Add position column back by joining with player_stats
    # Ensure player_positions is truly unique per player
    player_positions = player_stats.select(['player_id', 'player_name', 'position']).unique(subset=['player_id', 'player_name'])
    
    # Calculate season aggregates for positional rankings (use RAW scores for fair comparison)
    raw_season_contributions = raw_contributions.group_by(['player_id', 'player_name', 'team']).agg([
        pl.col('player_overall_contribution').mean().alias('raw_score')
    ])
    raw_season_contributions = raw_season_contributions.join(player_positions, on=['player_id', 'player_name'], how='left')
    
    # NOTE: Position rankings will be calculated AFTER Phase 5 adjustments
    # to match the rankings shown in position-specific tables
    # See line ~1015 where position_rankings is recalculated
    
    # First, calculate positional rankings (use raw scores for ranking)
    # Rank by (player_id, player_name, team) to handle players with same name on different teams
    position_rankings_old = {}
    for pos in SKILL_POSITIONS:
        pos_stats = raw_season_contributions.filter(
            pl.col('position') == pos
        ).sort('raw_score', descending=True)
        
        position_rankings_old[pos] = {
            (row['player_id'], row['player_name'], row['team']): rank + 1 
            for rank, row in enumerate(pos_stats.iter_rows(named=True))
        }
    
    
    # Add position to contributions for later use
    contributions = contributions.join(player_positions, on=['player_id', 'player_name'], how='left')
    
    # Apply Phase 4 adjustments (catch rate + blocking quality)
    # These are player-level adjustments calculated once per player after aggregation
    contributions = apply_phase4_adjustments(contributions, year)
    
    # Save per-game contributions before Phase 5 aggregation for peak/notable game tracking and consistency metrics
    # Clone the ENTIRE dataframe first, then select columns to ensure we have a deep copy
    per_game_contributions = contributions.clone().select([
        'player_id', 'player_name', 'team', 'position', 'week', 'player_overall_contribution'
    ])
    
    peak_games = (
        per_game_contributions.group_by(['player_id', 'player_name', 'team'])
        .agg([
            pl.col('player_overall_contribution').max().alias('peak_game_score')
        ])
    )
    
    # Apply Phase 5 adjustments (talent context + sample size dampening)
    # Two-pass system: use baseline scores to calculate teammate quality, then dampen by games played
    contributions = apply_phase5_adjustments(contributions, year)
    
    # Join peak game scores back
    contributions = contributions.join(peak_games, on=['player_id', 'player_name', 'team'], how='left')
    
    # Calculate average difficulty multipliers for all players
    avg_difficulty = calculate_average_difficulty(year, player_stats)
    
    # Initialize markdown strings for both files
    overview_md = f"# Top Contributors - {year}\n\n"
    overview_md += "*High-level overview of top offensive contributors*\n\n"
    overview_md += "## Overall Rankings\n\n"
    
    deep_dive_md = f"# Top Contributors - Deep Dive - {year}\n\n"
    deep_dive_md += "*Detailed analysis with consistency metrics and notable performances*\n\n"
    deep_dive_md += "## Overall Rankings\n\n"
    
    # Create PrettyTable for overview (simplified columns)
    overview_table = PrettyTable()
    overview_table.field_names = ["Rank", "Player", "Team", "Position", "Adjusted Score", 
                                    "Difficulty", "Games", "Avg/Game", "Typical", "Consistency", "Trend"]
    overview_table.align = "l"
    overview_table.float_format = '.2'
    overview_table.set_style(TableStyle.MARKDOWN)
    
    # Create PrettyTable for deep dive (all columns)
    deep_dive_table = PrettyTable()
    deep_dive_table.field_names = ["Rank", "Player", "Team", "Position (Pos. Rank)", "Raw Score", "Adjusted Score", 
                        "Difficulty", "Games", "Avg/Game", "Typical", "Consistency", "Floor", "Ceiling", "Peak", "Trend", "Notable Games"]
    deep_dive_table.align = "l"
    deep_dive_table.float_format = '.2'
    deep_dive_table.set_style(TableStyle.MARKDOWN)
    
    # Get top 10 by adjusted contribution with consistency metrics
    top_contributors = (
        contributions.group_by(['player_id', 'player_name', 'team'])
        .agg([
            pl.col('player_overall_contribution').sum().alias('adjusted_score'),  # Total adjusted score
            pl.col('player_overall_contribution').mean().alias('avg_per_game'),  # Average per game
            pl.col('peak_game_score').max().alias('peak_score'),
            pl.col('week').count().alias('games_played')
        ])
        .sort('adjusted_score', descending=True)
        .head(10)
    )
    
    # Calculate consistency metrics from per_game_contributions
    # Need to apply the same Phase 5 scaling (talent + dampening) to make metrics comparable to Avg/Game
    # Strategy: Calculate the scaling factor for each player (Phase 5 adjusted / Phase 4 mean) and apply to all metrics
    
    # Get pre-Phase 5 means for each player
    pre_phase5_means = (
        per_game_contributions.group_by(['player_id', 'player_name', 'team'])
        .agg([
            pl.col('player_overall_contribution').mean().alias('pre_phase5_mean')
        ])
    )
    
    # Get post-Phase 5 values (dampened) for each player
    post_phase5_values = (
        contributions.group_by(['player_id', 'player_name', 'team'])
        .agg([
            pl.col('player_overall_contribution').mean().alias('post_phase5_value')  # All weeks have same value after Phase 5
        ])
    )
    
    # Calculate scaling factor
    scaling_factors = pre_phase5_means.join(post_phase5_values, on=['player_id', 'player_name', 'team'])
    scaling_factors = scaling_factors.with_columns([
        (pl.col('post_phase5_value') / pl.col('pre_phase5_mean')).alias('phase5_scaling_factor')
    ])
    
    # Calculate consistency metrics from per_game_contributions
    consistency_metrics = (
        per_game_contributions.group_by(['player_id', 'player_name', 'team'])
        .agg([
            pl.col('player_overall_contribution').median().alias('typical_game_raw'),
            pl.col('player_overall_contribution').quantile(0.25).alias('floor_raw'),
            pl.col('player_overall_contribution').quantile(0.75).alias('ceiling_raw')
        ])
    )
    
    # Join scaling factors and apply to all consistency metrics
    consistency_metrics = consistency_metrics.join(scaling_factors, on=['player_id', 'player_name', 'team'], how='left')
    consistency_metrics = consistency_metrics.with_columns([
        (pl.col('typical_game_raw') * pl.col('phase5_scaling_factor')).alias('typical_game'),
        (pl.col('floor_raw') * pl.col('phase5_scaling_factor')).alias('floor'),
        (pl.col('ceiling_raw') * pl.col('phase5_scaling_factor')).alias('ceiling')
    ])
    
    # Calculate consistency profile (below/at/above average games)
    # Join typical_game back to per_game data to count performance levels
    per_game_with_typical = per_game_contributions.join(
        consistency_metrics.select(['player_id', 'player_name', 'team', 'typical_game']),
        on=['player_id', 'player_name', 'team'],
        how='left'
    )
    
    consistency_profile = (
        per_game_with_typical.group_by(['player_id', 'player_name', 'team'])
        .agg([
            # Below: < 95% of typical
            (pl.col('player_overall_contribution') < (pl.col('typical_game') * 0.95)).sum().alias('below_avg'),
            # At: 95-105% of typical
            ((pl.col('player_overall_contribution') >= (pl.col('typical_game') * 0.95)) & 
             (pl.col('player_overall_contribution') <= (pl.col('typical_game') * 1.05))).sum().alias('at_avg'),
            # Above: > 105% of typical
            (pl.col('player_overall_contribution') > (pl.col('typical_game') * 1.05)).sum().alias('above_avg')
        ])
    )
    
    # Join consistency profile to consistency metrics
    consistency_metrics = consistency_metrics.join(consistency_profile, on=['player_id', 'player_name', 'team'], how='left')
    
    # Select metrics for joining (removed std_dev, added consistency profile)
    consistency_metrics = consistency_metrics.select([
        'player_id', 'player_name', 'team', 'typical_game', 'floor', 'ceiling', 
        'below_avg', 'at_avg', 'above_avg'
    ])
    
    top_contributors = top_contributors.join(consistency_metrics, on=['player_id', 'player_name', 'team'], how='left')
    
    # Recalculate positional rankings based on ADJUSTED scores (after Phase 5)
    # This ensures the position ranks in "Overall Rankings" match the position-specific tables
    adjusted_season_contributions = contributions.group_by(['player_id', 'player_name', 'team']).agg([
        pl.col('player_overall_contribution').sum().alias('adjusted_score'),  # Total adjusted score
        pl.col('position').first().alias('position')  # Carry position through
    ])
    
    position_rankings = {}
    for pos in SKILL_POSITIONS:
        pos_stats = adjusted_season_contributions.filter(
            pl.col('position') == pos
        ).sort('adjusted_score', descending=True)
        
        position_rankings[pos] = {
            (row['player_id'], row['player_name'], row['team']): rank + 1 
            for rank, row in enumerate(pos_stats.iter_rows(named=True))
        }
    
    # Add raw scores from raw_contributions
    raw_scores = (
        raw_contributions.group_by(['player_id', 'player_name', 'team'])
        .agg([
            pl.col('player_overall_contribution').sum().alias('raw_score')  # Total raw score
        ])
    )
    top_contributors = top_contributors.join(raw_scores, on=['player_id', 'player_name', 'team'], how='left')
    
    # Join difficulty metrics
    if avg_difficulty is not None:
        top_contributors = top_contributors.join(
            avg_difficulty.select(['player_id', 'player_name', 'avg_difficulty_multiplier']),
            on=['player_id', 'player_name'],
            how='left'
        )
    
    # Get notable games (weeks where player had >150% of their average contribution)
    # Need to get opponent info from player_stats since it's not in contributions
    notable_games = {}
    for row in top_contributors.iter_rows(named=True):
        player_id = row['player_id']
        player_name = row['player_name']
        team_name = row['team']
        typical_game = row['typical_game']  # Use median for notable games threshold
        
        # Get player weeks with opponent information from player_stats
        player_weeks = player_stats.filter(
            (pl.col('player_id') == player_id) & (pl.col('player_name') == player_name) & (pl.col('team') == team_name)
        ).select(['week', 'opponent_team']).unique()
        
        # Get per-game contribution scores (before Phase 5 aggregation)
        player_contrib = per_game_contributions.filter(
            (pl.col('player_id') == player_id) & (pl.col('player_name') == player_name) & (pl.col('team') == team_name)
        ).select(['week', 'player_overall_contribution'])
        
        # Join to get both opponent and contribution
        player_full = player_contrib.join(player_weeks, on='week', how='left')
        
        high_games = player_full.filter(
            pl.col('player_overall_contribution') > typical_game * 1.5
        ).sort('player_overall_contribution', descending=True).head(3)
        
        if len(high_games) > 0:
            # Format as "Wk # (vs OPP)"
            game_strings = []
            for game_row in high_games.iter_rows(named=True):
                week_num = game_row['week']
                opponent = game_row.get('opponent_team', '?')
                game_strings.append(f"Wk {week_num} (vs {opponent})")
            notable_games[player_name] = ", ".join(game_strings)
        else:
            notable_games[player_name] = ""
    
    table = PrettyTable()
    table.field_names = ["Rank", "Player", "Team", "Position (Pos. Rank)", "Raw Score", "Adjusted Score", 
                        "Difficulty", "Games", "Avg/Game", "Typical", "Std Dev", "Floor", "Ceiling", "Peak", "Trend", "Notable Games"]
    table.align = "l"  # Left align text
    table.float_format = '.2'  # Two decimal places for floats
    
    for rank, row in enumerate(top_contributors.iter_rows(named=True), 1):
        player_id = row['player_id']
        player_name = row['player_name']
        team = row['team']
        position = player_stats.filter((pl.col('player_id') == player_id) & (pl.col('player_name') == player_name))['position'].head(1)[0]
        pos_rank = position_rankings[position].get((player_id, player_name, team), "N/A")
        position_display = f"{position} (#{pos_rank})"
        position_only = position
        raw_score = row['raw_score']
        adj_score_total = row['adjusted_score']  # Now the total adjusted score
        games_played = row['games_played']
        avg_per_game = row['avg_per_game']  # Now separate from adjusted_score
        peak = f"{row['peak_score']:.2f}"
        
        # Calculate trend (compare first half to second half median - more robust to outliers)
        # Use per_game_contributions (before Phase 5) to get actual weekly performance
        player_games = per_game_contributions.filter(
            (pl.col('player_id') == player_id) & (pl.col('player_name') == player_name)
        ).sort('week')
        if len(player_games) >= 4:
            mid_point = len(player_games) // 2
            first_half_median = player_games.head(mid_point)['player_overall_contribution'].median()
            second_half_median = player_games.tail(len(player_games) - mid_point)['player_overall_contribution'].median()
            if second_half_median > first_half_median * 1.15:  # Stricter threshold for median
                trend = "Increasing"
            elif second_half_median < first_half_median * 0.85:  # Stricter threshold for median
                trend = "Decreasing"
            else:
                trend = "Stable"
        else:
            trend = "Stable"
        
        games = notable_games.get(player_name, "")
        
        # Get consistency metrics (with None handling)
        typical = row.get('typical_game', avg_per_game)
        floor = row.get('floor', 0)
        ceiling = row.get('ceiling', 0)
        below_avg = row.get('below_avg', 0)
        at_avg = row.get('at_avg', 0)
        above_avg = row.get('above_avg', 0)
        consistency = f"{below_avg}/{at_avg}/{above_avg}"

        # Get difficulty multiplier
        difficulty_mult = row.get('avg_difficulty_multiplier', 1.0)
        difficulty_str = f"{difficulty_mult:.3f}" if difficulty_mult else "1.000"

        # Format values with None handling
        raw_score_str = f"{raw_score:.2f}" if raw_score is not None else "N/A"
        adj_score_str = f"{adj_score_total:.2f}" if adj_score_total is not None else "N/A"
        avg_per_game_str = f"{avg_per_game:.2f}" if avg_per_game is not None else "N/A"
        typical_str = f"{typical:.2f}" if typical is not None else "N/A"
        floor_str = f"{floor:.1f}" if floor is not None else "N/A"
        ceiling_str = f"{ceiling:.1f}" if ceiling is not None else "N/A"

        # Add to overview table (simplified)
        overview_table.add_row([rank, player_name, team, position_only, adj_score_str,
                                difficulty_str, games_played, avg_per_game_str, typical_str,
                                consistency, trend])

        # Add to deep dive table (full details)
        deep_dive_table.add_row([rank, player_name, team, position_display, raw_score_str, adj_score_str,
                                 difficulty_str, games_played, avg_per_game_str, typical_str, consistency,
                                 floor_str, ceiling_str, peak, trend, games])
    
    overview_md += overview_table.get_string() + "\n\n"
    deep_dive_md += deep_dive_table.get_string() + "\n\n"
    
    # Add positional rankings for RB, WR, TE
    for pos in ['RB', 'WR', 'TE']:
        overview_md += f"## {pos} Rankings\n\n"
        deep_dive_md += f"## {pos} Rankings\n\n"
        
        # Get top 10 players at this position
        pos_contributors = (
            contributions.filter(pl.col('position') == pos)
            .group_by(['player_id', 'player_name', 'team'])
            .agg([
                pl.col('player_overall_contribution').sum().alias('adjusted_score'),  # Total adjusted score
                pl.col('player_overall_contribution').mean().alias('avg_per_game'),  # Average per game
                pl.col('peak_game_score').max().alias('peak_score'),
                pl.col('week').count().alias('games_played')
            ])
            .sort('adjusted_score', descending=True)
            .head(10)
        )
        
        # Add consistency metrics for this position
        # Apply Phase 5 scaling to make metrics comparable to Avg/Game
        pos_per_game = per_game_contributions.filter(pl.col('position') == pos)
        
        # Get pre-Phase 5 means for this position
        pos_pre_phase5_means = (
            pos_per_game.group_by(['player_id', 'player_name', 'team'])
            .agg([
                pl.col('player_overall_contribution').mean().alias('pre_phase5_mean')
            ])
        )
        
        # Get post-Phase 5 values (already calculated in scaling_factors from above)
        # Join with the overall scaling_factors we calculated earlier
        pos_consistency = (
            pos_per_game.group_by(['player_id', 'player_name', 'team'])
            .agg([
                pl.col('player_overall_contribution').median().alias('typical_game_raw'),
                pl.col('player_overall_contribution').quantile(0.25).alias('floor_raw'),
                pl.col('player_overall_contribution').quantile(0.75).alias('ceiling_raw')
            ])
        )
        
        # Join scaling factors and apply
        pos_consistency = pos_consistency.join(scaling_factors.select(['player_id', 'player_name', 'team', 'phase5_scaling_factor']), 
                                               on=['player_id', 'player_name', 'team'], how='left')
        pos_consistency = pos_consistency.with_columns([
            (pl.col('typical_game_raw') * pl.col('phase5_scaling_factor')).alias('typical_game'),
            (pl.col('floor_raw') * pl.col('phase5_scaling_factor')).alias('floor'),
            (pl.col('ceiling_raw') * pl.col('phase5_scaling_factor')).alias('ceiling')
        ])
        
        # Calculate consistency profile for position
        pos_per_game_with_typical = pos_per_game.join(
            pos_consistency.select(['player_id', 'player_name', 'team', 'typical_game']),
            on=['player_id', 'player_name', 'team'],
            how='left'
        )
        
        pos_consistency_profile = (
            pos_per_game_with_typical.group_by(['player_id', 'player_name', 'team'])
            .agg([
                (pl.col('player_overall_contribution') < (pl.col('typical_game') * 0.95)).sum().alias('below_avg'),
                ((pl.col('player_overall_contribution') >= (pl.col('typical_game') * 0.95)) & 
                 (pl.col('player_overall_contribution') <= (pl.col('typical_game') * 1.05))).sum().alias('at_avg'),
                (pl.col('player_overall_contribution') > (pl.col('typical_game') * 1.05)).sum().alias('above_avg')
            ])
        )
        
        pos_consistency = pos_consistency.join(pos_consistency_profile, on=['player_id', 'player_name', 'team'], how='left')
        
        pos_consistency = pos_consistency.select([
            'player_id', 'player_name', 'team', 'typical_game', 'floor', 'ceiling',
            'below_avg', 'at_avg', 'above_avg'
        ])
        
        pos_contributors = pos_contributors.join(pos_consistency, on=['player_id', 'player_name', 'team'], how='left')
        
        # Add raw scores
        pos_contributors = pos_contributors.join(raw_scores, on=['player_id', 'player_name', 'team'], how='left')
        
        # Add difficulty metrics
        if avg_difficulty is not None:
            pos_contributors = pos_contributors.join(
                avg_difficulty.select(['player_id', 'player_name', 'avg_difficulty_multiplier']),
                on=['player_id', 'player_name'],
                how='left'
            )
        
        # Create overview table for this position (simplified)
        pos_overview_table = PrettyTable()
        pos_overview_table.field_names = ["Rank", "Player", "Team", "Adjusted Score", 
                                          "Difficulty", "Games", "Avg/Game", "Typical", "Consistency", "Trend"]
        pos_overview_table.align = "l"
        pos_overview_table.float_format = '.2'
        pos_overview_table.set_style(TableStyle.MARKDOWN)
        
        # Create deep dive table for this position (all columns)
        pos_deep_dive_table = PrettyTable()
        pos_deep_dive_table.field_names = ["Rank", "Player", "Team", "Raw Score", "Adjusted Score", 
                                "Difficulty", "Games", "Avg/Game", "Typical", "Consistency", "Floor", "Ceiling", "Peak", "Trend", "Notable Games (>150% Typical)"]
        pos_deep_dive_table.align = "l"
        pos_deep_dive_table.float_format = '.2'
        pos_deep_dive_table.set_style(TableStyle.MARKDOWN)
        
        for rank, row in enumerate(pos_contributors.iter_rows(named=True), 1):
            player_id = row['player_id']
            player_name = row['player_name']
            team = row['team']
            raw_score = row['raw_score']
            adj_score_total = row['adjusted_score']  # Now the total adjusted score
            games_played = row['games_played']
            avg_per_game = row['avg_per_game']  # Now separate from adjusted_score
            peak = f"{row['peak_score']:.2f}"
            
            # Calculate trend using median (more robust to outliers)
            player_games = per_game_contributions.filter(
                (pl.col('player_id') == player_id) & (pl.col('player_name') == player_name) & (pl.col('team') == team)
            ).sort('week')
            if len(player_games) >= 4:
                mid_point = len(player_games) // 2
                first_half_median = player_games.head(mid_point)['player_overall_contribution'].median()
                second_half_median = player_games.tail(len(player_games) - mid_point)['player_overall_contribution'].median()
                if second_half_median > first_half_median * 1.15:
                    trend = "Increasing"
                elif second_half_median < first_half_median * 0.85:
                    trend = "Decreasing"
                else:
                    trend = "Stable"
            else:
                trend = "Stable"
            
            # Get notable games for this player (reuse from notable_games dict or recalculate)
            # Use (player_id, player_name) key since that's unique
            games = notable_games.get(player_name, "")
            
            # Get consistency metrics
            typical = row.get('typical_game', avg_per_game)
            floor = row.get('floor', 0)
            ceiling = row.get('ceiling', 0)
            below_avg = row.get('below_avg', 0)
            at_avg = row.get('at_avg', 0)
            above_avg = row.get('above_avg', 0)
            consistency = f"{below_avg}/{at_avg}/{above_avg}"
            
            # Get difficulty multiplier
            difficulty_mult = row.get('avg_difficulty_multiplier', 1.0)
            difficulty_str = f"{difficulty_mult:.3f}" if difficulty_mult else "1.000"
            
            # Add to overview table (simplified)
            pos_overview_table.add_row([rank, player_name, team, f"{adj_score_total:.2f}", 
                                        difficulty_str, games_played, f"{avg_per_game:.2f}", 
                                        f"{typical:.2f}", consistency, trend])
            
            # Add to deep dive table (full details)
            pos_deep_dive_table.add_row([rank, player_name, team, f"{raw_score:.2f}", f"{adj_score_total:.2f}", 
                                         difficulty_str, games_played, f"{avg_per_game:.2f}", f"{typical:.2f}", consistency,
                                         f"{floor:.1f}", f"{ceiling:.1f}", peak, trend, games])
        
        overview_md += pos_overview_table.get_string() + "\n\n"
        deep_dive_md += pos_deep_dive_table.get_string() + "\n\n"
    
    # Add difficulty context section (only to deep dive)
    difficulty_context = calculate_difficulty_context(year, player_stats)
    if difficulty_context is not None:
        deep_dive_md += "## Difficulty Context\n\n"
        deep_dive_md += "*How challenging were the situations these players faced?*\n\n"
        
        # Create tables for each position group
        for pos in ['RB', 'WR', 'TE']:
            pos_difficulty = difficulty_context.filter(pl.col('position') == pos)
            if len(pos_difficulty) == 0:
                continue
            
            # Join with season contributions to get top players
            pos_season = raw_season_contributions.filter(pl.col('position') == pos).head(15)
            pos_with_difficulty = pos_season.join(
                pos_difficulty, 
                on=['player_id', 'player_name'], 
                how='left'
            ).sort('raw_score', descending=True)
            
            if pos == 'RB':
                # Filter to players with at least 50 rush attempts
                pos_with_difficulty = pos_with_difficulty.filter(
                    pl.col('rush_attempts') >= 50
                )
                
                if len(pos_with_difficulty) == 0:
                    continue
                
                deep_dive_md += f"### {pos} - Box Count Context\n\n"
                deep_dive_md += "*Higher box counts = tougher runs (8+ defenders = stacked box)*\n\n"
                
                diff_table = PrettyTable()
                diff_table.field_names = ["Player", "Avg Box", "8+ Box %", "Avg Multiplier", "Rush Att"]
                diff_table.align = "l"
                diff_table.float_format = '.2'
                diff_table.set_style(TableStyle.MARKDOWN)
                
                for row in pos_with_difficulty.iter_rows(named=True):
                    if row.get('avg_defenders_in_box') is not None:
                        player_name = row['player_name']
                        avg_box = row['avg_defenders_in_box']
                        pct_8plus = row['pct_vs_8plus_box'] * 100
                        avg_mult = row['avg_box_multiplier']
                        rush_att = int(row['rush_attempts'])
                        diff_table.add_row([player_name, f"{avg_box:.1f}", f"{pct_8plus:.1f}%", 
                                          f"{avg_mult:.3f}", rush_att])
                
                deep_dive_md += diff_table.get_string() + "\n\n"
            
            else:  # WR or TE
                # Filter to players with at least 20 targets
                pos_with_difficulty = pos_with_difficulty.filter(
                    pl.col('targets') >= 20
                )
                
                if len(pos_with_difficulty) == 0:
                    continue
                
                deep_dive_md += f"### {pos} - Coverage Context\n\n"
                deep_dive_md += "*Higher man coverage % = tougher matchups (2-Man, 1-Man, 0-Man, Cover-1)*\n\n"
                
                diff_table = PrettyTable()
                diff_table.field_names = ["Player", "Man Cov %", "Avg Multiplier", "Targets"]
                diff_table.align = "l"
                diff_table.float_format = '.2'
                diff_table.set_style(TableStyle.MARKDOWN)
                
                for row in pos_with_difficulty.iter_rows(named=True):
                    if row.get('pct_vs_man_coverage') is not None:
                        player_name = row['player_name']
                        pct_man = row['pct_vs_man_coverage'] * 100
                        avg_mult = row['avg_coverage_multiplier']
                        targets = int(row['targets'])
                        diff_table.add_row([player_name, f"{pct_man:.1f}%", 
                                          f"{avg_mult:.3f}", targets])
                
                deep_dive_md += diff_table.get_string() + "\n\n"
        
        deep_dive_md += "*Note: Multipliers > 1.0 indicate tougher situations (bonus), < 1.0 indicate easier situations (penalty)*\n\n"
    
    # Add FTN context section (only for years 2022+)
    from modules.ftn_cache_builder import FTN_START_YEAR
    if year >= FTN_START_YEAR:
        ftn_context = generate_ftn_context(year, player_stats, raw_season_contributions)
        if ftn_context:
            deep_dive_md += ftn_context
    
    return overview_md, deep_dive_md

