"""Rebuild PBP caches for all years with participation data (2016-2024)"""

from modules.logger import get_logger
from modules.injury_cache_builder import build_injury_cache
from modules.penalty_cache_builder import build_penalty_cache
from modules.weather_enricher import enrich_all_years as enrich_github_weather
from modules.nflweather_scraper import enrich_all_years as enrich_nflweather
from pathlib import Path
import subprocess
import sys

logger = get_logger(__name__)

# Years with participation data available
YEARS_TO_REBUILD = list(range(2016, 2025))

if __name__ == "__main__":
    logger.info(f"Rebuilding PBP caches for {len(YEARS_TO_REBUILD)} years (2016-2024)")
    
    # First, rebuild injury cache
    logger.info("=" * 80)
    logger.info("Building injury and roster caches...")
    logger.info("=" * 80)
    build_injury_cache(YEARS_TO_REBUILD[0], YEARS_TO_REBUILD[-1])
    
    # Build penalty cache
    logger.info("=" * 80)
    logger.info("Building penalty caches...")
    logger.info("=" * 80)
    build_penalty_cache(YEARS_TO_REBUILD[0], YEARS_TO_REBUILD[-1])
    
    python_exe = sys.executable
    
    for year in YEARS_TO_REBUILD:
        logger.info(f"=" * 80)
        logger.info(f"Rebuilding cache for {year}...")
        logger.info(f"=" * 80)
        
        try:
            result = subprocess.run(
                [python_exe, "-m", "modules.pbp_cache_builder", "--year", str(year), "--force"],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Print last few lines of output
            output_lines = result.stdout.strip().split('\n')
            for line in output_lines[-5:]:
                print(line)
            
            logger.info(f"Successfully rebuilt cache for {year}")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to rebuild cache for {year}: {e}")
            logger.error(f"Error output: {e.stderr}")

    # Enrich weather data for all years after PBP rebuild
    logger.info("=" * 80)
    logger.info("Enriching weather data from GitHub (2000-2020)...")
    logger.info("=" * 80)
    enrich_github_weather(start_year=2000, end_year=2020, dry_run=False)

    logger.info("=" * 80)
    logger.info("Enriching weather data from NFLWeather (2021-2025)...")
    logger.info("=" * 80)
    enrich_nflweather(start_year=2021, end_year=2025, dry_run=False)

    logger.info("=" * 80)
    logger.info("Cache rebuild complete!")
    logger.info("=" * 80)
