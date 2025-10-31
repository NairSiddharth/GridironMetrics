"""Rebuild PBP caches for all years with participation data (2016-2024)"""

from modules.logger import get_logger
from modules.injury_cache_builder import build_injury_cache
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
            
    logger.info("=" * 80)
    logger.info("Cache rebuild complete!")
    logger.info("=" * 80)
