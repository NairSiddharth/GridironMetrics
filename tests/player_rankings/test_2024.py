"""Quick test script for 2024 personnel multipliers"""

from pathlib import Path
from modules.logger import get_logger

logger = get_logger(__name__)

# Import main processing functions
import sys
sys.path.insert(0, str(Path(__file__).parent))

from main import process_year

if __name__ == "__main__":
    logger.info("Testing 2024 with personnel multipliers...")
    
    try:
        success, weekly, summary, top_contributors, qb_rankings = process_year(2024)
        
        if success:
            logger.info("✅ 2024 processing completed successfully")
            
            # Save just 2024 results
            output_dir = Path("output") / "2024"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            (output_dir / "top_contributors.md").write_text(top_contributors, encoding='utf-8')
            (output_dir / "season_summary.md").write_text(summary, encoding='utf-8')
            
            logger.info(f"✅ Results saved to {output_dir}")
            
            # Show top 5 contributors
            print("\n" + "="*80)
            print("TOP 5 CONTRIBUTORS (2024)")
            print("="*80)
            lines = top_contributors.split('\n')
            for i, line in enumerate(lines):
                if '| Rank |' in line:
                    # Print header and next 7 lines (header, separator, top 5)
                    for j in range(i, min(i+8, len(lines))):
                        print(lines[j])
                    break
            print("="*80)
            
        else:
            logger.error("❌ 2024 processing failed")
            
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
