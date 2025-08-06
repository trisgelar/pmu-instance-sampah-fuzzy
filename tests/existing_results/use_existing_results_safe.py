#!/usr/bin/env python3
"""
Safe script to use existing training results for analysis and RKNN conversion.
This script skips the problematic inference visualization.
"""

import sys
import os
import logging

# Add the project root to the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from main_colab import WasteDetectionSystemColab

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def use_existing_results_safe(model_version: str = "v8n"):
    """
    Use existing training results for analysis and RKNN conversion (safe version).
    
    Args:
        model_version: Model version to use (e.g., "v8n", "v10n", "v11n")
    """
    try:
        logger.info(f"🔍 Looking for existing training runs for YOLO{model_version}")
        
        # Initialize the system
        system_colab = WasteDetectionSystemColab()
        
        # Find existing training run
        existing_run = system_colab.find_existing_training_run(model_version)
        
        if not existing_run:
            logger.error(f"❌ No existing training run found for YOLO{model_version}")
            return False
        
        logger.info(f"✅ Found existing training run: {existing_run}")
        
        # Run analysis (this works perfectly)
        logger.info(f"📊 Running analyze_training_run...")
        system_colab.analyze_training_run(existing_run, model_version)
        
        # Skip inference visualization (has RGBA issue)
        logger.info(f"⏭️ Skipping run_inference_and_visualization (RGBA issue)")
        
        # Run RKNN conversion (this should work)
        logger.info(f"📦 Running convert_and_zip_rknn_models...")
        system_colab.convert_and_zip_rknn_models(model_version)
        
        logger.info("🎉 Successfully completed analysis and RKNN conversion using existing training!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        return False


if __name__ == "__main__":
    # Use v8n by default
    success = use_existing_results_safe("v8n")
    
    if success:
        print("\n🎉 Successfully completed using existing training results!")
        print("📁 Check the results in the output directories.")
        print("📊 Training metrics plot saved")
        print("📦 RKNN models converted and zipped")
    else:
        print("\n💥 Failed to use existing training results.")
        sys.exit(1) 