#!/usr/bin/env python3
"""
Script to use existing training results without retraining.
This script will find existing training runs and use them for analysis and inference.
"""

import sys
import os
import logging
from typing import Optional

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


def use_existing_training(model_version: str = "v8n", num_inference_images: int = 6):
    """
    Use existing training results for analysis and inference.
    
    Args:
        model_version: Model version to use (e.g., "v8n", "v10n", "v11n")
        num_inference_images: Number of images for inference
    """
    try:
        logger.info(f"🔍 Looking for existing training runs for YOLO{model_version}")
        
        # Initialize the system
        system_colab = WasteDetectionSystemColab()
        
        # Find existing training run
        existing_run = system_colab.find_existing_training_run(model_version)
        
        if not existing_run:
            logger.error(f"❌ No existing training run found for YOLO{model_version}")
            logger.info("💡 You need to train a model first using main_colab.py")
            return False
        
        logger.info(f"✅ Found existing training run: {existing_run}")
        
        # Run analysis and inference
        logger.info(f"📊 Analyzing training run for YOLO{model_version}")
        system_colab.analyze_training_run(existing_run, model_version)
        
        logger.info(f"🔍 Running inference and visualization for YOLO{model_version}")
        system_colab.run_inference_and_visualization(existing_run, model_version, num_inference_images)
        
        logger.info(f"📦 Converting and zipping RKNN models for YOLO{model_version}")
        system_colab.convert_and_zip_rknn_models(model_version)
        
        logger.info("🎉 Successfully completed analysis and inference using existing training!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error using existing training: {str(e)}")
        return False


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Use existing training results")
    parser.add_argument("--model", choices=["v8n", "v10n", "v11n"], 
                       default="v8n", help="Model version to use")
    parser.add_argument("--images", type=int, default=6, 
                       help="Number of images for inference")
    
    args = parser.parse_args()
    
    success = use_existing_training(args.model, args.images)
    
    if success:
        print("\n🎉 Successfully completed using existing training results!")
        print("📁 Check the results in the output directories.")
    else:
        print("\n💥 Failed to use existing training results.")
        sys.exit(1)


if __name__ == "__main__":
    main() 