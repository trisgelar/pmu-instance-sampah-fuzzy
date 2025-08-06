#!/usr/bin/env python3
"""
Enhanced Training System with Checkpoint Support

This module provides separate execution functions for local and Colab environments
with enhanced checkpoint functionality for stable long-term training.
"""

import os
import sys
import logging
import json
import time
from typing import Optional, Dict, Any, List
from datetime import datetime

# Add the project root to the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from main_colab import WasteDetectionSystemColab
from modules.exceptions import ModelError, ConfigurationError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class EnhancedTrainingSystem:
    """
    Enhanced training system with checkpoint support and separate execution modes.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the enhanced training system.
        
        Args:
            config_path: Path to configuration file
        """
        self.system = WasteDetectionSystemColab(config_path)
        self.checkpoint_dir = "checkpoints"
        self.checkpoint_file = os.path.join(self.checkpoint_dir, "training_state.json")
        
        # Create checkpoint directory
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        logger.info("Enhanced Training System initialized")
    
    def save_checkpoint(self, model_version: str, epoch: int, status: str, 
                       additional_info: Optional[Dict] = None) -> bool:
        """
        Save training checkpoint.
        
        Args:
            model_version: Model version being trained
            epoch: Current epoch
            status: Training status
            additional_info: Additional information to save
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            checkpoint_data = {
                "model_version": model_version,
                "epoch": epoch,
                "status": status,
                "timestamp": datetime.now().isoformat(),
                "additional_info": additional_info or {}
            }
            
            with open(self.checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            
            logger.info(f"Checkpoint saved: {model_version} at epoch {epoch} - {status}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {str(e)}")
            return False
    
    def load_checkpoint(self) -> Optional[Dict]:
        """
        Load training checkpoint.
        
        Returns:
            Optional[Dict]: Checkpoint data or None if not found
        """
        try:
            if not os.path.exists(self.checkpoint_file):
                return None
            
            with open(self.checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
            
            logger.info(f"Checkpoint loaded: {checkpoint_data.get('model_version')} at epoch {checkpoint_data.get('epoch')}")
            return checkpoint_data
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {str(e)}")
            return None
    
    def clear_checkpoint(self) -> bool:
        """
        Clear training checkpoint.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if os.path.exists(self.checkpoint_file):
                os.remove(self.checkpoint_file)
                logger.info("Checkpoint cleared")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear checkpoint: {str(e)}")
            return False
    
    def train_with_checkpoints(self, model_version: str, epochs: int = 200, 
                              batch_size: Optional[int] = None, 
                              resume_from_checkpoint: bool = True) -> Optional[str]:
        """
        Train model with checkpoint support for long training sessions.
        
        Args:
            model_version: Model version to train
            epochs: Total number of epochs
            batch_size: Training batch size
            resume_from_checkpoint: Whether to resume from checkpoint
            
        Returns:
            Optional[str]: Path to training run directory or None if failed
        """
        try:
            # Load checkpoint if resuming
            checkpoint = None
            if resume_from_checkpoint:
                checkpoint = self.load_checkpoint()
                if checkpoint and checkpoint.get('model_version') == model_version:
                    logger.info(f"Resuming training from epoch {checkpoint.get('epoch', 0)}")
                else:
                    logger.info("Starting fresh training")
                    self.clear_checkpoint()
            
            # Start training
            logger.info(f"Starting enhanced training for YOLO{model_version} with {epochs} epochs")
            
            # Save initial checkpoint
            self.save_checkpoint(model_version, 0, "started")
            
            # Train model
            run_dir = self.system.train_and_export_model(
                model_version=model_version,
                epochs=epochs,
                batch_size=batch_size
            )
            
            if run_dir:
                # Save completion checkpoint
                self.save_checkpoint(model_version, epochs, "completed", {
                    "run_dir": run_dir,
                    "final_epochs": epochs
                })
                logger.info(f"Training completed successfully for YOLO{model_version}")
                return run_dir
            else:
                # Save failure checkpoint
                self.save_checkpoint(model_version, 0, "failed", {
                    "error": "Training failed"
                })
                logger.error(f"Training failed for YOLO{model_version}")
                return None
                
        except Exception as e:
            # Save error checkpoint
            self.save_checkpoint(model_version, 0, "error", {
                "error": str(e)
            })
            logger.error(f"Training error for YOLO{model_version}: {str(e)}")
            raise ModelError(f"Training failed: {str(e)}") from e
    
    def execute_local_training(self, model_version: str = "v8n", epochs: int = 200, 
                              batch_size: Optional[int] = None) -> bool:
        """
        Execute training in local environment with enhanced monitoring.
        
        Args:
            model_version: Model version to train
            epochs: Number of training epochs
            batch_size: Training batch size
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info("=== LOCAL TRAINING EXECUTION ===")
            logger.info(f"Environment: Local")
            logger.info(f"Model: YOLO{model_version}")
            logger.info(f"Epochs: {epochs}")
            logger.info(f"Batch Size: {batch_size or 'default'}")
            
            # Display system status
            status = self.system.get_system_status()
            logger.info(f"System Status: {status}")
            
            # Train with checkpoints
            run_dir = self.train_with_checkpoints(
                model_version=model_version,
                epochs=epochs,
                batch_size=batch_size,
                resume_from_checkpoint=True
            )
            
            if run_dir:
                logger.info("✅ Local training completed successfully")
                return True
            else:
                logger.error("❌ Local training failed")
                return False
                
        except Exception as e:
            logger.error(f"Local training execution failed: {str(e)}")
            return False
    
    def execute_colab_training(self, model_version: str = "v8n", epochs: int = 200, 
                              batch_size: Optional[int] = None) -> bool:
        """
        Execute training in Google Colab environment with enhanced monitoring.
        
        Args:
            model_version: Model version to train
            epochs: Number of training epochs
            batch_size: Training batch size
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info("=== GOOGLE COLAB TRAINING EXECUTION ===")
            logger.info(f"Environment: Google Colab")
            logger.info(f"Model: YOLO{model_version}")
            logger.info(f"Epochs: {epochs}")
            logger.info(f"Batch Size: {batch_size or 'default'}")
            
            # Display system status
            status = self.system.get_system_status()
            logger.info(f"System Status: {status}")
            
            # Check CUDA status for Colab
            cuda_status = self.system.get_cuda_status()
            logger.info(f"CUDA Status: {cuda_status}")
            
            # Train with checkpoints
            run_dir = self.train_with_checkpoints(
                model_version=model_version,
                epochs=epochs,
                batch_size=batch_size,
                resume_from_checkpoint=True
            )
            
            if run_dir:
                logger.info("✅ Colab training completed successfully")
                return True
            else:
                logger.error("❌ Colab training failed")
                return False
                
        except Exception as e:
            logger.error(f"Colab training execution failed: {str(e)}")
            return False
    
    def execute_dataset_preparation(self) -> bool:
        """
        Execute dataset preparation step.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info("=== DATASET PREPARATION ===")
            
            # Prepare datasets
            success = self.system.dataset_manager.prepare_datasets()
            if success:
                logger.info("✅ Dataset preparation completed")
                
                # Validate dataset
                validation_results = self.system.dataset_manager.validate_dataset_format()
                if not validation_results.get('issues'):
                    logger.info("✅ Dataset validation passed")
                    return True
                else:
                    logger.warning("⚠️ Dataset validation issues found:")
                    for issue in validation_results['issues']:
                        logger.warning(f"  - {issue}")
                    return False
            else:
                logger.error("❌ Dataset preparation failed")
                return False
                
        except Exception as e:
            logger.error(f"Dataset preparation failed: {str(e)}")
            return False
    
    def get_training_status(self) -> Dict[str, Any]:
        """
        Get current training status including checkpoint information.
        
        Returns:
            Dict[str, Any]: Training status information
        """
        checkpoint = self.load_checkpoint()
        
        status = {
            "checkpoint_exists": checkpoint is not None,
            "system_status": self.system.get_system_status(),
            "cuda_status": self.system.get_cuda_status()
        }
        
        if checkpoint:
            status.update({
                "checkpoint_model": checkpoint.get('model_version'),
                "checkpoint_epoch": checkpoint.get('epoch'),
                "checkpoint_status": checkpoint.get('status'),
                "checkpoint_timestamp": checkpoint.get('timestamp')
            })
        
        return status


def main_local():
    """
    Main execution function for local environment.
    """
    try:
        enhanced_system = EnhancedTrainingSystem()
        
        # Display initial status
        status = enhanced_system.get_training_status()
        logger.info(f"Initial Status: {status}")
        
        # Execute local training
        success = enhanced_system.execute_local_training(
            model_version="v8n",
            epochs=200,
            batch_size=16
        )
        
        if success:
            logger.info("🎉 Local training completed successfully!")
        else:
            logger.error("💥 Local training failed!")
            
    except Exception as e:
        logger.error(f"Critical error in local execution: {str(e)}")
        sys.exit(1)


def main_colab():
    """
    Main execution function for Google Colab environment.
    """
    try:
        enhanced_system = EnhancedTrainingSystem()
        
        # Display initial status
        status = enhanced_system.get_training_status()
        logger.info(f"Initial Status: {status}")
        
        # Execute Colab training
        success = enhanced_system.execute_colab_training(
            model_version="v8n",
            epochs=200,
            batch_size=16
        )
        
        if success:
            logger.info("🎉 Colab training completed successfully!")
        else:
            logger.error("💥 Colab training failed!")
            
    except Exception as e:
        logger.error(f"Critical error in Colab execution: {str(e)}")
        sys.exit(1)


def main_dataset_prep():
    """
    Main execution function for dataset preparation only.
    """
    try:
        enhanced_system = EnhancedTrainingSystem()
        
        # Execute dataset preparation
        success = enhanced_system.execute_dataset_preparation()
        
        if success:
            logger.info("🎉 Dataset preparation completed successfully!")
        else:
            logger.error("💥 Dataset preparation failed!")
            
    except Exception as e:
        logger.error(f"Critical error in dataset preparation: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Training System")
    parser.add_argument("--mode", choices=["local", "colab", "dataset"], 
                       default="local", help="Execution mode")
    parser.add_argument("--model", choices=["v8n", "v10n", "v11n"], 
                       default="v8n", help="Model version")
    parser.add_argument("--epochs", type=int, default=200, 
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, 
                       help="Training batch size")
    
    args = parser.parse_args()
    
    if args.mode == "local":
        main_local()
    elif args.mode == "colab":
        main_colab()
    elif args.mode == "dataset":
        main_dataset_prep() 