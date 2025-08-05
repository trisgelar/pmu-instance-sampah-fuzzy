# file: modules/metrics_analyzer.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import logging
from typing import Optional, Dict, Any
import numpy as np

from modules.exceptions import FileOperationError, ValidationError

# Configure logging
logger = logging.getLogger(__name__)

class TrainingMetricsAnalyzer:
    """
    Mengelola visualisasi metrik pelatihan dan menampilkan tabel metrik performa akhir.
    """
    def __init__(self):
        """Initialize TrainingMetricsAnalyzer."""
        logger.info("TrainingMetricsAnalyzer initialized")

    def _validate_run_directory(self, train_run_dir: str) -> None:
        """
        Validate training run directory.
        
        Args:
            train_run_dir: Path to training run directory
            
        Raises:
            ValidationError: If directory is invalid
        """
        if not train_run_dir or not isinstance(train_run_dir, str):
            raise ValidationError("train_run_dir must be a non-empty string")
            
        if not os.path.exists(train_run_dir):
            raise ValidationError(f"Training run directory does not exist: {train_run_dir}")
            
        if not os.path.isdir(train_run_dir):
            raise ValidationError(f"Path is not a directory: {train_run_dir}")

    def _validate_model_version(self, model_version: str) -> None:
        """
        Validate model version parameter.
        
        Args:
            model_version: Model version to validate
            
        Raises:
            ValidationError: If model version is invalid
        """
        if not model_version or not isinstance(model_version, str):
            raise ValidationError("model_version must be a non-empty string")

    def _load_training_results(self, train_run_dir: str) -> pd.DataFrame:
        """
        Load training results from CSV file.
        
        Args:
            train_run_dir: Path to training run directory
            
        Returns:
            pd.DataFrame: Training results data
            
        Raises:
            FileOperationError: If results file cannot be loaded
        """
        results_path = os.path.join(train_run_dir, "results.csv")
        
        if not os.path.exists(results_path):
            raise FileOperationError(f"Results file not found at {results_path}")
            
        try:
            df = pd.read_csv(results_path)
            df.columns = df.columns.str.strip()
            
            # Validate required columns exist
            required_columns = ['epoch', 'train/box_loss', 'val/box_loss', 'train/seg_loss', 'val/seg_loss']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise FileOperationError(f"Missing required columns in results.csv: {missing_columns}")
                
            logger.info(f"Successfully loaded training results with {len(df)} epochs")
            return df
            
        except pd.errors.EmptyDataError:
            raise FileOperationError(f"Results file is empty: {results_path}")
        except pd.errors.ParserError as e:
            raise FileOperationError(f"Failed to parse results.csv: {str(e)}") from e
        except Exception as e:
            raise FileOperationError(f"Failed to load training results: {str(e)}") from e

    def plot_training_metrics(self, train_run_dir: str, model_version: str) -> bool:
        """
        Membuat plot metrik pelatihan dari file results.csv untuk segmentasi.
        
        Args:
            train_run_dir: Path to training run directory
            model_version: Model version for plot title
            
        Returns:
            bool: True if successful, False otherwise
            
        Raises:
            ValidationError: If parameters are invalid
            FileOperationError: If plotting fails
        """
        try:
            # Validate parameters
            self._validate_run_directory(train_run_dir)
            self._validate_model_version(model_version)
            
            # Load training results
            df = self._load_training_results(train_run_dir)
            
            logger.info(f"Creating training metrics plot for YOLO{model_version} (segment)")
            
            # Create plot
            plt.style.use('seaborn-v0_8-darkgrid')
            fig, axes = plt.subplots(1, 2, figsize=(18, 7))
            fig.suptitle(f'Training Metrics for YOLO{model_version} (Instance Segmentation)', fontsize=16)

            # Plot loss curves
            try:
                axes[0].plot(df['epoch'], df['train/box_loss'], label='Train Box Loss', 
                           color='blue', linestyle='-', marker='o', markersize=4)
                axes[0].plot(df['epoch'], df['val/box_loss'], label='Val Box Loss', 
                           color='red', linestyle='--', marker='o', markersize=4)
                axes[0].plot(df['epoch'], df['train/seg_loss'], label='Train Seg Loss', 
                           color='cyan', linestyle='-', marker='x', markersize=4)
                axes[0].plot(df['epoch'], df['val/seg_loss'], label='Val Seg Loss', 
                           color='magenta', linestyle='--', marker='x', markersize=4)
                axes[0].set_title('Loss Curves')
                axes[0].set_xlabel('Epoch')
                axes[0].set_ylabel('Loss')
                axes[0].legend()
                axes[0].grid(True)
                axes[0].set_ylim(bottom=0)
            except KeyError as e:
                raise FileOperationError(f"Missing required loss column in results: {str(e)}") from e

            # Plot performance metrics
            try:
                if 'metrics/mAP50(M)' in df.columns and 'metrics/mAP50-95(M)' in df.columns:
                    axes[1].plot(df['epoch'], df['metrics/mAP50(M)'], label='mAP50 (Mask)', 
                               color='purple', linestyle='-', marker='x', markersize=4)
                    axes[1].plot(df['epoch'], df['metrics/mAP50-95(M)'], label='mAP50-95 (Mask)', 
                               color='brown', linestyle='--', marker='+', markersize=4)
                    axes[1].set_title('Performance Metrics')
                    axes[1].set_xlabel('Epoch')
                    axes[1].set_ylabel('Metric Value')
                    axes[1].legend()
                    axes[1].grid(True)
                    axes[1].set_ylim(bottom=0)
                else:
                    logger.warning("Performance metrics columns not found, skipping performance plot")
                    axes[1].text(0.5, 0.5, 'Performance metrics not available', 
                               ha='center', va='center', transform=axes[1].transAxes)
                    axes[1].set_title('Performance Metrics (Not Available)')
            except Exception as e:
                logger.warning(f"Failed to plot performance metrics: {str(e)}")
                axes[1].text(0.5, 0.5, 'Error plotting performance metrics', 
                           ha='center', va='center', transform=axes[1].transAxes)
                axes[1].set_title('Performance Metrics (Error)')

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            # Save plot
            plot_filename = f"training_metrics_yolo{model_version}_segment.png"
            plot_path = os.path.join(train_run_dir, plot_filename)
            
            try:
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                logger.info(f"Training metrics plot saved to: {plot_path}")
                plt.close(fig)
                return True
            except Exception as e:
                raise FileOperationError(f"Failed to save plot: {str(e)}") from e
                
        except Exception as e:
            if isinstance(e, (ValidationError, FileOperationError)):
                raise
            error_msg = f"Unexpected error during metrics plotting: {str(e)}"
            logger.error(error_msg)
            raise FileOperationError(error_msg) from e

    def display_final_metrics_table(self, train_run_dir: str, model_version: str) -> bool:
        """
        Menampilkan tabel metrik performa akhir untuk segmentasi.
        
        Args:
            train_run_dir: Path to training run directory
            model_version: Model version for display
            
        Returns:
            bool: True if successful, False otherwise
            
        Raises:
            ValidationError: If parameters are invalid
            FileOperationError: If metrics display fails
        """
        try:
            # Validate parameters
            self._validate_run_directory(train_run_dir)
            self._validate_model_version(model_version)
            
            # Load training results
            df = self._load_training_results(train_run_dir)
            
            logger.info(f"Displaying final metrics table for YOLO{model_version} (Instance Segmentation)")
            
            # Get final metrics
            final_metrics = df.iloc[-1]
            
            # Prepare metrics data
            metrics_data = []
            
            # Add available metrics
            if 'metrics/mAP50(M)' in final_metrics:
                metrics_data.append({
                    "Metric": "mAP50 (Mask)",
                    "Value": f"{final_metrics['metrics/mAP50(M)']:.4f}"
                })
            
            if 'metrics/mAP50-95(M)' in final_metrics:
                metrics_data.append({
                    "Metric": "mAP50-95 (Mask)",
                    "Value": f"{final_metrics['metrics/mAP50-95(M)']:.4f}"
                })
            
            # Add loss metrics
            if 'val/box_loss' in final_metrics:
                metrics_data.append({
                    "Metric": "Final Val Box Loss",
                    "Value": f"{final_metrics['val/box_loss']:.4f}"
                })
            
            if 'val/seg_loss' in final_metrics:
                metrics_data.append({
                    "Metric": "Final Val Seg Loss",
                    "Value": f"{final_metrics['val/seg_loss']:.4f}"
                })
            
            if not metrics_data:
                logger.warning("No valid metrics found in results")
                return False
            
            # Create and display table
            metrics_df = pd.DataFrame(metrics_data)
            print(f"\n--- Final Performance Metrics for YOLO{model_version} (Instance Segmentation) ---")
            print(metrics_df.to_string(index=False))
            print("-" * 70)
            
            logger.info("Final metrics table displayed successfully")
            return True
            
        except Exception as e:
            if isinstance(e, (ValidationError, FileOperationError)):
                raise
            error_msg = f"Unexpected error during metrics display: {str(e)}"
            logger.error(error_msg)
            raise FileOperationError(error_msg) from e

    def get_training_summary(self, train_run_dir: str) -> Dict[str, Any]:
        """
        Get comprehensive training summary.
        
        Args:
            train_run_dir: Path to training run directory
            
        Returns:
            Dict[str, Any]: Training summary dictionary
        """
        try:
            self._validate_run_directory(train_run_dir)
            df = self._load_training_results(train_run_dir)
            
            summary = {
                "total_epochs": len(df),
                "final_epoch": df['epoch'].iloc[-1] if 'epoch' in df.columns else None,
                "best_metrics": {},
                "final_metrics": {},
                "training_completed": True
            }
            
            # Get final metrics
            final_metrics = df.iloc[-1]
            
            # Add available metrics to summary
            metric_columns = [col for col in df.columns if col.startswith('metrics/')]
            for col in metric_columns:
                if col in final_metrics:
                    summary["final_metrics"][col] = final_metrics[col]
                    
                    # Find best value for this metric
                    if col in df.columns:
                        best_value = df[col].max() if 'mAP' in col else df[col].min()
                        best_epoch = df.loc[df[col] == best_value, 'epoch'].iloc[0] if len(df[df[col] == best_value]) > 0 else None
                        summary["best_metrics"][col] = {
                            "value": best_value,
                            "epoch": best_epoch
                        }
            
            logger.info(f"Training summary generated for {len(df)} epochs")
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate training summary: {str(e)}")
            return {
                "error": str(e),
                "training_completed": False
            }
