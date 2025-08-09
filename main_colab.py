# file: main_colab.py
import os
import shutil
import logging
import sys
from typing import Optional, Dict, Any

# Impor semua kelas dari modul terpisah
from modules.dataset_manager import DatasetManager
from modules.model_processor import ModelProcessor
from modules.rknn_converter import RknnConverter
from modules.metrics_analyzer import TrainingMetricsAnalyzer
from modules.yolov8_visualizer_simple import YOLOv8Visualizer
from modules.inference_visualizer import InferenceVisualizer
from modules.drive_manager import DriveManager
from modules.config_manager import ConfigManager
from modules.cuda_manager import CUDAManager
from modules.exceptions import (
    WasteDetectionError, DatasetError, ModelError, FuzzyLogicError, 
    InferenceError, ConfigurationError, FileOperationError, APIError, ValidationError
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/waste_detection_system.log')
    ]
)
logger = logging.getLogger(__name__)

# --- PENTING: Langkah 0 - Buat file secrets.yaml ---
# Sebelum menjalankan program ini, jalankan cell berikut di Colab dan isikan kunci API Anda:
#
# import yaml
# with open('secrets.yaml', 'w') as f:
#     yaml.dump({'roboflow_api_key': 'YOUR_ROBOFLOW_API_KEY'}, f)
#
# Pastikan untuk mengganti 'YOUR_ROBOFLOW_API_KEY' dengan kunci API Anda.
# Setelah file dibuat, Anda tidak perlu mengubah kode di bawah ini.

class WasteDetectionSystemColab:
    """
    Kelas orkestrator yang mengkoordinasikan proses deteksi sampah di Google Colab,
    fokus pada Instance Segmentation dengan konfigurasi terpusat.
    """
    def __init__(self, config_path: Optional[str] = None, environment: Optional[str] = None):
        """
        Initialize WasteDetectionSystemColab with centralized configuration.
        
        Args:
            config_path: Path to configuration file
            environment: Environment to load (development, production, testing, colab)
            
        Raises:
            ConfigurationError: If initialization fails
        """
        try:
            # Initialize configuration manager
            self.config_manager = ConfigManager(config_path, environment)
            
            # Get configuration sections
            self.model_config = self.config_manager.get_model_config()
            self.dataset_config = self.config_manager.get_dataset_config()
            self.fuzzy_config = self.config_manager.get_fuzzy_config()
            self.logging_config = self.config_manager.get_logging_config()
            self.system_config = self.config_manager.get_system_config()
            
            # Validate and set configuration
            self._validate_configuration()
            
            # Initialize components with configuration
            self._initialize_components()
            
            logger.info("WasteDetectionSystemColab initialized successfully with configuration")
            
        except Exception as e:
            error_msg = f"Failed to initialize WasteDetectionSystemColab: {str(e)}"
            logger.error(error_msg)
            raise ConfigurationError(error_msg) from e

    def _validate_configuration(self) -> None:
        """
        Validate system configuration.
        
        Raises:
            ConfigurationError: If configuration is invalid
        """
        # Check if secrets.yaml exists
        if not os.path.exists('secrets.yaml'):
            raise ConfigurationError(
                "secrets.yaml file not found. Please create this file with your Roboflow API key."
            )

    def _initialize_components(self) -> None:
        """
        Initialize all system components with configuration.
        
        Raises:
            ConfigurationError: If component initialization fails
        """
        try:
            # Initialize DatasetManager with configuration
            self.dataset_manager = DatasetManager(
                dataset_dir=self.dataset_config.default_dataset_dir,
                is_project=self.dataset_config.roboflow_project,
                is_version=self.dataset_config.roboflow_version
            )
            logger.info("DatasetManager initialized successfully")
            
            # Initialize ModelProcessor with configuration
            self.model_processor = ModelProcessor(
                model_dir=self.dataset_config.default_model_dir,
                onnx_model_dir=self.dataset_config.default_onnx_dir,
                img_size=self.model_config.default_img_size
            )
            logger.info("ModelProcessor initialized successfully")
            
            # Initialize RknnConverter with configuration
            self.rknn_converter = RknnConverter(
                rknn_model_dir=self.dataset_config.default_rknn_dir,
                img_size=self.model_config.default_img_size
            )
            logger.info("RknnConverter initialized successfully")
            
            # Initialize CUDA manager
            self.cuda_manager = CUDAManager()
            logger.info("CUDAManager initialized successfully")
            
            # Initialize other components
            self.metrics_analyzer = TrainingMetricsAnalyzer()
            logger.info("TrainingMetricsAnalyzer initialized successfully")
            
            self.inference_visualizer = InferenceVisualizer(
                model_dir=self.dataset_config.default_model_dir,
                img_size=self.model_config.default_img_size
            )
            logger.info("InferenceVisualizer initialized successfully")
            
            # Initialize YOLOv8 specialized visualizer for academic papers
            self.yolov8_visualizer = YOLOv8Visualizer(
                base_output_dir="results/inference_outputs", 
                img_size=self.model_config.default_img_size
            )
            logger.info("YOLOv8Visualizer initialized successfully")
            
            self.drive_manager = DriveManager()
            logger.info("DriveManager initialized successfully")
            
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize components: {str(e)}") from e

    def train_and_export_model(self, model_version: str, epochs: Optional[int] = None, 
                              batch_size: Optional[int] = None) -> Optional[str]:
        """
        Menjalankan pipeline pelatihan dan ekspor model dengan konfigurasi terpusat.
        Mengembalikan run_dir untuk digunakan pada analisis selanjutnya.
        
        Args:
            model_version: YOLO model version to train
            epochs: Number of training epochs (uses default from config if None)
            batch_size: Training batch size (uses default from config if None)
            
        Returns:
            Optional[str]: Path to training run directory or None if failed
            
        Raises:
            ValidationError: If parameters are invalid
            ModelError: If training fails
        """
        try:
            # Use configuration defaults if not provided
            if epochs is None:
                epochs = self.model_config.default_epochs
            if batch_size is None:
                batch_size = self.model_config.default_batch_size
            
            # Validate parameters against configuration
            if not model_version or not isinstance(model_version, str):
                raise ValidationError("model_version must be a non-empty string")
                
            if model_version not in self.model_config.supported_versions:
                raise ValidationError(f"Model version '{model_version}' not supported. Supported versions: {self.model_config.supported_versions}")
                
            if not (self.model_config.min_epochs <= epochs <= self.model_config.max_epochs):
                raise ValidationError(f"Epochs ({epochs}) out of range [{self.model_config.min_epochs}, {self.model_config.max_epochs}]")
                
            if not (self.model_config.min_batch_size <= batch_size <= self.model_config.max_batch_size):
                raise ValidationError(f"Batch size ({batch_size}) out of range [{self.model_config.min_batch_size}, {self.model_config.max_batch_size}]")

            logger.info(f"Starting training and export pipeline for YOLO{model_version}")
            logger.info(f"Configuration: epochs={epochs}, batch_size={batch_size}, img_size={self.model_config.default_img_size}")
            
            data_yaml_path = self.dataset_manager.IS_DATA_YAML
            
            # Validate data.yaml exists
            if not os.path.exists(data_yaml_path):
                raise ModelError(f"data.yaml not found at {data_yaml_path}. Please prepare datasets first.")
            
            results, run_dir = self.model_processor.train_yolo_model(
                model_version=model_version,
                data_yaml_path=data_yaml_path,
                epochs=epochs,
                batch_size=batch_size
            )

            if run_dir:
                logger.info(f"Training completed successfully for YOLO{model_version}")
                self.model_processor.zip_weights_folder(model_version=model_version)
                logger.info(f"Weights folder compressed for YOLO{model_version}")
            else:
                logger.error(f"Training failed for YOLO{model_version}. Skipping further steps.")
                run_dir = None
            
            return run_dir
            
        except Exception as e:
            if isinstance(e, (ValidationError, ModelError)):
                raise
            error_msg = f"Unexpected error during training and export: {str(e)}"
            logger.error(error_msg)
            raise ModelError(error_msg) from e

    def analyze_training_run(self, run_dir: Optional[str], model_version: str) -> bool:
        """
        Menganalisis dan memvisualisasikan hasil dari sesi pelatihan tertentu.
        
        Args:
            run_dir: Path to training run directory
            model_version: Model version for analysis
            
        Returns:
            bool: True if successful, False otherwise
            
        Raises:
            ValidationError: If parameters are invalid
        """
        try:
            if not model_version or not isinstance(model_version, str):
                raise ValidationError("model_version must be a non-empty string")
                
            if run_dir:
                logger.info(f"Analyzing training run for YOLO{model_version}")
                
                # Plot training metrics
                try:
                    self.metrics_analyzer.plot_training_metrics(run_dir, model_version)
                    logger.info(f"Training metrics plotted for YOLO{model_version}")
                except Exception as e:
                    logger.error(f"Failed to plot training metrics: {str(e)}")
                
                # Display final metrics table
                try:
                    self.metrics_analyzer.display_final_metrics_table(run_dir, model_version)
                    logger.info(f"Final metrics table displayed for YOLO{model_version}")
                except Exception as e:
                    logger.error(f"Failed to display final metrics table: {str(e)}")
                    
                return True
            else:
                logger.warning("No training run directory provided for analysis.")
                return False
                
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            error_msg = f"Unexpected error during training analysis: {str(e)}"
            logger.error(error_msg)
            return False

    def run_inference_and_visualization(self, run_dir: Optional[str], model_version: str, 
                                      num_inference_images: int = 6) -> bool:
        """
        Menjalankan inferensi pada gambar sampel dan memvisualisasikan hasilnya.
        
        Args:
            run_dir: Path to training run directory
            model_version: Model version for inference
            num_inference_images: Number of images to process
            
        Returns:
            bool: True if successful, False otherwise
            
        Raises:
            ValidationError: If parameters are invalid
            InferenceError: If inference fails
        """
        try:
            # Validate parameters
            if not model_version or not isinstance(model_version, str):
                raise ValidationError("model_version must be a non-empty string")
                
            if num_inference_images <= 0 or not isinstance(num_inference_images, int):
                raise ValidationError("num_inference_images must be a positive integer")

            if run_dir:
                logger.info(f"Running inference and visualization for YOLO{model_version}")
                logger.info(f"Using confidence threshold: {self.model_config.default_conf_threshold}")
                
                data_yaml_path = self.dataset_manager.IS_DATA_YAML
                model_path_for_inference = os.path.join(run_dir, "weights", "best.pt")
                
                # Validate model file exists
                if not os.path.exists(model_path_for_inference):
                    raise InferenceError(f"Model file not found at {model_path_for_inference}")

                try:
                    inference_results = self.inference_visualizer.run_inference_on_sample_images(
                        model_path=model_path_for_inference,
                        data_yaml_path=data_yaml_path,
                        num_images=num_inference_images,
                        conf_threshold=self.model_config.default_conf_threshold
                    )
                    
                    if inference_results:
                        # Use specialized YOLOv8 visualizer for structured academic output
                        if model_version.startswith('v8'):
                            logger.info("Using YOLOv8 specialized visualizer for academic paper output")
                            pipeline_results = self.yolov8_visualizer.run_complete_visualization_pipeline(
                                model_path=model_path_for_inference,
                                data_yaml_path=data_yaml_path,
                                num_images=num_inference_images,
                                conf_threshold=self.model_config.default_conf_threshold,
                                model_version=model_version
                            )
                            
                            if pipeline_results["status"] == "completed":
                                total_files = sum(len(files) for files in pipeline_results["generated_files"].values())
                                logger.info(f"YOLOv8 visualization pipeline completed: {total_files} files generated")
                                return True
                            else:
                                logger.warning(f"YOLOv8 visualization pipeline failed: {pipeline_results.get('error', 'Unknown error')}")
                                # Fallback to standard visualizer
                        
                        # Fallback or standard visualization for other YOLO versions
                        logger.info("Using standard visualization pipeline")
                        self.inference_visualizer.visualize_inference_results_grid(
                            inference_results, 
                            title=f"YOLO{model_version} Instance Segmentation Inference",
                            save_only=True  # Save only for better compatibility
                        )
                        
                        # Create publication-ready figures for academic papers
                        pub_files = self.inference_visualizer.create_publication_ready_figures(
                            inference_results, 
                            model_version,
                            title=f"YOLO{model_version} Instance Segmentation"
                        )
                        
                        # Save additional outputs
                        self.inference_visualizer.save_superimposed_images(inference_results, model_version)
                        self.inference_visualizer.save_inference_results_csv(inference_results, model_version)
                        self.inference_visualizer.zip_superimposed_images_folder(model_version)
                        
                        logger.info(f"Inference and visualization completed for YOLO{model_version}")
                        logger.info(f"Publication figures created: {len(pub_files)} files")
                        return True
                    else:
                        logger.warning(f"No inference results generated for YOLO{model_version}")
                        return False
                        
                except Exception as e:
                    raise InferenceError(f"Inference failed for YOLO{model_version}: {str(e)}") from e
            else:
                logger.warning("No training run directory provided for inference.")
                return False
                
        except Exception as e:
            if isinstance(e, (ValidationError, InferenceError)):
                raise
            error_msg = f"Unexpected error during inference and visualization: {str(e)}"
            logger.error(error_msg)
            raise InferenceError(error_msg) from e

    def convert_and_zip_rknn_models(self, model_version: str) -> bool:
        """
        Mengkonversi model ONNX ke RKNN dan mengkompres folder RKNN.
        
        Args:
            model_version: Model version to convert
            
        Returns:
            bool: True if successful, False otherwise
            
        Raises:
            ValidationError: If parameters are invalid
        """
        try:
            if not model_version or not isinstance(model_version, str):
                raise ValidationError("model_version must be a non-empty string")

            logger.info(f"Converting and zipping RKNN models for YOLO{model_version}")
            
            onnx_model_path = ""
            if model_version == "v8n":
                onnx_model_path = self.model_processor.YOLOV8N_IS_ONNX_PATH
            elif model_version == "v10n":
                onnx_model_path = self.model_processor.YOLOV10N_IS_ONNX_PATH
            elif model_version == "v11n":
                onnx_model_path = self.model_processor.YOLOV11N_IS_ONNX_PATH
            else:
                raise ValidationError(f"Unknown model version '{model_version}' for RKNN conversion")
            
            # Validate ONNX model exists
            if not os.path.exists(onnx_model_path):
                logger.warning(f"ONNX model not found at {onnx_model_path}. Skipping RKNN conversion.")
                return False
            
            self.rknn_converter.convert_onnx_to_rknn(
                onnx_model_path=onnx_model_path,
                model_version=model_version
            )
            self.rknn_converter.zip_rknn_models_folder()
            
            logger.info(f"RKNN conversion and zipping completed for YOLO{model_version}")
            return True
            
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            error_msg = f"Unexpected error during RKNN conversion: {str(e)}"
            logger.error(error_msg)
            return False

    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status including configuration.
        
        Returns:
            Dict[str, Any]: System status information
        """
        try:
            status = {
                "configuration": self.config_manager.get_config_summary(),
                "dataset_ready": os.path.exists(self.dataset_manager.IS_DATA_YAML),
                "model_dir_exists": os.path.exists(self.dataset_config.default_model_dir),
                "onnx_dir_exists": os.path.exists(self.dataset_config.default_onnx_dir),
                "rknn_dir_exists": os.path.exists(self.dataset_config.default_rknn_dir),
                "secrets_file_exists": os.path.exists('secrets.yaml'),
                "components_initialized": True
            }
            
            # Check for existing models
            status["existing_models"] = {}
            for version in self.model_config.supported_versions:
                model_paths = self.model_processor.get_model_paths(version)
                if "error" not in model_paths:
                    status["existing_models"][version] = {
                        "pytorch_exists": os.path.exists(model_paths["pytorch_model"]),
                        "onnx_exists": os.path.exists(model_paths["onnx_model"])
                    }
            
            logger.info("System status retrieved successfully")
            return status
            
        except Exception as e:
            logger.error(f"Failed to get system status: {str(e)}")
            return {"error": str(e)}

    def update_configuration(self, section: str, key: str, value: Any) -> bool:
        """
        Update system configuration.
        
        Args:
            section: Configuration section (model, dataset, fuzzy, logging, system)
            key: Parameter key
            value: New value
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            success = self.config_manager.update_config(section, key, value)
            if success:
                logger.info(f"Configuration updated: {section}.{key} = {value}")
                # Re-initialize components if needed
                if section in ['model', 'dataset']:
                    self._initialize_components()
            return success
        except Exception as e:
            logger.error(f"Failed to update configuration: {str(e)}")
            return False

    def get_configuration_summary(self) -> Dict[str, Any]:
        """
        Get configuration summary for display.
        
        Returns:
            Dict[str, Any]: Configuration summary
        """
        return self.config_manager.get_config_summary()
    
    def get_cuda_status(self) -> Dict[str, Any]:
        """
        Get CUDA/GPU status and recommendations.
        
        Returns:
            Dict containing CUDA status and optimization recommendations
        """
        try:
            memory_info = self.cuda_manager.get_memory_info()
            optimization = self.cuda_manager.optimize_for_training(
                batch_size=self.model_config.default_batch_size,
                img_size=self.model_config.default_img_size
            )
            
            return {
                'memory_info': memory_info,
                'optimization_recommendations': optimization,
                'optimal_batch_size': self.cuda_manager.get_optimal_batch_size(
                    self.model_config.default_img_size
                )
            }
        except Exception as e:
            logger.error(f"Failed to get CUDA status: {e}")
            return {'error': str(e)}
    
    def clear_cuda_cache(self) -> bool:
        """
        Clear CUDA cache to free GPU memory.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.cuda_manager.clear_cache()
            return True
        except Exception as e:
            logger.error(f"Failed to clear CUDA cache: {e}")
            return False

    def find_existing_training_run(self, model_version: str) -> Optional[str]:
        """
        Find existing training run directory for a given model version.
        
        Args:
            model_version: Model version to find (e.g., "v8n", "v10n", "v11n")
            
        Returns:
            Optional[str]: Path to existing training run directory or None if not found
        """
        try:
            train_name = f"segment_train_{model_version}"
            model_dir = self.model_processor.MODEL_DIR
            
            # Find the most recent training directory
            existing_runs = []
            for item in os.listdir(model_dir):
                if item.startswith(train_name) and os.path.isdir(os.path.join(model_dir, item)):
                    run_path = os.path.join(model_dir, item)
                    # Check if it has weights
                    weights_path = os.path.join(run_path, "weights", "best.pt")
                    if os.path.exists(weights_path):
                        existing_runs.append(run_path)
            
            if existing_runs:
                # Sort by modification time (most recent first)
                existing_runs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                latest_run = existing_runs[0]
                logger.info(f"Found existing training run for YOLO{model_version}: {latest_run}")
                return latest_run
            else:
                logger.info(f"No existing training run found for YOLO{model_version}")
                return None
                
        except Exception as e:
            logger.error(f"Error finding existing training run: {str(e)}")
            return None

    def get_or_train_model(self, model_version: str, force_retrain: bool = False, 
                          epochs: Optional[int] = None, batch_size: Optional[int] = None) -> Optional[str]:
        """
        Get existing training run or train new model based on preference.
        
        Args:
            model_version: Model version to train/find
            force_retrain: If True, always train new model
            epochs: Number of training epochs (if training)
            batch_size: Training batch size (if training)
            
        Returns:
            Optional[str]: Path to training run directory
        """
        try:
            if not force_retrain:
                # Try to find existing training run
                existing_run = self.find_existing_training_run(model_version)
                if existing_run:
                    logger.info(f"Using existing training run for YOLO{model_version}")
                    return existing_run
            
            # Train new model if no existing run found or force_retrain is True
            logger.info(f"Training new model for YOLO{model_version}")
            return self.train_and_export_model(model_version, epochs, batch_size)
            
        except Exception as e:
            logger.error(f"Error in get_or_train_model: {str(e)}")
            return None

    def execute_yolo_pipeline(self, model_version: str, force_retrain: bool = False, 
                             epochs: Optional[int] = None, batch_size: Optional[int] = None,
                             num_inference_images: int = 6) -> bool:
        """
        Execute complete YOLO pipeline for a specific model version.
        
        Args:
            model_version: Model version to train/use (e.g., "v8n", "v10n", "v11n")
            force_retrain: If True, always train new model
            epochs: Number of training epochs (if training)
            batch_size: Training batch size (if training)
            num_inference_images: Number of images for inference
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"🚀 Starting YOLO{model_version} pipeline")
            
            # Get or train model
            run_dir = self.get_or_train_model(model_version, force_retrain, epochs, batch_size)
            
            if not run_dir:
                logger.error(f"❌ Failed to get or train model for YOLO{model_version}")
                return False
            
            logger.info(f"✅ Using training run: {run_dir}")
            
            # Execute pipeline steps
            logger.info(f"📊 Analyzing training run for YOLO{model_version}")
            self.analyze_training_run(run_dir, model_version)
            
            logger.info(f"🔍 Running inference and visualization for YOLO{model_version}")
            self.run_inference_and_visualization(run_dir, model_version, num_inference_images)
            
            logger.info(f"📦 Converting and zipping RKNN models for YOLO{model_version}")
            self.convert_and_zip_rknn_models(model_version)
            
            logger.info(f"🎉 YOLO{model_version} pipeline completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ YOLO{model_version} pipeline failed: {str(e)}")
            return False

    def execute_yolo_pipeline_safe(self, model_version: str, force_retrain: bool = False, 
                                  epochs: Optional[int] = None, batch_size: Optional[int] = None) -> bool:
        """
        Execute safe YOLO pipeline for a specific model version (skips problematic inference).
        
        Args:
            model_version: Model version to train/use (e.g., "v8n", "v10n", "v11n")
            force_retrain: If True, always train new model
            epochs: Number of training epochs (if training)
            batch_size: Training batch size (if training)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"🚀 Starting safe YOLO{model_version} pipeline")
            
            # Get or train model
            run_dir = self.get_or_train_model(model_version, force_retrain, epochs, batch_size)
            
            if not run_dir:
                logger.error(f"❌ Failed to get or train model for YOLO{model_version}")
                return False
            
            logger.info(f"✅ Using training run: {run_dir}")
            
            # Execute safe pipeline steps
            logger.info(f"📊 Analyzing training run for YOLO{model_version}")
            self.analyze_training_run(run_dir, model_version)
            
            # Skip inference visualization (has RGBA issue)
            logger.info(f"⏭️ Skipping run_inference_and_visualization (RGBA issue)")
            
            logger.info(f"📦 Converting and zipping RKNN models for YOLO{model_version}")
            self.convert_and_zip_rknn_models(model_version)
            
            logger.info(f"🎉 Safe YOLO{model_version} pipeline completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Safe YOLO{model_version} pipeline failed: {str(e)}")
            return False

    def export_onnx_from_existing_model(self, model_version: str) -> bool:
        """
        Export ONNX model from existing training results.
        
        Args:
            model_version: Model version to export (e.g., "v8n", "v10n", "v11n")
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"Exporting ONNX model for YOLO{model_version} from existing training results")
            
            # Find existing training run
            existing_run = self.find_existing_training_run(model_version)
            if not existing_run:
                logger.error(f"No existing training run found for YOLO{model_version}")
                return False
            
            # Path to the trained model
            pytorch_model_path = os.path.join(existing_run, "weights", "best.pt")
            
            if not os.path.exists(pytorch_model_path):
                logger.error(f"Trained model not found at {pytorch_model_path}")
                return False
            
            # ONNX output path
            onnx_output_path = ""
            if model_version == "v8n":
                onnx_output_path = self.model_processor.YOLOV8N_IS_ONNX_PATH
            elif model_version == "v10n":
                onnx_output_path = self.model_processor.YOLOV10N_IS_ONNX_PATH
            elif model_version == "v11n":
                onnx_output_path = self.model_processor.YOLOV11N_IS_ONNX_PATH
            else:
                logger.error(f"Unknown model version '{model_version}' for ONNX export")
                return False
            
            logger.info(f"Exporting PyTorch model ({pytorch_model_path}) to ONNX at {onnx_output_path}")
            
            # Load the trained model and export to ONNX
            from ultralytics import YOLO
            
            try:
                # Load the trained model
                model = YOLO(pytorch_model_path)
                
                # Export to ONNX
                model.export(
                    format="onnx", 
                    imgsz=self.model_config.default_img_size, 
                    opset=12, 
                    simplify=True
                )
                
                # Move the exported file to the correct location
                exported_files = [f for f in os.listdir(os.path.dirname(pytorch_model_path)) if f.endswith('.onnx')]
                if exported_files:
                    source_path = os.path.join(os.path.dirname(pytorch_model_path), exported_files[0])
                    shutil.move(source_path, onnx_output_path)
                
                if os.path.exists(onnx_output_path):
                    logger.info(f"✅ ONNX model successfully exported to: {onnx_output_path}")
                    return True
                else:
                    logger.error(f"❌ Failed to export ONNX model to {onnx_output_path}")
                    return False
                    
            except Exception as e:
                logger.error(f"Failed to export model to ONNX: {str(e)}")
                return False
                
        except Exception as e:
            logger.error(f"Unexpected error during ONNX export: {str(e)}")
            return False

    def execute_yolo_pipeline_complete(self, model_version: str, force_retrain: bool = False, 
                                      epochs: Optional[int] = None, batch_size: Optional[int] = None,
                                      num_inference_images: int = 6) -> bool:
        """
        Execute complete YOLO pipeline with ONNX export from existing models.
        
        Args:
            model_version: Model version to train/use (e.g., "v8n", "v10n", "v11n")
            force_retrain: If True, always train new model
            epochs: Number of training epochs (if training)
            batch_size: Training batch size (if training)
            num_inference_images: Number of images for inference
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"🚀 Starting complete YOLO{model_version} pipeline")
            
            # Get or train model
            run_dir = self.get_or_train_model(model_version, force_retrain, epochs, batch_size)
            
            if not run_dir:
                logger.error(f"❌ Failed to get or train model for YOLO{model_version}")
                return False
            
            logger.info(f"✅ Using training run: {run_dir}")
            
            # Export ONNX from existing model if not training new
            if not force_retrain:
                logger.info(f"📦 Exporting ONNX model for YOLO{model_version}")
                onnx_success = self.export_onnx_from_existing_model(model_version)
                if not onnx_success:
                    logger.warning(f"⚠️ ONNX export failed for YOLO{model_version}, continuing with pipeline")
            
            # Execute pipeline steps
            logger.info(f"📊 Analyzing training run for YOLO{model_version}")
            self.analyze_training_run(run_dir, model_version)
            
            logger.info(f"🔍 Running inference and visualization for YOLO{model_version}")
            self.run_inference_and_visualization(run_dir, model_version, num_inference_images)
            
            logger.info(f"📦 Converting and zipping RKNN models for YOLO{model_version}")
            self.convert_and_zip_rknn_models(model_version)
            
            logger.info(f"🎉 Complete YOLO{model_version} pipeline completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Complete YOLO{model_version} pipeline failed: {str(e)}")
            return False

    def execute_yolo_pipeline_with_onnx(self, model_version: str, force_retrain: bool = False, 
                                       epochs: Optional[int] = None, batch_size: Optional[int] = None) -> bool:
        """
        Execute YOLO pipeline with ONNX export but skip problematic inference.
        
        Args:
            model_version: Model version to train/use (e.g., "v8n", "v10n", "v11n")
            force_retrain: If True, always train new model
            epochs: Number of training epochs (if training)
            batch_size: Training batch size (if training)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"🚀 Starting YOLO{model_version} pipeline with ONNX export")
            
            # Get or train model
            run_dir = self.get_or_train_model(model_version, force_retrain, epochs, batch_size)
            
            if not run_dir:
                logger.error(f"❌ Failed to get or train model for YOLO{model_version}")
                return False
            
            logger.info(f"✅ Using training run: {run_dir}")
            
            # Export ONNX from existing model if not training new
            if not force_retrain:
                logger.info(f"📦 Exporting ONNX model for YOLO{model_version}")
                onnx_success = self.export_onnx_from_existing_model(model_version)
                if not onnx_success:
                    logger.warning(f"⚠️ ONNX export failed for YOLO{model_version}, continuing with pipeline")
            
            # Execute pipeline steps
            logger.info(f"📊 Analyzing training run for YOLO{model_version}")
            self.analyze_training_run(run_dir, model_version)
            
            # Skip inference visualization (has RGBA issue)
            logger.info(f"⏭️ Skipping run_inference_and_visualization (RGBA issue)")
            
            logger.info(f"📦 Converting and zipping RKNN models for YOLO{model_version}")
            self.convert_and_zip_rknn_models(model_version)
            
            logger.info(f"🎉 YOLO{model_version} pipeline with ONNX export completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ YOLO{model_version} pipeline with ONNX export failed: {str(e)}")
            return False


def initialize_system() -> WasteDetectionSystemColab:
    """
    Initialize the Waste Detection System with proper logging and status display.
    
    Returns:
        WasteDetectionSystemColab: Initialized system instance
    """
    logger.info("Starting Waste Detection System with Configuration Management")
    
    # Initialize system with configuration
    system_colab = WasteDetectionSystemColab()
    
    # Display system status and configuration
    status = system_colab.get_system_status()
    config_summary = system_colab.get_configuration_summary()
    
    logger.info(f"System Status: {status}")
    logger.info(f"Configuration Summary: {config_summary}")
    
    return system_colab


def prepare_datasets(system_colab: WasteDetectionSystemColab) -> bool:
    """
    Prepare datasets with Ultralytics normalization.
    
    Args:
        system_colab: Initialized system instance
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info("🔧 Preparing datasets with Ultralytics normalization...")
        
        # Prepare datasets with integrated Ultralytics normalization
        success = system_colab.dataset_manager.prepare_datasets()
        if success:
            logger.info("✅ Dataset preparation completed with Ultralytics normalization")
            
            # Validate the prepared dataset
            validation_results = system_colab.dataset_manager.validate_dataset_format()
            if not validation_results['issues']:
                logger.info("✅ Dataset validation passed - ready for training!")
                
                # Zip datasets for backup
                system_colab.dataset_manager.zip_datasets_folder()
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


def fix_existing_dataset(system_colab: WasteDetectionSystemColab) -> bool:
    """
    Fix existing dataset if preparation failed.
    
    Args:
        system_colab: Initialized system instance
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info("🔧 Attempting to fix existing dataset...")
        
        # First, fix path issues in data.yaml
        path_fix_success = system_colab.dataset_manager.fix_data_yaml_paths()
        if path_fix_success:
            logger.info("✅ Data.yaml path fixing completed successfully")
        else:
            logger.warning("⚠️ Data.yaml path fixing failed or not needed")
        
        # Then fix dataset classes and segmentation labels
        fix_success = system_colab.dataset_manager.fix_dataset_classes()
        if fix_success:
            logger.info("✅ Dataset fixing completed successfully")
            
            # Additional: Fix segmentation labels specifically
            seg_fix_success = system_colab.dataset_manager.fix_segmentation_labels()
            if seg_fix_success:
                logger.info("✅ Segmentation labels fixed successfully")
            else:
                logger.warning("⚠️ Segmentation label fixing failed or not needed")
            
            return True
        else:
            logger.error("❌ Dataset fixing failed")
            return False
            
    except Exception as e:
        logger.error(f"Dataset fixing failed: {str(e)}")
        return False


def execute_yolo_pipelines(system_colab: WasteDetectionSystemColab) -> None:
    """
    Execute YOLO pipelines for all model versions.
    
    Args:
        system_colab: Initialized system instance
    """
    # YOLOv8n Pipeline
    try:
        logger.info("🚀 Executing YOLOv8n pipeline...")
        system_colab.execute_yolo_pipeline_safe("v8n", force_retrain=False, epochs=200, batch_size=16)
    except Exception as e:
        logger.error(f"YOLOv8n pipeline failed: {str(e)}")
    
    # YOLOv10n Pipeline (commented out)
    # try:
    #     logger.info("🚀 Executing YOLOv10n pipeline...")
    #     system_colab.execute_yolo_pipeline_safe("v10n", force_retrain=False, epochs=200, batch_size=16)
    # except Exception as e:
    #     logger.error(f"YOLOv10n pipeline failed: {str(e)}")
    
    # YOLOv11n Pipeline (commented out)
    # try:
    #     logger.info("🚀 Executing YOLOv11n pipeline...")
    #     system_colab.execute_yolo_pipeline_safe("v11n", force_retrain=False, epochs=200, batch_size=16)
    # except Exception as e:
    #     logger.error(f"YOLOv11n pipeline failed: {str(e)}")


def save_results_to_drive(system_colab: WasteDetectionSystemColab) -> None:
    """
    Save all results to Google Drive.
    
    Args:
        system_colab: Initialized system instance
    """
    try:
        logger.info("💾 Saving results to Google Drive...")
        system_colab.drive_manager.save_all_results_to_drive(folder_to_save="yolo_issat_results")
        logger.info("✅ Results saved to Google Drive successfully")
    except Exception as e:
        logger.error(f"Drive save failed: {str(e)}")


def display_completion_message() -> None:
    """Display completion message to user."""
    logger.info("Waste Detection System completed successfully")
    print("\n🎉 Proses di Google Colab selesai!")
    print("📁 Model .pt, .onnx, .rknn, dan file .zip yang dihasilkan telah disimpan")
    print("📥 File dapat diunduh secara manual dari output directories")


def setup_colab_environment() -> None:
    """
    Setup instructions for Google Colab environment.
    This function provides guidance for Colab-specific setup.
    """
    print("\n📋 Google Colab Setup Instructions:")
    print("=" * 50)
    print("1. Mount Google Drive:")
    print("   from google.colab import drive")
    print("   drive.mount('/content/gdrive')")
    print("\n2. Enable TensorBoard:")
    print("   %load_ext tensorboard")
    print("   %tensorboard --logdir runs")
    print("\n3. Install dependencies if needed:")
    print("   !pip install -r requirements.txt")
    print("=" * 50)


def run_single_model_pipeline(system_colab: WasteDetectionSystemColab, 
                             model_version: str, force_retrain: bool = False,
                             epochs: int = 200, batch_size: int = 16) -> bool:
    """
    Run pipeline for a single model version.
    
    Args:
        system_colab: Initialized system instance
        model_version: Model version to run (e.g., "v8n", "v10n", "v11n")
        force_retrain: If True, always train new model
        epochs: Number of training epochs
        batch_size: Training batch size
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info(f"🚀 Executing YOLO{model_version} pipeline...")
        success = system_colab.execute_yolo_pipeline_safe(
            model_version, 
            force_retrain=force_retrain, 
            epochs=epochs, 
            batch_size=batch_size
        )
        
        if success:
            logger.info(f"✅ YOLO{model_version} pipeline completed successfully")
        else:
            logger.error(f"❌ YOLO{model_version} pipeline failed")
            
        return success
        
    except Exception as e:
        logger.error(f"YOLO{model_version} pipeline failed: {str(e)}")
        return False


def run_all_model_pipelines(system_colab: WasteDetectionSystemColab, 
                           models: list = None, force_retrain: bool = False,
                           epochs: int = 200, batch_size: int = 16) -> dict:
    """
    Run pipelines for multiple model versions.
    
    Args:
        system_colab: Initialized system instance
        models: List of model versions to run (default: ["v8n"])
        force_retrain: If True, always train new models
        epochs: Number of training epochs
        batch_size: Training batch size
        
    Returns:
        dict: Results for each model version
    """
    if models is None:
        models = ["v8n"]
    
    results = {}
    
    for model_version in models:
        logger.info(f"🔄 Processing YOLO{model_version}...")
        success = run_single_model_pipeline(
            system_colab, 
            model_version, 
            force_retrain, 
            epochs, 
            batch_size
        )
        results[model_version] = success
    
    return results


def print_pipeline_summary(results: dict) -> None:
    """
    Print a summary of pipeline execution results.
    
    Args:
        results: Dictionary of results from run_all_model_pipelines
    """
    print("\n📊 Pipeline Execution Summary:")
    print("=" * 40)
    
    for model_version, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"YOLO{model_version}: {status}")
    
    successful = sum(results.values())
    total = len(results)
    print(f"\nOverall: {successful}/{total} pipelines completed successfully")
    print("=" * 40)


def parse_arguments():
    """
    Parse command-line arguments for flexible execution.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Waste Detection System - YOLO Pipeline")
    
    # Model selection
    parser.add_argument("--models", nargs="+", choices=["v8n", "v10n", "v11n"], 
                       default=["v8n"], help="Model versions to run")
    
    # Training options
    parser.add_argument("--force-retrain", action="store_true", 
                       help="Force retraining even if existing models found")
    parser.add_argument("--epochs", type=int, default=200, 
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, 
                       help="Training batch size")
    
    # Dataset options
    parser.add_argument("--prepare-datasets", action="store_true", 
                       help="Prepare datasets before training")
    parser.add_argument("--fix-datasets", action="store_true", 
                       help="Fix existing datasets if preparation fails")
    
    # Drive options
    parser.add_argument("--save-to-drive", action="store_true", 
                       help="Save results to Google Drive")
    
    # Setup options
    parser.add_argument("--show-setup", action="store_true", 
                       help="Show Google Colab setup instructions")
    
    # Pipeline options
    parser.add_argument("--complete-pipeline", action="store_true", 
                       help="Use complete pipeline (includes inference and ONNX export)")
    parser.add_argument("--onnx-export", action="store_true", 
                       help="Use pipeline with ONNX export (skips inference)")
    
    return parser.parse_args()


def main():
    """
    Main execution function with comprehensive error handling and configuration management.
    """
    try:
        # Parse command-line arguments
        args = parse_arguments()
        
        # Show setup instructions if requested
        if args.show_setup:
            setup_colab_environment()
            return
        
        # Initialize system
        system_colab = initialize_system()
        
        # Optional: Dataset preparation
        if args.prepare_datasets:
            logger.info("🔧 Preparing datasets...")
            prepare_datasets(system_colab)
        
        if args.fix_datasets:
            logger.info("🔧 Fixing existing datasets...")
            fix_existing_dataset(system_colab)
        
        # Execute YOLO pipelines with better organization
        logger.info("🎯 Starting YOLO pipeline execution...")
        
        if args.complete_pipeline:
            # Use complete pipeline with inference and ONNX export
            logger.info("🔄 Using complete pipeline (includes inference and ONNX export)")
            results = {}
            for model_version in args.models:
                success = system_colab.execute_yolo_pipeline_complete(
                    model_version,
                    force_retrain=args.force_retrain,
                    epochs=args.epochs,
                    batch_size=args.batch_size
                )
                results[model_version] = success
        elif args.onnx_export:
            # Use pipeline with ONNX export but skip inference
            logger.info("🔄 Using pipeline with ONNX export (skips inference)")
            results = {}
            for model_version in args.models:
                success = system_colab.execute_yolo_pipeline_with_onnx(
                    model_version,
                    force_retrain=args.force_retrain,
                    epochs=args.epochs,
                    batch_size=args.batch_size
                )
                results[model_version] = success
        else:
            # Use safe pipeline (skips inference)
            results = run_all_model_pipelines(
                system_colab, 
                models=args.models,
                force_retrain=args.force_retrain,
                epochs=args.epochs,
                batch_size=args.batch_size
            )
        
        # Print summary
        print_pipeline_summary(results)
        
        # Optional: Save to Google Drive
        if args.save_to_drive:
            save_results_to_drive(system_colab)
        
        # Display completion message
        display_completion_message()
        
    except Exception as e:
        error_msg = f"Critical error in main execution: {str(e)}"
        logger.error(error_msg)
        print(f"Error: {error_msg}")
        sys.exit(1)

if __name__ == "__main__":
    main()
