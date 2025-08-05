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
        logging.FileHandler('waste_detection_system.log')
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
                        self.inference_visualizer.visualize_inference_results_grid(
                            inference_results, 
                            title=f"YOLO{model_version} Instance Segmentation Inference"
                        )
                        self.inference_visualizer.save_superimposed_images(inference_results, model_version)
                        self.inference_visualizer.save_inference_results_csv(inference_results, model_version)
                        self.inference_visualizer.zip_superimposed_images_folder(model_version)
                        
                        logger.info(f"Inference and visualization completed for YOLO{model_version}")
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


# --- Fungsi Utama (Main Execution Flow untuk Colab) ---
def main():
    """
    Main execution function with comprehensive error handling and configuration management.
    """
    try:
        logger.info("Starting Waste Detection System with Configuration Management")
        
        # Initialize system with configuration
        system_colab = WasteDetectionSystemColab()
        
        # Display system status and configuration
        status = system_colab.get_system_status()
        config_summary = system_colab.get_configuration_summary()
        
        logger.info(f"System Status: {status}")
        logger.info(f"Configuration Summary: {config_summary}")

        # --- PENTING: Langkah Tambahan - Mount Google Drive ---
        # Jalankan ini di sel terpisah di Colab untuk menghubungkan ke Drive
        # from google.colab import drive
        # drive.mount('/content/gdrive')

        # --- Untuk melihat log pelatihan di TensorBoard ---
        # %load_ext tensorboard
        # %tensorboard --logdir runs

        # Langkah 1: Persiapan Dataset (jalankan sekali)
        try:
            system_colab.dataset_manager.prepare_datasets()
            system_colab.dataset_manager.zip_datasets_folder()
        except Exception as e:
            logger.error(f"Dataset preparation failed: {str(e)}")

        # --- Eksekusi Pipeline untuk YOLOv8n Instance Segmentation ---
        try:
            run_dir_v8n = system_colab.train_and_export_model("v8n")
            system_colab.analyze_training_run(run_dir_v8n, "v8n")
            system_colab.run_inference_and_visualization(run_dir_v8n, "v8n", num_inference_images=6)
            # system_colab.convert_and_zip_rknn_models("v8n")
        except Exception as e:
            logger.error(f"YOLOv8n pipeline failed: {str(e)}")
        
        # --- Eksekusi Pipeline untuk YOLOv10n Instance Segmentation ---
        # try:
        #     run_dir_v10n = system_colab.train_and_export_model("v10n")
        #     system_colab.analyze_training_run(run_dir_v10n, "v10n")
        #     system_colab.run_inference_and_visualization(run_dir_v10n, "v10n", num_inference_images=6)
        #     #system_colab.convert_and_zip_rknn_models("v10n")
        # except Exception as e:
        #     logger.error(f"YOLOv10n pipeline failed: {str(e)}")

        # --- Eksekusi Pipeline untuk YOLOv11n Instance Segmentation ---
        # try:
        #     run_dir_v11n = system_colab.train_and_export_model("v11n")
        #     system_colab.analyze_training_run(run_dir_v11n, "v11n")
        #     system_colab.run_inference_and_visualization(run_dir_v11n, "v11n", num_inference_images=6)
        #     #system_colab.convert_and_zip_rknn_models("v11n")
        # except Exception as e:
        #     logger.error(f"YOLOv11n pipeline failed: {str(e)}")

        # Langkah Tambahan: Menyimpan semua hasil ke Google Drive
        # try:
        #     system_colab.drive_manager.save_all_results_to_drive(folder_to_save="yolo_issat_results")
        # except Exception as e:
        #     logger.error(f"Drive save failed: {str(e)}")
        
        logger.info("Waste Detection System completed successfully")
        print("\nProses di Google Colab selesai. Model .pt, .onnx, .rknn, dan file .zip yang dihasilkan telah disimpan dan juga dapat diunduh secara manual.")
        
    except Exception as e:
        error_msg = f"Critical error in main execution: {str(e)}"
        logger.error(error_msg)
        print(f"Error: {error_msg}")
        sys.exit(1)

if __name__ == "__main__":
    main()
