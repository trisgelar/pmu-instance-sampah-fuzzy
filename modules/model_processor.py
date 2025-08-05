# file: modules/model_processor.py
import os
import shutil
import logging
from typing import Optional, Tuple, Any
from ultralytics import YOLO

from modules.exceptions import ModelError, FileOperationError, ValidationError

# Configure logging
logger = logging.getLogger(__name__)

class ModelProcessor:
    """
    Mengelola pelatihan model YOLO, dan ekspor ke ONNX.
    """
    def __init__(self, model_dir: str, onnx_model_dir: str, img_size: tuple[int, int]):
        """
        Initialize ModelProcessor with validation.
        
        Args:
            model_dir: Directory to store model outputs
            onnx_model_dir: Directory to store ONNX models
            img_size: Image size tuple (width, height)
            
        Raises:
            ConfigurationError: If required parameters are invalid
        """
        try:
            self._validate_parameters(model_dir, onnx_model_dir, img_size)
            self.MODEL_DIR = model_dir
            self.ONNX_MODEL_DIR = onnx_model_dir
            self.img_size = img_size

            os.makedirs(self.ONNX_MODEL_DIR, exist_ok=True)
            logger.info(f"ONNX model directory created/verified: {self.ONNX_MODEL_DIR}")
            
            # ONNX paths for different YOLO versions
            self.YOLOV8N_IS_ONNX_PATH = os.path.join(self.ONNX_MODEL_DIR, "yolov8n_is.onnx")
            self.YOLOV10N_IS_ONNX_PATH = os.path.join(self.ONNX_MODEL_DIR, "yolov10n_is.onnx")
            self.YOLOV11N_IS_ONNX_PATH = os.path.join(self.ONNX_MODEL_DIR, "yolov11n_is.onnx")
            
            logger.info("ModelProcessor initialized successfully")
            
        except Exception as e:
            raise ValidationError(f"Failed to initialize ModelProcessor: {str(e)}") from e

    def _validate_parameters(self, model_dir: str, onnx_model_dir: str, img_size: tuple[int, int]) -> None:
        """
        Validate initialization parameters.
        
        Args:
            model_dir: Model directory path
            onnx_model_dir: ONNX model directory path
            img_size: Image size tuple
            
        Raises:
            ValidationError: If parameters are invalid
        """
        if not model_dir or not isinstance(model_dir, str):
            raise ValidationError("model_dir must be a non-empty string")
        
        if not onnx_model_dir or not isinstance(onnx_model_dir, str):
            raise ValidationError("onnx_model_dir must be a non-empty string")
            
        if not isinstance(img_size, tuple) or len(img_size) != 2:
            raise ValidationError("img_size must be a tuple of two integers")
            
        if not all(isinstance(x, int) and x > 0 for x in img_size):
            raise ValidationError("img_size values must be positive integers")

    def _validate_model_version(self, model_version: str) -> None:
        """
        Validate YOLO model version.
        
        Args:
            model_version: Model version to validate
            
        Raises:
            ValidationError: If model version is invalid
        """
        valid_versions = ["v8n", "v10n", "v11n"]
        if model_version not in valid_versions:
            raise ValidationError(f"Invalid model version '{model_version}'. Valid versions: {valid_versions}")

    def _get_onnx_path(self, model_version: str) -> str:
        """
        Get ONNX output path for model version.
        
        Args:
            model_version: Model version
            
        Returns:
            str: ONNX output path
        """
        if model_version == "v8n":
            return self.YOLOV8N_IS_ONNX_PATH
        elif model_version == "v10n":
            return self.YOLOV10N_IS_ONNX_PATH
        elif model_version == "v11n":
            return self.YOLOV11N_IS_ONNX_PATH
        else:
            raise ValidationError(f"Unknown model version '{model_version}' for ONNX export")

    def train_yolo_model(self, model_version: str = "v8n", data_yaml_path: str = "", 
                        epochs: int = 50, batch_size: int = 16) -> Tuple[Optional[Any], Optional[str]]:
        """
        Melatih model YOLO (v8n, v10n, v11n) untuk Instance Segmentation.
        Setelah pelatihan, model akan otomatis diekspor ke ONNX.
        
        Args:
            model_version: YOLO model version to train
            data_yaml_path: Path to data.yaml file
            epochs: Number of training epochs
            batch_size: Training batch size
            
        Returns:
            Tuple[Optional[Any], Optional[str]]: (training_results, run_directory)
            
        Raises:
            ValidationError: If parameters are invalid
            ModelError: If training or export fails
        """
        try:
            # Validate parameters
            self._validate_model_version(model_version)
            
            if not data_yaml_path or not isinstance(data_yaml_path, str):
                raise ValidationError("data_yaml_path must be a non-empty string")
                
            if not os.path.exists(data_yaml_path):
                raise ModelError(f"data.yaml not found at {data_yaml_path}. Cannot start training.")
                
            if epochs <= 0 or not isinstance(epochs, int):
                raise ValidationError("epochs must be a positive integer")
                
            if batch_size <= 0 or not isinstance(batch_size, int):
                raise ValidationError("batch_size must be a positive integer")

            logger.info(f"Starting YOLO{model_version} (segment) training")
            
            # Load model
            try:
                model = YOLO(f"yolo{model_version}-seg.pt")
                logger.info(f"Loaded YOLO{model_version} segmentation model")
            except Exception as e:
                raise ModelError(f"Failed to load YOLO{model_version} model: {str(e)}") from e

            # Train model
            train_name = f"segment_train_{model_version}"
            try:
                results = model.train(
                    data=data_yaml_path,
                    epochs=epochs,
                    imgsz=self.img_size[0],
                    batch=batch_size,
                    name=train_name
                )
                logger.info(f"YOLO{model_version} (segment) training completed successfully")
            except Exception as e:
                raise ModelError(f"Training failed for YOLO{model_version}: {str(e)}") from e

            # Export to ONNX
            pytorch_model_path = os.path.join(self.MODEL_DIR, "segment", train_name, "weights", "best.pt")
            
            if not os.path.exists(pytorch_model_path):
                raise ModelError(f"Trained model not found at {pytorch_model_path}")
            
            onnx_output_path = self._get_onnx_path(model_version)
            
            logger.info(f"Exporting PyTorch model ({pytorch_model_path}) to ONNX at {onnx_output_path}")
            
            try:
                model.export(format="onnx", imgsz=self.img_size, opset=12, simplify=True, filename=onnx_output_path)
                
                if os.path.exists(onnx_output_path):
                    logger.info(f"ONNX model successfully exported to: {onnx_output_path}")
                else:
                    raise ModelError(f"Failed to export ONNX model to {onnx_output_path}")
                    
            except Exception as e:
                raise ModelError(f"Failed to export model to ONNX: {str(e)}") from e

            train_run_dir = os.path.join(self.MODEL_DIR, "segment", train_name)
            return results, train_run_dir
            
        except Exception as e:
            if isinstance(e, (ValidationError, ModelError)):
                raise
            raise ModelError(f"Unexpected error during model training: {str(e)}") from e

    def zip_weights_folder(self, model_version: str = "v8n") -> bool:
        """
        Melakukan kompresi (zip) pada folder 'weights' yang dihasilkan dari pelatihan model.
        
        Args:
            model_version: Model version to compress weights for
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self._validate_model_version(model_version)
            
            train_name = f"segment_train_{model_version}"
            source_folder = os.path.join(self.MODEL_DIR, "segment", train_name, "weights")
            output_filename = f"segment_{model_version}_weights"
            
            if not os.path.exists(source_folder):
                logger.warning(f"Weights folder '{source_folder}' not found. Cannot compress.")
                return False
            
            logger.info(f"Compressing weights folder for segmentation {model_version}")
            
            try:
                shutil.make_archive(output_filename, 'zip', source_folder)
                logger.info(f"Folder '{source_folder}' successfully compressed to '{output_filename}.zip'")
                return True
            except Exception as e:
                error_msg = f"Failed to compress weights folder '{source_folder}': {str(e)}"
                logger.error(error_msg)
                raise FileOperationError(error_msg) from e
                
        except Exception as e:
            if isinstance(e, (ValidationError, FileOperationError)):
                raise
            logger.error(f"Unexpected error during weights compression: {str(e)}")
            return False

    def get_model_paths(self, model_version: str) -> dict:
        """
        Get model file paths for a specific version.
        
        Args:
            model_version: Model version
            
        Returns:
            dict: Dictionary containing model paths
        """
        try:
            self._validate_model_version(model_version)
            
            train_name = f"segment_train_{model_version}"
            return {
                "pytorch_model": os.path.join(self.MODEL_DIR, "segment", train_name, "weights", "best.pt"),
                "onnx_model": self._get_onnx_path(model_version),
                "weights_dir": os.path.join(self.MODEL_DIR, "segment", train_name, "weights"),
                "run_dir": os.path.join(self.MODEL_DIR, "segment", train_name)
            }
        except Exception as e:
            logger.error(f"Failed to get model paths: {str(e)}")
            return {"error": str(e)}
