# file: modules/dataset_manager.py
import os
import shutil
import yaml
import logging
from typing import Optional, Dict, Any
from roboflow import Roboflow

from modules.exceptions import DatasetError, ConfigurationError, APIError, FileOperationError, ValidationError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetManager:
    """
    Mengelola semua operasi terkait dataset, termasuk pengunduhan dari Roboflow
    dan penyiapan struktur folder lokal untuk Instance Segmentation.
    """
    def __init__(self, dataset_dir: str, is_project: str, is_version: str):
        """
        Initialize DatasetManager with validation.
        
        Args:
            dataset_dir: Directory to store datasets
            is_project: Roboflow project name
            is_version: Roboflow project version
            
        Raises:
            ConfigurationError: If required parameters are invalid
        """
        try:
            self._validate_parameters(dataset_dir, is_project, is_version)
            self.DATASET_DIR = dataset_dir
            self.ROBOFLOW_IS_PROJECT = is_project
            self.ROBOFLOW_IS_VERSION = is_version
            self.IS_DATA_YAML = os.path.join(self.DATASET_DIR, self.ROBOFLOW_IS_PROJECT, "data.yaml")
            
            # Create dataset directory
            os.makedirs(self.DATASET_DIR, exist_ok=True)
            logger.info(f"Dataset directory created/verified: {self.DATASET_DIR}")
            
            # Load API key
            self.ROBOFLOW_API_KEY = self._load_api_key()
            logger.info("Roboflow API key loaded successfully")
            
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize DatasetManager: {str(e)}") from e

    def _validate_parameters(self, dataset_dir: str, is_project: str, is_version: str) -> None:
        """
        Validate initialization parameters.
        
        Args:
            dataset_dir: Directory path
            is_project: Project name
            is_version: Version string
            
        Raises:
            ValidationError: If parameters are invalid
        """
        if not dataset_dir or not isinstance(dataset_dir, str):
            raise ValidationError("dataset_dir must be a non-empty string")
        
        if not is_project or not isinstance(is_project, str):
            raise ValidationError("is_project must be a non-empty string")
            
        if not is_version or not isinstance(is_version, str):
            raise ValidationError("is_version must be a non-empty string")

    def _load_api_key(self) -> str:
        """
        Memuat kunci API Roboflow dari file secrets.yaml.
        
        Returns:
            str: Roboflow API key
            
        Raises:
            ConfigurationError: If secrets file is missing or invalid
        """
        secrets_file = 'secrets.yaml'
        
        try:
            if not os.path.exists(secrets_file):
                raise ConfigurationError(
                    f"File '{secrets_file}' tidak ditemukan. "
                    "Harap buat file ini dengan kunci API Roboflow Anda."
                )
            
            with open(secrets_file, 'r', encoding='utf-8') as f:
                secrets = yaml.safe_load(f)
                if not secrets:
                    raise ConfigurationError(f"File '{secrets_file}' is empty or invalid")
                
                api_key = secrets.get('roboflow_api_key')
                if not api_key:
                    raise ConfigurationError(
                        f"Kunci 'roboflow_api_key' tidak ditemukan di '{secrets_file}'."
                    )
                
                # Validate API key format (basic check)
                if len(api_key) < 10:
                    raise ConfigurationError("API key appears to be invalid (too short)")
                
                return api_key
                
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML format in '{secrets_file}': {str(e)}") from e
        except Exception as e:
            raise ConfigurationError(f"Failed to load API key: {str(e)}") from e

    def _download_from_roboflow(self, project_name: str, version: str, dataset_type: str) -> Optional[str]:
        """
        Helper function to download dataset from Roboflow.
        
        Args:
            project_name: Name of the Roboflow project
            version: Version of the dataset
            dataset_type: Type of dataset to download
            
        Returns:
            Optional[str]: Path to downloaded dataset or None if failed
            
        Raises:
            APIError: If Roboflow API call fails
        """
        logger.info(f"Downloading dataset '{project_name}' version {version}")
        
        try:
            rf = Roboflow(api_key=self.ROBOFLOW_API_KEY)
            project = rf.workspace().project(project_name)
            dataset = project.version(version).download(dataset_type)
            
            logger.info(f"Dataset '{project_name}' successfully downloaded to: {dataset.location}")
            return dataset.location
            
        except Exception as e:
            error_msg = f"Failed to download dataset from Roboflow: {str(e)}"
            logger.error(error_msg)
            logger.error("Please ensure your Roboflow API key is correct and project/version is available.")
            raise APIError(error_msg) from e

    def prepare_datasets(self) -> bool:
        """
        Mempersiapkan dataset untuk Instance Segmentation.
        
        Returns:
            bool: True if successful, False otherwise
            
        Raises:
            DatasetError: If dataset preparation fails
        """
        logger.info("Preparing Instance Segmentation Dataset")
        
        try:
            is_dataset_root_path = self._download_from_roboflow(
                self.ROBOFLOW_IS_PROJECT, 
                self.ROBOFLOW_IS_VERSION, 
                dataset_type="coco-segmentation"
            )
            
            if not is_dataset_root_path:
                raise DatasetError("Failed to download dataset from Roboflow")
            
            is_dataset_target_path = os.path.join(self.DATASET_DIR, self.ROBOFLOW_IS_PROJECT)
            os.makedirs(is_dataset_target_path, exist_ok=True)

            # Move dataset contents
            try:
                for item in os.listdir(is_dataset_root_path):
                    source_path = os.path.join(is_dataset_root_path, item)
                    target_path = os.path.join(is_dataset_target_path, item)
                    shutil.move(source_path, target_path)
                os.rmdir(is_dataset_root_path)
                logger.info("Dataset contents moved successfully")
            except Exception as e:
                logger.warning(f"Failed to move dataset contents: {str(e)}")
                # Continue anyway as the dataset might still be usable

            # Determine correct paths
            path_train, path_val, path_test = self._determine_dataset_paths(is_dataset_target_path)
            
            # Get class names
            class_names = self._get_class_names()
            
            # Create data.yaml
            self._create_data_yaml(is_dataset_target_path, path_train, path_val, path_test, class_names)
            
            logger.info(f"Dataset Instance Segmentation ready at: {is_dataset_target_path}")
            logger.info(f"data.yaml file created at: {self.IS_DATA_YAML}")
            return True
            
        except Exception as e:
            error_msg = f"Failed to prepare Instance Segmentation dataset: {str(e)}"
            logger.error(error_msg)
            raise DatasetError(error_msg) from e

    def _determine_dataset_paths(self, dataset_path: str) -> tuple[str, str, str]:
        """
        Determine the correct paths for train, validation, and test sets.
        
        Args:
            dataset_path: Path to the dataset directory
            
        Returns:
            tuple: (train_path, val_path, test_path)
        """
        path_train = 'train'
        path_val = 'valid'
        path_test = 'test'
        
        # Check if images are in subdirectories
        if os.path.exists(os.path.join(dataset_path, path_train, 'images')):
            path_train = 'train/images'
        if os.path.exists(os.path.join(dataset_path, path_val, 'images')):
            path_val = 'valid/images'
        if os.path.exists(os.path.join(dataset_path, path_test, 'images')):
            path_test = 'test/images'
            
        return path_train, path_val, path_test

    def _get_class_names(self) -> list[str]:
        """
        Get class names from Roboflow project.
        
        Returns:
            list: List of class names
        """
        try:
            rf = Roboflow(api_key=self.ROBOFLOW_API_KEY)
            project = rf.workspace().project(self.ROBOFLOW_IS_PROJECT)
            class_names = list(project.classes.keys())
            class_names.sort()
            logger.info(f"Retrieved {len(class_names)} class names from Roboflow")
            return class_names
        except Exception as e:
            logger.warning(f"Failed to get class names from Roboflow: {str(e)}. Using placeholder.")
            return ["class0", "class1"]

    def _create_data_yaml(self, dataset_path: str, path_train: str, path_val: str, 
                          path_test: str, class_names: list[str]) -> None:
        """
        Create data.yaml file for YOLO training.
        
        Args:
            dataset_path: Path to dataset
            path_train: Training data path
            path_val: Validation data path
            path_test: Test data path
            class_names: List of class names
        """
        is_data_yaml_content = {
            'path': os.path.abspath(dataset_path),
            'train': path_train,
            'val': path_val,
            'test': path_test,
            'names': {i: name for i, name in enumerate(class_names)}
        }
        
        try:
            with open(self.IS_DATA_YAML, 'w', encoding='utf-8') as f:
                yaml.dump(is_data_yaml_content, f, sort_keys=False, default_flow_style=False)
            logger.info(f"data.yaml created successfully at {self.IS_DATA_YAML}")
        except Exception as e:
            raise FileOperationError(f"Failed to create data.yaml: {str(e)}") from e

    def zip_datasets_folder(self) -> bool:
        """
        Melakukan kompresi (zip) pada folder 'datasets'.
        
        Returns:
            bool: True if successful, False otherwise
        """
        source_folder = self.DATASET_DIR
        output_filename = "datasets"
        
        if not os.path.exists(source_folder):
            logger.warning(f"Dataset folder '{source_folder}' not found. Cannot compress.")
            return False
        
        logger.info("Compressing datasets folder")
        
        try:
            shutil.make_archive(output_filename, 'zip', source_folder)
            logger.info(f"Folder '{source_folder}' successfully compressed to '{output_filename}.zip'")
            return True
        except Exception as e:
            error_msg = f"Failed to compress datasets folder '{source_folder}': {str(e)}"
            logger.error(error_msg)
            raise FileOperationError(error_msg) from e
