# file: modules/dataset_manager.py
import os
import shutil
import yaml
import json
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

    def _extract_local_dataset(self, dataset_zip_path: str) -> Optional[str]:
        """
        Extract dataset from local dataset.zip file.
        
        Args:
            dataset_zip_path: Path to the dataset.zip file
            
        Returns:
            Optional[str]: Path to extracted dataset directory or None if failed
            
        Raises:
            FileOperationError: If extraction fails
        """
        try:
            import zipfile
            import tempfile
            
            logger.info(f"Extracting {dataset_zip_path}...")
            
            # Create temporary directory for extraction
            temp_dir = tempfile.mkdtemp(prefix="dataset_extract_")
            
            # Extract the zip file
            with zipfile.ZipFile(dataset_zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            logger.info(f"Dataset extracted to: {temp_dir}")
            
            # Find the dataset directory (usually contains train, valid, test folders)
            extracted_contents = os.listdir(temp_dir)
            if len(extracted_contents) == 1 and os.path.isdir(os.path.join(temp_dir, extracted_contents[0])):
                # Single directory containing the dataset
                dataset_dir = os.path.join(temp_dir, extracted_contents[0])
            else:
                # Multiple files/folders, use temp_dir as dataset directory
                dataset_dir = temp_dir
            
            logger.info(f"Using dataset directory: {dataset_dir}")
            return dataset_dir
            
        except Exception as e:
            error_msg = f"Failed to extract local dataset: {str(e)}"
            logger.error(error_msg)
            raise FileOperationError(error_msg) from e

    def _normalize_dataset_ultralytics(self, dataset_path: str) -> bool:
        """
        Normalize dataset using Ultralytics tools to ensure proper YOLO format.
        This method converts COCO annotations to YOLO format and ensures only 'sampah' class is used.
        
        Args:
            dataset_path: Path to the dataset directory
            
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info("Normalizing dataset using Ultralytics tools...")
        
        try:
            # Create backup of original dataset
            backup_path = f"{dataset_path}_backup"
            if os.path.exists(backup_path):
                shutil.rmtree(backup_path)
            shutil.copytree(dataset_path, backup_path)
            logger.info(f"Created backup at: {backup_path}")
            
            # Convert each split to YOLO format using the proven approach
            for split in ['train', 'valid', 'test']:
                split_path = os.path.join(dataset_path, split)
                coco_file = os.path.join(split_path, "_annotations.coco.json")
                
                if not os.path.exists(coco_file):
                    logger.warning(f"COCO file not found for {split}: {coco_file}")
                    continue
                
                logger.info(f"Processing {split} split...")
                
                # Create images and labels directories
                images_dir = os.path.join(split_path, "images")
                labels_dir = os.path.join(split_path, "labels")
                
                os.makedirs(images_dir, exist_ok=True)
                os.makedirs(labels_dir, exist_ok=True)
                
                # Move images to images directory
                image_files = []
                for file in os.listdir(split_path):
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')) and file != "_annotations.coco.json":
                        src = os.path.join(split_path, file)
                        dst = os.path.join(images_dir, file)
                        shutil.move(src, dst)
                        image_files.append(file)
                
                logger.info(f"  📸 Moved {len(image_files)} images to images/")
                
                # Parse COCO JSON and create YOLO labels
                try:
                    with open(coco_file, 'r', encoding='utf-8') as f:
                        coco_data = json.load(f)
                    
                    # Create a mapping from image ID to image info
                    image_id_to_info = {}
                    for image in coco_data.get('images', []):
                        image_id = image['id']
                        image_id_to_info[image_id] = {
                            'filename': image['file_name'],
                            'width': image['width'],
                            'height': image['height']
                        }
                    
                    # Group annotations by image_id
                    annotations_by_image = {}
                    for annotation in coco_data.get('annotations', []):
                        image_id = annotation['image_id']
                        if image_id not in annotations_by_image:
                            annotations_by_image[image_id] = []
                        annotations_by_image[image_id].append(annotation)
                    
                    # Process each image
                    label_count = 0
                    for image_id, image_info in image_id_to_info.items():
                        filename = image_info['filename']
                        img_width = image_info['width']
                        img_height = image_info['height']
                        base_name = os.path.splitext(filename)[0]
                        
                        # Create YOLO label file
                        label_file = os.path.join(labels_dir, f"{base_name}.txt")
                        
                        # Get annotations for this image
                        annotations = annotations_by_image.get(image_id, [])
                        
                        # Create YOLO format lines
                        yolo_lines = []
                        for annotation in annotations:
                            # Get bounding box
                            bbox = annotation.get('bbox', [0, 0, 100, 100])
                            x, y, w, h = bbox
                            
                            # Validate image dimensions
                            if img_width <= 0 or img_height <= 0:
                                logger.warning(f"    ⚠️ Invalid image dimensions for {filename}: {img_width}x{img_height}")
                                continue
                            
                            # Convert to YOLO format (normalized coordinates)
                            # YOLO format: class_id x_center y_center width height
                            x_center = (x + w/2) / img_width
                            y_center = (y + h/2) / img_height
                            width = w / img_width
                            height = h / img_height
                            
                            # Validate coordinates are within bounds
                            if (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                                0 <= width <= 1 and 0 <= height <= 1):
                                # Use class 0 for 'sampah'
                                yolo_line = f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                                yolo_lines.append(yolo_line)
                            else:
                                logger.warning(f"    ⚠️ Skipping annotation with out-of-bounds coordinates for {filename}")
                                logger.warning(f"      x_center={x_center:.4f}, y_center={y_center:.4f}, width={width:.4f}, height={height:.4f}")
                        
                        # Write the label file
                        with open(label_file, 'w') as f:
                            f.writelines(yolo_lines)
                        
                        if yolo_lines:
                            label_count += 1
                    
                    logger.info(f"  🏷️ Created {label_count} label files")
                    
                except Exception as e:
                    logger.error(f"  ❌ Error processing COCO JSON for {split}: {e}")
                    continue
            
            # Create proper data.yaml for YOLO format
            self._create_yolo_data_yaml(dataset_path)
            
            logger.info("✅ Dataset normalization completed successfully")
            return True
            
        except ImportError:
            logger.error("❌ Ultralytics not installed. Install with: pip install ultralytics")
            return False
        except Exception as e:
            logger.error(f"❌ Dataset normalization failed: {str(e)}")
            return False

    def _create_yolo_data_yaml(self, dataset_path: str) -> None:
        """
        Create YOLO format data.yaml file with only 'sampah' class.
        
        Args:
            dataset_path: Path to the dataset directory
        """
        # Determine the correct paths for train, validation, and test sets
        path_train, path_val, path_test = self._determine_dataset_paths(dataset_path)
        
        # Use relative path to avoid platform-specific absolute path issues
        yolo_data_yaml = {
            'path': '.',  # Use relative path
            'train': path_train,
            'val': path_val,
            'test': path_test,
            'names': {0: 'sampah'}  # Only 'sampah' class
        }
        
        data_yaml_path = os.path.join(dataset_path, "data.yaml")
        
        try:
            with open(data_yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(yolo_data_yaml, f, sort_keys=False, default_flow_style=False)
            
            logger.info(f"✅ Created YOLO data.yaml: {data_yaml_path}")
            logger.info(f"Using paths: train={path_train}, val={path_val}, test={path_test}")
            
        except Exception as e:
            raise FileOperationError(f"Failed to create YOLO data.yaml: {str(e)}") from e

    def prepare_datasets(self) -> bool:
        """
        Mempersiapkan dataset untuk Instance Segmentation.
        Checks for existing dataset.zip first, then downloads from Roboflow if needed.
        Uses Ultralytics tools to ensure proper YOLO format with only 'sampah' class.
        
        Returns:
            bool: True if successful, False otherwise
            
        Raises:
            DatasetError: If dataset preparation fails
        """
        logger.info("Preparing Instance Segmentation Dataset")
        
        try:
            # Check for existing dataset.zip in root folder
            dataset_zip_path = "dataset.zip"
            if os.path.exists(dataset_zip_path):
                logger.info(f"Found existing {dataset_zip_path} in root folder. Using local dataset.")
                is_dataset_root_path = self._extract_local_dataset(dataset_zip_path)
            else:
                logger.info(f"No {dataset_zip_path} found. Downloading from Roboflow...")
                is_dataset_root_path = self._download_from_roboflow(
                    self.ROBOFLOW_IS_PROJECT, 
                    self.ROBOFLOW_IS_VERSION, 
                    dataset_type="coco-segmentation"
                )
            
            if not is_dataset_root_path:
                raise DatasetError("Failed to prepare dataset")
            
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

            # Normalize dataset using Ultralytics tools
            if not self._normalize_dataset_ultralytics(is_dataset_target_path):
                logger.warning("Dataset normalization failed, but continuing with original format")
                # Fallback to original method
                class_names = self._get_class_names()
                self._create_data_yaml_standard(is_dataset_target_path, class_names)
            
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
        Get class names from Roboflow project or use default for local dataset.
        Only returns 'sampah' label as that's the main label for this project.
        
        Returns:
            list: List of class names (only 'sampah')
        """
        try:
            rf = Roboflow(api_key=self.ROBOFLOW_API_KEY)
            project = rf.workspace().project(self.ROBOFLOW_IS_PROJECT)
            all_class_names = list(project.classes.keys())
            logger.info(f"Retrieved {len(all_class_names)} class names from Roboflow: {all_class_names}")
            
            # Filter to only use 'sampah' label and ignore other categories
            if 'sampah' in all_class_names:
                logger.info("Using 'sampah' label as requested (ignoring other categories)")
                return ["sampah"]
            else:
                logger.warning("'sampah' label not found in Roboflow classes")
                logger.warning(f"Available categories: {all_class_names}")
                logger.warning("Using first available class as fallback")
                return [all_class_names[0]] if all_class_names else ["sampah"]
                
        except Exception as e:
            logger.warning(f"Failed to get class names from Roboflow: {str(e)}. Using 'sampah' as default.")
            # Default to 'sampah' since that's the actual label in your Roboflow project
            return ["sampah"]

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
        # Use relative path to avoid platform-specific absolute path issues
        is_data_yaml_content = {
            'path': '.',  # Use relative path instead of absolute
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

    def _create_data_yaml_standard(self, dataset_path: str, class_names: list[str]) -> None:
        """
        Create data.yaml file for YOLO training with standard format.
        This method points to image directories and uses the COCO annotations that come with Roboflow.
        Forces all segmentations to use 'sampah' category.
        
        Args:
            dataset_path: Path to dataset
            class_names: List of class names
        """
        # Determine the correct paths for train, validation, and test sets
        path_train, path_val, path_test = self._determine_dataset_paths(dataset_path)
        
        # Force all segmentations to use 'sampah' category
        self._normalize_coco_annotations(dataset_path)
        
        # Use relative path to avoid platform-specific absolute path issues
        is_data_yaml_content = {
            'path': '.',  # Use relative path instead of absolute
            'train': path_train,
            'val': path_val,
            'test': path_test,
            'names': {i: name for i, name in enumerate(class_names)}
        }
        
        try:
            with open(self.IS_DATA_YAML, 'w', encoding='utf-8') as f:
                yaml.dump(is_data_yaml_content, f, sort_keys=False, default_flow_style=False)
            logger.info(f"data.yaml with standard YOLO format created successfully at {self.IS_DATA_YAML}")
            logger.info(f"Using paths: train={path_train}, val={path_val}, test={path_test}")
        except Exception as e:
            raise FileOperationError(f"Failed to create data.yaml: {str(e)}") from e

    def _normalize_coco_annotations(self, dataset_path: str) -> None:
        """
        Normalize COCO annotations to use only 'sampah' category.
        This converts all segmentations to use the 'sampah' category ID.
        
        Args:
            dataset_path: Path to dataset
        """
        logger.info("Normalizing COCO annotations to use only 'sampah' category")
        
        for split in ['train', 'valid', 'test']:
            split_path = os.path.join(dataset_path, split)
            coco_file = os.path.join(split_path, "_annotations.coco.json")
            
            if not os.path.exists(coco_file):
                logger.warning(f"COCO file not found for {split}")
                continue
            
            try:
                with open(coco_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Find 'sampah' category ID
                sampah_category_id = None
                for cat in data.get('categories', []):
                    if cat.get('name') == 'sampah':
                        sampah_category_id = cat.get('id')
                        break
                
                if sampah_category_id is None:
                    logger.warning(f"No 'sampah' category found in {split}, skipping")
                    continue
                
                # Update all annotations to use 'sampah' category
                annotations_updated = 0
                for ann in data.get('annotations', []):
                    old_category_id = ann.get('category_id')
                    if old_category_id != sampah_category_id:
                        ann['category_id'] = sampah_category_id
                        annotations_updated += 1
                
                # Update categories to only include 'sampah'
                data['categories'] = [cat for cat in data.get('categories', []) 
                                   if cat.get('name') == 'sampah']
                
                # Save updated annotations
                with open(coco_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Updated {split}: {annotations_updated} annotations normalized to 'sampah' category")
                
            except Exception as e:
                logger.error(f"Failed to normalize {split} annotations: {str(e)}")

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

    def fix_dataset_classes(self, dataset_path: Optional[str] = None) -> bool:
        """
        Fix dataset class issues using Ultralytics tools.
        This method can be called independently to fix existing datasets.
        
        Args:
            dataset_path: Path to dataset directory (uses default if None)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if dataset_path is None:
            dataset_path = os.path.join(self.DATASET_DIR, self.ROBOFLOW_IS_PROJECT)
        
        if not os.path.exists(dataset_path):
            logger.error(f"Dataset path not found: {dataset_path}")
            return False
        
        logger.info(f"Fixing dataset classes in: {dataset_path}")
        
        try:
            # Check if dataset needs fixing
            data_yaml_path = os.path.join(dataset_path, "data.yaml")
            needs_fixing = False
            
            if os.path.exists(data_yaml_path):
                with open(data_yaml_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                    names = data.get('names', {})
                    path_value = data.get('path', '')
                    
                    # Check for absolute path issues
                    if path_value and (path_value.startswith('D:\\') or path_value.startswith('/') or '\\' in path_value):
                        logger.info("Dataset has absolute path in data.yaml, fixing...")
                        needs_fixing = True
                    elif names != {0: 'sampah'}:
                        logger.info("Dataset has incorrect class configuration, fixing...")
                        needs_fixing = True
            else:
                logger.info("No data.yaml found, creating proper YOLO format...")
                needs_fixing = True
            
            if needs_fixing:
                return self._normalize_dataset_ultralytics(dataset_path)
            else:
                logger.info("Dataset already has correct class configuration")
                return True
                
        except Exception as e:
            logger.error(f"Failed to fix dataset classes: {str(e)}")
            return False

    def validate_dataset_format(self, dataset_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate dataset format and provide detailed information.
        
        Args:
            dataset_path: Path to dataset directory (uses default if None)
            
        Returns:
            Dict[str, Any]: Validation results
        """
        if dataset_path is None:
            dataset_path = os.path.join(self.DATASET_DIR, self.ROBOFLOW_IS_PROJECT)
        
        results = {
            'dataset_path': dataset_path,
            'exists': os.path.exists(dataset_path),
            'data_yaml_exists': False,
            'data_yaml_content': None,
            'splits': {},
            'issues': [],
            'recommendations': []
        }
        
        if not results['exists']:
            results['issues'].append("Dataset directory does not exist")
            return results

    def fix_data_yaml_paths(self, dataset_path: Optional[str] = None) -> bool:
        """
        Fix absolute paths in existing data.yaml files to use relative paths.
        
        Args:
            dataset_path: Path to dataset directory (uses default if None)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if dataset_path is None:
            dataset_path = os.path.join(self.DATASET_DIR, self.ROBOFLOW_IS_PROJECT)
        
        if not os.path.exists(dataset_path):
            logger.error(f"Dataset path not found: {dataset_path}")
            return False
        
        data_yaml_path = os.path.join(dataset_path, "data.yaml")
        
        if not os.path.exists(data_yaml_path):
            logger.warning(f"No data.yaml found at: {data_yaml_path}")
            return False
        
        try:
            # Read current data.yaml
            with open(data_yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            # Check if path is absolute
            current_path = data.get('path', '')
            if current_path and (current_path.startswith('D:\\') or current_path.startswith('/') or '\\' in current_path):
                logger.info(f"Found absolute path in data.yaml: {current_path}")
                logger.info("Converting to relative path...")
                
                # Update to relative path
                data['path'] = '.'
                
                # Write back the updated data.yaml
                with open(data_yaml_path, 'w', encoding='utf-8') as f:
                    yaml.dump(data, f, sort_keys=False, default_flow_style=False)
                
                logger.info("✅ Successfully converted absolute path to relative path")
                return True
            else:
                logger.info("Data.yaml already uses relative path or no path found")
                return True
                
        except Exception as e:
            logger.error(f"Failed to fix data.yaml paths: {str(e)}")
            return False
        
        # Check data.yaml
        data_yaml_path = os.path.join(dataset_path, "data.yaml")
        results['data_yaml_exists'] = os.path.exists(data_yaml_path)
        
        if results['data_yaml_exists']:
            try:
                with open(data_yaml_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                    results['data_yaml_content'] = data
                    
                    # Check class names
                    names = data.get('names', {})
                    if names != {0: 'sampah'}:
                        results['issues'].append(f"Incorrect class names: {names}")
                        results['recommendations'].append("Run fix_dataset_classes() to correct class configuration")
                    
                    # Check paths
                    for split in ['train', 'val', 'test']:
                        split_path = data.get(split, '')
                        if split_path:
                            full_path = os.path.join(dataset_path, split_path)
                            if not os.path.exists(full_path):
                                results['issues'].append(f"Path not found: {full_path}")
                            else:
                                results['splits'][split] = {
                                    'path': full_path,
                                    'exists': True,
                                    'images_count': 0,
                                    'labels_count': 0
                                }
                                
                                # Count files
                                images_dir = os.path.join(full_path, "images")
                                labels_dir = os.path.join(full_path, "labels")
                                
                                if os.path.exists(images_dir):
                                    results['splits'][split]['images_count'] = len([
                                        f for f in os.listdir(images_dir) 
                                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                                    ])
                                
                                if os.path.exists(labels_dir):
                                    results['splits'][split]['labels_count'] = len([
                                        f for f in os.listdir(labels_dir) 
                                        if f.endswith('.txt')
                                    ])
            except Exception as e:
                results['issues'].append(f"Error reading data.yaml: {str(e)}")
        else:
            results['issues'].append("data.yaml not found")
            results['recommendations'].append("Run fix_dataset_classes() to create proper YOLO format")
        
        return results
