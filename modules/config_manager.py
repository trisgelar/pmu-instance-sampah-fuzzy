# file: modules/config_manager.py

import os
import yaml
import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

from modules.exceptions import ConfigurationError, ValidationError

logger = logging.getLogger(__name__)

class Environment(Enum):
    """Environment types for configuration."""
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"
    COLAB = "colab"

@dataclass
class ModelConfig:
    """Configuration for model training and inference."""
    # Model versions
    supported_versions: list[str] = field(default_factory=lambda: ["v8n", "v10n", "v11n"])
    
    # Training parameters
    default_epochs: int = 50
    default_batch_size: int = 16
    min_epochs: int = 1
    max_epochs: int = 1000
    min_batch_size: int = 1
    max_batch_size: int = 128
    
    # Image processing
    default_img_size: tuple[int, int] = (640, 640)
    supported_img_sizes: list[tuple[int, int]] = field(default_factory=lambda: [
        (416, 416), (512, 512), (640, 640), (832, 832), (1024, 1024)
    ])
    
    # Model export settings
    onnx_opset: int = 12
    onnx_simplify: bool = True
    
    # Inference settings
    default_conf_threshold: float = 0.25
    min_conf_threshold: float = 0.0
    max_conf_threshold: float = 1.0

@dataclass
class DatasetConfig:
    """Configuration for dataset management."""
    # Roboflow settings
    roboflow_project: str = "abcd12efghijklmn34opqrstuvwxyza1bc2de3f_segmentation"
    roboflow_version: str = "1"
    dataset_type: str = "coco-segmentation"
    
    # Directory structure
    default_dataset_dir: str = "datasets"
    default_model_dir: str = "runs"
    default_onnx_dir: str = "onnx_models"
    default_rknn_dir: str = "rknn_models"
    
    # File patterns
    supported_image_extensions: list[str] = field(default_factory=lambda: [".jpg", ".jpeg", ".png", ".bmp"])
    supported_video_extensions: list[str] = field(default_factory=lambda: [".mp4", ".avi", ".mov"])

@dataclass
class FuzzyConfig:
    """Configuration for fuzzy logic classification."""
    # Membership function parameters
    area_percent_ranges: Dict[str, list[int]] = field(default_factory=lambda: {
        "sedikit": [0, 0, 5],
        "sedang": [3, 10, 20],
        "banyak": [15, 100, 100]
    })
    
    classification_score_ranges: Dict[str, list[int]] = field(default_factory=lambda: {
        "low": [0, 0, 30],
        "medium": [20, 50, 80],
        "high": [70, 100, 100]
    })
    
    # Classification thresholds
    sedikit_threshold: float = 33.0
    sedang_threshold: float = 66.0
    
    # Fallback thresholds
    fallback_sedikit_threshold: float = 1.0
    fallback_sedang_threshold: float = 10.0

@dataclass
class LoggingConfig:
    """Configuration for logging system."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_logging: bool = True
    console_logging: bool = True
    log_file: str = "waste_detection_system.log"
    max_log_size_mb: int = 10
    backup_count: int = 5

@dataclass
class SystemConfig:
    """System-wide configuration."""
    # Environment
    environment: Environment = Environment.COLAB
    
    # Performance settings
    num_workers: int = 4
    pin_memory: bool = True
    mixed_precision: bool = True
    
    # File operations
    create_backups: bool = True
    backup_retention_days: int = 30
    
    # API settings
    request_timeout: int = 300
    max_retries: int = 3
    retry_delay: int = 5

class ConfigManager:
    """
    Centralized configuration management for the waste detection system.
    Supports multiple environments, validation, and dynamic configuration updates.
    """
    
    def __init__(self, config_path: Optional[str] = None, environment: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
            environment: Environment to load (development, production, testing, colab)
        """
        self.config_path = config_path or "config.yaml"
        self.environment = Environment(environment) if environment else Environment.COLAB
        
        # Initialize default configurations
        self.model_config = ModelConfig()
        self.dataset_config = DatasetConfig()
        self.fuzzy_config = FuzzyConfig()
        self.logging_config = LoggingConfig()
        self.system_config = SystemConfig()
        
        # Load configuration
        self._load_configuration()
        self._validate_configuration()
        
        logger.info(f"Configuration loaded for environment: {self.environment.value}")

    def _load_configuration(self) -> None:
        """Load configuration from file or create default."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
                    if config_data:
                        self._apply_configuration(config_data)
                        logger.info(f"Configuration loaded from {self.config_path}")
                    else:
                        logger.warning(f"Empty configuration file: {self.config_path}")
                        self._create_default_config()
            else:
                logger.info(f"Configuration file not found: {self.config_path}. Creating default configuration.")
                self._create_default_config()
                
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML in configuration file: {str(e)}")
            self._create_default_config()
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            self._create_default_config()

    def _apply_configuration(self, config_data: Dict[str, Any]) -> None:
        """Apply configuration data to components."""
        try:
            # Apply model configuration
            if 'model' in config_data:
                model_data = config_data['model']
                for key, value in model_data.items():
                    if hasattr(self.model_config, key):
                        # Convert lists to tuples for img_size fields
                        if key == 'default_img_size' and isinstance(value, list):
                            value = tuple(value)
                        elif key == 'supported_img_sizes' and isinstance(value, list):
                            value = [tuple(item) if isinstance(item, list) else item for item in value]
                        setattr(self.model_config, key, value)
            
            # Apply dataset configuration
            if 'dataset' in config_data:
                dataset_data = config_data['dataset']
                for key, value in dataset_data.items():
                    if hasattr(self.dataset_config, key):
                        setattr(self.dataset_config, key, value)
            
            # Apply fuzzy configuration
            if 'fuzzy' in config_data:
                fuzzy_data = config_data['fuzzy']
                for key, value in fuzzy_data.items():
                    if hasattr(self.fuzzy_config, key):
                        setattr(self.fuzzy_config, key, value)
            
            # Apply logging configuration
            if 'logging' in config_data:
                logging_data = config_data['logging']
                for key, value in logging_data.items():
                    if hasattr(self.logging_config, key):
                        setattr(self.logging_config, key, value)
            
            # Apply system configuration
            if 'system' in config_data:
                system_data = config_data['system']
                for key, value in system_data.items():
                    if hasattr(self.system_config, key):
                        if key == 'environment':
                            self.system_config.environment = Environment(value)
                        else:
                            setattr(self.system_config, key, value)
                            
        except Exception as e:
            logger.error(f"Failed to apply configuration: {str(e)}")
            raise ConfigurationError(f"Configuration application failed: {str(e)}") from e

    def _create_default_config(self) -> None:
        """Create and save default configuration."""
        default_config = {
            'model': {
                'supported_versions': self.model_config.supported_versions,
                'default_epochs': self.model_config.default_epochs,
                'default_batch_size': self.model_config.default_batch_size,
                'default_img_size': self.model_config.default_img_size,
                'default_conf_threshold': self.model_config.default_conf_threshold
            },
            'dataset': {
                'roboflow_project': self.dataset_config.roboflow_project,
                'roboflow_version': self.dataset_config.roboflow_version,
                'default_dataset_dir': self.dataset_config.default_dataset_dir,
                'default_model_dir': self.dataset_config.default_model_dir
            },
            'fuzzy': {
                'area_percent_ranges': self.fuzzy_config.area_percent_ranges,
                'classification_score_ranges': self.fuzzy_config.classification_score_ranges,
                'sedikit_threshold': self.fuzzy_config.sedikit_threshold,
                'sedang_threshold': self.fuzzy_config.sedang_threshold
            },
            'logging': {
                'level': self.logging_config.level,
                'file_logging': self.logging_config.file_logging,
                'console_logging': self.logging_config.console_logging
            },
            'system': {
                'environment': self.system_config.environment.value,
                'num_workers': self.system_config.num_workers,
                'request_timeout': self.system_config.request_timeout
            }
        }
        
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(default_config, f, default_flow_style=False, sort_keys=False)
            logger.info(f"Default configuration created at {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to create default configuration: {str(e)}")

    def _validate_configuration(self) -> None:
        """Validate configuration parameters."""
        try:
            # Validate model configuration
            if not (self.model_config.min_epochs <= self.model_config.default_epochs <= self.model_config.max_epochs):
                raise ValidationError(f"Default epochs ({self.model_config.default_epochs}) out of range [{self.model_config.min_epochs}, {self.model_config.max_epochs}]")
            
            if not (self.model_config.min_batch_size <= self.model_config.default_batch_size <= self.model_config.max_batch_size):
                raise ValidationError(f"Default batch size ({self.model_config.default_batch_size}) out of range [{self.model_config.min_batch_size}, {self.model_config.max_batch_size}]")
            
            if self.model_config.default_img_size not in self.model_config.supported_img_sizes:
                raise ValidationError(f"Default image size {self.model_config.default_img_size} not in supported sizes: {self.model_config.supported_img_sizes}")
            
            # Validate fuzzy configuration
            if not (0 <= self.fuzzy_config.sedikit_threshold <= 100):
                raise ValidationError(f"Sedikit threshold ({self.fuzzy_config.sedikit_threshold}) must be between 0 and 100")
            
            if not (0 <= self.fuzzy_config.sedang_threshold <= 100):
                raise ValidationError(f"Sedang threshold ({self.fuzzy_config.sedang_threshold}) must be between 0 and 100")
            
            # Validate system configuration
            if self.system_config.num_workers < 0:
                raise ValidationError(f"Number of workers ({self.system_config.num_workers}) must be non-negative")
            
            logger.info("Configuration validation passed")
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {str(e)}")
            raise ValidationError(f"Configuration validation failed: {str(e)}") from e

    def get_model_config(self) -> ModelConfig:
        """Get model configuration."""
        return self.model_config

    def get_dataset_config(self) -> DatasetConfig:
        """Get dataset configuration."""
        return self.dataset_config

    def get_fuzzy_config(self) -> FuzzyConfig:
        """Get fuzzy logic configuration."""
        return self.fuzzy_config

    def get_logging_config(self) -> LoggingConfig:
        """Get logging configuration."""
        return self.logging_config

    def get_system_config(self) -> SystemConfig:
        """Get system configuration."""
        return self.system_config

    def update_config(self, section: str, key: str, value: Any) -> bool:
        """
        Update configuration parameter.
        
        Args:
            section: Configuration section (model, dataset, fuzzy, logging, system)
            key: Parameter key
            value: New value
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Determine which config object to update
            config_objects = {
                'model': self.model_config,
                'dataset': self.dataset_config,
                'fuzzy': self.fuzzy_config,
                'logging': self.logging_config,
                'system': self.system_config
            }
            
            if section not in config_objects:
                raise ValidationError(f"Invalid configuration section: {section}")
            
            config_obj = config_objects[section]
            
            # Special handling for environment
            if section == 'system' and key == 'environment':
                value = Environment(value)
            
            # Update the configuration
            if hasattr(config_obj, key):
                setattr(config_obj, key, value)
                logger.info(f"Updated configuration: {section}.{key} = {value}")
                
                # Re-validate configuration
                self._validate_configuration()
                
                # Save to file
                self._save_configuration()
                
                return True
            else:
                raise ValidationError(f"Invalid configuration key: {key} in section {section}")
                
        except Exception as e:
            logger.error(f"Failed to update configuration: {str(e)}")
            return False

    def _save_configuration(self) -> None:
        """Save current configuration to file."""
        try:
            config_data = {
                'model': {
                    'supported_versions': self.model_config.supported_versions,
                    'default_epochs': self.model_config.default_epochs,
                    'default_batch_size': self.model_config.default_batch_size,
                    'default_img_size': self.model_config.default_img_size,
                    'default_conf_threshold': self.model_config.default_conf_threshold
                },
                'dataset': {
                    'roboflow_project': self.dataset_config.roboflow_project,
                    'roboflow_version': self.dataset_config.roboflow_version,
                    'default_dataset_dir': self.dataset_config.default_dataset_dir,
                    'default_model_dir': self.dataset_config.default_model_dir
                },
                'fuzzy': {
                    'area_percent_ranges': self.fuzzy_config.area_percent_ranges,
                    'classification_score_ranges': self.fuzzy_config.classification_score_ranges,
                    'sedikit_threshold': self.fuzzy_config.sedikit_threshold,
                    'sedang_threshold': self.fuzzy_config.sedang_threshold
                },
                'logging': {
                    'level': self.logging_config.level,
                    'file_logging': self.logging_config.file_logging,
                    'console_logging': self.logging_config.console_logging
                },
                'system': {
                    'environment': self.system_config.environment.value,
                    'num_workers': self.system_config.num_workers,
                    'request_timeout': self.system_config.request_timeout
                }
            }
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
            
            logger.info(f"Configuration saved to {self.config_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {str(e)}")

    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary for display."""
        return {
            'environment': self.system_config.environment.value,
            'model': {
                'supported_versions': self.model_config.supported_versions,
                'default_epochs': self.model_config.default_epochs,
                'default_batch_size': self.model_config.default_batch_size,
                'default_img_size': self.model_config.default_img_size
            },
            'dataset': {
                'roboflow_project': self.dataset_config.roboflow_project,
                'roboflow_version': self.dataset_config.roboflow_version,
                'directories': {
                    'dataset': self.dataset_config.default_dataset_dir,
                    'model': self.dataset_config.default_model_dir,
                    'onnx': self.dataset_config.default_onnx_dir,
                    'rknn': self.dataset_config.default_rknn_dir
                }
            },
            'fuzzy': {
                'area_ranges': self.fuzzy_config.area_percent_ranges,
                'thresholds': {
                    'sedikit': self.fuzzy_config.sedikit_threshold,
                    'sedang': self.fuzzy_config.sedang_threshold
                }
            },
            'system': {
                'num_workers': self.system_config.num_workers,
                'request_timeout': self.system_config.request_timeout
            }
        }

    def create_environment_config(self, environment: Environment) -> None:
        """
        Create environment-specific configuration.
        
        Args:
            environment: Target environment
        """
        env_config_path = f"config_{environment.value}.yaml"
        
        # Adjust configuration for environment
        if environment == Environment.PRODUCTION:
            self.logging_config.level = "WARNING"
            self.system_config.num_workers = 8
            self.model_config.default_batch_size = 32
        elif environment == Environment.DEVELOPMENT:
            self.logging_config.level = "DEBUG"
            self.system_config.num_workers = 2
            self.model_config.default_batch_size = 8
        elif environment == Environment.TESTING:
            self.logging_config.level = "INFO"
            self.system_config.num_workers = 1
            self.model_config.default_epochs = 2
            self.model_config.default_batch_size = 4
        
        # Save environment-specific configuration
        self.config_path = env_config_path
        self.system_config.environment = environment
        self._save_configuration()
        
        logger.info(f"Environment-specific configuration created: {env_config_path}") 