# file: modules/fuzzy_area_classifier.py

import numpy as np
import logging
from typing import Optional, Union, Dict, Any
import skfuzzy as fuzz
from skfuzzy import control as ctrl

from modules.exceptions import FuzzyLogicError, ValidationError

# Configure logging
logger = logging.getLogger(__name__)

class FuzzyAreaClassifier:
    """
    Mengklasifikasikan area sampah menggunakan sistem fuzzy.
    Kelas ini terpisah untuk modularitas dan kemudahan kustomisasi.
    """
    def __init__(self, fuzzy_config: Optional[Dict[str, Any]] = None):
        """
        Initialize FuzzyAreaClassifier with optional configuration.
        
        Args:
            fuzzy_config: Optional configuration dictionary for fuzzy parameters
            
        Raises:
            FuzzyLogicError: If fuzzy system setup fails
        """
        try:
            # Setup configuration
            self._setup_configuration(fuzzy_config)
            
            # Setup sistem fuzzy saat inisialisasi
            self.area_ctrl = self._setup_fuzzy_system()
            logger.info("Fuzzy system initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize fuzzy system: {str(e)}")
            raise FuzzyLogicError(f"Failed to initialize fuzzy system: {str(e)}") from e

    def _setup_configuration(self, fuzzy_config: Optional[Dict[str, Any]] = None) -> None:
        """
        Setup fuzzy configuration parameters.
        
        Args:
            fuzzy_config: Optional configuration dictionary
        """
        # Default configuration
        self.area_percent_ranges = {
            "sedikit": [0, 0, 5],
            "sedang": [3, 10, 20],
            "banyak": [15, 100, 100]
        }
        
        self.classification_score_ranges = {
            "low": [0, 0, 30],
            "medium": [20, 50, 80],
            "high": [70, 100, 100]
        }
        
        self.sedikit_threshold = 33.0
        self.sedang_threshold = 66.0
        
        self.fallback_sedikit_threshold = 1.0
        self.fallback_sedang_threshold = 10.0
        
        # Apply custom configuration if provided
        if fuzzy_config:
            if 'area_percent_ranges' in fuzzy_config:
                self.area_percent_ranges = fuzzy_config['area_percent_ranges']
            if 'classification_score_ranges' in fuzzy_config:
                self.classification_score_ranges = fuzzy_config['classification_score_ranges']
            if 'sedikit_threshold' in fuzzy_config:
                self.sedikit_threshold = fuzzy_config['sedikit_threshold']
            if 'sedang_threshold' in fuzzy_config:
                self.sedang_threshold = fuzzy_config['sedang_threshold']
            if 'fallback_sedikit_threshold' in fuzzy_config:
                self.fallback_sedikit_threshold = fuzzy_config['fallback_sedikit_threshold']
            if 'fallback_sedang_threshold' in fuzzy_config:
                self.fallback_sedang_threshold = fuzzy_config['fallback_sedang_threshold']
            
            logger.info("Custom fuzzy configuration applied")

    def _setup_fuzzy_system(self):
        """
        Mendefinisikan variabel linguistic, fungsi keanggotaan, dan aturan fuzzy
        untuk klasifikasi area sampah.
        
        Returns:
            ControlSystemSimulation: Configured fuzzy control system
            
        Raises:
            FuzzyLogicError: If fuzzy system setup fails
        """
        try:
            # Universe of discourse untuk persentase area sampah (0-100%)
            area_percent = ctrl.Antecedent(np.arange(0, 101, 1), 'area_percent')
            
            # Fungsi Keanggotaan untuk area_percent menggunakan konfigurasi
            area_percent['sedikit'] = fuzz.trimf(area_percent.universe, self.area_percent_ranges['sedikit'])
            area_percent['sedang'] = fuzz.trimf(area_percent.universe, self.area_percent_ranges['sedang'])
            area_percent['banyak'] = fuzz.trimf(area_percent.universe, self.area_percent_ranges['banyak'])

            # Universe of discourse untuk skor klasifikasi (0-100)
            classification_score = ctrl.Consequent(np.arange(0, 101, 1), 'classification_score')

            # Fungsi Keanggotaan untuk classification_score menggunakan konfigurasi
            classification_score['low'] = fuzz.trimf(classification_score.universe, self.classification_score_ranges['low'])
            classification_score['medium'] = fuzz.trimf(classification_score.universe, self.classification_score_ranges['medium'])
            classification_score['high'] = fuzz.trimf(classification_score.universe, self.classification_score_ranges['high'])

            # Aturan Fuzzy
            rule1 = ctrl.Rule(area_percent['sedikit'], classification_score['low'])
            rule2 = ctrl.Rule(area_percent['sedang'], classification_score['medium'])
            rule3 = ctrl.Rule(area_percent['banyak'], classification_score['high'])

            control_system = ctrl.ControlSystem([rule1, rule2, rule3])
            
            logger.info("Fuzzy rules and membership functions configured successfully")
            return ctrl.ControlSystemSimulation(control_system)
            
        except Exception as e:
            error_msg = f"Failed to setup fuzzy system: {str(e)}"
            logger.error(error_msg)
            raise FuzzyLogicError(error_msg) from e

    def _validate_input(self, normalized_area_percent: Union[int, float]) -> None:
        """
        Validate input parameters for fuzzy classification.
        
        Args:
            normalized_area_percent: Area percentage to classify
            
        Raises:
            ValidationError: If input is invalid
        """
        if not isinstance(normalized_area_percent, (int, float)):
            raise ValidationError("normalized_area_percent must be a number")
        
        if normalized_area_percent < 0:
            raise ValidationError("normalized_area_percent cannot be negative")
        
        if normalized_area_percent > 100:
            logger.warning(f"normalized_area_percent ({normalized_area_percent}) exceeds 100%, capping to 100%")
            normalized_area_percent = 100

    def classify_area(self, normalized_area_percent: Union[int, float]) -> str:
        """
        Mengklasifikasikan area sampah menggunakan sistem fuzzy.
        
        Args:
            normalized_area_percent: Area percentage (0-100) to classify
            
        Returns:
            str: Classification result ("sedikit", "sedang", or "banyak")
            
        Raises:
            ValidationError: If input is invalid
            FuzzyLogicError: If fuzzy computation fails
        """
        try:
            # Validate input
            self._validate_input(normalized_area_percent)
            
            if self.area_ctrl is None:
                logger.warning("Fuzzy system not available, using fallback classification")
                return self._fallback_classification(normalized_area_percent)

            # Perform fuzzy computation
            self.area_ctrl.input['area_percent'] = normalized_area_percent
            self.area_ctrl.compute()
            fuzzy_score = self.area_ctrl.output['classification_score']

            # Determine classification based on fuzzy score using configured thresholds
            classification = self._determine_classification(fuzzy_score)
            
            logger.debug(f"Area {normalized_area_percent}% classified as '{classification}' (fuzzy score: {fuzzy_score:.2f})")
            return classification
            
        except ValueError as e:
            error_msg = f"Error during fuzzy computation: {str(e)}"
            logger.error(error_msg)
            logger.info("Using fallback classification due to fuzzy computation error")
            return self._fallback_classification(normalized_area_percent)
        except Exception as e:
            error_msg = f"Unexpected error during area classification: {str(e)}"
            logger.error(error_msg)
            raise FuzzyLogicError(error_msg) from e

    def _fallback_classification(self, normalized_area_percent: Union[int, float]) -> str:
        """
        Fallback classification method when fuzzy system is unavailable.
        
        Args:
            normalized_area_percent: Area percentage to classify
            
        Returns:
            str: Classification result
        """
        if normalized_area_percent < self.fallback_sedikit_threshold:
            return "sedikit"
        elif normalized_area_percent < self.fallback_sedang_threshold:
            return "sedang"
        else:
            return "banyak"

    def _determine_classification(self, fuzzy_score: float) -> str:
        """
        Determine classification based on fuzzy score using configured thresholds.
        
        Args:
            fuzzy_score: Computed fuzzy score
            
        Returns:
            str: Classification result
        """
        if fuzzy_score <= self.sedikit_threshold:
            return "sedikit"
        elif fuzzy_score <= self.sedang_threshold:
            return "sedang"
        else:
            return "banyak"

    def get_membership_functions(self) -> dict:
        """
        Get current membership function parameters for debugging/analysis.
        
        Returns:
            dict: Membership function parameters
        """
        try:
            if self.area_ctrl is None:
                return {"error": "Fuzzy system not initialized"}
            
            return {
                "area_percent": self.area_percent_ranges,
                "classification_score": self.classification_score_ranges,
                "thresholds": {
                    "sedikit": self.sedikit_threshold,
                    "sedang": self.sedang_threshold
                },
                "fallback_thresholds": {
                    "sedikit": self.fallback_sedikit_threshold,
                    "sedang": self.fallback_sedang_threshold
                }
            }
        except Exception as e:
            logger.error(f"Failed to get membership functions: {str(e)}")
            return {"error": str(e)}

    def update_membership_functions(self, area_params: dict, score_params: dict, 
                                  thresholds: Optional[dict] = None) -> bool:
        """
        Update membership function parameters and reinitialize fuzzy system.
        
        Args:
            area_params: New area_percent membership function parameters
            score_params: New classification_score membership function parameters
            thresholds: Optional new classification thresholds
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Update parameters
            if area_params:
                self.area_percent_ranges.update(area_params)
            if score_params:
                self.classification_score_ranges.update(score_params)
            if thresholds:
                if 'sedikit' in thresholds:
                    self.sedikit_threshold = thresholds['sedikit']
                if 'sedang' in thresholds:
                    self.sedang_threshold = thresholds['sedang']
            
            # Reinitialize fuzzy system with new parameters
            self.area_ctrl = self._setup_fuzzy_system()
            logger.info("Membership functions updated successfully")
            return True
        except Exception as e:
            error_msg = f"Failed to update membership functions: {str(e)}"
            logger.error(error_msg)
            return False

    def get_configuration(self) -> Dict[str, Any]:
        """
        Get current fuzzy configuration.
        
        Returns:
            Dict[str, Any]: Current configuration
        """
        return {
            "area_percent_ranges": self.area_percent_ranges,
            "classification_score_ranges": self.classification_score_ranges,
            "sedikit_threshold": self.sedikit_threshold,
            "sedang_threshold": self.sedang_threshold,
            "fallback_sedikit_threshold": self.fallback_sedikit_threshold,
            "fallback_sedang_threshold": self.fallback_sedang_threshold
        }

