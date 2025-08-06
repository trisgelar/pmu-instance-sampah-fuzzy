#!/usr/bin/env python3
"""
Test script to verify that the restructured training files work correctly.
"""

import sys
import os
import unittest
import logging

# Add the project root to the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestTrainingImports(unittest.TestCase):
    """Test that training modules can be imported and used correctly."""
    
    def test_enhanced_training_import(self):
        """Test that EnhancedTrainingSystem can be imported."""
        try:
            from tests.training.enhanced_training_system import EnhancedTrainingSystem
            logger.info("✅ EnhancedTrainingSystem imported successfully")
            
            # Test instantiation
            system = EnhancedTrainingSystem()
            self.assertIsNotNone(system)
            logger.info("✅ EnhancedTrainingSystem instantiated successfully")
            
        except ImportError as e:
            logger.warning(f"⚠️ EnhancedTrainingSystem not available (missing dependencies): {e}")
            self.skipTest("EnhancedTrainingSystem requires additional dependencies")
        except Exception as e:
            self.fail(f"Failed to instantiate EnhancedTrainingSystem: {e}")
    
    def test_simple_training_import(self):
        """Test that SimpleEnhancedTrainingSystem can be imported."""
        try:
            from tests.training.simple_enhanced_training import SimpleEnhancedTrainingSystem
            logger.info("✅ SimpleEnhancedTrainingSystem imported successfully")
            
            # Test instantiation
            system = SimpleEnhancedTrainingSystem()
            self.assertIsNotNone(system)
            logger.info("✅ SimpleEnhancedTrainingSystem instantiated successfully")
            
        except ImportError as e:
            self.fail(f"Failed to import SimpleEnhancedTrainingSystem: {e}")
        except Exception as e:
            self.fail(f"Failed to instantiate SimpleEnhancedTrainingSystem: {e}")
    
    def test_training_module_import(self):
        """Test that the training module can be imported as a whole."""
        try:
            from tests.training import EnhancedTrainingSystem, SimpleEnhancedTrainingSystem
            logger.info("✅ Training module imported successfully")
            
            # Test simple system (should always be available)
            simple = SimpleEnhancedTrainingSystem()
            self.assertIsNotNone(simple)
            logger.info("✅ SimpleEnhancedTrainingSystem instantiated successfully")
            
            # Test enhanced system (may not be available)
            if EnhancedTrainingSystem is not None:
                enhanced = EnhancedTrainingSystem()
                self.assertIsNotNone(enhanced)
                logger.info("✅ EnhancedTrainingSystem instantiated successfully")
            else:
                logger.warning("⚠️ EnhancedTrainingSystem not available (missing dependencies)")
            
        except ImportError as e:
            self.fail(f"Failed to import training module: {e}")
        except Exception as e:
            self.fail(f"Failed to instantiate training systems: {e}")
    
    def test_checkpoint_functionality(self):
        """Test basic checkpoint functionality."""
        try:
            from tests.training.simple_enhanced_training import SimpleEnhancedTrainingSystem
            
            system = SimpleEnhancedTrainingSystem()
            
            # Test checkpoint operations
            success = system.save_checkpoint("v8n", 0, "test")
            self.assertTrue(success)
            
            checkpoint = system.load_checkpoint()
            self.assertIsNotNone(checkpoint)
            self.assertEqual(checkpoint.get('model_version'), 'v8n')
            
            success = system.clear_checkpoint()
            self.assertTrue(success)
            
            logger.info("✅ Checkpoint functionality works correctly")
            
        except Exception as e:
            self.fail(f"Checkpoint functionality test failed: {e}")


def main():
    """Run the training import tests."""
    logger.info("🧪 Testing training module imports and functionality...")
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTrainingImports)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    if result.wasSuccessful():
        logger.info("🎉 All training import tests passed!")
        return 0
    else:
        logger.error("💥 Some training import tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 