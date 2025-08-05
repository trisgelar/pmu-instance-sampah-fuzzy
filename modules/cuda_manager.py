# file: modules/cuda_manager.py
import logging
import torch
from typing import Optional, Dict, Any, Tuple
from modules.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

class CUDAManager:
    """
    Manages CUDA/GPU configuration and memory for the waste detection system.
    """
    
    def __init__(self):
        """Initialize CUDA manager."""
        self.device = None
        self.gpu_info = {}
        self._setup_device()
    
    def _setup_device(self) -> None:
        """Setup the best available device (GPU or CPU)."""
        try:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                self._get_gpu_info()
                logger.info(f"Using GPU: {self.gpu_info.get('name', 'Unknown')}")
            else:
                self.device = torch.device('cpu')
                logger.info("CUDA not available, using CPU")
                
        except Exception as e:
            logger.warning(f"Failed to setup CUDA device: {e}")
            self.device = torch.device('cpu')
    
    def _get_gpu_info(self) -> None:
        """Get GPU information."""
        try:
            self.gpu_info = {
                'name': torch.cuda.get_device_name(),
                'memory_total': torch.cuda.get_device_properties(0).total_memory,
                'memory_allocated': torch.cuda.memory_allocated(),
                'memory_cached': torch.cuda.memory_reserved(),
                'device_count': torch.cuda.device_count(),
                'current_device': torch.cuda.current_device()
            }
        except Exception as e:
            logger.warning(f"Failed to get GPU info: {e}")
            self.gpu_info = {}
    
    def get_device(self) -> torch.device:
        """Get the current device."""
        return self.device
    
    def is_cuda_available(self) -> bool:
        """Check if CUDA is available."""
        return torch.cuda.is_available()
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Get current memory information."""
        info = {
            'device': str(self.device),
            'cuda_available': self.is_cuda_available()
        }
        
        if self.is_cuda_available():
            try:
                info.update({
                    'memory_allocated_mb': round(torch.cuda.memory_allocated() / 1024**2, 2),
                    'memory_reserved_mb': round(torch.cuda.memory_reserved() / 1024**2, 2),
                    'memory_total_mb': round(self.gpu_info.get('memory_total', 0) / 1024**2, 2),
                    'gpu_name': self.gpu_info.get('name', 'Unknown')
                })
            except Exception as e:
                logger.warning(f"Failed to get memory info: {e}")
        
        return info
    
    def clear_cache(self) -> None:
        """Clear CUDA cache to free memory."""
        if self.is_cuda_available():
            try:
                torch.cuda.empty_cache()
                logger.info("CUDA cache cleared")
            except Exception as e:
                logger.warning(f"Failed to clear CUDA cache: {e}")
    
    def optimize_for_training(self, batch_size: int, img_size: Tuple[int, int]) -> Dict[str, Any]:
        """
        Optimize CUDA settings for training.
        
        Args:
            batch_size: Training batch size
            img_size: Image size (width, height)
            
        Returns:
            Dict with optimization recommendations
        """
        recommendations = {
            'device': str(self.device),
            'batch_size': batch_size,
            'img_size': img_size,
            'optimizations': []
        }
        
        if not self.is_cuda_available():
            recommendations['optimizations'].append("Use CPU - no GPU optimizations available")
            return recommendations
        
        # Check memory requirements
        estimated_memory = self._estimate_memory_usage(batch_size, img_size)
        available_memory = self.gpu_info.get('memory_total', 0) / 1024**2  # MB
        
        if estimated_memory > available_memory * 0.8:  # Use 80% of available memory
            recommendations['optimizations'].append(
                f"Reduce batch size or image size - estimated {estimated_memory:.1f}MB > {available_memory * 0.8:.1f}MB"
            )
        
        # Enable mixed precision if supported
        if hasattr(torch, 'autocast'):
            recommendations['optimizations'].append("Enable mixed precision training")
        
        # Memory pinning
        recommendations['optimizations'].append("Enable pin_memory=True for DataLoader")
        
        return recommendations
    
    def _estimate_memory_usage(self, batch_size: int, img_size: Tuple[int, int]) -> float:
        """
        Estimate GPU memory usage for training.
        
        Args:
            batch_size: Batch size
            img_size: Image size (width, height)
            
        Returns:
            Estimated memory usage in MB
        """
        # Rough estimation: 4 bytes per pixel * 3 channels * batch_size * image_size
        # Plus model parameters and gradients
        pixel_memory = batch_size * img_size[0] * img_size[1] * 3 * 4  # bytes
        model_memory = 100 * 1024 * 1024  # ~100MB for model
        gradient_memory = model_memory  # gradients are similar size to model
        
        total_bytes = pixel_memory + model_memory + gradient_memory
        return total_bytes / (1024 * 1024)  # Convert to MB
    
    def get_optimal_batch_size(self, img_size: Tuple[int, int], max_memory_usage: float = 0.8) -> int:
        """
        Calculate optimal batch size based on available GPU memory.
        
        Args:
            img_size: Image size (width, height)
            max_memory_usage: Maximum memory usage as fraction of total GPU memory
            
        Returns:
            Optimal batch size
        """
        if not self.is_cuda_available():
            return 1  # CPU fallback
        
        available_memory = self.gpu_info.get('memory_total', 0) / 1024**2  # MB
        max_memory = available_memory * max_memory_usage
        
        # Start with batch size 1 and increase until we exceed memory
        batch_size = 1
        while batch_size <= 128:  # Maximum reasonable batch size
            estimated_memory = self._estimate_memory_usage(batch_size, img_size)
            if estimated_memory > max_memory:
                break
            batch_size *= 2
        
        return max(1, batch_size // 2)  # Return the last working batch size
    
    def monitor_memory(self) -> Dict[str, Any]:
        """Monitor current GPU memory usage."""
        if not self.is_cuda_available():
            return {'status': 'No GPU available'}
        
        try:
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            total = self.gpu_info.get('memory_total', 0) / 1024**2
            
            return {
                'allocated_mb': round(allocated, 2),
                'reserved_mb': round(reserved, 2),
                'total_mb': round(total, 2),
                'usage_percent': round((allocated / total) * 100, 2) if total > 0 else 0
            }
        except Exception as e:
            logger.warning(f"Failed to monitor memory: {e}")
            return {'status': 'Failed to get memory info'}
    
    def __str__(self) -> str:
        """String representation of CUDA manager."""
        info = self.get_memory_info()
        return f"CUDAManager(device={info['device']}, cuda_available={info['cuda_available']})" 