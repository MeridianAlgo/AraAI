"""
Performance Optimization Utilities
GPU acceleration, model quantization, ONNX export, and batch processing
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ara.core.exceptions import ModelError
from ara.utils import get_logger

logger = get_logger(__name__)


class GPUAccelerator:
    """
    GPU acceleration utilities for deep learning models
    Supports CUDA, ROCm, and MPS (Apple Silicon)
    """

    def __init__(self):
        """Initialize GPU accelerator"""
        self.device = self._detect_device()
        self.device_name = self._get_device_name()
        self.is_available = self.device != "cpu"

        logger.info(
            "GPU Accelerator initialized",
            device=self.device,
            device_name=self.device_name,
            available=self.is_available,
        )

    def _detect_device(self) -> str:
        """
        Detect available GPU device

        Returns:
            Device string ('cuda', 'mps', 'cpu')
        """
        try:
            import torch

            # Check for CUDA (NVIDIA)
            if torch.cuda.is_available():
                return "cuda"

            # Check for MPS (Apple Silicon)
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"

            # Check for ROCm (AMD) - uses same API as CUDA
            if hasattr(torch, "hip") and torch.hip.is_available():
                return "cuda"  # ROCm uses CUDA API

        except ImportError:
            logger.warning("PyTorch not available, GPU acceleration disabled")

        return "cpu"

    def _get_device_name(self) -> str:
        """Get GPU device name"""
        try:
            import torch

            if self.device == "cuda":
                return torch.cuda.get_device_name(0)
            elif self.device == "mps":
                return "Apple Silicon GPU"

        except Exception as e:
            logger.debug(f"Could not get device name: {e}")

        return "CPU"

    def get_device(self):
        """
        Get PyTorch device object

        Returns:
            torch.device object
        """
        try:
            import torch

            return torch.device(self.device)
        except ImportError:
            raise ModelError("PyTorch not installed")

    def move_to_device(self, model):
        """
        Move model to GPU device

        Args:
            model: PyTorch model

        Returns:
            Model on device
        """
        try:
            device = self.get_device()
            model = model.to(device)
            logger.info(f"Moved model to {self.device}")
            return model
        except Exception as e:
            logger.warning(f"Failed to move model to device: {e}")
            return model

    def enable_mixed_precision(self) -> bool:
        """
        Enable mixed precision training (FP16)

        Returns:
            True if enabled successfully
        """
        if self.device == "cuda":
            try:
                import torch

                # Check if AMP is available
                if hasattr(torch.cuda.amp, "autocast"):
                    logger.info("Mixed precision (FP16) enabled")
                    return True
            except Exception as e:
                logger.warning(f"Failed to enable mixed precision: {e}")

        return False

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get GPU memory statistics

        Returns:
            Dict with memory stats
        """
        stats = {"device": self.device, "available": self.is_available}

        if self.device == "cuda":
            try:
                import torch

                stats["allocated_mb"] = torch.cuda.memory_allocated() / (1024**2)
                stats["reserved_mb"] = torch.cuda.memory_reserved() / (1024**2)
                stats["max_allocated_mb"] = torch.cuda.max_memory_allocated() / (1024**2)
            except Exception as e:
                logger.debug(f"Could not get memory stats: {e}")

        return stats

    def clear_cache(self) -> None:
        """Clear GPU cache"""
        if self.device == "cuda":
            try:
                import torch

                torch.cuda.empty_cache()
                logger.info("Cleared GPU cache")
            except Exception as e:
                logger.warning(f"Failed to clear GPU cache: {e}")


class ModelQuantizer:
    """
    Model quantization utilities for reducing model size and improving inference speed
    Supports FP32 → FP16 and FP32 → INT8 quantization
    """

    @staticmethod
    def quantize_to_fp16(model, inplace: bool = False):
        """
        Quantize model to FP16 (half precision)

        Args:
            model: PyTorch model
            inplace: Modify model in place

        Returns:
            Quantized model
        """
        try:
            if not inplace:
                model = model.clone() if hasattr(model, "clone") else model

            model = model.half()

            logger.info("Quantized model to FP16")
            return model

        except Exception as e:
            logger.error(f"FP16 quantization failed: {e}")
            raise ModelError(f"Quantization failed: {e}")

    @staticmethod
    def quantize_to_int8(model, calibration_data: Optional[np.ndarray] = None):
        """
        Quantize model to INT8

        Args:
            model: PyTorch model
            calibration_data: Data for calibration (optional)

        Returns:
            Quantized model
        """
        try:
            import torch
            from torch.quantization import quantize_dynamic

            # Dynamic quantization (no calibration needed)
            quantized_model = quantize_dynamic(
                model, {torch.nn.Linear, torch.nn.LSTM, torch.nn.GRU}, dtype=torch.qint8
            )

            logger.info("Quantized model to INT8")
            return quantized_model

        except Exception as e:
            logger.error(f"INT8 quantization failed: {e}")
            raise ModelError(f"Quantization failed: {e}")

    @staticmethod
    def measure_model_size(model) -> Dict[str, float]:
        """
        Measure model size

        Args:
            model: PyTorch model

        Returns:
            Dict with size metrics
        """
        try:
            # Count parameters
            param_count = sum(p.numel() for p in model.parameters())

            # Estimate size in MB
            param_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (
                1024**2
            )

            buffer_size_mb = sum(b.numel() * b.element_size() for b in model.buffers()) / (1024**2)

            total_size_mb = param_size_mb + buffer_size_mb

            return {
                "parameter_count": param_count,
                "parameter_size_mb": param_size_mb,
                "buffer_size_mb": buffer_size_mb,
                "total_size_mb": total_size_mb,
            }

        except Exception as e:
            logger.error(f"Failed to measure model size: {e}")
            return {}


class ONNXExporter:
    """
    ONNX export utilities for cross-platform model optimization
    """

    @staticmethod
    def export_to_onnx(
        model,
        input_shape: Tuple[int, ...],
        output_path: Path,
        opset_version: int = 14,
        dynamic_axes: Optional[Dict] = None,
    ) -> bool:
        """
        Export PyTorch model to ONNX format

        Args:
            model: PyTorch model
            input_shape: Input tensor shape
            output_path: Path to save ONNX model
            opset_version: ONNX opset version
            dynamic_axes: Dynamic axes configuration

        Returns:
            True if successful
        """
        try:
            import torch

            # Create dummy input
            dummy_input = torch.randn(*input_shape)

            # Move to same device as model
            device = next(model.parameters()).device
            dummy_input = dummy_input.to(device)

            # Set model to eval mode
            model.eval()

            # Export
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            torch.onnx.export(
                model,
                dummy_input,
                str(output_path),
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes=dynamic_axes
                or {"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            )

            logger.info(f"Exported model to ONNX: {output_path}")
            return True

        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            return False

    @staticmethod
    def load_onnx_model(model_path: Path):
        """
        Load ONNX model for inference

        Args:
            model_path: Path to ONNX model

        Returns:
            ONNX inference session
        """
        try:
            import onnxruntime as ort

            # Create inference session
            session = ort.InferenceSession(
                str(model_path),
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )

            logger.info(f"Loaded ONNX model: {model_path}")
            return session

        except ImportError:
            logger.error("onnxruntime not installed. Install with: pip install onnxruntime")
            raise ModelError("onnxruntime not available")
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            raise ModelError(f"ONNX load failed: {e}")

    @staticmethod
    def onnx_inference(session, input_data: np.ndarray) -> np.ndarray:
        """
        Run inference with ONNX model

        Args:
            session: ONNX inference session
            input_data: Input numpy array

        Returns:
            Output numpy array
        """
        try:
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name

            result = session.run([output_name], {input_name: input_data.astype(np.float32)})

            return result[0]

        except Exception as e:
            logger.error(f"ONNX inference failed: {e}")
            raise ModelError(f"Inference failed: {e}")


class BatchProcessor:
    """
    Batch processing optimization for predictions
    """

    def __init__(self, batch_size: int = 32, max_queue_size: int = 1000):
        """
        Initialize batch processor

        Args:
            batch_size: Batch size for processing
            max_queue_size: Maximum queue size
        """
        self.batch_size = batch_size
        self.max_queue_size = max_queue_size
        self._stats = {"batches_processed": 0, "items_processed": 0, "total_time": 0.0}

        logger.info(f"Initialized BatchProcessor (batch_size={batch_size})")

    def process_batch(self, items: List[Any], process_func: callable) -> List[Any]:
        """
        Process items in batches

        Args:
            items: List of items to process
            process_func: Function to process each batch

        Returns:
            List of results
        """
        results = []
        start_time = time.time()

        # Process in batches
        for i in range(0, len(items), self.batch_size):
            batch = items[i : i + self.batch_size]

            try:
                batch_results = process_func(batch)
                results.extend(batch_results)

                self._stats["batches_processed"] += 1
                self._stats["items_processed"] += len(batch)

            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                # Add None for failed items
                results.extend([None] * len(batch))

        elapsed = time.time() - start_time
        self._stats["total_time"] += elapsed

        logger.debug(
            f"Processed {len(items)} items in {elapsed:.2f}s ({len(items) / elapsed:.1f} items/s)"
        )

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get batch processing statistics"""
        avg_time_per_batch = (
            self._stats["total_time"] / self._stats["batches_processed"]
            if self._stats["batches_processed"] > 0
            else 0.0
        )

        avg_time_per_item = (
            self._stats["total_time"] / self._stats["items_processed"]
            if self._stats["items_processed"] > 0
            else 0.0
        )

        return {
            "batch_size": self.batch_size,
            "batches_processed": self._stats["batches_processed"],
            "items_processed": self._stats["items_processed"],
            "total_time": self._stats["total_time"],
            "avg_time_per_batch": avg_time_per_batch,
            "avg_time_per_item": avg_time_per_item,
            "throughput": 1.0 / avg_time_per_item if avg_time_per_item > 0 else 0.0,
        }


class ParallelFeatureCalculator:
    """
    Parallel processing for feature calculation
    """

    def __init__(self, n_workers: Optional[int] = None):
        """
        Initialize parallel feature calculator

        Args:
            n_workers: Number of worker processes (None = CPU count)
        """
        import multiprocessing as mp

        self.n_workers = n_workers or mp.cpu_count()
        logger.info(f"Initialized ParallelFeatureCalculator (workers={self.n_workers})")

    def calculate_features_parallel(
        self, data_list: List[Any], feature_func: callable
    ) -> List[Any]:
        """
        Calculate features in parallel

        Args:
            data_list: List of data items
            feature_func: Function to calculate features

        Returns:
            List of feature results
        """
        from multiprocessing import Pool

        start_time = time.time()

        try:
            with Pool(processes=self.n_workers) as pool:
                results = pool.map(feature_func, data_list)

            elapsed = time.time() - start_time
            logger.info(
                f"Calculated features for {len(data_list)} items in {elapsed:.2f}s "
                f"({len(data_list) / elapsed:.1f} items/s)"
            )

            return results

        except Exception as e:
            logger.error(f"Parallel feature calculation failed: {e}")
            # Fallback to sequential processing
            logger.info("Falling back to sequential processing")
            return [feature_func(data) for data in data_list]


class PerformanceProfiler:
    """
    Performance profiling utilities
    """

    def __init__(self):
        """Initialize performance profiler"""
        self._timings: Dict[str, List[float]] = {}

    def time_function(self, name: str):
        """
        Decorator to time function execution

        Args:
            name: Function name for tracking
        """

        def decorator(func):
            def wrapper(*args, **kwargs):
                start = time.time()
                result = func(*args, **kwargs)
                elapsed = time.time() - start

                if name not in self._timings:
                    self._timings[name] = []
                self._timings[name].append(elapsed)

                return result

            return wrapper

        return decorator

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get timing statistics"""
        stats = {}

        for name, timings in self._timings.items():
            stats[name] = {
                "count": len(timings),
                "total": sum(timings),
                "mean": np.mean(timings),
                "median": np.median(timings),
                "min": min(timings),
                "max": max(timings),
                "std": np.std(timings),
            }

        return stats

    def print_stats(self) -> None:
        """Print timing statistics"""
        stats = self.get_stats()

        print("\n=== Performance Profile ===")
        for name, metrics in sorted(stats.items(), key=lambda x: x[1]["total"], reverse=True):
            print(f"\n{name}:")
            print(f"  Count: {metrics['count']}")
            print(f"  Total: {metrics['total']:.3f}s")
            print(f"  Mean: {metrics['mean']:.3f}s")
            print(f"  Median: {metrics['median']:.3f}s")
            print(f"  Min: {metrics['min']:.3f}s")
            print(f"  Max: {metrics['max']:.3f}s")
