"""
MeridianAlgo Smart Trader GPU Module
Universal GPU Support: AMD • Intel • NVIDIA • Apple Silicon
"""

import torch
from typing import Dict, Tuple

def detect_all_gpus() -> Dict:
    """
    Detect ALL available GPUs from all vendors
    
    Returns:
        Dict with GPU availability information
    """
    gpu_info = {
        'nvidia': False,
        'amd': False, 
        'intel': False,
        'apple': False,
        'details': []
    }
    
    # Check NVIDIA CUDA
    if torch.cuda.is_available():
        gpu_info['nvidia'] = True
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            gpu_info['details'].append(f"NVIDIA {gpu_name} ({gpu_memory:.1f} GB)")
    
    # Check AMD ROCm/DirectML
    try:
        import torch_directml
        gpu_info['amd'] = True
        gpu_info['details'].append("AMD GPU (DirectML)")
    except ImportError:
        try:
            if hasattr(torch.version, 'hip') and torch.version.hip is not None:
                gpu_info['amd'] = True
                gpu_info['details'].append("AMD GPU (ROCm)")
        except:
            pass
    
    # Check Intel XPU
    try:
        import intel_extension_for_pytorch as ipex
        if hasattr(ipex, 'xpu') and ipex.xpu.is_available():
            gpu_info['intel'] = True
            gpu_info['details'].append("Intel Arc GPU (XPU)")
    except ImportError:
        pass
    
    # Check Apple Silicon MPS
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        gpu_info['apple'] = True
        gpu_info['details'].append("Apple Silicon MPS")
    
    return gpu_info

def get_best_device() -> Tuple[torch.device, str]:
    """
    Get the best available device with universal GPU support
    
    Priority: NVIDIA CUDA > AMD ROCm/DirectML > Intel XPU > Apple MPS > CPU
    
    Returns:
        Tuple of (device, device_name)
    """
    gpu_info = detect_all_gpus()
    
    # 1. NVIDIA CUDA (best ML performance)
    if gpu_info['nvidia'] and torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🟢 Using NVIDIA GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        return device, f"NVIDIA {gpu_name} ({gpu_memory:.1f} GB)"
    
    # 2. AMD GPU (DirectML/ROCm)
    elif gpu_info['amd']:
        try:
            import torch_directml
            device = torch_directml.device()
            print("🔴 Using AMD GPU with DirectML")
            return device, "AMD GPU (DirectML)"
        except ImportError:
            try:
                device = torch.device('cuda')  # ROCm uses CUDA API
                print("🔴 Using AMD GPU with ROCm")
                return device, "AMD GPU (ROCm)"
            except:
                pass
    
    # 3. Intel Arc GPU
    elif gpu_info['intel']:
        try:
            import intel_extension_for_pytorch as ipex
            device = ipex.xpu.device()
            print("🔵 Using Intel Arc GPU")
            return device, "Intel Arc GPU (XPU)"
        except:
            pass
    
    # 4. Apple Silicon MPS
    elif gpu_info['apple']:
        device = torch.device('mps')
        print("🍎 Using Apple Silicon MPS")
        return device, "Apple MPS GPU"
    
    # 5. Optimized CPU
    else:
        torch.set_num_threads(torch.get_num_threads())
        device = torch.device('cpu')
        cpu_count = torch.get_num_threads()
        print(f"💻 Using CPU with {cpu_count} threads")
        return device, f"CPU ({cpu_count} threads)"

def optimize_for_device(device: torch.device) -> None:
    """
    Optimize PyTorch settings based on device type
    
    Args:
        device: PyTorch device to optimize for
    """
    device_str = str(device)
    
    if device.type == 'cuda':
        # NVIDIA CUDA or AMD ROCm optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
            
    elif 'directml' in device_str.lower():
        # AMD DirectML optimizations (Windows)
        pass  # DirectML handles optimization internally
        
    elif 'xpu' in device_str.lower():
        # Intel XPU optimizations
        pass  # Intel extension handles optimization
        
    elif device.type == 'mps':
        # Apple Silicon optimizations
        pass  # MPS handles optimization internally
        
    else:
        # CPU optimizations
        torch.set_num_threads(torch.get_num_threads())
        torch.set_num_interop_threads(1)

def get_optimal_batch_size(device_name: str) -> int:
    """
    Get optimal batch size based on device type
    
    Args:
        device_name: Name of the device
        
    Returns:
        Optimal batch size
    """
    if 'GPU' in device_name or 'CUDA' in device_name or 'MPS' in device_name:
        return 64  # Larger batches for GPU
    else:
        return 32  # Smaller batches for CPU

def show_gpu_info() -> None:
    """Display comprehensive GPU information"""
    from rich.console import Console
    from rich.table import Table
    
    console = Console()
    gpu_info = detect_all_gpus()
    
    console.print("\n🚀 Universal GPU Support Status")
    console.print("Works with AMD • Intel • NVIDIA • Apple Silicon\n")
    
    # Create GPU status table
    table = Table(title="GPU Support Matrix")
    table.add_column("Vendor", style="bold")
    table.add_column("Status", style="bold")
    table.add_column("Details")
    
    table.add_row("🟢 NVIDIA", "✅ Supported" if gpu_info['nvidia'] else "❌ Not detected", "CUDA acceleration")
    table.add_row("🔴 AMD", "✅ Supported" if gpu_info['amd'] else "❌ Not detected", "ROCm/DirectML acceleration")
    table.add_row("🔵 Intel", "✅ Supported" if gpu_info['intel'] else "❌ Not detected", "XPU acceleration")
    table.add_row("🍎 Apple", "✅ Supported" if gpu_info['apple'] else "❌ Not detected", "MPS acceleration")
    
    console.print(table)
    
    if gpu_info['details']:
        console.print("\n🔧 Available GPUs:")
        for detail in gpu_info['details']:
            console.print(f"  • {detail}")
    else:
        console.print("\n💻 No GPUs detected - using optimized CPU")
    
    console.print(f"\n⚡ Current Device: {get_best_device()[1]}")
    console.print(f"🧵 CPU Threads: {torch.get_num_threads()}")
    console.print(f"🔥 PyTorch Version: {torch.__version__}")
    
    if not any([gpu_info['nvidia'], gpu_info['amd'], gpu_info['intel'], gpu_info['apple']]):
        console.print("\n📖 For GPU setup instructions:")
        console.print("   • See GPU_SETUP_GUIDE.md")
        console.print("   • Visit: https://github.com/MeridianAlgo/In-Python")
        console.print("🚀 GPU acceleration provides 2-5x speed improvement")