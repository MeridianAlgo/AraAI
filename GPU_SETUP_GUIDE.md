# 🚀 Multi-GPU Acceleration Setup Guide for Smart Trader
## AMD • Intel • NVIDIA • Apple Silicon Support

## Current Status
Your system is currently using **CPU with 8 threads** for training. To enable GPU acceleration for much faster training, follow this guide for your specific GPU vendor.

## 🎯 Why Use GPU?
- **2-100x faster training** for neural networks
- **Larger batch sizes** for better model performance
- **More epochs** in the same time
- **Better ensemble performance** with complex models

## 🔧 Multi-Vendor GPU Setup Options

### Option 1: AMD GPU (Recommended for AMD Users) 🔴
**Best for**: AMD Radeon RX 6000/7000 series, AMD Instinct, etc.

#### Windows (DirectML - Easiest):
1. **Install PyTorch with DirectML:**
   ```bash
   pip install torch-directml
   pip install torch torchvision torchaudio
   ```

2. **Verify AMD GPU detection:**
   ```bash
   python -c "import torch_directml; print('DirectML device:', torch_directml.device())"
   ```

#### Linux (ROCm - Best Performance):
1. **Check AMD GPU compatibility:**
   ```bash
   rocm-smi
   # or
   lspci | grep -i amd
   ```

2. **Install ROCm:**
   ```bash
   # Ubuntu/Debian
   wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
   echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/debian/ ubuntu main' | sudo tee /etc/apt/sources.list.d/rocm.list
   sudo apt update
   sudo apt install rocm-dkms
   ```

3. **Install PyTorch with ROCm:**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6
   ```

4. **Verify ROCm installation:**
   ```bash
   python -c "import torch; print('ROCm available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count())"
   ```

### Option 2: NVIDIA GPU (CUDA) 🟢
**Best for**: GeForce RTX, GTX, Quadro, Tesla series

1. **Check GPU compatibility:**
   ```bash
   nvidia-smi
   ```

2. **Install CUDA Toolkit:**
   - Download from: https://developer.nvidia.com/cuda-downloads
   - Choose your OS and follow installation instructions

3. **Install PyTorch with CUDA:**
   ```bash
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

4. **Verify installation:**
   ```bash
   python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
   ```

### Option 3: Intel Arc GPU (XPU) 🔵
**Best for**: Intel Arc A-series, Intel Data Center GPU

1. **Install Intel Extension for PyTorch:**
   ```bash
   pip install intel-extension-for-pytorch
   pip install torch torchvision torchaudio
   ```

2. **Verify Intel GPU:**
   ```bash
   python -c "import intel_extension_for_pytorch as ipex; print('Intel XPU available:', ipex.xpu.is_available() if hasattr(ipex, 'xpu') else False)"
   ```

### Option 4: Apple Silicon (M1/M2/M3/M4) 🍎
**Best for**: MacBook Pro/Air, Mac Studio, Mac Pro with Apple Silicon

1. **Install PyTorch with MPS support:**
   ```bash
   pip install torch torchvision torchaudio
   ```

2. **Verify MPS availability:**
   ```bash
   python -c "import torch; print('MPS available:', torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False)"
   ```

## 🔴 **DETECTED: AMD Radeon RX 7600 XT** 
**Your GPU has 16GB VRAM - Perfect for ML acceleration!**

## 🚀 AMD RX 7600 XT Setup Options (Ranked by Performance)

### 🥇 Option A: Python 3.11 + ROCm (Best Performance)
**Recommended for maximum speed**

1. **Install Python 3.11:**
   ```bash
   # Download Python 3.11 from python.org
   # Create new environment: python3.11 -m venv amd_env
   # Activate: amd_env\Scripts\activate
   ```

2. **Install PyTorch with ROCm:**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6
   ```

### 🥈 Option B: OpenCL Backend (Current Python 3.13)
**Works with your current setup**

1. **Install PyOpenCL:**
   ```bash
   pip install pyopencl
   pip install plaidml-keras plaidml
   ```

2. **Configure PlaidML:**
   ```bash
   plaidml-setup
   # Select your AMD GPU when prompted
   ```

### 🥉 Option C: CPU Optimization (Current - Already Working)
**Your current setup is already optimized**

- ✅ Using 8 CPU threads
- ✅ Optimized batch sizes
- ✅ Multi-threaded processing

## 🚀 Expected Performance Improvements

### Current CPU Performance (RX 7600 XT System):
- **Training Time**: ~1-2 seconds for 5 epochs
- **Batch Size**: 32
- **Memory Usage**: ~2-4 GB RAM
- **Status**: ✅ Already very fast!

### With AMD GPU Acceleration:
- **Training Time**: ~0.3-0.8 seconds for 5 epochs (2-4x faster)
- **Batch Size**: 64-128 (better model performance)
- **Memory Usage**: 16GB GPU VRAM + reduced RAM usage
- **Larger Models**: Can train more complex ensembles

## 🔍 Check Your Current Setup

Run this command to see detailed hardware information:
```bash
python -c "
import torch
import platform
print('System:', platform.system(), platform.release())
print('Python:', platform.python_version())
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA version:', torch.version.cuda)
    print('GPU count:', torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
        print(f'Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB')
print('MPS available:', torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False)
print('CPU threads:', torch.get_num_threads())
"
```

## 🎯 Smart Trader GPU Features

When GPU is detected, Smart Trader automatically:

1. **🚀 Enables GPU optimizations:**
   - CUDA benchmark mode for consistent performance
   - Optimized batch sizes (64 vs 32 for CPU)
   - Higher learning rates for faster convergence
   - GPU memory management

2. **📊 Shows enhanced device info:**
   - GPU name and memory
   - CUDA/MPS version
   - Optimization status

3. **⚡ Performance improvements:**
   - Faster ensemble training
   - Larger models possible
   - More sophisticated features

## 🔧 Troubleshooting

### Common Issues:

1. **"CUDA out of memory"**
   - Reduce batch size in the code
   - Use fewer epochs
   - Close other GPU applications

2. **"CUDA not available" after installation**
   - Restart terminal/IDE
   - Check CUDA version compatibility
   - Verify GPU drivers

3. **Slow performance with GPU**
   - Check if other applications are using GPU
   - Verify CUDA version matches PyTorch
   - Monitor GPU utilization with `nvidia-smi`

## 🎯 Next Steps

1. **Install GPU support** using the appropriate option above
2. **Restart your terminal/IDE**
3. **Run Smart Trader** - it will automatically detect and use GPU
4. **Enjoy 2-10x faster training!** 🚀

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Run the hardware check command
3. Verify your GPU is compatible with the chosen option

---

**Current Status**: Using CPU with 8 threads (optimized for multi-core performance)
**Recommended**: Install GPU support for significant speed improvements!