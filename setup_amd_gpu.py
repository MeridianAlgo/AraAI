#!/usr/bin/env python3
"""
AMD GPU Setup Script for Smart Trader
Specifically for AMD Radeon RX 7600 XT and similar GPUs
"""

import subprocess
import sys
import os
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

def detect_amd_gpu():
    """Detect AMD GPU on Windows"""
    try:
        result = subprocess.run(['wmic', 'path', 'win32_VideoController', 'get', 'name'], 
                              capture_output=True, text=True)
        gpu_info = result.stdout
        
        amd_gpus = []
        for line in gpu_info.split('\n'):
            if 'AMD' in line or 'Radeon' in line:
                amd_gpus.append(line.strip())
        
        return amd_gpus
    except:
        return []

def check_current_pytorch():
    """Check current PyTorch installation"""
    try:
        import torch
        return torch.__version__, torch.cuda.is_available()
    except ImportError:
        return None, False

def install_amd_support():
    """Install AMD GPU support for PyTorch"""
    console.print("\n[bold cyan]🔴 Installing AMD GPU Support...[/]")
    
    # Option 1: DirectML (Windows - Easiest)
    console.print("[yellow]Installing PyTorch with DirectML support...[/]")
    
    commands = [
        "pip install torch-directml",
        "pip install torch torchvision torchaudio"
    ]
    
    for cmd in commands:
        console.print(f"Running: {cmd}")
        try:
            result = subprocess.run(cmd.split(), check=True, capture_output=True, text=True)
            console.print("[green]✅ Success[/]")
        except subprocess.CalledProcessError as e:
            console.print(f"[red]❌ Error: {e}[/]")
            return False
    
    return True

def test_amd_gpu():
    """Test AMD GPU functionality"""
    console.print("\n[bold cyan]🧪 Testing AMD GPU...[/]")
    
    try:
        import torch_directml
        device = torch_directml.device()
        console.print(f"[green]✅ DirectML device: {device}[/]")
        
        # Test tensor operations
        x = torch.randn(1000, 1000).to(device)
        y = torch.randn(1000, 1000).to(device)
        z = torch.mm(x, y)
        console.print("[green]✅ Matrix multiplication test passed[/]")
        
        return True
    except Exception as e:
        console.print(f"[red]❌ AMD GPU test failed: {e}[/]")
        return False

def main():
    console.print(Panel.fit(
        "[bold cyan]🔴 AMD GPU Setup for Smart Trader[/]\n"
        "[dim]Optimized for AMD Radeon RX 7600 XT[/]",
        border_style="cyan"
    ))
    
    # Detect AMD GPU
    amd_gpus = detect_amd_gpu()
    if amd_gpus:
        console.print("\n[bold green]🔴 AMD GPU Detected:[/]")
        for gpu in amd_gpus:
            console.print(f"  • {gpu}")
    else:
        console.print("\n[bold red]❌ No AMD GPU detected[/]")
        return
    
    # Check current PyTorch
    pytorch_version, cuda_available = check_current_pytorch()
    if pytorch_version:
        console.print(f"\n[bold blue]Current PyTorch: {pytorch_version}[/]")
        console.print(f"CUDA Available: {cuda_available}")
    else:
        console.print("\n[bold yellow]PyTorch not installed[/]")
    
    # Show setup options
    table = Table(title="AMD GPU Setup Options")
    table.add_column("Option", style="cyan")
    table.add_column("Platform", style="green")
    table.add_column("Performance", style="yellow")
    table.add_column("Difficulty", style="red")
    
    table.add_row("DirectML", "Windows", "Good", "Easy")
    table.add_row("ROCm", "Linux", "Excellent", "Advanced")
    
    console.print(table)
    
    # Ask user for installation
    response = input("\n🔴 Install AMD GPU support with DirectML? (y/n): ")
    if response.lower() == 'y':
        if install_amd_support():
            console.print("\n[bold green]🎉 Installation completed![/]")
            
            # Test the installation
            if test_amd_gpu():
                console.print("\n[bold green]🚀 AMD GPU is ready for Smart Trader![/]")
                console.print("\nNext steps:")
                console.print("1. Restart your terminal")
                console.print("2. Run: python smart_trader.py AAPL")
                console.print("3. Enjoy 2-5x faster training! 🔴⚡")
            else:
                console.print("\n[bold yellow]⚠️ Installation completed but testing failed[/]")
                console.print("Try restarting your terminal and running Smart Trader")
        else:
            console.print("\n[bold red]❌ Installation failed[/]")
    else:
        console.print("\n[bold blue]📖 Manual setup instructions:[/]")
        console.print("1. pip install torch-directml")
        console.print("2. pip install torch torchvision torchaudio")
        console.print("3. Restart terminal and run Smart Trader")

if __name__ == "__main__":
    main()