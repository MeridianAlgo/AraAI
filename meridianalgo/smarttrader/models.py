"""
MeridianAlgo Smart Trader Models Module
Ensemble ML models with universal GPU support
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from torch.utils.data import DataLoader, Dataset

from .gpu import optimize_for_device, get_optimal_batch_size

class SimpleStockDataset(Dataset):
    """Simple PyTorch dataset for stock data"""
    
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.features = X
        self.targets = y
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]

def train_ensemble_models(X: np.ndarray, y: np.ndarray, epochs: int, device: torch.device) -> Optional[Dict]:
    """
    Train ensemble models with universal GPU optimization
    
    Args:
        X: Feature array
        y: Target array  
        epochs: Training epochs
        device: PyTorch device
        
    Returns:
        Dict with trained models or None if failed
    """
    try:
        # Try to use the advanced ensemble system
        try:
            import sys
            import os
            sys.path.append('src/python')
            from ensemble_system import EnsemblePredictor
            
            # Universal GPU optimization
            optimize_for_device(device)
            
            input_size = X.shape[1]
            ensemble_predictor = EnsemblePredictor(input_size=input_size)
            
            X_tensor = torch.FloatTensor(X).to(device)
            y_tensor = torch.FloatTensor(y).to(device)
            
            dataset_size = len(X_tensor)
            train_size = int(0.8 * dataset_size)
            
            X_train, X_val = X_tensor[:train_size], X_tensor[train_size:]
            y_train, y_val = y_tensor[:train_size], y_tensor[train_size:]
            
            train_dataset = SimpleStockDataset(X_train, y_train)
            val_dataset = SimpleStockDataset(X_val, y_val)
            
            # Optimize batch size for device
            device_name = str(device)
            batch_size = get_optimal_batch_size(device_name)
            
            train_loader = DataLoader(
                train_dataset, 
                batch_size=batch_size, 
                shuffle=True, 
                num_workers=0
            )
            val_loader = DataLoader(
                val_dataset, 
                batch_size=batch_size, 
                shuffle=False, 
                num_workers=0
            )
            
            training_results = ensemble_predictor.train_ensemble(
                train_loader, val_loader,
                epochs=epochs,
                learning_rate=0.001
            )
            
            # Clear GPU cache if using GPU
            if device.type == 'cuda' and hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            
            return {'ensemble': ensemble_predictor, 'results': training_results}
            
        except ImportError:
            # Fallback to simple neural network
            return train_simple_model(X, y, epochs, device)
            
    except Exception as e:
        print(f"Ensemble training failed: {e}")
        return None

def train_simple_model(X: np.ndarray, y: np.ndarray, epochs: int, device: torch.device) -> Dict:
    """
    Train a simple neural network as fallback
    
    Args:
        X: Feature array
        y: Target array
        epochs: Training epochs
        device: PyTorch device
        
    Returns:
        Dict with trained model
    """
    try:
        # Simple feedforward neural network
        class SimplePredictor(nn.Module):
            def __init__(self, input_size: int):
                super(SimplePredictor, self).__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_size, 64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(32, 1)
                )
            
            def forward(self, x):
                return self.network(x)
        
        input_size = X.shape[1]
        model = SimplePredictor(input_size).to(device)
        
        X_tensor = torch.FloatTensor(X).to(device)
        y_tensor = torch.FloatTensor(y).to(device)
        
        # Split data
        dataset_size = len(X_tensor)
        train_size = int(0.8 * dataset_size)
        
        X_train, X_val = X_tensor[:train_size], X_tensor[train_size:]
        y_train, y_val = y_tensor[:train_size], y_tensor[train_size:]
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_train).squeeze()
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val).squeeze()
            val_loss = criterion(val_outputs, y_val)
        
        return {
            'simple_model': model,
            'train_loss': loss.item(),
            'val_loss': val_loss.item(),
            'success': True
        }
        
    except Exception as e:
        print(f"Simple model training failed: {e}")
        return {'success': False, 'error': str(e)}

def make_accurate_predictions(X: np.ndarray, trained_models: Dict, days: int = 5) -> Optional[List[float]]:
    """
    Make predictions using trained models
    
    Args:
        X: Feature array
        trained_models: Dict with trained models
        days: Number of days to predict
        
    Returns:
        List of predictions or None if failed
    """
    try:
        # Try ensemble predictor first
        if 'ensemble' in trained_models:
            ensemble_predictor = trained_models['ensemble']
            
            last_sequence = X[-1:] if len(X.shape) > 1 else X.reshape(1, -1)
            X_pred = torch.FloatTensor(last_sequence)
            
            predictions = []
            current_input = X_pred
            
            for i in range(days):
                pred_result = ensemble_predictor.predict_with_ensemble(current_input)
                pred_value = pred_result['ensemble_prediction']
                
                if hasattr(pred_value, 'item'):
                    pred_value = pred_value.item()
                elif isinstance(pred_value, np.ndarray):
                    pred_value = pred_value.flatten()[0]
                
                predictions.append(pred_value)
                
                if len(current_input.shape) > 1:
                    new_features = current_input.clone()
                    new_features[0, -1] = pred_value
                    current_input = new_features
            
            return predictions
        
        # Try simple model
        elif 'simple_model' in trained_models:
            model = trained_models['simple_model']
            model.eval()
            
            last_sequence = X[-1:] if len(X.shape) > 1 else X.reshape(1, -1)
            X_pred = torch.FloatTensor(last_sequence)
            
            predictions = []
            current_input = X_pred
            
            with torch.no_grad():
                for i in range(days):
                    pred = model(current_input).squeeze()
                    pred_value = pred.item() if hasattr(pred, 'item') else float(pred)
                    predictions.append(pred_value)
                    
                    # Simple update for next prediction
                    if len(current_input.shape) > 1:
                        new_features = current_input.clone()
                        new_features[0, -1] = pred_value
                        current_input = new_features
            
            return predictions
        
        else:
            return None
            
    except Exception as e:
        print(f"Prediction failed: {e}")
        return None