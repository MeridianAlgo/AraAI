"""
Utility classes for GPU management, caching, and accuracy tracking
"""

import torch
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class GPUManager:
    """Enhanced GPU manager with multi-vendor support"""
    
    def __init__(self):
        self.device = None
        self.device_name = None
        self._initialize_device()
    
    def _initialize_device(self):
        """Initialize the best available device"""
        self.device, self.device_name = self.get_best_device()
    
    def detect_gpu_vendor(self):
        """Detect available GPU vendors and capabilities"""
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
    
    def get_best_device(self):
        """Get the best available device for computation"""
        gpu_info = self.detect_gpu_vendor()
        
        # Priority: NVIDIA CUDA > AMD > Intel XPU > Apple MPS > CPU
        
        if gpu_info['nvidia'] and torch.cuda.is_available():
            device = torch.device('cuda')
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return device, f"NVIDIA {gpu_name} ({gpu_memory:.1f} GB)"
        
        elif gpu_info['amd']:
            try:
                import torch_directml
                device = torch_directml.device()
                return device, "AMD GPU (DirectML)"
            except ImportError:
                try:
                    device = torch.device('cuda')  # ROCm uses CUDA API
                    return device, "AMD GPU (ROCm)"
                except:
                    pass
        
        elif gpu_info['intel']:
            try:
                import intel_extension_for_pytorch as ipex
                device = ipex.xpu.device()
                return device, "Intel Arc GPU (XPU)"
            except:
                pass
        
        elif gpu_info['apple']:
            device = torch.device('mps')
            return device, "Apple MPS GPU"
        
        else:
            torch.set_num_threads(torch.get_num_threads())
            device = torch.device('cpu')
            cpu_count = torch.get_num_threads()
            return device, f"CPU ({cpu_count} threads)"
    
    def get_device(self):
        """Get the current device"""
        return self.device
    
    def get_device_name(self):
        """Get the current device name"""
        return self.device_name

class CacheManager:
    """Enhanced cache manager for intelligent prediction caching"""
    
    def __init__(self):
        self.cache_file = 'predictions.csv'
        self.cache_timeout = 3600  # 1 hour
    
    def check_cached_predictions(self, symbol, days=5):
        """Check for existing cached predictions"""
        try:
            if not os.path.exists(self.cache_file):
                return None
            
            df = pd.read_csv(self.cache_file)
            if df.empty:
                return None
            
            # Find predictions for this symbol from today
            today = datetime.now().date()
            symbol_predictions = df[
                (df['Symbol'] == symbol) & 
                (pd.to_datetime(df['Timestamp']).dt.date == today)
            ]
            
            if symbol_predictions.empty:
                return None
            
            # Convert to standard format
            predictions = []
            for _, row in symbol_predictions.iterrows():
                pred_date = datetime.fromisoformat(row['Date'])
                predictions.append({
                    'day': (pred_date.date() - today).days + 1,
                    'date': row['Date'],
                    'predicted_price': row['Predicted_Price'],
                    'change': row.get('Change', 0),
                    'change_pct': row.get('Change_Pct', 0)
                })
            
            return {
                'symbol': symbol,
                'predictions': predictions,
                'timestamp': symbol_predictions.iloc[0]['Timestamp'],
                'current_price': symbol_predictions.iloc[0].get('Current_Price', 0)
            }
            
        except Exception as e:
            print(f"Error checking cached predictions: {e}")
            return None
    
    def save_predictions(self, symbol, prediction_result):
        """Save predictions to cache"""
        try:
            # Prepare data for CSV
            rows = []
            current_price = prediction_result.get('current_price', 0)
            timestamp = prediction_result.get('timestamp', datetime.now().isoformat())
            
            for pred in prediction_result.get('predictions', []):
                rows.append({
                    'Symbol': symbol,
                    'Date': pred['date'],
                    'Predicted_Price': pred['predicted_price'],
                    'Current_Price': current_price,
                    'Change': pred.get('change', 0),
                    'Change_Pct': pred.get('change_pct', 0),
                    'Timestamp': timestamp
                })
            
            # Save to CSV
            new_df = pd.DataFrame(rows)
            
            if os.path.exists(self.cache_file):
                existing_df = pd.read_csv(self.cache_file)
                # Remove old predictions for this symbol from today
                today = datetime.now().date()
                existing_df = existing_df[
                    ~((existing_df['Symbol'] == symbol) & 
                      (pd.to_datetime(existing_df['Timestamp']).dt.date == today))
                ]
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            else:
                combined_df = new_df
            
            combined_df.to_csv(self.cache_file, index=False)
            
        except Exception as e:
            print(f"Error saving predictions: {e}")
    
    def ask_user_choice(self, symbol, cached_data):
        """Ask user whether to use cached predictions or generate new ones"""
        try:
            print(f"\n🔍 Found existing predictions for {symbol} from today!")
            print("\nCached Predictions:")
            
            for pred in cached_data['predictions']:
                print(f"Day {pred['day']}: ${pred['predicted_price']:.2f} ({pred['change_pct']:+.1f}%)")
            
            cache_age = self._calculate_cache_age(cached_data['timestamp'])
            print(f"\nCache age: {cache_age}")
            
            print("\nOptions:")
            print("1. Use cached predictions (faster)")
            print("2. Generate new predictions (fresh analysis)")
            
            while True:
                try:
                    choice = input("\nEnter your choice (1-2): ").strip()
                    if choice == "1":
                        return "use_cached"
                    elif choice == "2":
                        return "generate_new"
                    else:
                        print("Invalid choice. Please enter 1 or 2.")
                except KeyboardInterrupt:
                    return "use_cached"
                except:
                    print("Invalid input. Please try again.")
                    
        except Exception as e:
            print(f"Error in user choice dialog: {e}")
            return "generate_new"
    
    def _calculate_cache_age(self, timestamp_str):
        """Calculate age of cached data"""
        try:
            if not timestamp_str:
                return "Unknown"
            
            timestamp = datetime.fromisoformat(timestamp_str)
            age = datetime.now() - timestamp
            
            if age.days > 0:
                return f"{age.days}d {age.seconds // 3600}h"
            else:
                return f"{age.seconds // 3600}h {(age.seconds % 3600) // 60}m"
                
        except Exception:
            return "Unknown"
    
    def get_cache_stats(self):
        """Get cache statistics"""
        try:
            if not os.path.exists(self.cache_file):
                return {'total_predictions': 0, 'symbols': 0, 'file_size': 0}
            
            df = pd.read_csv(self.cache_file)
            file_size = os.path.getsize(self.cache_file)
            
            return {
                'total_predictions': len(df),
                'symbols': df['Symbol'].nunique() if not df.empty else 0,
                'file_size': file_size,
                'last_updated': datetime.fromtimestamp(os.path.getmtime(self.cache_file)).isoformat()
            }
            
        except Exception as e:
            print(f"Error getting cache stats: {e}")
            return {'total_predictions': 0, 'symbols': 0, 'file_size': 0}

class AccuracyTracker:
    """Enhanced accuracy tracking and validation system"""
    
    def __init__(self):
        self.accuracy_file = 'prediction_accuracy.csv'
        self.predictions_file = 'predictions.csv'
    
    def validate_predictions(self):
        """Validate old predictions and calculate accuracy"""
        try:
            if not os.path.exists(self.predictions_file):
                return None
            
            df = pd.read_csv(self.predictions_file)
            if df.empty:
                return None
            
            validation_results = []
            current_date = datetime.now().date()
            
            # Process predictions that can be validated
            for _, row in df.iterrows():
                symbol = row['Symbol']
                pred_date = datetime.fromisoformat(row['Date'])
                predicted_price = row['Predicted_Price']
                
                # Check if prediction date has passed
                if pred_date.date() < current_date:
                    try:
                        # Get actual price for that date
                        import yfinance as yf
                        ticker = yf.Ticker(symbol)
                        actual_data = ticker.history(start=pred_date.date(), end=pred_date.date() + timedelta(days=2))
                        
                        if not actual_data.empty:
                            actual_price = actual_data['Close'].iloc[0]
                            error_pct = abs(predicted_price - actual_price) / actual_price * 100
                            
                            validation_results.append({
                                'symbol': symbol,
                                'date': pred_date.date(),
                                'predicted': predicted_price,
                                'actual': actual_price,
                                'error_pct': error_pct,
                                'accurate': error_pct < 3.0,  # 3% threshold
                                'excellent': error_pct < 1.0,
                                'good': error_pct < 2.0,
                                'timestamp': row['Timestamp']
                            })
                            
                    except Exception as e:
                        print(f"Error validating {symbol} for {pred_date.date()}: {e}")
                        continue
            
            # Save validation results
            if validation_results:
                self._save_accuracy_results(validation_results)
                
                # Calculate summary statistics
                total_validated = len(validation_results)
                accurate_count = sum(1 for r in validation_results if r['accurate'])
                excellent_count = sum(1 for r in validation_results if r['excellent'])
                good_count = sum(1 for r in validation_results if r['good'])
                avg_error = sum(r['error_pct'] for r in validation_results) / total_validated
                
                return {
                    'validated': total_validated,
                    'accuracy_rate': (accurate_count / total_validated) * 100,
                    'excellent_rate': (excellent_count / total_validated) * 100,
                    'good_rate': (good_count / total_validated) * 100,
                    'avg_error': avg_error
                }
            
            return None
            
        except Exception as e:
            print(f"Validation failed: {e}")
            return None
    
    def _save_accuracy_results(self, validation_results):
        """Save accuracy results to CSV"""
        try:
            new_accuracy_df = pd.DataFrame(validation_results)
            new_accuracy_df['validation_timestamp'] = datetime.now().isoformat()
            
            if os.path.exists(self.accuracy_file):
                existing_df = pd.read_csv(self.accuracy_file)
                combined_df = pd.concat([existing_df, new_accuracy_df], ignore_index=True)
                
                # Remove duplicates and keep recent data only (90 days)
                combined_df['date'] = pd.to_datetime(combined_df['date'])
                ninety_days_ago = datetime.now().date() - timedelta(days=90)
                combined_df = combined_df[combined_df['date'].dt.date >= ninety_days_ago]
                combined_df = combined_df.drop_duplicates(subset=['symbol', 'date'], keep='last')
            else:
                combined_df = new_accuracy_df
            
            combined_df.to_csv(self.accuracy_file, index=False)
            
        except Exception as e:
            print(f"Error saving accuracy results: {e}")
    
    def analyze_accuracy(self, symbol=None):
        """Analyze prediction accuracy for a symbol or all symbols"""
        try:
            if not os.path.exists(self.accuracy_file):
                return None
            
            df = pd.read_csv(self.accuracy_file)
            if df.empty:
                return None
            
            # Filter by symbol if specified
            if symbol:
                df = df[df['symbol'] == symbol]
                if df.empty:
                    return None
            
            # Calculate statistics
            total_predictions = len(df)
            accurate_count = df['accurate'].sum()
            excellent_count = df['excellent'].sum()
            good_count = df['good'].sum()
            avg_error = df['error_pct'].mean()
            
            # Recent performance (last 30 days)
            df['date'] = pd.to_datetime(df['date'])
            thirty_days_ago = datetime.now().date() - timedelta(days=30)
            recent_df = df[df['date'].dt.date >= thirty_days_ago]
            
            recent_stats = {}
            if not recent_df.empty:
                recent_stats = {
                    'total': len(recent_df),
                    'accuracy_rate': (recent_df['accurate'].sum() / len(recent_df)) * 100,
                    'avg_error': recent_df['error_pct'].mean()
                }
            
            return {
                'symbol': symbol or 'All',
                'total_predictions': total_predictions,
                'accuracy_rate': (accurate_count / total_predictions) * 100,
                'excellent_rate': (excellent_count / total_predictions) * 100,
                'good_rate': (good_count / total_predictions) * 100,
                'avg_error': avg_error,
                'recent_stats': recent_stats
            }
            
        except Exception as e:
            print(f"Error analyzing accuracy: {e}")
            return None
    
    def get_accuracy_stats(self):
        """Get overall accuracy statistics"""
        try:
            return self.analyze_accuracy()
        except Exception as e:
            print(f"Error getting accuracy stats: {e}")
            return None