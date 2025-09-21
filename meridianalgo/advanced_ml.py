"""
Advanced Self-Learning ML System with Chart Pattern Recognition
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class SelfLearningLSTM(nn.Module):
    """
    Advanced LSTM with self-learning capabilities and error correction
    """
    
    def __init__(self, input_size=50, hidden_size=256, num_layers=4, dropout=0.3):
        super(SelfLearningLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Multi-layer LSTM with residual connections
        self.lstm1 = nn.LSTM(input_size, hidden_size, 1, dropout=dropout, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(hidden_size * 2, hidden_size, 1, dropout=dropout, batch_first=True, bidirectional=True)
        self.lstm3 = nn.LSTM(hidden_size * 2, hidden_size, 1, dropout=dropout, batch_first=True, bidirectional=True)
        
        # Self-attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=16,
            dropout=dropout,
            batch_first=True
        )
        
        # Error correction network
        self.error_corrector = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Confidence estimator
        self.confidence_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # Output layers with skip connections
        self.fc_layers = nn.ModuleList([
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.Linear(hidden_size // 4, 1)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size * 2),
            nn.LayerNorm(hidden_size),
            nn.LayerNorm(hidden_size // 2)
        ])
        
        # Self-learning parameters
        self.prediction_history = []
        self.error_history = []
        self.learning_rate_scheduler = None
        
    def forward(self, x, return_confidence=False):
        # Multi-layer LSTM with residual connections
        out1, _ = self.lstm1(x)
        out1 = self.layer_norms[0](out1)
        
        out2, _ = self.lstm2(out1)
        out2 = self.layer_norms[0](out2 + out1)  # Residual connection
        
        out3, _ = self.lstm3(out2)
        out3 = self.layer_norms[0](out3 + out2)  # Residual connection
        
        # Self-attention
        attn_out, attention_weights = self.attention(out3, out3, out3)
        
        # Take the last output
        final_out = attn_out[:, -1, :]
        
        # Main prediction path
        x_main = final_out
        for i, (fc, norm) in enumerate(zip(self.fc_layers[:-1], self.layer_norms[1:])):
            x_main = self.relu(fc(x_main))
            x_main = norm(x_main)
            x_main = self.dropout(x_main)
        
        prediction = self.fc_layers[-1](x_main)
        
        # Error correction
        error_correction = self.error_corrector(final_out)
        corrected_prediction = prediction + error_correction
        
        if return_confidence:
            confidence = self.confidence_net(final_out)
            return corrected_prediction, confidence, attention_weights
        
        return corrected_prediction
    
    def update_learning_history(self, prediction, actual, error):
        """Update learning history for self-improvement"""
        self.prediction_history.append({
            'prediction': prediction,
            'actual': actual,
            'error': error,
            'timestamp': pd.Timestamp.now()
        })
        
        self.error_history.append(error)
        
        # Keep only recent history (last 1000 predictions)
        if len(self.prediction_history) > 1000:
            self.prediction_history = self.prediction_history[-1000:]
            self.error_history = self.error_history[-1000:]
    
    def get_learning_insights(self):
        """Get insights from learning history"""
        if len(self.error_history) < 10:
            return {}
        
        recent_errors = self.error_history[-100:]
        return {
            'avg_error': np.mean(recent_errors),
            'error_trend': np.polyfit(range(len(recent_errors)), recent_errors, 1)[0],
            'error_std': np.std(recent_errors),
            'improvement_rate': (self.error_history[0] - self.error_history[-1]) / len(self.error_history)
        }

class ChartPatternRecognizer:
    """
    Advanced chart pattern recognition system
    """
    
    def __init__(self):
        self.patterns = {}
        self.pattern_confidence = {}
    
    def detect_triangles(self, prices, window=20):
        """Detect triangle patterns (ascending, descending, symmetrical)"""
        patterns = []
        
        if len(prices) < window * 2:
            return patterns
        
        for i in range(window, len(prices) - window):
            segment = prices[i-window:i+window]
            
            # Find peaks and troughs
            peaks = self._find_peaks(segment)
            troughs = self._find_troughs(segment)
            
            if len(peaks) >= 2 and len(troughs) >= 2:
                # Ascending triangle
                if self._is_ascending_triangle(segment, peaks, troughs):
                    patterns.append({
                        'type': 'ascending_triangle',
                        'position': i,
                        'confidence': self._calculate_pattern_confidence(segment, 'ascending_triangle'),
                        'breakout_direction': 'bullish',
                        'target_price': self._calculate_triangle_target(segment, 'ascending')
                    })
                
                # Descending triangle
                elif self._is_descending_triangle(segment, peaks, troughs):
                    patterns.append({
                        'type': 'descending_triangle',
                        'position': i,
                        'confidence': self._calculate_pattern_confidence(segment, 'descending_triangle'),
                        'breakout_direction': 'bearish',
                        'target_price': self._calculate_triangle_target(segment, 'descending')
                    })
                
                # Symmetrical triangle
                elif self._is_symmetrical_triangle(segment, peaks, troughs):
                    patterns.append({
                        'type': 'symmetrical_triangle',
                        'position': i,
                        'confidence': self._calculate_pattern_confidence(segment, 'symmetrical_triangle'),
                        'breakout_direction': 'neutral',
                        'target_price': self._calculate_triangle_target(segment, 'symmetrical')
                    })
        
        return patterns
    
    def detect_wedges(self, prices, window=25):
        """Detect wedge patterns (rising, falling)"""
        patterns = []
        
        if len(prices) < window * 2:
            return patterns
        
        for i in range(window, len(prices) - window):
            segment = prices[i-window:i+window]
            
            # Rising wedge (bearish)
            if self._is_rising_wedge(segment):
                patterns.append({
                    'type': 'rising_wedge',
                    'position': i,
                    'confidence': self._calculate_pattern_confidence(segment, 'rising_wedge'),
                    'breakout_direction': 'bearish',
                    'target_price': self._calculate_wedge_target(segment, 'rising')
                })
            
            # Falling wedge (bullish)
            elif self._is_falling_wedge(segment):
                patterns.append({
                    'type': 'falling_wedge',
                    'position': i,
                    'confidence': self._calculate_pattern_confidence(segment, 'falling_wedge'),
                    'breakout_direction': 'bullish',
                    'target_price': self._calculate_wedge_target(segment, 'falling')
                })
        
        return patterns
    
    def detect_head_and_shoulders(self, prices, window=30):
        """Detect head and shoulders patterns"""
        patterns = []
        
        if len(prices) < window * 2:
            return patterns
        
        for i in range(window, len(prices) - window):
            segment = prices[i-window:i+window]
            peaks = self._find_peaks(segment)
            
            if len(peaks) >= 3:
                # Head and shoulders (bearish)
                if self._is_head_and_shoulders(segment, peaks):
                    patterns.append({
                        'type': 'head_and_shoulders',
                        'position': i,
                        'confidence': self._calculate_pattern_confidence(segment, 'head_and_shoulders'),
                        'breakout_direction': 'bearish',
                        'target_price': self._calculate_hs_target(segment, peaks)
                    })
                
                # Inverse head and shoulders (bullish)
                elif self._is_inverse_head_and_shoulders(segment, peaks):
                    patterns.append({
                        'type': 'inverse_head_and_shoulders',
                        'position': i,
                        'confidence': self._calculate_pattern_confidence(segment, 'inverse_head_and_shoulders'),
                        'breakout_direction': 'bullish',
                        'target_price': self._calculate_ihs_target(segment, peaks)
                    })
        
        return patterns
    
    def detect_double_patterns(self, prices, window=20):
        """Detect double top and double bottom patterns"""
        patterns = []
        
        if len(prices) < window * 2:
            return patterns
        
        for i in range(window, len(prices) - window):
            segment = prices[i-window:i+window]
            
            # Double top (bearish)
            if self._is_double_top(segment):
                patterns.append({
                    'type': 'double_top',
                    'position': i,
                    'confidence': self._calculate_pattern_confidence(segment, 'double_top'),
                    'breakout_direction': 'bearish',
                    'target_price': self._calculate_double_target(segment, 'top')
                })
            
            # Double bottom (bullish)
            elif self._is_double_bottom(segment):
                patterns.append({
                    'type': 'double_bottom',
                    'position': i,
                    'confidence': self._calculate_pattern_confidence(segment, 'double_bottom'),
                    'breakout_direction': 'bullish',
                    'target_price': self._calculate_double_target(segment, 'bottom')
                })
        
        return patterns
    
    def _find_peaks(self, prices, prominence=0.02):
        """Find price peaks"""
        peaks = []
        for i in range(1, len(prices) - 1):
            if (prices[i] > prices[i-1] and prices[i] > prices[i+1] and
                prices[i] > np.mean(prices) * (1 + prominence)):
                peaks.append(i)
        return peaks
    
    def _find_troughs(self, prices, prominence=0.02):
        """Find price troughs"""
        troughs = []
        for i in range(1, len(prices) - 1):
            if (prices[i] < prices[i-1] and prices[i] < prices[i+1] and
                prices[i] < np.mean(prices) * (1 - prominence)):
                troughs.append(i)
        return troughs
    
    def _is_ascending_triangle(self, prices, peaks, troughs):
        """Check if pattern is ascending triangle"""
        if len(peaks) < 2 or len(troughs) < 2:
            return False
        
        # Resistance line should be relatively flat
        peak_prices = [prices[p] for p in peaks]
        resistance_slope = np.polyfit(peaks, peak_prices, 1)[0]
        
        # Support line should be ascending
        trough_prices = [prices[t] for t in troughs]
        support_slope = np.polyfit(troughs, trough_prices, 1)[0]
        
        return abs(resistance_slope) < 0.001 and support_slope > 0.001
    
    def _is_descending_triangle(self, prices, peaks, troughs):
        """Check if pattern is descending triangle"""
        if len(peaks) < 2 or len(troughs) < 2:
            return False
        
        # Resistance line should be descending
        peak_prices = [prices[p] for p in peaks]
        resistance_slope = np.polyfit(peaks, peak_prices, 1)[0]
        
        # Support line should be relatively flat
        trough_prices = [prices[t] for t in troughs]
        support_slope = np.polyfit(troughs, trough_prices, 1)[0]
        
        return resistance_slope < -0.001 and abs(support_slope) < 0.001
    
    def _is_symmetrical_triangle(self, prices, peaks, troughs):
        """Check if pattern is symmetrical triangle"""
        if len(peaks) < 2 or len(troughs) < 2:
            return False
        
        # Resistance line should be descending
        peak_prices = [prices[p] for p in peaks]
        resistance_slope = np.polyfit(peaks, peak_prices, 1)[0]
        
        # Support line should be ascending
        trough_prices = [prices[t] for t in troughs]
        support_slope = np.polyfit(troughs, trough_prices, 1)[0]
        
        return resistance_slope < -0.001 and support_slope > 0.001
    
    def _is_rising_wedge(self, prices):
        """Check if pattern is rising wedge"""
        # Both support and resistance lines ascending, but converging
        peaks = self._find_peaks(prices)
        troughs = self._find_troughs(prices)
        
        if len(peaks) < 2 or len(troughs) < 2:
            return False
        
        peak_prices = [prices[p] for p in peaks]
        trough_prices = [prices[t] for t in troughs]
        
        resistance_slope = np.polyfit(peaks, peak_prices, 1)[0]
        support_slope = np.polyfit(troughs, trough_prices, 1)[0]
        
        return (resistance_slope > 0 and support_slope > 0 and 
                support_slope > resistance_slope)
    
    def _is_falling_wedge(self, prices):
        """Check if pattern is falling wedge"""
        # Both support and resistance lines descending, but converging
        peaks = self._find_peaks(prices)
        troughs = self._find_troughs(prices)
        
        if len(peaks) < 2 or len(troughs) < 2:
            return False
        
        peak_prices = [prices[p] for p in peaks]
        trough_prices = [prices[t] for t in troughs]
        
        resistance_slope = np.polyfit(peaks, peak_prices, 1)[0]
        support_slope = np.polyfit(troughs, trough_prices, 1)[0]
        
        return (resistance_slope < 0 and support_slope < 0 and 
                support_slope < resistance_slope)
    
    def _is_head_and_shoulders(self, prices, peaks):
        """Check if pattern is head and shoulders"""
        if len(peaks) < 3:
            return False
        
        # Middle peak should be highest (head)
        peak_prices = [prices[p] for p in peaks[:3]]
        return peak_prices[1] > peak_prices[0] and peak_prices[1] > peak_prices[2]
    
    def _is_inverse_head_and_shoulders(self, prices, peaks):
        """Check if pattern is inverse head and shoulders"""
        troughs = self._find_troughs(prices)
        if len(troughs) < 3:
            return False
        
        # Middle trough should be lowest (head)
        trough_prices = [prices[t] for t in troughs[:3]]
        return trough_prices[1] < trough_prices[0] and trough_prices[1] < trough_prices[2]
    
    def _is_double_top(self, prices):
        """Check if pattern is double top"""
        peaks = self._find_peaks(prices)
        if len(peaks) < 2:
            return False
        
        # Two peaks at similar levels
        peak_prices = [prices[p] for p in peaks[:2]]
        return abs(peak_prices[0] - peak_prices[1]) / peak_prices[0] < 0.03
    
    def _is_double_bottom(self, prices):
        """Check if pattern is double bottom"""
        troughs = self._find_troughs(prices)
        if len(troughs) < 2:
            return False
        
        # Two troughs at similar levels
        trough_prices = [prices[t] for t in troughs[:2]]
        return abs(trough_prices[0] - trough_prices[1]) / trough_prices[0] < 0.03
    
    def _calculate_pattern_confidence(self, prices, pattern_type):
        """Calculate confidence score for pattern"""
        # Base confidence calculation
        volatility = np.std(prices) / np.mean(prices)
        volume_consistency = 0.8  # Placeholder - would use actual volume data
        
        base_confidence = max(0.5, 1.0 - volatility * 2)
        
        # Pattern-specific adjustments
        pattern_multipliers = {
            'ascending_triangle': 0.85,
            'descending_triangle': 0.85,
            'symmetrical_triangle': 0.75,
            'rising_wedge': 0.80,
            'falling_wedge': 0.80,
            'head_and_shoulders': 0.90,
            'inverse_head_and_shoulders': 0.90,
            'double_top': 0.85,
            'double_bottom': 0.85
        }
        
        multiplier = pattern_multipliers.get(pattern_type, 0.70)
        return min(0.95, base_confidence * multiplier * volume_consistency)
    
    def _calculate_triangle_target(self, prices, triangle_type):
        """Calculate price target for triangle patterns"""
        height = max(prices) - min(prices)
        current_price = prices[-1]
        
        if triangle_type == 'ascending':
            return current_price + height * 0.618  # Golden ratio
        elif triangle_type == 'descending':
            return current_price - height * 0.618
        else:  # symmetrical
            return current_price + height * 0.5 * (1 if np.random.random() > 0.5 else -1)
    
    def _calculate_wedge_target(self, prices, wedge_type):
        """Calculate price target for wedge patterns"""
        height = max(prices) - min(prices)
        current_price = prices[-1]
        
        if wedge_type == 'rising':
            return current_price - height * 0.618
        else:  # falling
            return current_price + height * 0.618
    
    def _calculate_hs_target(self, prices, peaks):
        """Calculate head and shoulders target"""
        neckline = min(prices)
        head = max([prices[p] for p in peaks])
        height = head - neckline
        return neckline - height
    
    def _calculate_ihs_target(self, prices, peaks):
        """Calculate inverse head and shoulders target"""
        neckline = max(prices)
        head = min(prices)
        height = neckline - head
        return neckline + height
    
    def _calculate_double_target(self, prices, pattern_type):
        """Calculate double top/bottom target"""
        if pattern_type == 'top':
            resistance = max(prices)
            support = min(prices)
            height = resistance - support
            return support - height
        else:  # bottom
            resistance = max(prices)
            support = min(prices)
            height = resistance - support
            return resistance + height

class AdvancedEnsembleSystem:
    """
    Enhanced ensemble system with self-learning and pattern recognition
    """
    
    def __init__(self, device=None):
        self.device = device or torch.device('cpu')
        self.models = {}
        self.scalers = {}
        self.pattern_recognizer = ChartPatternRecognizer()
        self.learning_history = []
        self.model_weights = {'lstm': 0.4, 'rf': 0.3, 'gb': 0.3}
        self.is_trained = False
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize enhanced models"""
        try:
            # Self-learning LSTM
            self.models['lstm'] = SelfLearningLSTM().to(self.device)
            self.lstm_optimizer = torch.optim.AdamW(
                self.models['lstm'].parameters(),
                lr=0.001,
                weight_decay=0.01
            )
            
            # Enhanced Random Forest
            self.models['rf'] = RandomForestRegressor(
                n_estimators=300,
                max_depth=20,
                min_samples_split=3,
                min_samples_leaf=1,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            )
            
            # Enhanced Gradient Boosting
            self.models['gb'] = GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.08,
                max_depth=10,
                min_samples_split=3,
                min_samples_leaf=1,
                subsample=0.8,
                random_state=42
            )
            
            # Scalers for different models
            self.scalers['lstm'] = MinMaxScaler()
            self.scalers['ml'] = StandardScaler()
            
        except Exception as e:
            print(f"Error initializing advanced models: {e}")
    
    def train_with_self_learning(self, features, targets, validation_split=0.2):
        """Train models with self-learning capabilities"""
        try:
            # Split data
            split_idx = int(len(features) * (1 - validation_split))
            train_features, val_features = features[:split_idx], features[split_idx:]
            train_targets, val_targets = targets[:split_idx], targets[split_idx:]
            
            # Train traditional ML models
            self._train_ml_models(train_features, train_targets)
            
            # Train LSTM with validation
            self._train_lstm_with_validation(train_features, train_targets, val_features, val_targets)
            
            # Optimize ensemble weights
            self._optimize_ensemble_weights(val_features, val_targets)
            
            self.is_trained = True
            return True
            
        except Exception as e:
            print(f"Advanced training failed: {e}")
            return False
    
    def _train_ml_models(self, features, targets):
        """Train ML models with feature scaling"""
        # Scale features
        scaled_features = self.scalers['ml'].fit_transform(features)
        
        # Train Random Forest
        self.models['rf'].fit(scaled_features, targets)
        
        # Train Gradient Boosting
        self.models['gb'].fit(scaled_features, targets)
    
    def _train_lstm_with_validation(self, train_features, train_targets, val_features, val_targets, epochs=150):
        """Train LSTM with validation and early stopping"""
        try:
            # Prepare LSTM data
            sequence_length = 60
            X_train, y_train = self._prepare_lstm_sequences(train_features, train_targets, sequence_length)
            X_val, y_val = self._prepare_lstm_sequences(val_features, val_targets, sequence_length)
            
            if X_train is None or X_val is None:
                return
            
            # Scale data
            X_train_scaled = self.scalers['lstm'].fit_transform(X_train.reshape(-1, X_train.shape[-1]))
            X_train_scaled = X_train_scaled.reshape(X_train.shape)
            
            X_val_scaled = self.scalers['lstm'].transform(X_val.reshape(-1, X_val.shape[-1]))
            X_val_scaled = X_val_scaled.reshape(X_val.shape)
            
            # Convert to tensors
            X_train_tensor = torch.FloatTensor(X_train_scaled).to(self.device)
            y_train_tensor = torch.FloatTensor(y_train).to(self.device)
            X_val_tensor = torch.FloatTensor(X_val_scaled).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).to(self.device)
            
            # Training with early stopping
            best_val_loss = float('inf')
            patience = 20
            patience_counter = 0
            
            criterion = nn.MSELoss()
            
            for epoch in range(epochs):
                # Training
                self.models['lstm'].train()
                self.lstm_optimizer.zero_grad()
                
                train_pred, train_conf, _ = self.models['lstm'](X_train_tensor, return_confidence=True)
                train_loss = criterion(train_pred.squeeze(), y_train_tensor)
                
                # Add confidence loss
                conf_loss = torch.mean((train_conf - 0.8) ** 2)  # Target confidence of 0.8
                total_loss = train_loss + 0.1 * conf_loss
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.models['lstm'].parameters(), 1.0)
                self.lstm_optimizer.step()
                
                # Validation
                self.models['lstm'].eval()
                with torch.no_grad():
                    val_pred, val_conf, _ = self.models['lstm'](X_val_tensor, return_confidence=True)
                    val_loss = criterion(val_pred.squeeze(), y_val_tensor)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model state
                    torch.save(self.models['lstm'].state_dict(), 'best_lstm_model.pth')
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    # Load best model
                    self.models['lstm'].load_state_dict(torch.load('best_lstm_model.pth'))
                    break
                
                if epoch % 25 == 0:
                    print(f"Epoch {epoch}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
        except Exception as e:
            print(f"LSTM training with validation failed: {e}")
    
    def _prepare_lstm_sequences(self, features, targets, sequence_length):
        """Prepare sequences for LSTM training"""
        try:
            if len(features) < sequence_length + 10:
                return None, None
            
            X, y = [], []
            for i in range(sequence_length, len(features)):
                X.append(features[i-sequence_length:i])
                y.append(targets[i])
            
            return np.array(X), np.array(y)
            
        except Exception as e:
            print(f"Error preparing LSTM sequences: {e}")
            return None, None
    
    def _optimize_ensemble_weights(self, val_features, val_targets):
        """Optimize ensemble weights based on validation performance"""
        try:
            # Get predictions from each model
            rf_pred = self._predict_rf(val_features)
            gb_pred = self._predict_gb(val_features)
            lstm_pred = self._predict_lstm(val_features)
            
            # Calculate individual model errors
            rf_error = mean_absolute_error(val_targets, rf_pred)
            gb_error = mean_absolute_error(val_targets, gb_pred)
            lstm_error = mean_absolute_error(val_targets, lstm_pred)
            
            # Calculate inverse error weights (better models get higher weights)
            total_inv_error = (1/rf_error) + (1/gb_error) + (1/lstm_error)
            
            self.model_weights = {
                'rf': (1/rf_error) / total_inv_error,
                'gb': (1/gb_error) / total_inv_error,
                'lstm': (1/lstm_error) / total_inv_error
            }
            
            print(f"Optimized weights: {self.model_weights}")
            
        except Exception as e:
            print(f"Weight optimization failed: {e}")
    
    def predict_with_patterns(self, features, prices, days=5):
        """Enhanced prediction with chart pattern analysis"""
        try:
            # Get base ensemble predictions
            base_predictions = self._get_ensemble_predictions(features, days)
            
            # Analyze chart patterns
            patterns = self._analyze_all_patterns(prices)
            
            # Adjust predictions based on patterns
            adjusted_predictions = self._adjust_predictions_with_patterns(
                base_predictions, patterns, prices[-1]
            )
            
            # Calculate confidence scores
            confidence_scores = self._calculate_prediction_confidence(
                features, patterns, adjusted_predictions
            )
            
            return {
                'predictions': adjusted_predictions,
                'confidence_scores': confidence_scores,
                'patterns': patterns,
                'base_predictions': base_predictions,
                'model_weights': self.model_weights
            }
            
        except Exception as e:
            print(f"Enhanced prediction failed: {e}")
            return self._fallback_prediction(features, days)
    
    def _get_ensemble_predictions(self, features, days):
        """Get ensemble predictions from all models"""
        try:
            rf_pred = self._predict_rf(features, days)
            gb_pred = self._predict_gb(features, days)
            lstm_pred = self._predict_lstm(features, days)
            
            # Weighted ensemble
            predictions = []
            for i in range(days):
                ensemble_pred = (
                    self.model_weights['rf'] * rf_pred[i] +
                    self.model_weights['gb'] * gb_pred[i] +
                    self.model_weights['lstm'] * lstm_pred[i]
                )
                predictions.append(ensemble_pred)
            
            return predictions
            
        except Exception as e:
            print(f"Ensemble prediction failed: {e}")
            return [features[-1, 0]] * days if len(features.shape) > 1 else [features[-1]] * days
    
    def _analyze_all_patterns(self, prices):
        """Analyze all chart patterns"""
        all_patterns = []
        
        try:
            # Detect various patterns
            triangles = self.pattern_recognizer.detect_triangles(prices)
            wedges = self.pattern_recognizer.detect_wedges(prices)
            hs_patterns = self.pattern_recognizer.detect_head_and_shoulders(prices)
            double_patterns = self.pattern_recognizer.detect_double_patterns(prices)
            
            all_patterns.extend(triangles)
            all_patterns.extend(wedges)
            all_patterns.extend(hs_patterns)
            all_patterns.extend(double_patterns)
            
            # Sort by confidence
            all_patterns.sort(key=lambda x: x['confidence'], reverse=True)
            
        except Exception as e:
            print(f"Pattern analysis failed: {e}")
        
        return all_patterns
    
    def _adjust_predictions_with_patterns(self, base_predictions, patterns, current_price):
        """Adjust predictions based on detected patterns"""
        adjusted_predictions = base_predictions.copy()
        
        if not patterns:
            return adjusted_predictions
        
        # Use the highest confidence pattern
        primary_pattern = patterns[0]
        
        try:
            pattern_influence = primary_pattern['confidence'] * 0.3  # Max 30% influence
            
            if primary_pattern['breakout_direction'] == 'bullish':
                for i in range(len(adjusted_predictions)):
                    adjustment = (primary_pattern['target_price'] - current_price) * pattern_influence * (i + 1) / len(adjusted_predictions)
                    adjusted_predictions[i] += adjustment
            
            elif primary_pattern['breakout_direction'] == 'bearish':
                for i in range(len(adjusted_predictions)):
                    adjustment = (current_price - primary_pattern['target_price']) * pattern_influence * (i + 1) / len(adjusted_predictions)
                    adjusted_predictions[i] -= adjustment
            
        except Exception as e:
            print(f"Pattern adjustment failed: {e}")
        
        return adjusted_predictions
    
    def _calculate_prediction_confidence(self, features, patterns, predictions):
        """Calculate confidence scores for predictions"""
        base_confidence = 0.75
        
        # Adjust based on patterns
        if patterns:
            pattern_confidence = patterns[0]['confidence']
            base_confidence = (base_confidence + pattern_confidence) / 2
        
        # Adjust based on data quality
        if len(features) > 100:
            base_confidence += 0.1
        
        # Adjust based on volatility
        if len(features.shape) > 1:
            recent_volatility = np.std(features[-20:, 0]) / np.mean(features[-20:, 0])
        else:
            recent_volatility = np.std(features[-20:]) / np.mean(features[-20:])
        
        volatility_adjustment = max(-0.2, -recent_volatility * 2)
        base_confidence += volatility_adjustment
        
        return [min(0.95, max(0.5, base_confidence - i * 0.05)) for i in range(len(predictions))]
    
    def _predict_rf(self, features, days=1):
        """Random Forest predictions"""
        try:
            if len(features.shape) > 1:
                last_features = features[-1:].copy()
            else:
                last_features = features.reshape(1, -1)
            
            scaled_features = self.scalers['ml'].transform(last_features)
            predictions = []
            
            for _ in range(days):
                pred = self.models['rf'].predict(scaled_features)[0]
                predictions.append(pred)
                # Update features for next prediction
                scaled_features = np.roll(scaled_features, -1)
                scaled_features[0, -1] = self.scalers['ml'].transform([[pred]])[0, 0]
            
            return predictions
            
        except Exception as e:
            print(f"RF prediction failed: {e}")
            return [features[-1, 0] if len(features.shape) > 1 else features[-1]] * days
    
    def _predict_gb(self, features, days=1):
        """Gradient Boosting predictions"""
        try:
            if len(features.shape) > 1:
                last_features = features[-1:].copy()
            else:
                last_features = features.reshape(1, -1)
            
            scaled_features = self.scalers['ml'].transform(last_features)
            predictions = []
            
            for _ in range(days):
                pred = self.models['gb'].predict(scaled_features)[0]
                predictions.append(pred)
                # Update features for next prediction
                scaled_features = np.roll(scaled_features, -1)
                scaled_features[0, -1] = self.scalers['ml'].transform([[pred]])[0, 0]
            
            return predictions
            
        except Exception as e:
            print(f"GB prediction failed: {e}")
            return [features[-1, 0] if len(features.shape) > 1 else features[-1]] * days
    
    def _predict_lstm(self, features, days=1):
        """LSTM predictions with confidence"""
        try:
            sequence_length = 60
            
            if len(features) >= sequence_length:
                sequence = features[-sequence_length:].copy()
            else:
                # Pad sequence
                padding = np.repeat(features[0:1], sequence_length - len(features), axis=0)
                sequence = np.vstack([padding, features])
            
            # Scale sequence
            sequence_scaled = self.scalers['lstm'].transform(sequence.reshape(-1, sequence.shape[-1]))
            sequence_scaled = sequence_scaled.reshape(sequence.shape)
            
            predictions = []
            self.models['lstm'].eval()
            
            with torch.no_grad():
                for _ in range(days):
                    seq_tensor = torch.FloatTensor(sequence_scaled).unsqueeze(0).to(self.device)
                    pred, confidence, _ = self.models['lstm'](seq_tensor, return_confidence=True)
                    
                    pred_value = pred.item()
                    predictions.append(pred_value)
                    
                    # Update sequence
                    new_row = sequence_scaled[-1].copy()
                    new_row[0] = self.scalers['lstm'].transform([[pred_value]])[0, 0]
                    sequence_scaled = np.vstack([sequence_scaled[1:], new_row.reshape(1, -1)])
            
            return predictions
            
        except Exception as e:
            print(f"LSTM prediction failed: {e}")
            return [features[-1, 0] if len(features.shape) > 1 else features[-1]] * days
    
    def _fallback_prediction(self, features, days):
        """Enhanced fallback prediction"""
        try:
            if len(features.shape) > 1:
                prices = features[:, 0]
            else:
                prices = features
            
            # Multiple trend analysis
            short_trend = np.mean(np.diff(prices[-5:])) if len(prices) >= 6 else 0
            medium_trend = np.mean(np.diff(prices[-10:])) if len(prices) >= 11 else 0
            long_trend = np.mean(np.diff(prices[-20:])) if len(prices) >= 21 else 0
            
            # Weighted trend
            trend = (0.5 * short_trend + 0.3 * medium_trend + 0.2 * long_trend)
            
            current_price = prices[-1]
            predictions = []
            
            for i in range(days):
                # Apply trend with dampening
                dampening = 0.9 ** i
                pred_price = current_price + (trend * (i + 1) * dampening)
                
                # Add some noise based on historical volatility
                volatility = np.std(prices[-20:]) if len(prices) >= 20 else np.std(prices)
                noise = np.random.normal(0, volatility * 0.1)
                pred_price += noise
                
                predictions.append(pred_price)
            
            return {
                'predictions': predictions,
                'confidence_scores': [0.6 - i * 0.05 for i in range(days)],
                'patterns': [],
                'base_predictions': predictions,
                'model_weights': {'fallback': 1.0}
            }
            
        except Exception as e:
            print(f"Fallback prediction failed: {e}")
            return {
                'predictions': [100.0] * days,
                'confidence_scores': [0.5] * days,
                'patterns': [],
                'base_predictions': [100.0] * days,
                'model_weights': {'fallback': 1.0}
            }
    
    def update_self_learning(self, prediction_data, actual_data):
        """Update self-learning system with new data"""
        try:
            # Update LSTM learning history
            if 'lstm_prediction' in prediction_data and actual_data:
                error = abs(prediction_data['lstm_prediction'] - actual_data) / actual_data
                self.models['lstm'].update_learning_history(
                    prediction_data['lstm_prediction'], 
                    actual_data, 
                    error
                )
            
            # Update ensemble weights based on performance
            self._update_ensemble_weights(prediction_data, actual_data)
            
            # Store learning data
            self.learning_history.append({
                'prediction': prediction_data,
                'actual': actual_data,
                'timestamp': pd.Timestamp.now(),
                'error': abs(prediction_data.get('ensemble_prediction', 0) - actual_data) / actual_data if actual_data else 0
            })
            
            # Keep only recent history
            if len(self.learning_history) > 1000:
                self.learning_history = self.learning_history[-1000:]
            
        except Exception as e:
            print(f"Self-learning update failed: {e}")
    
    def _update_ensemble_weights(self, prediction_data, actual_data):
        """Update ensemble weights based on recent performance"""
        try:
            if len(self.learning_history) < 50:
                return
            
            # Calculate recent performance for each model
            recent_history = self.learning_history[-50:]
            
            model_errors = {'rf': [], 'gb': [], 'lstm': []}
            
            for entry in recent_history:
                pred = entry['prediction']
                actual = entry['actual']
                
                if actual and actual > 0:
                    for model in model_errors:
                        if f'{model}_prediction' in pred:
                            error = abs(pred[f'{model}_prediction'] - actual) / actual
                            model_errors[model].append(error)
            
            # Update weights based on inverse of average errors
            new_weights = {}
            total_inv_error = 0
            
            for model, errors in model_errors.items():
                if errors:
                    avg_error = np.mean(errors)
                    inv_error = 1 / (avg_error + 0.001)  # Add small constant to avoid division by zero
                    new_weights[model] = inv_error
                    total_inv_error += inv_error
                else:
                    new_weights[model] = 1.0
                    total_inv_error += 1.0
            
            # Normalize weights
            for model in new_weights:
                new_weights[model] /= total_inv_error
            
            # Smooth weight updates (blend with current weights)
            alpha = 0.1  # Learning rate for weight updates
            for model in self.model_weights:
                if model in new_weights:
                    self.model_weights[model] = (
                        (1 - alpha) * self.model_weights[model] + 
                        alpha * new_weights[model]
                    )
            
        except Exception as e:
            print(f"Weight update failed: {e}")
    
    def get_learning_insights(self):
        """Get insights from the learning system"""
        insights = {}
        
        try:
            # LSTM insights
            if hasattr(self.models['lstm'], 'get_learning_insights'):
                insights['lstm'] = self.models['lstm'].get_learning_insights()
            
            # Ensemble insights
            if len(self.learning_history) > 10:
                recent_errors = [entry['error'] for entry in self.learning_history[-100:]]
                insights['ensemble'] = {
                    'avg_error': np.mean(recent_errors),
                    'error_trend': np.polyfit(range(len(recent_errors)), recent_errors, 1)[0],
                    'current_weights': self.model_weights.copy(),
                    'total_predictions': len(self.learning_history)
                }
            
        except Exception as e:
            print(f"Learning insights failed: {e}")
        
        return insights