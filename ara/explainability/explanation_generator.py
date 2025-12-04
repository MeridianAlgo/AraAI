"""
Natural Language Explanation Generator
Generates human-readable explanations for predictions
"""

import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime

from ara.core.exceptions import ValidationError


class ExplanationGenerator:
    """
    Generates natural language explanations for model predictions
    Combines SHAP values, feature contributions, and domain knowledge
    """
    
    def __init__(
        self,
        feature_descriptions: Optional[Dict[str, str]] = None,
        domain_context: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize explanation generator
        
        Args:
            feature_descriptions: Human-readable descriptions of features
            domain_context: Domain-specific context for better explanations
        """
        self.feature_descriptions = feature_descriptions or {}
        self.domain_context = domain_context or {}
    
    def generate_prediction_explanation(
        self,
        prediction: float,
        base_value: float,
        top_features: List[Dict[str, Any]],
        confidence: Optional[float] = None,
        prediction_type: str = "price"
    ) -> str:
        """
        Generate comprehensive natural language explanation
        
        Args:
            prediction: Predicted value
            base_value: Base/expected value
            top_features: List of top contributing features
            confidence: Prediction confidence (0-1)
            prediction_type: Type of prediction ('price', 'return', 'direction')
            
        Returns:
            Natural language explanation string
        """
        explanation_parts = []
        
        # Opening statement
        if prediction_type == "price":
            explanation_parts.append(
                f"The model predicts a price of ${prediction:.2f}."
            )
        elif prediction_type == "return":
            direction = "gain" if prediction > 0 else "loss"
            explanation_parts.append(
                f"The model predicts a {abs(prediction):.2f}% {direction}."
            )
        else:
            explanation_parts.append(
                f"The model predicts a value of {prediction:.4f}."
            )
        
        # Confidence statement
        if confidence is not None:
            confidence_level = self._get_confidence_level(confidence)
            explanation_parts.append(
                f"This prediction has {confidence_level} confidence ({confidence*100:.1f}%)."
            )
        
        # Baseline comparison
        change_from_base = prediction - base_value
        if abs(change_from_base) > 0.01:
            direction = "above" if change_from_base > 0 else "below"
            explanation_parts.append(
                f"This is {abs(change_from_base):.2f} {direction} the baseline expectation of {base_value:.2f}."
            )
        
        # Top contributing factors
        if top_features:
            explanation_parts.append("\nKey factors driving this prediction:")
            
            for i, feature in enumerate(top_features[:5], 1):
                feature_explanation = self._explain_feature_contribution(
                    feature, i
                )
                explanation_parts.append(feature_explanation)
        
        # Summary
        positive_features = [f for f in top_features if f['contribution'] > 0]
        negative_features = [f for f in top_features if f['contribution'] < 0]
        
        if positive_features and negative_features:
            explanation_parts.append(
                f"\nOverall, {len(positive_features)} factors push the prediction higher, "
                f"while {len(negative_features)} factors push it lower."
            )
        elif positive_features:
            explanation_parts.append(
                f"\nAll major factors ({len(positive_features)}) push the prediction higher."
            )
        elif negative_features:
            explanation_parts.append(
                f"\nAll major factors ({len(negative_features)}) push the prediction lower."
            )
        
        return "\n".join(explanation_parts)
    
    def _explain_feature_contribution(
        self,
        feature: Dict[str, Any],
        rank: int
    ) -> str:
        """
        Generate explanation for a single feature contribution
        
        Args:
            feature: Feature dict with name, value, contribution, etc.
            rank: Rank of this feature (1-based)
            
        Returns:
            Explanation string
        """
        name = feature['feature_name']
        value = feature['feature_value']
        contribution = feature['contribution']
        percentage = feature.get('contribution_percentage', 0)
        
        # Get human-readable description
        description = self.feature_descriptions.get(name, name)
        
        # Determine impact direction and strength
        if contribution > 0:
            impact = "increases"
            direction = "upward"
        else:
            impact = "decreases"
            direction = "downward"
        
        # Determine strength
        if percentage > 20:
            strength = "strongly"
        elif percentage > 10:
            strength = "moderately"
        else:
            strength = "slightly"
        
        # Build explanation
        explanation = f"{rank}. {description} (value: {value:.4f}) {strength} {impact} the prediction"
        
        if percentage > 0:
            explanation += f", contributing {percentage:.1f}% of the total {direction} pressure"
        
        explanation += "."
        
        return explanation
    
    def _get_confidence_level(self, confidence: float) -> str:
        """Convert confidence score to descriptive level"""
        if confidence >= 0.9:
            return "very high"
        elif confidence >= 0.75:
            return "high"
        elif confidence >= 0.6:
            return "moderate"
        elif confidence >= 0.4:
            return "low"
        else:
            return "very low"
    
    def generate_comparison_explanation(
        self,
        prediction1: float,
        prediction2: float,
        changed_features: List[Dict[str, Any]],
        label1: str = "Scenario 1",
        label2: str = "Scenario 2"
    ) -> str:
        """
        Generate explanation comparing two predictions
        
        Args:
            prediction1: First prediction
            prediction2: Second prediction
            changed_features: Features that changed between scenarios
            label1: Label for first scenario
            label2: Label for second scenario
            
        Returns:
            Comparison explanation
        """
        explanation_parts = []
        
        # Overall comparison
        diff = prediction2 - prediction1
        pct_change = (diff / prediction1 * 100) if prediction1 != 0 else 0
        
        if diff > 0:
            explanation_parts.append(
                f"{label2} predicts a value {abs(diff):.2f} higher than {label1} "
                f"({abs(pct_change):.1f}% increase)."
            )
        else:
            explanation_parts.append(
                f"{label2} predicts a value {abs(diff):.2f} lower than {label1} "
                f"({abs(pct_change):.1f}% decrease)."
            )
        
        # Key changes
        if changed_features:
            explanation_parts.append("\nKey differences:")
            
            for i, feature in enumerate(changed_features[:5], 1):
                name = feature['feature_name']
                change = feature['change']
                
                description = self.feature_descriptions.get(name, name)
                
                if change > 0:
                    explanation_parts.append(
                        f"{i}. {description} has increased impact (+{abs(change):.4f})"
                    )
                else:
                    explanation_parts.append(
                        f"{i}. {description} has decreased impact ({change:.4f})"
                    )
        
        return "\n".join(explanation_parts)
    
    def generate_feature_importance_explanation(
        self,
        feature_importance: Dict[str, float],
        top_n: int = 10
    ) -> str:
        """
        Generate explanation of overall feature importance
        
        Args:
            feature_importance: Dict mapping features to importance scores
            top_n: Number of top features to explain
            
        Returns:
            Feature importance explanation
        """
        explanation_parts = [
            "Feature Importance Analysis:",
            "\nThe following features have the most influence on predictions:\n"
        ]
        
        # Sort by importance
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:top_n]
        
        # Calculate total importance for percentages
        total_importance = sum(abs(score) for _, score in sorted_features)
        
        for i, (feature_name, importance) in enumerate(sorted_features, 1):
            description = self.feature_descriptions.get(feature_name, feature_name)
            percentage = (abs(importance) / total_importance * 100) if total_importance > 0 else 0
            
            explanation_parts.append(
                f"{i}. {description}: {percentage:.1f}% of total importance"
            )
        
        return "\n".join(explanation_parts)
    
    def generate_uncertainty_explanation(
        self,
        confidence: float,
        uncertainty_factors: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate explanation of prediction uncertainty
        
        Args:
            confidence: Confidence score (0-1)
            uncertainty_factors: Dict with factors affecting uncertainty
            
        Returns:
            Uncertainty explanation
        """
        explanation_parts = []
        
        confidence_level = self._get_confidence_level(confidence)
        explanation_parts.append(
            f"Prediction Confidence: {confidence_level} ({confidence*100:.1f}%)"
        )
        
        # Explain confidence level
        if confidence >= 0.75:
            explanation_parts.append(
                "\nThe model is confident in this prediction because:"
            )
            reasons = [
                "- Multiple models agree on the prediction",
                "- Input data quality is high",
                "- Similar historical patterns have been accurate"
            ]
        else:
            explanation_parts.append(
                "\nThe model has lower confidence in this prediction due to:"
            )
            reasons = [
                "- Models show some disagreement",
                "- Market conditions are uncertain",
                "- Limited historical data for similar scenarios"
            ]
        
        explanation_parts.extend(reasons)
        
        # Add specific uncertainty factors if provided
        if uncertainty_factors:
            explanation_parts.append("\nSpecific uncertainty factors:")
            
            for factor_name, factor_value in uncertainty_factors.items():
                if isinstance(factor_value, (int, float)):
                    explanation_parts.append(
                        f"- {factor_name}: {factor_value:.2f}"
                    )
                else:
                    explanation_parts.append(
                        f"- {factor_name}: {factor_value}"
                    )
        
        return "\n".join(explanation_parts)
    
    def generate_time_series_explanation(
        self,
        predictions: List[float],
        time_labels: List[str],
        trend: str = "unknown"
    ) -> str:
        """
        Generate explanation for time series predictions
        
        Args:
            predictions: List of predicted values over time
            time_labels: Labels for each time point
            trend: Overall trend ('increasing', 'decreasing', 'stable', 'volatile')
            
        Returns:
            Time series explanation
        """
        if len(predictions) != len(time_labels):
            raise ValidationError(
                "predictions and time_labels must have same length",
                {"predictions_len": len(predictions), "time_labels_len": len(time_labels)}
            )
        
        explanation_parts = ["Time Series Prediction Analysis:\n"]
        
        # Overall trend
        if trend == "unknown":
            # Calculate trend
            if len(predictions) > 1:
                changes = np.diff(predictions)
                avg_change = np.mean(changes)
                volatility = np.std(changes)
                
                if volatility > abs(avg_change) * 2:
                    trend = "volatile"
                elif avg_change > 0.01:
                    trend = "increasing"
                elif avg_change < -0.01:
                    trend = "decreasing"
                else:
                    trend = "stable"
        
        trend_descriptions = {
            "increasing": "The predictions show an upward trend over time.",
            "decreasing": "The predictions show a downward trend over time.",
            "stable": "The predictions remain relatively stable over time.",
            "volatile": "The predictions show high volatility with no clear trend."
        }
        
        explanation_parts.append(trend_descriptions.get(trend, "Trend analysis unavailable."))
        
        # Key points
        if len(predictions) >= 3:
            min_idx = np.argmin(predictions)
            max_idx = np.argmax(predictions)
            
            explanation_parts.append(
                f"\nLowest predicted value: {predictions[min_idx]:.2f} at {time_labels[min_idx]}"
            )
            explanation_parts.append(
                f"Highest predicted value: {predictions[max_idx]:.2f} at {time_labels[max_idx]}"
            )
            
            # Calculate range
            value_range = predictions[max_idx] - predictions[min_idx]
            explanation_parts.append(
                f"Prediction range: {value_range:.2f}"
            )
        
        return "\n".join(explanation_parts)
    
    def generate_model_explanation(
        self,
        model_name: str,
        model_type: str,
        model_performance: Optional[Dict[str, float]] = None
    ) -> str:
        """
        Generate explanation about the model itself
        
        Args:
            model_name: Name of the model
            model_type: Type of model (e.g., 'transformer', 'ensemble', 'xgboost')
            model_performance: Dict with performance metrics
            
        Returns:
            Model explanation
        """
        explanation_parts = [f"Model: {model_name}"]
        
        # Model type description
        model_descriptions = {
            "transformer": "A state-of-the-art deep learning model using attention mechanisms to capture complex patterns in time series data.",
            "ensemble": "A combination of multiple machine learning models that vote on predictions to improve accuracy and robustness.",
            "xgboost": "A gradient boosting model that builds an ensemble of decision trees optimized for prediction accuracy.",
            "lstm": "A recurrent neural network that can learn long-term dependencies in sequential data.",
            "cnn_lstm": "A hybrid model combining convolutional layers for pattern recognition with LSTM for temporal modeling."
        }
        
        if model_type.lower() in model_descriptions:
            explanation_parts.append(f"\n{model_descriptions[model_type.lower()]}")
        
        # Performance metrics
        if model_performance:
            explanation_parts.append("\nModel Performance:")
            
            for metric_name, metric_value in model_performance.items():
                if isinstance(metric_value, (int, float)):
                    explanation_parts.append(
                        f"- {metric_name}: {metric_value:.4f}"
                    )
        
        return "\n".join(explanation_parts)
    
    def generate_full_explanation(
        self,
        prediction_data: Dict[str, Any]
    ) -> str:
        """
        Generate comprehensive explanation from prediction data
        
        Args:
            prediction_data: Dict containing all prediction information
            
        Returns:
            Full explanation string
        """
        sections = []
        
        # Prediction explanation
        if all(k in prediction_data for k in ['prediction', 'base_value', 'top_features']):
            pred_explanation = self.generate_prediction_explanation(
                prediction=prediction_data['prediction'],
                base_value=prediction_data['base_value'],
                top_features=prediction_data['top_features'],
                confidence=prediction_data.get('confidence'),
                prediction_type=prediction_data.get('prediction_type', 'price')
            )
            sections.append(pred_explanation)
        
        # Uncertainty explanation
        if 'confidence' in prediction_data:
            uncertainty_explanation = self.generate_uncertainty_explanation(
                confidence=prediction_data['confidence'],
                uncertainty_factors=prediction_data.get('uncertainty_factors')
            )
            sections.append("\n" + uncertainty_explanation)
        
        # Model explanation
        if 'model_name' in prediction_data and 'model_type' in prediction_data:
            model_explanation = self.generate_model_explanation(
                model_name=prediction_data['model_name'],
                model_type=prediction_data['model_type'],
                model_performance=prediction_data.get('model_performance')
            )
            sections.append("\n" + model_explanation)
        
        # Timestamp
        sections.append(f"\n\nGenerated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return "\n".join(sections)
