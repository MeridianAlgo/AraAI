"""
Feature Contribution Analyzer
Analyzes and ranks feature contributions to predictions
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import warnings

from ara.core.exceptions import ValidationError


class FeatureContributionAnalyzer:
    """
    Analyzes feature contributions to model predictions
    Provides top contributing factors and contribution percentages
    """
    
    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        feature_descriptions: Optional[Dict[str, str]] = None
    ):
        """
        Initialize feature contribution analyzer
        
        Args:
            feature_names: Names of features
            feature_descriptions: Human-readable descriptions of features
        """
        self.feature_names = feature_names
        self.feature_descriptions = feature_descriptions or {}
    
    def analyze_contributions(
        self,
        feature_values: np.ndarray,
        contributions: np.ndarray,
        top_n: int = 5
    ) -> Dict[str, Any]:
        """
        Analyze feature contributions for a prediction
        
        Args:
            feature_values: Feature values for the sample
            contributions: Contribution values (e.g., SHAP values, coefficients)
            top_n: Number of top features to return
            
        Returns:
            Dict with contribution analysis
        """
        if len(feature_values) != len(contributions):
            raise ValidationError(
                "feature_values and contributions must have same length",
                {
                    "feature_values_len": len(feature_values),
                    "contributions_len": len(contributions)
                }
            )
        
        # Calculate absolute contributions
        abs_contributions = np.abs(contributions)
        
        # Get top contributing features
        top_indices = np.argsort(abs_contributions)[-top_n:][::-1]
        
        # Calculate contribution percentages
        total_abs_contribution = np.sum(abs_contributions)
        if total_abs_contribution > 0:
            contribution_percentages = (abs_contributions / total_abs_contribution) * 100
        else:
            contribution_percentages = np.zeros_like(abs_contributions)
        
        # Build top features list
        top_features = []
        for idx in top_indices:
            feature_name = (
                self.feature_names[idx] if self.feature_names
                else f"feature_{idx}"
            )
            
            feature_info = {
                'rank': len(top_features) + 1,
                'feature_name': feature_name,
                'feature_value': float(feature_values[idx]),
                'contribution': float(contributions[idx]),
                'abs_contribution': float(abs_contributions[idx]),
                'contribution_percentage': float(contribution_percentages[idx]),
                'direction': 'positive' if contributions[idx] > 0 else 'negative',
                'description': self.feature_descriptions.get(feature_name, '')
            }
            
            top_features.append(feature_info)
        
        # Calculate summary statistics
        positive_contribution = float(np.sum(contributions[contributions > 0]))
        negative_contribution = float(np.sum(contributions[contributions < 0]))
        net_contribution = float(np.sum(contributions))
        
        return {
            'top_features': top_features,
            'summary': {
                'total_features': len(feature_values),
                'positive_contribution': positive_contribution,
                'negative_contribution': negative_contribution,
                'net_contribution': net_contribution,
                'total_abs_contribution': float(total_abs_contribution),
                'top_n_percentage': float(
                    np.sum(contribution_percentages[top_indices])
                )
            }
        }
    
    def calculate_contribution_percentages(
        self,
        contributions: np.ndarray
    ) -> np.ndarray:
        """
        Calculate contribution percentages for all features
        
        Args:
            contributions: Contribution values
            
        Returns:
            Array of contribution percentages
        """
        abs_contributions = np.abs(contributions)
        total = np.sum(abs_contributions)
        
        if total > 0:
            return (abs_contributions / total) * 100
        else:
            return np.zeros_like(abs_contributions)
    
    def get_top_n_features(
        self,
        feature_values: np.ndarray,
        contributions: np.ndarray,
        n: int = 5
    ) -> List[Tuple[str, float, float]]:
        """
        Get top N contributing features
        
        Args:
            feature_values: Feature values
            contributions: Contribution values
            n: Number of features to return
            
        Returns:
            List of (feature_name, feature_value, contribution) tuples
        """
        abs_contributions = np.abs(contributions)
        top_indices = np.argsort(abs_contributions)[-n:][::-1]
        
        result = []
        for idx in top_indices:
            feature_name = (
                self.feature_names[idx] if self.feature_names
                else f"feature_{idx}"
            )
            result.append((
                feature_name,
                float(feature_values[idx]),
                float(contributions[idx])
            ))
        
        return result
    
    def generate_factor_descriptions(
        self,
        top_features: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Generate human-readable descriptions for top factors
        
        Args:
            top_features: List of top feature dicts from analyze_contributions
            
        Returns:
            List of description strings
        """
        descriptions = []
        
        for feature in top_features:
            name = feature['feature_name']
            value = feature['feature_value']
            contribution = feature['contribution']
            percentage = feature['contribution_percentage']
            direction = feature['direction']
            
            # Get custom description if available
            custom_desc = feature.get('description', '')
            
            if custom_desc:
                desc = f"{custom_desc}: {value:.4f}"
            else:
                desc = f"{name}: {value:.4f}"
            
            # Add contribution info
            if direction == 'positive':
                desc += f" (pushes prediction UP by {percentage:.1f}%)"
            else:
                desc += f" (pushes prediction DOWN by {percentage:.1f}%)"
            
            descriptions.append(desc)
        
        return descriptions
    
    def compare_contributions(
        self,
        contributions1: np.ndarray,
        contributions2: np.ndarray,
        feature_values1: Optional[np.ndarray] = None,
        feature_values2: Optional[np.ndarray] = None,
        top_n: int = 5
    ) -> Dict[str, Any]:
        """
        Compare feature contributions between two predictions
        
        Args:
            contributions1: Contributions for first prediction
            contributions2: Contributions for second prediction
            feature_values1: Feature values for first prediction (optional)
            feature_values2: Feature values for second prediction (optional)
            top_n: Number of top features to compare
            
        Returns:
            Dict with comparison results
        """
        if len(contributions1) != len(contributions2):
            raise ValidationError(
                "Contribution arrays must have same length",
                {
                    "contributions1_len": len(contributions1),
                    "contributions2_len": len(contributions2)
                }
            )
        
        # Calculate differences
        contribution_diff = contributions2 - contributions1
        abs_diff = np.abs(contribution_diff)
        
        # Get features with largest changes
        top_changed_indices = np.argsort(abs_diff)[-top_n:][::-1]
        
        changed_features = []
        for idx in top_changed_indices:
            feature_name = (
                self.feature_names[idx] if self.feature_names
                else f"feature_{idx}"
            )
            
            feature_info = {
                'feature_name': feature_name,
                'contribution_1': float(contributions1[idx]),
                'contribution_2': float(contributions2[idx]),
                'change': float(contribution_diff[idx]),
                'abs_change': float(abs_diff[idx]),
                'change_direction': 'increased' if contribution_diff[idx] > 0 else 'decreased'
            }
            
            if feature_values1 is not None and feature_values2 is not None:
                feature_info['value_1'] = float(feature_values1[idx])
                feature_info['value_2'] = float(feature_values2[idx])
                feature_info['value_change'] = float(feature_values2[idx] - feature_values1[idx])
            
            changed_features.append(feature_info)
        
        return {
            'changed_features': changed_features,
            'summary': {
                'total_contribution_change': float(np.sum(contribution_diff)),
                'mean_abs_change': float(np.mean(abs_diff)),
                'max_change': float(np.max(abs_diff)),
                'features_increased': int(np.sum(contribution_diff > 0)),
                'features_decreased': int(np.sum(contribution_diff < 0))
            }
        }
    
    def create_contribution_matrix(
        self,
        contributions: np.ndarray,
        feature_groups: Optional[Dict[str, List[int]]] = None
    ) -> Dict[str, Any]:
        """
        Create a contribution matrix grouped by feature categories
        
        Args:
            contributions: Contribution values (n_samples, n_features)
            feature_groups: Dict mapping group names to feature indices
            
        Returns:
            Dict with grouped contributions
        """
        if contributions.ndim == 1:
            contributions = contributions.reshape(1, -1)
        
        if feature_groups is None:
            # No grouping - return all features
            return {
                'all_features': {
                    'contributions': contributions,
                    'mean_contribution': float(np.mean(np.abs(contributions))),
                    'total_contribution': float(np.sum(np.abs(contributions)))
                }
            }
        
        grouped_results = {}
        
        for group_name, feature_indices in feature_groups.items():
            group_contributions = contributions[:, feature_indices]
            
            grouped_results[group_name] = {
                'feature_indices': feature_indices,
                'n_features': len(feature_indices),
                'contributions': group_contributions,
                'mean_contribution': float(np.mean(np.abs(group_contributions))),
                'total_contribution': float(np.sum(np.abs(group_contributions))),
                'positive_contribution': float(np.sum(group_contributions[group_contributions > 0])),
                'negative_contribution': float(np.sum(group_contributions[group_contributions < 0]))
            }
        
        return grouped_results
    
    def rank_features_by_importance(
        self,
        contributions: np.ndarray,
        method: str = "mean_abs"
    ) -> List[Tuple[str, float]]:
        """
        Rank all features by importance
        
        Args:
            contributions: Contribution values (n_samples, n_features) or (n_features,)
            method: Ranking method ('mean_abs', 'median_abs', 'max_abs')
            
        Returns:
            List of (feature_name, importance_score) tuples, sorted by importance
        """
        if contributions.ndim == 1:
            contributions = contributions.reshape(1, -1)
        
        # Calculate importance scores
        if method == "mean_abs":
            importance = np.mean(np.abs(contributions), axis=0)
        elif method == "median_abs":
            importance = np.median(np.abs(contributions), axis=0)
        elif method == "max_abs":
            importance = np.max(np.abs(contributions), axis=0)
        else:
            raise ValidationError(
                f"Unknown ranking method: {method}",
                {"valid_methods": ["mean_abs", "median_abs", "max_abs"]}
            )
        
        # Create ranked list
        ranked_features = []
        for idx, score in enumerate(importance):
            feature_name = (
                self.feature_names[idx] if self.feature_names
                else f"feature_{idx}"
            )
            ranked_features.append((feature_name, float(score)))
        
        # Sort by importance (descending)
        ranked_features.sort(key=lambda x: x[1], reverse=True)
        
        return ranked_features
