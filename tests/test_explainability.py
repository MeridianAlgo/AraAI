"""
Tests for Explainability Module
Tests SHAP explainer, feature analyzer, attention visualizer, and explanation generator
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ara.explainability import (
    SHAPExplainer,
    FeatureContributionAnalyzer,
    AttentionVisualizer,
    ExplanationGenerator
)


# Test fixtures
@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = 2.0 * X[:, 0] + 1.5 * X[:, 1] - 1.0 * X[:, 2] + 0.1 * np.random.randn(100)
    return X, y


@pytest.fixture
def trained_model(sample_data):
    """Create a trained model for testing"""
    X, y = sample_data
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model


@pytest.fixture
def feature_names():
    """Create feature names for testing"""
    return [f"feature_{i}" for i in range(10)]


# SHAP Explainer Tests
class TestSHAPExplainer:
    """Test SHAP explainer functionality"""
    
    def test_shap_explainer_creation(self, trained_model, sample_data, feature_names):
        """Test SHAP explainer initialization"""
        X, _ = sample_data
        
        try:
            explainer = SHAPExplainer(
                model=trained_model,
                background_data=X[:20],
                feature_names=feature_names,
                explainer_type='tree'
            )
            assert explainer is not None
            assert explainer.model == trained_model
            assert explainer.feature_names == feature_names
        except ImportError:
            pytest.skip("SHAP not available")
    
    def test_shap_explain(self, trained_model, sample_data, feature_names):
        """Test SHAP value calculation"""
        X, _ = sample_data
        
        try:
            explainer = SHAPExplainer(
                model=trained_model,
                background_data=X[:20],
                feature_names=feature_names,
                explainer_type='tree'
            )
            
            explanation = explainer.explain(X[:5])
            
            assert 'shap_values' in explanation
            assert 'base_value' in explanation
            assert explanation['shap_values'].shape == (5, 10)
        except ImportError:
            pytest.skip("SHAP not available")
    
    def test_feature_importance(self, trained_model, sample_data, feature_names):
        """Test feature importance calculation"""
        X, _ = sample_data
        
        try:
            explainer = SHAPExplainer(
                model=trained_model,
                background_data=X[:20],
                feature_names=feature_names,
                explainer_type='tree'
            )
            
            importance = explainer.get_feature_importance(X[:20])
            
            assert isinstance(importance, dict)
            assert len(importance) == 10
            assert all(isinstance(v, float) for v in importance.values())
        except ImportError:
            pytest.skip("SHAP not available")
    
    def test_top_features(self, trained_model, sample_data, feature_names):
        """Test top features extraction"""
        X, _ = sample_data
        
        try:
            explainer = SHAPExplainer(
                model=trained_model,
                background_data=X[:20],
                feature_names=feature_names,
                explainer_type='tree'
            )
            
            top_features = explainer.get_top_features(X[:5], n=3, sample_idx=0)
            
            assert len(top_features) == 3
            assert all('feature_name' in f for f in top_features)
            assert all('shap_value' in f for f in top_features)
            assert all('contribution' in f for f in top_features)
        except ImportError:
            pytest.skip("SHAP not available")
    
    def test_explain_prediction(self, trained_model, sample_data, feature_names):
        """Test comprehensive prediction explanation"""
        X, _ = sample_data
        
        try:
            explainer = SHAPExplainer(
                model=trained_model,
                background_data=X[:20],
                feature_names=feature_names,
                explainer_type='tree'
            )
            
            explanation = explainer.explain_prediction(X[:5], sample_idx=0, n_features=5)
            
            assert 'prediction' in explanation
            assert 'base_value' in explanation
            assert 'top_features' in explanation
            assert len(explanation['top_features']) == 5
        except ImportError:
            pytest.skip("SHAP not available")


# Feature Contribution Analyzer Tests
class TestFeatureContributionAnalyzer:
    """Test feature contribution analyzer"""
    
    def test_analyzer_creation(self, feature_names):
        """Test analyzer initialization"""
        analyzer = FeatureContributionAnalyzer(feature_names=feature_names)
        assert analyzer is not None
        assert analyzer.feature_names == feature_names
    
    def test_analyze_contributions(self, feature_names):
        """Test contribution analysis"""
        analyzer = FeatureContributionAnalyzer(feature_names=feature_names)
        
        feature_values = np.random.randn(10)
        contributions = np.random.randn(10)
        
        analysis = analyzer.analyze_contributions(
            feature_values=feature_values,
            contributions=contributions,
            top_n=5
        )
        
        assert 'top_features' in analysis
        assert 'summary' in analysis
        assert len(analysis['top_features']) == 5
        
        # Check feature structure
        for feature in analysis['top_features']:
            assert 'feature_name' in feature
            assert 'contribution' in feature
            assert 'contribution_percentage' in feature
            assert 'direction' in feature
    
    def test_contribution_percentages(self):
        """Test contribution percentage calculation"""
        analyzer = FeatureContributionAnalyzer()
        
        contributions = np.array([1.0, -0.5, 0.3, -0.2, 0.1])
        percentages = analyzer.calculate_contribution_percentages(contributions)
        
        assert len(percentages) == 5
        assert np.isclose(np.sum(percentages), 100.0)
        assert all(p >= 0 for p in percentages)
    
    def test_top_n_features(self, feature_names):
        """Test top N features extraction"""
        analyzer = FeatureContributionAnalyzer(feature_names=feature_names)
        
        feature_values = np.random.randn(10)
        contributions = np.array([2.0, -1.5, 1.0, -0.5, 0.3, 0.2, -0.1, 0.05, 0.02, 0.01])
        
        top_features = analyzer.get_top_n_features(
            feature_values=feature_values,
            contributions=contributions,
            n=3
        )
        
        assert len(top_features) == 3
        # Should be sorted by absolute contribution
        assert abs(top_features[0][2]) >= abs(top_features[1][2])
        assert abs(top_features[1][2]) >= abs(top_features[2][2])
    
    def test_factor_descriptions(self, feature_names):
        """Test factor description generation"""
        analyzer = FeatureContributionAnalyzer(feature_names=feature_names)
        
        top_features = [
            {
                'feature_name': 'feature_0',
                'feature_value': 1.5,
                'contribution': 0.8,
                'contribution_percentage': 40.0,
                'direction': 'positive',
                'description': 'Test feature'
            }
        ]
        
        descriptions = analyzer.generate_factor_descriptions(top_features)
        
        assert len(descriptions) == 1
        assert isinstance(descriptions[0], str)
        assert 'feature_0' in descriptions[0] or 'Test feature' in descriptions[0]
    
    def test_compare_contributions(self, feature_names):
        """Test contribution comparison"""
        analyzer = FeatureContributionAnalyzer(feature_names=feature_names)
        
        contributions1 = np.random.randn(10)
        contributions2 = np.random.randn(10)
        
        comparison = analyzer.compare_contributions(
            contributions1=contributions1,
            contributions2=contributions2,
            top_n=5
        )
        
        assert 'changed_features' in comparison
        assert 'summary' in comparison
        assert len(comparison['changed_features']) == 5
    
    def test_rank_features(self, feature_names):
        """Test feature ranking"""
        analyzer = FeatureContributionAnalyzer(feature_names=feature_names)
        
        contributions = np.random.randn(5, 10)
        
        ranked = analyzer.rank_features_by_importance(contributions, method='mean_abs')
        
        assert len(ranked) == 10
        # Should be sorted descending
        for i in range(len(ranked) - 1):
            assert ranked[i][1] >= ranked[i+1][1]


# Attention Visualizer Tests
class TestAttentionVisualizer:
    """Test attention visualizer"""
    
    def test_visualizer_creation(self):
        """Test visualizer initialization"""
        visualizer = AttentionVisualizer()
        assert visualizer is not None
    
    def test_attention_summary(self):
        """Test attention summary generation"""
        visualizer = AttentionVisualizer()
        
        # Create sample attention weights (n_heads, seq_len, seq_len)
        attention_weights = np.random.rand(8, 20, 20)
        # Normalize to sum to 1 along last dimension
        attention_weights = attention_weights / attention_weights.sum(axis=2, keepdims=True)
        
        summary = visualizer.get_attention_summary(attention_weights)
        
        assert 'n_heads' in summary
        assert 'seq_length' in summary
        assert 'avg_attention' in summary
        assert summary['n_heads'] == 8
        assert summary['seq_length'] == 20
    
    def test_top_attended_positions(self):
        """Test top attended positions extraction"""
        visualizer = AttentionVisualizer()
        
        attention_weights = np.random.rand(4, 10, 10)
        attention_weights = attention_weights / attention_weights.sum(axis=2, keepdims=True)
        
        top_positions = visualizer.get_top_attended_positions(
            attention_weights=attention_weights,
            query_position=5,
            top_k=3
        )
        
        assert len(top_positions) == 3
        assert all('position' in p for p in top_positions)
        assert all('attention_score' in p for p in top_positions)
    
    def test_analyze_patterns(self):
        """Test attention pattern analysis"""
        visualizer = AttentionVisualizer()
        
        attention_weights = np.random.rand(4, 10, 10)
        attention_weights = attention_weights / attention_weights.sum(axis=2, keepdims=True)
        
        patterns = visualizer.analyze_attention_patterns(attention_weights)
        
        assert 'heads' in patterns
        assert len(patterns['heads']) == 4
        
        for head in patterns['heads']:
            assert 'pattern_type' in head
            assert 'diagonal_strength' in head
            assert 'uniformity' in head
    
    def test_heatmap_data(self):
        """Test heatmap data preparation"""
        visualizer = AttentionVisualizer()
        
        attention_weights = np.random.rand(4, 10, 10)
        attention_weights = attention_weights / attention_weights.sum(axis=2, keepdims=True)
        
        heatmap_data = visualizer.create_attention_heatmap_data(
            attention_weights=attention_weights,
            head_idx=0
        )
        
        assert 'data' in heatmap_data
        assert 'x_labels' in heatmap_data
        assert 'y_labels' in heatmap_data
        assert 'title' in heatmap_data
    
    def test_temporal_flow(self):
        """Test temporal attention flow analysis"""
        visualizer = AttentionVisualizer()
        
        attention_weights = np.random.rand(4, 10, 10)
        attention_weights = attention_weights / attention_weights.sum(axis=2, keepdims=True)
        
        flow = visualizer.get_temporal_attention_flow(attention_weights)
        
        assert 'flow_by_position' in flow
        assert 'overall_past_bias' in flow
        assert 'overall_current_bias' in flow
        assert 'overall_future_bias' in flow
        assert len(flow['flow_by_position']) == 10


# Explanation Generator Tests
class TestExplanationGenerator:
    """Test explanation generator"""
    
    def test_generator_creation(self):
        """Test generator initialization"""
        generator = ExplanationGenerator()
        assert generator is not None
    
    def test_prediction_explanation(self):
        """Test prediction explanation generation"""
        generator = ExplanationGenerator()
        
        top_features = [
            {
                'feature_name': 'RSI',
                'feature_value': 65.0,
                'contribution': 0.5,
                'contribution_percentage': 40.0,
                'direction': 'positive'
            },
            {
                'feature_name': 'MACD',
                'feature_value': 1.2,
                'contribution': 0.3,
                'contribution_percentage': 25.0,
                'direction': 'positive'
            }
        ]
        
        explanation = generator.generate_prediction_explanation(
            prediction=152.50,
            base_value=150.00,
            top_features=top_features,
            confidence=0.85,
            prediction_type='price'
        )
        
        assert isinstance(explanation, str)
        assert len(explanation) > 0
        assert '152.50' in explanation
        assert 'confidence' in explanation.lower()
    
    def test_uncertainty_explanation(self):
        """Test uncertainty explanation generation"""
        generator = ExplanationGenerator()
        
        explanation = generator.generate_uncertainty_explanation(
            confidence=0.75,
            uncertainty_factors={'model_agreement': 0.8, 'data_quality': 0.7}
        )
        
        assert isinstance(explanation, str)
        assert 'confidence' in explanation.lower()
    
    def test_feature_importance_explanation(self):
        """Test feature importance explanation"""
        generator = ExplanationGenerator()
        
        feature_importance = {
            'RSI': 0.25,
            'MACD': 0.20,
            'Volume': 0.15
        }
        
        explanation = generator.generate_feature_importance_explanation(
            feature_importance=feature_importance,
            top_n=3
        )
        
        assert isinstance(explanation, str)
        assert 'RSI' in explanation
        assert 'MACD' in explanation
    
    def test_time_series_explanation(self):
        """Test time series explanation"""
        generator = ExplanationGenerator()
        
        predictions = [100.0, 102.0, 104.0, 103.0, 105.0]
        time_labels = ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5']
        
        explanation = generator.generate_time_series_explanation(
            predictions=predictions,
            time_labels=time_labels,
            trend='increasing'
        )
        
        assert isinstance(explanation, str)
        assert 'trend' in explanation.lower()
    
    def test_comparison_explanation(self):
        """Test comparison explanation"""
        generator = ExplanationGenerator()
        
        changed_features = [
            {
                'feature_name': 'RSI',
                'change': 0.5
            }
        ]
        
        explanation = generator.generate_comparison_explanation(
            prediction1=100.0,
            prediction2=105.0,
            changed_features=changed_features
        )
        
        assert isinstance(explanation, str)
        assert len(explanation) > 0


# Integration Tests
class TestExplainabilityIntegration:
    """Test integration of explainability components"""
    
    def test_complete_workflow(self, trained_model, sample_data, feature_names):
        """Test complete explainability workflow"""
        X, _ = sample_data
        
        try:
            # 1. SHAP Explainer
            explainer = SHAPExplainer(
                model=trained_model,
                background_data=X[:20],
                feature_names=feature_names,
                explainer_type='tree'
            )
            
            explanation = explainer.explain_prediction(X[:5], sample_idx=0, n_features=5)
            
            # 2. Feature Analyzer
            analyzer = FeatureContributionAnalyzer(feature_names=feature_names)
            
            shap_result = explainer.explain(X[:5])
            analysis = analyzer.analyze_contributions(
                feature_values=X[0],
                contributions=shap_result['shap_values'][0],
                top_n=5
            )
            
            # 3. Explanation Generator
            generator = ExplanationGenerator()
            
            text_explanation = generator.generate_prediction_explanation(
                prediction=explanation['prediction'],
                base_value=explanation['base_value'],
                top_features=explanation['top_features'],
                confidence=0.85
            )
            
            # Verify all components worked
            assert explanation is not None
            assert analysis is not None
            assert text_explanation is not None
            assert len(text_explanation) > 0
            
        except ImportError:
            pytest.skip("SHAP not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
