# Explainable AI Module

This module provides comprehensive explainability features for ARA AI predictions, including SHAP values, feature contribution analysis, attention visualization, and natural language explanations.

## Features

### 1. SHAP Explainer (`shap_explainer.py`)
- **SHAP Values**: Calculate Shapley Additive Explanations for any model
- **Feature Importance**: Rank features by their impact on predictions
- **Top Contributors**: Identify the most influential features for each prediction
- **Model-Agnostic**: Works with tree-based, linear, and deep learning models
- **Interaction Values**: Analyze feature interactions (for tree models)

### 2. Feature Contribution Analyzer (`feature_analyzer.py`)
- **Contribution Analysis**: Calculate top 5 contributing factors for predictions
- **Percentage Calculation**: Show contribution as percentage of total impact
- **Factor Descriptions**: Generate human-readable descriptions
- **Comparison**: Compare contributions between different predictions
- **Grouping**: Group features by category for analysis

### 3. Attention Visualizer (`attention_visualizer.py`)
- **Attention Extraction**: Extract attention weights from Transformer models
- **Pattern Analysis**: Identify attention patterns (local, global, recency-biased)
- **Temporal Flow**: Analyze how attention flows through time
- **Heatmap Data**: Prepare data for attention heatmap visualization
- **Top Positions**: Find which positions receive most attention

### 4. Explanation Generator (`explanation_generator.py`)
- **Natural Language**: Generate human-readable explanations
- **Prediction Explanations**: Explain why a specific prediction was made
- **Confidence Explanations**: Explain prediction uncertainty
- **Comparison Explanations**: Compare two predictions
- **Time Series Explanations**: Explain trends over time

### 5. Visualizer (`visualizer.py`)
- **Bar Charts**: Visualize feature contributions
- **Heatmaps**: Display attention weights and correlations
- **Waterfall Plots**: Show cumulative feature contributions
- **Comparison Charts**: Side-by-side contribution comparisons
- **Importance Rankings**: Visualize feature importance

## Installation

```bash
# Install required dependencies
pip install shap matplotlib seaborn
```

## Quick Start

### Basic SHAP Explanation

```python
from ara.explainability import SHAPExplainer
import numpy as np

# Assume you have a trained model and data
model = your_trained_model
X = your_feature_data  # shape: (n_samples, n_features)
feature_names = ['RSI', 'MACD', 'Volume', 'SMA_20', 'ATR']

# Create explainer
explainer = SHAPExplainer(
    model=model,
    background_data=X[:100],  # Use subset as background
    feature_names=feature_names,
    explainer_type='auto'  # Automatically select best explainer
)

# Explain a prediction
explanation = explainer.explain_prediction(X, sample_idx=0, n_features=5)

print(f"Prediction: {explanation['prediction']}")
print(f"Base value: {explanation['base_value']}")
print("\nTop contributing features:")
for feature in explanation['top_features']:
    print(f"  {feature['feature_name']}: {feature['contribution']:.4f}")
```

### Feature Contribution Analysis

```python
from ara.explainability import FeatureContributionAnalyzer

# Create analyzer
analyzer = FeatureContributionAnalyzer(
    feature_names=feature_names,
    feature_descriptions={
        'RSI': 'Relative Strength Index (momentum indicator)',
        'MACD': 'Moving Average Convergence Divergence',
        'Volume': 'Trading volume',
        'SMA_20': '20-day Simple Moving Average',
        'ATR': 'Average True Range (volatility)'
    }
)

# Analyze contributions
feature_values = X[0]  # Feature values for one sample
contributions = shap_values[0]  # SHAP values for one sample

analysis = analyzer.analyze_contributions(
    feature_values=feature_values,
    contributions=contributions,
    top_n=5
)

print("Top 5 Contributing Features:")
for feature in analysis['top_features']:
    print(f"{feature['rank']}. {feature['feature_name']}")
    print(f"   Value: {feature['feature_value']:.4f}")
    print(f"   Contribution: {feature['contribution']:.4f}")
    print(f"   Percentage: {feature['contribution_percentage']:.1f}%")
    print(f"   Direction: {feature['direction']}")
```

### Natural Language Explanations

```python
from ara.explainability import ExplanationGenerator

# Create generator
generator = ExplanationGenerator(
    feature_descriptions={
        'RSI': 'Relative Strength Index',
        'MACD': 'MACD indicator',
        'Volume': 'Trading volume'
    }
)

# Generate explanation
explanation_text = generator.generate_prediction_explanation(
    prediction=152.50,
    base_value=150.00,
    top_features=analysis['top_features'],
    confidence=0.85,
    prediction_type='price'
)

print(explanation_text)
```

**Output:**
```
The model predicts a price of $152.50.
This prediction has high confidence (85.0%).
This is 2.50 above the baseline expectation of 150.00.

Key factors driving this prediction:
1. Relative Strength Index (value: 65.2341) strongly increases the prediction, contributing 35.2% of the total upward pressure.
2. MACD indicator (value: 1.2500) moderately increases the prediction, contributing 22.8% of the total upward pressure.
3. Trading volume (value: 1250000.0000) slightly increases the prediction, contributing 15.3% of the total upward pressure.

Overall, 3 factors push the prediction higher.
```

### Attention Visualization (for Transformer models)

```python
from ara.explainability import AttentionVisualizer

# Create visualizer
visualizer = AttentionVisualizer(model=transformer_model)

# Register hooks to capture attention
visualizer.register_hooks()

# Extract attention weights
attention_weights = visualizer.extract_attention_weights(
    model=transformer_model,
    input_data=X_sequence
)

# Analyze attention patterns
patterns = visualizer.analyze_attention_patterns(attention_weights['layer_0'])

print("Attention Pattern Analysis:")
for head in patterns['heads']:
    print(f"Head {head['head_idx']}: {head['pattern_type']}")
    print(f"  Diagonal strength: {head['diagonal_strength']:.3f}")
    print(f"  Uniformity: {head['uniformity']:.3f}")
```

### Create Visualizations

```python
from ara.explainability import ExplainabilityVisualizer

# Create visualizer
viz = ExplainabilityVisualizer()

# Plot feature contributions
fig = viz.plot_feature_contributions(
    top_features=analysis['top_features'],
    title="Top Feature Contributions to Prediction",
    save_path="feature_contributions.png"
)

# Plot contribution percentages
fig = viz.plot_contribution_percentages(
    top_features=analysis['top_features'],
    title="Feature Contribution Percentages",
    save_path="contribution_percentages.png"
)

# Plot attention heatmap
fig = viz.plot_attention_heatmap(
    attention_weights=attention_matrix,
    x_labels=[f"t-{i}" for i in range(60, 0, -1)],
    y_labels=[f"t-{i}" for i in range(60, 0, -1)],
    title="Transformer Attention Heatmap",
    save_path="attention_heatmap.png"
)

# Plot feature importance ranking
feature_importance = explainer.get_feature_importance(X)
fig = viz.plot_feature_importance_ranking(
    feature_importance=feature_importance,
    top_n=20,
    title="Overall Feature Importance",
    save_path="feature_importance.png"
)
```

## Complete Example

```python
from ara.explainability import (
    SHAPExplainer,
    FeatureContributionAnalyzer,
    ExplanationGenerator,
    ExplainabilityVisualizer
)
import numpy as np

# Setup
model = your_trained_model
X = your_data
feature_names = ['RSI', 'MACD', 'Volume', 'SMA_20', 'ATR']

# 1. Calculate SHAP values
explainer = SHAPExplainer(
    model=model,
    background_data=X[:100],
    feature_names=feature_names
)

shap_explanation = explainer.explain_prediction(X, sample_idx=0, n_features=5)

# 2. Analyze contributions
analyzer = FeatureContributionAnalyzer(feature_names=feature_names)
contribution_analysis = analyzer.analyze_contributions(
    feature_values=X[0],
    contributions=shap_explanation['top_features'][0]['shap_value'],
    top_n=5
)

# 3. Generate natural language explanation
generator = ExplanationGenerator()
text_explanation = generator.generate_prediction_explanation(
    prediction=shap_explanation['prediction'],
    base_value=shap_explanation['base_value'],
    top_features=shap_explanation['top_features'],
    confidence=0.85
)

print(text_explanation)

# 4. Create visualizations
viz = ExplainabilityVisualizer()
viz.plot_feature_contributions(
    top_features=shap_explanation['top_features'],
    save_path="explanation.png"
)

# 5. Get feature importance
feature_importance = explainer.get_feature_importance(X)
print("\nOverall Feature Importance:")
for feature, importance in list(feature_importance.items())[:10]:
    print(f"  {feature}: {importance:.4f}")
```

## API Reference

### SHAPExplainer

#### Methods

- `explain(X)`: Calculate SHAP values for predictions
- `get_feature_importance(X, method='mean_abs')`: Get overall feature importance
- `get_top_features(X, n=5, sample_idx=0)`: Get top N contributing features
- `explain_prediction(X, sample_idx=0, n_features=5)`: Comprehensive explanation
- `get_interaction_values(X, feature_idx1, feature_idx2)`: Feature interactions

### FeatureContributionAnalyzer

#### Methods

- `analyze_contributions(feature_values, contributions, top_n=5)`: Analyze contributions
- `calculate_contribution_percentages(contributions)`: Calculate percentages
- `get_top_n_features(feature_values, contributions, n=5)`: Get top features
- `generate_factor_descriptions(top_features)`: Generate descriptions
- `compare_contributions(contributions1, contributions2)`: Compare scenarios
- `rank_features_by_importance(contributions, method='mean_abs')`: Rank features

### AttentionVisualizer

#### Methods

- `register_hooks(model)`: Register hooks to capture attention
- `extract_attention_weights(model, input_data)`: Extract attention weights
- `get_attention_summary(attention_weights)`: Get attention statistics
- `get_top_attended_positions(attention_weights, query_position, top_k=5)`: Top positions
- `analyze_attention_patterns(attention_weights)`: Analyze patterns
- `create_attention_heatmap_data(attention_weights)`: Prepare heatmap data

### ExplanationGenerator

#### Methods

- `generate_prediction_explanation(prediction, base_value, top_features, confidence)`: Explain prediction
- `generate_comparison_explanation(prediction1, prediction2, changed_features)`: Compare predictions
- `generate_feature_importance_explanation(feature_importance, top_n=10)`: Explain importance
- `generate_uncertainty_explanation(confidence, uncertainty_factors)`: Explain uncertainty
- `generate_time_series_explanation(predictions, time_labels, trend)`: Explain time series
- `generate_full_explanation(prediction_data)`: Comprehensive explanation

### ExplainabilityVisualizer

#### Methods

- `plot_feature_contributions(top_features)`: Bar chart of contributions
- `plot_contribution_percentages(top_features)`: Percentage bar chart
- `plot_shap_waterfall(base_value, shap_values, feature_values, feature_names)`: Waterfall plot
- `plot_attention_heatmap(attention_weights, x_labels, y_labels)`: Attention heatmap
- `plot_feature_importance_ranking(feature_importance, top_n=20)`: Importance ranking
- `plot_contribution_comparison(contributions1, contributions2, feature_names)`: Compare scenarios

## Best Practices

1. **Use Background Data**: For KernelExplainer, use a representative subset (100-1000 samples)
2. **Feature Names**: Always provide meaningful feature names for better explanations
3. **Feature Descriptions**: Add human-readable descriptions for clearer explanations
4. **Visualization**: Save visualizations to files for reports and documentation
5. **Confidence**: Always include confidence scores in explanations
6. **Validation**: Verify SHAP values sum to prediction - base_value
7. **Performance**: Use TreeExplainer for tree models (much faster than KernelExplainer)

## Troubleshooting

### SHAP Installation Issues
```bash
# If SHAP installation fails, try:
pip install shap --no-cache-dir

# Or install from source:
pip install git+https://github.com/slundberg/shap.git
```

### Matplotlib Backend Issues
```python
# If you get display errors, use non-interactive backend:
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
```

### Memory Issues with SHAP
```python
# For large datasets, use smaller background data:
background_data = X[:100]  # Use only 100 samples

# Or use TreeExplainer which is more efficient:
explainer = SHAPExplainer(model=model, explainer_type='tree')
```

## Requirements

- Python 3.8+
- numpy >= 1.20.0
- shap >= 0.42.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- torch >= 2.0.0 (for attention visualization)

## License

Part of the ARA AI prediction system.
