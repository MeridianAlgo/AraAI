"""
Explainability Demo
Demonstrates all explainable AI features including SHAP, feature contributions,
attention visualization, and natural language explanations
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ara.explainability import (
    SHAPExplainer,
    FeatureContributionAnalyzer,
    AttentionVisualizer,
    ExplanationGenerator,
    ExplainabilityVisualizer
)


def create_sample_data(n_samples=1000, n_features=20):
    """Create sample data for demonstration"""
    np.random.seed(42)
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Create target with known relationships
    # Features 0-4 have strong positive impact
    # Features 5-9 have moderate negative impact
    # Features 10-19 have weak random impact
    y = (
        2.0 * X[:, 0] +  # Strong positive
        1.5 * X[:, 1] +
        1.2 * X[:, 2] +
        -1.0 * X[:, 5] +  # Moderate negative
        -0.8 * X[:, 6] +
        0.1 * np.random.randn(n_samples)  # Noise
    )
    
    return X, y


def demo_shap_explainer():
    """Demonstrate SHAP explainer"""
    print("=" * 80)
    print("SHAP EXPLAINER DEMO")
    print("=" * 80)
    
    # Create sample data
    X, y = create_sample_data()
    
    # Train a simple model
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    
    # Create feature names
    feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
    feature_names[0] = "RSI"
    feature_names[1] = "MACD"
    feature_names[2] = "Volume"
    feature_names[5] = "Volatility"
    feature_names[6] = "Momentum"
    
    print("\n1. Creating SHAP Explainer...")
    explainer = SHAPExplainer(
        model=model,
        background_data=X[:100],
        feature_names=feature_names,
        explainer_type='tree'  # Use TreeExplainer for Random Forest
    )
    print("   ✓ Explainer created")
    
    print("\n2. Explaining a single prediction...")
    explanation = explainer.explain_prediction(X, sample_idx=0, n_features=5)
    
    print(f"\n   Prediction: {explanation['prediction']:.4f}")
    print(f"   Base value: {explanation['base_value']:.4f}")
    print(f"   Total SHAP contribution: {explanation['total_shap_contribution']:.4f}")
    print(f"   Expected prediction: {explanation['expected_prediction']:.4f}")
    
    print("\n   Top 5 Contributing Features:")
    for feature in explanation['top_features']:
        print(f"     {feature['feature_name']:15s}: "
              f"value={feature['feature_value']:7.4f}, "
              f"contribution={feature['contribution']:7.4f}")
    
    print("\n3. Getting overall feature importance...")
    feature_importance = explainer.get_feature_importance(X[:100])
    
    print("\n   Top 10 Most Important Features:")
    for i, (feature, importance) in enumerate(list(feature_importance.items())[:10], 1):
        print(f"     {i:2d}. {feature:15s}: {importance:.6f}")
    
    return explainer, X, y, feature_names, explanation


def demo_feature_analyzer(X, feature_names, shap_values):
    """Demonstrate feature contribution analyzer"""
    print("\n" + "=" * 80)
    print("FEATURE CONTRIBUTION ANALYZER DEMO")
    print("=" * 80)
    
    # Create analyzer with descriptions
    feature_descriptions = {
        'RSI': 'Relative Strength Index (momentum indicator)',
        'MACD': 'Moving Average Convergence Divergence',
        'Volume': 'Trading volume',
        'Volatility': 'Price volatility measure',
        'Momentum': 'Price momentum indicator'
    }
    
    print("\n1. Creating Feature Contribution Analyzer...")
    analyzer = FeatureContributionAnalyzer(
        feature_names=feature_names,
        feature_descriptions=feature_descriptions
    )
    print("   ✓ Analyzer created")
    
    print("\n2. Analyzing contributions for a sample...")
    analysis = analyzer.analyze_contributions(
        feature_values=X[0],
        contributions=shap_values[0],
        top_n=5
    )
    
    print("\n   Top 5 Contributing Features:")
    for feature in analysis['top_features']:
        print(f"\n     Rank {feature['rank']}: {feature['feature_name']}")
        print(f"       Value: {feature['feature_value']:.4f}")
        print(f"       Contribution: {feature['contribution']:.4f}")
        print(f"       Percentage: {feature['contribution_percentage']:.1f}%")
        print(f"       Direction: {feature['direction']}")
        if feature['description']:
            print(f"       Description: {feature['description']}")
    
    print("\n   Summary Statistics:")
    summary = analysis['summary']
    print(f"     Total features: {summary['total_features']}")
    print(f"     Positive contribution: {summary['positive_contribution']:.4f}")
    print(f"     Negative contribution: {summary['negative_contribution']:.4f}")
    print(f"     Net contribution: {summary['net_contribution']:.4f}")
    print(f"     Top 5 account for: {summary['top_n_percentage']:.1f}% of total impact")
    
    print("\n3. Generating factor descriptions...")
    descriptions = analyzer.generate_factor_descriptions(analysis['top_features'])
    print("\n   Human-readable descriptions:")
    for desc in descriptions:
        print(f"     • {desc}")
    
    print("\n4. Ranking all features by importance...")
    ranked_features = analyzer.rank_features_by_importance(
        shap_values[:100],
        method='mean_abs'
    )
    
    print("\n   Top 10 Features by Importance:")
    for i, (feature, score) in enumerate(ranked_features[:10], 1):
        print(f"     {i:2d}. {feature:15s}: {score:.6f}")
    
    return analyzer, analysis


def demo_explanation_generator(explanation, analysis):
    """Demonstrate natural language explanation generator"""
    print("\n" + "=" * 80)
    print("EXPLANATION GENERATOR DEMO")
    print("=" * 80)
    
    # Create generator
    feature_descriptions = {
        'RSI': 'Relative Strength Index',
        'MACD': 'MACD indicator',
        'Volume': 'Trading volume',
        'Volatility': 'Price volatility',
        'Momentum': 'Price momentum'
    }
    
    print("\n1. Creating Explanation Generator...")
    generator = ExplanationGenerator(
        feature_descriptions=feature_descriptions
    )
    print("   ✓ Generator created")
    
    print("\n2. Generating prediction explanation...")
    pred_explanation = generator.generate_prediction_explanation(
        prediction=explanation['prediction'],
        base_value=explanation['base_value'],
        top_features=explanation['top_features'],
        confidence=0.85,
        prediction_type='price'
    )
    
    print("\n" + "-" * 80)
    print(pred_explanation)
    print("-" * 80)
    
    print("\n3. Generating uncertainty explanation...")
    uncertainty_explanation = generator.generate_uncertainty_explanation(
        confidence=0.85,
        uncertainty_factors={
            'model_agreement': 0.92,
            'data_quality': 0.88,
            'regime_stability': 0.75
        }
    )
    
    print("\n" + "-" * 80)
    print(uncertainty_explanation)
    print("-" * 80)
    
    print("\n4. Generating feature importance explanation...")
    feature_importance = {
        'RSI': 0.25,
        'MACD': 0.18,
        'Volume': 0.15,
        'Volatility': 0.12,
        'Momentum': 0.10,
        'Feature_7': 0.08,
        'Feature_8': 0.06,
        'Feature_9': 0.04,
        'Feature_10': 0.02
    }
    
    importance_explanation = generator.generate_feature_importance_explanation(
        feature_importance=feature_importance,
        top_n=5
    )
    
    print("\n" + "-" * 80)
    print(importance_explanation)
    print("-" * 80)
    
    print("\n5. Generating time series explanation...")
    predictions = [100.0, 102.5, 105.0, 103.5, 106.0, 108.5, 107.0]
    time_labels = ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7']
    
    ts_explanation = generator.generate_time_series_explanation(
        predictions=predictions,
        time_labels=time_labels,
        trend='increasing'
    )
    
    print("\n" + "-" * 80)
    print(ts_explanation)
    print("-" * 80)
    
    return generator


def demo_visualizer(analysis, feature_importance):
    """Demonstrate explainability visualizer"""
    print("\n" + "=" * 80)
    print("EXPLAINABILITY VISUALIZER DEMO")
    print("=" * 80)
    
    try:
        print("\n1. Creating Explainability Visualizer...")
        viz = ExplainabilityVisualizer()
        print("   ✓ Visualizer created")
        
        print("\n2. Creating feature contribution bar chart...")
        fig = viz.plot_feature_contributions(
            top_features=analysis['top_features'],
            title="Top Feature Contributions",
            save_path="feature_contributions.png"
        )
        if fig:
            print("   ✓ Saved to feature_contributions.png")
        
        print("\n3. Creating contribution percentage chart...")
        fig = viz.plot_contribution_percentages(
            top_features=analysis['top_features'],
            title="Feature Contribution Percentages",
            save_path="contribution_percentages.png"
        )
        if fig:
            print("   ✓ Saved to contribution_percentages.png")
        
        print("\n4. Creating feature importance ranking chart...")
        fig = viz.plot_feature_importance_ranking(
            feature_importance=feature_importance,
            top_n=15,
            title="Overall Feature Importance Ranking",
            save_path="feature_importance_ranking.png"
        )
        if fig:
            print("   ✓ Saved to feature_importance_ranking.png")
        
        print("\n5. Creating attention heatmap (simulated)...")
        # Create simulated attention weights
        attention_weights = np.random.rand(10, 10)
        attention_weights = attention_weights / attention_weights.sum(axis=1, keepdims=True)
        
        fig = viz.plot_attention_heatmap(
            attention_weights=attention_weights,
            x_labels=[f"t-{i}" for i in range(10, 0, -1)],
            y_labels=[f"t-{i}" for i in range(10, 0, -1)],
            title="Simulated Attention Heatmap",
            save_path="attention_heatmap.png"
        )
        if fig:
            print("   ✓ Saved to attention_heatmap.png")
        
        print("\n6. Closing all figures...")
        viz.close_all()
        print("   ✓ All figures closed")
        
        print("\n✓ All visualizations created successfully!")
        print("  Check the current directory for PNG files.")
        
    except Exception as e:
        print(f"\n⚠ Visualization demo skipped: {e}")
        print("  Install matplotlib and seaborn to enable visualizations:")
        print("  pip install matplotlib seaborn")


def demo_complete_workflow():
    """Demonstrate complete explainability workflow"""
    print("\n" + "=" * 80)
    print("COMPLETE EXPLAINABILITY WORKFLOW")
    print("=" * 80)
    
    print("\nThis workflow demonstrates how to use all explainability components together")
    print("to create comprehensive, interpretable predictions.\n")
    
    # Step 1: SHAP Explainer
    explainer, X, y, feature_names, explanation = demo_shap_explainer()
    
    # Get SHAP values for analysis
    shap_result = explainer.explain(X[:100])
    shap_values = shap_result['shap_values']
    
    # Step 2: Feature Analyzer
    analyzer, analysis = demo_feature_analyzer(X, feature_names, shap_values)
    
    # Step 3: Explanation Generator
    generator = demo_explanation_generator(explanation, analysis)
    
    # Step 4: Visualizer
    feature_importance = explainer.get_feature_importance(X[:100])
    demo_visualizer(analysis, feature_importance)
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETE!")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("  1. SHAP values provide model-agnostic explanations")
    print("  2. Feature contribution analysis identifies key drivers")
    print("  3. Natural language explanations make results accessible")
    print("  4. Visualizations help communicate insights")
    print("\nFor production use, integrate these components into your prediction pipeline.")


def main():
    """Main demo function"""
    print("\n" + "=" * 80)
    print("ARA AI EXPLAINABILITY MODULE DEMO")
    print("=" * 80)
    print("\nThis demo showcases all explainability features:")
    print("  • SHAP value calculation")
    print("  • Feature contribution analysis")
    print("  • Natural language explanations")
    print("  • Attention visualization (for Transformers)")
    print("  • Visual explanations (charts and heatmaps)")
    
    try:
        demo_complete_workflow()
        
        print("\n" + "=" * 80)
        print("SUCCESS!")
        print("=" * 80)
        print("\nAll explainability features demonstrated successfully.")
        print("Check the generated PNG files for visualizations.")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
