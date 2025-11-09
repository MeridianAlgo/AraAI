#!/usr/bin/env python3
"""
Enhanced AI Analysis Demo
Demonstrates the comprehensive AI-powered company analysis features
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from meridianalgo.ai_analysis import LightweightAIAnalyzer
from meridianalgo.console import ConsoleManager
import time

def demo_ai_models():
    """Demo AI model capabilities"""
    print(" AI MODEL CAPABILITIES DEMO")
    print("=" * 60)
    
    console = ConsoleManager(verbose=True)
    
    # Initialize AI analyzer with GPU support
    console.print_info("Initializing AI analyzer with GPU support...")
    ai_analyzer = LightweightAIAnalyzer(use_gpu=True)
    
    # Show model information
    if hasattr(ai_analyzer, 'model_configs'):
        console.print_info("Available AI Models:")
        for model_type, config in ai_analyzer.model_configs.items():
            print(f"   {model_type.title()}: {config['model_name']}")
            print(f"     Size: {config['size_mb']}MB | Accuracy: {config['accuracy']}")
    
    return ai_analyzer

def demo_comprehensive_analysis():
    """Demo comprehensive AI analysis"""
    print("\n COMPREHENSIVE AI ANALYSIS DEMO")
    print("=" * 60)
    
    console = ConsoleManager(verbose=True)
    ai_analyzer = LightweightAIAnalyzer(use_gpu=True)
    
    # Test symbols with different characteristics
    test_symbols = [
        ('AAPL', 'Large-cap tech leader'),
        ('TSLA', 'High-growth EV company'),
        ('JNJ', 'Defensive healthcare'),
        ('XOM', 'Energy sector cyclical')
    ]
    
    for symbol, description in test_symbols:
        try:
            console.print_info(f"Analyzing {symbol} ({description})...")
            
            start_time = time.time()
            analysis = ai_analyzer.analyze_company_with_ai(symbol)
            analysis_time = time.time() - start_time
            
            if analysis and 'error' not in analysis:
                print(f" Analysis completed in {analysis_time:.2f}s")
                
                # Show key AI insights
                if 'ai_insights' in analysis:
                    print(" AI Insights:")
                    for insight in analysis['ai_insights'][:3]:
                        print(f"   • {insight}")
                
                # Show AI sentiment
                if 'ai_sentiment' in analysis:
                    sentiment = analysis['ai_sentiment']
                    print(f" AI Sentiment: {sentiment.get('sentiment', 'N/A').upper()} ({sentiment.get('confidence', 0)*100:.0f}% confidence)")
                
                # Show AI recommendation
                if 'ai_recommendation' in analysis:
                    rec = analysis['ai_recommendation']
                    print(f" AI Recommendation: {rec.get('recommendation', 'N/A')} ({rec.get('confidence', 0)*100:.0f}% confidence)")
                
                # Show overall score
                print(f" Overall AI Score: {analysis.get('overall_score', 0):.1f}/100")
                
            else:
                error_msg = analysis.get('error', 'Unknown error') if analysis else 'Analysis failed'
                console.print_error(f"Analysis failed: {error_msg}")
            
            print("-" * 40)
            time.sleep(1)  # Brief pause between analyses
            
        except Exception as e:
            console.print_error(f"Error analyzing {symbol}: {e}")

def demo_ai_features():
    """Demo specific AI features"""
    print("\n AI FEATURE BREAKDOWN DEMO")
    print("=" * 60)
    
    console = ConsoleManager(verbose=True)
    ai_analyzer = LightweightAIAnalyzer(use_gpu=True)
    
    symbol = 'MSFT'
    console.print_info(f"Detailed AI feature analysis for {symbol}...")
    
    try:
        analysis = ai_analyzer.analyze_company_with_ai(symbol)
        
        if analysis and 'error' not in analysis:
            # Financial Sentiment Analysis
            if 'ai_sentiment' in analysis:
                sentiment = analysis['ai_sentiment']
                console.print_info(" Financial Sentiment Analysis:")
                print(f"   Overall: {sentiment.get('sentiment', 'N/A').upper()}")
                print(f"   Confidence: {sentiment.get('confidence', 0)*100:.1f}%")
                print(f"   Reasoning: {sentiment.get('reasoning', 'N/A')}")
                
                if 'detailed_sentiments' in sentiment:
                    print("   Detailed Analysis:")
                    for detail in sentiment['detailed_sentiments']:
                        print(f"     • {detail['aspect']}: {detail['sentiment']} ({detail['confidence']*100:.0f}%)")
            
            # Risk Assessment
            if 'ai_risk_assessment' in analysis:
                risk = analysis['ai_risk_assessment']
                console.print_info(" AI Risk Assessment:")
                print(f"   Risk Score: {risk.get('risk_score', 0):.1f}/100")
                print(f"   Primary Risk: {risk.get('primary_risk_category', 'N/A')}")
                
                if 'risk_factors' in risk:
                    print("   Risk Factors:")
                    for factor in risk['risk_factors']:
                        print(f"     • {factor['type']}: {factor['level']} (Impact: {factor['impact']*100:.0f}%)")
            
            # Growth Analysis
            if 'ai_growth_potential' in analysis:
                growth = analysis['ai_growth_potential']
                console.print_info(" AI Growth Analysis:")
                print(f"   Growth Category: {growth.get('growth_category', 'N/A')}")
                print(f"   Confidence: {growth.get('growth_confidence', 0)*100:.1f}%")
                print(f"   Reasoning: {growth.get('growth_reasoning', 'N/A')}")
            
            # Competitive Analysis
            if 'ai_competitive_analysis' in analysis:
                comp = analysis['ai_competitive_analysis']
                console.print_info(" AI Competitive Analysis:")
                print(f"   Market Position: {comp.get('market_position', 'N/A')}")
                print(f"   Moat Strength: {comp.get('moat_strength', 'N/A')}")
                
                if 'competitive_advantages' in comp:
                    print("   Competitive Advantages:")
                    for advantage in comp['competitive_advantages']:
                        print(f"     • {advantage}")
            
            # ESG Analysis
            if 'ai_esg_analysis' in analysis:
                esg = analysis['ai_esg_analysis']
                console.print_info(" AI ESG Analysis:")
                print(f"   ESG Score: {esg.get('esg_score', 0):.1f}/100")
                
                if 'governance_factors' in esg:
                    print("   Governance Factors:")
                    for factor in esg['governance_factors']:
                        print(f"     • {factor}")
            
            # Market Outlook
            if 'ai_market_outlook' in analysis:
                outlook = analysis['ai_market_outlook']
                console.print_info(" AI Market Outlook:")
                print(f"   Market Trend: {outlook.get('market_trend', 'N/A')}")
                print(f"   Sector Outlook: {outlook.get('sector_outlook', 'N/A')}")
                
                if 'key_drivers' in outlook:
                    print("   Key Drivers:")
                    for driver in outlook['key_drivers']:
                        print(f"     • {driver}")
            
            # AI Price Targets
            if 'ai_price_targets' in analysis:
                targets = analysis['ai_price_targets']
                console.print_info(" AI Price Targets:")
                print(f"   Bull Case: ${targets.get('bull_case_target', 0):.2f}")
                print(f"   Base Case: ${targets.get('base_case_target', 0):.2f}")
                print(f"   Bear Case: ${targets.get('bear_case_target', 0):.2f}")
                print(f"   Time Horizon: {targets.get('time_horizon', 'N/A')}")
                print(f"   Confidence: {targets.get('confidence', 0)*100:.1f}%")
        
        else:
            error_msg = analysis.get('error', 'Unknown error') if analysis else 'Analysis failed'
            console.print_error(f"Detailed analysis failed: {error_msg}")
    
    except Exception as e:
        console.print_error(f"Feature demo failed: {e}")

def demo_performance_comparison():
    """Demo performance comparison between CPU and GPU"""
    print("\n PERFORMANCE COMPARISON DEMO")
    print("=" * 60)
    
    console = ConsoleManager(verbose=True)
    
    symbol = 'GOOGL'
    
    # Test CPU performance
    console.print_info("Testing CPU performance...")
    start_time = time.time()
    ai_analyzer_cpu = LightweightAIAnalyzer(use_gpu=False)
    cpu_analysis = ai_analyzer_cpu.analyze_company_with_ai(symbol)
    cpu_time = time.time() - start_time
    
    # Test GPU performance (if available)
    console.print_info("Testing GPU performance...")
    start_time = time.time()
    ai_analyzer_gpu = LightweightAIAnalyzer(use_gpu=True)
    gpu_analysis = ai_analyzer_gpu.analyze_company_with_ai(symbol)
    gpu_time = time.time() - start_time
    
    # Compare results
    print("\n Performance Results:")
    print(f"CPU Analysis Time: {cpu_time:.2f}s")
    print(f"GPU Analysis Time: {gpu_time:.2f}s")
    
    if gpu_time < cpu_time:
        speedup = cpu_time / gpu_time
        print(f" GPU Speedup: {speedup:.1f}x faster")
    else:
        print(" CPU performed similarly or better")
    
    # Compare analysis quality
    cpu_score = cpu_analysis.get('overall_score', 0) if cpu_analysis else 0
    gpu_score = gpu_analysis.get('overall_score', 0) if gpu_analysis else 0
    
    print(f"\n Analysis Quality:")
    print(f"CPU Analysis Score: {cpu_score:.1f}/100")
    print(f"GPU Analysis Score: {gpu_score:.1f}/100")

def demo_model_accuracy():
    """Demo model accuracy with known companies"""
    print("\n MODEL ACCURACY DEMO")
    print("=" * 60)
    
    console = ConsoleManager(verbose=True)
    ai_analyzer = LightweightAIAnalyzer(use_gpu=True)
    
    # Test with companies with known characteristics
    test_cases = [
        {
            'symbol': 'AAPL',
            'expected_sentiment': 'positive',
            'expected_growth': 'moderate',
            'expected_risk': 'low'
        },
        {
            'symbol': 'NFLX',
            'expected_sentiment': 'mixed',
            'expected_growth': 'moderate',
            'expected_risk': 'moderate'
        }
    ]
    
    correct_predictions = 0
    total_predictions = 0
    
    for test_case in test_cases:
        symbol = test_case['symbol']
        console.print_info(f"Testing accuracy for {symbol}...")
        
        try:
            analysis = ai_analyzer.analyze_company_with_ai(symbol)
            
            if analysis and 'error' not in analysis:
                # Check sentiment accuracy
                ai_sentiment = analysis.get('ai_sentiment', {}).get('sentiment', 'neutral')
                expected_sentiment = test_case['expected_sentiment']
                
                sentiment_correct = (
                    (ai_sentiment == 'positive' and expected_sentiment == 'positive') or
                    (ai_sentiment == 'negative' and expected_sentiment == 'negative') or
                    (ai_sentiment == 'neutral' and expected_sentiment in ['mixed', 'neutral'])
                )
                
                if sentiment_correct:
                    correct_predictions += 1
                    print(f"    Sentiment: {ai_sentiment} (Expected: {expected_sentiment})")
                else:
                    print(f"    Sentiment: {ai_sentiment} (Expected: {expected_sentiment})")
                
                total_predictions += 1
                
                # Show other predictions
                growth = analysis.get('ai_growth_potential', {}).get('growth_category', 'unknown')
                risk_score = analysis.get('ai_risk_assessment', {}).get('risk_score', 50)
                
                print(f"    Growth: {growth}")
                print(f"    Risk Score: {risk_score:.1f}/100")
            
        except Exception as e:
            console.print_error(f"Accuracy test failed for {symbol}: {e}")
    
    if total_predictions > 0:
        accuracy = (correct_predictions / total_predictions) * 100
        print(f"\n Model Accuracy: {accuracy:.1f}% ({correct_predictions}/{total_predictions})")
    else:
        print("\n No predictions to evaluate")

def main():
    """Run AI analysis demo"""
    print(" ARA AI - ENHANCED AI ANALYSIS DEMO")
    print("Comprehensive AI-Powered Company Analysis")
    print("=" * 60)
    
    demos = [
        ("AI Model Capabilities", demo_ai_models),
        ("Comprehensive Analysis", demo_comprehensive_analysis),
        ("AI Feature Breakdown", demo_ai_features),
        ("Performance Comparison", demo_performance_comparison),
        ("Model Accuracy Test", demo_model_accuracy)
    ]
    
    print("\nAvailable Demos:")
    for i, (name, _) in enumerate(demos, 1):
        print(f"  {i}. {name}")
    
    print("\nRunning all demos...")
    
    for name, demo_func in demos:
        try:
            print(f"\n Starting: {name}")
            demo_func()
            print(f" Completed: {name}")
        except KeyboardInterrupt:
            print(f"\n⏹ Demo interrupted by user")
            break
        except Exception as e:
            print(f" Error in {name}: {e}")
            continue
    
    print("\n" + "="*60)
    print(" AI ANALYSIS DEMO COMPLETED!")
    print("="*60)
    print("\n Key Features Demonstrated:")
    print("• FinBERT financial sentiment analysis")
    print("• Comprehensive risk assessment")
    print("• AI-powered growth analysis")
    print("• Competitive position evaluation")
    print("• ESG factor analysis")
    print("• Market outlook prediction")
    print("• AI-generated price targets")
    print("• GPU acceleration support")
    
    print("\n Usage Tips:")
    print("• Use 'python ara.py --ai-analysis SYMBOL' for AI analysis")
    print("• GPU acceleration improves performance significantly")
    print("• Models are cached locally for faster subsequent use")
    print("• Combine with technical analysis for complete picture")

if __name__ == "__main__":
    main()