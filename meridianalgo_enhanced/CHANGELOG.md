# Changelog

All notable changes to this project will be documented in this file.

## [0.3.0] - 2025-01-28

### 🚀 Major Changes
- **BREAKING**: Removed all API key dependencies - now uses pure Yahoo Finance
- **Zero Setup**: No API keys or registration required
- **Enhanced Validation**: Multi-tier accuracy system with intelligent failsafes

### ✨ New Features
- **Yahoo Finance Integration**: Complete migration to free Yahoo Finance data
- **Prediction Validation**: 6-layer validation system with failsafes
- **Multi-Tier Accuracy**: Excellent (<1%), Good (<2%), Acceptable (<3%) thresholds
- **Conservative Fallbacks**: Ultra-safe predictions when validation fails
- **Enhanced AI Analyzer**: Yahoo Finance-based insights without external APIs

### 🛡️ Prediction Failsafes
- Extreme change detection (>50% flagged as unreliable)
- Consistency validation (smooth day-to-day transitions)
- Confidence thresholds (minimum 60% required)
- Volatility context analysis (stable vs volatile stocks)
- Volume validation (low-volume stock adjustments)
- Conservative fallbacks when validation fails

### 🔧 Technical Improvements
- Removed `requests` dependency for API calls
- Enhanced error handling and user feedback
- Improved prediction constraints and reality checks
- Better confidence calculation algorithms
- Stricter validation thresholds (removed 5% threshold)

### 📦 Dependencies Updated
- Added `python-dotenv>=1.0.0`
- Added `rich>=13.0.0` for better UI
- Added `scipy>=1.9.0` for advanced calculations
- Removed `requests` dependency
- Updated minimum versions for better stability

### 🐛 Bug Fixes
- Fixed confidence variable scope issues
- Fixed data variable naming conflicts
- Resolved runtime errors in prediction validation
- Improved error messages and user feedback

## [0.2.1] - 2024-12-15

### 🔧 Bug Fixes
- Fixed installation issues with PyTorch dependencies
- Improved error handling in ensemble models
- Better compatibility with different Python versions

### 📚 Documentation
- Updated README with better examples
- Added more detailed API documentation
- Improved installation instructions

## [0.2.0] - 2024-11-20

### ✨ New Features
- Added MLPredictor class for machine learning predictions
- Introduced EnsembleModels for advanced prediction strategies
- Added AIAnalyzer for market sentiment analysis
- Enhanced technical indicators with more options

### 🔧 Improvements
- Better error handling across all modules
- Improved performance for large datasets
- Enhanced backtesting capabilities
- More robust data validation

## [0.1.0] - 2024-10-01

### 🎉 Initial Release
- Basic trading engine functionality
- Technical indicators calculation
- Simple backtesting capabilities
- Yahoo Finance data integration