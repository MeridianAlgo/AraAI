# ML Stock Predictor

A comprehensive machine learning stock analysis and price prediction platform using PyTorch, Node.js, and Python.

## Features

- Real-time stock data analysis with technical indicators
- PyTorch neural network for next-day price predictions
- Comprehensive error tracking and performance metrics
- Terminal-based interface with analytics dashboard
- Model diagnostics and feature importance analysis
- Profit/loss simulation with risk-adjusted returns

## Setup

### Prerequisites

- Node.js (v16 or higher)
- Python (v3.8 or higher)
- pip and npm

### Installation

1. Install Node.js dependencies:
```bash
npm install
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

### Usage

Start the CLI application:
```bash
npm start
```

Start the API server:
```bash
npm run api
```

## Project Structure

```
├── src/
│   ├── cli/           # Terminal CLI interface
│   ├── api/           # Node.js API server
│   └── python/        # Python ML engine
├── data/
│   ├── models/        # Trained model storage
│   └── stock_data.db  # SQLite database
├── config/            # Configuration files
├── tests/             # Test suites
└── logs/              # Application logs
```

## Development

Run in development mode:
```bash
npm run dev
```

Run tests:
```bash
npm test
```

## License

MIT