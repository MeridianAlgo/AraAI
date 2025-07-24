/**
 * Node.js API Server for ML Stock Predictor
 * Provides endpoints for stock data fetching and analysis
 */

const express = require('express');
const cors = require('cors');
const DataFetcher = require('./data-fetcher');

const app = express();
const port = process.env.API_PORT || 3000;
const dataFetcher = new DataFetcher();

// Middleware
app.use(cors());
app.use(express.json());

// Request logging middleware
app.use((req, res, next) => {
    console.log(`${new Date().toISOString()} - ${req.method} ${req.path}`);
    next();
});

// Health check endpoint
app.get('/health', (req, res) => {
    res.json({
        status: 'healthy',
        timestamp: new Date().toISOString(),
        uptime: process.uptime()
    });
});

// Get current stock data
app.get('/api/stock/:symbol', async (req, res) => {
    try {
        const { symbol } = req.params;
        const { useCache = 'true' } = req.query;
        
        console.log(`📊 Fetching stock data for ${symbol.toUpperCase()}`);
        
        const stockData = await dataFetcher.fetchStockData(symbol.toUpperCase(), {
            useCache: useCache === 'true'
        });
        
        const normalizedData = dataFetcher.normalizeStockData(stockData);
        
        res.json({
            success: true,
            data: normalizedData,
            timestamp: new Date().toISOString()
        });
        
    } catch (error) {
        console.error(`❌ Error fetching stock data:`, error.message);
        
        res.status(500).json({
            success: false,
            error: error.message,
            timestamp: new Date().toISOString()
        });
    }
});

// Get historical stock data
app.get('/api/stock/:symbol/history', async (req, res) => {
    try {
        const { symbol } = req.params;
        const { days = '30' } = req.query;
        
        console.log(`📈 Fetching ${days} days of historical data for ${symbol.toUpperCase()}`);
        
        const historicalData = await dataFetcher.fetchHistoricalData(
            symbol.toUpperCase(), 
            parseInt(days)
        );
        
        const normalizedData = historicalData.map(data => 
            dataFetcher.normalizeStockData(data)
        );
        
        res.json({
            success: true,
            data: normalizedData,
            count: normalizedData.length,
            symbol: symbol.toUpperCase(),
            days: parseInt(days),
            timestamp: new Date().toISOString()
        });
        
    } catch (error) {
        console.error(`❌ Error fetching historical data:`, error.message);
        
        res.status(500).json({
            success: false,
            error: error.message,
            timestamp: new Date().toISOString()
        });
    }
});

// Validate stock symbol
app.get('/api/validate/:symbol', async (req, res) => {
    try {
        const { symbol } = req.params;
        
        console.log(`🔍 Validating symbol ${symbol.toUpperCase()}`);
        
        // Try to fetch current data to validate symbol
        const stockData = await dataFetcher.fetchStockData(symbol.toUpperCase());
        
        res.json({
            success: true,
            valid: true,
            symbol: symbol.toUpperCase(),
            currentPrice: stockData.close,
            timestamp: new Date().toISOString()
        });
        
    } catch (error) {
        console.warn(`⚠️ Symbol validation failed for ${symbol}:`, error.message);
        
        res.json({
            success: true,
            valid: false,
            symbol: symbol.toUpperCase(),
            error: error.message,
            timestamp: new Date().toISOString()
        });
    }
});

// Get cache statistics
app.get('/api/cache/stats', (req, res) => {
    const stats = dataFetcher.getCacheStats();
    
    res.json({
        success: true,
        cache: stats,
        timestamp: new Date().toISOString()
    });
});

// Clear cache
app.delete('/api/cache', (req, res) => {
    dataFetcher.clearCache();
    
    res.json({
        success: true,
        message: 'Cache cleared successfully',
        timestamp: new Date().toISOString()
    });
});

// Batch stock data endpoint
app.post('/api/stocks/batch', async (req, res) => {
    try {
        const { symbols } = req.body;
        
        if (!Array.isArray(symbols) || symbols.length === 0) {
            return res.status(400).json({
                success: false,
                error: 'symbols array is required',
                timestamp: new Date().toISOString()
            });
        }
        
        if (symbols.length > 10) {
            return res.status(400).json({
                success: false,
                error: 'Maximum 10 symbols allowed per batch request',
                timestamp: new Date().toISOString()
            });
        }
        
        console.log(`📊 Batch fetching data for ${symbols.length} symbols`);
        
        const results = [];
        const errors = [];
        
        for (const symbol of symbols) {
            try {
                const stockData = await dataFetcher.fetchStockData(symbol.toUpperCase());
                const normalizedData = dataFetcher.normalizeStockData(stockData);
                results.push(normalizedData);
            } catch (error) {
                errors.push({
                    symbol: symbol.toUpperCase(),
                    error: error.message
                });
            }
        }
        
        res.json({
            success: true,
            data: results,
            errors: errors,
            requested: symbols.length,
            successful: results.length,
            failed: errors.length,
            timestamp: new Date().toISOString()
        });
        
    } catch (error) {
        console.error(`❌ Batch request error:`, error.message);
        
        res.status(500).json({
            success: false,
            error: error.message,
            timestamp: new Date().toISOString()
        });
    }
});

// Error handling middleware
app.use((error, req, res, next) => {
    console.error('Unhandled error:', error);
    
    res.status(500).json({
        success: false,
        error: 'Internal server error',
        timestamp: new Date().toISOString()
    });
});

// 404 handler
app.use((req, res) => {
    res.status(404).json({
        success: false,
        error: 'Endpoint not found',
        path: req.path,
        timestamp: new Date().toISOString()
    });
});

// Start server
const server = app.listen(port, () => {
    console.log(`🚀 ML Stock Predictor API Server running on port ${port}`);
    console.log(`📡 Health check: http://localhost:${port}/health`);
    console.log(`📊 Stock data: http://localhost:${port}/api/stock/{symbol}`);
    console.log(`📈 Historical: http://localhost:${port}/api/stock/{symbol}/history`);
});

// Graceful shutdown
process.on('SIGTERM', () => {
    console.log('🛑 SIGTERM received, shutting down gracefully');
    server.close(() => {
        console.log('✅ Server closed');
        process.exit(0);
    });
});

process.on('SIGINT', () => {
    console.log('🛑 SIGINT received, shutting down gracefully');
    server.close(() => {
        console.log('✅ Server closed');
        process.exit(0);
    });
});

module.exports = app;