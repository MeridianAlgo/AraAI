/**
 * Financial Data API Integration Layer
 * Handles fetching stock data from multiple sources with retry logic and validation
 */

const axios = require('axios');
const moment = require('moment');

class DataFetcher {
    constructor() {
        this.primarySource = 'yfinance';
        this.fallbackSource = 'alpha_vantage';
        this.cache = new Map();
        this.cacheTimeout = 5 * 60 * 1000; // 5 minutes
        this.rateLimits = {
            yfinance: { requests: 0, resetTime: 0, maxRequests: 2000 },
            alpha_vantage: { requests: 0, resetTime: 0, maxRequests: 5 }
        };
    }

    /**
     * Fetch current stock data with fallback sources
     */
    async fetchStockData(symbol, options = {}) {
        const { useCache = true, retries = 3 } = options;
        
        // Check cache first
        if (useCache) {
            const cached = this.getFromCache(symbol);
            if (cached) {
                console.log(`📦 Using cached data for ${symbol}`);
                return cached;
            }
        }

        let lastError = null;
        
        // Try primary source first
        try {
            console.log(`🔄 Fetching ${symbol} from primary source (${this.primarySource})`);
            const data = await this.fetchFromYahooFinance(symbol, retries);
            
            if (this.validateStockData(data)) {
                this.setCache(symbol, data);
                return data;
            }
        } catch (error) {
            console.warn(`⚠️ Primary source failed for ${symbol}:`, error.message);
            lastError = error;
        }

        // Try fallback source
        try {
            console.log(`🔄 Trying fallback source (${this.fallbackSource}) for ${symbol}`);
            const data = await this.fetchFromAlphaVantage(symbol, retries);
            
            if (this.validateStockData(data)) {
                this.setCache(symbol, data);
                return data;
            }
        } catch (error) {
            console.warn(`⚠️ Fallback source failed for ${symbol}:`, error.message);
            lastError = error;
        }

        // All sources failed
        throw new Error(`Failed to fetch data for ${symbol} from all sources. Last error: ${lastError?.message}`);
    }

    /**
     * Fetch data from Yahoo Finance (free tier)
     */
    async fetchFromYahooFinance(symbol, retries = 3) {
        await this.checkRateLimit('yfinance');
        
        const url = `https://query1.finance.yahoo.com/v8/finance/chart/${symbol}`;
        const params = {
            range: '1d',
            interval: '1d',
            includePrePost: false
        };

        for (let attempt = 1; attempt <= retries; attempt++) {
            try {
                console.log(`📡 Yahoo Finance API call for ${symbol} (attempt ${attempt}/${retries})`);
                
                const response = await axios.get(url, {
                    params,
                    timeout: 10000,
                    headers: {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                    }
                });

                this.updateRateLimit('yfinance');

                if (response.data?.chart?.result?.[0]) {
                    const result = response.data.chart.result[0];
                    const meta = result.meta;
                    const quote = result.indicators.quote[0];
                    
                    return {
                        symbol: meta.symbol,
                        date: new Date().toISOString().split('T')[0],
                        open: quote.open[0] || meta.previousClose,
                        high: quote.high[0] || meta.previousClose,
                        low: quote.low[0] || meta.previousClose,
                        close: meta.regularMarketPrice || quote.close[0],
                        volume: quote.volume[0] || 0,
                        previousClose: meta.previousClose,
                        source: 'yahoo_finance'
                    };
                }
                
                throw new Error('Invalid response format from Yahoo Finance');
                
            } catch (error) {
                console.warn(`❌ Yahoo Finance attempt ${attempt} failed:`, error.message);
                
                if (attempt === retries) {
                    throw error;
                }
                
                // Exponential backoff
                const delay = Math.min(1000 * Math.pow(2, attempt - 1), 10000);
                console.log(`⏳ Waiting ${delay}ms before retry...`);
                await this.sleep(delay);
            }
        }
    }

    /**
     * Fetch data from Alpha Vantage (requires API key)
     */
    async fetchFromAlphaVantage(symbol, retries = 3) {
        const apiKey = process.env.ALPHA_VANTAGE_API_KEY;
        
        if (!apiKey) {
            throw new Error('Alpha Vantage API key not configured');
        }

        await this.checkRateLimit('alpha_vantage');
        
        const url = 'https://www.alphavantage.co/query';
        const params = {
            function: 'GLOBAL_QUOTE',
            symbol: symbol,
            apikey: apiKey
        };

        for (let attempt = 1; attempt <= retries; attempt++) {
            try {
                console.log(`📡 Alpha Vantage API call for ${symbol} (attempt ${attempt}/${retries})`);
                
                const response = await axios.get(url, {
                    params,
                    timeout: 15000
                });

                this.updateRateLimit('alpha_vantage');

                const quote = response.data['Global Quote'];
                if (quote && quote['01. symbol']) {
                    return {
                        symbol: quote['01. symbol'],
                        date: new Date().toISOString().split('T')[0],
                        open: parseFloat(quote['02. open']),
                        high: parseFloat(quote['03. high']),
                        low: parseFloat(quote['04. low']),
                        close: parseFloat(quote['05. price']),
                        volume: parseInt(quote['06. volume']),
                        previousClose: parseFloat(quote['08. previous close']),
                        source: 'alpha_vantage'
                    };
                }
                
                // Check for API limit message
                if (response.data.Note) {
                    throw new Error('Alpha Vantage API rate limit exceeded');
                }
                
                throw new Error('Invalid response format from Alpha Vantage');
                
            } catch (error) {
                console.warn(`❌ Alpha Vantage attempt ${attempt} failed:`, error.message);
                
                if (attempt === retries) {
                    throw error;
                }
                
                // Longer delay for Alpha Vantage due to stricter rate limits
                const delay = Math.min(2000 * Math.pow(2, attempt - 1), 30000);
                console.log(`⏳ Waiting ${delay}ms before retry...`);
                await this.sleep(delay);
            }
        }
    }

    /**
     * Validate stock data quality and completeness
     */
    validateStockData(data) {
        if (!data || typeof data !== 'object') {
            console.warn('❌ Invalid data: not an object');
            return false;
        }

        const required = ['symbol', 'open', 'high', 'low', 'close', 'volume'];
        const missing = required.filter(field => data[field] === undefined || data[field] === null);
        
        if (missing.length > 0) {
            console.warn(`❌ Missing required fields: ${missing.join(', ')}`);
            return false;
        }

        // Validate price consistency
        const { open, high, low, close } = data;
        if (high < Math.max(open, close, low) || low > Math.min(open, close, high)) {
            console.warn('❌ Invalid OHLC price consistency');
            return false;
        }

        // Validate reasonable price ranges
        if (open <= 0 || high <= 0 || low <= 0 || close <= 0) {
            console.warn('❌ Invalid prices: must be positive');
            return false;
        }

        if (data.volume < 0) {
            console.warn('❌ Invalid volume: cannot be negative');
            return false;
        }

        console.log(`✅ Data validation passed for ${data.symbol}`);
        return true;
    }

    /**
     * Normalize data format for consistent processing
     */
    normalizeStockData(data) {
        return {
            symbol: data.symbol.toUpperCase(),
            date: data.date,
            open_price: parseFloat(data.open),
            high_price: parseFloat(data.high),
            low_price: parseFloat(data.low),
            close_price: parseFloat(data.close),
            volume: parseInt(data.volume),
            previous_close: data.previousClose ? parseFloat(data.previousClose) : null,
            source: data.source,
            fetched_at: new Date().toISOString()
        };
    }

    /**
     * Check and enforce API rate limits
     */
    async checkRateLimit(source) {
        const limit = this.rateLimits[source];
        const now = Date.now();
        
        // Reset counter if hour has passed
        if (now > limit.resetTime) {
            limit.requests = 0;
            limit.resetTime = now + (60 * 60 * 1000); // 1 hour
        }
        
        // Check if we've hit the limit
        if (limit.requests >= limit.maxRequests) {
            const waitTime = limit.resetTime - now;
            console.warn(`⏳ Rate limit reached for ${source}. Waiting ${Math.ceil(waitTime / 1000)}s...`);
            await this.sleep(waitTime);
            
            // Reset after waiting
            limit.requests = 0;
            limit.resetTime = now + (60 * 60 * 1000);
        }
    }

    /**
     * Update rate limit counter
     */
    updateRateLimit(source) {
        this.rateLimits[source].requests++;
    }

    /**
     * Cache management
     */
    getFromCache(symbol) {
        const cached = this.cache.get(symbol);
        if (cached && (Date.now() - cached.timestamp) < this.cacheTimeout) {
            return cached.data;
        }
        return null;
    }

    setCache(symbol, data) {
        this.cache.set(symbol, {
            data: data,
            timestamp: Date.now()
        });
    }

    clearCache() {
        this.cache.clear();
        console.log('🗑️ Cache cleared');
    }

    /**
     * Utility function for delays
     */
    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    /**
     * Get cache statistics
     */
    getCacheStats() {
        return {
            size: this.cache.size,
            timeout: this.cacheTimeout,
            rateLimits: this.rateLimits
        };
    }

    /**
     * Fetch historical data (basic implementation)
     */
    async fetchHistoricalData(symbol, days = 30) {
        console.log(`📊 Fetching ${days} days of historical data for ${symbol}`);
        
        try {
            // For now, we'll use a simple approach - in production you'd want proper historical API
            const url = `https://query1.finance.yahoo.com/v8/finance/chart/${symbol}`;
            const params = {
                range: `${days}d`,
                interval: '1d'
            };

            const response = await axios.get(url, {
                params,
                timeout: 15000,
                headers: {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
            });

            if (response.data?.chart?.result?.[0]) {
                const result = response.data.chart.result[0];
                const timestamps = result.timestamp;
                const quote = result.indicators.quote[0];
                
                const historicalData = [];
                
                for (let i = 0; i < timestamps.length; i++) {
                    if (quote.open[i] !== null) {
                        historicalData.push({
                            symbol: symbol,
                            date: new Date(timestamps[i] * 1000).toISOString().split('T')[0],
                            open: quote.open[i],
                            high: quote.high[i],
                            low: quote.low[i],
                            close: quote.close[i],
                            volume: quote.volume[i] || 0,
                            source: 'yahoo_finance_historical'
                        });
                    }
                }
                
                console.log(`✅ Retrieved ${historicalData.length} historical data points for ${symbol}`);
                return historicalData;
            }
            
            throw new Error('No historical data available');
            
        } catch (error) {
            console.error(`❌ Failed to fetch historical data for ${symbol}:`, error.message);
            throw error;
        }
    }
}

module.exports = DataFetcher;