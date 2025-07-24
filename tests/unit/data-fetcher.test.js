/**
 * Unit tests for DataFetcher API integration
 */

const DataFetcher = require('../../src/api/data-fetcher');

// Mock axios for testing
jest.mock('axios');
const axios = require('axios');

describe('DataFetcher', () => {
    let dataFetcher;

    beforeEach(() => {
        dataFetcher = new DataFetcher();
        jest.clearAllMocks();
    });

    describe('validateStockData', () => {
        test('should validate correct stock data', () => {
            const validData = {
                symbol: 'AAPL',
                open: 150.0,
                high: 155.0,
                low: 149.0,
                close: 152.0,
                volume: 1000000
            };

            expect(dataFetcher.validateStockData(validData)).toBe(true);
        });

        test('should reject data with missing fields', () => {
            const invalidData = {
                symbol: 'AAPL',
                open: 150.0,
                // missing high, low, close, volume
            };

            expect(dataFetcher.validateStockData(invalidData)).toBe(false);
        });

        test('should reject data with invalid OHLC consistency', () => {
            const invalidData = {
                symbol: 'AAPL',
                open: 150.0,
                high: 140.0, // High is lower than open
                low: 149.0,
                close: 152.0,
                volume: 1000000
            };

            expect(dataFetcher.validateStockData(invalidData)).toBe(false);
        });

        test('should reject data with negative prices', () => {
            const invalidData = {
                symbol: 'AAPL',
                open: -150.0,
                high: 155.0,
                low: 149.0,
                close: 152.0,
                volume: 1000000
            };

            expect(dataFetcher.validateStockData(invalidData)).toBe(false);
        });

        test('should reject data with negative volume', () => {
            const invalidData = {
                symbol: 'AAPL',
                open: 150.0,
                high: 155.0,
                low: 149.0,
                close: 152.0,
                volume: -1000000
            };

            expect(dataFetcher.validateStockData(invalidData)).toBe(false);
        });
    });

    describe('normalizeStockData', () => {
        test('should normalize stock data format', () => {
            const rawData = {
                symbol: 'aapl',
                date: '2023-01-01',
                open: '150.0',
                high: '155.0',
                low: '149.0',
                close: '152.0',
                volume: '1000000',
                source: 'yahoo_finance'
            };

            const normalized = dataFetcher.normalizeStockData(rawData);

            expect(normalized.symbol).toBe('AAPL');
            expect(normalized.open_price).toBe(150.0);
            expect(normalized.high_price).toBe(155.0);
            expect(normalized.low_price).toBe(149.0);
            expect(normalized.close_price).toBe(152.0);
            expect(normalized.volume).toBe(1000000);
            expect(normalized.source).toBe('yahoo_finance');
            expect(normalized.fetched_at).toBeDefined();
        });
    });

    describe('cache management', () => {
        test('should cache and retrieve data', () => {
            const testData = { symbol: 'AAPL', close: 150.0 };
            
            dataFetcher.setCache('AAPL', testData);
            const cached = dataFetcher.getFromCache('AAPL');
            
            expect(cached).toEqual(testData);
        });

        test('should return null for expired cache', async () => {
            const testData = { symbol: 'AAPL', close: 150.0 };
            
            // Set a very short cache timeout for testing
            dataFetcher.cacheTimeout = 1;
            dataFetcher.setCache('AAPL', testData);
            
            // Wait for cache to expire
            await new Promise(resolve => setTimeout(resolve, 2));
            
            const cached = dataFetcher.getFromCache('AAPL');
            expect(cached).toBeNull();
        });

        test('should clear cache', () => {
            dataFetcher.setCache('AAPL', { symbol: 'AAPL' });
            dataFetcher.setCache('GOOGL', { symbol: 'GOOGL' });
            
            expect(dataFetcher.cache.size).toBe(2);
            
            dataFetcher.clearCache();
            expect(dataFetcher.cache.size).toBe(0);
        });
    });

    describe('fetchFromYahooFinance', () => {
        test('should fetch and parse Yahoo Finance data', async () => {
            const mockResponse = {
                data: {
                    chart: {
                        result: [{
                            meta: {
                                symbol: 'AAPL',
                                regularMarketPrice: 152.0,
                                previousClose: 150.0
                            },
                            indicators: {
                                quote: [{
                                    open: [151.0],
                                    high: [155.0],
                                    low: [149.0],
                                    close: [152.0],
                                    volume: [1000000]
                                }]
                            }
                        }]
                    }
                }
            };

            axios.get.mockResolvedValue(mockResponse);

            const result = await dataFetcher.fetchFromYahooFinance('AAPL');

            expect(result.symbol).toBe('AAPL');
            expect(result.open).toBe(151.0);
            expect(result.high).toBe(155.0);
            expect(result.low).toBe(149.0);
            expect(result.close).toBe(152.0);
            expect(result.volume).toBe(1000000);
            expect(result.source).toBe('yahoo_finance');
        });

        test('should retry on failure', async () => {
            axios.get
                .mockRejectedValueOnce(new Error('Network error'))
                .mockRejectedValueOnce(new Error('Network error'))
                .mockResolvedValueOnce({
                    data: {
                        chart: {
                            result: [{
                                meta: { symbol: 'AAPL', regularMarketPrice: 152.0 },
                                indicators: { quote: [{ open: [151.0], high: [155.0], low: [149.0], close: [152.0], volume: [1000000] }] }
                            }]
                        }
                    }
                });

            const result = await dataFetcher.fetchFromYahooFinance('AAPL', 3);
            expect(result.symbol).toBe('AAPL');
            expect(axios.get).toHaveBeenCalledTimes(3);
        });

        test('should throw error after max retries', async () => {
            axios.get.mockRejectedValue(new Error('Network error'));

            await expect(dataFetcher.fetchFromYahooFinance('AAPL', 2))
                .rejects.toThrow('Network error');
            
            expect(axios.get).toHaveBeenCalledTimes(2);
        });
    });

    describe('fetchStockData', () => {
        test('should return cached data when available', async () => {
            const cachedData = { symbol: 'AAPL', close: 150.0 };
            dataFetcher.setCache('AAPL', cachedData);

            const result = await dataFetcher.fetchStockData('AAPL');
            expect(result).toEqual(cachedData);
            expect(axios.get).not.toHaveBeenCalled();
        });

        test('should fetch from primary source when cache miss', async () => {
            const mockResponse = {
                data: {
                    chart: {
                        result: [{
                            meta: { symbol: 'AAPL', regularMarketPrice: 152.0 },
                            indicators: { quote: [{ open: [151.0], high: [155.0], low: [149.0], close: [152.0], volume: [1000000] }] }
                        }]
                    }
                }
            };

            axios.get.mockResolvedValue(mockResponse);

            const result = await dataFetcher.fetchStockData('AAPL', { useCache: false });
            expect(result.symbol).toBe('AAPL');
            expect(axios.get).toHaveBeenCalledTimes(1);
        });
    });

    describe('rate limiting', () => {
        test('should track rate limit requests', () => {
            const initialRequests = dataFetcher.rateLimits.yfinance.requests;
            dataFetcher.updateRateLimit('yfinance');
            
            expect(dataFetcher.rateLimits.yfinance.requests).toBe(initialRequests + 1);
        });

        test('should get cache statistics', () => {
            dataFetcher.setCache('AAPL', { symbol: 'AAPL' });
            dataFetcher.setCache('GOOGL', { symbol: 'GOOGL' });
            
            const stats = dataFetcher.getCacheStats();
            
            expect(stats.size).toBe(2);
            expect(stats.timeout).toBeDefined();
            expect(stats.rateLimits).toBeDefined();
        });
    });
});