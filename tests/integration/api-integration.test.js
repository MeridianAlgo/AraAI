/**
 * Integration tests for API endpoints
 */

const request = require('supertest');
const app = require('../../src/api/server');

describe('API Integration Tests', () => {
    let server;

    beforeAll(() => {
        // Start server for testing
        server = app.listen(0); // Use random port for testing
    });

    afterAll((done) => {
        server.close(done);
    });

    describe('Health Check', () => {
        test('GET /health should return healthy status', async () => {
            const response = await request(app)
                .get('/health')
                .expect(200);

            expect(response.body.status).toBe('healthy');
            expect(response.body.timestamp).toBeDefined();
            expect(response.body.uptime).toBeDefined();
        });
    });

    describe('Stock Data Endpoints', () => {
        test('GET /api/stock/:symbol should return stock data', async () => {
            // This test might fail if external APIs are down
            // In a real environment, you'd mock the DataFetcher
            const response = await request(app)
                .get('/api/stock/AAPL')
                .expect('Content-Type', /json/);

            if (response.status === 200) {
                expect(response.body.success).toBe(true);
                expect(response.body.data).toBeDefined();
                expect(response.body.data.symbol).toBe('AAPL');
                expect(response.body.data.close_price).toBeDefined();
                expect(response.body.timestamp).toBeDefined();
            } else {
                // API might be down, check error format
                expect(response.body.success).toBe(false);
                expect(response.body.error).toBeDefined();
            }
        }, 30000); // 30 second timeout for external API calls

        test('GET /api/validate/:symbol should validate symbol', async () => {
            const response = await request(app)
                .get('/api/validate/AAPL')
                .expect(200)
                .expect('Content-Type', /json/);

            expect(response.body.success).toBe(true);
            expect(response.body.symbol).toBe('AAPL');
            expect(typeof response.body.valid).toBe('boolean');
        }, 30000);

        test('GET /api/validate/:symbol should handle invalid symbol', async () => {
            const response = await request(app)
                .get('/api/validate/INVALID123')
                .expect(200)
                .expect('Content-Type', /json/);

            expect(response.body.success).toBe(true);
            expect(response.body.symbol).toBe('INVALID123');
            expect(response.body.valid).toBe(false);
            expect(response.body.error).toBeDefined();
        }, 30000);
    });

    describe('Cache Endpoints', () => {
        test('GET /api/cache/stats should return cache statistics', async () => {
            const response = await request(app)
                .get('/api/cache/stats')
                .expect(200)
                .expect('Content-Type', /json/);

            expect(response.body.success).toBe(true);
            expect(response.body.cache).toBeDefined();
            expect(response.body.cache.size).toBeDefined();
            expect(response.body.cache.timeout).toBeDefined();
            expect(response.body.cache.rateLimits).toBeDefined();
        });

        test('DELETE /api/cache should clear cache', async () => {
            const response = await request(app)
                .delete('/api/cache')
                .expect(200)
                .expect('Content-Type', /json/);

            expect(response.body.success).toBe(true);
            expect(response.body.message).toBe('Cache cleared successfully');
        });
    });

    describe('Batch Endpoints', () => {
        test('POST /api/stocks/batch should handle batch requests', async () => {
            const response = await request(app)
                .post('/api/stocks/batch')
                .send({ symbols: ['AAPL', 'GOOGL'] })
                .expect('Content-Type', /json/);

            expect(response.body.success).toBe(true);
            expect(response.body.requested).toBe(2);
            expect(Array.isArray(response.body.data)).toBe(true);
            expect(Array.isArray(response.body.errors)).toBe(true);
        }, 60000); // Longer timeout for batch requests

        test('POST /api/stocks/batch should reject empty symbols array', async () => {
            const response = await request(app)
                .post('/api/stocks/batch')
                .send({ symbols: [] })
                .expect(400)
                .expect('Content-Type', /json/);

            expect(response.body.success).toBe(false);
            expect(response.body.error).toBe('symbols array is required');
        });

        test('POST /api/stocks/batch should reject too many symbols', async () => {
            const symbols = Array(15).fill('AAPL'); // 15 symbols, over the limit
            
            const response = await request(app)
                .post('/api/stocks/batch')
                .send({ symbols })
                .expect(400)
                .expect('Content-Type', /json/);

            expect(response.body.success).toBe(false);
            expect(response.body.error).toBe('Maximum 10 symbols allowed per batch request');
        });

        test('POST /api/stocks/batch should require symbols array', async () => {
            const response = await request(app)
                .post('/api/stocks/batch')
                .send({})
                .expect(400)
                .expect('Content-Type', /json/);

            expect(response.body.success).toBe(false);
            expect(response.body.error).toBe('symbols array is required');
        });
    });

    describe('Error Handling', () => {
        test('should return 404 for unknown endpoints', async () => {
            const response = await request(app)
                .get('/api/unknown')
                .expect(404)
                .expect('Content-Type', /json/);

            expect(response.body.success).toBe(false);
            expect(response.body.error).toBe('Endpoint not found');
            expect(response.body.path).toBe('/api/unknown');
        });

        test('should handle malformed JSON in POST requests', async () => {
            const response = await request(app)
                .post('/api/stocks/batch')
                .set('Content-Type', 'application/json')
                .send('invalid json')
                .expect(400);

            // Express automatically handles malformed JSON
        });
    });

    describe('CORS and Headers', () => {
        test('should include CORS headers', async () => {
            const response = await request(app)
                .get('/health')
                .expect(200);

            expect(response.headers['access-control-allow-origin']).toBeDefined();
        });

        test('should handle OPTIONS requests', async () => {
            await request(app)
                .options('/api/stock/AAPL')
                .expect(204);
        });
    });
});