#!/usr/bin/env node
/**
 * ML Stock Predictor CLI - Enhanced Terminal Interface with Analytics
 */

const { Command } = require('commander');
const inquirer = require('inquirer');
const chalk = require('chalk');
const Table = require('cli-table3');
const cliProgress = require('cli-progress');
const axios = require('axios');
const { spawn } = require('child_process');
const path = require('path');
const boxen = require('boxen');
const gradient = require('gradient-string');
const figlet = require('figlet');
const ora = require('ora');

const program = new Command();

// Configuration
const API_BASE_URL = 'http://localhost:3000';
const PYTHON_PATH = 'python';

class MLStockCLI {
    constructor() {
        this.progressBar = null;
        this.spinner = null;
        this.setupCommands();
    }

    showBanner() {
        console.clear();
        const title = figlet.textSync('ML STOCK', {
            font: 'ANSI Shadow',
            horizontalLayout: 'default',
            verticalLayout: 'default'
        });
        
        const subtitle = figlet.textSync('PREDICTOR', {
            font: 'ANSI Shadow',
            horizontalLayout: 'default',
            verticalLayout: 'default'
        });

        console.log(gradient.rainbow(title));
        console.log(gradient.rainbow(subtitle));
        console.log(chalk.gray('━'.repeat(80)));
        console.log(chalk.cyan.bold('🤖 AI-Powered Stock Analysis & Price Prediction Platform'));
        console.log(chalk.gray('━'.repeat(80)));
    }

    createBox(content, title, color = 'cyan') {
        return boxen(content, {
            title: title,
            titleAlignment: 'center',
            padding: 1,
            margin: 1,
            borderStyle: 'round',
            borderColor: color,
            backgroundColor: 'black'
        });
    }

    createInfoBox(title, data, color = 'blue') {
        let content = '';
        for (const [key, value] of Object.entries(data)) {
            content += `${chalk.gray('•')} ${chalk.white(key)}: ${chalk[color](value)}\n`;
        }
        return this.createBox(content.trim(), title, color);
    }

    createMetricsTable(data, title) {
        const table = new Table({
            head: [chalk.cyan.bold('Metric'), chalk.cyan.bold('Value')],
            colWidths: [25, 25],
            style: {
                head: [],
                border: ['cyan'],
                'padding-left': 2,
                'padding-right': 2
            },
            chars: {
                'top': '═',
                'top-mid': '╤',
                'top-left': '╔',
                'top-right': '╗',
                'bottom': '═',
                'bottom-mid': '╧',
                'bottom-left': '╚',
                'bottom-right': '╝',
                'left': '║',
                'left-mid': '╟',
                'mid': '─',
                'mid-mid': '┼',
                'right': '║',
                'right-mid': '╢',
                'middle': '│'
            }
        });

        Object.entries(data).forEach(([key, value]) => {
            table.push([chalk.white(key), chalk.green(value)]);
        });

        return this.createBox(table.toString(), title, 'cyan');
    }

    startSpinner(text, color = 'cyan') {
        this.spinner = ora({
            text: chalk[color](text),
            spinner: 'dots12',
            color: color
        }).start();
    }

    updateSpinner(text, color = 'cyan') {
        if (this.spinner) {
            this.spinner.text = chalk[color](text);
        }
    }

    stopSpinner(symbol = '✅', text = 'Done', color = 'green') {
        if (this.spinner) {
            this.spinner.stopAndPersist({
                symbol: symbol,
                text: chalk[color](text)
            });
            this.spinner = null;
        }
    }

    failSpinner(text = 'Failed', color = 'red') {
        if (this.spinner) {
            this.spinner.stopAndPersist({
                symbol: '❌',
                text: chalk[color](text)
            });
            this.spinner = null;
        }
    }

    setupCommands() {
        program
            .name('ml-stock-predictor')
            .description('ML Stock Predictor - Terminal Interface with Analytics')
            .version('1.0.0');

        // Main prediction command
        program
            .command('predict <symbol>')
            .description('Predict next-day price for a stock symbol')
            .option('-t, --train', 'Train model before prediction')
            .option('-d, --days <days>', 'Days of historical data for training', '60')
            .option('-v, --verbose', 'Verbose output')
            .action(async (symbol, options) => {
                await this.handlePredict(symbol.toUpperCase(), options);
            });

        // Training command
        program
            .command('train <symbol>')
            .description('Train a new model for a stock symbol')
            .option('-d, --days <days>', 'Days of historical data', '60')
            .option('-e, --epochs <epochs>', 'Training epochs', '100')
            .option('-l, --learning-rate <rate>', 'Learning rate', '0.001')
            .action(async (symbol, options) => {
                await this.handleTrain(symbol.toUpperCase(), options);
            });

        // Analytics command
        program
            .command('analytics <symbol>')
            .description('Show comprehensive analytics for a symbol')
            .option('-d, --days <days>', 'Days to analyze', '30')
            .option('--report', 'Generate detailed report')
            .action(async (symbol, options) => {
                await this.handleAnalytics(symbol.toUpperCase(), options);
            });

        // Performance command
        program
            .command('performance <symbol>')
            .description('Show model performance metrics')
            .option('-d, --days <days>', 'Days to analyze', '30')
            .action(async (symbol, options) => {
                await this.handlePerformance(symbol.toUpperCase(), options);
            });

        // Interactive mode
        program
            .command('interactive')
            .alias('i')
            .description('Start interactive mode')
            .action(async () => {
                await this.handleInteractive();
            });

        // Market data command
        program
            .command('data <symbol>')
            .description('Fetch and display current market data')
            .option('-h, --history <days>', 'Show historical data', '5')
            .action(async (symbol, options) => {
                await this.handleMarketData(symbol.toUpperCase(), options);
            });

        // Status command
        program
            .command('status')
            .description('Show system status and health')
            .action(async () => {
                await this.handleStatus();
            });

        // Volatility analysis command
        program
            .command('volatility <symbol>')
            .description('Show historical volatility analysis for a symbol')
            .option('-d, --days <days>', 'Days of historical data to analyze', '252')
            .action(async (symbol, options) => {
                await this.handleVolatility(symbol.toUpperCase(), options);
            });
    }

    async handlePredict(symbol, options) {
        this.showBanner();
        
        console.log(this.createBox(
            `🔮 Generating AI-powered prediction for ${chalk.cyan.bold(symbol)}`,
            '🤖 ML STOCK PREDICTION',
            'magenta'
        ));

        try {
            // Validate symbol first
            this.startSpinner(`Validating symbol ${symbol}...`, 'cyan');
            await this.validateSymbol(symbol);
            this.stopSpinner('✅', `Symbol ${symbol} validated`, 'green');

            // Train model if requested
            if (options.train) {
                this.startSpinner('Training neural network model...', 'yellow');
                await this.trainModel(symbol, options);
                this.stopSpinner('🧠', 'Model training completed', 'green');
            }

            // Make prediction
            this.startSpinner('Analyzing market data and generating prediction...', 'blue');
            const prediction = await this.callPythonScript('predict', {
                symbol: symbol,
                verbose: options.verbose
            });
            this.stopSpinner('🔮', 'Prediction generated successfully', 'green');

            this.displayPrediction(prediction);

        } catch (error) {
            this.failSpinner(`Error: ${error.message}`);
            process.exit(1);
        }
    }

    async handleTrain(symbol, options) {
        console.log(chalk.blue.bold(`\n🏋️  Training Model - ${symbol}\n`));

        try {
            await this.validateSymbol(symbol);

            const trainingParams = {
                symbol: symbol,
                days: parseInt(options.days),
                epochs: parseInt(options.epochs),
                learning_rate: parseFloat(options.learningRate)
            };

            console.log(chalk.cyan('📊 Training parameters:'));
            console.log(`  • Symbol: ${symbol}`);
            console.log(`  • Historical days: ${trainingParams.days}`);
            console.log(`  • Epochs: ${trainingParams.epochs}`);
            console.log(`  • Learning rate: ${trainingParams.learning_rate}`);

            // Start training
            this.startProgressBar('Training model');
            const result = await this.callPythonScript('train', trainingParams);
            this.stopProgressBar();

            this.displayTrainingResults(result);

        } catch (error) {
            this.stopProgressBar();
            console.error(chalk.red(`❌ Training failed: ${error.message}`));
            process.exit(1);
        }
    }

    async handleAnalytics(symbol, options) {
        console.log(chalk.blue.bold(`\n📊 Analytics Dashboard - ${symbol}\n`));

        try {
            const analyticsParams = {
                symbol: symbol,
                days: parseInt(options.days),
                detailed: options.report
            };

            console.log(chalk.cyan('🔄 Generating analytics...'));
            const analytics = await this.callPythonScript('analytics', analyticsParams);

            this.displayAnalytics(analytics);

            if (options.report) {
                console.log(chalk.green('\n📄 Detailed Report:'));
                console.log(analytics.detailed_report);
            }

        } catch (error) {
            console.error(chalk.red(`❌ Analytics error: ${error.message}`));
        }
    }

    async handlePerformance(symbol, options) {
        console.log(chalk.blue.bold(`\n📈 Performance Metrics - ${symbol}\n`));

        try {
            const performanceData = await this.callPythonScript('performance', {
                symbol: symbol,
                days: parseInt(options.days)
            });

            this.displayPerformanceMetrics(performanceData);

        } catch (error) {
            console.error(chalk.red(`❌ Performance analysis error: ${error.message}`));
        }
    }

    async handleMarketData(symbol, options) {
        console.log(chalk.blue.bold(`\n📊 Market Data - ${symbol}\n`));

        try {
            // Check if API server is running, if not start it
            await this.ensureApiServer();
            
            // Get current data
            const response = await axios.get(`${API_BASE_URL}/api/stock/${symbol}`);
            const currentData = response.data.data;

            this.displayMarketData(currentData);

            // Get historical data if requested
            if (options.history) {
                const histResponse = await axios.get(`${API_BASE_URL}/api/stock/${symbol}/history?days=${options.history}`);
                const historicalData = histResponse.data.data;

                this.displayHistoricalData(historicalData);
            }

        } catch (error) {
            console.error(chalk.red(`❌ Market data error: ${error.message}`));
        }
    }

    async handleStatus() {
        console.log(chalk.blue.bold('\n🏥 System Status\n'));

        try {
            // Check API server
            const apiHealth = await axios.get(`${API_BASE_URL}/health`);
            console.log(chalk.green('✅ API Server: Online'));
            console.log(`   Uptime: ${Math.floor(apiHealth.data.uptime)}s`);

            // Check cache stats
            const cacheStats = await axios.get(`${API_BASE_URL}/api/cache/stats`);
            console.log(chalk.green('✅ Cache System: Active'));
            console.log(`   Cached items: ${cacheStats.data.cache.size}`);

            // Check Python engine
            const pythonStatus = await this.callPythonScript('status', {});
            console.log(chalk.green('✅ ML Engine: Ready'));
            console.log(`   PyTorch version: ${pythonStatus.pytorch_version}`);

        } catch (error) {
            console.error(chalk.red(`❌ System check failed: ${error.message}`));
        }
    }

    async handleVolatility(symbol, options) {
        this.showBanner();
        
        console.log(this.createBox(
            `📊 Historical volatility analysis for ${chalk.cyan.bold(symbol)}`,
            '📈 VOLATILITY ANALYSIS',
            'yellow'
        ));

        try {
            this.startSpinner(`Analyzing ${options.days} days of historical data...`, 'yellow');
            
            const volatilityData = await this.callPythonScript('volatility', {
                symbol: symbol,
                days: parseInt(options.days)
            });
            
            this.stopSpinner('📊', 'Volatility analysis completed', 'green');
            
            this.displayVolatilityAnalysis(volatilityData);

        } catch (error) {
            this.failSpinner(`Error: ${error.message}`);
            console.error(chalk.red(`❌ Volatility analysis error: ${error.message}`));
        }
    }

    displayVolatilityAnalysis(data) {
        if (data.error) {
            console.log(chalk.red(`❌ ${data.error}`));
            return;
        }

        // Daily changes overview
        const dailyChanges = {
            'Average Daily Change': `${(data.mean_change * 100).toFixed(2)}%`,
            'Daily Volatility (Std Dev)': `${(data.std_change * 100).toFixed(2)}%`,
            'Largest Single Day Gain': `${(data.max_change * 100).toFixed(2)}%`,
            'Largest Single Day Loss': `${(data.min_change * 100).toFixed(2)}%`,
            'Volatility Regime': data.volatility_regime
        };

        console.log(this.createInfoBox('📊 DAILY PRICE CHANGES', dailyChanges, 'blue'));

        // Typical ranges
        const typicalRanges = {
            '50% of Days Range': `${(data.percentiles.p25 * 100).toFixed(2)}% to ${(data.percentiles.p75 * 100).toFixed(2)}%`,
            '80% of Days Range': `${(data.percentiles.p10 * 100).toFixed(2)}% to ${(data.percentiles.p90 * 100).toFixed(2)}%`,
            '90% of Days Range': `${(data.percentiles.p5 * 100).toFixed(2)}% to ${(data.percentiles.p95 * 100).toFixed(2)}%`,
            'Median Daily Change': `${(data.percentiles.p50 * 100).toFixed(2)}%`
        };

        console.log(this.createInfoBox('📈 TYPICAL RANGES', typicalRanges, 'green'));

        // Extreme moves
        const extremeMoves = {
            'Days with >5% Gains': `${data.extreme_moves.up_5pct_days} (${(data.extreme_moves.up_5pct_days/data.extreme_moves.total_days*100).toFixed(1)}%)`,
            'Days with >5% Losses': `${data.extreme_moves.down_5pct_days} (${(data.extreme_moves.down_5pct_days/data.extreme_moves.total_days*100).toFixed(1)}%)`,
            'Days with >10% Moves': `${data.extreme_moves.up_10pct_days + data.extreme_moves.down_10pct_days} total`,
            'Analysis Period': `${data.extreme_moves.total_days} trading days`
        };

        console.log(this.createInfoBox('🚨 EXTREME MOVES', extremeMoves, 'red'));

        // Prediction constraints info
        const constraintInfo = `Based on this analysis, predictions will be constrained to realistic ranges:\n\n` +
                              `${chalk.green('•')} High confidence predictions: up to ${(data.percentiles.p90 * 100).toFixed(1)}% moves\n` +
                              `${chalk.yellow('•')} Medium confidence predictions: up to ${(data.percentiles.p75 * 100).toFixed(1)}% moves\n` +
                              `${chalk.gray('•')} Low confidence predictions: up to ${(data.typical_range.normal_up * 100).toFixed(1)}% moves`;

        console.log(this.createBox(constraintInfo, '🎯 PREDICTION CONSTRAINTS', 'cyan'));
    }

    async handleInteractive() {
        console.log(chalk.blue.bold('\n🎯 Interactive Mode\n'));
        console.log(chalk.gray('Type "exit" to quit\n'));

        while (true) {
            try {
                const answers = await inquirer.prompt([
                    {
                        type: 'list',
                        name: 'action',
                        message: 'What would you like to do?',
                        choices: [
                            { name: '🔮 Make Prediction', value: 'predict' },
                            { name: '🏋️  Train Model', value: 'train' },
                            { name: '📊 View Analytics', value: 'analytics' },
                            { name: '📈 Performance Metrics', value: 'performance' },
                            { name: '📊 Market Data', value: 'data' },
                            { name: '🏥 System Status', value: 'status' },
                            { name: '🚪 Exit', value: 'exit' }
                        ]
                    }
                ]);

                if (answers.action === 'exit') {
                    console.log(chalk.green('👋 Goodbye!'));
                    break;
                }

                // Get symbol for most actions
                let symbol = '';
                if (['predict', 'train', 'analytics', 'performance', 'data'].includes(answers.action)) {
                    const symbolAnswer = await inquirer.prompt([
                        {
                            type: 'input',
                            name: 'symbol',
                            message: 'Enter stock symbol:',
                            validate: (input) => input.length > 0 || 'Symbol is required'
                        }
                    ]);
                    symbol = symbolAnswer.symbol.toUpperCase();
                }

                // Execute action
                switch (answers.action) {
                    case 'predict':
                        await this.handlePredict(symbol, {});
                        break;
                    case 'train':
                        await this.handleTrain(symbol, { days: '60', epochs: '100', learningRate: '0.001' });
                        break;
                    case 'analytics':
                        await this.handleAnalytics(symbol, { days: '30' });
                        break;
                    case 'performance':
                        await this.handlePerformance(symbol, { days: '30' });
                        break;
                    case 'data':
                        await this.handleMarketData(symbol, { history: '5' });
                        break;
                    case 'status':
                        await this.handleStatus();
                        break;
                }

                console.log('\n' + '─'.repeat(50) + '\n');

            } catch (error) {
                console.error(chalk.red(`❌ Error: ${error.message}`));
            }
        }
    }

    async ensureApiServer() {
        try {
            // Check if API server is already running
            await axios.get(`${API_BASE_URL}/health`, { timeout: 2000 });
        } catch (error) {
            // API server not running, start it
            console.log(chalk.yellow('🚀 Starting API server...'));
            
            // Start API server in background
            const apiServer = spawn('node', ['src/api/server.js'], {
                detached: true,
                stdio: 'ignore'
            });
            
            apiServer.unref();
            
            // Wait for server to start
            await this.sleep(3000);
            
            // Verify it's running
            try {
                await axios.get(`${API_BASE_URL}/health`, { timeout: 5000 });
                console.log(chalk.green('✅ API server started'));
            } catch (e) {
                throw new Error('Failed to start API server');
            }
        }
    }

    async validateSymbol(symbol) {
        try {
            await this.ensureApiServer();
            const response = await axios.get(`${API_BASE_URL}/api/validate/${symbol}`);
            if (!response.data.valid) {
                throw new Error(`Invalid symbol: ${symbol}`);
            }
        } catch (error) {
            throw new Error(`Symbol validation failed: ${error.message}`);
        }
    }

    async trainModel(symbol, options) {
        const trainingParams = {
            symbol: symbol,
            days: parseInt(options.days || '60'),
            epochs: parseInt(options.epochs || '100'),
            learning_rate: parseFloat(options.learningRate || '0.001')
        };

        this.startProgressBar('Training model');
        const result = await this.callPythonScript('train', trainingParams);
        this.stopProgressBar();

        return result;
    }

    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    async callPythonScript(action, params) {
        return new Promise((resolve, reject) => {
            const pythonScript = path.join(__dirname, '..', 'python', 'cli_handler.py');
            const args = [pythonScript, action, JSON.stringify(params)];

            const pythonProcess = spawn(PYTHON_PATH, args);
            let output = '';
            let errorOutput = '';

            pythonProcess.stdout.on('data', (data) => {
                output += data.toString();
            });

            pythonProcess.stderr.on('data', (data) => {
                errorOutput += data.toString();
            });

            pythonProcess.on('close', (code) => {
                if (code === 0) {
                    try {
                        const result = JSON.parse(output);
                        resolve(result);
                    } catch (e) {
                        reject(new Error(`Invalid JSON response: ${output}`));
                    }
                } else {
                    reject(new Error(`Python script failed: ${errorOutput || output}`));
                }
            });
        });
    }

    displayPrediction(prediction) {
        const table = new Table({
            head: ['Metric', 'Value'],
            colWidths: [25, 25]
        });

        const direction = prediction.direction === 'UP' ? '📈' : '📉';
        const confidence = (prediction.confidence * 100).toFixed(1);
        const priceChange = prediction.predicted_price - prediction.current_price;
        const priceChangePct = (priceChange / prediction.current_price * 100).toFixed(2);

        table.push(
            ['Symbol', prediction.symbol],
            ['Current Price', `$${prediction.current_price.toFixed(2)}`],
            ['Predicted Price', `$${prediction.predicted_price.toFixed(2)}`],
            ['Direction', `${direction} ${prediction.direction}`],
            ['Price Change', `$${priceChange.toFixed(2)} (${priceChangePct}%)`],
            ['Confidence', `${confidence}%`],
            ['Risk Level', prediction.risk_level],
            ['Timestamp', new Date(prediction.timestamp).toLocaleString()]
        );

        console.log(table.toString());

        // Risk warning
        if (prediction.risk_level === 'HIGH') {
            console.log(chalk.red.bold('\n⚠️  HIGH RISK PREDICTION - Use with caution!'));
        } else if (prediction.risk_level === 'MEDIUM') {
            console.log(chalk.yellow('\n⚠️  Medium risk prediction - Consider additional analysis'));
        } else {
            console.log(chalk.green('\n✅ Low risk prediction'));
        }
    }

    displayTrainingResults(result) {
        console.log(chalk.green.bold('\n✅ Training Completed!\n'));

        const table = new Table({
            head: ['Metric', 'Value'],
            colWidths: [25, 25]
        });

        table.push(
            ['Symbol', result.symbol],
            ['Final Train Loss', result.training_results.final_train_loss.toFixed(6)],
            ['Final Val Loss', result.training_results.final_val_loss.toFixed(6)],
            ['Epochs Trained', result.training_results.epochs_trained],
            ['Converged', result.training_results.converged ? 'Yes' : 'No'],
            ['Model Parameters', result.model_info.total_parameters.toLocaleString()],
            ['Model Size', `${result.model_info.model_size_mb.toFixed(2)} MB`]
        );

        console.log(table.toString());

        if (result.test_metrics) {
            console.log(chalk.cyan('\n📊 Test Metrics:'));
            console.log(`  • MAE: ${result.test_metrics.mae.toFixed(4)}`);
            console.log(`  • RMSE: ${result.test_metrics.rmse.toFixed(4)}`);
            console.log(`  • Directional Accuracy: ${(result.test_metrics.directional_accuracy * 100).toFixed(1)}%`);
        }
    }

    displayAnalytics(analytics) {
        if (analytics.error) {
            console.log(chalk.red(`❌ ${analytics.error}`));
            return;
        }

        console.log(chalk.cyan('📊 Performance Overview:'));
        console.log(`  • Total Predictions: ${analytics.total_predictions}`);
        console.log(`  • Predictions with Actuals: ${analytics.predictions_with_actuals}`);
        console.log(`  • Directional Accuracy: ${(analytics.directional_accuracy * 100).toFixed(1)}%`);

        if (analytics.error_metrics) {
            console.log(chalk.cyan('\n📈 Error Metrics:'));
            console.log(`  • MAE: $${analytics.error_metrics.mae.toFixed(2)}`);
            console.log(`  • RMSE: $${analytics.error_metrics.rmse.toFixed(2)}`);
            console.log(`  • MAPE: ${analytics.error_metrics.mape.toFixed(1)}%`);
        }

        if (analytics.confidence_analysis) {
            console.log(chalk.cyan('\n🎯 Confidence Analysis:'));
            console.log(`  • Average Confidence: ${(analytics.confidence_analysis.avg_confidence * 100).toFixed(1)}%`);
            console.log(`  • High Confidence: ${analytics.confidence_analysis.confidence_distribution['high (0.7-1.0)']} predictions`);
        }
    }

    displayPerformanceMetrics(performance) {
        // Implementation for performance display
        console.log(chalk.cyan('📈 Performance Metrics:'));
        console.log(JSON.stringify(performance, null, 2));
    }

    displayMarketData(data) {
        const table = new Table({
            head: ['Metric', 'Value'],
            colWidths: [20, 20]
        });

        table.push(
            ['Symbol', data.symbol],
            ['Open', `$${data.open_price.toFixed(2)}`],
            ['High', `$${data.high_price.toFixed(2)}`],
            ['Low', `$${data.low_price.toFixed(2)}`],
            ['Close', `$${data.close_price.toFixed(2)}`],
            ['Volume', data.volume.toLocaleString()],
            ['Source', data.source]
        );

        console.log(table.toString());
    }

    displayHistoricalData(data) {
        console.log(chalk.cyan('\n📊 Historical Data (Last 5 days):'));
        
        const table = new Table({
            head: ['Date', 'Open', 'High', 'Low', 'Close', 'Volume'],
            colWidths: [12, 10, 10, 10, 10, 12]
        });

        data.slice(-5).forEach(day => {
            table.push([
                day.date,
                `$${day.open_price.toFixed(2)}`,
                `$${day.high_price.toFixed(2)}`,
                `$${day.low_price.toFixed(2)}`,
                `$${day.close_price.toFixed(2)}`,
                day.volume.toLocaleString()
            ]);
        });

        console.log(table.toString());
    }

    startProgressBar(message) {
        this.progressBar = new cliProgress.SingleBar({
            format: chalk.cyan(`${message} |`) + chalk.green('{bar}') + chalk.cyan('| {percentage}%'),
            barCompleteChar: '\u2588',
            barIncompleteChar: '\u2591',
            hideCursor: true
        });
        this.progressBar.start(100, 0);

        // Simulate progress
        let progress = 0;
        this.progressInterval = setInterval(() => {
            progress += Math.random() * 10;
            if (progress > 95) progress = 95;
            this.progressBar.update(progress);
        }, 500);
    }

    stopProgressBar() {
        if (this.progressBar) {
            clearInterval(this.progressInterval);
            this.progressBar.update(100);
            this.progressBar.stop();
            this.progressBar = null;
        }
    }
}

// Create CLI instance and parse arguments
const cli = new MLStockCLI();

// Handle uncaught errors
process.on('uncaughtException', (error) => {
    console.error(chalk.red(`\n❌ Uncaught Error: ${error.message}`));
    process.exit(1);
});

process.on('unhandledRejection', (error) => {
    console.error(chalk.red(`\n❌ Unhandled Promise Rejection: ${error.message}`));
    process.exit(1);
});

// Parse command line arguments
program.parse(process.argv);

// Show help if no command provided
if (!process.argv.slice(2).length) {
    console.log(chalk.blue.bold('\n🤖 ML Stock Predictor CLI\n'));
    program.outputHelp();
    console.log(chalk.gray('\nExamples:'));
    console.log(chalk.gray('  ml-stock-predictor predict AAPL'));
    console.log(chalk.gray('  ml-stock-predictor train GOOGL --days 90'));
    console.log(chalk.gray('  ml-stock-predictor analytics TSLA --days 30'));
    console.log(chalk.gray('  ml-stock-predictor interactive'));
}