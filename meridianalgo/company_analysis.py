"""
Comprehensive Company Analysis System
Risk Analysis, Financial Metrics, and Market Intelligence
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class CompanyAnalyzer:
    """
    Comprehensive company analysis with risk assessment and financial metrics
    """
    
    def __init__(self):
        self.risk_factors = {}
        self.financial_metrics = {}
        self.market_intelligence = {}
    
    def analyze_company(self, symbol):
        """
        Perform comprehensive company analysis
        """
        try:
            # Get company data
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Get historical data
            hist_data = ticker.history(period="2y")
            
            if hist_data.empty:
                return {"error": f"No data available for {symbol}"}
            
            # Perform all analyses
            analysis = {
                'symbol': symbol,
                'company_name': info.get('longName', symbol),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'current_price': hist_data['Close'].iloc[-1],
                
                # Financial Analysis
                'financial_health': self._analyze_financial_health(info, hist_data),
                
                # Risk Analysis
                'risk_assessment': self._perform_risk_analysis(info, hist_data),
                
                # Technical Analysis
                'technical_analysis': self._perform_technical_analysis(hist_data),
                
                # Valuation Analysis
                'valuation_metrics': self._analyze_valuation(info, hist_data),
                
                # Market Intelligence
                'market_intelligence': self._gather_market_intelligence(info, hist_data),
                
                # Growth Analysis
                'growth_analysis': self._analyze_growth_potential(info, hist_data),
                
                # Competitive Analysis
                'competitive_position': self._analyze_competitive_position(info),
                
                # ESG and Sustainability
                'esg_analysis': self._analyze_esg_factors(info),
                
                # Overall Score
                'overall_score': 0,
                'recommendation': 'HOLD'
            }
            
            # Calculate overall score and recommendation
            analysis['overall_score'], analysis['recommendation'] = self._calculate_overall_score(analysis)
            
            return analysis
            
        except Exception as e:
            return {"error": f"Analysis failed for {symbol}: {str(e)}"}
    
    def _analyze_financial_health(self, info, hist_data):
        """Analyze financial health and stability"""
        try:
            financial_health = {
                'debt_to_equity': info.get('debtToEquity', 0),
                'current_ratio': info.get('currentRatio', 0),
                'quick_ratio': info.get('quickRatio', 0),
                'return_on_equity': info.get('returnOnEquity', 0),
                'return_on_assets': info.get('returnOnAssets', 0),
                'profit_margin': info.get('profitMargins', 0),
                'operating_margin': info.get('operatingMargins', 0),
                'gross_margin': info.get('grossMargins', 0),
                'free_cash_flow': info.get('freeCashflow', 0),
                'total_cash': info.get('totalCash', 0),
                'total_debt': info.get('totalDebt', 0),
                'revenue_growth': info.get('revenueGrowth', 0),
                'earnings_growth': info.get('earningsGrowth', 0)
            }
            
            # Calculate financial health score
            score = 0
            max_score = 100
            
            # Debt analysis (20 points)
            debt_to_equity = financial_health['debt_to_equity']
            if debt_to_equity < 0.3:
                score += 20
            elif debt_to_equity < 0.6:
                score += 15
            elif debt_to_equity < 1.0:
                score += 10
            elif debt_to_equity < 2.0:
                score += 5
            
            # Liquidity analysis (20 points)
            current_ratio = financial_health['current_ratio']
            if current_ratio > 2.0:
                score += 20
            elif current_ratio > 1.5:
                score += 15
            elif current_ratio > 1.0:
                score += 10
            elif current_ratio > 0.8:
                score += 5
            
            # Profitability analysis (30 points)
            roe = financial_health['return_on_equity']
            if roe > 0.20:
                score += 15
            elif roe > 0.15:
                score += 12
            elif roe > 0.10:
                score += 8
            elif roe > 0.05:
                score += 4
            
            profit_margin = financial_health['profit_margin']
            if profit_margin > 0.20:
                score += 15
            elif profit_margin > 0.15:
                score += 12
            elif profit_margin > 0.10:
                score += 8
            elif profit_margin > 0.05:
                score += 4
            
            # Growth analysis (20 points)
            revenue_growth = financial_health['revenue_growth']
            if revenue_growth > 0.20:
                score += 10
            elif revenue_growth > 0.10:
                score += 8
            elif revenue_growth > 0.05:
                score += 5
            elif revenue_growth > 0:
                score += 2
            
            earnings_growth = financial_health['earnings_growth']
            if earnings_growth > 0.20:
                score += 10
            elif earnings_growth > 0.10:
                score += 8
            elif earnings_growth > 0.05:
                score += 5
            elif earnings_growth > 0:
                score += 2
            
            # Cash flow analysis (10 points)
            free_cash_flow = financial_health['free_cash_flow']
            if free_cash_flow > 0:
                score += 10
            elif free_cash_flow > -1000000000:  # -1B
                score += 5
            
            financial_health['health_score'] = (score / max_score) * 100
            financial_health['health_grade'] = self._get_grade(financial_health['health_score'])
            
            return financial_health
            
        except Exception as e:
            return {"error": f"Financial health analysis failed: {str(e)}"}
    
    def _perform_risk_analysis(self, info, hist_data):
        """Comprehensive risk analysis"""
        try:
            risk_analysis = {
                'beta': info.get('beta', 1.0),
                'volatility_52w': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'var_95': 0,  # Value at Risk 95%
                'liquidity_risk': 'Low',
                'concentration_risk': 'Medium',
                'market_risk': 'Medium',
                'credit_risk': 'Low',
                'operational_risk': 'Medium'
            }
            
            # Calculate volatility metrics
            returns = hist_data['Close'].pct_change().dropna()
            
            if len(returns) > 0:
                # Annualized volatility
                risk_analysis['volatility_52w'] = returns.std() * np.sqrt(252)
                
                # Maximum drawdown
                cumulative = (1 + returns).cumprod()
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max
                risk_analysis['max_drawdown'] = drawdown.min()
                
                # Sharpe ratio (assuming 2% risk-free rate)
                excess_returns = returns - 0.02/252
                if returns.std() > 0:
                    risk_analysis['sharpe_ratio'] = (excess_returns.mean() * 252) / (returns.std() * np.sqrt(252))
                
                # Value at Risk (95% confidence)
                risk_analysis['var_95'] = np.percentile(returns, 5)
            
            # Risk categorization based on beta
            beta = risk_analysis['beta']
            if beta < 0.8:
                risk_analysis['market_risk'] = 'Low'
            elif beta < 1.2:
                risk_analysis['market_risk'] = 'Medium'
            else:
                risk_analysis['market_risk'] = 'High'
            
            # Volatility risk
            volatility = risk_analysis['volatility_52w']
            if volatility < 0.2:
                risk_analysis['volatility_risk'] = 'Low'
            elif volatility < 0.4:
                risk_analysis['volatility_risk'] = 'Medium'
            else:
                risk_analysis['volatility_risk'] = 'High'
            
            # Liquidity risk based on volume
            avg_volume = hist_data['Volume'].mean()
            if avg_volume > 1000000:
                risk_analysis['liquidity_risk'] = 'Low'
            elif avg_volume > 100000:
                risk_analysis['liquidity_risk'] = 'Medium'
            else:
                risk_analysis['liquidity_risk'] = 'High'
            
            # Credit risk based on debt levels
            debt_to_equity = info.get('debtToEquity', 0)
            if debt_to_equity < 0.3:
                risk_analysis['credit_risk'] = 'Low'
            elif debt_to_equity < 1.0:
                risk_analysis['credit_risk'] = 'Medium'
            else:
                risk_analysis['credit_risk'] = 'High'
            
            # Overall risk score
            risk_score = 0
            
            # Beta risk (25 points)
            if beta < 0.8:
                risk_score += 25
            elif beta < 1.2:
                risk_score += 15
            elif beta < 1.5:
                risk_score += 5
            
            # Volatility risk (25 points)
            if volatility < 0.2:
                risk_score += 25
            elif volatility < 0.4:
                risk_score += 15
            elif volatility < 0.6:
                risk_score += 5
            
            # Drawdown risk (25 points)
            max_dd = abs(risk_analysis['max_drawdown'])
            if max_dd < 0.1:
                risk_score += 25
            elif max_dd < 0.2:
                risk_score += 15
            elif max_dd < 0.3:
                risk_score += 10
            elif max_dd < 0.5:
                risk_score += 5
            
            # Sharpe ratio (25 points)
            sharpe = risk_analysis['sharpe_ratio']
            if sharpe > 1.5:
                risk_score += 25
            elif sharpe > 1.0:
                risk_score += 20
            elif sharpe > 0.5:
                risk_score += 15
            elif sharpe > 0:
                risk_score += 10
            elif sharpe > -0.5:
                risk_score += 5
            
            risk_analysis['risk_score'] = risk_score
            risk_analysis['risk_grade'] = self._get_grade(risk_score)
            
            return risk_analysis
            
        except Exception as e:
            return {"error": f"Risk analysis failed: {str(e)}"}
    
    def _perform_technical_analysis(self, hist_data):
        """Advanced technical analysis"""
        try:
            # Calculate technical indicators
            close = hist_data['Close']
            high = hist_data['High']
            low = hist_data['Low']
            volume = hist_data['Volume']
            
            technical = {
                'trend_direction': 'Neutral',
                'trend_strength': 0,
                'support_levels': [],
                'resistance_levels': [],
                'momentum_score': 0,
                'volume_analysis': {},
                'pattern_signals': []
            }
            
            # Moving averages
            sma_20 = close.rolling(20).mean()
            sma_50 = close.rolling(50).mean()
            sma_200 = close.rolling(200).mean()
            ema_12 = close.ewm(span=12).mean()
            ema_26 = close.ewm(span=26).mean()
            
            # Current price vs moving averages
            current_price = close.iloc[-1]
            
            # Trend analysis
            trend_signals = 0
            if current_price > sma_20.iloc[-1]:
                trend_signals += 1
            if current_price > sma_50.iloc[-1]:
                trend_signals += 1
            if current_price > sma_200.iloc[-1]:
                trend_signals += 1
            if sma_20.iloc[-1] > sma_50.iloc[-1]:
                trend_signals += 1
            if sma_50.iloc[-1] > sma_200.iloc[-1]:
                trend_signals += 1
            
            if trend_signals >= 4:
                technical['trend_direction'] = 'Bullish'
                technical['trend_strength'] = trend_signals * 20
            elif trend_signals <= 1:
                technical['trend_direction'] = 'Bearish'
                technical['trend_strength'] = (5 - trend_signals) * 20
            else:
                technical['trend_direction'] = 'Neutral'
                technical['trend_strength'] = 50
            
            # RSI
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
            
            # MACD
            macd = ema_12 - ema_26
            macd_signal = macd.ewm(span=9).mean()
            macd_histogram = macd - macd_signal
            
            # Momentum analysis
            momentum_score = 0
            
            # RSI momentum
            if 30 < current_rsi < 70:
                momentum_score += 20
            elif current_rsi > 70:
                momentum_score += 10  # Overbought but strong
            elif current_rsi < 30:
                momentum_score += 10  # Oversold but weak
            
            # MACD momentum
            if macd.iloc[-1] > macd_signal.iloc[-1]:
                momentum_score += 20
            if macd_histogram.iloc[-1] > macd_histogram.iloc[-2]:
                momentum_score += 10
            
            # Price momentum
            price_change_1d = (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2]
            price_change_5d = (close.iloc[-1] - close.iloc[-6]) / close.iloc[-6]
            price_change_20d = (close.iloc[-1] - close.iloc[-21]) / close.iloc[-21]
            
            if price_change_1d > 0:
                momentum_score += 10
            if price_change_5d > 0:
                momentum_score += 15
            if price_change_20d > 0:
                momentum_score += 15
            
            technical['momentum_score'] = momentum_score
            
            # Support and resistance levels
            recent_data = close.tail(60)
            technical['support_levels'] = self._find_support_levels(recent_data)
            technical['resistance_levels'] = self._find_resistance_levels(recent_data)
            
            # Volume analysis
            avg_volume = volume.tail(20).mean()
            recent_volume = volume.iloc[-1]
            
            technical['volume_analysis'] = {
                'current_vs_average': recent_volume / avg_volume,
                'volume_trend': 'Increasing' if volume.tail(5).mean() > volume.tail(10).mean() else 'Decreasing',
                'volume_strength': 'High' if recent_volume > avg_volume * 1.5 else 'Normal' if recent_volume > avg_volume * 0.8 else 'Low'
            }
            
            # Pattern signals
            technical['pattern_signals'] = self._detect_simple_patterns(close, high, low)
            
            # Technical indicators summary
            technical['indicators'] = {
                'rsi': current_rsi,
                'macd': macd.iloc[-1],
                'macd_signal': macd_signal.iloc[-1],
                'sma_20': sma_20.iloc[-1],
                'sma_50': sma_50.iloc[-1],
                'sma_200': sma_200.iloc[-1],
                'bollinger_upper': (sma_20 + 2 * close.rolling(20).std()).iloc[-1],
                'bollinger_lower': (sma_20 - 2 * close.rolling(20).std()).iloc[-1]
            }
            
            return technical
            
        except Exception as e:
            return {"error": f"Technical analysis failed: {str(e)}"}
    
    def _analyze_valuation(self, info, hist_data):
        """Valuation analysis"""
        try:
            valuation = {
                'pe_ratio': info.get('trailingPE', 0),
                'forward_pe': info.get('forwardPE', 0),
                'peg_ratio': info.get('pegRatio', 0),
                'price_to_book': info.get('priceToBook', 0),
                'price_to_sales': info.get('priceToSalesTrailing12Months', 0),
                'enterprise_value': info.get('enterpriseValue', 0),
                'ev_to_revenue': info.get('enterpriseToRevenue', 0),
                'ev_to_ebitda': info.get('enterpriseToEbitda', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'payout_ratio': info.get('payoutRatio', 0)
            }
            
            # Valuation score
            score = 0
            
            # P/E ratio analysis (25 points)
            pe = valuation['pe_ratio']
            if 0 < pe < 15:
                score += 25
            elif pe < 20:
                score += 20
            elif pe < 25:
                score += 15
            elif pe < 30:
                score += 10
            elif pe < 40:
                score += 5
            
            # PEG ratio analysis (25 points)
            peg = valuation['peg_ratio']
            if 0 < peg < 1:
                score += 25
            elif peg < 1.5:
                score += 20
            elif peg < 2:
                score += 15
            elif peg < 3:
                score += 10
            
            # Price to book analysis (25 points)
            pb = valuation['price_to_book']
            if 0 < pb < 1:
                score += 25
            elif pb < 2:
                score += 20
            elif pb < 3:
                score += 15
            elif pb < 5:
                score += 10
            elif pb < 10:
                score += 5
            
            # Dividend analysis (25 points)
            div_yield = valuation['dividend_yield']
            if div_yield > 0.04:  # 4%+
                score += 15
            elif div_yield > 0.02:  # 2%+
                score += 10
            elif div_yield > 0:
                score += 5
            
            payout_ratio = valuation['payout_ratio']
            if 0 < payout_ratio < 0.6:
                score += 10
            elif payout_ratio < 0.8:
                score += 5
            
            valuation['valuation_score'] = score
            valuation['valuation_grade'] = self._get_grade(score)
            
            # Valuation summary
            if score >= 80:
                valuation['summary'] = 'Undervalued'
            elif score >= 60:
                valuation['summary'] = 'Fair Value'
            elif score >= 40:
                valuation['summary'] = 'Slightly Overvalued'
            else:
                valuation['summary'] = 'Overvalued'
            
            return valuation
            
        except Exception as e:
            return {"error": f"Valuation analysis failed: {str(e)}"}
    
    def _gather_market_intelligence(self, info, hist_data):
        """Market intelligence and sentiment analysis"""
        try:
            intelligence = {
                'market_cap_category': 'Unknown',
                'sector_performance': 'Average',
                'analyst_recommendations': {},
                'institutional_ownership': info.get('heldByInsiders', 0),
                'insider_ownership': info.get('heldByInstitutions', 0),
                'short_interest': info.get('shortRatio', 0),
                'price_targets': {
                    'high': info.get('targetHighPrice', 0),
                    'low': info.get('targetLowPrice', 0),
                    'mean': info.get('targetMeanPrice', 0),
                    'median': info.get('targetMedianPrice', 0)
                },
                'earnings_estimates': {},
                'market_sentiment': 'Neutral'
            }
            
            # Market cap categorization
            market_cap = info.get('marketCap', 0)
            if market_cap > 200_000_000_000:  # 200B+
                intelligence['market_cap_category'] = 'Mega Cap'
            elif market_cap > 10_000_000_000:  # 10B+
                intelligence['market_cap_category'] = 'Large Cap'
            elif market_cap > 2_000_000_000:  # 2B+
                intelligence['market_cap_category'] = 'Mid Cap'
            elif market_cap > 300_000_000:  # 300M+
                intelligence['market_cap_category'] = 'Small Cap'
            else:
                intelligence['market_cap_category'] = 'Micro Cap'
            
            # Analyst recommendations
            recommendations = info.get('recommendationKey', 'hold')
            intelligence['analyst_recommendations'] = {
                'recommendation': recommendations,
                'strong_buy': info.get('recommendationMean', 3.0) < 1.5,
                'buy': 1.5 <= info.get('recommendationMean', 3.0) < 2.5,
                'hold': 2.5 <= info.get('recommendationMean', 3.0) < 3.5,
                'sell': info.get('recommendationMean', 3.0) >= 3.5
            }
            
            # Market sentiment based on various factors
            sentiment_score = 0
            
            # Price vs targets
            current_price = hist_data['Close'].iloc[-1]
            target_mean = intelligence['price_targets']['mean']
            if target_mean > 0:
                upside = (target_mean - current_price) / current_price
                if upside > 0.2:
                    sentiment_score += 30
                elif upside > 0.1:
                    sentiment_score += 20
                elif upside > 0:
                    sentiment_score += 10
                elif upside > -0.1:
                    sentiment_score += 5
            
            # Institutional ownership
            inst_ownership = intelligence['institutional_ownership']
            if inst_ownership > 0.7:
                sentiment_score += 20
            elif inst_ownership > 0.5:
                sentiment_score += 15
            elif inst_ownership > 0.3:
                sentiment_score += 10
            
            # Short interest
            short_ratio = intelligence['short_interest']
            if short_ratio < 2:
                sentiment_score += 15
            elif short_ratio < 5:
                sentiment_score += 10
            elif short_ratio < 10:
                sentiment_score += 5
            
            # Recent performance
            price_change_30d = (hist_data['Close'].iloc[-1] - hist_data['Close'].iloc[-31]) / hist_data['Close'].iloc[-31]
            if price_change_30d > 0.1:
                sentiment_score += 20
            elif price_change_30d > 0:
                sentiment_score += 10
            elif price_change_30d > -0.1:
                sentiment_score += 5
            
            # Volume trend
            recent_volume = hist_data['Volume'].tail(10).mean()
            older_volume = hist_data['Volume'].tail(30).head(20).mean()
            if recent_volume > older_volume * 1.2:
                sentiment_score += 15
            elif recent_volume > older_volume:
                sentiment_score += 10
            
            if sentiment_score >= 70:
                intelligence['market_sentiment'] = 'Very Bullish'
            elif sentiment_score >= 50:
                intelligence['market_sentiment'] = 'Bullish'
            elif sentiment_score >= 30:
                intelligence['market_sentiment'] = 'Neutral'
            elif sentiment_score >= 15:
                intelligence['market_sentiment'] = 'Bearish'
            else:
                intelligence['market_sentiment'] = 'Very Bearish'
            
            intelligence['sentiment_score'] = sentiment_score
            
            return intelligence
            
        except Exception as e:
            return {"error": f"Market intelligence failed: {str(e)}"}
    
    def _analyze_growth_potential(self, info, hist_data):
        """Growth potential analysis"""
        try:
            growth = {
                'revenue_growth_rate': info.get('revenueGrowth', 0),
                'earnings_growth_rate': info.get('earningsGrowth', 0),
                'book_value_growth': 0,
                'dividend_growth': 0,
                'market_expansion': 'Medium',
                'competitive_advantages': [],
                'innovation_score': 0,
                'scalability_score': 0
            }
            
            # Calculate historical growth rates
            if len(hist_data) >= 252:  # 1 year of data
                # Price growth
                price_1y_ago = hist_data['Close'].iloc[-252]
                current_price = hist_data['Close'].iloc[-1]
                price_growth = (current_price - price_1y_ago) / price_1y_ago
                
                growth['price_growth_1y'] = price_growth
            
            # Growth score calculation
            score = 0
            
            # Revenue growth (30 points)
            rev_growth = growth['revenue_growth_rate']
            if rev_growth > 0.3:  # 30%+
                score += 30
            elif rev_growth > 0.2:  # 20%+
                score += 25
            elif rev_growth > 0.15:  # 15%+
                score += 20
            elif rev_growth > 0.1:  # 10%+
                score += 15
            elif rev_growth > 0.05:  # 5%+
                score += 10
            elif rev_growth > 0:
                score += 5
            
            # Earnings growth (30 points)
            earn_growth = growth['earnings_growth_rate']
            if earn_growth > 0.3:
                score += 30
            elif earn_growth > 0.2:
                score += 25
            elif earn_growth > 0.15:
                score += 20
            elif earn_growth > 0.1:
                score += 15
            elif earn_growth > 0.05:
                score += 10
            elif earn_growth > 0:
                score += 5
            
            # Market position (20 points)
            market_cap = info.get('marketCap', 0)
            if market_cap > 50_000_000_000:  # Large established companies
                score += 10  # Stable but limited growth
            elif market_cap > 2_000_000_000:  # Mid cap
                score += 20  # Good growth potential
            else:  # Small cap
                score += 15  # High growth potential but risky
            
            # Innovation indicators (20 points)
            sector = info.get('sector', '')
            if sector in ['Technology', 'Healthcare', 'Communication Services']:
                score += 15
            elif sector in ['Consumer Discretionary', 'Industrials']:
                score += 10
            elif sector in ['Consumer Staples', 'Financials']:
                score += 5
            
            # Additional growth factors
            if info.get('pegRatio', 0) < 1 and info.get('pegRatio', 0) > 0:
                score += 5  # Growth at reasonable price
            
            growth['growth_score'] = score
            growth['growth_grade'] = self._get_grade(score)
            
            # Growth category
            if score >= 80:
                growth['growth_category'] = 'High Growth'
            elif score >= 60:
                growth['growth_category'] = 'Moderate Growth'
            elif score >= 40:
                growth['growth_category'] = 'Slow Growth'
            else:
                growth['growth_category'] = 'Declining'
            
            return growth
            
        except Exception as e:
            return {"error": f"Growth analysis failed: {str(e)}"}
    
    def _analyze_competitive_position(self, info):
        """Analyze competitive position"""
        try:
            competitive = {
                'market_share': 'Unknown',
                'competitive_advantages': [],
                'moat_strength': 'Medium',
                'brand_strength': 'Medium',
                'pricing_power': 'Medium',
                'barriers_to_entry': 'Medium'
            }
            
            # Analyze based on available metrics
            profit_margin = info.get('profitMargins', 0)
            gross_margin = info.get('grossMargins', 0)
            roe = info.get('returnOnEquity', 0)
            
            # Competitive advantages based on margins
            if profit_margin > 0.2:
                competitive['competitive_advantages'].append('High Profit Margins')
                competitive['pricing_power'] = 'High'
            
            if gross_margin > 0.6:
                competitive['competitive_advantages'].append('Strong Gross Margins')
            
            if roe > 0.2:
                competitive['competitive_advantages'].append('Efficient Capital Use')
            
            # Market cap as proxy for market position
            market_cap = info.get('marketCap', 0)
            if market_cap > 100_000_000_000:
                competitive['market_share'] = 'Market Leader'
                competitive['moat_strength'] = 'Strong'
                competitive['brand_strength'] = 'Strong'
            elif market_cap > 10_000_000_000:
                competitive['market_share'] = 'Major Player'
                competitive['moat_strength'] = 'Medium'
            else:
                competitive['market_share'] = 'Niche Player'
                competitive['moat_strength'] = 'Weak'
            
            # Sector-specific analysis
            sector = info.get('sector', '')
            if sector == 'Technology':
                competitive['barriers_to_entry'] = 'High'
                competitive['competitive_advantages'].append('Technology Innovation')
            elif sector == 'Utilities':
                competitive['barriers_to_entry'] = 'Very High'
                competitive['moat_strength'] = 'Strong'
            elif sector == 'Healthcare':
                competitive['barriers_to_entry'] = 'High'
                competitive['competitive_advantages'].append('Regulatory Protection')
            
            return competitive
            
        except Exception as e:
            return {"error": f"Competitive analysis failed: {str(e)}"}
    
    def _analyze_esg_factors(self, info):
        """ESG (Environmental, Social, Governance) analysis"""
        try:
            esg = {
                'esg_score': 0,
                'environmental_score': 0,
                'social_score': 0,
                'governance_score': 0,
                'sustainability_initiatives': [],
                'governance_quality': 'Medium'
            }
            
            # Basic governance metrics
            insider_ownership = info.get('heldByInsiders', 0)
            institutional_ownership = info.get('heldByInstitutions', 0)
            
            # Governance scoring
            gov_score = 0
            
            # Insider ownership (should be reasonable, not too high or low)
            if 0.05 < insider_ownership < 0.3:
                gov_score += 25
            elif 0.01 < insider_ownership < 0.5:
                gov_score += 15
            
            # Institutional ownership (higher is generally better)
            if institutional_ownership > 0.7:
                gov_score += 25
            elif institutional_ownership > 0.5:
                gov_score += 20
            elif institutional_ownership > 0.3:
                gov_score += 15
            
            # Dividend policy (consistent dividends indicate good governance)
            dividend_yield = info.get('dividendYield', 0)
            payout_ratio = info.get('payoutRatio', 0)
            
            if dividend_yield > 0 and 0.2 < payout_ratio < 0.8:
                gov_score += 25
                esg['sustainability_initiatives'].append('Consistent Dividend Policy')
            elif dividend_yield > 0:
                gov_score += 15
            
            # Profitability consistency (proxy for management quality)
            roe = info.get('returnOnEquity', 0)
            if roe > 0.15:
                gov_score += 25
            elif roe > 0.1:
                gov_score += 15
            elif roe > 0.05:
                gov_score += 10
            
            esg['governance_score'] = gov_score
            
            # Overall ESG score (governance-heavy due to limited data)
            esg['esg_score'] = gov_score * 0.6 + 40  # Base score of 40 for E&S
            
            if esg['esg_score'] >= 80:
                esg['esg_grade'] = 'A'
            elif esg['esg_score'] >= 70:
                esg['esg_grade'] = 'B'
            elif esg['esg_score'] >= 60:
                esg['esg_grade'] = 'C'
            else:
                esg['esg_grade'] = 'D'
            
            return esg
            
        except Exception as e:
            return {"error": f"ESG analysis failed: {str(e)}"}
    
    def _find_support_levels(self, prices):
        """Find support levels"""
        try:
            support_levels = []
            
            # Find local minima
            for i in range(2, len(prices) - 2):
                if (prices.iloc[i] < prices.iloc[i-1] and 
                    prices.iloc[i] < prices.iloc[i+1] and
                    prices.iloc[i] < prices.iloc[i-2] and 
                    prices.iloc[i] < prices.iloc[i+2]):
                    support_levels.append(prices.iloc[i])
            
            # Remove duplicates and sort
            support_levels = sorted(list(set([round(level, 2) for level in support_levels])))
            
            # Return top 3 strongest support levels
            return support_levels[-3:] if len(support_levels) >= 3 else support_levels
            
        except Exception:
            return []
    
    def _find_resistance_levels(self, prices):
        """Find resistance levels"""
        try:
            resistance_levels = []
            
            # Find local maxima
            for i in range(2, len(prices) - 2):
                if (prices.iloc[i] > prices.iloc[i-1] and 
                    prices.iloc[i] > prices.iloc[i+1] and
                    prices.iloc[i] > prices.iloc[i-2] and 
                    prices.iloc[i] > prices.iloc[i+2]):
                    resistance_levels.append(prices.iloc[i])
            
            # Remove duplicates and sort
            resistance_levels = sorted(list(set([round(level, 2) for level in resistance_levels])))
            
            # Return top 3 strongest resistance levels
            return resistance_levels[-3:] if len(resistance_levels) >= 3 else resistance_levels
            
        except Exception:
            return []
    
    def _detect_simple_patterns(self, close, high, low):
        """Detect simple chart patterns"""
        try:
            patterns = []
            
            if len(close) < 20:
                return patterns
            
            recent_close = close.tail(20)
            recent_high = high.tail(20)
            recent_low = low.tail(20)
            
            # Bullish patterns
            if (recent_close.iloc[-1] > recent_close.iloc[-5] and
                recent_close.iloc[-5] > recent_close.iloc[-10]):
                patterns.append({
                    'pattern': 'Uptrend',
                    'signal': 'Bullish',
                    'strength': 'Medium'
                })
            
            # Bearish patterns
            if (recent_close.iloc[-1] < recent_close.iloc[-5] and
                recent_close.iloc[-5] < recent_close.iloc[-10]):
                patterns.append({
                    'pattern': 'Downtrend',
                    'signal': 'Bearish',
                    'strength': 'Medium'
                })
            
            # Consolidation
            if (max(recent_high) - min(recent_low)) / recent_close.iloc[-1] < 0.05:
                patterns.append({
                    'pattern': 'Consolidation',
                    'signal': 'Neutral',
                    'strength': 'Low'
                })
            
            return patterns
            
        except Exception:
            return []
    
    def _get_grade(self, score):
        """Convert score to letter grade"""
        if score >= 90:
            return 'A+'
        elif score >= 85:
            return 'A'
        elif score >= 80:
            return 'A-'
        elif score >= 75:
            return 'B+'
        elif score >= 70:
            return 'B'
        elif score >= 65:
            return 'B-'
        elif score >= 60:
            return 'C+'
        elif score >= 55:
            return 'C'
        elif score >= 50:
            return 'C-'
        elif score >= 45:
            return 'D+'
        elif score >= 40:
            return 'D'
        else:
            return 'F'
    
    def _calculate_overall_score(self, analysis):
        """Calculate overall investment score and recommendation"""
        try:
            # Weight different aspects
            weights = {
                'financial_health': 0.25,
                'risk_assessment': 0.20,
                'technical_analysis': 0.15,
                'valuation_metrics': 0.20,
                'growth_analysis': 0.15,
                'market_intelligence': 0.05
            }
            
            total_score = 0
            total_weight = 0
            
            for aspect, weight in weights.items():
                if aspect in analysis and isinstance(analysis[aspect], dict):
                    aspect_data = analysis[aspect]
                    
                    # Get score from each aspect
                    if aspect == 'financial_health' and 'health_score' in aspect_data:
                        total_score += aspect_data['health_score'] * weight
                        total_weight += weight
                    elif aspect == 'risk_assessment' and 'risk_score' in aspect_data:
                        total_score += aspect_data['risk_score'] * weight
                        total_weight += weight
                    elif aspect == 'technical_analysis' and 'momentum_score' in aspect_data:
                        total_score += aspect_data['momentum_score'] * weight
                        total_weight += weight
                    elif aspect == 'valuation_metrics' and 'valuation_score' in aspect_data:
                        total_score += aspect_data['valuation_score'] * weight
                        total_weight += weight
                    elif aspect == 'growth_analysis' and 'growth_score' in aspect_data:
                        total_score += aspect_data['growth_score'] * weight
                        total_weight += weight
                    elif aspect == 'market_intelligence' and 'sentiment_score' in aspect_data:
                        total_score += aspect_data['sentiment_score'] * weight
                        total_weight += weight
            
            # Normalize score
            if total_weight > 0:
                overall_score = total_score / total_weight
            else:
                overall_score = 50  # Default neutral score
            
            # Determine recommendation
            if overall_score >= 80:
                recommendation = 'STRONG BUY'
            elif overall_score >= 70:
                recommendation = 'BUY'
            elif overall_score >= 60:
                recommendation = 'WEAK BUY'
            elif overall_score >= 40:
                recommendation = 'HOLD'
            elif overall_score >= 30:
                recommendation = 'WEAK SELL'
            elif overall_score >= 20:
                recommendation = 'SELL'
            else:
                recommendation = 'STRONG SELL'
            
            return round(overall_score, 1), recommendation
            
        except Exception as e:
            return 50.0, 'HOLD'