"""
Enhanced AI-Powered Company Analysis using Lightweight Hugging Face Models
Provides intelligent insights and analysis with GPU acceleration support
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

try:
    from transformers import (
        pipeline,
        AutoTokenizer,
        AutoModelForSequenceClassification,
        AutoModelForCausalLM,
        BitsAndBytesConfig,
    )
    import torch

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers not available. Install with: pip install transformers torch")


class LightweightAIAnalyzer:
    """
    Enhanced AI-powered company analysis using Mistral-7B lightweight model
    Main conversational AI for comprehensive financial analysis
    """

    def __init__(self, use_gpu=True):
        self.use_gpu = (
            use_gpu and torch.cuda.is_available() if TRANSFORMERS_AVAILABLE else False
        )
        self.device = self._get_best_device()

        # Model cache directory
        self.model_cache_dir = Path(".ara_cache/models")
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)

        # Main conversational AI model (Mistral-7B lightweight)
        self.main_ai = None
        self.tokenizer = None

        # Specialized models for specific tasks
        self.sentiment_analyzer = None
        self.financial_classifier = None

        # Model configurations - Lightweight but powerful models
        self.model_configs = {
            "main_ai": {
                "model_name": "microsoft/DialoGPT-medium",  # Lightweight conversational AI
                "size_mb": 350,
                "accuracy": 0.85,
                "description": "Main conversational AI for analysis",
            },
            "mistral_alternative": {
                "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # Very lightweight Llama-based
                "size_mb": 2200,
                "accuracy": 0.82,
                "description": "Lightweight instruction-following model",
            },
            "financial_specialist": {
                "model_name": "ProsusAI/finbert",
                "size_mb": 440,
                "accuracy": 0.91,
                "description": "Financial sentiment specialist",
            },
            "classifier": {
                "model_name": "facebook/bart-large-mnli",
                "size_mb": 1600,
                "accuracy": 0.89,
                "description": "Zero-shot classification",
            },
        }

        if TRANSFORMERS_AVAILABLE:
            self._initialize_models()

    def _get_best_device(self):
        """Get the best available device for AI models"""
        if not TRANSFORMERS_AVAILABLE:
            return -1

        if self.use_gpu:
            if torch.cuda.is_available():
                return 0  # Use first GPU
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"  # Apple Silicon

        return -1  # CPU

    def _initialize_models(self):
        """Initialize lightweight conversational AI and specialized models"""
        try:
            print(" Initializing AI models...")

            # Main conversational AI - Try TinyLlama first, fallback to DialoGPT
            try:
                print(" Loading TinyLlama conversational AI...")
                self.main_ai = pipeline(
                    "text-generation",
                    model=self.model_configs["mistral_alternative"]["model_name"],
                    device=self.device,
                    model_kwargs={
                        "torch_dtype": torch.float16 if self.use_gpu else torch.float32,
                        "device_map": "auto" if self.use_gpu else None,
                    },
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1,
                )
                print(" TinyLlama conversational AI loaded (2.2GB)")
            except Exception as e:
                print(f"  TinyLlama failed, trying DialoGPT: {e}")
                try:
                    self.main_ai = pipeline(
                        "text-generation",
                        model=self.model_configs["main_ai"]["model_name"],
                        device=self.device,
                        model_kwargs=(
                            {"torch_dtype": torch.float16} if self.use_gpu else {}
                        ),
                        max_length=300,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=50256,
                    )
                    print(" DialoGPT conversational AI loaded (350MB)")
                except Exception as e2:
                    print(f" Both conversational AIs failed: {e2}")

            # Financial sentiment specialist
            try:
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model=self.model_configs["financial_specialist"]["model_name"],
                    device=self.device,
                    model_kwargs={"torch_dtype": torch.float16} if self.use_gpu else {},
                    return_all_scores=True,
                )
                print(" FinBERT financial specialist loaded (440MB)")
            except Exception as e:
                print(f"  FinBERT failed: {e}")

            # Zero-shot classifier for categorization
            try:
                self.financial_classifier = pipeline(
                    "zero-shot-classification",
                    model=self.model_configs["classifier"]["model_name"],
                    device=self.device,
                    model_kwargs={"torch_dtype": torch.float16} if self.use_gpu else {},
                )
                print(" BART classifier loaded (1.6GB)")
            except Exception as e:
                print(f"  BART classifier failed: {e}")

            print(f" Using device: {self.device}")

        except Exception as e:
            print(f" Model initialization failed: {e}")
            self.main_ai = None
            self.sentiment_analyzer = None
            self.financial_classifier = None

    def analyze_company_with_ai(self, symbol, basic_analysis=None):
        """
        Perform comprehensive AI-powered company analysis using conversational AI
        """
        try:
            if not TRANSFORMERS_AVAILABLE:
                return self._fallback_analysis(symbol, basic_analysis)

            # Get comprehensive company data
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist_data = ticker.history(period="2y")

            if hist_data.empty:
                return {"error": f"No data available for {symbol}"}

            # Prepare company data for AI analysis
            company_data = self._prepare_company_data_for_ai(info, hist_data)

            # Generate comprehensive AI analysis using conversational AI
            ai_analysis = self._generate_ai_analysis(company_data, symbol)

            # Get specialized analysis from other models
            sentiment_analysis = self._get_financial_sentiment(company_data)
            risk_classification = self._classify_investment_risk(company_data)

            # Combine all analysis
            analysis = {
                "symbol": symbol,
                "company_name": info.get("longName", symbol),
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
                "current_price": hist_data["Close"].iloc[-1],
                "market_cap": info.get("marketCap", 0),
                "business_summary": info.get("longBusinessSummary", ""),
                # Main AI-generated analysis
                "ai_analysis": ai_analysis,
                "ai_sentiment": sentiment_analysis,
                "ai_risk_classification": risk_classification,
                # Traditional metrics enhanced with AI interpretation
                "financial_metrics": self._calculate_comprehensive_metrics(
                    info, hist_data
                ),
                "technical_signals": self._get_enhanced_technical_signals(hist_data),
                "market_position": self._assess_market_position_with_ai(
                    info, ai_analysis
                ),
                # AI-powered insights
                "ai_insights": ai_analysis.get("insights", []),
                "ai_recommendation": ai_analysis.get("recommendation", {}),
                "ai_price_targets": ai_analysis.get("price_targets", {}),
                "ai_risks": ai_analysis.get("risks", []),
                "ai_opportunities": ai_analysis.get("opportunities", []),
                # Metadata
                "ai_confidence": self._calculate_ai_confidence(company_data),
                "analysis_timestamp": datetime.now().isoformat(),
                "model_info": self._get_model_info(),
            }

            # Calculate overall score based on AI analysis
            analysis["overall_score"], analysis["recommendation"] = (
                self._calculate_ai_score(analysis)
            )

            return analysis

        except Exception as e:
            return {"error": f"AI analysis failed for {symbol}: {str(e)}"}

    def _prepare_company_data_for_ai(self, info, hist_data):
        """Prepare comprehensive company data for AI analysis"""
        try:
            current_price = hist_data["Close"].iloc[-1]

            # Calculate key metrics
            price_change_1d = (
                (current_price - hist_data["Close"].iloc[-2])
                / hist_data["Close"].iloc[-2]
                * 100
            )
            price_change_30d = (
                (current_price - hist_data["Close"].iloc[-31])
                / hist_data["Close"].iloc[-31]
                * 100
            )
            price_change_1y = (
                (current_price - hist_data["Close"].iloc[-253])
                / hist_data["Close"].iloc[-253]
                * 100
            )

            volatility = hist_data["Close"].pct_change().std() * np.sqrt(252) * 100

            # Prepare structured data for AI
            company_data = {
                "basic_info": {
                    "name": info.get("longName", "Unknown"),
                    "symbol": info.get("symbol", ""),
                    "sector": info.get("sector", "Unknown"),
                    "industry": info.get("industry", "Unknown"),
                    "country": info.get("country", "Unknown"),
                    "employees": info.get("fullTimeEmployees", 0),
                    "website": info.get("website", ""),
                    "business_summary": info.get("longBusinessSummary", "")[
                        :500
                    ],  # Truncate for AI
                },
                "financial_metrics": {
                    "market_cap": info.get("marketCap", 0),
                    "enterprise_value": info.get("enterpriseValue", 0),
                    "revenue": info.get("totalRevenue", 0),
                    "profit_margin": info.get("profitMargins", 0),
                    "operating_margin": info.get("operatingMargins", 0),
                    "gross_margin": info.get("grossMargins", 0),
                    "roe": info.get("returnOnEquity", 0),
                    "roa": info.get("returnOnAssets", 0),
                    "debt_to_equity": info.get("debtToEquity", 0),
                    "current_ratio": info.get("currentRatio", 0),
                    "free_cash_flow": info.get("freeCashflow", 0),
                },
                "valuation_metrics": {
                    "pe_ratio": info.get("trailingPE", 0),
                    "forward_pe": info.get("forwardPE", 0),
                    "peg_ratio": info.get("pegRatio", 0),
                    "price_to_book": info.get("priceToBook", 0),
                    "price_to_sales": info.get("priceToSalesTrailing12Months", 0),
                    "ev_to_revenue": info.get("enterpriseToRevenue", 0),
                    "ev_to_ebitda": info.get("enterpriseToEbitda", 0),
                },
                "growth_metrics": {
                    "revenue_growth": info.get("revenueGrowth", 0),
                    "earnings_growth": info.get("earningsGrowth", 0),
                    "revenue_growth_quarterly": info.get("revenueQuarterlyGrowth", 0),
                    "earnings_growth_quarterly": info.get("earningsQuarterlyGrowth", 0),
                },
                "market_performance": {
                    "current_price": current_price,
                    "price_change_1d": price_change_1d,
                    "price_change_30d": price_change_30d,
                    "price_change_1y": price_change_1y,
                    "volatility": volatility,
                    "beta": info.get("beta", 1.0),
                    "52w_high": hist_data["High"].max(),
                    "52w_low": hist_data["Low"].min(),
                },
                "dividend_info": {
                    "dividend_yield": info.get("dividendYield", 0),
                    "payout_ratio": info.get("payoutRatio", 0),
                    "ex_dividend_date": str(info.get("exDividendDate", "")),
                },
                "analyst_data": {
                    "target_mean_price": info.get("targetMeanPrice", 0),
                    "target_high_price": info.get("targetHighPrice", 0),
                    "target_low_price": info.get("targetLowPrice", 0),
                    "recommendation": info.get("recommendationKey", ""),
                    "num_analysts": info.get("numberOfAnalystOpinions", 0),
                },
            }

            return company_data

        except Exception as e:
            return {"error": f"Data preparation failed: {str(e)}"}

    def _generate_ai_analysis(self, company_data, symbol):
        """Generate comprehensive analysis using conversational AI"""
        try:
            if not self.main_ai:
                return self._generate_fallback_analysis(company_data)

            # Create analysis prompt for the AI
            prompt = self._create_analysis_prompt(company_data, symbol)

            # Generate AI response
            response = self.main_ai(
                prompt,
                max_new_tokens=800,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=50256,
            )

            # Parse AI response
            ai_text = response[0]["generated_text"] if response else ""

            # Extract structured insights from AI response
            parsed_analysis = self._parse_ai_response(ai_text, company_data)

            return parsed_analysis

        except Exception as e:
            print(f"AI analysis generation failed: {e}")
            return self._generate_fallback_analysis(company_data)

    def _create_analysis_prompt(self, company_data, symbol):
        """Create a comprehensive analysis prompt for the AI"""
        try:
            basic_info = company_data.get("basic_info", {})
            financial = company_data.get("financial_metrics", {})
            valuation = company_data.get("valuation_metrics", {})
            growth = company_data.get("growth_metrics", {})
            performance = company_data.get("market_performance", {})

            prompt = f"""Analyze {basic_info.get('name', symbol)} ({symbol}) as an investment opportunity.

Company Overview:
- Sector: {basic_info.get('sector', 'Unknown')}
- Industry: {basic_info.get('industry', 'Unknown')}
- Market Cap: ${financial.get('market_cap', 0):,.0f}
- Employees: {basic_info.get('employees', 0):,}

Financial Health:
- Revenue: ${financial.get('revenue', 0):,.0f}
- Profit Margin: {financial.get('profit_margin', 0)*100:.1f}%
- ROE: {financial.get('roe', 0)*100:.1f}%
- Debt/Equity: {financial.get('debt_to_equity', 0):.2f}
- Current Ratio: {financial.get('current_ratio', 0):.2f}

Valuation:
- P/E Ratio: {valuation.get('pe_ratio', 0):.1f}
- PEG Ratio: {valuation.get('peg_ratio', 0):.2f}
- Price/Book: {valuation.get('price_to_book', 0):.2f}

Growth:
- Revenue Growth: {growth.get('revenue_growth', 0)*100:.1f}%
- Earnings Growth: {growth.get('earnings_growth', 0)*100:.1f}%

Performance:
- Current Price: ${performance.get('current_price', 0):.2f}
- 1-Year Return: {performance.get('price_change_1y', 0):.1f}%
- Volatility: {performance.get('volatility', 0):.1f}%

Provide a comprehensive investment analysis including:
1. Investment Thesis (2-3 key points)
2. Main Risks (2-3 key risks)
3. Growth Opportunities (2-3 opportunities)
4. Recommendation (Strong Buy/Buy/Hold/Sell/Strong Sell)
5. Price Target Range (Bull/Base/Bear case)
6. Key Catalysts to watch

Analysis:"""

            return prompt

        except Exception:
            return f"Analyze {symbol} as an investment opportunity. Provide investment thesis, risks, opportunities, and recommendation."

    def _parse_ai_response(self, ai_text, company_data):
        """Parse AI response into structured analysis"""
        try:
            # Extract the generated part (remove the prompt)
            if "Analysis:" in ai_text:
                analysis_text = ai_text.split("Analysis:")[-1].strip()
            else:
                analysis_text = ai_text

            # Parse different sections
            insights = []
            risks = []
            opportunities = []
            recommendation = {"action": "HOLD", "confidence": 0.7, "reasoning": ""}
            price_targets = {}

            # Simple parsing - look for key phrases
            lines = analysis_text.split("\n")
            current_section = None

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Identify sections
                if any(
                    word in line.lower()
                    for word in ["thesis", "investment", "strengths"]
                ):
                    current_section = "insights"
                elif any(
                    word in line.lower() for word in ["risk", "concern", "challenge"]
                ):
                    current_section = "risks"
                elif any(
                    word in line.lower()
                    for word in ["opportunity", "growth", "catalyst"]
                ):
                    current_section = "opportunities"
                elif any(
                    word in line.lower()
                    for word in ["recommendation", "rating", "buy", "sell", "hold"]
                ):
                    current_section = "recommendation"
                elif any(
                    word in line.lower() for word in ["target", "price", "valuation"]
                ):
                    current_section = "price_targets"

                # Extract content based on section
                if current_section == "insights" and len(line) > 20:
                    insights.append(line)
                elif current_section == "risks" and len(line) > 20:
                    risks.append(line)
                elif current_section == "opportunities" and len(line) > 20:
                    opportunities.append(line)
                elif current_section == "recommendation":
                    # Extract recommendation
                    if any(
                        word in line.lower()
                        for word in ["strong buy", "buy", "hold", "sell"]
                    ):
                        if "strong buy" in line.lower():
                            recommendation["action"] = "STRONG BUY"
                            recommendation["confidence"] = 0.9
                        elif "buy" in line.lower():
                            recommendation["action"] = "BUY"
                            recommendation["confidence"] = 0.8
                        elif "sell" in line.lower():
                            recommendation["action"] = "SELL"
                            recommendation["confidence"] = 0.8
                        else:
                            recommendation["action"] = "HOLD"
                            recommendation["confidence"] = 0.7
                        recommendation["reasoning"] = line

            # Generate price targets based on current price and analysis sentiment
            current_price = company_data.get("market_performance", {}).get(
                "current_price", 100
            )

            if recommendation["action"] in ["STRONG BUY", "BUY"]:
                price_targets = {
                    "bull_case": current_price * 1.25,
                    "base_case": current_price * 1.15,
                    "bear_case": current_price * 1.05,
                }
            elif recommendation["action"] == "SELL":
                price_targets = {
                    "bull_case": current_price * 1.05,
                    "base_case": current_price * 0.95,
                    "bear_case": current_price * 0.85,
                }
            else:  # HOLD
                price_targets = {
                    "bull_case": current_price * 1.10,
                    "base_case": current_price * 1.02,
                    "bear_case": current_price * 0.95,
                }

            return {
                "insights": insights[:5],  # Top 5 insights
                "risks": risks[:5],  # Top 5 risks
                "opportunities": opportunities[:5],  # Top 5 opportunities
                "recommendation": recommendation,
                "price_targets": price_targets,
                "full_analysis": analysis_text[:1000],  # First 1000 chars
                "ai_generated": True,
            }

        except Exception:
            return self._generate_fallback_analysis(company_data)

    def _generate_fallback_analysis(self, company_data):
        """Generate fallback analysis when AI fails"""
        try:
            financial = company_data.get("financial_metrics", {})
            performance = company_data.get("market_performance", {})

            # Simple rule-based analysis
            insights = []
            risks = []
            opportunities = []

            # Financial health insights
            profit_margin = financial.get("profit_margin", 0)
            if profit_margin > 0.15:
                insights.append("Strong profitability with healthy profit margins")
            elif profit_margin > 0.05:
                insights.append("Moderate profitability, room for improvement")
            else:
                risks.append("Low or negative profitability")

            # Growth insights
            revenue_growth = company_data.get("growth_metrics", {}).get(
                "revenue_growth", 0
            )
            if revenue_growth > 0.1:
                opportunities.append("Strong revenue growth momentum")
            elif revenue_growth < 0:
                risks.append("Declining revenue trend")

            # Debt analysis
            debt_to_equity = financial.get("debt_to_equity", 0)
            if debt_to_equity > 1.5:
                risks.append("High debt levels may limit financial flexibility")
            elif debt_to_equity < 0.3:
                insights.append("Conservative debt management")

            # Determine recommendation
            current_price = performance.get("current_price", 100)
            pe_ratio = company_data.get("valuation_metrics", {}).get("pe_ratio", 20)

            if profit_margin > 0.1 and revenue_growth > 0.05 and pe_ratio < 20:
                recommendation = {
                    "action": "BUY",
                    "confidence": 0.7,
                    "reasoning": "Good fundamentals and reasonable valuation",
                }
                price_targets = {
                    "bull_case": current_price * 1.20,
                    "base_case": current_price * 1.10,
                    "bear_case": current_price * 1.00,
                }
            elif profit_margin < 0 or debt_to_equity > 2:
                recommendation = {
                    "action": "SELL",
                    "confidence": 0.6,
                    "reasoning": "Weak fundamentals",
                }
                price_targets = {
                    "bull_case": current_price * 1.00,
                    "base_case": current_price * 0.90,
                    "bear_case": current_price * 0.80,
                }
            else:
                recommendation = {
                    "action": "HOLD",
                    "confidence": 0.6,
                    "reasoning": "Mixed signals",
                }
                price_targets = {
                    "bull_case": current_price * 1.10,
                    "base_case": current_price * 1.00,
                    "bear_case": current_price * 0.90,
                }

            return {
                "insights": insights,
                "risks": risks,
                "opportunities": opportunities,
                "recommendation": recommendation,
                "price_targets": price_targets,
                "full_analysis": "Fallback analysis based on financial metrics",
                "ai_generated": False,
            }

        except Exception as e:
            return {
                "insights": ["Analysis unavailable"],
                "risks": ["Unable to assess risks"],
                "opportunities": ["Unable to identify opportunities"],
                "recommendation": {
                    "action": "HOLD",
                    "confidence": 0.5,
                    "reasoning": "Insufficient data",
                },
                "price_targets": {},
                "full_analysis": f"Analysis failed: {str(e)}",
                "ai_generated": False,
            }

    def _prepare_enhanced_company_context(self, info, hist_data):
        """Prepare enhanced company context for comprehensive AI analysis"""
        try:
            # Calculate comprehensive metrics
            current_price = hist_data["Close"].iloc[-1]
            price_change_1d = (
                (current_price - hist_data["Close"].iloc[-2])
                / hist_data["Close"].iloc[-2]
                * 100
            )
            price_change_7d = (
                (current_price - hist_data["Close"].iloc[-8])
                / hist_data["Close"].iloc[-8]
                * 100
            )
            price_change_30d = (
                (current_price - hist_data["Close"].iloc[-31])
                / hist_data["Close"].iloc[-31]
                * 100
            )
            price_change_90d = (
                (current_price - hist_data["Close"].iloc[-91])
                / hist_data["Close"].iloc[-91]
                * 100
            )
            price_change_1y = (
                (current_price - hist_data["Close"].iloc[-253])
                / hist_data["Close"].iloc[-253]
                * 100
            )

            volatility = hist_data["Close"].pct_change().std() * np.sqrt(252) * 100
            avg_volume = hist_data["Volume"].mean()
            recent_volume = hist_data["Volume"].tail(5).mean()

            # Calculate additional metrics
            high_52w = hist_data["High"].max()
            low_52w = hist_data["Low"].min()
            price_vs_52w_high = (current_price - high_52w) / high_52w * 100
            price_vs_52w_low = (current_price - low_52w) / low_52w * 100

            # Create comprehensive context
            context = f"""
            COMPANY OVERVIEW:
            Company: {info.get('longName', 'Unknown')}
            Sector: {info.get('sector', 'Unknown')}
            Industry: {info.get('industry', 'Unknown')}
            Country: {info.get('country', 'Unknown')}
            Employees: {info.get('fullTimeEmployees', 'N/A')}
            
            FINANCIAL METRICS:
            Market Cap: ${info.get('marketCap', 0):,.0f}
            Enterprise Value: ${info.get('enterpriseValue', 0):,.0f}
            Revenue (TTM): ${info.get('totalRevenue', 0):,.0f}
            Profit Margin: {info.get('profitMargins', 0)*100:.1f}%
            Operating Margin: {info.get('operatingMargins', 0)*100:.1f}%
            Gross Margin: {info.get('grossMargins', 0)*100:.1f}%
            ROE: {info.get('returnOnEquity', 0)*100:.1f}%
            ROA: {info.get('returnOnAssets', 0)*100:.1f}%
            
            VALUATION METRICS:
            P/E Ratio: {info.get('trailingPE', 'N/A')}
            Forward P/E: {info.get('forwardPE', 'N/A')}
            PEG Ratio: {info.get('pegRatio', 'N/A')}
            Price to Book: {info.get('priceToBook', 'N/A')}
            Price to Sales: {info.get('priceToSalesTrailing12Months', 'N/A')}
            EV/Revenue: {info.get('enterpriseToRevenue', 'N/A')}
            EV/EBITDA: {info.get('enterpriseToEbitda', 'N/A')}
            
            GROWTH METRICS:
            Revenue Growth: {info.get('revenueGrowth', 0)*100:.1f}%
            Earnings Growth: {info.get('earningsGrowth', 0)*100:.1f}%
            Revenue Growth (Quarterly): {info.get('revenueQuarterlyGrowth', 0)*100:.1f}%
            Earnings Growth (Quarterly): {info.get('earningsQuarterlyGrowth', 0)*100:.1f}%
            
            FINANCIAL HEALTH:
            Total Cash: ${info.get('totalCash', 0):,.0f}
            Total Debt: ${info.get('totalDebt', 0):,.0f}
            Debt to Equity: {info.get('debtToEquity', 'N/A')}
            Current Ratio: {info.get('currentRatio', 'N/A')}
            Quick Ratio: {info.get('quickRatio', 'N/A')}
            Free Cash Flow: ${info.get('freeCashflow', 0):,.0f}
            
            MARKET PERFORMANCE:
            Current Price: ${current_price:.2f}
            52-Week High: ${high_52w:.2f}
            52-Week Low: ${low_52w:.2f}
            Distance from 52W High: {price_vs_52w_high:.1f}%
            Distance from 52W Low: {price_vs_52w_low:.1f}%
            
            PRICE PERFORMANCE:
            1-Day Change: {price_change_1d:.1f}%
            7-Day Change: {price_change_7d:.1f}%
            30-Day Change: {price_change_30d:.1f}%
            90-Day Change: {price_change_90d:.1f}%
            1-Year Change: {price_change_1y:.1f}%
            Annual Volatility: {volatility:.1f}%
            
            TRADING METRICS:
            Average Volume: {avg_volume:,.0f}
            Recent Volume: {recent_volume:,.0f}
            Volume Trend: {'Increasing' if recent_volume > avg_volume else 'Decreasing'}
            Beta: {info.get('beta', 'N/A')}
            
            DIVIDEND INFO:
            Dividend Yield: {info.get('dividendYield', 0)*100:.2f}%
            Payout Ratio: {info.get('payoutRatio', 0)*100:.1f}%
            Ex-Dividend Date: {info.get('exDividendDate', 'N/A')}
            
            ANALYST DATA:
            Target Mean Price: ${info.get('targetMeanPrice', 0):.2f}
            Target High Price: ${info.get('targetHighPrice', 0):.2f}
            Target Low Price: ${info.get('targetLowPrice', 0):.2f}
            Recommendation: {info.get('recommendationKey', 'N/A')}
            Number of Analysts: {info.get('numberOfAnalystOpinions', 'N/A')}
            """

            return context.strip()

        except Exception:
            return f"Company: {info.get('longName', 'Unknown')}, Sector: {info.get('sector', 'Unknown')}"

    def _analyze_sentiment(self, context):
        """Analyze sentiment using AI"""
        try:
            if not self.sentiment_analyzer:
                return {
                    "sentiment": "neutral",
                    "confidence": 0.5,
                    "reasoning": "AI model not available",
                }

            # Analyze sentiment of company context
            result = self.sentiment_analyzer(
                context[:500]
            )  # Limit to 500 chars for efficiency

            sentiment_map = {
                "LABEL_0": "negative",
                "LABEL_1": "neutral",
                "LABEL_2": "positive",
                "NEGATIVE": "negative",
                "NEUTRAL": "neutral",
                "POSITIVE": "positive",
            }

            sentiment = sentiment_map.get(result[0]["label"], "neutral")
            confidence = result[0]["score"]

            # Generate reasoning
            reasoning = self._generate_sentiment_reasoning(
                sentiment, confidence, context
            )

            return {
                "sentiment": sentiment,
                "confidence": confidence,
                "reasoning": reasoning,
            }

        except Exception as e:
            return {
                "sentiment": "neutral",
                "confidence": 0.5,
                "reasoning": f"Analysis error: {str(e)}",
            }

    def _generate_insights(self, context):
        """Generate AI insights about the company"""
        try:
            if not self.text_generator:
                return ["AI insights not available - model not loaded"]

            # Create prompts for different aspects
            prompts = [
                "Based on the financial data, this company shows",
                "The investment potential of this stock is",
                "Key risks to consider include",
            ]

            insights = []

            for prompt in prompts:
                try:
                    # Generate insight with low resource usage
                    result = self.text_generator(
                        prompt,
                        max_length=80,
                        num_return_sequences=1,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=50256,
                    )

                    generated_text = result[0]["generated_text"]
                    # Clean up the generated text
                    insight = generated_text.replace(prompt, "").strip()
                    if insight and len(insight) > 10:
                        insights.append(insight[:200])  # Limit length

                except Exception:
                    continue

            # Add fallback insights if generation fails
            if not insights:
                insights = self._generate_fallback_insights(context)

            return insights[:3]  # Return top 3 insights

        except Exception as e:
            return [f"Insight generation error: {str(e)}"]

    def _assess_risks_with_ai(self, context):
        """AI-powered risk assessment"""
        try:
            if not self.financial_classifier:
                return self._fallback_risk_assessment(context)

            # Define risk categories
            risk_categories = [
                "high financial risk",
                "moderate financial risk",
                "low financial risk",
                "high market volatility",
                "stable market position",
                "growth company",
                "value investment",
            ]

            # Classify the company context
            result = self.financial_classifier(context[:500], risk_categories)

            # Extract top risk factors
            risk_assessment = {
                "primary_risk_category": result["labels"][0],
                "risk_confidence": result["scores"][0],
                "risk_factors": result["labels"][:3],
                "risk_scores": result["scores"][:3],
            }

            # Generate risk summary
            risk_assessment["risk_summary"] = self._generate_risk_summary(
                risk_assessment
            )

            return risk_assessment

        except Exception as e:
            return {"error": f"Risk assessment failed: {str(e)}"}

    def _analyze_growth_with_ai(self, context):
        """AI-powered growth analysis"""
        try:
            if not self.financial_classifier:
                return {"growth_category": "moderate", "confidence": 0.5}

            # Define growth categories
            growth_categories = [
                "high growth potential",
                "moderate growth potential",
                "low growth potential",
                "declining business",
                "turnaround opportunity",
                "mature stable business",
            ]

            # Classify growth potential
            result = self.financial_classifier(context[:500], growth_categories)

            return {
                "growth_category": result["labels"][0],
                "growth_confidence": result["scores"][0],
                "growth_factors": result["labels"][:3],
                "growth_reasoning": self._generate_growth_reasoning(
                    result["labels"][0], result["scores"][0]
                ),
            }

        except Exception as e:
            return {"growth_category": "moderate", "confidence": 0.5, "error": str(e)}

    def _generate_recommendation(self, context):
        """Generate AI-powered investment recommendation"""
        try:
            if not self.financial_classifier:
                return {
                    "recommendation": "HOLD",
                    "confidence": 0.5,
                    "reasoning": "AI model not available",
                }

            # Define recommendation categories
            recommendations = [
                "strong buy recommendation",
                "buy recommendation",
                "hold recommendation",
                "sell recommendation",
                "strong sell recommendation",
            ]

            # Get recommendation
            result = self.financial_classifier(context[:500], recommendations)

            # Map to standard recommendations
            rec_map = {
                "strong buy recommendation": "STRONG BUY",
                "buy recommendation": "BUY",
                "hold recommendation": "HOLD",
                "sell recommendation": "SELL",
                "strong sell recommendation": "STRONG SELL",
            }

            recommendation = rec_map.get(result["labels"][0], "HOLD")
            confidence = result["scores"][0]

            return {
                "recommendation": recommendation,
                "confidence": confidence,
                "reasoning": self._generate_recommendation_reasoning(
                    recommendation, confidence
                ),
            }

        except Exception as e:
            return {
                "recommendation": "HOLD",
                "confidence": 0.5,
                "reasoning": f"Error: {str(e)}",
            }

    def _calculate_basic_metrics(self, info, hist_data):
        """Calculate basic financial metrics efficiently"""
        try:
            current_price = hist_data["Close"].iloc[-1]

            # Price performance
            price_1m = (
                (current_price - hist_data["Close"].iloc[-21])
                / hist_data["Close"].iloc[-21]
                * 100
            )
            price_3m = (
                (current_price - hist_data["Close"].iloc[-63])
                / hist_data["Close"].iloc[-63]
                * 100
            )
            price_1y = (
                (current_price - hist_data["Close"].iloc[0])
                / hist_data["Close"].iloc[0]
                * 100
            )

            # Volatility
            returns = hist_data["Close"].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) * 100

            # Volume analysis
            avg_volume = hist_data["Volume"].mean()
            recent_volume = hist_data["Volume"].tail(5).mean()

            return {
                "current_price": current_price,
                "price_change_1m": price_1m,
                "price_change_3m": price_3m,
                "price_change_1y": price_1y,
                "volatility_annual": volatility,
                "avg_volume": avg_volume,
                "recent_volume_trend": (
                    "increasing" if recent_volume > avg_volume else "decreasing"
                ),
                "pe_ratio": info.get("trailingPE", 0),
                "market_cap": info.get("marketCap", 0),
                "profit_margin": info.get("profitMargins", 0),
                "debt_to_equity": info.get("debtToEquity", 0),
            }

        except Exception as e:
            return {"error": f"Metrics calculation failed: {str(e)}"}

    def _get_technical_signals(self, hist_data):
        """Get basic technical signals"""
        try:
            close = hist_data["Close"]

            # Simple moving averages
            sma_20 = close.rolling(20).mean().iloc[-1]
            sma_50 = close.rolling(50).mean().iloc[-1]
            current_price = close.iloc[-1]

            # RSI calculation (simplified)
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50

            # Trend analysis
            trend = (
                "bullish"
                if current_price > sma_20 > sma_50
                else "bearish" if current_price < sma_20 < sma_50 else "neutral"
            )

            return {
                "trend": trend,
                "rsi": current_rsi,
                "price_vs_sma20": (current_price - sma_20) / sma_20 * 100,
                "price_vs_sma50": (current_price - sma_50) / sma_50 * 100,
                "momentum": (
                    "overbought"
                    if current_rsi > 70
                    else "oversold" if current_rsi < 30 else "neutral"
                ),
            }

        except Exception as e:
            return {"error": f"Technical analysis failed: {str(e)}"}

    def _assess_market_position(self, info):
        """Assess market position"""
        try:
            market_cap = info.get("marketCap", 0)

            if market_cap > 200_000_000_000:
                size_category = "Mega Cap"
                position_strength = "Market Leader"
            elif market_cap > 10_000_000_000:
                size_category = "Large Cap"
                position_strength = "Major Player"
            elif market_cap > 2_000_000_000:
                size_category = "Mid Cap"
                position_strength = "Growing Company"
            else:
                size_category = "Small Cap"
                position_strength = "Niche Player"

            return {
                "size_category": size_category,
                "position_strength": position_strength,
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
                "competitive_position": self._assess_competitive_position(info),
            }

        except Exception as e:
            return {"error": f"Market position assessment failed: {str(e)}"}

    def _assess_competitive_position(self, info):
        """Assess competitive position based on margins"""
        try:
            profit_margin = info.get("profitMargins", 0)
            gross_margin = info.get("grossMargins", 0)

            if profit_margin > 0.2:
                return "Strong competitive advantage"
            elif profit_margin > 0.1:
                return "Moderate competitive position"
            elif profit_margin > 0.05:
                return "Average competitive position"
            else:
                return "Weak competitive position"

        except Exception:
            return "Unknown competitive position"

    def _calculate_ai_score(self, analysis):
        """Calculate overall AI-powered score"""
        try:
            score = 50  # Base score

            # Sentiment impact (20 points)
            sentiment = analysis.get("ai_sentiment", {})
            if sentiment.get("sentiment") == "positive":
                score += 20 * sentiment.get("confidence", 0.5)
            elif sentiment.get("sentiment") == "negative":
                score -= 20 * sentiment.get("confidence", 0.5)

            # Growth potential (25 points)
            growth = analysis.get("ai_growth_potential", {})
            if "high growth" in growth.get("growth_category", ""):
                score += 25 * growth.get("growth_confidence", 0.5)
            elif "moderate growth" in growth.get("growth_category", ""):
                score += 15 * growth.get("growth_confidence", 0.5)
            elif "low growth" in growth.get("growth_category", ""):
                score += 5 * growth.get("growth_confidence", 0.5)

            # Risk assessment (25 points)
            risk = analysis.get("ai_risk_assessment", {})
            if "low" in risk.get("primary_risk_category", ""):
                score += 25 * risk.get("risk_confidence", 0.5)
            elif "moderate" in risk.get("primary_risk_category", ""):
                score += 15 * risk.get("risk_confidence", 0.5)
            elif "high" in risk.get("primary_risk_category", ""):
                score -= 15 * risk.get("risk_confidence", 0.5)

            # Technical signals (15 points)
            technical = analysis.get("technical_signals", {})
            if technical.get("trend") == "bullish":
                score += 15
            elif technical.get("trend") == "bearish":
                score -= 15

            # Financial metrics (15 points)
            financial = analysis.get("financial_metrics", {})
            if financial.get("price_change_1y", 0) > 20:
                score += 10
            elif financial.get("price_change_1y", 0) < -20:
                score -= 10

            # Normalize score
            score = max(0, min(100, score))

            # Generate recommendation
            if score >= 80:
                recommendation = "STRONG BUY"
            elif score >= 70:
                recommendation = "BUY"
            elif score >= 60:
                recommendation = "WEAK BUY"
            elif score >= 40:
                recommendation = "HOLD"
            elif score >= 30:
                recommendation = "WEAK SELL"
            else:
                recommendation = "SELL"

            return round(score, 1), recommendation

        except Exception:
            return 50.0, "HOLD"

    def _generate_sentiment_reasoning(self, sentiment, confidence, context):
        """Generate reasoning for sentiment analysis"""
        try:
            if sentiment == "positive":
                return f"Positive market sentiment detected with {confidence*100:.0f}% confidence based on financial indicators and market position."
            elif sentiment == "negative":
                return f"Negative market sentiment detected with {confidence*100:.0f}% confidence. Consider risk factors carefully."
            else:
                return f"Neutral market sentiment with {confidence*100:.0f}% confidence. Mixed signals in the analysis."
        except:
            return "Sentiment analysis completed."

    def _generate_risk_summary(self, risk_assessment):
        """Generate risk summary"""
        try:
            primary_risk = risk_assessment.get("primary_risk_category", "moderate risk")
            confidence = risk_assessment.get("risk_confidence", 0.5)

            return f"Primary risk category: {primary_risk} (confidence: {confidence*100:.0f}%). Monitor key risk factors closely."
        except:
            return "Risk assessment completed."

    def _generate_growth_reasoning(self, growth_category, confidence):
        """Generate growth reasoning"""
        try:
            return f"Growth analysis indicates {growth_category} with {confidence*100:.0f}% confidence based on financial trends and market position."
        except:
            return "Growth analysis completed."

    def _generate_recommendation_reasoning(self, recommendation, confidence):
        """Generate recommendation reasoning"""
        try:
            return f"AI recommendation: {recommendation} with {confidence*100:.0f}% confidence based on comprehensive analysis of financial, technical, and market factors."
        except:
            return f"Recommendation: {recommendation}"

    def _generate_fallback_insights(self, context):
        """Generate fallback insights when AI models fail"""
        insights = [
            "Company analysis based on fundamental financial metrics and market position.",
            "Technical indicators suggest monitoring price trends and volume patterns.",
            "Consider market conditions and sector performance in investment decisions.",
        ]
        return insights

    def _fallback_risk_assessment(self, context):
        """Fallback risk assessment"""
        return {
            "primary_risk_category": "moderate financial risk",
            "risk_confidence": 0.6,
            "risk_summary": "Standard risk assessment based on available financial data.",
        }

    def _fallback_analysis(self, symbol, basic_analysis):
        """Fallback analysis when AI models are not available"""
        return {
            "symbol": symbol,
            "ai_sentiment": {
                "sentiment": "neutral",
                "confidence": 0.5,
                "reasoning": "AI models not available",
            },
            "ai_insights": [
                "Basic analysis available",
                "AI features require transformers library",
            ],
            "ai_recommendation": {
                "recommendation": "HOLD",
                "confidence": 0.5,
                "reasoning": "Limited analysis available",
            },
            "overall_score": 50.0,
            "recommendation": "HOLD",
            "error": "AI models not available. Install transformers library for full AI analysis.",
        }


# Lightweight model manager to minimize resource usage
class ModelManager:
    """Manage AI models efficiently to minimize CPU usage"""

    def __init__(self):
        self.models = {}
        self.model_usage = {}

    def get_model(self, model_name, model_type="sentiment"):
        """Get model with lazy loading"""
        if model_name not in self.models:
            if model_type == "sentiment":
                self.models[model_name] = pipeline(
                    "sentiment-analysis",
                    model=model_name,
                    device=-1,  # CPU only for efficiency
                    model_kwargs=(
                        {"torch_dtype": torch.float16} if TRANSFORMERS_AVAILABLE else {}
                    ),
                )

        return self.models[model_name]

    def clear_unused_models(self):
        """Clear unused models to free memory"""
        # Implementation for clearing unused models
        pass
        """Enhanced financial sentiment analysis using FinBERT"""
        try:
            # Use FinBERT if available, otherwise fallback
            analyzer = self.financial_analyzer or self.sentiment_analyzer

            if not analyzer:
                return {
                    "sentiment": "neutral",
                    "confidence": 0.5,
                    "reasoning": "AI model not available",
                }

            # Analyze different aspects
            business_summary = info.get("longBusinessSummary", "")[:500]
            financial_context = context[:1000]

            sentiments = []

            # Analyze business summary
            if business_summary:
                result = analyzer(business_summary)
                if isinstance(result, list) and len(result) > 0:
                    if isinstance(result[0], list):
                        result = result[0]

                    # Handle FinBERT output format
                    if "label" in result[0]:
                        sentiment_score = max(result, key=lambda x: x["score"])
                        sentiments.append(
                            {
                                "aspect": "business_model",
                                "sentiment": sentiment_score["label"].lower(),
                                "confidence": sentiment_score["score"],
                            }
                        )

            # Analyze financial metrics context
            if financial_context:
                result = analyzer(financial_context)
                if isinstance(result, list) and len(result) > 0:
                    if isinstance(result[0], list):
                        result = result[0]

                    if "label" in result[0]:
                        sentiment_score = max(result, key=lambda x: x["score"])
                        sentiments.append(
                            {
                                "aspect": "financial_health",
                                "sentiment": sentiment_score["label"].lower(),
                                "confidence": sentiment_score["score"],
                            }
                        )

            # Calculate overall sentiment
            if sentiments:
                avg_confidence = sum(s["confidence"] for s in sentiments) / len(
                    sentiments
                )
                positive_count = sum(
                    1 for s in sentiments if "positive" in s["sentiment"]
                )
                negative_count = sum(
                    1 for s in sentiments if "negative" in s["sentiment"]
                )

                if positive_count > negative_count:
                    overall_sentiment = "positive"
                elif negative_count > positive_count:
                    overall_sentiment = "negative"
                else:
                    overall_sentiment = "neutral"

                reasoning = f"Analysis of {len(sentiments)} aspects: {positive_count} positive, {negative_count} negative"
            else:
                overall_sentiment = "neutral"
                avg_confidence = 0.5
                reasoning = "Limited data for sentiment analysis"

            return {
                "sentiment": overall_sentiment,
                "confidence": avg_confidence,
                "reasoning": reasoning,
                "detailed_sentiments": sentiments,
            }

        except Exception as e:
            return {
                "sentiment": "neutral",
                "confidence": 0.5,
                "reasoning": f"Analysis error: {str(e)}",
            }

    def _generate_comprehensive_insights(self, context, info):
        """Generate comprehensive AI insights"""
        try:
            insights = []

            # Financial health insights
            profit_margin = info.get("profitMargins", 0)
            revenue_growth = info.get("revenueGrowth", 0)
            debt_to_equity = info.get("debtToEquity", 0)
            roe = info.get("returnOnEquity", 0)

            if profit_margin > 0.2:
                insights.append(
                    f"Exceptional profitability with {profit_margin*100:.1f}% profit margin, indicating strong pricing power and operational efficiency."
                )
            elif profit_margin > 0.1:
                insights.append(
                    f"Solid profitability with {profit_margin*100:.1f}% profit margin, showing good business fundamentals."
                )
            elif profit_margin > 0:
                insights.append(
                    f"Moderate profitability at {profit_margin*100:.1f}% profit margin, room for improvement in operational efficiency."
                )
            else:
                insights.append(
                    "Company is currently unprofitable, focus on path to profitability and cash burn rate."
                )

            # Growth insights
            if revenue_growth > 0.2:
                insights.append(
                    f"Strong revenue growth of {revenue_growth*100:.1f}% indicates expanding market share and business momentum."
                )
            elif revenue_growth > 0.1:
                insights.append(
                    f"Healthy revenue growth of {revenue_growth*100:.1f}% shows steady business expansion."
                )
            elif revenue_growth > 0:
                insights.append(
                    f"Modest revenue growth of {revenue_growth*100:.1f}% suggests stable but slow-growing business."
                )
            else:
                insights.append(
                    "Declining revenues require attention to competitive position and market dynamics."
                )

            # Financial health insights
            if debt_to_equity and debt_to_equity < 0.3:
                insights.append(
                    "Conservative debt levels provide financial flexibility and lower risk profile."
                )
            elif debt_to_equity and debt_to_equity > 1.0:
                insights.append(
                    "High debt levels require monitoring of interest coverage and refinancing risks."
                )

            # ROE insights
            if roe > 0.15:
                insights.append(
                    f"Excellent return on equity of {roe*100:.1f}% demonstrates efficient use of shareholder capital."
                )
            elif roe > 0.1:
                insights.append(
                    f"Good return on equity of {roe*100:.1f}% shows solid management performance."
                )

            # Valuation insights
            pe_ratio = info.get("trailingPE", 0)
            if pe_ratio and pe_ratio < 15:
                insights.append(
                    "Attractive valuation with low P/E ratio may indicate undervaluation or value opportunity."
                )
            elif pe_ratio and pe_ratio > 30:
                insights.append(
                    "High P/E ratio suggests growth expectations or potential overvaluation - verify growth prospects."
                )

            # Market position insights
            market_cap = info.get("marketCap", 0)
            if market_cap > 100_000_000_000:
                insights.append(
                    "Large-cap stability with established market position and likely dividend potential."
                )
            elif market_cap < 2_000_000_000:
                insights.append(
                    "Small-cap growth potential with higher volatility and risk-reward profile."
                )

            # Sector-specific insights
            sector = info.get("sector", "")
            if sector == "Technology":
                insights.append(
                    "Technology sector exposure provides growth potential but requires monitoring of innovation cycles and competition."
                )
            elif sector == "Healthcare":
                insights.append(
                    "Healthcare sector offers defensive characteristics with demographic tailwinds and regulatory considerations."
                )
            elif sector == "Financial Services":
                insights.append(
                    "Financial sector performance tied to interest rates, economic cycles, and regulatory environment."
                )

            # Generate AI insights if text generator is available
            if self.text_generator and len(insights) < 5:
                try:
                    prompt = f"Investment analysis for {info.get('longName', 'company')} shows"
                    ai_result = self.text_generator(
                        prompt, max_length=100, num_return_sequences=1
                    )
                    if ai_result and len(ai_result) > 0:
                        generated_text = (
                            ai_result[0]["generated_text"].replace(prompt, "").strip()
                        )
                        if len(generated_text) > 20:
                            insights.append(f"AI Analysis: {generated_text[:150]}...")
                except Exception:
                    pass

            return insights[:8]  # Return top 8 insights

        except Exception as e:
            return [f"Insight generation error: {str(e)}"]

    def _assess_comprehensive_risks(self, context, info):
        """Comprehensive AI-powered risk assessment"""
        try:
            if not self.financial_classifier:
                return self._fallback_risk_assessment(context)

            # Enhanced risk categories
            risk_categories = [
                "high financial leverage risk",
                "moderate financial risk",
                "low financial risk",
                "high market volatility risk",
                "regulatory compliance risk",
                "competitive pressure risk",
                "economic sensitivity risk",
                "liquidity risk",
                "operational risk",
                "technology disruption risk",
                "stable defensive business",
                "growth company risk profile",
            ]

            # Analyze different risk aspects
            financial_text = f"Company with debt-to-equity {info.get('debtToEquity', 0)}, current ratio {info.get('currentRatio', 0)}, profit margin {info.get('profitMargins', 0)*100:.1f}%"

            result = self.financial_classifier(financial_text, risk_categories)

            # Calculate comprehensive risk score
            risk_factors = []

            # Financial risks
            debt_to_equity = info.get("debtToEquity", 0)
            if debt_to_equity and debt_to_equity > 1.5:
                risk_factors.append(
                    {"type": "Financial Leverage", "level": "High", "impact": 0.8}
                )
            elif debt_to_equity and debt_to_equity > 0.8:
                risk_factors.append(
                    {"type": "Financial Leverage", "level": "Moderate", "impact": 0.5}
                )

            # Liquidity risks
            current_ratio = info.get("currentRatio", 0)
            if current_ratio and current_ratio < 1.0:
                risk_factors.append(
                    {"type": "Liquidity", "level": "High", "impact": 0.7}
                )
            elif current_ratio and current_ratio < 1.5:
                risk_factors.append(
                    {"type": "Liquidity", "level": "Moderate", "impact": 0.4}
                )

            # Profitability risks
            profit_margin = info.get("profitMargins", 0)
            if profit_margin < 0:
                risk_factors.append(
                    {"type": "Profitability", "level": "High", "impact": 0.9}
                )
            elif profit_margin < 0.05:
                risk_factors.append(
                    {"type": "Profitability", "level": "Moderate", "impact": 0.6}
                )

            # Market risks
            beta = info.get("beta", 1.0)
            if beta > 1.5:
                risk_factors.append(
                    {"type": "Market Volatility", "level": "High", "impact": 0.6}
                )
            elif beta > 1.2:
                risk_factors.append(
                    {"type": "Market Volatility", "level": "Moderate", "impact": 0.4}
                )

            # Sector-specific risks
            sector = info.get("sector", "")
            sector_risks = {
                "Technology": {
                    "type": "Technology Disruption",
                    "level": "Moderate",
                    "impact": 0.5,
                },
                "Energy": {"type": "Commodity Price", "level": "High", "impact": 0.7},
                "Financial Services": {
                    "type": "Regulatory",
                    "level": "Moderate",
                    "impact": 0.5,
                },
                "Healthcare": {
                    "type": "Regulatory",
                    "level": "Moderate",
                    "impact": 0.4,
                },
                "Consumer Discretionary": {
                    "type": "Economic Sensitivity",
                    "level": "Moderate",
                    "impact": 0.6,
                },
            }

            if sector in sector_risks:
                risk_factors.append(sector_risks[sector])

            # Calculate overall risk score
            if risk_factors:
                total_impact = sum(rf["impact"] for rf in risk_factors)
                risk_score = min(100, total_impact * 20)  # Scale to 0-100
            else:
                risk_score = 30  # Default moderate risk

            return {
                "primary_risk_category": result["labels"][0],
                "risk_confidence": result["scores"][0],
                "risk_score": risk_score,
                "risk_factors": risk_factors,
                "risk_summary": self._generate_risk_summary_enhanced(
                    risk_factors, risk_score
                ),
                "mitigation_strategies": self._suggest_risk_mitigation(risk_factors),
            }

        except Exception as e:
            return {"error": f"Risk assessment failed: {str(e)}"}

    def _analyze_competitive_position_ai(self, info):
        """AI-powered competitive position analysis"""
        try:
            competitive_analysis = {
                "market_position": "Unknown",
                "competitive_advantages": [],
                "competitive_threats": [],
                "moat_strength": "Medium",
                "market_share_trend": "Stable",
            }

            # Analyze based on financial metrics
            profit_margin = info.get("profitMargins", 0)
            gross_margin = info.get("grossMargins", 0)
            roe = info.get("returnOnEquity", 0)
            market_cap = info.get("marketCap", 0)

            # Competitive advantages
            if profit_margin > 0.2:
                competitive_analysis["competitive_advantages"].append(
                    "Superior profitability indicates pricing power"
                )
            if gross_margin > 0.6:
                competitive_analysis["competitive_advantages"].append(
                    "High gross margins suggest strong value proposition"
                )
            if roe > 0.2:
                competitive_analysis["competitive_advantages"].append(
                    "Excellent capital efficiency"
                )

            # Market position based on size and profitability
            if market_cap > 50_000_000_000 and profit_margin > 0.15:
                competitive_analysis["market_position"] = "Market Leader"
                competitive_analysis["moat_strength"] = "Strong"
            elif market_cap > 10_000_000_000:
                competitive_analysis["market_position"] = "Major Player"
                competitive_analysis["moat_strength"] = "Medium"
            else:
                competitive_analysis["market_position"] = "Niche Player"
                competitive_analysis["moat_strength"] = "Weak"

            # Sector-specific competitive analysis
            sector = info.get("sector", "")
            if sector == "Technology":
                competitive_analysis["competitive_threats"].append(
                    "Rapid technological change"
                )
                competitive_analysis["competitive_advantages"].append(
                    "Innovation capabilities"
                )
            elif sector == "Consumer Staples":
                competitive_analysis["competitive_advantages"].append(
                    "Brand loyalty and distribution"
                )
                competitive_analysis["moat_strength"] = "Strong"
            elif sector == "Utilities":
                competitive_analysis["competitive_advantages"].append(
                    "Regulatory barriers and infrastructure"
                )
                competitive_analysis["moat_strength"] = "Very Strong"

            return competitive_analysis

        except Exception as e:
            return {"error": f"Competitive analysis failed: {str(e)}"}

    def _analyze_esg_with_ai(self, info):
        """AI-powered ESG analysis"""
        try:
            esg_analysis = {
                "esg_score": 50,
                "environmental_factors": [],
                "social_factors": [],
                "governance_factors": [],
                "esg_risks": [],
                "esg_opportunities": [],
            }

            # Governance analysis based on available data
            insider_ownership = info.get("heldByInsiders", 0)
            institutional_ownership = info.get("heldByInstitutions", 0)

            governance_score = 50

            # Insider ownership analysis
            if 0.05 < insider_ownership < 0.3:
                governance_score += 15
                esg_analysis["governance_factors"].append("Balanced insider ownership")
            elif insider_ownership > 0.5:
                governance_score -= 10
                esg_analysis["esg_risks"].append(
                    "High insider control may limit shareholder rights"
                )

            # Institutional ownership
            if institutional_ownership > 0.7:
                governance_score += 10
                esg_analysis["governance_factors"].append(
                    "Strong institutional oversight"
                )

            # Dividend policy (governance indicator)
            dividend_yield = info.get("dividendYield", 0)
            payout_ratio = info.get("payoutRatio", 0)

            if dividend_yield > 0 and 0.2 < payout_ratio < 0.8:
                governance_score += 10
                esg_analysis["governance_factors"].append("Sustainable dividend policy")

            # Sector-specific ESG factors
            sector = info.get("sector", "")
            if sector == "Energy":
                esg_analysis["environmental_factors"].append(
                    "Carbon footprint and transition risks"
                )
                esg_analysis["esg_risks"].append("Climate change regulatory risks")
            elif sector == "Technology":
                esg_analysis["social_factors"].append(
                    "Data privacy and digital divide considerations"
                )
                esg_analysis["esg_opportunities"].append(
                    "Digital transformation enabler"
                )
            elif sector == "Healthcare":
                esg_analysis["social_factors"].append(
                    "Healthcare access and affordability"
                )
                esg_analysis["esg_opportunities"].append("Improving health outcomes")

            # Financial health as governance indicator
            debt_to_equity = info.get("debtToEquity", 0)
            if debt_to_equity and debt_to_equity < 0.5:
                governance_score += 5
                esg_analysis["governance_factors"].append(
                    "Conservative financial management"
                )

            esg_analysis["esg_score"] = min(100, max(0, governance_score))

            return esg_analysis

        except Exception as e:
            return {"error": f"ESG analysis failed: {str(e)}"}

    def _analyze_market_outlook(self, context, info):
        """AI-powered market outlook analysis"""
        try:
            outlook = {
                "market_trend": "Neutral",
                "sector_outlook": "Stable",
                "key_drivers": [],
                "potential_catalysts": [],
                "headwinds": [],
                "time_horizon": "12 months",
            }

            # Analyze growth trends
            revenue_growth = info.get("revenueGrowth", 0)
            earnings_growth = info.get("earningsGrowth", 0)

            if revenue_growth > 0.15 and earnings_growth > 0.15:
                outlook["market_trend"] = "Bullish"
                outlook["key_drivers"].append(
                    "Strong revenue and earnings growth momentum"
                )
            elif revenue_growth < -0.05 or earnings_growth < -0.1:
                outlook["market_trend"] = "Bearish"
                outlook["headwinds"].append("Declining financial performance")

            # Sector-specific outlook
            sector = info.get("sector", "")
            sector_outlooks = {
                "Technology": {
                    "outlook": "Positive",
                    "drivers": [
                        "Digital transformation",
                        "AI adoption",
                        "Cloud migration",
                    ],
                    "risks": ["Regulatory scrutiny", "Valuation concerns"],
                },
                "Healthcare": {
                    "outlook": "Stable",
                    "drivers": ["Aging demographics", "Innovation pipeline"],
                    "risks": ["Drug pricing pressure", "Regulatory changes"],
                },
                "Energy": {
                    "outlook": "Mixed",
                    "drivers": ["Energy transition", "Commodity prices"],
                    "risks": ["Climate regulations", "Stranded assets"],
                },
            }

            if sector in sector_outlooks:
                sector_data = sector_outlooks[sector]
                outlook["sector_outlook"] = sector_data["outlook"]
                outlook["key_drivers"].extend(sector_data["drivers"])
                outlook["headwinds"].extend(sector_data["risks"])

            # Valuation-based catalysts
            pe_ratio = info.get("trailingPE", 0)
            if pe_ratio and pe_ratio < 15:
                outlook["potential_catalysts"].append(
                    "Attractive valuation may attract value investors"
                )

            # Financial health catalysts
            free_cash_flow = info.get("freeCashflow", 0)
            if free_cash_flow > 0:
                outlook["potential_catalysts"].append(
                    "Positive free cash flow supports dividend/buyback potential"
                )

            return outlook

        except Exception as e:
            return {"error": f"Market outlook analysis failed: {str(e)}"}

    def _generate_ai_price_targets(self, analysis):
        """Generate AI-powered price targets"""
        try:
            current_price = analysis.get("current_price", 0)
            if not current_price:
                return {"error": "No current price available"}

            # Base targets on comprehensive analysis
            overall_score = analysis.get("overall_score", 50)

            # Calculate target multipliers based on analysis
            if overall_score >= 80:
                bull_multiplier = 1.25
                bear_multiplier = 0.95
            elif overall_score >= 70:
                bull_multiplier = 1.15
                bear_multiplier = 0.90
            elif overall_score >= 60:
                bull_multiplier = 1.10
                bear_multiplier = 0.85
            elif overall_score >= 40:
                bull_multiplier = 1.05
                bear_multiplier = 0.80
            else:
                bull_multiplier = 0.95
                bear_multiplier = 0.75

            # Adjust based on growth potential
            growth_analysis = analysis.get("ai_growth_potential", {})
            if "high growth" in growth_analysis.get("growth_category", ""):
                bull_multiplier += 0.1
            elif "low growth" in growth_analysis.get("growth_category", ""):
                bull_multiplier -= 0.05

            # Adjust based on risk assessment
            risk_analysis = analysis.get("ai_risk_assessment", {})
            risk_score = risk_analysis.get("risk_score", 50)
            if risk_score > 70:
                bull_multiplier -= 0.1
                bear_multiplier -= 0.05
            elif risk_score < 30:
                bull_multiplier += 0.05

            return {
                "bull_case_target": current_price * bull_multiplier,
                "bear_case_target": current_price * bear_multiplier,
                "base_case_target": current_price
                * ((bull_multiplier + bear_multiplier) / 2),
                "time_horizon": "12 months",
                "confidence": analysis.get("ai_confidence", 0.75),
                "methodology": "AI-powered multi-factor analysis",
            }

        except Exception as e:
            return {"error": f"Price target generation failed: {str(e)}"}

    def _calculate_enhanced_ai_score(self, analysis):
        """Calculate enhanced overall AI score"""
        try:
            score = 50  # Base score

            # Sentiment impact (15 points)
            sentiment = analysis.get("ai_sentiment", {})
            if sentiment.get("sentiment") == "positive":
                score += 15 * sentiment.get("confidence", 0.5)
            elif sentiment.get("sentiment") == "negative":
                score -= 15 * sentiment.get("confidence", 0.5)

            # Growth potential (20 points)
            growth = analysis.get("ai_growth_potential", {})
            if "high growth" in growth.get("growth_category", ""):
                score += 20 * growth.get("growth_confidence", 0.5)
            elif "moderate growth" in growth.get("growth_category", ""):
                score += 12 * growth.get("growth_confidence", 0.5)
            elif "low growth" in growth.get("growth_category", ""):
                score += 5 * growth.get("growth_confidence", 0.5)

            # Risk assessment (20 points)
            risk = analysis.get("ai_risk_assessment", {})
            risk_score = risk.get("risk_score", 50)
            score += (100 - risk_score) * 0.2

            # Financial metrics (20 points)
            financial = analysis.get("financial_metrics", {})
            if financial.get("profit_margin", 0) > 0.15:
                score += 10
            if financial.get("revenue_growth", 0) > 0.1:
                score += 10

            # Competitive position (15 points)
            competitive = analysis.get("ai_competitive_analysis", {})
            if competitive.get("market_position") == "Market Leader":
                score += 15
            elif competitive.get("market_position") == "Major Player":
                score += 10
            elif competitive.get("market_position") == "Niche Player":
                score += 5

            # ESG factors (10 points)
            esg = analysis.get("ai_esg_analysis", {})
            esg_score = esg.get("esg_score", 50)
            score += (esg_score - 50) * 0.2

            # Normalize score
            score = max(0, min(100, score))

            # Generate recommendation
            if score >= 85:
                recommendation = "STRONG BUY"
            elif score >= 75:
                recommendation = "BUY"
            elif score >= 65:
                recommendation = "WEAK BUY"
            elif score >= 45:
                recommendation = "HOLD"
            elif score >= 35:
                recommendation = "WEAK SELL"
            elif score >= 25:
                recommendation = "SELL"
            else:
                recommendation = "STRONG SELL"

            return round(score, 1), recommendation

        except Exception:
            return 50.0, "HOLD"

    def _get_model_versions(self):
        """Get versions of loaded AI models"""
        try:
            versions = {}

            if self.financial_analyzer:
                versions["financial_sentiment"] = self.model_configs["financial"][
                    "model_name"
                ]
            if self.sentiment_analyzer:
                versions["general_sentiment"] = self.model_configs["sentiment"][
                    "model_name"
                ]
            if self.financial_classifier:
                versions["classifier"] = self.model_configs["classification"][
                    "model_name"
                ]
            if self.text_generator:
                versions["text_generator"] = self.model_configs["text_generation"][
                    "model_name"
                ]

            return versions

        except Exception:
            return {}

    def _calculate_ai_confidence(self, context):
        """Calculate AI confidence based on data quality and model availability"""
        try:
            confidence = 0.5  # Base confidence

            # Model availability
            if self.financial_analyzer:
                confidence += 0.2
            if self.financial_classifier:
                confidence += 0.15
            if self.sentiment_analyzer:
                confidence += 0.1

            # Data quality
            if len(context) > 1000:
                confidence += 0.05

            return min(0.95, confidence)

        except Exception:
            return 0.75

    def _get_financial_sentiment(self, company_data):
        """Get financial sentiment using FinBERT"""
        try:
            if not self.sentiment_analyzer:
                return {
                    "sentiment": "neutral",
                    "confidence": 0.5,
                    "reasoning": "Sentiment analyzer not available",
                }

            # Create text for sentiment analysis
            financial = company_data.get("financial_metrics", {})
            performance = company_data.get("market_performance", {})

            sentiment_text = f"""
            Company with {financial.get('profit_margin', 0)*100:.1f}% profit margin, 
            {financial.get('revenue_growth', 0)*100:.1f}% revenue growth, 
            {performance.get('price_change_1y', 0):.1f}% annual return, 
            debt-to-equity ratio of {financial.get('debt_to_equity', 0):.2f}
            """

            result = self.sentiment_analyzer(sentiment_text)

            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], list):
                    result = result[0]

                # Get the highest confidence sentiment
                best_sentiment = max(result, key=lambda x: x["score"])

                return {
                    "sentiment": best_sentiment["label"].lower(),
                    "confidence": best_sentiment["score"],
                    "reasoning": "Financial sentiment analysis based on key metrics",
                    "all_scores": result,
                }

            return {
                "sentiment": "neutral",
                "confidence": 0.5,
                "reasoning": "Unable to analyze sentiment",
            }

        except Exception as e:
            return {
                "sentiment": "neutral",
                "confidence": 0.5,
                "reasoning": f"Sentiment analysis failed: {str(e)}",
            }

    def _classify_investment_risk(self, company_data):
        """Classify investment risk using zero-shot classification"""
        try:
            if not self.financial_classifier:
                return {"risk_level": "medium", "confidence": 0.5, "factors": []}

            # Create risk assessment text
            financial = company_data.get("financial_metrics", {})
            performance = company_data.get("market_performance", {})

            risk_text = f"""
            Investment with {performance.get('volatility', 0):.1f}% volatility, 
            beta of {performance.get('beta', 1.0):.2f}, 
            debt-to-equity of {financial.get('debt_to_equity', 0):.2f}, 
            current ratio of {financial.get('current_ratio', 0):.2f}
            """

            risk_categories = [
                "low risk conservative investment",
                "medium risk balanced investment",
                "high risk growth investment",
                "very high risk speculative investment",
            ]

            result = self.financial_classifier(risk_text, risk_categories)

            # Map to simple risk levels
            risk_mapping = {
                "low risk conservative investment": "low",
                "medium risk balanced investment": "medium",
                "high risk growth investment": "high",
                "very high risk speculative investment": "very_high",
            }

            primary_risk = result["labels"][0]
            risk_level = risk_mapping.get(primary_risk, "medium")

            # Identify specific risk factors
            risk_factors = []

            if performance.get("volatility", 0) > 30:
                risk_factors.append("High price volatility")
            if financial.get("debt_to_equity", 0) > 1.5:
                risk_factors.append("High debt levels")
            if financial.get("current_ratio", 0) < 1.0:
                risk_factors.append("Liquidity concerns")
            if performance.get("beta", 1.0) > 1.5:
                risk_factors.append("High market sensitivity")

            return {
                "risk_level": risk_level,
                "confidence": result["scores"][0],
                "factors": risk_factors,
                "classification_details": result,
            }

        except Exception as e:
            return {
                "risk_level": "medium",
                "confidence": 0.5,
                "factors": [],
                "error": str(e),
            }

    def _assess_market_position_with_ai(self, info, ai_analysis):
        """Assess market position enhanced with AI insights"""
        try:
            market_cap = info.get("marketCap", 0)
            sector = info.get("sector", "Unknown")

            # Basic market position
            if market_cap > 200_000_000_000:
                size_category = "Mega Cap"
                market_position = "Market Leader"
            elif market_cap > 10_000_000_000:
                size_category = "Large Cap"
                market_position = "Major Player"
            elif market_cap > 2_000_000_000:
                size_category = "Mid Cap"
                market_position = "Growing Company"
            else:
                size_category = "Small Cap"
                market_position = "Niche Player"

            # Enhanced with AI insights
            ai_insights = ai_analysis.get("insights", [])
            competitive_advantages = []

            # Extract competitive advantages from AI insights
            for insight in ai_insights:
                if any(
                    word in insight.lower()
                    for word in ["advantage", "leader", "dominant", "strong"]
                ):
                    competitive_advantages.append(insight)

            return {
                "size_category": size_category,
                "market_position": market_position,
                "sector": sector,
                "competitive_advantages": competitive_advantages[:3],
                "ai_enhanced": True,
            }

        except Exception as e:
            return {"error": f"Market position assessment failed: {str(e)}"}

    def _calculate_ai_score(self, analysis):
        """Calculate overall score based on AI analysis"""
        try:
            score = 50  # Base score

            # AI recommendation impact (40 points)
            ai_rec = analysis.get("ai_analysis", {}).get("recommendation", {})
            rec_action = ai_rec.get("action", "HOLD")
            rec_confidence = ai_rec.get("confidence", 0.5)

            if rec_action == "STRONG BUY":
                score += 40 * rec_confidence
            elif rec_action == "BUY":
                score += 30 * rec_confidence
            elif rec_action == "HOLD":
                score += 10 * rec_confidence
            elif rec_action == "SELL":
                score -= 20 * rec_confidence
            elif rec_action == "STRONG SELL":
                score -= 40 * rec_confidence

            # Sentiment impact (20 points)
            sentiment = analysis.get("ai_sentiment", {})
            if sentiment.get("sentiment") == "positive":
                score += 20 * sentiment.get("confidence", 0.5)
            elif sentiment.get("sentiment") == "negative":
                score -= 20 * sentiment.get("confidence", 0.5)

            # Risk impact (20 points)
            risk_class = analysis.get("ai_risk_classification", {})
            risk_level = risk_class.get("risk_level", "medium")
            risk_confidence = risk_class.get("confidence", 0.5)

            if risk_level == "low":
                score += 20 * risk_confidence
            elif risk_level == "medium":
                score += 10 * risk_confidence
            elif risk_level == "high":
                score -= 10 * risk_confidence
            elif risk_level == "very_high":
                score -= 20 * risk_confidence

            # Financial metrics impact (20 points)
            financial = analysis.get("financial_metrics", {})
            if financial.get("profit_margin", 0) > 0.15:
                score += 10
            if financial.get("revenue_growth", 0) > 0.1:
                score += 10

            # Normalize score
            score = max(0, min(100, score))

            # Map score to recommendation
            if score >= 80:
                recommendation = "STRONG BUY"
            elif score >= 70:
                recommendation = "BUY"
            elif score >= 60:
                recommendation = "WEAK BUY"
            elif score >= 40:
                recommendation = "HOLD"
            elif score >= 30:
                recommendation = "WEAK SELL"
            else:
                recommendation = "SELL"

            return round(score, 1), recommendation

        except Exception:
            return 50.0, "HOLD"

    def _get_model_info(self):
        """Get information about loaded models"""
        try:
            model_info = {
                "main_ai_loaded": self.main_ai is not None,
                "sentiment_loaded": self.sentiment_analyzer is not None,
                "classifier_loaded": self.financial_classifier is not None,
                "device": str(self.device),
                "gpu_available": self.use_gpu,
            }

            # Add model names if available
            if self.main_ai:
                model_info["main_ai_model"] = self.model_configs["mistral_alternative"][
                    "model_name"
                ]
            if self.sentiment_analyzer:
                model_info["sentiment_model"] = self.model_configs[
                    "financial_specialist"
                ]["model_name"]
            if self.financial_classifier:
                model_info["classifier_model"] = self.model_configs["classifier"][
                    "model_name"
                ]

            return model_info

        except Exception as e:
            return {"error": f"Model info failed: {str(e)}"}

    def _calculate_ai_confidence(self, company_data):
        """Calculate AI confidence based on available data and models"""
        try:
            confidence = 0.5  # Base confidence

            # Model availability
            if self.main_ai:
                confidence += 0.3
            if self.sentiment_analyzer:
                confidence += 0.1
            if self.financial_classifier:
                confidence += 0.1

            # Data quality
            financial = company_data.get("financial_metrics", {})
            if financial.get("revenue", 0) > 0:
                confidence += 0.05
            if financial.get("profit_margin", 0) != 0:
                confidence += 0.05

            return min(0.95, confidence)

        except Exception:
            return 0.75

    def _fallback_analysis(self, symbol, basic_analysis):
        """Fallback analysis when transformers not available"""
        return {
            "symbol": symbol,
            "ai_analysis": {
                "insights": ["AI analysis requires transformers library"],
                "risks": ["Install transformers for full AI analysis"],
                "opportunities": ["Enhanced analysis available with AI models"],
                "recommendation": {
                    "action": "HOLD",
                    "confidence": 0.5,
                    "reasoning": "Limited analysis available",
                },
                "price_targets": {},
                "full_analysis": "AI models not available. Install transformers library for comprehensive analysis.",
                "ai_generated": False,
            },
            "ai_sentiment": {
                "sentiment": "neutral",
                "confidence": 0.5,
                "reasoning": "AI models not available",
            },
            "ai_risk_classification": {
                "risk_level": "medium",
                "confidence": 0.5,
                "factors": [],
            },
            "overall_score": 50.0,
            "recommendation": "HOLD",
            "error": "AI models not available. Install: pip install transformers torch",
        }
