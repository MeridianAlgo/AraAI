"""
Data models for multi-currency support
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, Optional


class Currency(str, Enum):
    """Supported currencies"""

    USD = "USD"  # US Dollar
    EUR = "EUR"  # Euro
    GBP = "GBP"  # British Pound
    JPY = "JPY"  # Japanese Yen
    CNY = "CNY"  # Chinese Yuan
    AUD = "AUD"  # Australian Dollar
    CAD = "CAD"  # Canadian Dollar
    CHF = "CHF"  # Swiss Franc
    HKD = "HKD"  # Hong Kong Dollar
    SGD = "SGD"  # Singapore Dollar
    INR = "INR"  # Indian Rupee
    KRW = "KRW"  # South Korean Won


@dataclass
class CurrencyPreference:
    """User currency preference settings"""

    preferred_currency: Currency
    auto_convert: bool = True
    show_original: bool = True

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "preferred_currency": self.preferred_currency.value,
            "auto_convert": self.auto_convert,
            "show_original": self.show_original,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "CurrencyPreference":
        """Create from dictionary"""
        return cls(
            preferred_currency=Currency(data["preferred_currency"]),
            auto_convert=data.get("auto_convert", True),
            show_original=data.get("show_original", True),
        )


@dataclass
class ConversionResult:
    """Result of currency conversion"""

    original_amount: float
    original_currency: Currency
    converted_amount: float
    target_currency: Currency
    exchange_rate: float
    timestamp: datetime
    source: str

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "original_amount": self.original_amount,
            "original_currency": self.original_currency.value,
            "converted_amount": self.converted_amount,
            "target_currency": self.target_currency.value,
            "exchange_rate": self.exchange_rate,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
        }


@dataclass
class ExchangeRate:
    """Exchange rate data"""

    from_currency: Currency
    to_currency: Currency
    rate: float
    timestamp: datetime
    source: str
    bid: Optional[float] = None
    ask: Optional[float] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "from_currency": self.from_currency.value,
            "to_currency": self.to_currency.value,
            "rate": self.rate,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "bid": self.bid,
            "ask": self.ask,
        }


@dataclass
class CurrencyRiskMetrics:
    """Currency risk metrics for portfolios"""

    base_currency: Currency
    currency_exposures: Dict[Currency, float]  # Currency -> exposure amount
    currency_volatilities: Dict[Currency, float]  # Currency -> volatility
    currency_correlations: Dict[tuple, float]  # (Currency, Currency) -> correlation
    total_currency_risk: float  # VaR in base currency
    hedged_return: float  # Currency-hedged return
    unhedged_return: float  # Unhedged return
    currency_contribution: float  # Currency contribution to return

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "base_currency": self.base_currency.value,
            "currency_exposures": {k.value: v for k, v in self.currency_exposures.items()},
            "currency_volatilities": {k.value: v for k, v in self.currency_volatilities.items()},
            "currency_correlations": {
                f"{k[0].value}_{k[1].value}": v for k, v in self.currency_correlations.items()
            },
            "total_currency_risk": self.total_currency_risk,
            "hedged_return": self.hedged_return,
            "unhedged_return": self.unhedged_return,
            "currency_contribution": self.currency_contribution,
        }
