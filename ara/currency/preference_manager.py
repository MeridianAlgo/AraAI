"""
Currency preference management
"""

import json
from pathlib import Path
from typing import Optional
from ara.currency.models import Currency, CurrencyPreference
from ara.utils import get_logger

logger = get_logger(__name__)


class CurrencyPreferenceManager:
    """
    Manage user currency preferences
    Stores preferences in a JSON file
    """

    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize preference manager

        Args:
            config_dir: Directory for config files (default: ~/.ara)
        """
        if config_dir is None:
            config_dir = Path.home() / ".ara"

        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)

        self.preference_file = self.config_dir / "currency_preferences.json"
        self._preference: Optional[CurrencyPreference] = None

    def load_preference(self) -> CurrencyPreference:
        """
        Load currency preference from file

        Returns:
            CurrencyPreference object (defaults to USD if not found)
        """
        if self._preference is not None:
            return self._preference

        if not self.preference_file.exists():
            # Return default preference
            logger.info("No currency preference found, using default (USD)")
            self._preference = CurrencyPreference(
                preferred_currency=Currency.USD, auto_convert=True, show_original=True
            )
            return self._preference

        try:
            with open(self.preference_file, "r") as f:
                data = json.load(f)

            self._preference = CurrencyPreference.from_dict(data)

            logger.info(
                f"Loaded currency preference: {self._preference.preferred_currency.value}",
                preferred_currency=self._preference.preferred_currency.value,
            )

            return self._preference

        except Exception as e:
            logger.error(f"Failed to load currency preference: {e}", error=str(e))
            # Return default on error
            self._preference = CurrencyPreference(
                preferred_currency=Currency.USD, auto_convert=True, show_original=True
            )
            return self._preference

    def save_preference(self, preference: CurrencyPreference) -> None:
        """
        Save currency preference to file

        Args:
            preference: CurrencyPreference to save
        """
        try:
            with open(self.preference_file, "w") as f:
                json.dump(preference.to_dict(), f, indent=2)

            self._preference = preference

            logger.info(
                f"Saved currency preference: {preference.preferred_currency.value}",
                preferred_currency=preference.preferred_currency.value,
            )

        except Exception as e:
            logger.error(f"Failed to save currency preference: {e}", error=str(e))
            raise

    def set_preferred_currency(self, currency: Currency) -> None:
        """
        Set preferred currency

        Args:
            currency: Currency to set as preferred
        """
        preference = self.load_preference()
        preference.preferred_currency = currency
        self.save_preference(preference)

    def get_preferred_currency(self) -> Currency:
        """
        Get preferred currency

        Returns:
            Preferred currency
        """
        preference = self.load_preference()
        return preference.preferred_currency

    def set_auto_convert(self, auto_convert: bool) -> None:
        """
        Set auto-convert flag

        Args:
            auto_convert: Whether to auto-convert
        """
        preference = self.load_preference()
        preference.auto_convert = auto_convert
        self.save_preference(preference)

    def set_show_original(self, show_original: bool) -> None:
        """
        Set show-original flag

        Args:
            show_original: Whether to show original currency
        """
        preference = self.load_preference()
        preference.show_original = show_original
        self.save_preference(preference)

    def reset_to_default(self) -> None:
        """Reset preference to default (USD)"""
        default_preference = CurrencyPreference(
            preferred_currency=Currency.USD, auto_convert=True, show_original=True
        )
        self.save_preference(default_preference)
        logger.info("Reset currency preference to default (USD)")
