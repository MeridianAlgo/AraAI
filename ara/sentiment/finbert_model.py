"""
FinBERT Model Wrapper

Provides sentiment analysis using the FinBERT model (BERT fine-tuned on financial text).
"""

import logging

logger = logging.getLogger(__name__)


class FinBERTModel:
    """
    Wrapper for FinBERT sentiment analysis model.

    FinBERT is a BERT model fine-tuned on financial text for sentiment analysis.
    Falls back to VADER sentiment if FinBERT is not available.
    """

    def __init__(self, model_name: str = "ProsusAI/finbert"):
        """
        Initialize FinBERT model.

        Args:
            model_name: HuggingFace model name
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.use_finbert = False
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize the sentiment model"""
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            logger.info(f"Loading FinBERT model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.eval()
            self.use_finbert = True
            logger.info("FinBERT model loaded successfully")

        except ImportError:
            logger.warning(
                "transformers library not installed. Install with: pip install transformers torch"
            )
            self._initialize_vader()
        except Exception as e:
            logger.warning(f"Failed to load FinBERT model: {e}. Falling back to VADER.")
            self._initialize_vader()

    def _initialize_vader(self) -> None:
        """Initialize VADER sentiment analyzer as fallback"""
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

            self.vader = SentimentIntensityAnalyzer()
            self.use_finbert = False
            logger.info("Using VADER sentiment analyzer")
        except ImportError:
            logger.warning(
                "vaderSentiment library not installed. Install with: pip install vaderSentiment"
            )
            self.vader = None

    def analyze_text(self, text: str) -> float:
        """
        Analyze sentiment of text.

        Args:
            text: Text to analyze

        Returns:
            Sentiment score from -1 (bearish) to +1 (bullish)
        """
        if not text or not text.strip():
            return 0.0

        if self.use_finbert and self.model is not None:
            return self._analyze_with_finbert(text)
        elif self.vader is not None:
            return self._analyze_with_vader(text)
        else:
            logger.warning("No sentiment model available, returning neutral")
            return 0.0

    def _analyze_with_finbert(self, text: str) -> float:
        """
        Analyze sentiment using FinBERT.

        Args:
            text: Text to analyze

        Returns:
            Sentiment score from -1 to +1
        """
        try:
            import torch

            # Tokenize
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512, padding=True
            )

            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

            # FinBERT outputs: [negative, neutral, positive]
            probs = predictions[0].numpy()

            # Convert to -1 to +1 scale
            # negative: -1, neutral: 0, positive: +1
            sentiment = float(probs[2] - probs[0])

            return sentiment

        except Exception as e:
            logger.error(f"Error in FinBERT analysis: {e}")
            return 0.0

    def _analyze_with_vader(self, text: str) -> float:
        """
        Analyze sentiment using VADER.

        Args:
            text: Text to analyze

        Returns:
            Sentiment score from -1 to +1
        """
        try:
            scores = self.vader.polarity_scores(text)
            # VADER compound score is already -1 to +1
            return float(scores["compound"])
        except Exception as e:
            logger.error(f"Error in VADER analysis: {e}")
            return 0.0

    def analyze_batch(self, texts: list) -> list:
        """
        Analyze sentiment for multiple texts.

        Args:
            texts: List of texts to analyze

        Returns:
            List of sentiment scores
        """
        if self.use_finbert and self.model is not None:
            return self._analyze_batch_finbert(texts)
        else:
            return [self.analyze_text(text) for text in texts]

    def _analyze_batch_finbert(self, texts: list) -> list:
        """
        Batch analyze with FinBERT for efficiency.

        Args:
            texts: List of texts

        Returns:
            List of sentiment scores
        """
        try:
            import torch

            # Tokenize all texts
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            )

            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

            # Convert to sentiment scores
            probs = predictions.numpy()
            sentiments = [float(p[2] - p[0]) for p in probs]

            return sentiments

        except Exception as e:
            logger.error(f"Error in batch FinBERT analysis: {e}")
            return [self.analyze_text(text) for text in texts]
