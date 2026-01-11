import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
import os
from datetime import datetime
import matplotlib.pyplot as plt


class StockDataset(Dataset):
    """PyTorch Dataset for stock price prediction"""

    def __init__(
        self, data: np.ndarray, sequence_length: int = 60, prediction_window: int = 1
    ):
        """
        Args:
            data: Normalized stock data (n_samples, n_features)
            sequence_length: Number of time steps to look back
            prediction_window: Number of days to predict ahead
        """
        self.sequence_length = sequence_length
        self.prediction_window = prediction_window
        self.x, self.y = self.prepare_sequences(data)

    def prepare_sequences(self, data: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert time series data into sequences for training"""
        x, y = [], []
        for i in range(len(data) - self.sequence_length - self.prediction_window + 1):
            x.append(data[i : (i + self.sequence_length)])
            y.append(
                data[
                    i
                    + self.sequence_length : i
                    + self.sequence_length
                    + self.prediction_window,
                    0,
                ]
            )  # Predict only the close price
        return torch.FloatTensor(np.array(x)), torch.FloatTensor(np.array(y))

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class StockPredictor(nn.Module):
    """Transformer-based stock price predictor"""

    def __init__(
        self,
        input_dim: int = 5,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        """
        Args:
            input_dim: Number of features per time step (e.g., OHLCV = 5)
            d_model: Dimension of the model
            nhead: Number of attention heads
            num_layers: Number of transformer encoder layers
            dropout: Dropout rate
        """
        super(StockPredictor, self).__init__()

        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),  # Predict next day's close price
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_dim)
        x = self.embedding(x)  # (batch_size, seq_len, d_model)
        x = self.transformer_encoder(x)  # (batch_size, seq_len, d_model)
        x = self.decoder(x[:, -1, :])  # Use last time step's output for prediction
        return x


class StockPredictorTrainer:
    """Handles training and evaluation of the StockPredictor model"""

    def __init__(
        self,
        ticker: str = "AAPL",
        start_date: str = "2020-01-01",
        end_date: str = datetime.now().strftime("%Y-%m-%d"),
        sequence_length: int = 60,
        prediction_window: int = 1,
        batch_size: int = 32,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
    ):
        """
        Initialize the stock predictor trainer

        Args:
            ticker: Stock ticker symbol
            start_date: Start date for data in 'YYYY-MM-DD' format
            end_date: End date for data in 'YYYY-MM-DD' format
            sequence_length: Number of days to look back for prediction
            prediction_window: Number of days to predict ahead
            batch_size: Batch size for training
            train_ratio: Ratio of data to use for training
            val_ratio: Ratio of data to use for validation
            test_ratio: Ratio of data to use for testing
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.sequence_length = sequence_length
        self.prediction_window = prediction_window
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )

        # Initialize data loaders
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.model = None

    def download_data(self) -> pd.DataFrame:
        """Download stock data from Yahoo Finance"""
        print(
            f"Downloading {self.ticker} data from {self.start_date} to {self.end_date}..."
        )
        data = yf.download(self.ticker, start=self.start_date, end=self.end_date)

        # Calculate additional technical indicators
        data = self._add_technical_indicators(data)
        return data

    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the data"""
        # Simple Moving Averages
        data["SMA_20"] = data["Close"].rolling(window=20).mean()
        data["SMA_50"] = data["Close"].rolling(window=50).mean()

        # RSI
        delta = data["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data["RSI"] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = data["Close"].ewm(span=12, adjust=False).mean()
        exp2 = data["Close"].ewm(span=26, adjust=False).mean()
        data["MACD"] = exp1 - exp2
        data["Signal_Line"] = data["MACD"].ewm(span=9, adjust=False).mean()

        # Bollinger Bands
        data["BB_upper"] = (
            data["Close"].rolling(window=20).mean()
            + 2 * data["Close"].rolling(window=20).std()
        )
        data["BB_lower"] = (
            data["Close"].rolling(window=20).mean()
            - 2 * data["Close"].rolling(window=20).std()
        )

        # Drop NaN values from technical indicators
        data = data.dropna()

        return data

    def prepare_data(
        self, data: pd.DataFrame
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare data for training, validation, and testing"""
        # Select features (OHLCV + technical indicators)
        feature_columns = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "SMA_20",
            "SMA_50",
            "RSI",
            "MACD",
            "Signal_Line",
            "BB_upper",
            "BB_lower",
        ]

        # Ensure all columns exist in the data
        feature_columns = [col for col in feature_columns if col in data.columns]
        data = data[feature_columns]

        # Normalize the data
        scaled_data = self.scaler.fit_transform(data)

        # Split into train, validation, and test sets
        train_size = int(len(scaled_data) * self.train_ratio)
        val_size = int(len(scaled_data) * self.val_ratio)

        train_data = scaled_data[:train_size]
        val_data = scaled_data[train_size : train_size + val_size]
        test_data = scaled_data[train_size + val_size :]

        # Create datasets
        train_dataset = StockDataset(
            train_data, self.sequence_length, self.prediction_window
        )
        val_dataset = StockDataset(
            val_data, self.sequence_length, self.prediction_window
        )
        test_dataset = StockDataset(
            test_data, self.sequence_length, self.prediction_window
        )

        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )

        return self.train_loader, self.val_loader, self.test_loader

    def train(
        self,
        num_epochs: int = 100,
        learning_rate: float = 1e-4,
        patience: int = 10,
        model_path: str = None,
    ) -> None:
        """Train the model"""
        if self.train_loader is None or self.val_loader is None:
            raise ValueError("Data loaders not initialized. Call prepare_data() first.")

        # Initialize model
        input_dim = next(iter(self.train_loader))[0].shape[
            2
        ]  # Get input dimension from data
        self.model = StockPredictor(input_dim=input_dim).to(self.device)

        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", patience=patience // 2, factor=0.5
        )

        # Early stopping
        best_val_loss = float("inf")
        patience_counter = 0

        # Training loop
        train_losses = []
        val_losses = []

        for epoch in range(num_epochs):
            # Training
            self.model.train()
            train_loss = 0.0

            for batch_x, batch_y in self.train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                # Forward pass
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)

                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0
                )  # Gradient clipping
                optimizer.step()

                train_loss += loss.item() * batch_x.size(0)

            # Calculate average training loss
            train_loss = train_loss / len(self.train_loader.dataset)
            train_losses.append(train_loss)

            # Validation
            val_loss = self.evaluate(self.val_loader, criterion)
            val_losses.append(val_loss)

            # Update learning rate
            scheduler.step(val_loss)

            # Print progress
            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
                )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save the best model
                if model_path:
                    self.save_model(model_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # Load the best model
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

        return train_losses, val_losses

    def evaluate(self, data_loader: DataLoader, criterion) -> float:
        """Evaluate the model on the given data loader"""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                total_loss += loss.item() * batch_x.size(0)

        return total_loss / len(data_loader.dataset)

    def predict(self, data_loader: DataLoader) -> np.ndarray:
        """Make predictions using the trained model"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        self.model.eval()
        predictions = []

        with torch.no_grad():
            for batch_x, _ in data_loader:
                batch_x = batch_x.to(self.device)
                outputs = self.model(batch_x).cpu().numpy()
                predictions.append(outputs)

        return np.concatenate(predictions, axis=0)

    def save_model(self, filepath: str) -> None:
        """Save the model to a file"""
        if self.model is None:
            raise ValueError("Model not trained. Nothing to save.")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Save model state and scaler
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "scaler": self.scaler,
                "sequence_length": self.sequence_length,
                "prediction_window": self.prediction_window,
                "ticker": self.ticker,
            },
            filepath,
        )

        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load a trained model from a file"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        checkpoint = torch.load(filepath, map_location=self.device)

        # Initialize model with correct input dimension
        input_dim = (
            next(iter(self.train_loader))[0].shape[2] if self.train_loader else 5
        )  # Default to 5 if no data loaded
        self.model = StockPredictor(input_dim=input_dim).to(self.device)

        # Load model state
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.scaler = checkpoint["scaler"]
        self.sequence_length = checkpoint.get("sequence_length", self.sequence_length)
        self.prediction_window = checkpoint.get(
            "prediction_window", self.prediction_window
        )
        self.ticker = checkpoint.get("ticker", self.ticker)

        print(f"Model loaded from {filepath}")

    def plot_predictions(
        self, actual: np.ndarray, predicted: np.ndarray, title: str = ""
    ) -> None:
        """Plot actual vs predicted values"""
        plt.figure(figsize=(12, 6))
        plt.plot(actual, label="Actual")
        plt.plot(predicted, label="Predicted")
        plt.title(f"{self.ticker} {title} - Actual vs Predicted")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)
        plt.show()


def train_and_evaluate(
    ticker: str = "AAPL",
    start_date: str = "2020-01-01",
    end_date: str = datetime.now().strftime("%Y-%m-%d"),
    sequence_length: int = 60,
    prediction_window: int = 1,
    batch_size: int = 32,
    num_epochs: int = 100,
    learning_rate: float = 1e-4,
    model_dir: str = "saved_models",
) -> None:
    """Train and evaluate the stock predictor model"""
    # Create model directory
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{ticker}_stock_predictor.pt")

    # Initialize trainer
    trainer = StockPredictorTrainer(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        sequence_length=sequence_length,
        prediction_window=prediction_window,
        batch_size=batch_size,
    )

    # Download and prepare data
    data = trainer.download_data()
    train_loader, val_loader, test_loader = trainer.prepare_data(data)

    # Train the model
    print(f"Training model for {ticker}...")
    train_losses, val_losses = trainer.train(
        num_epochs=num_epochs, learning_rate=learning_rate, model_path=model_path
    )

    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title(f"Training and Validation Loss - {ticker}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Evaluate on test set
    test_loss = trainer.evaluate(test_loader, nn.MSELoss())
    print(f"Test Loss: {test_loss:.6f}")

    # Make predictions
    test_predictions = trainer.predict(test_loader)

    # Get actual values (only close prices)
    actual_values = []
    for _, batch_y in test_loader:
        actual_values.append(batch_y.numpy())
    actual_values = np.concatenate(actual_values, axis=0)

    # Plot predictions
    trainer.plot_predictions(actual_values, test_predictions, "Test Set")

    return trainer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a stock price predictor model")
    parser.add_argument(
        "--ticker", type=str, default="AAPL", help="Stock ticker symbol"
    )
    parser.add_argument(
        "--start_date", type=str, default="2020-01-01", help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end_date",
        type=str,
        default=datetime.now().strftime("%Y-%m-%d"),
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=60,
        help="Number of days to look back for prediction",
    )
    parser.add_argument(
        "--prediction_window",
        type=int,
        default=1,
        help="Number of days to predict ahead",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="saved_models",
        help="Directory to save the trained model",
    )

    args = parser.parse_args()

    # Train and evaluate the model
    trainer = train_and_evaluate(
        ticker=args.ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        sequence_length=args.sequence_length,
        prediction_window=args.prediction_window,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        model_dir=args.model_dir,
    )
