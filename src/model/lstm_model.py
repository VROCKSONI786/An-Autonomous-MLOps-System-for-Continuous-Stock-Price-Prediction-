import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import logging
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.data_preprocessor import DataPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LSTMStockModel:
    """Bidirectional LSTM model for stock price prediction"""
    
    def __init__(self, sequence_length=60, n_features=21):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None
        self.history = None
        # placeholder for configuration parameters supplied by external
        # orchestrators (ModelTrainer, streamlit UI, etc).  Historically
        # `ModelTrainer` did a ``lstm.model_config.update(...)`` after
        # creating an instance which blew up because no such attribute
        # existed.  Add it here so that callers can freely merge their
        # settings without having to worry about initialization order.
        self.model_config: dict = {}
        
    def build_model(self, lstm_units=128, dropout_rate=0.2):
        """Build Bidirectional LSTM model"""
        # allow callers to override via model_config if available
        lstm_units = self.model_config.get("lstm_units", lstm_units)
        dropout_rate = self.model_config.get("dropout_rate", dropout_rate)

        logger.info("Building Bidirectional LSTM model...")
        
        model = Sequential([
            Input(shape=(self.sequence_length, self.n_features)),
            
            # First Bidirectional LSTM layer
            Bidirectional(LSTM(lstm_units, return_sequences=True)),
            Dropout(dropout_rate),
            
            # Second Bidirectional LSTM layer
            Bidirectional(LSTM(lstm_units // 2, return_sequences=True)),
            Dropout(dropout_rate),
            
            # Third LSTM layer
            LSTM(lstm_units // 4, return_sequences=False),
            Dropout(dropout_rate),
            
            # Dense layers
            Dense(64, activation='relu'),
            Dropout(dropout_rate),
            
            Dense(32, activation='relu'),
            
            # Output layer
            Dense(1)
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mean_squared_error',
            metrics=['mae', 'mse']
        )
        
        self.model = model
        
        logger.info("Model built successfully")
        logger.info(f"\nModel Summary:")
        model.summary()
        
        return model
    
    def prepare_data(self, symbol='RELIANCE.NS', test_size=0.2, validation_split=0.1):
        """Prepare training, validation, and test data"""
        logger.info(f"Preparing data for {symbol}...")
        
        # Load preprocessed data
        preprocessor = DataPreprocessor()
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        processed_file = os.path.join(project_root, "data", "processed", "preprocessed_data.csv")
        
        data = pd.read_csv(processed_file, index_col=0, parse_dates=True)
        
        # Load scaler to get feature columns
        scaler_file = os.path.join(project_root, "models", "scaler.pkl")
        preprocessor.load_scaler(scaler_file)
        
        # Create sequences
        X, y = preprocessor.create_sequences(data, symbol, self.sequence_length)
        
        logger.info(f"Created {len(X)} sequences")
        logger.info(f"X shape: {X.shape}, y shape: {y.shape}")
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=validation_split, shuffle=False
        )
        
        logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs: int | None = None,
        batch_size: int | None = None,
        symbol: str | None = None,
        save_dir: str | None = None,
    ) -> dict:
        """Train the model.

        This method is used both by the standalone script in ``lstm_model.py``
        and the higher level ``ModelTrainer`` class.  The latter passes
        ``symbol``/``save_dir`` so it can persist the checkpoint and return a
        small summary dict that is later consumed for MLflow logging.  To
        remain backwards compatible we continue to accept the simple
        ``epochs``/``batch_size`` arguments and default them to values stored
        in ``self.model_config`` when not provided.
        """
        logger.info("Starting model training...")

        if self.model is None:
            self.build_model()

        # allow defaults to be driven from configuration if unspecified
        if epochs is None:
            epochs = int(self.model_config.get("epochs", 50))
        if batch_size is None:
            batch_size = int(self.model_config.get("batch_size", 32))
        
        # Callbacks
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        checkpoint_dir = os.path.join(project_root, "models", "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=os.path.join(checkpoint_dir, 'best_model.keras'),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001,
                verbose=1
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
        )

        logger.info("Training complete!")

        # summarize results for callers that expect a dict (e.g. ModelTrainer)
        hist = self.history.history
        best_val_loss = min(hist.get("val_loss", [float("inf")]))
        best_val_mae = min(hist.get("val_mae", [float("inf")]))
        epochs_run = len(hist.get("loss", []))

        model_path = None
        if symbol and save_dir:
            os.makedirs(save_dir, exist_ok=True)
            model_path = os.path.join(save_dir, f"{symbol}_lstm_model.keras")
            try:
                # ``save_model`` will create parent dir if needed
                self.save_model(model_path)
            except Exception:
                logger.warning(f"Failed to save model to {model_path}", exc_info=True)

        return {
            "history": hist,
            "best_val_loss": best_val_loss,
            "best_val_mae": best_val_mae,
            "epochs_run": epochs_run,
            "model_path": model_path,
        }
    
    def evaluate(self, X_test, y_test):
        """Evaluate model on test data"""
        logger.info("Evaluating model...")
        
        results = self.model.evaluate(X_test, y_test, verbose=0)
        
        logger.info(f"Test Loss: {results[0]:.6f}")
        logger.info(f"Test MAE: {results[1]:.6f}")
        logger.info(f"Test MSE: {results[2]:.6f}")
        
        return results
    
    def predict(self, X):
        """Make predictions"""
        predictions = self.model.predict(X, verbose=0)
        return predictions
    
    def _find_project_root(self) -> str:
        """Search upward for repository root by looking for common markers.

        Falls back to two-levels-up if no marker is found.
        """
        cur = os.path.dirname(os.path.abspath(__file__))
        search_limit = 6
        for _ in range(search_limit):
            # markers that indicate project root
            if os.path.isdir(os.path.join(cur, "config")) or os.path.exists(os.path.join(cur, "requirements.txt")):
                return cur
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            cur = parent
        # fallback (previous behavior)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.dirname(os.path.dirname(current_dir))

    def save_model(self, filepath=None):
        """Save the trained model to the canonical project `models/saved_models` by default."""
        try:
            if filepath is None:
                project_root = self._find_project_root()
                filepath = os.path.join(project_root, "models", "saved_models", "lstm_model.keras")

            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            self.model.save(filepath)
            logger.info(f"Model saved to {filepath}")

        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")

    def load_model(self, filepath=None):
        """Load a saved model from the canonical project `models/saved_models` by default."""
        try:
            if filepath is None:
                project_root = self._find_project_root()
                filepath = os.path.join(project_root, "models", "saved_models", "lstm_model.keras")

            self.model = keras.models.load_model(filepath)
            logger.info(f"Model loaded from {filepath}")

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")

if __name__ == "__main__":
    # Initialize model
    lstm_model = LSTMStockModel(sequence_length=60, n_features=21)
    
    # Build model
    lstm_model.build_model(lstm_units=128, dropout_rate=0.2)
    
    # Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test = lstm_model.prepare_data(
        symbol='RELIANCE.NS',
        test_size=0.2,
        validation_split=0.1
    )
    
    # Train model (returns a summary dict when called with symbol/save_dir)
    result = lstm_model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=50,
        batch_size=32
    )
    # ``result`` will be a Keras ``History`` object when using the simple
    # signature above; if more kwargs are supplied (see docstring) it will be
    # a dict with keys such as ``history``, ``best_val_loss`` etc.
    history = result if not isinstance(result, dict) else result.get("history")
    
    # Evaluate
    results = lstm_model.evaluate(X_test, y_test)
    
    # Save model
    lstm_model.save_model()
    
    print("\n=== Training Complete ===")
    print(f"Final Test Loss: {results[0]:.6f}")
    print(f"Final Test MAE: {results[1]:.6f}")