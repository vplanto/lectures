"""
Модуль для предиктивного тестування з використанням LSTM.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')


class LSTMPredictor:
    """
    LSTM модель для прогнозування та виявлення аномалій через помилку реконструкції.
    """
    
    def __init__(self, window_size: int = 60, hidden_size: int = 50, 
                 num_layers: int = 2, learning_rate: float = 0.001,
                 device: Optional[str] = None):
        """
        Ініціалізація LSTM моделі.
        
        Parameters:
        -----------
        window_size : int
            Розмір вікна для sliding window
        hidden_size : int
            Розмір прихованого шару LSTM
        num_layers : int
            Кількість шарів LSTM
        learning_rate : float
            Швидкість навчання
        device : str, optional
            Пристрій для обчислень ('cpu' або 'cuda')
        """
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Створення послідовностей для навчання (sliding window).
        
        Parameters:
        -----------
        data : np.ndarray
            Вхідні дані
        
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            (X, y) - вхідні послідовності та цільові значення
        """
        X, y = [], []
        for i in range(len(data) - self.window_size):
            X.append(data[i:i + self.window_size])
            y.append(data[i + self.window_size])
        return np.array(X), np.array(y)
    
    def _build_model(self, input_size: int = 1) -> nn.Module:
        """
        Побудова LSTM моделі.
        
        Parameters:
        -----------
        input_size : int
            Розмір вхідного шару (за замовчуванням 1 для уніваріантного ряду)
        
        Returns:
        --------
        nn.Module
            LSTM модель
        """
        class LSTMModel(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers):
                super(LSTMModel, self).__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                                   batch_first=True)
                self.fc = nn.Linear(hidden_size, 1)
            
            def forward(self, x):
                # x shape: (batch, seq_len, input_size)
                lstm_out, _ = self.lstm(x)
                # Беремо останній вихід
                last_output = lstm_out[:, -1, :]
                output = self.fc(last_output)
                return output
        
        return LSTMModel(input_size, self.hidden_size, self.num_layers)
    
    def train(self, series: pd.Series, epochs: int = 50, batch_size: int = 32,
              train_ratio: float = 0.8) -> List[float]:
        """
        Навчання LSTM моделі.
        
        Parameters:
        -----------
        series : pd.Series
            Часовий ряд для навчання
        epochs : int
            Кількість епох навчання
        batch_size : int
            Розмір батчу
        train_ratio : float
            Частка даних для навчання
        
        Returns:
        --------
        List[float]
            Історія втрат під час навчання
        """
        # Нормалізація даних
        data = series.values.reshape(-1, 1)
        data_scaled = self.scaler.fit_transform(data)
        
        # Створення послідовностей
        X, y = self._create_sequences(data_scaled.flatten())
        
        # Розділення на train/test
        split_idx = int(len(X) * train_ratio)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Перетворення в тензори
        X_train = torch.FloatTensor(X_train).unsqueeze(-1).to(self.device)
        y_train = torch.FloatTensor(y_train).unsqueeze(-1).to(self.device)
        X_test = torch.FloatTensor(X_test).unsqueeze(-1).to(self.device)
        y_test = torch.FloatTensor(y_test).unsqueeze(-1).to(self.device)
        
        # Побудова моделі
        self.model = self._build_model(input_size=1).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Навчання
        history = []
        self.model.train()
        
        for epoch in range(epochs):
            # Batch training
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i + batch_size]
                batch_y = y_train[i:i + batch_size]
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            
            # Обчислення втрат на тестовому наборі
            with torch.no_grad():
                test_outputs = self.model(X_test)
                test_loss = criterion(test_outputs, y_test).item()
                history.append(test_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {test_loss:.6f}")
        
        self.is_trained = True
        return history
    
    def predict(self, series: pd.Series) -> np.ndarray:
        """
        Прогнозування для всього ряду.
        
        Parameters:
        -----------
        series : pd.Series
            Часовий ряд для прогнозування
        
        Returns:
        --------
        np.ndarray
            Прогнозовані значення
        """
        if not self.is_trained:
            raise ValueError("Модель не навчена. Спочатку викличте train()")
        
        # Нормалізація
        data = series.values.reshape(-1, 1)
        data_scaled = self.scaler.transform(data)
        
        # Створення послідовностей
        X, _ = self._create_sequences(data_scaled.flatten())
        
        # Прогнозування
        X_tensor = torch.FloatTensor(X).unsqueeze(-1).to(self.device)
        self.model.eval()
        
        predictions = []
        with torch.no_grad():
            for i in range(len(X_tensor)):
                pred = self.model(X_tensor[i:i+1])
                predictions.append(pred.cpu().numpy()[0, 0])
        
        # Денормалізація
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions).flatten()
        
        return predictions
    
    def compute_reconstruction_error(self, series: pd.Series) -> Tuple[np.ndarray, dict]:
        """
        Обчислення помилки реконструкції.
        
        Parameters:
        -----------
        series : pd.Series
            Часовий ряд для аналізу
        
        Returns:
        --------
        Tuple[np.ndarray, dict]
            (errors, stats) - помилки реконструкції та статистика
        """
        if not self.is_trained:
            raise ValueError("Модель не навчена. Спочатку викличте train()")
        
        # Прогнозування
        predictions = self.predict(series)
        
        # Обчислення помилок (тільки для точок, де є прогноз)
        actual = series.values[self.window_size:]
        errors = np.abs(actual - predictions)
        
        # Статистика
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        threshold = mean_error + 3 * std_error
        
        stats = {
            'mean': mean_error,
            'std': std_error,
            'threshold': threshold,
            'anomalies': np.sum(errors > threshold)
        }
        
        return errors, stats


