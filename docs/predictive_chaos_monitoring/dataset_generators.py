"""
Модуль для генерації синтетичних датасетів для тестування системи предиктивного моніторингу.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional


class ControlledChaosGenerator:
    """
    Генератор датасету "Controlled Chaos" на основі логістичного відображення.
    x_{n+1} = r * x_n * (1 - x_n)
    """
    
    def __init__(self, r: float = 3.9, noise_std: float = 0.01, 
                 random_seed: Optional[int] = None):
        """
        Ініціалізація генератора.
        
        Parameters:
        -----------
        r : float
            Параметр логістичного відображення (r=3.9 для хаотичного режиму)
        noise_std : float
            Стандартне відхилення білого шуму
        random_seed : int, optional
            Seed для відтворюваності результатів
        """
        self.r = r
        self.noise_std = noise_std
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def generate(self, n_points: int = 1000, x0: float = 0.5,
                 anomaly_indices: Optional[List[int]] = None,
                 anomaly_type: str = 'spike') -> pd.Series:
        """
        Генерація хаотичного ряду з аномаліями.
        
        Parameters:
        -----------
        n_points : int
            Кількість точок у ряду
        x0 : float
            Початкове значення (0 < x0 < 1)
        anomaly_indices : List[int], optional
            Індекси, де потрібно інжектувати аномалії
        anomaly_type : str
            Тип аномалії: 'spike' (збільшення значення) або 'drift' (дрейф параметра r)
        
        Returns:
        --------
        pd.Series
            Згенерований часовий ряд
        """
        if anomaly_indices is None:
            # Автоматично генеруємо 3-5 випадкових аномалій
            n_anomalies = np.random.randint(3, 6)
            anomaly_indices = sorted(np.random.choice(
                range(100, n_points - 100), 
                size=n_anomalies, 
                replace=False
            ))
        
        series = np.zeros(n_points)
        series[0] = x0
        current_r = self.r
        drift_active_until = -1  # Трекінг активного дрейфу
        
        for i in range(1, n_points):
            # Повернення параметра r до нормального значення після дрейфу
            if anomaly_type == 'drift' and i > drift_active_until:
                current_r = self.r
            
            # Логістичне відображення
            series[i] = current_r * series[i-1] * (1 - series[i-1])
            
            # Додавання білого шуму
            series[i] += np.random.normal(0, self.noise_std)
            
            # Обмеження значення в межах [0, 1]
            series[i] = np.clip(series[i], 0, 1)
            
            # Інжекція аномалій
            if i in anomaly_indices:
                if anomaly_type == 'spike':
                    # Різке збільшення значення
                    series[i] = series[i] * 1.5
                    series[i] = np.clip(series[i], 0, 1)
                elif anomaly_type == 'drift':
                    # Дрейф параметра r на короткий період (5 точок)
                    current_r = self.r * 1.2
                    drift_active_until = i + 5
        
        return pd.Series(series, name='ControlledChaos')


class PureRandomGenerator:
    """
    Генератор датасету "Pure Random" на основі випадкового блукання.
    X_t = X_{t-1} + ε_t, де ε_t ~ i.i.d.
    """
    
    def __init__(self, noise_std: float = 0.1, random_seed: Optional[int] = None):
        """
        Ініціалізація генератора.
        
        Parameters:
        -----------
        noise_std : float
            Стандартне відхилення шуму ε_t
        random_seed : int, optional
            Seed для відтворюваності результатів
        """
        self.noise_std = noise_std
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def generate(self, n_points: int = 1000, x0: float = 0.0) -> pd.Series:
        """
        Генерація випадкового блукання.
        
        Parameters:
        -----------
        n_points : int
            Кількість точок у ряду
        x0 : float
            Початкове значення
        
        Returns:
        --------
        pd.Series
            Згенерований часовий ряд
        """
        # Генерація i.i.d. шуму
        noise = np.random.normal(0, self.noise_std, n_points)
        
        # Випадкове блукання
        series = np.zeros(n_points)
        series[0] = x0
        
        for i in range(1, n_points):
            series[i] = series[i-1] + noise[i]
        
        return pd.Series(series, name='PureRandom')

