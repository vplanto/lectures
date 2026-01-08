"""
Модуль для математичного аналізу часових рядів.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class TimeSeriesAnalyzer:
    """
    Клас для математичного аналізу часових рядів.
    """
    
    def __init__(self):
        """Ініціалізація аналізатора."""
        pass
    
    def compute_hurst_exponent(self, series: pd.Series, max_lag: Optional[int] = None) -> float:
        """
        Обчислення показника Херста (H) за допомогою R/S аналізу.
        
        Parameters:
        -----------
        series : pd.Series
            Часовий ряд для аналізу
        max_lag : int, optional
            Максимальний лаг для аналізу (за замовчуванням len(series)//4)
        
        Returns:
        --------
        float
            Показник Херста H
        """
        n = len(series)
        if max_lag is None:
            max_lag = n // 4
        
        # Перетворення в numpy array
        values = series.values
        
        # R/S аналіз
        lags = []
        rs_values = []
        
        for lag in range(10, min(max_lag, n // 2), 5):
            # Розбиття на підпослідовності довжиною lag
            n_subsets = n // lag
            if n_subsets < 2:
                continue
            
            rs_subset = []
            
            for i in range(n_subsets):
                subset = values[i * lag:(i + 1) * lag]
                
                if len(subset) < 2:
                    continue
                
                # Обчислення середнього
                mean_subset = np.mean(subset)
                
                # Відхилення від середнього
                deviations = subset - mean_subset
                
                # Накопичені відхилення
                cumsum_deviations = np.cumsum(deviations)
                
                # Range (R)
                R = np.max(cumsum_deviations) - np.min(cumsum_deviations)
                
                # Standard deviation (S)
                S = np.std(subset)
                
                if S > 1e-10:
                    rs_subset.append(R / S)
            
            if rs_subset:
                lags.append(lag)
                rs_values.append(np.mean(rs_subset))
        
        if len(lags) < 2:
            return 0.5  # За замовчуванням для випадкового процесу
        
        # Логарифмічна регресія: log(R/S) = H * log(lag) + c
        log_lags = np.log(lags)
        log_rs = np.log(rs_values)
        
        # Лінійна регресія
        H = np.polyfit(log_lags, log_rs, 1)[0]
        
        return H
    
    def adf_test(self, series: pd.Series) -> Dict[str, float]:
        """
        Augmented Dickey-Fuller тест на стаціонарність.
        
        Parameters:
        -----------
        series : pd.Series
            Часовий ряд для тестування
        
        Returns:
        --------
        Dict[str, float]
            Словник з результатами тесту (statistic, pvalue, критичні значення)
        """
        result = adfuller(series.values, autolag='AIC')
        
        return {
            'adf_statistic': result[0],
            'pvalue': result[1],
            'critical_values': result[4],
            'is_stationary': result[1] < 0.05
        }
    
    def compute_acf(self, series: pd.Series, nlags: int = 40) -> Tuple[np.ndarray, np.ndarray]:
        """
        Обчислення автокореляційної функції (ACF).
        
        Parameters:
        -----------
        series : pd.Series
            Часовий ряд для аналізу
        nlags : int
            Кількість лагів для обчислення
        
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            (lags, acf_values) - лаги та значення ACF
        """
        acf_values = acf(series.values, nlags=nlags, fft=True)
        lags = np.arange(len(acf_values))
        
        return lags, acf_values
    
    def analyze(self, series: pd.Series, series_name: str = "Series") -> Dict:
        """
        Повний аналіз часового ряду.
        
        Parameters:
        -----------
        series : pd.Series
            Часовий ряд для аналізу
        series_name : str
            Назва ряду для виведення
        
        Returns:
        --------
        Dict
            Словник з результатами аналізу
        """
        print(f"\n{'='*60}")
        print(f"Аналіз ряду: {series_name}")
        print(f"{'='*60}")
        
        # Показник Херста
        H = self.compute_hurst_exponent(series)
        print(f"\nПоказник Херста (H): {H:.4f}")
        if H > 0.5:
            print(f"  → Ряд має пам'ять (персистентність), H > 0.5")
        elif H < 0.5:
            print(f"  → Ряд має антиперсистентність, H < 0.5")
        else:
            print(f"  → Ряд близький до випадкового блукання, H ≈ 0.5")
        
        # ADF тест
        adf_results = self.adf_test(series)
        print(f"\nADF тест на стаціонарність:")
        print(f"  ADF статистика: {adf_results['adf_statistic']:.4f}")
        print(f"  p-value: {adf_results['pvalue']:.6f}")
        print(f"  Стаціонарний: {'Так' if adf_results['is_stationary'] else 'Ні'}")
        print(f"  Критичні значення:")
        for key, value in adf_results['critical_values'].items():
            print(f"    {key}: {value:.4f}")
        
        # ACF
        lags, acf_values = self.compute_acf(series)
        significant_lags = np.sum(np.abs(acf_values[1:]) > 0.1)  # Поріг значущості
        print(f"\nАвтокореляційна функція (ACF):")
        print(f"  Значущих кореляцій (|ACF| > 0.1): {significant_lags}")
        if significant_lags > 5:
            print(f"  → Ряд має структуру (наявні автокореляції)")
        else:
            print(f"  → Ряд близький до білого шуму (відсутність структури)")
        
        return {
            'hurst_exponent': H,
            'adf_test': adf_results,
            'acf': (lags, acf_values),
            'significant_acf_lags': significant_lags
        }

