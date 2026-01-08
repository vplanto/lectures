"""
Семінар 1: Математичне моделювання та пошук інваріантів

Реалізує механізм виявлення стійких зв'язків між метриками через PCA та лінійну регресію.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Invariant:
    """Клас для зберігання інформації про інваріант."""
    x_component_idx: int  # Індекс компоненти x
    y_component_idx: int  # Індекс компоненти y
    coefficient_a: float  # Коефіцієнт a в рівнянні y = ax + b
    intercept_b: float   # Коефіцієнт b в рівнянні y = ax + b
    correlation: float   # Кореляція між x та y
    max_residual: float  # Максимальний залишок на тренувальному наборі
    pca_x: PCA          # PCA трансформація для x
    pca_y: PCA          # PCA трансформація для y
    scaler_x: StandardScaler  # Скалер для x
    scaler_y: StandardScaler  # Скалер для y
    feature_names_x: List[str]  # Назви ознак для x
    feature_names_y: List[str]  # Назви ознак для y


class InvariantDiscovery:
    """
    Клас для виявлення статистичних інваріантів між групами метрик.
    
    Інваріанти - це стійкі лінійні залежності між групами метрик,
    які описують "нормальну" поведінку системи.
    """
    
    def __init__(
        self,
        variance_threshold: float = 0.95,
        correlation_threshold: float = 0.93,
        overhead: float = 0.1
    ):
        """
        Ініціалізація класу.
        
        Parameters:
        -----------
        variance_threshold : float
            Мінімальна частка варіативності, яку повинні пояснювати PCA компоненти (за замовчуванням 0.95)
        correlation_threshold : float
            Мінімальна абсолютна кореляція для валідації інваріанта (за замовчуванням 0.93)
        overhead : float
            Додатковий відсоток для порогу аномалій (за замовчуванням 0.1 = 10%)
        """
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.overhead = overhead
        self.invariants: List[Invariant] = []
        self.feature_groups: Dict[str, List[str]] = {}
        
    def add_feature_group(self, group_name: str, features: List[str]):
        """
        Додати групу ознак для аналізу.
        
        Parameters:
        -----------
        group_name : str
            Назва групи (наприклад, "cpu_metrics", "memory_metrics")
        features : List[str]
            Список назв колонок у DataFrame, що належать до цієї групи
        """
        self.feature_groups[group_name] = features
    
    def _apply_pca(
        self,
        data: pd.DataFrame,
        features: List[str],
        variance_threshold: float
    ) -> Tuple[PCA, StandardScaler, np.ndarray, int]:
        """
        Застосувати PCA до групи ознак.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Дані для аналізу
        features : List[str]
            Список ознак для PCA
        variance_threshold : float
            Мінімальна частка варіативності
            
        Returns:
        --------
        pca : PCA
            Навчений PCA об'єкт
        scaler : StandardScaler
            Навчений скалер
        transformed_data : np.ndarray
            Трансформовані дані
        n_components : int
            Кількість компонент, що пояснюють >= variance_threshold варіативності
        """
        # Вибірка та стандартизація даних
        X = data[features].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Застосування PCA
        pca = PCA()
        X_transformed = pca.fit_transform(X_scaled)
        
        # Визначення кількості компонент для >= variance_threshold
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
        
        # Переобучення PCA з оптимальною кількістю компонент
        if n_components < len(features):
            pca = PCA(n_components=n_components)
            X_transformed = pca.fit_transform(X_scaled)
        
        return pca, scaler, X_transformed, n_components
    
    def _find_linear_relationship(
        self,
        x: np.ndarray,
        y: np.ndarray
    ) -> Tuple[float, float, float, float]:
        """
        Знайти лінійну залежність y = ax + b між двома компонентами.
        
        Parameters:
        -----------
        x : np.ndarray
            Незалежна змінна (1D масив)
        y : np.ndarray
            Залежна змінна (1D масив)
            
        Returns:
        --------
        a : float
            Коефіцієнт нахилу
        b : float
            Перетин
        correlation : float
            Кореляція між x та y
        max_residual : float
            Максимальний залишок
        """
        # Лінійна регресія
        X = x.reshape(-1, 1)
        reg = LinearRegression()
        reg.fit(X, y)
        
        a = reg.coef_[0]
        b = reg.intercept_
        
        # Обчислення кореляції
        correlation = np.corrcoef(x, y)[0, 1]
        
        # Обчислення залишків
        y_pred = reg.predict(X)
        residuals = np.abs(y - y_pred)
        max_residual = np.max(residuals)
        
        return a, b, correlation, max_residual
    
    def fit(self, data: pd.DataFrame) -> List[Invariant]:
        """
        Навчити модель на даних та знайти інваріанти.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Тренувальні дані (DataFrame з метриками)
            
        Returns:
        --------
        invariants : List[Invariant]
            Список знайдених інваріантів
        """
        if not self.feature_groups:
            raise ValueError("Спочатку додайте групи ознак через add_feature_group()")
        
        self.invariants = []
        group_names = list(self.feature_groups.keys())
        
        # Застосування PCA до кожної групи
        pca_results = {}
        for group_name, features in self.feature_groups.items():
            # Перевірка наявності всіх ознак
            missing_features = [f for f in features if f not in data.columns]
            if missing_features:
                raise ValueError(f"Ознаки {missing_features} відсутні в даних")
            
            pca, scaler, transformed, n_components = self._apply_pca(
                data, features, self.variance_threshold
            )
            pca_results[group_name] = {
                'pca': pca,
                'scaler': scaler,
                'transformed': transformed,
                'features': features
            }
        
        # Пошук інваріантів між парами груп
        for i, group_x_name in enumerate(group_names):
            for j, group_y_name in enumerate(group_names):
                if i >= j:  # Уникаємо дублікатів та пар з самою собою
                    continue
                
                result_x = pca_results[group_x_name]
                result_y = pca_results[group_y_name]
                
                # Перевірка всіх пар компонент
                for comp_x_idx in range(result_x['transformed'].shape[1]):
                    for comp_y_idx in range(result_y['transformed'].shape[1]):
                        x_component = result_x['transformed'][:, comp_x_idx]
                        y_component = result_y['transformed'][:, comp_y_idx]
                        
                        # Знаходження лінійної залежності
                        a, b, correlation, max_residual = self._find_linear_relationship(
                            x_component, y_component
                        )
                        
                        # Перевірка критерію валідності
                        if abs(correlation) >= self.correlation_threshold:
                            invariant = Invariant(
                                x_component_idx=comp_x_idx,
                                y_component_idx=comp_y_idx,
                                coefficient_a=a,
                                intercept_b=b,
                                correlation=correlation,
                                max_residual=max_residual,
                                pca_x=result_x['pca'],
                                pca_y=result_y['pca'],
                                scaler_x=result_x['scaler'],
                                scaler_y=result_y['scaler'],
                                feature_names_x=result_x['features'],
                                feature_names_y=result_y['features']
                            )
                            self.invariants.append(invariant)
        
        return self.invariants
    
    def detect_anomalies(
        self,
        data: pd.DataFrame,
        invariant: Optional[Invariant] = None
    ) -> Dict:
        """
        Детекція аномалій на нових даних.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Нові дані для перевірки
        invariant : Invariant, optional
            Конкретний інваріант для перевірки. Якщо None, перевіряються всі інваріанти.
            
        Returns:
        --------
        results : Dict
            Словник з результатами детекції:
            - 'violations': список порушень
            - 'status': загальний статус
        """
        if not self.invariants:
            raise ValueError("Спочатку навчіть модель через fit()")
        
        invariants_to_check = [invariant] if invariant else self.invariants
        violations = []
        
        for inv in invariants_to_check:
            # Трансформація даних через PCA
            X_x = data[inv.feature_names_x].values
            X_x_scaled = inv.scaler_x.transform(X_x)
            x_component = inv.pca_x.transform(X_x_scaled)[:, inv.x_component_idx]
            
            X_y = data[inv.feature_names_y].values
            X_y_scaled = inv.scaler_y.transform(X_y)
            y_component = inv.pca_y.transform(X_y_scaled)[:, inv.y_component_idx]
            
            # Прогноз за інваріантом
            y_pred = inv.coefficient_a * x_component + inv.intercept_b
            
            # Обчислення залишків
            residuals = np.abs(y_component - y_pred)
            threshold = inv.max_residual * (1 + self.overhead)
            
            # Виявлення порушень
            violation_indices = np.where(residuals > threshold)[0]
            
            if len(violation_indices) > 0:
                for idx in violation_indices:
                    violations.append({
                        'invariant_id': len(violations),
                        'x_component_idx': inv.x_component_idx,
                        'y_component_idx': inv.y_component_idx,
                        'row_index': idx,
                        'residual': residuals[idx],
                        'threshold': threshold,
                        'x_value': x_component[idx],
                        'y_actual': y_component[idx],
                        'y_predicted': y_pred[idx],
                        'correlation': inv.correlation
                    })
        
        return {
            'violations': violations,
            'status': 'anomaly_detected' if violations else 'normal',
            'total_violations': len(violations)
        }
    
    def get_invariants_summary(self) -> pd.DataFrame:
        """
        Отримати зведену інформацію про всі знайдені інваріанти.
        
        Returns:
        --------
        summary : pd.DataFrame
            DataFrame з інформацією про інваріанти
        """
        if not self.invariants:
            return pd.DataFrame()
        
        summary_data = []
        for i, inv in enumerate(self.invariants):
            summary_data.append({
                'invariant_id': i,
                'x_component': inv.x_component_idx,
                'y_component': inv.y_component_idx,
                'coefficient_a': inv.coefficient_a,
                'intercept_b': inv.intercept_b,
                'correlation': inv.correlation,
                'max_residual': inv.max_residual,
                'threshold': inv.max_residual * (1 + self.overhead)
            })
        
        return pd.DataFrame(summary_data)


