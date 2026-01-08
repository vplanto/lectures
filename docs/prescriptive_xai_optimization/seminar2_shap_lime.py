"""
Семінар 2: Діагностика розриву інваріантів через SHAP та LIME

Пояснює, чому саме математичний інваріант було порушено.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from seminar1_invariant_discovery import Invariant
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Попередження: бібліотека shap не встановлена. Використовується спрощена реалізація.")


@dataclass
class ExplanationResult:
    """Результат пояснення порушення інваріанта."""
    feature_names: List[str]
    shap_values: np.ndarray
    base_value: float
    predicted_value: float
    actual_value: float
    contribution_sum: float  # Сума SHAP значень (має дорівнювати різниці)


class KernelSHAPExplainer:
    """
    Спрощена реалізація KernelSHAP для пояснення відхилень від інваріантів.
    """
    
    def __init__(self, n_samples: int = 100):
        """
        Ініціалізація explainer.
        
        Parameters:
        -----------
        n_samples : int
            Кількість зразків для апроксимації SHAP значень
        """
        self.n_samples = n_samples
    
    def explain_invariant_violation(
        self,
        invariant: Invariant,
        data: pd.DataFrame,
        violation_row_idx: int,
        background_data: Optional[pd.DataFrame] = None
    ) -> ExplanationResult:
        """
        Пояснити порушення інваріанта через SHAP значення.
        
        Parameters:
        -----------
        invariant : Invariant
            Порушений інваріант
        data : pd.DataFrame
            Дані, що містять порушення
        violation_row_idx : int
            Індекс рядка з порушенням
        background_data : pd.DataFrame, optional
            Фонові дані для обчислення базового значення
            
        Returns:
        --------
        explanation : ExplanationResult
            Результат пояснення
        """
        # Вибірка даних
        violation_row = data.iloc[[violation_row_idx]]
        
        # Визначення фонового набору
        if background_data is None:
            background_data = data.drop([violation_row_idx])
        
        # Об'єднання всіх ознак
        all_features = invariant.feature_names_x + invariant.feature_names_y
        feature_names = all_features
        
        # Трансформація через PCA для отримання прогнозу
        X_x = violation_row[invariant.feature_names_x].values
        X_x_scaled = invariant.scaler_x.transform(X_x)
        x_component = invariant.pca_x.transform(X_x_scaled)[:, invariant.x_component_idx]
        
        X_y = violation_row[invariant.feature_names_y].values
        X_y_scaled = invariant.scaler_y.transform(X_y)
        y_component_actual = invariant.pca_y.transform(X_y_scaled)[:, invariant.y_component_idx]
        
        # Прогноз за інваріантом
        y_component_predicted = invariant.coefficient_a * x_component[0] + invariant.intercept_b
        
        # Базове значення (середнє на фонових даних)
        X_y_bg = background_data[invariant.feature_names_y].values
        X_y_bg_scaled = invariant.scaler_y.transform(X_y_bg)
        y_bg_components = invariant.pca_y.transform(X_y_bg_scaled)[:, invariant.y_component_idx]
        base_value = np.mean(y_bg_components)
        
        # Обчислення SHAP значень
        shap_values = self._compute_shap_values(
            invariant, violation_row, background_data, all_features
        )
        
        # Перевірка: сума SHAP значень має дорівнювати різниці
        contribution_sum = np.sum(shap_values)
        expected_delta = y_component_actual[0] - base_value
        
        return ExplanationResult(
            feature_names=feature_names,
            shap_values=shap_values,
            base_value=base_value,
            predicted_value=y_component_predicted[0],
            actual_value=y_component_actual[0],
            contribution_sum=contribution_sum
        )
    
    def _compute_shap_values(
        self,
        invariant: Invariant,
        instance: pd.DataFrame,
        background: pd.DataFrame,
        feature_names: List[str]
    ) -> np.ndarray:
        """
        Обчислити SHAP значення для ознак.
        
        Використовує спрощений підхід KernelSHAP.
        """
        n_features = len(feature_names)
        shap_values = np.zeros(n_features)
        
        # Функція-модель: прогноз y компоненти на основі ознак
        def model_predict(X: pd.DataFrame) -> np.ndarray:
            """Прогноз y компоненти через інваріант."""
            X_x = X[invariant.feature_names_x].values
            X_x_scaled = invariant.scaler_x.transform(X_x)
            x_comp = invariant.pca_x.transform(X_x_scaled)[:, invariant.x_component_idx]
            
            X_y = X[invariant.feature_names_y].values
            X_y_scaled = invariant.scaler_y.transform(X_y)
            y_comp = invariant.pca_y.transform(X_y_scaled)[:, invariant.y_component_idx]
            
            return y_comp
        
        # Базове значення (середнє на фоні)
        y_base = np.mean(model_predict(background))
        
        # Значення для поточного інстансу
        y_instance = model_predict(instance)[0]
        
        # Спрощений підхід: обчислення внеску кожної ознаки
        # через різницю між прогнозом з та без цієї ознаки
        instance_values = instance[feature_names].values[0]
        background_mean = background[feature_names].mean().values
        
        for i, feature_name in enumerate(feature_names):
            # Створення гібридного інстансу: поточне значення для цієї ознаки,
            # середнє значення для інших
            hybrid_instance = instance.copy()
            hybrid_instance[feature_name] = instance_values[i]
            
            # Прогноз з поточним значенням ознаки
            y_with_feature = model_predict(hybrid_instance)[0]
            
            # Прогноз з середнім значенням ознаки
            hybrid_instance[feature_name] = background_mean[i]
            y_without_feature = model_predict(hybrid_instance)[0]
            
            # Внесок ознаки
            shap_values[i] = y_with_feature - y_without_feature
        
        # Нормалізація: сума має дорівнювати різниці між фактичним та базовим
        total_contribution = np.sum(shap_values)
        target_delta = y_instance - y_base
        
        if abs(total_contribution) > 1e-10:
            shap_values = shap_values * (target_delta / total_contribution)
        
        return shap_values


class LIMExplainer:
    """
    Локальна апроксимація через LIME для швидкої інтерпретації аномалій.
    """
    
    def __init__(self, n_samples: int = 1000, kernel_width: float = 0.75):
        """
        Ініціалізація LIME explainer.
        
        Parameters:
        -----------
        n_samples : int
            Кількість зразків для локальної моделі
        kernel_width : float
            Ширина ядра для вагування зразків
        """
        self.n_samples = n_samples
        self.kernel_width = kernel_width
    
    def explain_invariant_violation(
        self,
        invariant: Invariant,
        data: pd.DataFrame,
        violation_row_idx: int,
        background_data: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Пояснити порушення через локальну лінійну модель.
        
        Parameters:
        -----------
        invariant : Invariant
            Порушений інваріант
        data : pd.DataFrame
            Дані з порушенням
        violation_row_idx : int
            Індекс рядка з порушенням
        background_data : pd.DataFrame, optional
            Фонові дані
            
        Returns:
        --------
        explanation : Dict
            Результат пояснення з коефіцієнтами лінійної моделі
        """
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        
        violation_row = data.iloc[[violation_row_idx]]
        
        if background_data is None:
            background_data = data.drop([violation_row_idx])
        
        all_features = invariant.feature_names_x + invariant.feature_names_y
        
        # Генерація зразків навколо точки аномалії
        instance_values = violation_row[all_features].values[0]
        background_mean = background_data[all_features].mean().values
        background_std = background_data[all_features].std().values
        background_std = np.where(background_std < 1e-10, 1.0, background_std)
        
        # Генерація випадкових зразків
        samples = []
        for _ in range(self.n_samples):
            # Зразок: інстанс + випадковий шум
            noise = np.random.normal(0, 1, len(all_features))
            sample = instance_values + noise * background_std * 0.1
            samples.append(sample)
        
        samples_df = pd.DataFrame(samples, columns=all_features)
        
        # Прогноз для зразків через інваріант
        def predict_y_component(X: pd.DataFrame) -> np.ndarray:
            X_x = X[invariant.feature_names_x].values
            X_x_scaled = invariant.scaler_x.transform(X_x)
            x_comp = invariant.pca_x.transform(X_x_scaled)[:, invariant.x_component_idx]
            
            X_y = X[invariant.feature_names_y].values
            X_y_scaled = invariant.scaler_y.transform(X_y)
            y_comp = invariant.pca_y.transform(X_y_scaled)[:, invariant.y_component_idx]
            
            return y_comp
        
        y_samples = predict_y_component(samples_df)
        y_instance = predict_y_component(violation_row)[0]
        
        # Обчислення відстаней та ваг
        distances = np.linalg.norm(
            samples_df.values - instance_values.reshape(1, -1),
            axis=1
        )
        weights = np.exp(-distances / (self.kernel_width * np.mean(distances)))
        
        # Навчання локальної лінійної моделі
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(samples_df)
        
        model = Ridge(alpha=0.1)
        model.fit(X_scaled, y_samples, sample_weight=weights)
        
        # Коефіцієнти моделі
        coefficients = model.coef_
        intercept = model.intercept_
        
        # Важливість ознак (абсолютні значення коефіцієнтів)
        feature_importance = np.abs(coefficients)
        
        return {
            'feature_names': all_features,
            'coefficients': coefficients,
            'intercept': intercept,
            'feature_importance': feature_importance,
            'predicted_value': model.predict(scaler.transform(violation_row[all_features]))[0],
            'actual_value': y_instance
        }


def create_waterfall_plot_data(explanation: ExplanationResult) -> pd.DataFrame:
    """
    Створити дані для waterfall візуалізації SHAP значень.
    
    Parameters:
    -----------
    explanation : ExplanationResult
        Результат пояснення
        
    Returns:
    --------
    plot_data : pd.DataFrame
        DataFrame з даними для візуалізації
    """
    # Сортування за абсолютним значенням внеску
    indices = np.argsort(np.abs(explanation.shap_values))[::-1]
    
    plot_data = []
    cumulative = explanation.base_value
    
    for idx in indices:
        feature_name = explanation.feature_names[idx]
        shap_value = explanation.shap_values[idx]
        cumulative += shap_value
        
        plot_data.append({
            'feature': feature_name,
            'contribution': shap_value,
            'cumulative': cumulative,
            'abs_contribution': abs(shap_value)
        })
    
    return pd.DataFrame(plot_data)


class InvariantDiagnostics:
    """
    Головний клас для діагностики порушень інваріантів.
    """
    
    def __init__(self, use_shap_library: bool = True):
        """
        Ініціалізація діагностики.
        
        Parameters:
        -----------
        use_shap_library : bool
            Використовувати офіційну бібліотеку shap, якщо доступна
        """
        self.use_shap_library = use_shap_library and SHAP_AVAILABLE
        self.kernel_shap = KernelSHAPExplainer()
        self.lime = LIMExplainer()
    
    def diagnose_violation(
        self,
        invariant: Invariant,
        data: pd.DataFrame,
        violation_row_idx: int,
        background_data: Optional[pd.DataFrame] = None,
        method: str = 'shap'
    ) -> Dict:
        """
        Діагностувати порушення інваріанта.
        
        Parameters:
        -----------
        invariant : Invariant
            Порушений інваріант
        data : pd.DataFrame
            Дані з порушенням
        violation_row_idx : int
            Індекс рядка з порушенням
        background_data : pd.DataFrame, optional
            Фонові дані
        method : str
            Метод пояснення: 'shap' або 'lime'
            
        Returns:
        --------
        diagnosis : Dict
            Результат діагностики
        """
        if method.lower() == 'shap':
            explanation = self.kernel_shap.explain_invariant_violation(
                invariant, data, violation_row_idx, background_data
            )
            waterfall_data = create_waterfall_plot_data(explanation)
            
            return {
                'method': 'SHAP',
                'explanation': explanation,
                'waterfall_data': waterfall_data,
                'top_contributors': waterfall_data.head(10).to_dict('records')
            }
        
        elif method.lower() == 'lime':
            explanation = self.lime.explain_invariant_violation(
                invariant, data, violation_row_idx, background_data
            )
            
            # Сортування за важливістю
            importance_df = pd.DataFrame({
                'feature': explanation['feature_names'],
                'importance': explanation['feature_importance'],
                'coefficient': explanation['coefficients']
            }).sort_values('importance', ascending=False)
            
            return {
                'method': 'LIME',
                'explanation': explanation,
                'top_contributors': importance_df.head(10).to_dict('records')
            }
        
        else:
            raise ValueError(f"Невідомий метод: {method}. Використовуйте 'shap' або 'lime'")


