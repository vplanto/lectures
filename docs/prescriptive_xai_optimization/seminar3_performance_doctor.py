"""
Семінар 3: Контрфактуальна терапія та прескриптивний аналіз

Перетворює діагноз "Чому це сталося" на дієву пораду "Що змінити".
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from scipy.optimize import minimize
from seminar1_invariant_discovery import Invariant
from seminar2_shap_lime import InvariantDiagnostics
from stability_evaluator import StabilityEvaluator, StabilityReport
import warnings
warnings.filterwarnings('ignore')


@dataclass
class Counterfactual:
    """Результат пошуку контрфактуала."""
    original_values: Dict[str, float]
    counterfactual_values: Dict[str, float]
    changes: Dict[str, float]  # Зміни в абсолютних значеннях
    relative_changes: Dict[str, float]  # Зміни у відсотках
    distance: float  # Відстань від оригіналу
    predicted_y: float  # Прогнозоване значення y для контрфактуала
    target_y: float  # Цільове значення y (норма)
    is_valid: bool  # Чи задовольняє контрфактуал інваріант


@dataclass
class Prescription:
    """Рекомендація для виправлення системи."""
    invariant_id: int
    diagnosis: str
    actionable_features: List[str]
    recommended_changes: Dict[str, float]  # Зміни у відсотках
    expected_improvement: float
    confidence: float
    alternative_solutions: List[Counterfactual]
    # Causal awareness fields
    causal_validity: Optional[float] = None  # Валідність з точки зору причинності (0-1)
    causal_effects: Optional[Dict[str, float]] = None  # Causal effects для кожної ознаки
    confounders: Optional[Dict[str, List[str]]] = None  # Confounders для кожної ознаки
    causal_warnings: Optional[List[str]] = None  # Попередження про кореляцію vs причинність
    # Stability evaluation fields
    stability_report: Optional[StabilityReport] = None  # Звіт про стабільність пояснень
    stability_warnings: Optional[List[str]] = None  # Попередження про нестабільність


class PerformanceDoctor:
    """
    Модуль для генерації прескриптивних рекомендацій на основі
    порушених інваріантів та контрфактуального аналізу.
    """
    
    def __init__(
        self,
        actionable_features: Optional[List[str]] = None,
        non_actionable_features: Optional[List[str]] = None,
        feature_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        n_counterfactuals: int = 5,
        causal_graph: Optional[Dict[str, List[str]]] = None,
        causal_threshold: float = 0.1
    ):
        """
        Ініціалізація PerformanceDoctor.
        
        Parameters:
        -----------
        actionable_features : List[str], optional
            Список ознак, які можна змінювати (наприклад, конфігураційні параметри)
        non_actionable_features : List[str], optional
            Список ознак, які неможливо змінити (timestamp, user_id тощо)
        feature_bounds : Dict[str, Tuple[float, float]], optional
            Обмеження на значення ознак: {feature_name: (min, max)}
        n_counterfactuals : int
            Кількість альтернативних контрфактуалів для генерації
        """
        self.actionable_features = set(actionable_features) if actionable_features else set()
        self.non_actionable_features = set(non_actionable_features) if non_actionable_features else set()
        self.feature_bounds = feature_bounds or {}
        self.n_counterfactuals = n_counterfactuals
        self.diagnostics = InvariantDiagnostics()
        # Causal awareness
        self.causal_graph = causal_graph or {}  # {feature: [effects]} або {confounder: [affected_features]}
        self.causal_threshold = causal_threshold  # Поріг значущості causal effect
        # Stability evaluation
        self.stability_evaluator = StabilityEvaluator()
        self.evaluate_stability = True  # За замовчуванням увімкнено
    
    def _identify_actionable_features(
        self,
        all_features: List[str],
        data: pd.DataFrame
    ) -> List[str]:
        """
        Визначити дієві ознаки зі списку всіх ознак.
        
        Parameters:
        -----------
        all_features : List[str]
            Всі ознаки
        data : pd.DataFrame
            Дані для аналізу
            
        Returns:
        --------
        actionable : List[str]
            Список дієвих ознак
        """
        if self.actionable_features:
            # Якщо вказано явно, використовуємо їх
            return [f for f in all_features if f in self.actionable_features]
        
        # Автоматичне визначення: виключаємо timestamp, id, та інші недієві
        non_actionable_patterns = ['timestamp', 'id', 'time', 'date', 'user_id', 'session_id']
        
        actionable = []
        for feature in all_features:
            feature_lower = feature.lower()
            # Пропускаємо явно недієві
            if feature in self.non_actionable_features:
                continue
            # Пропускаємо ознаки з недієвими паттернами
            if any(pattern in feature_lower for pattern in non_actionable_patterns):
                continue
            # Перевірка на числові значення (дієві ознаки зазвичай числові)
            if pd.api.types.is_numeric_dtype(data[feature]):
                actionable.append(feature)
        
        return actionable
    
    def _get_feature_bounds(
        self,
        feature: str,
        data: pd.DataFrame,
        current_value: float
    ) -> Tuple[float, float]:
        """
        Отримати обмеження для ознаки.
        
        Parameters:
        -----------
        feature : str
            Назва ознаки
        data : pd.DataFrame
            Дані для визначення діапазону
        current_value : float
            Поточне значення
            
        Returns:
        --------
        bounds : Tuple[float, float]
            (min, max) значення
        """
        if feature in self.feature_bounds:
            return self.feature_bounds[feature]
        
        # Автоматичне визначення на основі даних
        feature_min = data[feature].min()
        feature_max = data[feature].max()
        
        # Дозволяємо зміни в межах ±50% від поточного значення
        # або в межах історичного діапазону
        min_val = max(feature_min, current_value * 0.5)
        max_val = min(feature_max, current_value * 1.5)
        
        return (min_val, max_val)
    
    def _predict_y_from_features(
        self,
        invariant: Invariant,
        feature_values: Dict[str, float]
    ) -> float:
        """
        Прогнозувати значення y компоненти на основі значень ознак.
        
        Parameters:
        -----------
        invariant : Invariant
            Інваріант для прогнозу
        feature_values : Dict[str, float]
            Значення ознак
            
        Returns:
        --------
        y_predicted : float
            Прогнозоване значення y компоненти
        """
        # Створення DataFrame з одним рядком
        row_data = {**feature_values}
        # Додаємо відсутні ознаки з середніми значеннями
        for feature in invariant.feature_names_x + invariant.feature_names_y:
            if feature not in row_data:
                row_data[feature] = 0.0  # Будемо використовувати середнє пізніше
        
        row_df = pd.DataFrame([row_data])
        
        # Трансформація через PCA
        X_x = row_df[invariant.feature_names_x].values
        X_x_scaled = invariant.scaler_x.transform(X_x)
        x_component = invariant.pca_x.transform(X_x_scaled)[:, invariant.x_component_idx]
        
        X_y = row_df[invariant.feature_names_y].values
        X_y_scaled = invariant.scaler_y.transform(X_y)
        y_component = invariant.pca_y.transform(X_y_scaled)[:, invariant.y_component_idx]
        
        # Прогноз за інваріантом
        y_predicted = invariant.coefficient_a * x_component[0] + invariant.intercept_b
        
        return y_predicted
    
    def _compute_target_y(
        self,
        invariant: Invariant,
        background_data: pd.DataFrame
    ) -> float:
        """
        Обчислити цільове (нормальне) значення y.
        
        Parameters:
        -----------
        invariant : Invariant
            Інваріант
        background_data : pd.DataFrame
            Фонові дані (нормальний стан)
            
        Returns:
        --------
        target_y : float
            Цільове значення y компоненти
        """
        X_y = background_data[invariant.feature_names_y].values
        X_y_scaled = invariant.scaler_y.transform(X_y)
        y_components = invariant.pca_y.transform(X_y_scaled)[:, invariant.y_component_idx]
        
        # Цільове значення - середнє на нормальних даних
        return np.mean(y_components)
    
    def find_counterfactual(
        self,
        invariant: Invariant,
        current_state: pd.DataFrame,
        row_idx: int,
        background_data: pd.DataFrame,
        target_y: Optional[float] = None
    ) -> Counterfactual:
        """
        Знайти контрфактуал - мінімальну зміну, що відновлює інваріант.
        
        Parameters:
        -----------
        invariant : Invariant
            Порушений інваріант
        current_state : pd.DataFrame
            Поточний стан системи
        row_idx : int
            Індекс рядка з порушенням
        background_data : pd.DataFrame
            Фонові дані (нормальний стан)
        target_y : float, optional
            Цільове значення y. Якщо None, обчислюється автоматично
            
        Returns:
        --------
        counterfactual : Counterfactual
            Знайдений контрфактуал
        """
        current_row = current_state.iloc[[row_idx]]
        all_features = invariant.feature_names_x + invariant.feature_names_y
        
        # Визначення дієвих ознак
        actionable = self._identify_actionable_features(all_features, current_state)
        
        if not actionable:
            raise ValueError("Не знайдено дієвих ознак для зміни")
        
        # Цільове значення y
        if target_y is None:
            target_y = self._compute_target_y(invariant, background_data)
        
        # Поточні значення
        original_values = current_row[actionable].iloc[0].to_dict()
        
        # Функція втрат для оптимізації
        def objective(x: np.ndarray) -> float:
            """Мінімізуємо відстань + штраф за відхилення від цільового y."""
            # Створення словника значень ознак
            feature_values = original_values.copy()
            for i, feature in enumerate(actionable):
                feature_values[feature] = x[i]
            
            # Додаємо недієві ознаки з поточного стану
            for feature in all_features:
                if feature not in feature_values:
                    feature_values[feature] = current_row[feature].iloc[0]
            
            # Прогноз y
            y_pred = self._predict_y_from_features(invariant, feature_values)
            
            # Відстань від оригіналу
            distance = np.sum((x - np.array([original_values[f] for f in actionable])) ** 2)
            
            # Штраф за відхилення від цільового y
            y_penalty = (y_pred - target_y) ** 2 * 10.0
            
            return distance + y_penalty
        
        # Обмеження
        bounds = []
        x0 = []
        for feature in actionable:
            current_val = original_values[feature]
            min_val, max_val = self._get_feature_bounds(feature, background_data, current_val)
            bounds.append((min_val, max_val))
            x0.append(current_val)
        
        # Оптимізація
        result = minimize(
            objective,
            x0=np.array(x0),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100}
        )
        
        # Формування результату
        counterfactual_values = {}
        changes = {}
        relative_changes = {}
        
        for i, feature in enumerate(actionable):
            new_val = result.x[i]
            old_val = original_values[feature]
            counterfactual_values[feature] = new_val
            changes[feature] = new_val - old_val
            if abs(old_val) > 1e-10:
                relative_changes[feature] = (new_val - old_val) / old_val * 100
            else:
                relative_changes[feature] = 0.0
        
        # Прогноз для контрфактуала
        cf_feature_values = {**original_values, **counterfactual_values}
        for feature in all_features:
            if feature not in cf_feature_values:
                cf_feature_values[feature] = current_row[feature].iloc[0]
        
        predicted_y = self._predict_y_from_features(invariant, cf_feature_values)
        
        # Обчислення фактичного y компоненти для контрфактуала
        X_y_cf = pd.DataFrame([cf_feature_values])[invariant.feature_names_y].values
        X_y_cf_scaled = invariant.scaler_y.transform(X_y_cf)
        y_actual_cf = invariant.pca_y.transform(X_y_cf_scaled)[:, invariant.y_component_idx][0]
        
        # Перевірка валідності (чи задовольняє інваріант)
        # Residual - це різниця між фактичним y та прогнозованим за інваріантом
        residual = abs(y_actual_cf - predicted_y)
        threshold = invariant.max_residual * 1.1
        is_valid = residual <= threshold
        
        return Counterfactual(
            original_values=original_values,
            counterfactual_values=counterfactual_values,
            changes=changes,
            relative_changes=relative_changes,
            distance=result.fun,
            predicted_y=predicted_y,
            target_y=target_y,
            is_valid=is_valid
        )
    
    def find_diverse_counterfactuals(
        self,
        invariant: Invariant,
        current_state: pd.DataFrame,
        row_idx: int,
        background_data: pd.DataFrame,
        n_solutions: Optional[int] = None
    ) -> List[Counterfactual]:
        """
        Знайти кілька різноманітних контрфактуалів (DiCE-підхід).
        
        Parameters:
        -----------
        invariant : Invariant
            Порушений інваріант
        current_state : pd.DataFrame
            Поточний стан
        row_idx : int
            Індекс рядка
        background_data : pd.DataFrame
            Фонові дані
        n_solutions : int, optional
            Кількість рішень (за замовчуванням self.n_counterfactuals)
            
        Returns:
        --------
        counterfactuals : List[Counterfactual]
            Список різноманітних контрфактуалів
        """
        n_solutions = n_solutions or self.n_counterfactuals
        counterfactuals = []
        
        # Знаходимо базовий контрфактуал
        base_cf = self.find_counterfactual(
            invariant, current_state, row_idx, background_data
        )
        counterfactuals.append(base_cf)
        
        # Генерація додаткових різноманітних рішень
        # Використовуємо різні початкові точки та ваги в функції втрат
        for i in range(n_solutions - 1):
            # Змінюємо ваги для різноманітності
            # (спрощена версія - можна покращити через генетичні алгоритми)
            try:
                cf = self.find_counterfactual(
                    invariant, current_state, row_idx, background_data
                )
                # Перевірка на унікальність
                is_unique = True
                for existing_cf in counterfactuals:
                    if np.allclose(
                        [existing_cf.counterfactual_values.get(f, 0) for f in 
                         sorted(cf.counterfactual_values.keys())],
                        [cf.counterfactual_values.get(f, 0) for f in 
                         sorted(cf.counterfactual_values.keys())],
                        atol=1e-3
                    ):
                        is_unique = False
                        break
                
                if is_unique:
                    counterfactuals.append(cf)
            except:
                continue
        
        return counterfactuals
    
    def prescribe_solution(
        self,
        invariant: Invariant,
        current_state: pd.DataFrame,
        row_idx: int,
        background_data: pd.DataFrame,
        diagnosis: Optional[str] = None
    ) -> Prescription:
        """
        Створити прескриптивну рекомендацію для виправлення системи.
        
        Parameters:
        -----------
        invariant : Invariant
            Порушений інваріант
        current_state : pd.DataFrame
            Поточний стан
        row_idx : int
            Індекс рядка з порушенням
        background_data : pd.DataFrame
            Фонові дані
        diagnosis : str, optional
            Діагноз (може бути отриманий через SHAP/LIME)
            
        Returns:
        --------
        prescription : Prescription
            Рекомендація для виправлення
        """
        # Діагностика через SHAP
        if diagnosis is None:
            shap_result = self.diagnostics.diagnose_violation(
                invariant, current_state, row_idx, background_data, method='shap'
            )
            top_contributors = shap_result['top_contributors'][:3]
            diagnosis = f"Порушення інваріанта через: {', '.join([c['feature'] for c in top_contributors])}"
        
        # Пошук контрфактуалів
        counterfactuals = self.find_diverse_counterfactuals(
            invariant, current_state, row_idx, background_data
        )
        
        if not counterfactuals:
            raise ValueError("Не вдалося знайти контрфактуали")
        
        # Вибір найкращого рішення (мінімальна відстань + валідність)
        best_cf = min(
            [cf for cf in counterfactuals if cf.is_valid],
            key=lambda x: x.distance,
            default=counterfactuals[0]
        )
        
        # Визначення дієвих ознак
        actionable = self._identify_actionable_features(
            invariant.feature_names_x + invariant.feature_names_y,
            current_state
        )
        
        # Очікуване покращення
        current_y = self._predict_y_from_features(
            invariant,
            current_state.iloc[[row_idx]].to_dict('records')[0]
        )
        expected_improvement = abs(best_cf.predicted_y - current_y)
        
        # Впевненість (на основі валідності та відстані)
        confidence = 0.9 if best_cf.is_valid else 0.5
        if best_cf.distance > 0:
            confidence *= min(1.0, 1.0 / (1.0 + best_cf.distance * 0.1))
        
        # Causal awareness: оцінка причинності
        causal_validity, causal_effects, confounders, causal_warnings = self._evaluate_causality(
            best_cf, invariant, current_state, row_idx, background_data
        )
        
        # Корекція впевненості з урахуванням причинності
        if causal_validity is not None:
            confidence *= causal_validity
        
        # Stability evaluation (обов'язковий етап валідації)
        stability_report = None
        stability_warnings = None
        if self.evaluate_stability:
            try:
                stability_report = self.stability_evaluator.evaluate(
                    invariant=invariant,
                    current_state=current_state,
                    row_idx=row_idx,
                    background_data=background_data,
                    diagnostics=self.diagnostics
                )
                
                # Корекція впевненості на основі стабільності
                if stability_report.robustness_score < self.stability_evaluator.stability_threshold:
                    confidence *= stability_report.robustness_score
                
                stability_warnings = stability_report.warnings
            except Exception as e:
                # Якщо stability evaluation не вдався, додаємо попередження
                stability_warnings = [f"⚠️ Не вдалося оцінити стабільність: {str(e)}"]
        
        return Prescription(
            invariant_id=id(invariant) % 10000,  # Простий ID
            diagnosis=diagnosis,
            actionable_features=actionable,
            recommended_changes=best_cf.relative_changes,
            expected_improvement=expected_improvement,
            confidence=confidence,
            alternative_solutions=counterfactuals[1:] if len(counterfactuals) > 1 else [],
            causal_validity=causal_validity,
            causal_effects=causal_effects,
            confounders=confounders,
            causal_warnings=causal_warnings,
            stability_report=stability_report,
            stability_warnings=stability_warnings
        )
    
    def format_prescription_message(self, prescription: Prescription) -> str:
        """
        Форматувати рекомендацію у зрозуміле повідомлення.
        
        Parameters:
        -----------
        prescription : Prescription
            Рекомендація
            
        Returns:
        --------
        message : str
            Форматоване повідомлення
        """
        message_parts = [
            f"🔍 Діагноз: {prescription.diagnosis}",
            f"\n💡 Рекомендації для відновлення стабільності інваріанта:",
        ]
        
        for feature, change_pct in prescription.recommended_changes.items():
            direction = "збільшити" if change_pct > 0 else "зменшити"
            message_parts.append(
                f"  • {direction.capitalize()} параметр '{feature}' на {abs(change_pct):.1f}%"
            )
        
        message_parts.append(
            f"\n📊 Очікуване покращення: {prescription.expected_improvement:.4f}"
        )
        message_parts.append(
            f"🎯 Впевненість: {prescription.confidence*100:.1f}%"
        )
        
        # Causal awareness інформація
        if prescription.causal_validity is not None:
            message_parts.append(
                f"\n🔬 Causal Validity: {prescription.causal_validity*100:.1f}%"
            )
            
            if prescription.causal_effects:
                message_parts.append("\n📊 Causal Effects:")
                for feature, effect in prescription.causal_effects.items():
                    change = prescription.recommended_changes.get(feature, 0)
                    message_parts.append(
                        f"  • {feature}: effect={effect:.4f} (зміна: {change:+.1f}%)"
                    )
            
            if prescription.causal_warnings:
                message_parts.append("\n⚠️ Causal Warnings:")
                for warning in prescription.causal_warnings:
                    message_parts.append(f"  {warning}")
        
        # Stability evaluation інформація
        if prescription.stability_report is not None:
            sr = prescription.stability_report
            status = "✅" if sr.is_stable else "⚠️"
            message_parts.append(
                f"\n{status} Stability Evaluation (Robustness): {sr.robustness_score*100:.1f}%"
            )
            message_parts.append(
                f"  • CV Score: {sr.cv_score:.3f} {'✅' if sr.cv_score < 0.2 else '⚠️'}"
            )
            message_parts.append(
                f"  • Rank Stability: {sr.rank_stability:.3f} {'✅' if sr.rank_stability >= 0.8 else '⚠️'}"
            )
            message_parts.append(
                f"  • Agreement (SHAP/LIME): {sr.agreement_score:.3f} {'✅' if sr.agreement_score >= 0.6 else '⚠️'}"
            )
            
            if prescription.stability_warnings:
                message_parts.append("\n⚠️ Stability Warnings:")
                for warning in prescription.stability_warnings:
                    message_parts.append(f"  {warning}")
        elif prescription.stability_warnings:
            message_parts.append("\n⚠️ Stability Warnings:")
            for warning in prescription.stability_warnings:
                message_parts.append(f"  {warning}")
        
        if prescription.alternative_solutions:
            message_parts.append(
                f"\n🔄 Альтернативні рішення: {len(prescription.alternative_solutions)} варіантів"
            )
        
        return "\n".join(message_parts)

