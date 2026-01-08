"""
Stability Evaluation: Оцінка стабільності пояснень (Robustness)

Обов'язковий етап валідації системи Performance Doctor.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from seminar1_invariant_discovery import Invariant
from seminar2_shap_lime import InvariantDiagnostics, KernelSHAPExplainer, LIMExplainer
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')


@dataclass
class StabilityReport:
    """Звіт про стабільність пояснень."""
    # Метрики стабільності
    robustness_score: float  # Загальна оцінка стабільності (0-1)
    cv_score: float  # Коефіцієнт варіації
    rank_stability: float  # Стабільність рангу
    top_k_stability: float  # Стабільність топ-k
    agreement_score: float  # Узгодженість між методами
    
    # Деталі
    bootstrap_metrics: Dict[str, float]
    perturbation_metrics: Dict[str, float]
    agreement_metrics: Dict[str, float]
    
    # Статус
    is_stable: bool  # Чи є пояснення стабільним
    warnings: List[str]  # Попередження про нестабільність


class StabilityEvaluator:
    """
    Клас для оцінки стабільності пояснень (Robustness).
    """
    
    def __init__(
        self,
        n_bootstrap: int = 50,
        n_perturbations: int = 30,
        noise_level: float = 0.05,
        stability_threshold: float = 0.7
    ):
        """
        Ініціалізація evaluator.
        
        Parameters:
        -----------
        n_bootstrap : int
            Кількість bootstrap зразків для оцінки
        n_perturbations : int
            Кількість perturbations для оцінки
        noise_level : float
            Рівень шуму для perturbations
        stability_threshold : float
            Поріг стабільності (robustness score)
        """
        self.n_bootstrap = n_bootstrap
        self.n_perturbations = n_perturbations
        self.noise_level = noise_level
        self.stability_threshold = stability_threshold
    
    def evaluate(
        self,
        invariant: Invariant,
        current_state: pd.DataFrame,
        row_idx: int,
        background_data: pd.DataFrame,
        diagnostics: InvariantDiagnostics
    ) -> StabilityReport:
        """
        Повна оцінка стабільності пояснень.
        
        Parameters:
        -----------
        invariant : Invariant
            Інваріант
        current_state : pd.DataFrame
            Поточний стан
        row_idx : int
            Індекс рядка з порушенням
        background_data : pd.DataFrame
            Фонові дані
        diagnostics : InvariantDiagnostics
            Об'єкт діагностики
            
        Returns:
        --------
        report : StabilityReport
            Звіт про стабільність
        """
        # Bootstrap evaluation
        bootstrap_metrics = self._evaluate_bootstrap(
            invariant, current_state, row_idx, background_data, diagnostics
        )
        
        # Perturbation evaluation
        perturbation_metrics = self._evaluate_perturbation(
            invariant, current_state, row_idx, background_data, diagnostics
        )
        
        # Agreement evaluation
        agreement_metrics = self._evaluate_agreement(
            invariant, current_state, row_idx, background_data, diagnostics
        )
        
        # Обчислення загальних метрик
        robustness_score = (
            bootstrap_metrics.get('robustness', 0.5) * 0.4 +
            perturbation_metrics.get('robustness', 0.5) * 0.4 +
            agreement_metrics.get('agreement_score', 0.5) * 0.2
        )
        
        cv_score = bootstrap_metrics.get('mean_cv', 0.0)
        rank_stability = bootstrap_metrics.get('rank_stability', 0.0)
        top_k_stability = bootstrap_metrics.get('top_k_stability', 0.0)
        agreement_score = agreement_metrics.get('agreement_score', 0.0)
        
        # Перевірка стабільності
        is_stable, warnings = self._check_stability(
            robustness_score, cv_score, rank_stability, agreement_score
        )
        
        return StabilityReport(
            robustness_score=robustness_score,
            cv_score=cv_score,
            rank_stability=rank_stability,
            top_k_stability=top_k_stability,
            agreement_score=agreement_score,
            bootstrap_metrics=bootstrap_metrics,
            perturbation_metrics=perturbation_metrics,
            agreement_metrics=agreement_metrics,
            is_stable=is_stable,
            warnings=warnings
        )
    
    def _evaluate_bootstrap(
        self,
        invariant: Invariant,
        current_state: pd.DataFrame,
        row_idx: int,
        background_data: pd.DataFrame,
        diagnostics: InvariantDiagnostics
    ) -> Dict[str, float]:
        """Оцінка стабільності через bootstrap sampling."""
        explanations = []
        
        for _ in range(self.n_bootstrap):
            # Генерація bootstrap зразка
            bootstrap_indices = np.random.choice(
                len(background_data),
                size=int(len(background_data) * 0.8),
                replace=True
            )
            bootstrap_data = background_data.iloc[bootstrap_indices]
            
            try:
                # Генерація пояснення через SHAP
                shap_result = diagnostics.diagnose_violation(
                    invariant, current_state, row_idx, bootstrap_data, method='shap'
                )
                
                # Витягування SHAP значень
                if 'explanation' in shap_result:
                    shap_values = shap_result['explanation'].shap_values
                    explanations.append(shap_values)
            except:
                continue
        
        if len(explanations) < 5:
            return {
                'robustness': 0.0,
                'mean_cv': 1.0,
                'rank_stability': 0.0,
                'top_k_stability': 0.0
            }
        
        # Обчислення метрик
        explanations_array = np.array(explanations)
        mean_explanation = np.mean(explanations_array, axis=0)
        var_explanation = np.var(explanations_array, axis=0)
        
        # Коефіцієнт варіації
        cv = np.sqrt(var_explanation) / (np.abs(mean_explanation) + 1e-10)
        mean_cv = np.mean(cv)
        
        # Robustness
        deviations = []
        for exp in explanations:
            deviation = np.linalg.norm(exp - mean_explanation) / (
                np.linalg.norm(mean_explanation) + 1e-10
            )
            deviations.append(deviation)
        robustness = 1.0 - np.mean(deviations)
        robustness = max(0.0, min(1.0, robustness))
        
        # Rank stability
        ranks = [np.argsort(np.abs(exp))[::-1] for exp in explanations]
        rank_correlations = []
        for i in range(min(10, len(ranks))):
            for j in range(i+1, min(10, len(ranks))):
                try:
                    corr, _ = spearmanr(ranks[i], ranks[j])
                    if not np.isnan(corr):
                        rank_correlations.append(corr)
                except:
                    continue
        rank_stability = np.mean(rank_correlations) if rank_correlations else 0.0
        
        # Top-K stability (k=5)
        k = 5
        top_k_sets = [set(np.argsort(np.abs(exp))[::-1][:k]) for exp in explanations]
        top_k_intersections = []
        for i in range(min(10, len(top_k_sets))):
            for j in range(i+1, min(10, len(top_k_sets))):
                intersection = len(top_k_sets[i] & top_k_sets[j]) / k
                top_k_intersections.append(intersection)
        top_k_stability = np.mean(top_k_intersections) if top_k_intersections else 0.0
        
        return {
            'robustness': robustness,
            'mean_cv': mean_cv,
            'rank_stability': rank_stability,
            'top_k_stability': top_k_stability
        }
    
    def _evaluate_perturbation(
        self,
        invariant: Invariant,
        current_state: pd.DataFrame,
        row_idx: int,
        background_data: pd.DataFrame,
        diagnostics: InvariantDiagnostics
    ) -> Dict[str, float]:
        """Оцінка стабільності через perturbation analysis."""
        violation_row = current_state.iloc[[row_idx]]
        all_features = invariant.feature_names_x + invariant.feature_names_y
        
        # Базове пояснення
        try:
            base_result = diagnostics.diagnose_violation(
                invariant, current_state, row_idx, background_data, method='shap'
            )
            if 'explanation' not in base_result:
                return {'robustness': 0.0, 'mean_deviation': 1.0, 'max_deviation': 1.0}
            base_explanation = base_result['explanation'].shap_values
        except:
            return {'robustness': 0.0, 'mean_deviation': 1.0, 'max_deviation': 1.0}
        
        explanations = [base_explanation]
        instance_values = violation_row[all_features].values[0]
        std_values = background_data[all_features].std().values
        std_values = np.where(std_values < 1e-10, 1.0, std_values)
        
        for _ in range(self.n_perturbations):
            # Додавання шуму
            noise = np.random.normal(0, self.noise_level, size=instance_values.shape)
            perturbed_values = instance_values + noise * std_values
            perturbed_row = violation_row.copy()
            perturbed_row[all_features] = perturbed_values
            
            try:
                # Генерація пояснення
                perturbed_result = diagnostics.diagnose_violation(
                    invariant, perturbed_row, 0, background_data, method='shap'
                )
                if 'explanation' in perturbed_result:
                    explanations.append(perturbed_result['explanation'].shap_values)
            except:
                continue
        
        if len(explanations) < 3:
            return {'robustness': 0.0, 'mean_deviation': 1.0, 'max_deviation': 1.0}
        
        # Обчислення метрик
        explanations_array = np.array(explanations)
        mean_explanation = np.mean(explanations_array, axis=0)
        
        # Robustness
        deviations = []
        for exp in explanations:
            deviation = np.linalg.norm(exp - mean_explanation) / (
                np.linalg.norm(mean_explanation) + 1e-10
            )
            deviations.append(deviation)
        
        robustness = 1.0 - np.mean(deviations)
        robustness = max(0.0, min(1.0, robustness))
        
        return {
            'robustness': robustness,
            'mean_deviation': np.mean(deviations),
            'max_deviation': np.max(deviations)
        }
    
    def _evaluate_agreement(
        self,
        invariant: Invariant,
        current_state: pd.DataFrame,
        row_idx: int,
        background_data: pd.DataFrame,
        diagnostics: InvariantDiagnostics
    ) -> Dict[str, float]:
        """Оцінка узгодженості між SHAP та LIME."""
        try:
            # SHAP пояснення
            shap_result = diagnostics.diagnose_violation(
                invariant, current_state, row_idx, background_data, method='shap'
            )
            if 'explanation' not in shap_result:
                return {'agreement_score': 0.0, 'pearson_correlation': 0.0, 'spearman_correlation': 0.0}
            shap_values = shap_result['explanation'].shap_values
            
            # LIME пояснення
            lime_result = diagnostics.diagnose_violation(
                invariant, current_state, row_idx, background_data, method='lime'
            )
            if 'explanation' not in lime_result:
                return {'agreement_score': 0.0, 'pearson_correlation': 0.0, 'spearman_correlation': 0.0}
            lime_importance = lime_result['explanation']['feature_importance']
            
            # Нормалізація
            shap_norm = shap_values / (np.linalg.norm(shap_values) + 1e-10)
            lime_norm = lime_importance / (np.linalg.norm(lime_importance) + 1e-10)
            
            # Кореляції
            try:
                pearson_corr, _ = pearsonr(shap_norm, lime_norm)
                if np.isnan(pearson_corr):
                    pearson_corr = 0.0
            except:
                pearson_corr = 0.0
            
            try:
                spearman_corr, _ = spearmanr(shap_norm, lime_norm)
                if np.isnan(spearman_corr):
                    spearman_corr = 0.0
            except:
                spearman_corr = 0.0
            
            # Косинусна подібність
            cosine_sim = np.dot(shap_norm, lime_norm) / (
                np.linalg.norm(shap_norm) * np.linalg.norm(lime_norm) + 1e-10
            )
            
            agreement_score = (abs(pearson_corr) + abs(spearman_corr) + abs(cosine_sim)) / 3
            
            return {
                'agreement_score': agreement_score,
                'pearson_correlation': pearson_corr,
                'spearman_correlation': spearman_corr,
                'cosine_similarity': cosine_sim
            }
        except Exception as e:
            return {'agreement_score': 0.0, 'pearson_correlation': 0.0, 'spearman_correlation': 0.0}
    
    def _check_stability(
        self,
        robustness_score: float,
        cv_score: float,
        rank_stability: float,
        agreement_score: float
    ) -> Tuple[bool, List[str]]:
        """Перевірка стабільності та генерація попереджень."""
        warnings = []
        
        if robustness_score < self.stability_threshold:
            warnings.append(
                f"⚠️ Низька стабільність (robustness={robustness_score:.2f}). "
                "Рекомендується збільшити кількість зразків або перевірити дані."
            )
        
        if cv_score >= 0.2:
            warnings.append(
                f"⚠️ Висока варіативність (CV={cv_score:.2f}). "
                "Пояснення може бути нестабільним."
            )
        
        if rank_stability < 0.8:
            warnings.append(
                f"⚠️ Нестабільний порядок важливості (rank_stability={rank_stability:.2f}). "
                "Топ-ознаки можуть змінюватися між запусками."
            )
        
        if agreement_score < 0.6:
            warnings.append(
                f"⚠️ Низька узгодженість між методами (agreement={agreement_score:.2f}). "
                "SHAP та LIME дають різні результати."
            )
        
        is_stable = (
            robustness_score >= self.stability_threshold and
            cv_score < 0.2 and
            rank_stability >= 0.8 and
            agreement_score >= 0.6
        )
        
        return is_stable, warnings


