"""
Приклад використання системи моніторингу IT-інфраструктур
з виявленням інваріантів, діагностикою та прескриптивним аналізом.
"""

import numpy as np
import pandas as pd
from seminar1_invariant_discovery import InvariantDiscovery
from seminar2_shap_lime import InvariantDiagnostics
from seminar3_performance_doctor import PerformanceDoctor


def generate_synthetic_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    Генерація синтетичних даних для демонстрації.
    
    Створює дані з логічними залежностями між метриками:
    - CPU метрики: cpu_usage, cpu_freq, cpu_temp
    - Memory метрики: ram_usage, swap_usage, cache_hits
    - Network метрики: bandwidth, latency, packet_loss
    """
    np.random.seed(42)
    
    # CPU метрики (корельовані між собою)
    cpu_usage = np.random.uniform(20, 80, n_samples)
    cpu_freq = 2.0 + cpu_usage * 0.03 + np.random.normal(0, 0.1, n_samples)
    cpu_temp = 40 + cpu_usage * 0.5 + np.random.normal(0, 2, n_samples)
    
    # Memory метрики (корельовані між собою)
    ram_usage = np.random.uniform(30, 70, n_samples)
    swap_usage = np.maximum(0, ram_usage - 50) * 2 + np.random.normal(0, 5, n_samples)
    cache_hits = 100 - ram_usage * 0.8 + np.random.normal(0, 5, n_samples)
    
    # Network метрики
    bandwidth = np.random.uniform(100, 1000, n_samples)
    latency = 50 + 1000 / bandwidth + np.random.normal(0, 5, n_samples)
    packet_loss = np.maximum(0, latency - 100) * 0.01 + np.random.normal(0, 0.1, n_samples)
    
    # Створення DataFrame
    data = pd.DataFrame({
        'cpu_usage': cpu_usage,
        'cpu_freq': cpu_freq,
        'cpu_temp': cpu_temp,
        'ram_usage': ram_usage,
        'swap_usage': swap_usage,
        'cache_hits': cache_hits,
        'bandwidth': bandwidth,
        'latency': latency,
        'packet_loss': packet_loss,
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='H')
    })
    
    return data


def seminar1_example():
    """Демонстрація Семінару 1: Виявлення інваріантів."""
    print("=" * 80)
    print("СЕМІНАР 1: Виявлення інваріантів")
    print("=" * 80)
    
    # Генерація даних
    data = generate_synthetic_data(1000)
    print(f"\n📊 Згенеровано {len(data)} зразків даних")
    
    # Створення об'єкта для виявлення інваріантів
    discovery = InvariantDiscovery(
        variance_threshold=0.95,
        correlation_threshold=0.93,
        overhead=0.1
    )
    
    # Додавання груп ознак
    discovery.add_feature_group('cpu_metrics', ['cpu_usage', 'cpu_freq', 'cpu_temp'])
    discovery.add_feature_group('memory_metrics', ['ram_usage', 'swap_usage', 'cache_hits'])
    discovery.add_feature_group('network_metrics', ['bandwidth', 'latency', 'packet_loss'])
    
    # Навчання моделі
    print("\n🔍 Пошук інваріантів...")
    invariants = discovery.fit(data)
    print(f"✅ Знайдено {len(invariants)} інваріантів")
    
    # Виведення зведення
    summary = discovery.get_invariants_summary()
    print("\n📋 Зведення інваріантів:")
    print(summary.to_string(index=False))
    
    # Детекція аномалій на нових даних
    print("\n🔎 Детекція аномалій...")
    # Створюємо аномальні дані
    anomaly_data = data.copy()
    anomaly_data.loc[0, 'cpu_usage'] = 95  # Аномальне значення
    anomaly_data.loc[0, 'ram_usage'] = 90  # Аномальне значення
    
    results = discovery.detect_anomalies(anomaly_data)
    print(f"⚠️  Виявлено {results['total_violations']} порушень інваріантів")
    
    if results['violations']:
        print("\nДеталі першого порушення:")
        violation = results['violations'][0]
        print(f"  - Residual: {violation['residual']:.4f}")
        print(f"  - Threshold: {violation['threshold']:.4f}")
        print(f"  - Correlation: {violation['correlation']:.4f}")
    
    return discovery, invariants, data


def seminar2_example(discovery, invariants, data):
    """Демонстрація Семінару 2: Діагностика через SHAP та LIME."""
    print("\n" + "=" * 80)
    print("СЕМІНАР 2: Діагностика порушень інваріантів")
    print("=" * 80)
    
    # Створення діагностики
    diagnostics = InvariantDiagnostics()
    
    # Створюємо дані з порушенням
    anomaly_data = data.copy()
    anomaly_data.loc[0, 'cpu_usage'] = 95
    anomaly_data.loc[0, 'ram_usage'] = 90
    
    # Детекція порушень
    results = discovery.detect_anomalies(anomaly_data)
    
    if results['violations']:
        violation = results['violations'][0]
        invariant = invariants[violation.get('invariant_id', 0) % len(invariants)]
        
        print(f"\n🔍 Діагностика порушення інваріанта...")
        print(f"   Correlation: {invariant.correlation:.4f}")
        print(f"   Residual: {violation['residual']:.4f}")
        
        # SHAP пояснення
        print("\n📊 SHAP аналіз:")
        shap_result = diagnostics.diagnose_violation(
            invariant,
            anomaly_data,
            violation['row_index'],
            background_data=data,
            method='shap'
        )
        
        print("\nТоп-5 ознак, що внесли найбільший внесок:")
        for i, contributor in enumerate(shap_result['top_contributors'][:5], 1):
            print(f"  {i}. {contributor['feature']}: {contributor['contribution']:.4f}")
        
        # LIME пояснення
        print("\n🔬 LIME аналіз:")
        lime_result = diagnostics.diagnose_violation(
            invariant,
            anomaly_data,
            violation['row_index'],
            background_data=data,
            method='lime'
        )
        
        print("\nТоп-5 найважливіших ознак:")
        for i, contributor in enumerate(lime_result['top_contributors'][:5], 1):
            print(f"  {i}. {contributor['feature']}: "
                  f"importance={contributor['importance']:.4f}, "
                  f"coef={contributor['coefficient']:.4f}")
        
        return diagnostics, invariant, violation, anomaly_data
    
    return diagnostics, None, None, anomaly_data


def seminar3_example(discovery, invariant, violation, anomaly_data, data):
    """Демонстрація Семінару 3: Прескриптивний аналіз."""
    print("\n" + "=" * 80)
    print("СЕМІНАР 3: Прескриптивний аналіз та контрфактуальна терапія")
    print("=" * 80)
    
    if invariant is None or violation is None:
        print("⚠️  Немає порушень для аналізу")
        return
    
    # Створення PerformanceDoctor з Causal Awareness
    doctor = PerformanceDoctor(
        actionable_features=['cpu_usage', 'ram_usage', 'cpu_freq', 'bandwidth'],
        non_actionable_features=['timestamp'],
        feature_bounds={
            'cpu_usage': (0, 100),
            'ram_usage': (0, 100),
            'cpu_freq': (1.0, 4.0),
            'bandwidth': (10, 2000)
        },
        n_counterfactuals=3,
        # Causal graph: визначаємо причинно-наслідкові зв'язки
        causal_graph={
            'cpu_usage': ['latency'],  # cpu_usage викликає latency
            'ram_usage': ['latency'],  # ram_usage викликає latency
            'bandwidth': ['latency'],   # bandwidth викликає latency
            # Примітка: якщо є confounders (наприклад, user_count),
            # їх потрібно додати в causal_graph
        },
        causal_threshold=0.1  # Поріг значущості causal effect
    )
    
    print("\n💊 Генерація рекомендацій...")
    
    # Створення прескрипції
    prescription = doctor.prescribe_solution(
        invariant,
        anomaly_data,
        violation['row_index'],
        background_data=data,
        diagnosis="Порушення інваріанта через аномальні значення CPU та RAM"
    )
    
    # Виведення рекомендації
    message = doctor.format_prescription_message(prescription)
    print("\n" + message)
    
    # Деталі контрфактуалів
    print("\n📋 Деталі рекомендованих змін:")
    for feature, change_pct in prescription.recommended_changes.items():
        current_val = anomaly_data.loc[violation['row_index'], feature]
        new_val = current_val * (1 + change_pct / 100)
        print(f"  • {feature}: {current_val:.2f} → {new_val:.2f} "
              f"({change_pct:+.1f}%)")
    
    # Causal awareness інформація
    if prescription.causal_validity is not None:
        print(f"\n🔬 Causal Validity: {prescription.causal_validity*100:.1f}%")
        if prescription.causal_effects:
            print("\n📊 Causal Effects (чи дійсно зміни спрацюють):")
            for feature, effect in prescription.causal_effects.items():
                change = prescription.recommended_changes.get(feature, 0)
                status = "✅" if abs(effect) >= 0.1 else "⚠️"
                print(f"  {status} {feature}: effect={effect:.4f} (зміна: {change:+.1f}%)")
        
        if prescription.causal_warnings:
            print("\n⚠️ Causal Warnings:")
            for warning in prescription.causal_warnings:
                print(f"  {warning}")
    
    # Stability evaluation інформація
    if prescription.stability_report is not None:
        sr = prescription.stability_report
        print(f"\n🔬 Stability Evaluation (Robustness):")
        print(f"  • Robustness Score: {sr.robustness_score*100:.1f}% {'✅' if sr.is_stable else '⚠️'}")
        print(f"  • CV Score: {sr.cv_score:.3f} {'✅' if sr.cv_score < 0.2 else '⚠️'}")
        print(f"  • Rank Stability: {sr.rank_stability:.3f} {'✅' if sr.rank_stability >= 0.8 else '⚠️'}")
        print(f"  • Agreement (SHAP/LIME): {sr.agreement_score:.3f} {'✅' if sr.agreement_score >= 0.6 else '⚠️'}")
        print(f"  • Status: {'✅ Стабільне' if sr.is_stable else '⚠️ Нестабільне'}")
        
        if prescription.stability_warnings:
            print("\n⚠️ Stability Warnings:")
            for warning in prescription.stability_warnings:
                print(f"  {warning}")
    elif prescription.stability_warnings:
        print("\n⚠️ Stability Warnings:")
        for warning in prescription.stability_warnings:
            print(f"  {warning}")
    
    if prescription.alternative_solutions:
        print(f"\n🔄 Альтернативні рішення ({len(prescription.alternative_solutions)}):")
        for i, alt_cf in enumerate(prescription.alternative_solutions[:2], 1):
            print(f"\n  Варіант {i}:")
            for feature, change_pct in alt_cf.relative_changes.items():
                print(f"    • {feature}: {change_pct:+.1f}%")


def main():
    """Головна функція для запуску всіх семінарів."""
    print("\n" + "=" * 80)
    print("СИСТЕМА МОНІТОРИНГУ IT-ІНФРАСТРУКТУР")
    print("Від виявлення інваріантів до прескриптивних рекомендацій")
    print("=" * 80)
    
    try:
        # Семінар 1
        discovery, invariants, data = seminar1_example()
        
        # Семінар 2
        diagnostics, invariant, violation, anomaly_data = seminar2_example(
            discovery, invariants, data
        )
        
        # Семінар 3
        seminar3_example(discovery, invariant, violation, anomaly_data, data)
        
        print("\n" + "=" * 80)
        print("✅ Всі семінари успішно виконані!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Помилка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

