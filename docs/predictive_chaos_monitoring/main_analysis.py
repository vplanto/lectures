"""
Головний скрипт для генерації, аналізу та предиктивного тестування синтетичних датасетів.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataset_generators import ControlledChaosGenerator, PureRandomGenerator
from time_series_analyzer import TimeSeriesAnalyzer
from lstm_predictor import LSTMPredictor
import warnings
warnings.filterwarnings('ignore')

# Налаштування для відтворюваності
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Параметри генерації
N_POINTS = 1000
WINDOW_SIZE = 60

def main():
    """
    Головна функція для виконання повного аналізу.
    """
    print("="*80)
    print("ГЕНЕРАЦІЯ ТА АНАЛІЗ СИНТЕТИЧНИХ ДАТАСЕТІВ")
    print("="*80)
    
    # ========================================================================
    # ЗАВДАННЯ 1: ГЕНЕРАЦІЯ ДАТАСЕТІВ
    # ========================================================================
    print("\n" + "="*80)
    print("ЗАВДАННЯ 1: ГЕНЕРАЦІЯ ДАТАСЕТІВ")
    print("="*80)
    
    # Генерація Controlled Chaos
    print("\nГенерація датасету 'Controlled Chaos'...")
    chaos_gen = ControlledChaosGenerator(r=3.9, noise_std=0.01, random_seed=RANDOM_SEED)
    chaos_series = chaos_gen.generate(
        n_points=N_POINTS, 
        x0=0.5,
        anomaly_type='spike'
    )
    print(f"  Згенеровано {len(chaos_series)} точок")
    
    # Генерація Pure Random
    print("\nГенерація датасету 'Pure Random'...")
    random_gen = PureRandomGenerator(noise_std=0.1, random_seed=RANDOM_SEED+1)
    random_series = random_gen.generate(n_points=N_POINTS, x0=0.0)
    print(f"  Згенеровано {len(random_series)} точок")
    
    # ========================================================================
    # ЗАВДАННЯ 2: МАТЕМАТИЧНИЙ АНАЛІЗ
    # ========================================================================
    print("\n" + "="*80)
    print("ЗАВДАННЯ 2: МАТЕМАТИЧНИЙ АНАЛІЗ ТА ПОРІВНЯННЯ")
    print("="*80)
    
    analyzer = TimeSeriesAnalyzer()
    
    # Аналіз Controlled Chaos
    chaos_analysis = analyzer.analyze(chaos_series, "Controlled Chaos")
    
    # Аналіз Pure Random
    random_analysis = analyzer.analyze(random_series, "Pure Random")
    
    # ========================================================================
    # ЗАВДАННЯ 3: ПРЕДИКТИВНИЙ ТЕСТ (BASELINE)
    # ========================================================================
    print("\n" + "="*80)
    print("ЗАВДАННЯ 3: ПРЕДИКТИВНИЙ ТЕСТ (BASELINE)")
    print("="*80)
    
    # Навчання моделей
    print("\nНавчання LSTM моделі для 'Controlled Chaos'...")
    chaos_predictor = LSTMPredictor(window_size=WINDOW_SIZE, hidden_size=50, 
                                     num_layers=2, learning_rate=0.001)
    chaos_history = chaos_predictor.train(chaos_series, epochs=50, batch_size=32)
    
    print("\nНавчання LSTM моделі для 'Pure Random'...")
    random_predictor = LSTMPredictor(window_size=WINDOW_SIZE, hidden_size=50, 
                                      num_layers=2, learning_rate=0.001)
    random_history = random_predictor.train(random_series, epochs=50, batch_size=32)
    
    # Обчислення помилок реконструкції
    print("\nОбчислення помилок реконструкції...")
    chaos_errors, chaos_stats = chaos_predictor.compute_reconstruction_error(chaos_series)
    random_errors, random_stats = random_predictor.compute_reconstruction_error(random_series)
    
    print(f"\nСтатистика помилок реконструкції:")
    print(f"  Controlled Chaos:")
    print(f"    Середнє: {chaos_stats['mean']:.6f}")
    print(f"    Стандартне відхилення: {chaos_stats['std']:.6f}")
    print(f"    Поріг (μ + 3σ): {chaos_stats['threshold']:.6f}")
    print(f"    Виявлено аномалій: {chaos_stats['anomalies']}")
    
    print(f"  Pure Random:")
    print(f"    Середнє: {random_stats['mean']:.6f}")
    print(f"    Стандартне відхилення: {random_stats['std']:.6f}")
    print(f"    Поріг (μ + 3σ): {random_stats['threshold']:.6f}")
    print(f"    Виявлено аномалій: {random_stats['anomalies']}")
    
    # ========================================================================
    # ВІЗУАЛІЗАЦІЯ
    # ========================================================================
    print("\n" + "="*80)
    print("ПОБУДОВА ГРАФІКІВ")
    print("="*80)
    
    # Створення фігури з двома сабплотами
    fig = plt.figure(figsize=(16, 12))
    
    # Графік 1: Controlled Chaos - часовий ряд та помилки
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(chaos_series.index, chaos_series.values, 'b-', alpha=0.7, linewidth=0.8, label='Controlled Chaos')
    ax1.set_xlabel('Час (t)', fontsize=10)
    ax1.set_ylabel('Значення', fontsize=10)
    ax1.set_title('Controlled Chaos: Часовий ряд', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2 = plt.subplot(2, 2, 2)
    error_indices = np.arange(WINDOW_SIZE, len(chaos_series))
    ax2.plot(error_indices, chaos_errors, 'r-', alpha=0.7, linewidth=0.8, label='Помилка реконструкції')
    ax2.axhline(y=chaos_stats['threshold'], color='g', linestyle='--', 
                linewidth=2, label=f'Поріг (μ + 3σ = {chaos_stats["threshold"]:.4f})')
    ax2.fill_between(error_indices, chaos_stats['threshold'], chaos_errors.max(), 
                     where=(chaos_errors > chaos_stats['threshold']), 
                     alpha=0.3, color='red', label='Аномалії')
    ax2.set_xlabel('Час (t)', fontsize=10)
    ax2.set_ylabel('Помилка реконструкції', fontsize=10)
    ax2.set_title('Controlled Chaos: Помилка реконструкції', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Графік 2: Pure Random - часовий ряд та помилки
    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(random_series.index, random_series.values, 'purple', alpha=0.7, linewidth=0.8, label='Pure Random')
    ax3.set_xlabel('Час (t)', fontsize=10)
    ax3.set_ylabel('Значення', fontsize=10)
    ax3.set_title('Pure Random: Часовий ряд', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    ax4 = plt.subplot(2, 2, 4)
    error_indices_random = np.arange(WINDOW_SIZE, len(random_series))
    ax4.plot(error_indices_random, random_errors, 'orange', alpha=0.7, linewidth=0.8, label='Помилка реконструкції')
    ax4.axhline(y=random_stats['threshold'], color='g', linestyle='--', 
                linewidth=2, label=f'Поріг (μ + 3σ = {random_stats["threshold"]:.4f})')
    ax4.set_xlabel('Час (t)', fontsize=10)
    ax4.set_ylabel('Помилка реконструкції', fontsize=10)
    ax4.set_title('Pure Random: Помилка реконструкції (рівномірно висока)', 
                  fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('predictive_monitoring_analysis.png', dpi=300, bbox_inches='tight')
    print("\nГрафік збережено в файл: predictive_monitoring_analysis.png")
    
    # Додаткова візуалізація: ACF для обох рядів
    fig2, (ax5, ax6) = plt.subplots(1, 2, figsize=(14, 5))
    
    # ACF для Controlled Chaos
    chaos_lags, chaos_acf = chaos_analysis['acf']
    ax5.stem(chaos_lags[:20], chaos_acf[:20], basefmt=" ")
    ax5.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax5.axhline(y=0.1, color='r', linestyle='--', linewidth=1, alpha=0.5)
    ax5.axhline(y=-0.1, color='r', linestyle='--', linewidth=1, alpha=0.5)
    ax5.set_xlabel('Лаг', fontsize=10)
    ax5.set_ylabel('ACF', fontsize=10)
    ax5.set_title('Controlled Chaos: Автокореляційна функція', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # ACF для Pure Random
    random_lags, random_acf = random_analysis['acf']
    ax6.stem(random_lags[:20], random_acf[:20], basefmt=" ")
    ax6.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax6.axhline(y=0.1, color='r', linestyle='--', linewidth=1, alpha=0.5)
    ax6.axhline(y=-0.1, color='r', linestyle='--', linewidth=1, alpha=0.5)
    ax6.set_xlabel('Лаг', fontsize=10)
    ax6.set_ylabel('ACF', fontsize=10)
    ax6.set_title('Pure Random: Автокореляційна функція', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('acf_analysis.png', dpi=300, bbox_inches='tight')
    print("Графік ACF збережено в файл: acf_analysis.png")
    
    print("\n" + "="*80)
    print("АНАЛІЗ ЗАВЕРШЕНО")
    print("="*80)
    
    # Підсумок
    print("\nПІДСУМОК:")
    print(f"  Controlled Chaos:")
    print(f"    - Показник Херста: {chaos_analysis['hurst_exponent']:.4f} {'(має пам\'ять)' if chaos_analysis['hurst_exponent'] > 0.5 else '(близький до випадкового)'}")
    print(f"    - Стаціонарність: {'Так' if chaos_analysis['adf_test']['is_stationary'] else 'Ні'}")
    print(f"    - Аномалії виявлено: {chaos_stats['anomalies']} точок виходять за поріг μ+3σ")
    
    print(f"  Pure Random:")
    print(f"    - Показник Херста: {random_analysis['hurst_exponent']:.4f} {'(має пам\'ять)' if random_analysis['hurst_exponent'] > 0.5 else '(близький до випадкового)'}")
    print(f"    - Стаціонарність: {'Так' if random_analysis['adf_test']['is_stationary'] else 'Ні'}")
    print(f"    - Аномалії виявлено: {random_stats['anomalies']} точок (помилка рівномірно висока)")
    
    plt.show()


if __name__ == "__main__":
    main()


