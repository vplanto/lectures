"""
СЕМІНАР 2: «Сліпе тестування пам'яті» (Hurst Blind Test)

Мета: Навчити студентів відрізняти випадковий шум від сигналу з пам'яттю 
      за допомогою показника Херста

Практика: Викладач надає три анонімні датасети:
          - Чисте випадкове блукання (H=0.5)
          - Антиперсистентний ряд (H<0.5)
          - Метрика з реальною пам'яттю (H>0.7)

Завдання: Студенти мають реалізувати R/S аналіз та за нахилом прямої в 
          логарифмічних координатах визначити, який із процесів є передбачуваним,
          а який — «білим шумом»

Результат: Оволодіння стратегією фільтрації метрик, щоб не витрачати ресурси 
           на прогнозування хаотичного шуму
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class HurstAnalyzer:
    """
    Клас для обчислення показника Херста за допомогою R/S аналізу.
    """
    
    def __init__(self):
        """Ініціалізація аналізатора."""
        pass
    
    def compute_hurst(self, series: np.ndarray, max_lag: int = None, 
                     min_lag: int = 10, step: int = 5) -> Tuple[float, Dict]:
        """
        Обчислення показника Херста за допомогою R/S аналізу.
        
        Parameters:
        -----------
        series : np.ndarray
            Часовий ряд
        max_lag : int, optional
            Максимальний лаг
        min_lag : int
            Мінімальний лаг
        step : int
            Крок для лагів
        
        Returns:
        --------
        Tuple[float, Dict]
            (H, results) - показник Херста та детальні результати
        """
        n = len(series)
        if max_lag is None:
            max_lag = n // 4
        
        lags = []
        rs_values = []
        
        for lag in range(min_lag, min(max_lag, n // 2), step):
            n_subsets = n // lag
            if n_subsets < 2:
                continue
            
            rs_subset = []
            
            for i in range(n_subsets):
                subset = series[i * lag:(i + 1) * lag]
                
                if len(subset) < 2:
                    continue
                
                # Середнє значення
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
            return 0.5, {'lags': [], 'rs_values': [], 'log_lags': [], 'log_rs': []}
        
        # Логарифмічна регресія: log(R/S) = H * log(lag) + c
        log_lags = np.log(lags)
        log_rs = np.log(rs_values)
        
        # Лінійна регресія
        H = np.polyfit(log_lags, log_rs, 1)[0]
        
        results = {
            'lags': lags,
            'rs_values': rs_values,
            'log_lags': log_lags,
            'log_rs': log_rs,
            'H': H
        }
        
        return H, results


class DatasetGenerator:
    """
    Генератор анонімних датасетів для сліпого тестування.
    """
    
    @staticmethod
    def generate_random_walk(n: int = 1000, noise_std: float = 0.1, 
                            seed: int = None) -> np.ndarray:
        """
        Генерація чистого випадкового блукання (H ≈ 0.5).
        """
        if seed is not None:
            np.random.seed(seed)
        
        noise = np.random.normal(0, noise_std, n)
        series = np.cumsum(noise)
        return series
    
    @staticmethod
    def generate_antipersistent(n: int = 1000, seed: int = None) -> np.ndarray:
        """
        Генерація антиперсистентного ряду (H < 0.5).
        Використовуємо фракційне блукання з негативним параметром Херста.
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Спрощена модель антиперсистентного процесу
        # Кожне наступне значення має тенденцію "повертатися" до середнього
        series = np.zeros(n)
        mean_reversion = 0.3  # Сила повернення до середнього
        
        for i in range(1, n):
            series[i] = series[i-1] - mean_reversion * series[i-1] + np.random.normal(0, 0.1)
        
        return series
    
    @staticmethod
    def generate_persistent(n: int = 1000, seed: int = None) -> np.ndarray:
        """
        Генерація персистентного ряду з пам'яттю (H > 0.7).
        Використовуємо логістичне відображення (хаос) або фракційне блукання.
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Використовуємо логістичне відображення (хаос має пам'ять)
        r = 3.9
        series = np.zeros(n)
        series[0] = 0.5
        
        for i in range(1, n):
            series[i] = r * series[i-1] * (1 - series[i-1])
            # Додаємо слабкий шум
            series[i] += np.random.normal(0, 0.01)
            series[i] = np.clip(series[i], 0, 1)
        
        return series


def blind_test(datasets: Dict[str, np.ndarray], show_answers: bool = False):
    """
    Проведення сліпого тестування.
    
    Parameters:
    -----------
    datasets : Dict[str, np.ndarray]
        Словник з анонімними датасетами
    show_answers : bool
        Чи показувати правильні відповіді
    """
    print("="*80)
    print("СЛІПЕ ТЕСТУВАННЯ ПАМ'ЯТІ")
    print("="*80)
    print("\nВам надано три анонімні датасети.")
    print("Завдання: визначити, який з них:")
    print("  A) Чисте випадкове блукання (H ≈ 0.5)")
    print("  B) Антиперсистентний ряд (H < 0.5)")
    print("  C) Метрика з реальною пам'яттю (H > 0.7)")
    print("\n" + "-"*80)
    
    analyzer = HurstAnalyzer()
    results = {}
    
    # Аналіз кожного датасету
    for name, data in datasets.items():
        H, details = analyzer.compute_hurst(data)
        results[name] = {
            'H': H,
            'details': details,
            'data': data
        }
        
        print(f"\n{name}:")
        print(f"  Показник Херста: H = {H:.4f}")
        
        if H < 0.5:
            interpretation = "Антиперсистентний процес (тенденція до повернення)"
        elif H > 0.7:
            interpretation = "Персистентний процес з пам'яттю (передбачуваний)"
        else:
            interpretation = "Випадкове блукання (білий шум)"
        
        print(f"  Інтерпретація: {interpretation}")
    
    # Правильні відповіді
    if show_answers:
        print("\n" + "="*80)
        print("ПРАВИЛЬНІ ВІДПОВІДІ:")
        print("="*80)
        print("  Dataset_1: Чисте випадкове блукання (H ≈ 0.5)")
        print("  Dataset_2: Антиперсистентний ряд (H < 0.5)")
        print("  Dataset_3: Метрика з реальною пам'яттю (H > 0.7)")
    
    return results


def plot_analysis(results: Dict, save_path: str = 'seminar_02_hurst_blind_test.png'):
    """
    Візуалізація результатів аналізу.
    """
    fig = plt.figure(figsize=(18, 12))
    
    dataset_names = list(results.keys())
    
    for idx, name in enumerate(dataset_names):
        data = results[name]['data']
        H = results[name]['H']
        details = results[name]['details']
        
        # Графік 1: Часовий ряд
        ax1 = plt.subplot(3, 3, idx * 3 + 1)
        ax1.plot(data, linewidth=0.8, alpha=0.7)
        ax1.set_title(f'{name}\nЧасовий ряд', fontsize=11, fontweight='bold')
        ax1.set_xlabel('Час (t)')
        ax1.set_ylabel('Значення')
        ax1.grid(True, alpha=0.3)
        
        # Графік 2: R/S аналіз (логарифмічні координати)
        ax2 = plt.subplot(3, 3, idx * 3 + 2)
        if len(details['log_lags']) > 0:
            ax2.scatter(details['log_lags'], details['log_rs'], alpha=0.6, s=50)
            
            # Лінія регресії
            if len(details['log_lags']) >= 2:
                z = np.polyfit(details['log_lags'], details['log_rs'], 1)
                p = np.poly1d(z)
                ax2.plot(details['log_lags'], p(details['log_lags']), 
                        "r--", alpha=0.8, linewidth=2, 
                        label=f'H = {H:.4f}')
                ax2.legend()
        
        ax2.set_xlabel('log(lag)')
        ax2.set_ylabel('log(R/S)')
        ax2.set_title(f'R/S аналіз\nНахил = H = {H:.4f}', 
                     fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Графік 3: Інтерпретація
        ax3 = plt.subplot(3, 3, idx * 3 + 3)
        ax3.axis('off')
        
        # Визначення типу процесу
        if H < 0.5:
            process_type = "Антиперсистентний"
            color = 'orange'
            description = "Тенденція до повернення до середнього\n(mean-reverting)"
        elif H > 0.7:
            process_type = "Персистентний з пам'яттю"
            color = 'green'
            description = "Передбачуваний процес\n(можна використовувати для прогнозування)"
        else:
            process_type = "Випадкове блукання"
            color = 'blue'
            description = "Білий шум\n(не варто витрачати ресурси на прогнозування)"
        
        text = f"Показник Херста: H = {H:.4f}\n\n"
        text += f"Тип процесу:\n{process_type}\n\n"
        text += f"{description}"
        
        ax3.text(0.5, 0.5, text, ha='center', va='center', 
                fontsize=12, bbox=dict(boxstyle='round', facecolor=color, alpha=0.3),
                transform=ax3.transAxes)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nГрафік збережено в файл: {save_path}")


def main():
    """
    Головна функція семінару.
    """
    print("="*80)
    print("СЕМІНАР 2: СЛІПЕ ТЕСТУВАННЯ ПАМ'ЯТІ")
    print("="*80)
    
    # Генерація анонімних датасетів
    generator = DatasetGenerator()
    
    datasets = {
        'Dataset_1': generator.generate_random_walk(n=1000, seed=42),
        'Dataset_2': generator.generate_antipersistent(n=1000, seed=43),
        'Dataset_3': generator.generate_persistent(n=1000, seed=44)
    }
    
    # Проведення сліпого тестування
    results = blind_test(datasets, show_answers=True)
    
    # Візуалізація
    plot_analysis(results)
    
    # Висновки
    print("\n" + "="*80)
    print("ВИСНОВКИ ТА РЕКОМЕНДАЦІЇ:")
    print("="*80)
    print("\n1. Показник Херста H дозволяє відрізнити:")
    print("   - H ≈ 0.5: Випадкове блукання (білий шум)")
    print("   - H < 0.5: Антиперсистентний процес (mean-reverting)")
    print("   - H > 0.7: Персистентний процес з пам'яттю")
    
    print("\n2. Стратегія фільтрації метрик:")
    print("   - Для H > 0.7: Варто використовувати LSTM та інші методи прогнозування")
    print("   - Для H ≈ 0.5: Не варто витрачати ресурси на складні моделі")
    print("   - Для H < 0.5: Можна використовувати простіші методи (mean-reversion)")
    
    print("\n3. R/S аналіз:")
    print("   - Нахил прямої в логарифмічних координатах log(R/S) vs log(lag) = H")
    print("   - Чим більше H, тим більше пам'яті у процесі")
    
    plt.show()


if __name__ == "__main__":
    main()


