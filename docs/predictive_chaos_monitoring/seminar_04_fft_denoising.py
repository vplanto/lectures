"""
СЕМІНАР 4: «Спектральна чистка» (FFT Denoising)

Мета: Навчитися виділяти аномалії, «відсікаючи» сезонність у частотній області

Практика: Робота з метриками, що мають чітку денну та тижневу сезонність

Завдання: Застосувати Fast Fourier Transform (FFT), знайти домінуючі частоти,
          «обнулити» їх у спектрі та виконати зворотне перетворення (IFFT)

Результат: Візуалізація «оголеного» аномального сигналу, який раніше був 
           замаскований під щоденний пік навантаження
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')


class FFTDenoiser:
    """
    Клас для спектральної чистки часових рядів.
    """
    
    def __init__(self, sampling_rate: float = 1.0):
        """
        Ініціалізація денойзера.
        
        Parameters:
        -----------
        sampling_rate : float
            Частота дискретизації (наприклад, 1.0 для 1 точка на годину)
        """
        self.sampling_rate = sampling_rate
    
    def compute_fft(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Обчислення FFT сигналу.
        
        Parameters:
        -----------
        signal : np.ndarray
            Вхідний сигнал
        
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            (frequencies, fft_values) - частоти та FFT коефіцієнти
        """
        fft_values = fft(signal)
        frequencies = fftfreq(len(signal), 1.0 / self.sampling_rate)
        
        return frequencies, fft_values
    
    def find_dominant_frequencies(self, frequencies: np.ndarray, 
                                  fft_values: np.ndarray, 
                                  n_top: int = 5) -> List[Tuple[float, float, int]]:
        """
        Знаходження домінуючих частот.
        
        Parameters:
        -----------
        frequencies : np.ndarray
            Масив частот
        fft_values : np.ndarray
            FFT коефіцієнти
        n_top : int
            Кількість топових частот для повернення
        
        Returns:
        --------
        List[Tuple[float, float, int]]
            Список (частота, амплітуда, індекс) для топових частот
        """
        # Обчислення амплітуд
        amplitudes = np.abs(fft_values)
        
        # Індекси для позитивних частот
        positive_freq_idx = frequencies > 0
        
        # Знаходження топових частот
        top_indices = np.argsort(amplitudes[positive_freq_idx])[-n_top:][::-1]
        positive_freq = frequencies[positive_freq_idx]
        
        dominant = []
        for idx in top_indices:
            freq = positive_freq[idx]
            amp = amplitudes[positive_freq_idx][idx]
            dominant.append((freq, amp, idx))
        
        return dominant
    
    def remove_frequencies(self, fft_values: np.ndarray, 
                          frequencies: np.ndarray,
                          freq_to_remove: List[float],
                          bandwidth: float = 0.01) -> np.ndarray:
        """
        Видалення заданих частот зі спектру.
        
        Parameters:
        -----------
        fft_values : np.ndarray
            FFT коефіцієнти
        frequencies : np.ndarray
            Масив частот
        freq_to_remove : List[float]
            Список частот для видалення
        bandwidth : float
            Ширина смуги навколо кожної частоти для видалення
        
        Returns:
        --------
        np.ndarray
            Відфільтровані FFT коефіцієнти
        """
        filtered_fft = fft_values.copy()
        
        for freq in freq_to_remove:
            # Знаходження індексів частот у заданій смузі
            mask = np.abs(np.abs(frequencies) - abs(freq)) < bandwidth
            filtered_fft[mask] = 0
        
        return filtered_fft
    
    def denoise(self, signal: np.ndarray, 
               frequencies_to_remove: List[float] = None,
               remove_dominant: int = 0) -> Tuple[np.ndarray, dict]:
        """
        Повна процедура денойзингу.
        
        Parameters:
        -----------
        signal : np.ndarray
            Вхідний сигнал
        frequencies_to_remove : List[float], optional
            Список частот для видалення
        remove_dominant : int
            Кількість домінуючих частот для автоматичного видалення
        
        Returns:
        --------
        Tuple[np.ndarray, dict]
            (denoised_signal, info) - очищений сигнал та інформація
        """
        # FFT
        frequencies, fft_values = self.compute_fft(signal)
        
        # Знаходження домінуючих частот
        dominant = self.find_dominant_frequencies(frequencies, fft_values, n_top=10)
        
        # Визначення частот для видалення
        if frequencies_to_remove is None:
            frequencies_to_remove = []
        
        if remove_dominant > 0:
            for i in range(min(remove_dominant, len(dominant))):
                frequencies_to_remove.append(dominant[i][0])
        
        # Видалення частот
        filtered_fft = self.remove_frequencies(fft_values, frequencies, 
                                               frequencies_to_remove)
        
        # Зворотне перетворення
        denoised_signal = np.real(ifft(filtered_fft))
        
        info = {
            'frequencies': frequencies,
            'fft_values': fft_values,
            'filtered_fft': filtered_fft,
            'dominant_frequencies': dominant,
            'removed_frequencies': frequencies_to_remove
        }
        
        return denoised_signal, info


def generate_seasonal_data_with_anomaly(n_points: int = 1000, 
                                       daily_period: int = 24,
                                       weekly_period: int = 168,
                                       anomaly_start: int = 500,
                                       anomaly_duration: int = 50) -> np.ndarray:
    """
    Генерація даних з сезонністю та аномалією.
    
    Parameters:
    -----------
    n_points : int
        Кількість точок (години)
    daily_period : int
        Період денної сезонності (години)
    weekly_period : int
        Період тижневої сезонності (години)
    anomaly_start : int
        Початок аномалії
    anomaly_duration : int
        Тривалість аномалії
    
    Returns:
    --------
    np.ndarray
        Згенерований сигнал
    """
    t = np.arange(n_points)
    
    # Денна сезонність (24 години)
    daily = 0.5 * np.sin(2 * np.pi * t / daily_period)
    
    # Тижнева сезонність (168 годин = 7 днів)
    weekly = 0.3 * np.sin(2 * np.pi * t / weekly_period)
    
    # Базовий тренд
    trend = 0.1 * t / n_points
    
    # Шум
    noise = 0.05 * np.random.randn(n_points)
    
    # Аномалія (тихий сплеск, замаскований під сезонність)
    anomaly = np.zeros(n_points)
    anomaly[anomaly_start:anomaly_start + anomaly_duration] = \
        0.4 * np.sin(2 * np.pi * np.arange(anomaly_duration) / daily_period) + 0.3
    
    # Комбінація
    signal = 0.5 + daily + weekly + trend + noise + anomaly
    
    return signal


def main():
    """
    Головна функція семінару.
    """
    print("="*80)
    print("СЕМІНАР 4: СПЕКТРАЛЬНА ЧИСТКА (FFT DENOISING)")
    print("="*80)
    
    # Генерація даних з сезонністю та аномалією
    print("\nГенерація синтетичних даних...")
    print("  - Денна сезонність (період = 24 години)")
    print("  - Тижнева сезонність (період = 168 годин)")
    print("  - Тиха аномалія (замаскована під сезонність)")
    
    signal = generate_seasonal_data_with_anomaly(
        n_points=1000,
        daily_period=24,
        weekly_period=168,
        anomaly_start=500,
        anomaly_duration=50
    )
    
    # Створення денойзера
    denoiser = FFTDenoiser(sampling_rate=1.0)  # 1 точка на годину
    
    # Аналіз спектру
    frequencies, fft_values = denoiser.compute_fft(signal)
    dominant = denoiser.find_dominant_frequencies(frequencies, fft_values, n_top=5)
    
    print("\n" + "-"*80)
    print("ДОМІНУЮЧІ ЧАСТОТИ В СПЕКТРІ:")
    print("-"*80)
    for i, (freq, amp, idx) in enumerate(dominant, 1):
        period = 1.0 / freq if freq > 0 else np.inf
        print(f"{i}. Частота: {freq:.6f} (період: {period:.2f} годин, амплітуда: {amp:.4f})")
    
    # Денойзинг: видалення денної та тижневої сезонності
    print("\n" + "-"*80)
    print("ВИДАЛЕННЯ СЕЗОННОСТІ:")
    print("-"*80)
    
    # Частоти для видалення (денна та тижнева сезонність)
    daily_freq = 1.0 / 24.0
    weekly_freq = 1.0 / 168.0
    
    print(f"  Денна сезонність: частота = {daily_freq:.6f} (період = 24 години)")
    print(f"  Тижнева сезонність: частота = {weekly_freq:.6f} (період = 168 годин)")
    
    denoised_signal, info = denoiser.denoise(
        signal,
        frequencies_to_remove=[daily_freq, weekly_freq]
    )
    
    # Візуалізація
    fig = plt.figure(figsize=(18, 12))
    
    # Графік 1: Оригінальний сигнал
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(signal, 'b-', linewidth=1, alpha=0.7, label='Оригінальний сигнал')
    ax1.axvspan(500, 550, alpha=0.2, color='red', label='Аномалія (замаскована)')
    ax1.set_xlabel('Час (години)')
    ax1.set_ylabel('Значення')
    ax1.set_title('Оригінальний сигнал з сезонністю та аномалією', 
                  fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Графік 2: Спектр (FFT)
    ax2 = plt.subplot(3, 2, 2)
    positive_freq = frequencies > 0
    ax2.plot(frequencies[positive_freq], np.abs(fft_values[positive_freq]), 
            'b-', linewidth=1, alpha=0.7)
    ax2.axvline(x=1.0/24.0, color='r', linestyle='--', alpha=0.7, label='Денна (1/24)')
    ax2.axvline(x=1.0/168.0, color='orange', linestyle='--', alpha=0.7, label='Тижнева (1/168)')
    ax2.set_xlabel('Частота (1/година)')
    ax2.set_ylabel('Амплітуда')
    ax2.set_title('Спектр сигналу (FFT)', fontweight='bold')
    ax2.set_xlim(0, 0.1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Графік 3: Відфільтрований спектр
    ax3 = plt.subplot(3, 2, 3)
    ax3.plot(frequencies[positive_freq], np.abs(info['filtered_fft'][positive_freq]), 
            'g-', linewidth=1, alpha=0.7, label='Відфільтрований спектр')
    ax3.axvline(x=1.0/24.0, color='r', linestyle='--', alpha=0.3)
    ax3.axvline(x=1.0/168.0, color='orange', linestyle='--', alpha=0.3)
    ax3.set_xlabel('Частота (1/година)')
    ax3.set_ylabel('Амплітуда')
    ax3.set_title('Відфільтрований спектр (сезонність видалена)', 
                  fontweight='bold')
    ax3.set_xlim(0, 0.1)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Графік 4: Очищений сигнал
    ax4 = plt.subplot(3, 2, 4)
    ax4.plot(denoised_signal, 'g-', linewidth=1.5, alpha=0.8, label='Очищений сигнал')
    ax4.axvspan(500, 550, alpha=0.3, color='red', label='Аномалія (тепер видна!)')
    ax4.set_xlabel('Час (години)')
    ax4.set_ylabel('Значення')
    ax4.set_title('Очищений сигнал (аномалія виявлена)', 
                  fontweight='bold', color='green')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Графік 5: Порівняння
    ax5 = plt.subplot(3, 2, 5)
    ax5.plot(signal, 'b-', linewidth=1, alpha=0.5, label='Оригінальний')
    ax5.plot(denoised_signal, 'g-', linewidth=1.5, alpha=0.8, label='Очищений')
    ax5.axvspan(500, 550, alpha=0.2, color='red')
    ax5.set_xlabel('Час (години)')
    ax5.set_ylabel('Значення')
    ax5.set_title('Порівняння: Оригінальний vs Очищений', fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Графік 6: Виділена аномалія
    ax6 = plt.subplot(3, 2, 6)
    # Виділення аномалії як різниця між очищеним сигналом та базовим рівнем
    baseline = np.median(denoised_signal)
    anomaly_signal = denoised_signal - baseline
    ax6.plot(anomaly_signal, 'r-', linewidth=2, alpha=0.8, label='Виділена аномалія')
    ax6.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax6.axvspan(500, 550, alpha=0.2, color='red')
    ax6.set_xlabel('Час (години)')
    ax6.set_ylabel('Відхилення від базового рівня')
    ax6.set_title('Виділена аномалія (оголений сигнал)', 
                  fontweight='bold', color='red')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('seminar_04_fft_denoising.png', dpi=300, bbox_inches='tight')
    print("\nГрафік збережено в файл: seminar_04_fft_denoising.png")
    
    # Аналіз результатів
    print("\n" + "="*80)
    print("АНАЛІЗ РЕЗУЛЬТАТІВ")
    print("="*80)
    
    # Виявлення аномалії
    anomaly_region = denoised_signal[500:550]
    normal_region = np.concatenate([denoised_signal[:500], denoised_signal[550:]])
    
    anomaly_mean = np.mean(anomaly_region)
    normal_mean = np.mean(normal_region)
    normal_std = np.std(normal_region)
    
    print(f"\nСтатистика:")
    print(f"  Нормальний рівень: {normal_mean:.4f} ± {normal_std:.4f}")
    print(f"  Рівень аномалії: {anomaly_mean:.4f}")
    print(f"  Відхилення: {(anomaly_mean - normal_mean) / normal_std:.2f}σ")
    
    if abs(anomaly_mean - normal_mean) > 3 * normal_std:
        print(f"\n  ✓ Аномалія чітко виявлена! (відхилення > 3σ)")
    else:
        print(f"\n  ⚠ Аномалія може бути не помітна без спектральної чистки")
    
    # Висновки
    print("\n" + "="*80)
    print("ВИСНОВКИ")
    print("="*80)
    print("\n1. FFT дозволяє виділити домінуючі частоти в сигналі")
    print("2. Видалення сезонних частот 'оголює' аномальні сигнали")
    print("3. Тиха аномалія, замаскована під сезонність, стає видимою")
    print("4. Це дозволяє виявляти аномалії, які не видно в оригінальному сигналі")
    
    plt.show()


if __name__ == "__main__":
    main()


