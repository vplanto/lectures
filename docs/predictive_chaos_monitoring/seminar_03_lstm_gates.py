"""
СЕМІНАР 3: «Ручний вентиль LSTM» (Hydraulic Gates Sandbox)

Мета: Демостифікація роботи LSTM через покроковий прорахунок стану комірки

Практика: Створення інтерактивного блокнота, де студенти можуть вручну змінювати 
          значення Forget, Input та Output gates (від 0 до 1)

Завдання: Промоделювати сценарій інциденту: встановити вентилі так, щоб мережа 
          «забула» старий нормальний тренд і «запам'ятала» новий різкий сплеск 
          навантаження

Результат: Глибоке розуміння того, як лінійний потік у Cell State вирішує 
           проблему зникаючого градієнта
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')


class LSTMCellManual:
    """
    Ручна реалізація LSTM комірки для демонстрації роботи вентилів.
    """
    
    def __init__(self, input_size: int = 1, hidden_size: int = 1):
        """
        Ініціалізація LSTM комірки.
        
        Parameters:
        -----------
        input_size : int
            Розмір вхідного вектора
        hidden_size : int
            Розмір прихованого стану
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Стани
        self.cell_state = 0.0  # Cell State (лінійний потік)
        self.hidden_state = 0.0  # Hidden State (вихід)
        
        # Історія для візуалізації
        self.history = {
            'cell_state': [],
            'hidden_state': [],
            'forget_gate': [],
            'input_gate': [],
            'output_gate': [],
            'candidate': []
        }
    
    def step(self, x_t: float, 
             forget_gate: float = 0.5,
             input_gate: float = 0.5,
             output_gate: float = 0.5,
             candidate_value: float = None) -> Tuple[float, float]:
        """
        Один крок LSTM комірки.
        
        Формули:
        - f_t = forget_gate (що забути з попереднього Cell State)
        - i_t = input_gate (що додати з нового вхідного значення)
        - C̃_t = candidate_value (нове значення для Cell State)
        - C_t = f_t * C_{t-1} + i_t * C̃_t (оновлення Cell State)
        - o_t = output_gate (що вивести з Cell State)
        - h_t = o_t * tanh(C_t) (вихідний Hidden State)
        
        Parameters:
        -----------
        x_t : float
            Вхідне значення на кроці t
        forget_gate : float
            Значення Forget Gate (0-1)
        input_gate : float
            Значення Input Gate (0-1)
        output_gate : float
            Значення Output Gate (0-1)
        candidate_value : float, optional
            Кандидат на нове значення Cell State (якщо None, використовується x_t)
        
        Returns:
        --------
        Tuple[float, float]
            (cell_state, hidden_state) - нові стани
        """
        if candidate_value is None:
            candidate_value = x_t
        
        # Крок 1: Забути частину попереднього Cell State
        # C_t = f_t * C_{t-1} + ...
        self.cell_state = forget_gate * self.cell_state
        
        # Крок 2: Додати нову інформацію
        # C_t = ... + i_t * C̃_t
        self.cell_state = self.cell_state + input_gate * candidate_value
        
        # Крок 3: Обчислити вихідний Hidden State
        # h_t = o_t * tanh(C_t)
        self.hidden_state = output_gate * np.tanh(self.cell_state)
        
        # Збереження історії
        self.history['cell_state'].append(self.cell_state)
        self.history['hidden_state'].append(self.hidden_state)
        self.history['forget_gate'].append(forget_gate)
        self.history['input_gate'].append(input_gate)
        self.history['output_gate'].append(output_gate)
        self.history['candidate'].append(candidate_value)
        
        return self.cell_state, self.hidden_state
    
    def reset(self):
        """Скидання станів та історії."""
        self.cell_state = 0.0
        self.hidden_state = 0.0
        self.history = {
            'cell_state': [],
            'hidden_state': [],
            'forget_gate': [],
            'input_gate': [],
            'output_gate': [],
            'candidate': []
        }


def scenario_normal_then_spike():
    """
    Сценарій: Нормальний тренд, потім різкий сплеск навантаження.
    Завдання: Налаштувати вентилі так, щоб мережа забула старий тренд 
              і запам'ятала новий сплеск.
    """
    print("="*80)
    print("СЦЕНАРІЙ: НОРМАЛЬНИЙ ТРЕНД → РІЗКИЙ СПЛЕСК")
    print("="*80)
    print("\nСитуація:")
    print("  - t=0 до t=50: Нормальне навантаження (CPU = 30%)")
    print("  - t=50 до t=100: Різкий сплеск (CPU = 90%)")
    print("\nЗавдання:")
    print("  Налаштувати вентилі так, щоб LSTM:")
    print("  1. Забула старий нормальний тренд (Forget Gate → 0)")
    print("  2. Запам'ятала новий сплеск (Input Gate → 1)")
    print("  3. Вивела новий стан (Output Gate → 1)")
    
    # Генерація даних
    n_steps = 100
    normal_load = 0.3
    spike_load = 0.9
    
    inputs = np.concatenate([
        np.full(50, normal_load),
        np.full(50, spike_load)
    ])
    
    # Сценарій 1: Неправильні налаштування (не забуває старий тренд)
    print("\n" + "-"*80)
    print("СЦЕНАРІЙ 1: НЕПРАВИЛЬНІ НАЛАШТУВАННЯ")
    print("-"*80)
    print("Forget Gate = 0.9 (майже не забуває)")
    print("Input Gate = 0.5 (середнє запам'ятовування)")
    print("Output Gate = 0.8")
    
    lstm1 = LSTMCellManual()
    for x in inputs:
        lstm1.step(x, forget_gate=0.9, input_gate=0.5, output_gate=0.8)
    
    # Сценарій 2: Правильні налаштування (забуває старий, запам'ятовує новий)
    print("\n" + "-"*80)
    print("СЦЕНАРІЙ 2: ПРАВИЛЬНІ НАЛАШТУВАННЯ")
    print("-"*80)
    print("До сплеску (t<50): Forget=0.9, Input=0.3, Output=0.7 (нормальна робота)")
    print("Під час сплеску (t>=50): Forget=0.1, Input=0.9, Output=1.0 (реагує на зміну)")
    
    lstm2 = LSTMCellManual()
    for i, x in enumerate(inputs):
        if i < 50:
            # Нормальна робота
            lstm2.step(x, forget_gate=0.9, input_gate=0.3, output_gate=0.7)
        else:
            # Реакція на сплеск
            lstm2.step(x, forget_gate=0.1, input_gate=0.9, output_gate=1.0)
    
    # Візуалізація
    fig = plt.figure(figsize=(18, 10))
    
    # Графік 1: Вхідні дані
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(inputs, 'b-', linewidth=2, label='Вхід (CPU %)')
    ax1.axvline(x=50, color='r', linestyle='--', alpha=0.7, label='Початок сплеску')
    ax1.set_xlabel('Час (t)')
    ax1.set_ylabel('CPU Utilization')
    ax1.set_title('Вхідні дані: Нормальний тренд → Сплеск', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Графік 2: Cell State (неправильні налаштування)
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(lstm1.history['cell_state'], 'r-', linewidth=2, label='Cell State')
    ax2.axvline(x=50, color='r', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Час (t)')
    ax2.set_ylabel('Cell State')
    ax2.set_title('Неправильні налаштування:\nCell State (повільно адаптується)', 
                  fontweight='bold', color='red')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Графік 3: Hidden State (неправильні налаштування)
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(lstm1.history['hidden_state'], 'r-', linewidth=2, label='Hidden State')
    ax3.plot(inputs, 'b--', alpha=0.3, label='Вхід (для порівняння)')
    ax3.axvline(x=50, color='r', linestyle='--', alpha=0.7)
    ax3.set_xlabel('Час (t)')
    ax3.set_ylabel('Hidden State')
    ax3.set_title('Неправильні налаштування:\nHidden State (відстає від реальності)', 
                  fontweight='bold', color='red')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Графік 4: Cell State (правильні налаштування)
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(lstm2.history['cell_state'], 'g-', linewidth=2, label='Cell State')
    ax4.axvline(x=50, color='r', linestyle='--', alpha=0.7)
    ax4.set_xlabel('Час (t)')
    ax4.set_ylabel('Cell State')
    ax4.set_title('Правильні налаштування:\nCell State (швидко адаптується)', 
                  fontweight='bold', color='green')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Графік 5: Hidden State (правильні налаштування)
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(lstm2.history['hidden_state'], 'g-', linewidth=2, label='Hidden State')
    ax5.plot(inputs, 'b--', alpha=0.3, label='Вхід (для порівняння)')
    ax5.axvline(x=50, color='r', linestyle='--', alpha=0.7)
    ax5.set_xlabel('Час (t)')
    ax5.set_ylabel('Hidden State')
    ax5.set_title('Правильні налаштування:\nHidden State (відстежує реальність)', 
                  fontweight='bold', color='green')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Графік 6: Порівняння вентилів
    ax6 = plt.subplot(2, 3, 6)
    time_steps = np.arange(len(inputs))
    
    # Вентилі для правильного сценарію
    forget_gates = [0.9 if t < 50 else 0.1 for t in time_steps]
    input_gates = [0.3 if t < 50 else 0.9 for t in time_steps]
    output_gates = [0.7 if t < 50 else 1.0 for t in time_steps]
    
    ax6.plot(time_steps, forget_gates, 'r-', linewidth=2, label='Forget Gate', alpha=0.7)
    ax6.plot(time_steps, input_gates, 'b-', linewidth=2, label='Input Gate', alpha=0.7)
    ax6.plot(time_steps, output_gates, 'g-', linewidth=2, label='Output Gate', alpha=0.7)
    ax6.axvline(x=50, color='k', linestyle='--', alpha=0.7, label='Початок сплеску')
    ax6.set_xlabel('Час (t)')
    ax6.set_ylabel('Значення вентиля (0-1)')
    ax6.set_title('Динаміка вентилів\n(правильні налаштування)', fontweight='bold')
    ax6.set_ylim(0, 1.1)
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('seminar_03_lstm_gates.png', dpi=300, bbox_inches='tight')
    print("\nГрафік збережено в файл: seminar_03_lstm_gates.png")
    
    # Аналіз результатів
    print("\n" + "="*80)
    print("АНАЛІЗ РЕЗУЛЬТАТІВ")
    print("="*80)
    
    # Помилка передбачення
    error1 = np.mean(np.abs(np.array(lstm1.history['hidden_state']) - inputs))
    error2 = np.mean(np.abs(np.array(lstm2.history['hidden_state']) - inputs))
    
    print(f"\nСередня помилка передбачення:")
    print(f"  Неправильні налаштування: {error1:.4f}")
    print(f"  Правильні налаштування: {error2:.4f}")
    print(f"  Покращення: {(1 - error2/error1)*100:.1f}%")
    
    # Швидкість адаптації до сплеску
    adaptation1 = np.argmax(np.array(lstm1.history['hidden_state'][50:]) > 0.8) if np.any(np.array(lstm1.history['hidden_state'][50:]) > 0.8) else 50
    adaptation2 = np.argmax(np.array(lstm2.history['hidden_state'][50:]) > 0.8) if np.any(np.array(lstm2.history['hidden_state'][50:]) > 0.8) else 0
    
    print(f"\nШвидкість адаптації до сплеску:")
    print(f"  Неправильні налаштування: {adaptation1} кроків")
    print(f"  Правильні налаштування: {adaptation2} кроків")
    
    plt.show()


def demonstrate_gradient_flow():
    """
    Демонстрація того, як лінійний потік у Cell State вирішує проблему 
    зникаючого градієнта.
    """
    print("\n" + "="*80)
    print("ДЕМОНСТРАЦІЯ: ЛІНІЙНИЙ ПОТІК У CELL STATE")
    print("="*80)
    print("\nПроблема зникаючого градієнта в RNN:")
    print("  - При навчанні градієнти експоненційно зменшуються")
    print("  - Це ускладнює навчання довгих залежностей")
    print("\nРішення в LSTM:")
    print("  - Cell State має лінійний потік: C_t = f_t * C_{t-1} + i_t * C̃_t")
    print("  - Градієнти можуть протікати без згасання через Forget Gate")
    print("  - Це дозволяє зберігати інформацію на довгих відстанях")
    
    # Симуляція потоку градієнта
    n_steps = 100
    
    # RNN: градієнт експоненційно згасає
    rnn_gradient = np.exp(-0.1 * np.arange(n_steps))
    
    # LSTM: градієнт може зберігатися (якщо Forget Gate близький до 1)
    forget_gate = 0.95
    lstm_gradient = forget_gate ** np.arange(n_steps)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(rnn_gradient, 'r-', linewidth=2, label='RNN (експоненційне згасання)')
    ax.plot(lstm_gradient, 'g-', linewidth=2, label=f'LSTM (Forget Gate={forget_gate})')
    ax.set_xlabel('Кількість кроків назад')
    ax.set_ylabel('Відносна сила градієнта')
    ax.set_title('Порівняння потоку градієнта: RNN vs LSTM', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('seminar_03_gradient_flow.png', dpi=300, bbox_inches='tight')
    print("\nГрафік збережено в файл: seminar_03_gradient_flow.png")
    plt.show()


def interactive_experiment():
    """
    Інтерактивний експеримент: студенти можуть змінювати значення вентилів.
    """
    print("\n" + "="*80)
    print("ІНТЕРАКТИВНИЙ ЕКСПЕРИМЕНТ")
    print("="*80)
    print("\nСпробуйте різні комбінації вентилів:")
    print("  - Forget Gate: 0.0 (забути все) до 1.0 (зберегти все)")
    print("  - Input Gate: 0.0 (ігнорувати вхід) до 1.0 (запам'ятати вхід)")
    print("  - Output Gate: 0.0 (не виводити) до 1.0 (вивести все)")
    
    # Приклад: різні комбінації
    test_cases = [
        {'name': 'Зберегти все', 'forget': 1.0, 'input': 0.0, 'output': 1.0},
        {'name': 'Забути все', 'forget': 0.0, 'input': 1.0, 'output': 1.0},
        {'name': 'Не виводити', 'forget': 0.9, 'input': 0.5, 'output': 0.0},
        {'name': 'Реакція на зміну', 'forget': 0.1, 'input': 0.9, 'output': 1.0},
    ]
    
    inputs = np.concatenate([np.full(25, 0.3), np.full(25, 0.9)])
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, case in enumerate(test_cases):
        lstm = LSTMCellManual()
        for x in inputs:
            lstm.step(x, 
                     forget_gate=case['forget'],
                     input_gate=case['input'],
                     output_gate=case['output'])
        
        ax = axes[idx]
        ax.plot(inputs, 'b--', alpha=0.5, linewidth=2, label='Вхід')
        ax.plot(lstm.history['cell_state'], 'r-', linewidth=2, label='Cell State')
        ax.plot(lstm.history['hidden_state'], 'g-', linewidth=2, label='Hidden State')
        ax.axvline(x=25, color='k', linestyle='--', alpha=0.5)
        ax.set_title(f"{case['name']}\nF={case['forget']:.1f}, I={case['input']:.1f}, O={case['output']:.1f}", 
                    fontweight='bold')
        ax.set_xlabel('Час (t)')
        ax.set_ylabel('Значення')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('seminar_03_interactive_experiment.png', dpi=300, bbox_inches='tight')
    print("\nГрафік збережено в файл: seminar_03_interactive_experiment.png")
    plt.show()


def main():
    """
    Головна функція семінару.
    """
    print("="*80)
    print("СЕМІНАР 3: РУЧНИЙ ВЕНТИЛЬ LSTM")
    print("="*80)
    
    # Сценарій інциденту
    scenario_normal_then_spike()
    
    # Демонстрація потоку градієнта
    demonstrate_gradient_flow()
    
    # Інтерактивний експеримент
    interactive_experiment()
    
    # Висновки
    print("\n" + "="*80)
    print("ВИСНОВКИ")
    print("="*80)
    print("\n1. LSTM вентилі контролюють інформаційний потік:")
    print("   - Forget Gate: що забути з попереднього стану")
    print("   - Input Gate: що запам'ятати з нового вхідного значення")
    print("   - Output Gate: що вивести з Cell State")
    
    print("\n2. Cell State має лінійний потік:")
    print("   - C_t = f_t * C_{t-1} + i_t * C̃_t")
    print("   - Це вирішує проблему зникаючого градієнта")
    print("   - Дозволяє зберігати інформацію на довгих відстанях")
    
    print("\n3. Для виявлення аномалій:")
    print("   - При зміні тренду: Forget Gate → 0, Input Gate → 1")
    print("   - Це дозволяє швидко адаптуватися до нових умов")
    print("   - Hidden State відстежує реальність, а не старий тренд")


if __name__ == "__main__":
    main()


