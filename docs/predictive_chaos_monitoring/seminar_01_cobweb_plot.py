"""
СЕМІНАР 1: «Павутинна діаграма хаосу» (Cobweb Plot Lab)

Мета: Візуалізувати перехід від стабільності до хаосу в логістичному відображенні

Практика: Побудова павутинної діаграми (cobweb plot), де на графіку y=f(x) та y=x 
           відстежується траєкторія x_n

Завдання: Експериментально знайти значення r, при яких нерухома точка стає нестабільною,
          і на власні очі побачити появу 2-циклу та 4-циклу (біфуркації)

Результат: Розуміння «зони турбулентності», де класичні статичні пороги моніторингу 
           перестають працювати
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import warnings
warnings.filterwarnings('ignore')


class CobwebPlot:
    """
    Клас для побудови павутинної діаграми логістичного відображення.
    """
    
    def __init__(self, r: float, x0: float = 0.5, n_iterations: int = 50):
        """
        Ініціалізація параметрів.
        
        Parameters:
        -----------
        r : float
            Параметр логістичного відображення
        x0 : float
            Початкове значення
        n_iterations : int
            Кількість ітерацій для побудови
        """
        self.r = r
        self.x0 = x0
        self.n_iterations = n_iterations
    
    def logistic_map(self, x: float) -> float:
        """
        Логістичне відображення: f(x) = r * x * (1 - x)
        """
        return self.r * x * (1 - x)
    
    def derivative(self, x: float) -> float:
        """
        Похідна: f'(x) = r * (1 - 2*x)
        """
        return self.r * (1 - 2 * x)
    
    def find_fixed_points(self) -> list:
        """
        Знаходження нерухомих точок: x* = f(x*)
        Нерухома точка: x* = 0 або x* = 1 - 1/r
        """
        fixed_points = [0.0]
        if self.r > 1:
            fixed_points.append(1 - 1/self.r)
        return fixed_points
    
    def check_stability(self, x_star: float) -> tuple:
        """
        Перевірка стабільності нерухомої точки.
        Стабільна, якщо |f'(x*)| < 1
        """
        if x_star < 0 or x_star > 1:
            return False, None
        
        f_prime = abs(self.derivative(x_star))
        is_stable = f_prime < 1
        
        return is_stable, f_prime
    
    def generate_trajectory(self) -> np.ndarray:
        """
        Генерація траєкторії для павутинної діаграми.
        """
        trajectory = np.zeros(self.n_iterations * 2 + 1)
        trajectory[0] = self.x0
        
        x = self.x0
        for i in range(self.n_iterations):
            # Вертикальна лінія до f(x)
            trajectory[2*i + 1] = x
            y = self.logistic_map(x)
            trajectory[2*i + 2] = y
            # Горизонтальна лінія до y=x
            x = y
        
        return trajectory
    
    def plot(self, ax=None, show_fixed_points=True, show_stability=True):
        """
        Побудова павутинної діаграми.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        
        # Діапазон значень x
        x_range = np.linspace(0, 1, 1000)
        
        # Графік y = f(x)
        y_logistic = [self.logistic_map(x) for x in x_range]
        ax.plot(x_range, y_logistic, 'b-', linewidth=2, label=f'f(x) = {self.r:.2f}x(1-x)')
        
        # Графік y = x
        ax.plot(x_range, x_range, 'r--', linewidth=2, label='y = x', alpha=0.7)
        
        # Генерація траєкторії
        trajectory = self.generate_trajectory()
        
        # Побудова павутинної діаграми
        for i in range(0, len(trajectory) - 2, 2):
            x_start = trajectory[i]
            y_start = trajectory[i + 1] if i > 0 else self.x0
            x_end = trajectory[i + 1]
            y_end = trajectory[i + 1]
            
            # Вертикальна лінія
            if i > 0:
                ax.plot([x_start, x_start], [x_start, y_end], 'g-', alpha=0.5, linewidth=1)
            
            # Горизонтальна лінія
            ax.plot([x_start, y_end], [y_end, y_end], 'g-', alpha=0.5, linewidth=1)
        
        # Позначення початкової точки
        ax.plot(self.x0, 0, 'go', markersize=10, label=f'Початкова точка: x₀ = {self.x0:.2f}')
        
        # Нерухомі точки
        if show_fixed_points:
            fixed_points = self.find_fixed_points()
            for x_star in fixed_points:
                if 0 <= x_star <= 1:
                    y_star = self.logistic_map(x_star)
                    color = 'green' if show_stability else 'black'
                    marker = 'o' if show_stability else 'x'
                    
                    if show_stability:
                        is_stable, f_prime = self.check_stability(x_star)
                        color = 'green' if is_stable else 'red'
                        label_text = f'x* = {x_star:.3f} ({"стабільна" if is_stable else "нестабільна"}, |f\'|={f_prime:.3f})'
                    else:
                        label_text = f'x* = {x_star:.3f}'
                    
                    ax.plot(x_star, y_star, color=color, marker=marker, 
                           markersize=12, label=label_text, zorder=5)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('f(x)', fontsize=12)
        ax.set_title(f'Павутинна діаграма (Cobweb Plot)\nЛогістичне відображення: r = {self.r:.3f}', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=9)
        ax.set_aspect('equal')
        
        return ax


def find_bifurcation_points():
    """
    Експериментальне знаходження точок біфуркації.
    """
    print("="*80)
    print("ЕКСПЕРИМЕНТАЛЬНЕ ЗНАХОДЖЕННЯ ТОЧОК БІФУРКАЦІЇ")
    print("="*80)
    
    # Теоретичні значення біфуркацій
    r_values = {
        'Стабільна нерухома точка': 1.0,
        'Перша біфуркація (2-цикл)': 3.0,
        'Друга біфуркація (4-цикл)': 3.449,
        'Хаос': 3.56995,
        'Повний хаос': 3.9
    }
    
    print("\nТеоретичні значення параметра r:")
    for name, r in r_values.items():
        print(f"  {name}: r = {r:.5f}")
    
    return r_values


def main():
    """
    Головна функція семінару.
    """
    print("="*80)
    print("СЕМІНАР 1: ПАВУТИННА ДІАГРАМА ХАОСУ")
    print("="*80)
    
    # Знаходження точок біфуркації
    bifurcation_points = find_bifurcation_points()
    
    # Створення фігури з кількома підграфіками
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Стабільна нерухома точка (r = 2.5)
    ax1 = plt.subplot(2, 3, 1)
    cobweb1 = CobwebPlot(r=2.5, x0=0.3, n_iterations=20)
    cobweb1.plot(ax1)
    
    # 2. Перша біфуркація - 2-цикл (r = 3.2)
    ax2 = plt.subplot(2, 3, 2)
    cobweb2 = CobwebPlot(r=3.2, x0=0.3, n_iterations=30)
    cobweb2.plot(ax2)
    
    # 3. Друга біфуркація - 4-цикл (r = 3.5)
    ax3 = plt.subplot(2, 3, 3)
    cobweb3 = CobwebPlot(r=3.5, x0=0.3, n_iterations=40)
    cobweb3.plot(ax3)
    
    # 4. Перехід до хаосу (r = 3.6)
    ax4 = plt.subplot(2, 3, 4)
    cobweb4 = CobwebPlot(r=3.6, x0=0.3, n_iterations=50)
    cobweb4.plot(ax4)
    
    # 5. Хаос (r = 3.9)
    ax5 = plt.subplot(2, 3, 5)
    cobweb5 = CobwebPlot(r=3.9, x0=0.3, n_iterations=50)
    cobweb5.plot(ax5)
    
    # 6. Діаграма біфуркацій
    ax6 = plt.subplot(2, 3, 6)
    plot_bifurcation_diagram(ax6)
    
    plt.tight_layout()
    plt.savefig('seminar_01_cobweb_plot.png', dpi=300, bbox_inches='tight')
    print("\nГрафік збережено в файл: seminar_01_cobweb_plot.png")
    
    # Аналіз стабільності
    print("\n" + "="*80)
    print("АНАЛІЗ СТАБІЛЬНОСТІ НЕРУХОМИХ ТОЧОК")
    print("="*80)
    
    test_r_values = [2.0, 2.5, 3.0, 3.2, 3.5, 3.9]
    
    for r in test_r_values:
        cobweb = CobwebPlot(r=r)
        fixed_points = cobweb.find_fixed_points()
        
        print(f"\nr = {r:.2f}:")
        for x_star in fixed_points:
            if 0 <= x_star <= 1:
                is_stable, f_prime = cobweb.check_stability(x_star)
                status = "СТАБІЛЬНА" if is_stable else "НЕСТАБІЛЬНА"
                print(f"  x* = {x_star:.4f}: {status} (|f'(x*)| = {f_prime:.4f})")
    
    print("\n" + "="*80)
    print("ВИСНОВКИ:")
    print("="*80)
    print("1. При r < 3.0: нерухома точка стабільна (|f'(x*)| < 1)")
    print("2. При r = 3.0: перша біфуркація - з'являється 2-цикл")
    print("3. При r ≈ 3.449: друга біфуркація - з'являється 4-цикл")
    print("4. При r > 3.56995: настає хаос - траєкторія не періодична")
    print("5. У зоні хаосу статичні пороги моніторингу перестають працювати!")
    
    plt.show()


def plot_bifurcation_diagram(ax, r_min=2.5, r_max=4.0, n_r=1000, n_iter=200, n_skip=100):
    """
    Побудова діаграми біфуркацій.
    """
    r_values = np.linspace(r_min, r_max, n_r)
    x_values = []
    r_plot = []
    
    for r in r_values:
        x = 0.5
        # Пропускаємо перші ітерації (транзієнт)
        for _ in range(n_skip):
            x = r * x * (1 - x)
        # Зберігаємо наступні значення
        for _ in range(n_iter):
            x = r * x * (1 - x)
            x_values.append(x)
            r_plot.append(r)
    
    ax.plot(r_plot, x_values, ',k', alpha=0.25, markersize=0.5)
    ax.set_xlabel('r', fontsize=12)
    ax.set_ylabel('x', fontsize=12)
    ax.set_title('Діаграма біфуркацій', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Позначення ключових точок
    ax.axvline(x=3.0, color='r', linestyle='--', alpha=0.7, label='2-цикл (r=3.0)')
    ax.axvline(x=3.449, color='orange', linestyle='--', alpha=0.7, label='4-цикл (r=3.449)')
    ax.axvline(x=3.56995, color='purple', linestyle='--', alpha=0.7, label='Хаос (r=3.57)')
    ax.legend(fontsize=8)


if __name__ == "__main__":
    main()


