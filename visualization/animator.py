import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

try:
    from ..ag_core.function_parser import safe_eval_function
except ImportError: # Para pruebas directas
    from ag_core.function_parser import safe_eval_function # Asume que está en el mismo nivel o PYTHONPATH

def create_ga_animation(population_history, func_str, x_range, optimization_type, best_solution_history, interval=200):
    """
    Crea una animación de la evolución del AG.
    - population_history: Lista de listas, donde cada sublista son los x-values de la población en una generación.
    - func_str: La función objetivo como string.
    - x_range: Tupla (min_x, max_x).
    - optimization_type: 'maximize' o 'minimize'.
    - best_solution_history: Lista de los mejores fitness reales por generación.
    - interval: Milisegundos entre frames.
    """
    if not population_history:
        print("Historial de población vacío, no se puede crear animación.")
        return None

    fig_anim, ax_anim = plt.subplots(figsize=(7, 5))
    plt.style.use('seaborn-v0_8-whitegrid')

    # Graficar la función objetivo una vez
    x_func = np.linspace(x_range[0], x_range[1], 300)
    y_func = []
    for x_val in x_func:
        try:
            y_func.append(safe_eval_function(func_str, x_val))
        except ValueError:
            y_func.append(np.nan)
    ax_anim.plot(x_func, y_func, color='darkgrey', linestyle='--', linewidth=1.5, label="f(x)")

    # Establecer límites fijos para los ejes para evitar reescalado en cada frame
    all_pop_x = [x for gen_pop in population_history for x in gen_pop]
    all_y_vals_for_limit = list(y_func) # Copiar
    if all_pop_x:
        for x_val in all_pop_x: # Solo para límites, no necesita ser super eficiente
            try: all_y_vals_for_limit.append(safe_eval_function(func_str, x_val))
            except: pass
    
    valid_y_for_limit = [y for y in all_y_vals_for_limit if y is not None and not np.isnan(y)]
    if valid_y_for_limit:
        min_y, max_y = np.min(valid_y_for_limit), np.max(valid_y_for_limit)
        padding = (max_y - min_y) * 0.15 if (max_y - min_y) > 1e-6 else 0.15
        ax_anim.set_ylim(min_y - padding, max_y + padding)
    ax_anim.set_xlim(x_range[0], x_range[1])

    # Elementos que se actualizarán
    pop_scatter = ax_anim.scatter([], [], color='deepskyblue', s=25, alpha=0.8, edgecolors='black', linewidth=0.5)
    best_gen_scatter = ax_anim.scatter([], [], color='crimson', s=80, marker='*', zorder=5, edgecolors='black')
    title_text = ax_anim.set_title("", fontsize=10)
    
    ax_anim.set_xlabel("Valor de x")
    ax_anim.set_ylabel("Valor de f(x)")
    ax_anim.legend(loc='upper right', fontsize='x-small')
    fig_anim.tight_layout()

    num_frames = len(population_history)

    def update_frame(gen_idx):
        current_pop_x = population_history[gen_idx]
        current_pop_y = []
        for x_val in current_pop_x:
            try:
                current_pop_y.append(safe_eval_function(func_str, x_val))
            except ValueError:
                current_pop_y.append(np.nan)
        
        pop_scatter.set_offsets(np.c_[current_pop_x, current_pop_y])
        
        # Marcar el mejor de esta generación (requiere recalcular o tenerlo guardado)
        # Por simplicidad, tomamos el x que produce el mejor fitness real de esta generación
        if best_solution_history and gen_idx < len(best_solution_history):
            # Encontrar el individuo que dio ese best_fitness_history[gen_idx]
            # Esto es un poco más complejo, ya que population_history solo tiene x.
            # Podríamos tomar el individuo con el f(x) más cercano al best_fitness_history[gen_idx]
            # O, si guardamos la población completa con sus fitness, sería más fácil.
            # Por ahora, vamos a encontrar el mejor de la población actual de la animación:
            
            valid_indices = [i for i, y in enumerate(current_pop_y) if y is not None and not np.isnan(y)]
            if valid_indices:
                current_pop_y_valid = np.array(current_pop_y)[valid_indices]
                current_pop_x_valid = np.array(current_pop_x)[valid_indices]

                if optimization_type == "maximize":
                    best_idx_in_gen = np.argmax(current_pop_y_valid)
                else:
                    best_idx_in_gen = np.argmin(current_pop_y_valid)
                
                best_x_this_gen = current_pop_x_valid[best_idx_in_gen]
                best_y_this_gen = current_pop_y_valid[best_idx_in_gen]
                best_gen_scatter.set_offsets(np.c_[[best_x_this_gen], [best_y_this_gen]])
            else:
                best_gen_scatter.set_offsets(np.c_[[], []])


        title_text.set_text(f"Población en Función Objetivo (Generación {gen_idx + 1}/{num_frames})")
        return pop_scatter, best_gen_scatter, title_text

    anim = FuncAnimation(fig_anim, update_frame, frames=num_frames, interval=interval, blit=False)
    # blit=False es a veces más robusto, especialmente si los elementos del título cambian.
    
    return fig_anim, anim # Devolver la figura también por si se quiere cerrar o manejar.

if __name__ == '__main__':
    # Prueba de la animación
    print("Probando módulo de animación...")
    # Simular datos
    hist_pop_test = [
        np.random.uniform(-5, 5, 20) for _ in range(10) # 10 generaciones, 20 individuos
    ]
    hist_best_fit_test = np.sin(np.linspace(0, np.pi, 10)) * 10 # Fitness simulado

    test_func = "x * np.sin(x) + 10"
    test_range = (-5, 5)
    test_opt_type = "maximize"

    if not hist_pop_test:
        print("Historial de prueba vacío.")
    else:
        fig, animation_obj = create_ga_animation(hist_pop_test, test_func, test_range, test_opt_type, hist_best_fit_test, interval=500)
        
        if animation_obj:
            try:
                # Para guardar, se necesita un writer (ffmpeg para mp4, pillow para gif)
                # animation_obj.save("test_animation.gif", writer='pillow', fps=5)
                # print("Animación de prueba guardada como test_animation.gif")
                plt.show() # Mostrar la animación
            except Exception as e:
                print(f"Error al mostrar/guardar animación de prueba: {e}")
                print("Asegúrate de tener 'Pillow' instalado para GIF o 'ffmpeg' para MP4 y configurado en tu PATH.")
        else:
            print("No se pudo crear el objeto de animación.")