# visualization/plotter.py
import matplotlib.pyplot as plt
# Importante: Usar el backend de Qt para FigureCanvas
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import numpy as np

try:
    from ..ag_core.function_parser import safe_eval_function
except ImportError: 
    # Para pruebas directas de este módulo, si function_parser.py está en una ruta accesible
    # o si se ajusta el PYTHONPATH. Asumir que está en el mismo nivel para prueba directa.
    # from ag_core.function_parser import safe_eval_function
    # Para pruebas, es mejor comentar la importación relativa si se ejecuta solo este archivo
    # y definir una función dummy si es necesario.
    # O asegurar que la estructura de carpetas permita la importación relativa al ejecutar directamente.
    # Por ahora, la dejaremos así asumiendo que la ejecución principal es desde main_app.py
    pass # Opcional: from ag_core.function_parser import safe_eval_function si se configura bien para pruebas


# --- Funciones específicas para la integración con PySide6 ---

def update_fitness_plot_qt(mpl_canvas: FigureCanvas, ga_optimizer_instance):
    """
    Actualiza el gráfico de evolución de la aptitud en un MplCanvas de Qt.
    :param mpl_canvas: La instancia de MplCanvas (de ui.main_window) donde se dibujará.
    :param ga_optimizer_instance: La instancia de GeneticOptimizer que contiene el historial.
    """
    if mpl_canvas is None or mpl_canvas.axes is None:
        print("Error: Canvas de fitness no proporcionado o no inicializado.")
        return
    
    ax = mpl_canvas.axes # Acceder a los ejes del canvas
    ax.clear() # Limpiar los ejes antes de redibujar

    if ga_optimizer_instance is None or not hasattr(ga_optimizer_instance, 'best_solution_fitness_history') or \
       not ga_optimizer_instance.best_solution_fitness_history: # Comprobar también si la lista está vacía
        # Mostrar mensaje si no hay datos o instancia
        ax.text(0.5, 0.5, "No hay datos de fitness", ha='center', va='center', transform=ax.transAxes)
    else:
        fitness_history = np.array(ga_optimizer_instance.best_solution_fitness_history)
        if not fitness_history.size: 
            ax.text(0.5, 0.5, "Esperando datos...", ha='center', va='center', transform=ax.transAxes)
        else:
            generations = np.arange(len(fitness_history))
            ax.plot(generations, fitness_history, marker='.', linestyle='-', color='dodgerblue', label='Mejor Aptitud (Real)')
            
            if len(fitness_history) > 0:
                if ga_optimizer_instance.optimization_type == "maximize":
                    best_gen_idx = np.argmax(fitness_history)
                else: 
                    best_gen_idx = np.argmin(fitness_history)
                ax.plot(best_gen_idx, fitness_history[best_gen_idx], '*', markersize=10, color='red', label='Mejor Global Histórico')
            ax.legend(fontsize='small')

    ax.set_xlabel("Generación")
    ax.set_ylabel("Aptitud")
    ax.grid(True, linestyle=':', alpha=0.7)
    # El suptitle se maneja en la clase MplCanvas, no es necesario establecerlo aquí cada vez.
    # if mpl_canvas.fig._suptitle: 
    #      mpl_canvas.fig.suptitle(mpl_canvas.fig._suptitle.get_text(), fontsize=10, fontweight='bold')
    
    # Asegurar que el layout se ajuste bien
    try:
        mpl_canvas.fig.tight_layout(rect=[0, 0.03, 1, 0.95] if mpl_canvas.fig._suptitle else None)
    except Exception: # tight_layout a veces puede fallar si el gráfico está muy vacío
        pass
    mpl_canvas.draw_idle()


def update_population_plot_qt(mpl_canvas: FigureCanvas, ga_instance_snapshot, func_str, x_range):
    """
    Actualiza el gráfico de población en un MplCanvas de Qt.
    :param mpl_canvas: La instancia de MplCanvas donde se dibujará.
    :param ga_instance_snapshot: La instancia de PyGAD de la generación actual.
    :param func_str: La función objetivo como string.
    :param x_range: Tupla (min_x, max_x).
    """
    if mpl_canvas is None or mpl_canvas.axes is None:
        print("Error: Canvas de población no proporcionado o no inicializado.")
        return

    ax = mpl_canvas.axes
    ax.clear()

    if ga_instance_snapshot is None or not func_str or not x_range :
        ax.text(0.5, 0.5, "No hay datos de población", ha='center', va='center', transform=ax.transAxes)
        ax.set_xlabel("Valor de x")
        ax.set_ylabel("Valor de f(x)")
        ax.grid(True, linestyle=':', alpha=0.7)
        # if mpl_canvas.fig._suptitle: 
        #      mpl_canvas.fig.suptitle(mpl_canvas.fig._suptitle.get_text(), fontsize=10, fontweight='bold')
        try:
            mpl_canvas.fig.tight_layout(rect=[0, 0.03, 1, 0.95] if mpl_canvas.fig._suptitle else None)
        except Exception:
            pass
        mpl_canvas.draw_idle()
        return

    generation_num = ga_instance_snapshot.generations_completed

    # Graficar la función objetivo
    x_func_vals = np.linspace(x_range[0], x_range[1], 200)
    y_func_vals = []
    for x_val_f in x_func_vals:
        try:
            y_func_vals.append(safe_eval_function(func_str, x_val_f))
        except (ValueError, TypeError): # Capturar TypeError también por si acaso
            y_func_vals.append(np.nan)
    ax.plot(x_func_vals, y_func_vals, color='darkgrey', linestyle='--', linewidth=1.5, label="f(x)")

    # Obtener la población actual
    population = ga_instance_snapshot.population
    pop_y_evaluated = [] 

    if population is not None and len(population) > 0:
        pop_x_coords = np.array([sol[0] for sol in population])
        current_pop_y_coords = []
        for x_val_p in pop_x_coords:
            try:
                y_val_p = safe_eval_function(func_str, x_val_p)
                current_pop_y_coords.append(y_val_p)
                if not np.isnan(y_val_p): pop_y_evaluated.append(y_val_p)
            except (ValueError, TypeError):
                current_pop_y_coords.append(np.nan)
        
        ax.scatter(pop_x_coords, current_pop_y_coords, color='deepskyblue', s=25, alpha=0.8, edgecolors='black', linewidth=0.5, label=f"Población")

    # Marcar la mejor solución de la generación actual
    if ga_instance_snapshot.best_solution_generation != -1: 
        best_sol_genes, _, _ = ga_instance_snapshot.best_solution()
        if best_sol_genes is not None and len(best_sol_genes) > 0:
            best_sol_x_coord = best_sol_genes[0]
            try:
                best_sol_y_coord = safe_eval_function(func_str, best_sol_x_coord)
                if not np.isnan(best_sol_y_coord): pop_y_evaluated.append(best_sol_y_coord)
                ax.scatter([best_sol_x_coord], [best_sol_y_coord], color='crimson', s=80, marker='*', zorder=5, edgecolors='black', label="Mejor de Gen.")
            except (ValueError, TypeError):
                pass 

    ax.set_xlabel("Valor de x")
    ax.set_ylabel("Valor de f(x)")
    # El suptitle ya está, podemos añadir un título de generación
    ax.set_title(f"Generación: {generation_num}", fontsize=9, loc='center') 
    # if mpl_canvas.fig._suptitle: 
    #     mpl_canvas.fig.suptitle(mpl_canvas.fig._suptitle.get_text(), fontsize=10, fontweight='bold')
    
    ax.legend(fontsize='x-small', loc='best') 
    ax.grid(True, linestyle=':', alpha=0.7)
    
    valid_y_func_for_limits = [y for y in y_func_vals if y is not None and not np.isnan(y)]
    all_valid_y_for_limits = valid_y_func_for_limits + pop_y_evaluated
    if all_valid_y_for_limits:
        min_y, max_y = np.min(all_valid_y_for_limits), np.max(all_valid_y_for_limits)
        if abs(max_y - min_y) < 1e-9: # Si el rango es muy pequeño o cero
            padding = 0.5 
        else: 
            padding = (max_y - min_y) * 0.15
        ax.set_ylim(min_y - padding, max_y + padding)
    ax.set_xlim(x_range[0], x_range[1])

    try:
        mpl_canvas.fig.tight_layout(rect=[0, 0.03, 1, 0.95] if mpl_canvas.fig._suptitle else None)
    except Exception:
        pass
    mpl_canvas.draw_idle()


def clear_plots_qt(fitness_canvas: FigureCanvas, population_canvas: FigureCanvas):
    """Limpia ambos gráficos de Matplotlib en los canvas de Qt."""
    canvases_details = [
        (fitness_canvas, "Generación", "Aptitud", "Evolución de la Aptitud"),
        (population_canvas, "Valor de x", "Valor de f(x)", "Población en Función Objetivo")
    ]

    for canvas, x_label, y_label, suptitle_text in canvases_details:
        if canvas and canvas.axes:
            canvas.axes.clear()
            canvas.axes.set_xlabel(x_label)
            canvas.axes.set_ylabel(y_label)
            # Restaurar el suptitle que se definió al crear MplCanvas (si existe en MplCanvas.fig._suptitle)
            # O usar el texto por defecto que pasamos aquí
            actual_suptitle = suptitle_text
            if hasattr(canvas.fig, '_suptitle') and canvas.fig._suptitle is not None:
                actual_suptitle = canvas.fig._suptitle.get_text() # Intentar usar el original
            
            canvas.fig.suptitle(actual_suptitle, fontsize=10, fontweight='bold') # Siempre poner el suptitle
            canvas.axes.set_title("") # Limpiar cualquier título de generación específico
            canvas.axes.grid(True, linestyle=':', alpha=0.7)
            try:
                canvas.fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Ajustar para el suptitle
            except Exception:
                pass # Ignorar errores de tight_layout si el gráfico está muy vacío
            canvas.draw_idle()


def get_fitness_plot_fig_from_canvas(fitness_canvas: FigureCanvas):
    """Devuelve el objeto Figure del canvas de aptitud."""
    return fitness_canvas.fig if fitness_canvas else None

def get_population_plot_fig_from_canvas(population_canvas: FigureCanvas):
    """Devuelve el objeto Figure del canvas de población."""
    return population_canvas.fig if population_canvas else None


if __name__ == '__main__':
    print("Módulo plotter.py (versión Qt). Para probar, ejecutar la aplicación principal.")