# ui/main_window.py
import sys
from PySide6.QtCore import Qt, Slot, Signal, QObject 
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QRadioButton, QComboBox, QProgressBar,
    QTextEdit, QFrame, QGroupBox, QFileDialog, QMessageBox, QStatusBar, QSizePolicy,
    QSpacerItem
)
from PySide6.QtGui import QFont, QIcon
import numpy as np # Necesario para np.isnan en handle_generation_update

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class MplCanvas(FigureCanvas):
    """Clase base para un canvas de Matplotlib embebido en Qt."""
    def __init__(self, parent=None, width=5, height=4, dpi=100, suptitle=None):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        if suptitle:
            self.fig.suptitle(suptitle, fontsize=10, fontweight='bold')
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.updateGeometry()
        self.fig.tight_layout(rect=[0, 0.03, 1, 0.95] if suptitle else None)


class MainWindow(QMainWindow):
    # --- REFERENCIAS QUE EL CONTROLADOR LLENARÁ ---
    plotter_module = None
    ga_optimizer_instance = None 
    current_params_dict = None   
    best_solution_details_dict = None 
    app_state = {"running": False, "paused": False}

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Optimizador con Algoritmos Genéticos (PySide6) v0.2.4") # Nueva versión
        self.setGeometry(50, 50, 1300, 850)
        self._init_ui()

    def _init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_app_layout = QHBoxLayout(main_widget)

        left_column_widget = QWidget()
        left_v_layout = QVBoxLayout(left_column_widget)
        left_v_layout.setSpacing(10)

        opt_config_group = QGroupBox("Configuración de Optimización")
        opt_config_group.setFont(QFont("Arial", 10, QFont.Bold))
        opt_config_layout = QGridLayout(opt_config_group)
        self.rb_maximize = QRadioButton("Maximizar")
        self.rb_maximize.setChecked(True)
        self.rb_minimize = QRadioButton("Minimizar")
        opt_type_layout = QHBoxLayout()
        opt_type_layout.addWidget(self.rb_maximize)
        opt_type_layout.addWidget(self.rb_minimize)
        opt_type_layout.addStretch()
        opt_config_layout.addWidget(QLabel("Tipo Optimización:"), 0, 0, Qt.AlignmentFlag.AlignTop)
        opt_config_layout.addLayout(opt_type_layout, 0, 1)
        opt_config_layout.addWidget(QLabel("Función Objetivo f(x):"), 1, 0)
        self.le_func_str = QLineEdit("x * math.sin(x) + 10")
        self.le_func_str.setToolTip("Ej: x * math.cos(x) o x**2.\nUse 'x'. Funciones 'math.' y 'np.' disponibles (np.sin, np.cos, np.exp).")
        opt_config_layout.addWidget(self.le_func_str, 1, 1)
        opt_config_layout.addWidget(QLabel("Intervalo Búsqueda [min, max]:"), 2, 0)
        self.le_range_min = QLineEdit("-10")
        self.le_range_max = QLineEdit("10")
        range_input_layout = QHBoxLayout()
        range_input_layout.addWidget(self.le_range_min)
        range_input_layout.addWidget(QLabel("a"))
        range_input_layout.addWidget(self.le_range_max)
        opt_config_layout.addLayout(range_input_layout, 2, 1)
        left_v_layout.addWidget(opt_config_group)

        ga_params_group = QGroupBox("Parámetros del Algoritmo Genético")
        ga_params_group.setFont(QFont("Arial", 10, QFont.Bold))
        ga_params_layout = QGridLayout(ga_params_group)
        self.le_pop_size = QLineEdit("50")
        self.le_num_generations = QLineEdit("100")
        self.le_crossover_prob = QLineEdit("0.8")
        self.le_mutation_prob = QLineEdit("0.1")
        self.combo_selection_type = QComboBox()
        self.combo_selection_type.addItems(['sss', 'rws', 'sus', 'random', 'tournament', 'rank'])
        self.combo_selection_type.setCurrentText('sss')
        self.combo_crossover_type = QComboBox()
        self.combo_crossover_type.addItems(['single_point', 'two_points', 'uniform', 'scattered'])
        self.combo_crossover_type.setCurrentText('single_point')
        self.le_keep_elitism = QLineEdit("2")
        ga_params_layout.addWidget(QLabel("Tamaño Población (P₀):"), 0, 0); ga_params_layout.addWidget(self.le_pop_size, 0, 1)
        ga_params_layout.addWidget(QLabel("Núm. Máx. Generaciones:"), 1, 0); ga_params_layout.addWidget(self.le_num_generations, 1, 1)
        ga_params_layout.addWidget(QLabel("Prob. Cruce (Pc) [0-1]:"), 2, 0); ga_params_layout.addWidget(self.le_crossover_prob, 2, 1)
        ga_params_layout.addWidget(QLabel("Prob. Mutación (Pm) [0-1]:"), 3, 0); ga_params_layout.addWidget(self.le_mutation_prob, 3, 1)
        ga_params_layout.addWidget(QLabel("Método Selección:"), 4, 0); ga_params_layout.addWidget(self.combo_selection_type, 4, 1)
        ga_params_layout.addWidget(QLabel("Tipo Cruce:"), 5, 0); ga_params_layout.addWidget(self.combo_crossover_type, 5, 1)
        ga_params_layout.addWidget(QLabel("Elitismo (N mejores):"), 6, 0); ga_params_layout.addWidget(self.le_keep_elitism, 6, 1)
        left_v_layout.addWidget(ga_params_group)

        control_results_group = QGroupBox("Control y Resultados")
        control_results_group.setFont(QFont("Arial", 10, QFont.Bold))
        control_results_layout = QVBoxLayout(control_results_group)
        control_buttons_layout = QHBoxLayout()
        self.btn_start = QPushButton(" Iniciar"); control_buttons_layout.addWidget(self.btn_start)
        self.btn_pause = QPushButton(" Pausar"); control_buttons_layout.addWidget(self.btn_pause)
        self.btn_reset = QPushButton(" Reiniciar"); control_buttons_layout.addWidget(self.btn_reset)
        control_results_layout.addLayout(control_buttons_layout)
        progress_layout = QHBoxLayout()
        progress_layout.addWidget(QLabel("Progreso:"))
        self.progress_bar = QProgressBar(); self.progress_bar.setTextVisible(True); self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        control_results_layout.addLayout(progress_layout)
        control_results_layout.addWidget(QLabel("Mejor Solución Encontrada:"))
        self.te_best_solution_info = QTextEdit()
        self.te_best_solution_info.setReadOnly(True); self.te_best_solution_info.setFixedHeight(80)
        self.te_best_solution_info.setFont(QFont("Courier New", 9))
        control_results_layout.addWidget(self.te_best_solution_info)
        left_v_layout.addWidget(control_results_group)
        left_v_layout.addStretch(1)
        main_app_layout.addWidget(left_column_widget, 1)

        line = QFrame(); line.setFrameShape(QFrame.Shape.VLine); line.setFrameShadow(QFrame.Shadow.Sunken)
        main_app_layout.addWidget(line)

        right_column_widget = QWidget()
        right_v_layout = QVBoxLayout(right_column_widget)
        self.fitness_plot_canvas = MplCanvas(self, suptitle="Evolución de la Aptitud")
        self.population_plot_canvas = MplCanvas(self, suptitle="Población en Función Objetivo")
        right_v_layout.addWidget(self.fitness_plot_canvas, 1)
        right_v_layout.addWidget(self.population_plot_canvas, 1)

        export_group = QGroupBox("Exportación de Resultados")
        export_group.setFont(QFont("Arial", 10, QFont.Bold))
        export_layout = QHBoxLayout(export_group)
        self.btn_export_csv = QPushButton("Exportar CSV"); export_layout.addWidget(self.btn_export_csv)
        self.btn_export_pdf = QPushButton("Exportar PDF"); export_layout.addWidget(self.btn_export_pdf)
        self.btn_export_gif = QPushButton("Exportar GIF"); export_layout.addWidget(self.btn_export_gif)
        self.lbl_export_status = QLabel(""); export_layout.addWidget(self.lbl_export_status); export_layout.addStretch(1)
        right_v_layout.addWidget(export_group)

        console_group = QGroupBox("Consola de Salida")
        console_group.setFont(QFont("Arial", 10, QFont.Bold))
        console_layout = QVBoxLayout(console_group)
        self.te_console_output = QTextEdit()
        self.te_console_output.setReadOnly(True); self.te_console_output.setFont(QFont("Courier New", 8))
        self.te_console_output.setMinimumHeight(100)
        console_layout.addWidget(self.te_console_output)
        right_v_layout.addWidget(console_group)
        main_app_layout.addWidget(right_column_widget, 2)

        self.status_bar_widget = QStatusBar()
        self.setStatusBar(self.status_bar_widget)
        self.status_bar_widget.showMessage("Listo.")

    def get_parameters_from_gui(self):
        """Recolecta y valida parámetros de los widgets de la GUI."""
        print("[DEBUG MainWindow] get_parameters_from_gui llamado.") # NUEVO PRINT
        try:
            params = {
                "optimization_type": "maximize" if self.rb_maximize.isChecked() else "minimize",
                "func_str": self.le_func_str.text().strip(),
                "range_min": float(self.le_range_min.text()),
                "range_max": float(self.le_range_max.text()),
                "pop_size": int(self.le_pop_size.text()),
                "num_generations": int(self.le_num_generations.text()),
                "crossover_prob": float(self.le_crossover_prob.text()),
                "mutation_prob": float(self.le_mutation_prob.text()),
                "selection_type": self.combo_selection_type.currentText(),
                "crossover_type": self.combo_crossover_type.currentText(),
                "keep_elitism": int(self.le_keep_elitism.text())
            }
            # Validaciones
            if not params["func_str"]: raise ValueError("La función objetivo no puede estar vacía.")
            if params["range_min"] >= params["range_max"]: raise ValueError("El mínimo del intervalo debe ser menor que el máximo.")
            if not (10 <= params["pop_size"] <= 2000): raise ValueError("Tamaño de población debe estar entre 10 y 2000.")
            if not (1 <= params["num_generations"] <= 10000): raise ValueError("Número de generaciones debe estar entre 1 y 10000.")
            if not (0.0 <= params["crossover_prob"] <= 1.0): raise ValueError("Prob. cruce debe estar entre 0.0 y 1.0.")
            if not (0.0 <= params["mutation_prob"] <= 1.0): raise ValueError("Prob. mutación debe estar entre 0.0 y 1.0.")
            if not (0 <= params["keep_elitism"] < params["pop_size"]): raise ValueError("Elitismo debe ser >= 0 y menor que el tamaño de la población.")
            
            self.status_bar_widget.showMessage("Parámetros recolectados y validados.")
            print(f"[DEBUG MainWindow] Parámetros validados y devueltos: {params}") # NUEVO PRINT
            return params
            
        except ValueError as e: # Errores de conversión (float, int) o validación
            QMessageBox.warning(self, "Error en Parámetros", str(e))
            self.status_bar_widget.showMessage(f"Error en parámetros: {str(e)}")
            print(f"[DEBUG MainWindow] Error de validación en get_parameters_from_gui: {e}") # NUEVO PRINT
            return None # MUY IMPORTANTE: devolver None si hay error
        except Exception as e_gen: # Otros errores inesperados
            QMessageBox.critical(self, "Error Inesperado", f"Error al parsear parámetros: {str(e_gen)}")
            self.status_bar_widget.showMessage(f"Error parseando: {str(e_gen)}")
            print(f"[DEBUG MainWindow] Error inesperado en get_parameters_from_gui: {e_gen}") # NUEVO PRINT
            return None # MUY IMPORTANTE: devolver None si hay error

    @Slot(object)
    def handle_generation_update(self, ga_instance_snapshot):
        current_params = self.current_params_dict
        best_solution_global = self.best_solution_details_dict
        optimizer_ref = self.ga_optimizer_instance

        # Usar el estado 'paused' de la instancia de MainWindow, que es actualizado por ApplicationController
        if self.app_state.get("paused", False) or not optimizer_ref or not current_params:
            return

        if ga_instance_snapshot:
            self.progress_bar.setValue(int((ga_instance_snapshot.generations_completed / current_params["num_generations"]) * 100))
            self.progress_bar.setFormat(f"{ga_instance_snapshot.generations_completed}/{current_params['num_generations']}")

            current_best_details_gen = optimizer_ref.get_best_solution_details()
            if current_best_details_gen:
                # La lógica para actualizar self.best_solution_details_dict (el "mejor global")
                # debe estar en ApplicationController. Aquí, MainWindow solo lo lee para mostrarlo.
                best_solution_global_display = self.best_solution_details_dict # Leer el valor actualizado por el controller

                info_text = (f"Gen: {ga_instance_snapshot.generations_completed} | "
                             f"Mejor Actual X: {current_best_details_gen['x_value']:.4f}, f(X): {current_best_details_gen['f_x_value']:.4f}\n")
                if best_solution_global_display:
                    info_text += (f"Mejor Global X: {best_solution_global_display['x_value']:.4f}, f(X): {best_solution_global_display['f_x_value']:.4f} (Gen {best_solution_global_display['generation']})")
                else:
                    info_text += "Mejor Global: Aún no determinado."
                self.te_best_solution_info.setText(info_text)

            if self.plotter_module:
                self.plotter_module.update_fitness_plot_qt(self.fitness_plot_canvas, optimizer_ref)
                self.plotter_module.update_population_plot_qt(
                    self.population_plot_canvas, ga_instance_snapshot,
                    current_params["func_str"],
                    (current_params["range_min"], current_params["range_max"])
                )
            self.status_bar_widget.showMessage(f"Generación {ga_instance_snapshot.generations_completed} procesada.")

    @Slot(object)
    def handle_ag_stopped(self, ga_final_instance):
        print("MainWindow: AG detenido (recibido de señal).")
        # self.app_state es actualizado por ApplicationController
        self.update_gui_for_run_state(self.app_state["running"], self.app_state["paused"]) 

        optimizer = self.ga_optimizer_instance 
        current_params = self.current_params_dict
        best_solution_global = self.best_solution_details_dict # Ya debería estar actualizado por el controller

        if optimizer and optimizer.ga_instance and current_params:
            final_progress_val = int((optimizer.ga_instance.generations_completed / current_params["num_generations"]) * 100)
            self.progress_bar.setValue(final_progress_val)
            self.progress_bar.setFormat(f"{optimizer.ga_instance.generations_completed}/{current_params['num_generations']} (Finalizado)")

            if best_solution_global:
                info_text = (f"FINAL: Mejor Global X: {best_solution_global['x_value']:.6f}\n"
                             f"f(X): {best_solution_global['f_x_value']:.6f} (Encontrado en Gen: {best_solution_global['generation']})\n"
                             f"Generaciones completadas: {optimizer.ga_instance.generations_completed}")
                self.te_best_solution_info.setText(info_text)
                self.status_bar_widget.showMessage("Optimización completada.")
                QMessageBox.information(self, "Información", "Optimización Completada!")
            else:
                self.te_best_solution_info.setText("No se encontró una solución válida al finalizar.")
                self.status_bar_widget.showMessage("Optimización terminada, sin solución válida.")

            if self.plotter_module:
                self.plotter_module.update_fitness_plot_qt(self.fitness_plot_canvas, optimizer)
                self.plotter_module.update_population_plot_qt(
                    self.population_plot_canvas, optimizer.ga_instance,
                    current_params["func_str"],
                    (current_params["range_min"], current_params["range_max"])
                )
        else:
            self.te_best_solution_info.setText("Ejecución terminada con errores o sin resultados válidos.")
            self.status_bar_widget.showMessage("Ejecución terminada (sin instancia de AG válida).")

    @Slot(str)
    def handle_thread_error(self, error_msg):
        QMessageBox.critical(self, "Error del AG", f"Error en el hilo del AG: {error_msg}")
        self.status_bar_widget.showMessage(f"Error en AG: {error_msg[:50]}...")
        # self.app_state es actualizado por ApplicationController
        self.update_gui_for_run_state(self.app_state["running"], self.app_state["paused"])

    def update_gui_for_run_state(self, running, paused=False):
        """Actualiza el estado de los widgets de la UI basado en el estado de ejecución."""
        self.btn_start.setEnabled(not running)
        self.btn_pause.setEnabled(running)
        self.btn_pause.setText("Reanudar" if paused else "Pausar")
        
        # El estado del optimizador (self.ga_optimizer_instance) debe ser la referencia principal
        # para habilitar el botón de reset cuando no se está ejecutando.
        can_reset = (self.ga_optimizer_instance is not None and not running) or running
        self.btn_reset.setEnabled(can_reset)

        config_widgets = [
            self.rb_maximize, self.rb_minimize, self.le_func_str, self.le_range_min, self.le_range_max,
            self.le_pop_size, self.le_num_generations, self.le_crossover_prob, self.le_mutation_prob,
            self.combo_selection_type, self.combo_crossover_type, self.le_keep_elitism
        ]
        for widget in config_widgets:
            widget.setEnabled(not running)
        
        has_results = self.ga_optimizer_instance is not None and self.ga_optimizer_instance.ga_instance is not None
        self.btn_export_csv.setEnabled(not running and has_results)
        self.btn_export_pdf.setEnabled(not running and has_results and self.best_solution_details_dict is not None)
        
        # --- CORRECCIÓN PARA TypeError ---
        # Asegurar que has_history_for_gif_bool sea un booleano explícito
        has_history_for_gif_bool = False
        if self.ga_optimizer_instance and hasattr(self.ga_optimizer_instance, 'population_history'):
            if self.ga_optimizer_instance.population_history: # Comprueba si la lista no está vacía
                has_history_for_gif_bool = True
        
        self.btn_export_gif.setEnabled(bool(not running and has_results and has_history_for_gif_bool))
        # ---------------------------------

    def append_to_console(self, message: str):
        self.te_console_output.append(str(message).strip())
        self.te_console_output.ensureCursorVisible()

    def closeEvent(self, event):
        # Se delega al ApplicationController para manejar el cierre si es necesario
        # (por ejemplo, para detener hilos)
        if hasattr(self, 'controller_ref') and self.controller_ref:
             if self.controller_ref.handle_window_close_attempt(event): # Si el controller lo maneja y quiere cerrar
                 super().closeEvent(event)
             # Si el controller dice event.ignore(), no se llamará a super().closeEvent(event)
        else: # Comportamiento por defecto si no hay controller_ref o no maneja el evento
             super().closeEvent(event)


class QtConsoleOutputRedirector(QObject):
    text_written = Signal(str)
    def write(self, text):
        self.text_written.emit(text)
    def flush(self):
        pass


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window_test = MainWindow() 
    console_redirector_test = QtConsoleOutputRedirector()
    console_redirector_test.text_written.connect(window_test.append_to_console)
    sys.stdout = console_redirector_test
    sys.stderr = console_redirector_test
    print("Ventana de prueba de ui/main_window.py iniciada.")
    window_test.show()
    sys.exit(app.exec())