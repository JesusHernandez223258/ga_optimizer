# main_app.py
import sys
import os
# import threading # QThread maneja los hilos de Qt
import traceback
import datetime
import logging # <--- AÑADIDO para logging

from PySide6.QtCore import QThread, Signal, QObject, Slot, QTimer
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

# Importaciones de módulos del proyecto
from ui.main_window import MainWindow, QtConsoleOutputRedirector
from ag_core.genetic_algorithm import GeneticOptimizer
from ag_core.function_parser import safe_eval_function
from visualization import plotter
from exporting import exporter

# --- Configuración del Logging ---
LOG_FILENAME = 'ga_optimizer_app.log'
logging.basicConfig(
    level=logging.DEBUG, # Capturar desde DEBUG hacia arriba
    format='%(asctime)s - %(levelname)s - %(name)s - %(threadName)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILENAME, mode='w'), # Escribir a archivo, 'w' para sobrescribir en cada ejecución
        logging.StreamHandler(sys.stdout) # También a la consola de la terminal
    ]
)
logger = logging.getLogger(__name__)


# --- Worker QThread para el Algoritmo Genético ---
class GAWorker(QObject):
    generation_update_signal = Signal(object)
    ag_stopped_signal = Signal(object) # object puede ser ga_instance o None
    error_occurred_signal = Signal(str)

    def __init__(self, params_dict):
        super().__init__()
        self.params = params_dict
        self.ga_optimizer_ref: GeneticOptimizer = None # Tipado para claridad
        self._should_stop = False
        logger.debug(f"GAWorker: Inicializado con params: {self.params}")

    @Slot()
    def run(self):
        logger.info("GAWorker: El método run() ha comenzado.")
        try:
            if self.params is None: # Doble verificación
                logger.error("GAWorker: self.params es None al inicio de run().")
                raise ValueError("Los parámetros para GAWorker no pueden ser None.")

            self.ga_optimizer_ref = GeneticOptimizer(
                self.params,
                self.params.get("func_str", ""), # Usar .get para seguridad
                on_generation_callback=self._emit_generation_update,
                on_stop_callback=self._emit_ag_stopped
            )
            
            if self._should_stop:
                logger.info("GAWorker: Se solicitó parada antes de iniciar el AG.")
                self.ag_stopped_signal.emit(None) # Indicar cancelación/parada
                return

            logger.info("GAWorker: Llamando a ga_optimizer_ref.run()...")
            self.ga_optimizer_ref.run() # Esto es bloqueante
            # ag_stopped_signal será emitido por el callback _on_stop_capture de GeneticOptimizer

        except Exception as e:
            error_msg = f"Error crítico en GAWorker.run(): {e}"
            logger.error(error_msg, exc_info=True) # exc_info=True para loggear el traceback completo
            self.error_occurred_signal.emit(str(e)) # Enviar solo el mensaje de error a la GUI
            self.ag_stopped_signal.emit(None) # Indicar parada con error
        logger.info("GAWorker: El método run() ha finalizado.")


    def _emit_generation_update(self, ga_instance):
        if not self._should_stop:
            self.generation_update_signal.emit(ga_instance)

    def _emit_ag_stopped(self, ga_instance):
        logger.info(f"GAWorker: AG detenido internamente, emitiendo ag_stopped_signal. Instancia: {'Presente' if ga_instance else 'Ausente/Error'}")
        self.ag_stopped_signal.emit(ga_instance)

    def get_optimizer(self):
        return self.ga_optimizer_ref

    @Slot()
    def request_stop(self):
        logger.info("GAWorker: Solicitud de parada (request_stop) recibida.")
        self._should_stop = True
        if self.ga_optimizer_ref and self.ga_optimizer_ref.ga_instance:
            # PyGAD no tiene un método stop() explícito que podamos llamar para interrumpir run()
            # Esta bandera _should_stop ayudará a prevenir nuevas emisiones de señales
            # y podría usarse si PyGAD se ejecutara iterativamente.
            logger.debug("GAWorker: _should_stop establecido a True. PyGAD.run() es bloqueante.")


class ApplicationController:
    def __init__(self, main_window: MainWindow):
        self.window = main_window
        self.is_running_ga = False
        self.is_paused_ga = False # Pausa simulada
        self.current_ga_optimizer: GeneticOptimizer = None
        self.current_params = None
        self.best_solution_ever = None
        
        self.ag_qthread: QThread = None
        self.ga_worker_obj: GAWorker = None

        # Pasar referencias importantes a la ventana para su uso interno
        self.window.plotter_module = plotter 
        self._update_window_references() # Inicializar refs en la ventana

        self._connect_ui_signals()
        logger.info("ApplicationController: Inicializado y señales conectadas.")

    def _connect_ui_signals(self):
        self.window.btn_start.clicked.connect(self.on_start_clicked)
        self.window.btn_pause.clicked.connect(self.on_pause_resume_clicked)
        self.window.btn_reset.clicked.connect(self.on_reset_clicked)
        self.window.btn_export_csv.clicked.connect(self.on_export_csv_clicked)
        self.window.btn_export_pdf.clicked.connect(self.on_export_pdf_clicked)
        self.window.btn_export_gif.clicked.connect(self.on_export_gif_clicked)
        
        # Conectar el evento de cierre de la ventana del ApplicationController
        # QApplication.instance().aboutToQuit.connect(self.handle_app_about_to_quit)
        # En lugar de aboutToQuit, podemos usar el closeEvent de MainWindow si lo configuramos
        # para que llame a un método del controlador. O el controlador puede manejarlo.

    def _update_window_references(self):
        """Actualiza las referencias en la instancia de MainWindow."""
        self.window.ga_optimizer_instance = self.current_ga_optimizer
        self.window.current_params_dict = self.current_params
        self.window.best_solution_details_dict = self.best_solution_ever
        # Actualizar el app_state de la ventana para que refleje el estado del controlador
        self.window.app_state["running"] = self.is_running_ga
        self.window.app_state["paused"] = self.is_paused_ga
        # self.window.app_state["ga_optimizer"] no es necesario ya que tiene ga_optimizer_instance

    @Slot()
    def on_start_clicked(self):
        logger.debug("ApplicationController: Botón Iniciar presionado.")
        if self.is_running_ga:
            QMessageBox.information(self.window, "Información", "El algoritmo ya está en ejecución.")
            logger.warning("ApplicationController: Intento de iniciar AG mientras ya está corriendo.")
            return

        params = self.window.get_parameters_from_gui()
        logger.debug(f"ApplicationController: Parámetros de GUI: {params}")

        if not params: 
            logger.error("ApplicationController: get_parameters_from_gui devolvió None. No se inicia el AG.")
            return

        try:
            # Validar función antes de pasarla al hilo
            test_x = (params["range_min"] + params["range_max"]) / 2
            logger.debug(f"ApplicationController: Validando función '{params['func_str']}' con x={test_x}")
            safe_eval_function(params["func_str"], test_x)
            logger.info("ApplicationController: Función objetivo validada exitosamente.")
        except ValueError as e:
            error_msg = f"Función objetivo inválida: {e}"
            QMessageBox.critical(self.window, "Error en Función Objetivo", error_msg)
            self.window.status_bar_widget.showMessage(f"Error en función: {str(e)[:50]}...")
            logger.error(f"ApplicationController: {error_msg}", exc_info=False) # No necesitamos el traceback aquí
            return
        except KeyError as e_key:
             error_msg = f"Falta el parámetro '{e_key}' para validar la función."
             QMessageBox.critical(self.window, "Error en Parámetros", error_msg)
             self.window.status_bar_widget.showMessage(f"Parámetro faltante: {e_key}")
             logger.error(f"ApplicationController: {error_msg}")
             return

        self.current_params = params
        self.best_solution_ever = None
        self.current_ga_optimizer = None # Limpiar instancia anterior
        self._update_window_references()

        plotter.clear_plots_qt(self.window.fitness_plot_canvas, self.window.population_plot_canvas)
        self.window.progress_bar.setValue(0)
        self.window.progress_bar.setFormat("0%")
        self.window.te_best_solution_info.setText("Iniciando...")
        self.window.status_bar_widget.showMessage("Iniciando algoritmo genético...")
        logger.info("\n" + "="*10 + " Iniciando Nueva Ejecución del AG " + "="*10)
        logger.info(f"ApplicationController: Parámetros para GAWorker: {self.current_params}")

        self.ag_qthread = QThread(parent=self.window) # Establecer parent para mejor gestión de Qt
        self.ga_worker_obj = GAWorker(self.current_params)
        self.ga_worker_obj.moveToThread(self.ag_qthread)

        self.ga_worker_obj.generation_update_signal.connect(self.handle_generation_update)
        self.ga_worker_obj.ag_stopped_signal.connect(self.handle_ag_stopped)
        self.ga_worker_obj.error_occurred_signal.connect(self.handle_thread_error)

        self.ag_qthread.started.connect(self.ga_worker_obj.run)
        self.ag_qthread.finished.connect(self.on_qthread_finished) # Para limpieza
        
        self.is_running_ga = True
        self.is_paused_ga = False
        self._update_window_references()
        self.window.update_gui_for_run_state(True, False)
        
        self.ag_qthread.start()
        logger.info("ApplicationController: QThread del AG iniciado.")

    @Slot()
    def on_pause_resume_clicked(self):
        if not self.is_running_ga: return
        self.is_paused_ga = not self.is_paused_ga
        self._update_window_references()
        self.window.update_gui_for_run_state(self.is_running_ga, self.is_paused_ga)
        status_msg = "Pausado (actualizaciones de GUI detenidas)." if self.is_paused_ga else "Reanudado."
        self.window.status_bar_widget.showMessage(status_msg)
        logger.info(f"ApplicationController: Estado de pausa (simulado): {self.is_paused_ga}")

    @Slot()
    def on_reset_clicked(self):
        logger.info("ApplicationController: Botón Reiniciar presionado.")
        self.window.status_bar_widget.showMessage("Intentando reiniciar...")
        if self.ag_qthread and self.ag_qthread.isRunning():
            logger.info("ApplicationController: Solicitando parada del worker y QThread...")
            if self.ga_worker_obj: self.ga_worker_obj.request_stop()
            self.ag_qthread.quit() # Pide al bucle de eventos del hilo que termine
            # Esperar un poco a que el hilo termine limpiamente
            if not self.ag_qthread.wait(1500): # Aumentar tiempo de espera
                 logger.warning("ApplicationController: El QThread no terminó limpiamente después de quit(). Intentando terminarlo forzosamente.")
                 # self.ag_qthread.terminate() # Usar con extrema precaución, puede corromper datos
                 # self.ag_qthread.wait() # Esperar a que realmente termine
                 # Es mejor dejar que on_qthread_finished y handle_ag_stopped hagan el reseteo
                 # si el hilo no para por sí mismo.
            else:
                logger.info("ApplicationController: QThread ha terminado después de quit() y wait().")
                # Si el hilo terminó bien, _perform_actual_reset_logic ya debería haber sido llamado
                # a través de on_qthread_finished o handle_ag_stopped.
                # Si no, el estado is_running_ga podría seguir en True.
                if self.is_running_ga: # Si el reseteo no ocurrió, forzarlo
                    self._perform_actual_reset_logic()

        else: # Si no hay hilo corriendo o ya terminó
            logger.info("ApplicationController: No hay hilo AG corriendo, reseteando directamente.")
            self._perform_actual_reset_logic()

    def _perform_actual_reset_logic(self):
        logger.info("ApplicationController: Realizando reseteo de lógica y UI.")
        self.is_running_ga = False
        self.is_paused_ga = False
        self.current_ga_optimizer = None
        self.current_params = None
        self.best_solution_ever = None
        self._update_window_references() # Actualizar referencias en MainWindow

        plotter.clear_plots_qt(self.window.fitness_plot_canvas, self.window.population_plot_canvas)
        self.window.progress_bar.setValue(0)
        self.window.progress_bar.setFormat("0%")
        self.window.te_best_solution_info.setText("Sistema reiniciado. Listo.")
        if hasattr(self.window, 'te_console_output'): self.window.te_console_output.clear()
        self.window.status_bar_widget.showMessage("Sistema reiniciado.")
        self.window.update_gui_for_run_state(False)
        logger.info("\n" + "="*10 + " Sistema Reiniciado (PySide6) " + "="*10)

    @Slot()
    def on_qthread_finished(self):
        logger.info("ApplicationController: QThread del AG ha finalizado (señal finished).")
        # Limpiar referencias
        self.ag_qthread = None 
        self.ga_worker_obj = None
        
        # Si el AG se detuvo (por ejemplo, por request_stop que llevó a quit())
        # pero el estado de 'running' no se actualizó a través de handle_ag_stopped,
        # forzamos una actualización de la UI aquí.
        if self.is_running_ga:
            logger.warning("ApplicationController: QThread finalizado, pero is_running_ga seguía True. Forzando reseteo de UI.")
            self.is_running_ga = False # Asegurar que el estado se actualice
            self.is_paused_ga = False
            self._update_window_references()
            self.window.update_gui_for_run_state(False, False)
            # Considerar si _perform_actual_reset_logic es más apropiado aquí
            # self._perform_actual_reset_logic()

    # --- Slots para manejar señales del GAWorker ---
    @Slot(object)
    def handle_generation_update(self, ga_instance_snapshot):
        # logger.debug(f"Controller: Recibida actualización de generación {ga_instance_snapshot.generations_completed}")
        if self.is_paused_ga or not self.current_ga_optimizer or not self.current_params:
            return

        # Actualizar el mejor global aquí en el controlador
        current_best_details_gen = self.current_ga_optimizer.get_best_solution_details()
        if current_best_details_gen:
            if self.best_solution_ever is None or \
               (self.current_params["optimization_type"] == "maximize" and current_best_details_gen["f_x_value"] > self.best_solution_ever["f_x_value"]) or \
               (self.current_params["optimization_type"] == "minimize" and current_best_details_gen["f_x_value"] < self.best_solution_ever["f_x_value"]):
                self.best_solution_ever = current_best_details_gen.copy()
                self._update_window_references() # Actualizar la referencia en la ventana

        # Dejar que la ventana actualice su UI
        self.window.handle_generation_update(ga_instance_snapshot)

    @Slot(object)
    def handle_ag_stopped(self, ga_final_instance):
        logger.info(f"Controller: AG detenido (señal de GAWorker). Instancia: {'Presente' if ga_final_instance else 'Ausente/Error'}")
        
        if self.ga_worker_obj: # Obtener la instancia final del optimizador del worker
            self.current_ga_optimizer = self.ga_worker_obj.get_optimizer()
        
        self.is_running_ga = False # Esencial: marcar como no corriendo
        self.is_paused_ga = False
        self._update_window_references() # Actualizar referencias y estado en la ventana

        # Llamar al handler de la ventana para actualizar la UI
        self.window.handle_ag_stopped(ga_final_instance) 

        # Asegurarse de que el hilo QThread se detenga si aún está activo
        if self.ag_qthread and self.ag_qthread.isRunning():
            logger.debug("Controller: AG Stoped, solicitando quit() al QThread.")
            self.ag_qthread.quit()
            # No hacer wait() aquí para no bloquear, finished se encargará de la limpieza del objeto QThread

    @Slot(str)
    def handle_thread_error(self, error_msg):
        logger.error(f"Controller: Error recibido del hilo del AG: {error_msg}")
        # La lógica de parada y actualización de UI ya debería estar cubierta por handle_ag_stopped(None)
        # que es llamado por el worker en caso de error.
        # Aquí solo nos aseguramos de que la ventana muestre el error.
        self.window.handle_thread_error(error_msg) # Dejar que la ventana muestre el popup
        # Asegurar que el estado de la app esté correcto
        if self.is_running_ga:
            self.is_running_ga = False
            self.is_paused_ga = False
            self._update_window_references()
            self.window.update_gui_for_run_state(False, False)


    # --- Slots para botones de exportación ---
    # (Estos métodos no cambian significativamente, solo usan las variables de instancia del controlador)
    @Slot()
    def on_export_csv_clicked(self):
        logger.debug("Controller: Exportar CSV presionado.")
        if self.current_ga_optimizer and self.current_ga_optimizer.ga_instance and self.current_params:
            default_name = f"ga_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            filename, _ = QFileDialog.getSaveFileName(self.window, "Guardar como CSV", default_name, "CSV Files (*.csv)")
            if filename:
                self.window.status_bar_widget.showMessage(f"Exportando a {os.path.basename(filename)}...")
                QApplication.processEvents() # Dar oportunidad a la GUI de actualizarse
                success, msg = exporter.export_population_to_csv(
                    self.current_ga_optimizer.ga_instance, self.current_params["func_str"], filename
                )
                self.window.lbl_export_status.setText(msg)
                self.window.status_bar_widget.showMessage(msg)
                if success: QMessageBox.information(self.window, "Exportación CSV", msg)
                else: QMessageBox.warning(self.window, "Error Exportación CSV", msg)
        else:
            QMessageBox.warning(self.window, "Error Exportación", "No hay datos para exportar.")
            logger.warning("Controller: Intento de exportar CSV sin datos suficientes.")


    @Slot()
    def on_export_pdf_clicked(self):
        logger.debug("Controller: Exportar PDF presionado.")
        if self.current_ga_optimizer and self.current_params and self.best_solution_ever:
            default_name = f"ga_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            filename, _ = QFileDialog.getSaveFileName(self.window, "Guardar como PDF", default_name, "PDF Files (*.pdf)")
            if filename:
                self.window.status_bar_widget.showMessage(f"Generando PDF: {os.path.basename(filename)}...")
                QApplication.processEvents()
                success, msg = exporter.export_report_to_pdf(
                    self.window, self.current_ga_optimizer, self.current_params, self.best_solution_ever, filename
                )
                self.window.lbl_export_status.setText(msg)
                self.window.status_bar_widget.showMessage(msg)
                if success: QMessageBox.information(self.window, "Exportación PDF", msg)
                else: QMessageBox.critical(self.window, "Error Exportación PDF", msg)
        else:
            QMessageBox.warning(self.window, "Error Exportación", "No hay datos suficientes para generar el PDF.")
            logger.warning("Controller: Intento de exportar PDF sin datos suficientes.")

    @Slot()
    def on_export_gif_clicked(self):
        logger.debug("Controller: Exportar GIF presionado.")
        if self.current_ga_optimizer and hasattr(self.current_ga_optimizer, 'population_history') and \
           self.current_ga_optimizer.population_history and self.current_params:
            default_name = f"ga_animation_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.gif"
            filename, _ = QFileDialog.getSaveFileName(self.window, "Guardar Animación como GIF", default_name, "GIF Files (*.gif)")
            if filename:
                self.window.status_bar_widget.showMessage(f"Generando GIF: {os.path.basename(filename)}... (puede tardar)")
                QApplication.processEvents()
                success, msg = exporter.export_animation_to_gif(
                    self.current_ga_optimizer, self.current_params["func_str"],
                    (self.current_params["range_min"], self.current_params["range_max"]),
                    self.current_params["optimization_type"], filename, fps=10 # Aumentado FPS para prueba
                )
                self.window.lbl_export_status.setText(msg)
                self.window.status_bar_widget.showMessage(msg)
                if success: QMessageBox.information(self.window, "Exportación GIF", msg)
                else: QMessageBox.critical(self.window, "Error Exportación GIF", msg)
        else:
            QMessageBox.warning(self.window, "Error Exportación", "No hay historial de población para la animación.")
            logger.warning("Controller: Intento de exportar GIF sin datos suficientes.")

    def handle_app_about_to_quit(self):
        """Maneja la señal de que la aplicación está a punto de cerrarse."""
        logger.info("ApplicationController: Aplicación a punto de cerrarse (aboutToQuit).")
        if self.ag_qthread and self.ag_qthread.isRunning():
            logger.info("ApplicationController: Deteniendo hilo del AG antes de salir...")
            if self.ga_worker_obj:
                self.ga_worker_obj.request_stop()
            self.ag_qthread.quit()
            if not self.ag_qthread.wait(2000): # Darle 2 segundos para terminar
                logger.warning("ApplicationController: El hilo del AG no terminó limpiamente al cerrar la app.")
        logger.info("ApplicationController: Limpieza de salida completada.")


def main_qt_app():
    # Configurar un excepthook global para capturar errores no manejados y loggearlos
    # ANTES de que Qt pueda cerrar la aplicación prematuramente.
    def global_except_hook(exctype, value, tb):
        logger.critical("Excepción global no manejada:", exc_info=(exctype, value, tb))
        # Mostrar un QMessageBox aquí puede ser problemático si la GUI ya está corrupta.
        # Es mejor asegurarse de que el logging a archivo funcione.
        # Podríamos intentar un QMessageBox simple:
        # QMessageBox.critical(None, "Error Crítico Inesperado", 
        #                      f"Ha ocurrido un error fatal:\n{value}\nConsulte el archivo de log para detalles.")
        sys.__excepthook__(exctype, value, tb) # Llamar al excepthook original para que imprima a stderr
        if isinstance(QApplication.instance(), QApplication): # Si la app Qt aún existe
             QApplication.instance().quit() # Intentar cerrar limpiamente

    sys.excepthook = global_except_hook

    app = QApplication(sys.argv)
    # app.setStyle("Fusion") 

    main_window_instance = MainWindow()
    
    console_redirector = QtConsoleOutputRedirector()
    console_redirector.text_written.connect(main_window_instance.append_to_console)
    # Guardar referencias originales de stdout/stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = console_redirector
    sys.stderr = console_redirector

    logger.info("Aplicación Optimizador AG (PySide6) iniciada.")
    try:
        import PySide6
        logger.info(f"Versión de PySide6: {PySide6.__version__}")
    except ImportError:
        logger.error("PySide6 no está instalado correctamente.")
        return # No continuar si PySide6 no se puede importar

    controller = ApplicationController(main_window_instance)
    QApplication.instance().aboutToQuit.connect(controller.handle_app_about_to_quit) # Conectar señal de cierre

    main_window_instance.show()
    
    exit_code = 0
    try:
        exit_code = app.exec()
    except Exception as e_main_loop:
        logger.critical(f"Error no manejado en el bucle principal de la aplicación (app.exec()): {e_main_loop}", exc_info=True)
        # Mostrar error al usuario
        QMessageBox.critical(None, "Error Fatal de Aplicación", 
                             f"Un error crítico ha ocurrido en el bucle de eventos de la aplicación:\n{e_main_loop}\n"
                             "La aplicación se cerrará. Revise el archivo 'ga_optimizer_app.log'.")
        exit_code = 1 # Indicar salida con error
    finally:
        # Restaurar stdout/stderr originales
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        logger.info(f"Aplicación finalizada con código de salida: {exit_code}")
        sys.exit(exit_code)


if __name__ == '__main__':
    # Wrapper para asegurar que cualquier excepción en main_qt_app también se loggee
    try:
        main_qt_app()
    except Exception as e_top:
        logger.critical("Excepción de nivel superior no manejada al iniciar la aplicación:", exc_info=True)
        # Aquí es difícil mostrar un QMessageBox porque QApplication podría no estar activo o estar corrupto
        print(f"ERROR FATAL AL INICIAR: {e_top}\nConsulte {LOG_FILENAME}", file=sys.__stderr__) # Usar stderr original