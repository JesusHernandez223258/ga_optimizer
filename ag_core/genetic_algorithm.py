import pygad
import numpy as np
from .function_parser import safe_eval_function
import logging

logger_ga = logging.getLogger(f"{__name__}.GeneticOptimizer")

class GeneticOptimizer:
    def __init__(self, params, fitness_func_str, on_generation_callback=None, on_stop_callback=None):
        logger_ga.debug(f"__init__: Recibidos params: {params}")
        if params is None:
            logger_ga.error("__init__: Los parámetros (params) son None.")
            raise ValueError("Los parámetros (params) no pueden ser None al crear GeneticOptimizer.")
        
        required_keys = [
            'range_min', 'range_max', 'pop_size', 'num_generations', 
            'selection_type', 'keep_elitism', 'crossover_type', 
            'crossover_prob', 'mutation_prob', 'optimization_type', 'func_str'
        ]
        for key in required_keys:
            if key not in params:
                logger_ga.error(f"__init__: Falta la clave requerida '{key}' en params: {params}")
                raise KeyError(f"Falta la clave requerida '{key}' en los parámetros de inicialización de GeneticOptimizer.")

        self.params = params
        self.fitness_func_str = params['func_str']
        self.on_generation_callback = on_generation_callback
        self.on_stop_callback = on_stop_callback
        self.ga_instance = None
        self.population_history = []
        self.best_solution_fitness_history = []
        self.optimization_type = params['optimization_type']
        self._fitness_penalty = -np.inf if self.optimization_type == 'maximize' else np.inf
        logger_ga.info(f"GeneticOptimizer inicializado para {self.optimization_type} f(x)={self.fitness_func_str}")

    def _fitness_wrapper(self, ga_inst, solution, sol_idx):
        try:
            x_val = solution[0]
            raw = safe_eval_function(self.fitness_func_str, x_val)
            return -raw if self.optimization_type == 'minimize' else raw
        except ValueError as ve:
            logger_ga.warning(f"_fitness_wrapper: ValueError: {ve} para solution {solution}. Aplicando penalización.")
            return self._fitness_penalty
        except Exception as e:
            logger_ga.error(f"_fitness_wrapper: CRÍTICO: {e} para solution {solution}. Aplicando penalización.", exc_info=True)
            return self._fitness_penalty

    def _on_generation_capture(self, ga_inst):
        if self.on_generation_callback:
            self.on_generation_callback(ga_inst)

    def _on_stop_capture(self, ga_inst, last_gen_fit):
        logger_ga.info(f"_on_stop_capture: AG detenido. Última gen fitness: {last_gen_fit}")
        if self.on_stop_callback:
            self.on_stop_callback(ga_inst)

    def setup_ga_instance(self):
        params = self.params
        logger_ga.debug(f"setup_ga_instance: Entrando. params: {params}")

        # Validar rango
        range_min = params['range_min']
        range_max = params['range_max']
        if not isinstance(range_min, (int, float)):
            raise TypeError(f"'range_min' debe ser numérico, pero se obtuvo {type(range_min)}.")
        if not isinstance(range_max, (int, float)):
            raise TypeError(f"'range_max' debe ser numérico, pero se obtuvo {type(range_max)}.")
        if range_min >= range_max:
            raise ValueError(f"range_min ({range_min}) debe ser menor que range_max ({range_max}).")

        # Definir espacio de genes
        num_genes_val = 1
        gene_space_val = [{'low': range_min, 'high': range_max}]
        logger_ga.info(f"setup_ga_instance: gene_space_val: {gene_space_val} (Tipo: {type(gene_space_val)})")

        # Validación adicional
        if not (isinstance(gene_space_val, list) and 
                len(gene_space_val) == 1 and isinstance(gene_space_val[0], dict) and 
                isinstance(gene_space_val[0].get('low'), (int, float)) and 
                isinstance(gene_space_val[0].get('high'), (int, float))):
            raise TypeError(f"Validación gene_space fallida: {gene_space_val}")

        # Padres para mating
        n_parents = max(2, int(params['pop_size'] * 0.4))
        if n_parents % 2 != 0:
            n_parents -= 1
            if n_parents < 2:
                n_parents = 2

        # Simplificación de la llamada a PyGAD
        try:
            self.ga_instance = pygad.GA(
                num_generations=int(params['num_generations']),
                num_parents_mating=n_parents,
                fitness_func=self._fitness_wrapper,
                sol_per_pop=int(params['pop_size']),
                num_genes=num_genes_val,
                gene_space=gene_space_val,
                gene_type=float
                # Aislado: agregar opcionales uno a uno si todo funciona
            )
            logger_ga.info("setup_ga_instance: PyGAD Instance configured con los parámetros esenciales.")
        except Exception as ex:
            logger_ga.error(f"Error inicializando pygad.GA con gene_space: {gene_space_val}", exc_info=True)
            raise

    def run(self):
        logger_ga.info("run: Iniciando optimización.")
        if self.ga_instance is None:
            self.setup_ga_instance()
        self.ga_instance.run()
        return self.ga_instance

    def get_best_solution_details(self):
        if not self.ga_instance or self.ga_instance.best_solution_generation == -1:
            return None
        sol, fit_int, _ = self.ga_instance.best_solution()
        fit_act = -fit_int if self.optimization_type == 'minimize' else fit_int
        return {
            'x_value': sol[0],
            'f_x_value': fit_act,
            'internal_fitness': fit_int,
            'generation': self.ga_instance.best_solution_generation
        }

# Fin de genetic_algorithm.py
