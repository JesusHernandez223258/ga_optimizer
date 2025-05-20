# ag_core/function_parser.py
import math
import numpy as np
from asteval import Interpreter # Interpreter class

def safe_eval_function(func_str, x_value):
    """
    Evalúa de forma segura una función matemática dada como string.
    Permite el uso de 'x', funciones de 'math' y 'numpy'.
    Esta versión crea una nueva instancia de Interpreter para cada evaluación
    para mayor compatibilidad con diferentes versiones de asteval.
    """
    if not func_str.strip():
        raise ValueError("La cadena de la función objetivo no puede estar vacía.")

    # Crear una NUEVA instancia de Interpreter para esta evaluación
    aeval_local = Interpreter()

    # Añadir símbolos necesarios a la tabla de esta instancia local
    # asteval por defecto incluye builtins.
    # Le añadimos math, numpy y nuestra variable 'x'.
    aeval_local.symtable['math'] = math
    aeval_local.symtable['np'] = np
    aeval_local.symtable['x'] = x_value
    
    # Evaluar la expresión. El método eval() sin argumentos extra
    # usará la symtable de la instancia.
    result = aeval_local.eval(func_str, show_errors=False) 
        
    if aeval_local.error:
        last_error = aeval_local.error[-1] # aeval.error es una lista de objetos Error
        # Un objeto Error tiene un método get_error() que devuelve una tupla (type, message, traceback_info)
        error_type, error_message, _ = last_error.get_error() 
        aeval_local.clear_errors() 
        raise ValueError(f"Error al evaluar la función '{func_str}' (tipo: {error_type}):\n{error_message}")
    
    if result is None:
        raise ValueError(f"La función '{func_str}' no retornó un valor numérico para x={x_value}.")
    
    if not isinstance(result, (int, float, np.number)):
        raise ValueError(f"La función '{func_str}' retornó un tipo no numérico ({type(result)}) para x={x_value}.")

    if np.isnan(result) or np.isinf(result):
        raise ValueError(f"La función '{func_str}' resultó en NaN o Infinito para x={x_value}.")
        
    return float(result)

if __name__ == '__main__':
    print("Probando safe_eval_function (con instancia local de Interpreter):")
    test_functions = [
        ("x * math.cos(x)", 2.0),
        ("x**2 - 3*x + 4", 5.0),
        ("math.sin(x) / (x + 0.00001)", 0.0),
        ("np.exp(-x**2)", 0.5),
        ("1/(1+x**2)", 1.0)
    ]

    for func_str, val in test_functions:
        try:
            res = safe_eval_function(func_str, val)
            print(f"f(x) = {func_str:<20} | x = {val:<5} | f(x) = {res:.4f}")
        except ValueError as e:
            print(f"Error para f(x) = {func_str}, x = {val}: {e}")

    print("\nProbando funciones erróneas:")
    error_tests = [
        ("import os", 0),
        ("x + y", 0),
        ("eval('1+1')", 0),
        ("lambda x: x", 0),
        ("math.log(-1)",1),
        ("", 1)
    ]
    for func_str, val in error_tests:
        try:
            res = safe_eval_function(func_str, val)
            print(f"f(x) = {func_str:<20} | x = {val:<5} | f(x) = {res:.4f} (INESPERADO)")
        except ValueError as e:
            print(f"Error (esperado) para f(x) = {func_str:<20}, x = {val:<5}: {e}")
        except Exception as e_gen:
            print(f"Error GENERAL (inesperado) para f(x) = {func_str:<20}, x = {val:<5}: {e_gen}")