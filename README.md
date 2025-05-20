# Optimizador con Algoritmos Genéticos

Aplicación de escritorio en Python para configurar, ejecutar y visualizar algoritmos genéticos aplicados a problemas de optimización.

## Características Implementadas (Versión PySide6)

*   Interfaz gráfica (PySide6) para configuración de parámetros del AG.
*   Definición de función objetivo personalizada.
*   Ejecución del algoritmo genético (usando PyGAD) en un hilo separado (QThread).
*   Visualización en tiempo real de:
    *   Evolución de la aptitud (mejor fitness por generación).
    *   Población sobre la curva de la función objetivo.
*   Exportación de:
    *   Resultados de la población a CSV.
    *   Reporte del experimento a PDF (incluyendo gráficos).
*   Animación básica del proceso evolutivo (exportable a GIF).
*   Consola de salida integrada en la GUI.

## Stack Tecnológico

*   **Lenguaje**: Python 3.9+
*   **GUI**: PySide6 (Qt for Python)
*   **Algoritmo Genético**: PyGAD
*   **Gráficos**: Matplotlib
*   **Animación**: Matplotlib.animation, MoviePy
*   **Exportación**: Pandas (CSV), ReportLab (PDF)
*   **Evaluación Segura de Funciones**: Asteval

## Configuración y Ejecución

1.  **Clonar el repositorio o descargar los archivos.**
2.  **Crear un entorno virtual (recomendado):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate    # Windows
    ```
3.  **Instalar dependencias:**
    ```bash
    pip install -r requirements.txt
    ```
    *Nota: Para la exportación de animaciones GIF/MP4, `moviepy` puede requerir `ffmpeg`. Si no lo tienes, MoviePy intentará descargarlo, o puedes instalarlo manualmente (`sudo apt install ffmpeg` en Debian/Ubuntu).*

4.  **Ejecutar la aplicación:**
    Asegúrate de estar en el directorio raíz `ga_optimizer_project`.
    ```bash
    python main_app.py
    ```

## Estructura del Proyecto

    ```bash
    └── ga_optimizer_project
    ├── ag_core
    │   ├── function_parser.py
    │   └── genetic_algorithm.py
    ├── assets
    ├── exporting
    │   └── exporter.py
    ├── main_app.py
    ├── README.md
    ├── requirements.txt
    ├── ui
    │   └── main_window.py
    └── visualization
        ├── animator.py
        └── plotter.py
    ```
