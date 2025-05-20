# exporting/exporter.py
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import io
import datetime
import traceback # Para errores
import matplotlib.pyplot as plt # Para cerrar figuras de animación

from visualization import plotter # Para get_..._fig_from_canvas
from visualization import animator 

try:
    from ..ag_core.function_parser import safe_eval_function
except ImportError:
    from ag_core.function_parser import safe_eval_function 

def export_population_to_csv(ga_instance, func_str, filename="ga_population_results.csv"):
    if not ga_instance or ga_instance.population is None:
        return False, "No hay datos de población para exportar."
    data = []
    population = ga_instance.population
    try:
        pop_fitness_internal = ga_instance.cal_pop_fitness()
    except Exception: # Si cal_pop_fitness falla (ej. no está listo)
        pop_fitness_internal = [None] * len(population)

    for i, sol_genes in enumerate(population):
        x_value = sol_genes[0] 
        f_x_value_str, fitness_internal_str = "Error", "Error"
        try:
            f_x_value = safe_eval_function(func_str, x_value)
            f_x_value_str = f"{f_x_value:.6f}"
        except Exception as e_fx:
            f_x_value_str = f"Error f(x): {e_fx}"
        
        if pop_fitness_internal[i] is not None:
            try:
                fitness_internal_str = f"{float(pop_fitness_internal[i]):.6f}"
            except (ValueError, TypeError):
                 fitness_internal_str = str(pop_fitness_internal[i]) # Si no es flotante
        else:
            fitness_internal_str = "N/A"


        data.append({
            "Individuo_ID": i + 1,
            "Genes_X (Valor_X)": f"{x_value:.6f}",
            "Valor_f(x)_Real": f_x_value_str,
            "Fitness_Interno_PyGAD": fitness_internal_str
        })
    
    df = pd.DataFrame(data)
    try:
        df.to_csv(filename, index=False)
        return True, f"Datos de la última población exportados a {filename}"
    except Exception as e:
        return False, f"Error al exportar CSV: {e}"

def export_report_to_pdf(main_window_ref, ga_optimizer, params_snapshot, best_solution_details, filename="ga_report.pdf"):
    if not main_window_ref or not ga_optimizer or not ga_optimizer.ga_instance or \
       not best_solution_details or not params_snapshot:
        return False, "Datos insuficientes para generar el PDF (faltan referencias o datos)."

    ga_instance = ga_optimizer.ga_instance
    doc = SimpleDocTemplate(filename, pagesize=letter,
                            rightMargin=0.7*inch, leftMargin=0.7*inch,
                            topMargin=0.7*inch, bottomMargin=0.7*inch)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='CodeSmall', parent=styles['Code'], fontSize=8))
    story = []

    title_str = "Reporte del Algoritmo Genético"
    story.append(Paragraph(title_str, styles['h1']))
    story.append(Paragraph(f"<i>Generado el: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))

    story.append(Paragraph("<b>Parámetros de Configuración</b>", styles['h2']))
    func_str_display = params_snapshot['func_str']
    if len(func_str_display) > 60: # Acortar si es muy larga para la tabla
        func_str_display = func_str_display[:57] + "..."

    config_data = [
        [Paragraph("<b>Función Objetivo f(x):</b>", styles['Normal']), Paragraph(f"<font face=Courier size=9>{func_str_display}</font>", styles['Normal'])],
        ["<b>Tipo Optimización:</b>", params_snapshot['optimization_type'].capitalize()],
        ["<b>Intervalo Búsqueda:</b>", f"[{params_snapshot['range_min']}, {params_snapshot['range_max']}]"],
        ["<b>Tamaño Población:</b>", str(params_snapshot['pop_size'])],
        ["<b>Generaciones Programadas:</b>", str(params_snapshot['num_generations'])],
        ["<b>Generaciones Completadas:</b>", str(ga_instance.generations_completed)],
        ["<b>Prob. Cruce (Pc):</b>", str(params_snapshot['crossover_prob'])],
        ["<b>Prob. Mutación (Pm):</b>", str(params_snapshot['mutation_prob'])],
        ["<b>Método Selección:</b>", params_snapshot['selection_type']],
        ["<b>Tipo Cruce:</b>", params_snapshot['crossover_type']],
        ["<b>Elitismo (N mejores):</b>", str(params_snapshot['keep_elitism'])],
    ]
    config_table = Table(config_data, colWidths=[2.2*inch, 4.6*inch], hAlign='LEFT')
    config_table.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.Color(0.8,0.8,0.8)), # Gris más claro
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        # ('BACKGROUND', (0,0), (0,-1), colors.whitesmoke),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'), # Fila de encabezado
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'), # Columna de etiquetas
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(config_table)
    story.append(Spacer(1, 0.2*inch))

    story.append(Paragraph("<b>Mejor Solución Encontrada Globalmente</b>", styles['h2']))
    best_sol_data = [
        ["<b>Valor X:</b>", f"{best_solution_details['x_value']:.8f}"],
        ["<b>Valor f(X) Real:</b>", f"{best_solution_details['f_x_value']:.8f}"],
        ["<b>Fitness Interno PyGAD:</b>", f"{best_solution_details['internal_fitness']:.8f}"],
        ["<b>Encontrada en Generación:</b>", str(best_solution_details['generation'])],
    ]
    best_sol_table = Table(best_sol_data, colWidths=[2.2*inch, 4.6*inch], hAlign='LEFT')
    best_sol_table.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.Color(0.8,0.8,0.8)),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        # ('BACKGROUND', (0,0), (0,-1), colors.whitesmoke),
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(best_sol_table)
    story.append(Spacer(1, 0.1*inch)) # Menos espacio antes del page break
    story.append(PageBreak())

    story.append(Paragraph("<b>Gráfico de Evolución de la Mejor Aptitud Real</b>", styles['h2']))
    fig_fitness = plotter.get_fitness_plot_fig_from_canvas(main_window_ref.fitness_plot_canvas)
    if fig_fitness:
        img_buffer_fitness = io.BytesIO()
        fig_fitness.savefig(img_buffer_fitness, format='PNG', dpi=150, bbox_inches='tight') # DPI reducido para PDF
        img_buffer_fitness.seek(0)
        # Ajustar tamaño de imagen para que quepa bien
        img_w, img_h = fig_fitness.get_size_inches()
        aspect = img_h / img_w
        report_img_width = 6.5 * inch 
        story.append(Image(img_buffer_fitness, width=report_img_width, height=report_img_width * aspect))
        img_buffer_fitness.close()
    else: story.append(Paragraph("Gráfico de aptitud no disponible.", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))

    story.append(Paragraph("<b>Gráfico de Población (Última Generación)</b>", styles['h2']))
    fig_population = plotter.get_population_plot_fig_from_canvas(main_window_ref.population_plot_canvas)
    if fig_population:
        img_buffer_pop = io.BytesIO()
        fig_population.savefig(img_buffer_pop, format='PNG', dpi=150, bbox_inches='tight')
        img_buffer_pop.seek(0)
        img_w_pop, img_h_pop = fig_population.get_size_inches()
        aspect_pop = img_h_pop / img_w_pop
        story.append(Image(img_buffer_pop, width=report_img_width, height=report_img_width * aspect_pop))
        img_buffer_pop.close()
    else: story.append(Paragraph("Gráfico de población no disponible.", styles['Normal']))
    
    try:
        doc.build(story)
        return True, f"Reporte PDF generado: {filename}"
    except Exception as e:
        return False, f"Error al generar PDF: {e}\n{traceback.format_exc()}"

def export_animation_to_gif(ga_optimizer, func_str, x_range, optimization_type, filename="ga_evolution.gif", fps=10):
    if not ga_optimizer or not hasattr(ga_optimizer, 'population_history') or not ga_optimizer.population_history:
        return False, "No hay historial de población para generar la animación."
    if not hasattr(ga_optimizer, 'best_solution_fitness_history') or not ga_optimizer.best_solution_fitness_history:
        return False, "No hay historial de fitness para la animación."

    print(f"Iniciando generación de GIF ({len(ga_optimizer.population_history)} frames, FPS={fps})... Esto puede tardar.")
    fig_anim, anim_object = animator.create_ga_animation(
        ga_optimizer.population_history, func_str, x_range, optimization_type,
        ga_optimizer.best_solution_fitness_history, interval=int(1000/fps) # Intervalo en ms
    )
    if anim_object:
        try:
            anim_object.save(filename, writer='pillow', fps=fps) 
            plt.close(fig_anim) # Muy importante cerrar la figura de la animación
            return True, f"Animación GIF guardada como {filename}"
        except Exception as e:
            plt.close(fig_anim) # Asegurar cierre incluso en error
            error_msg = f"Error al guardar GIF: {e}.\n{traceback.format_exc()}\n"
            error_msg += "Asegúrate de tener 'Pillow' instalado y actualizado.\n"
            error_msg += "Para MP4, necesitarías 'moviepy' y 'ffmpeg'."
            return False, error_msg
    else:
        return False, "No se pudo generar el objeto de animación (anim_object es None)."