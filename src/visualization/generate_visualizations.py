import os
import sys
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
import numpy as np
from datetime import datetime

# Añadir el directorio raíz al path de Python
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

from src.visualization.task2_visualizer import main as task2_main
from src.visualization.task3_visualizer import main as task3_main

def load_predictions(task2_file, task3_file):
    """Cargar predicciones de los archivos JSON."""
    with open(task2_file, 'r') as f:
        task2_pred = json.load(f)
    with open(task3_file, 'r') as f:
        task3_pred = json.load(f)
    return task2_pred, task3_pred

def create_task2_visualization(predictions):
    """Crear visualización para Task 2."""
    # Crear figura con subplots para cada contaminante
    fig = go.Figure()
    
    for station_code, data in predictions['target'].items():
        pollutant = data['pollutant']
        values = data['values']
        dates = pd.date_range(start=data['start'], end=data['end'], freq='H')
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=values,
            name=f'{pollutant} - Estación {station_code}',
            mode='lines+markers'
        ))
    
    fig.update_layout(
        title='Predicciones de Contaminantes por Estación',
        xaxis_title='Fecha',
        yaxis_title='Concentración',
        template='plotly_white',
        height=600
    )
    
    # Guardar figura
    output_path = Path('reports/figures/task2_predictions.png')
    fig.write_image(str(output_path))

def create_task3_visualization(predictions):
    """Crear visualización para Task 3."""
    # Crear figura con subplots para cada contaminante
    fig = go.Figure()
    
    for station_code, data in predictions['target'].items():
        pollutant = data['pollutant']
        values = data['values']
        dates = pd.date_range(start=data['start'], end=data['end'], freq='H')
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=values,
            name=f'{pollutant} - Estación {station_code}',
            mode='lines+markers'
        ))
    
    fig.update_layout(
        title='Predicciones Optimizadas de Calidad del Aire',
        xaxis_title='Fecha',
        yaxis_title='Concentración',
        template='plotly_white',
        height=600
    )
    
    # Guardar figura
    output_path = Path('reports/figures/task3_predictions.png')
    fig.write_image(str(output_path))

def main():
    """Función principal para generar visualizaciones."""
    print("Generando visualizaciones...")
    
    # Cargar predicciones
    task2_pred, task3_pred = load_predictions(
        'predictions/task2_predictions.json',
        'predictions/task3_predictions.json'
    )
    
    # Crear visualizaciones
    create_task2_visualization(task2_pred)
    create_task3_visualization(task3_pred)
    
    print("Visualizaciones generadas exitosamente!")

if __name__ == "__main__":
    main() 