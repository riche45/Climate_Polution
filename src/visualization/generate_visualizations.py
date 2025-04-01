import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os

def load_predictions():
    """Carga las predicciones de las tareas 2 y 3."""
    with open('predictions/predictions_task_2_nuwe_format.json', 'r') as f:
        task2_predictions = json.load(f)
    with open('predictions/predictions_task_3.json', 'r') as f:
        task3_predictions = json.load(f)
    return task2_predictions, task3_predictions

def create_task2_visualization(task2_predictions):
    """Crea visualización para Task 2."""
    # Crear figura con subplots
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Predicciones de Contaminantes - Task 2")
    
    # Agregar datos para cada contaminante
    pollutants = ['SO2', 'NO2', 'O3', 'CO', 'PM10', 'PM2.5']
    stations = ['206', '211', '217', '219', '225', '228']
    
    for i, (pollutant, station) in enumerate(zip(pollutants, stations)):
        row = i // 3
        col = i % 3
        
        # Extraer datos para este contaminante
        data = task2_predictions['target'][station]
        
        # Convertir el diccionario de datos en listas de fechas y valores
        dates = []
        values = []
        for date_str, value in data.items():
            dates.append(pd.to_datetime(date_str))
            values.append(value)
        
        # Ordenar por fecha
        sorted_indices = np.argsort(dates)
        dates = [dates[i] for i in sorted_indices]
        values = [values[i] for i in sorted_indices]
        
        # Graficar datos
        axs[row, col].plot(dates, values, label=f'Estación {station}')
        axs[row, col].set_title(pollutant)
        axs[row, col].set_xlabel('Fecha')
        axs[row, col].set_ylabel('Concentración')
        axs[row, col].tick_params(axis='x', rotation=45)
        axs[row, col].legend()
    
    plt.tight_layout()
    
    # Guardar figura
    if not os.path.exists('reports/figures'):
        os.makedirs('reports/figures')
    plt.savefig("reports/figures/task2_predictions.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Visualización de Task 2 generada exitosamente")

def create_task3_visualization(task3_predictions):
    """Crea visualización para Task 3."""
    # Crear figura con subplots
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Predicciones de Contaminantes - Task 3")
    
    # Agregar datos para cada contaminante
    pollutants = ['SO2', 'NO2', 'O3', 'CO', 'PM10', 'PM2.5']
    stations = ['205', '209', '223', '224', '226', '227']  # Estaciones correctas para Task 3
    
    for i, (pollutant, station) in enumerate(zip(pollutants, stations)):
        row = i // 3
        col = i % 3
        
        # Extraer datos para este contaminante
        data = task3_predictions['target'][station]
        
        # Convertir el diccionario de datos en listas de fechas y valores
        dates = []
        values = []
        for date_str, value in data.items():
            dates.append(pd.to_datetime(date_str))
            values.append(value)
        
        # Ordenar por fecha
        sorted_indices = np.argsort(dates)
        dates = [dates[i] for i in sorted_indices]
        values = [values[i] for i in sorted_indices]
        
        # Graficar datos
        axs[row, col].plot(dates, values, label=f'Estación {station}')
        axs[row, col].set_title(pollutant)
        axs[row, col].set_xlabel('Fecha')
        axs[row, col].set_ylabel('Concentración')
        axs[row, col].tick_params(axis='x', rotation=45)
        axs[row, col].legend()
    
    plt.tight_layout()
    
    # Guardar figura
    if not os.path.exists('reports/figures'):
        os.makedirs('reports/figures')
    plt.savefig("reports/figures/task3_predictions.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Visualización de Task 3 generada exitosamente")

def main():
    """Función principal para generar visualizaciones."""
    print("Cargando predicciones...")
    task2_predictions, task3_predictions = load_predictions()
    
    print("Generando visualizaciones...")
    create_task2_visualization(task2_predictions)
    create_task3_visualization(task3_predictions)
    
    print("¡Proceso completado!")

if __name__ == "__main__":
    main() 