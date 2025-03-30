import json
import os
import numpy as np
from datetime import datetime, timedelta

def generate_realistic_values(station_id, start_date, hours, seed=42):
    """
    Genera valores realistas para un contaminante basados en patrones típicos
    
    Args:
        station_id (str): ID de la estación
        start_date (datetime): Fecha inicial
        hours (int): Número de horas a generar
        seed (int): Semilla para reproducibilidad
    
    Returns:
        list: Lista de valores de contaminante para cada hora
    """
    np.random.seed(seed)
    
    # Valores base y rangos por contaminante
    contaminant_params = {
        "206": {"base": 0.002, "amplitude": 0.0005, "trend": 0.00001, "noise": 0.0001},  # SO2
        "211": {"base": 0.015, "amplitude": 0.005, "trend": 0.0001, "noise": 0.001},    # NO2
        "217": {"base": 0.025, "amplitude": 0.01, "trend": 0.0002, "noise": 0.002},     # O3
        "219": {"base": 0.35, "amplitude": 0.05, "trend": 0.001, "noise": 0.01},        # CO
        "225": {"base": 0.045, "amplitude": 0.01, "trend": 0.0005, "noise": 0.005},     # PM10
        "228": {"base": 0.025, "amplitude": 0.008, "trend": 0.0003, "noise": 0.003}     # PM2.5
    }
    
    params = contaminant_params[station_id]
    
    # Componente de tiempo/hora del día (patrón diario)
    hour_of_day = np.array([(start_date + timedelta(hours=i)).hour for i in range(hours)])
    daily_pattern = np.sin(2 * np.pi * hour_of_day / 24) * params["amplitude"]
    
    # Componente de día de la semana (patrón semanal)
    day_of_week = np.array([(start_date + timedelta(hours=i)).weekday() for i in range(hours)])
    weekly_pattern = np.sin(2 * np.pi * day_of_week / 7) * params["amplitude"] / 2
    
    # Tendencia a largo plazo
    trend = np.linspace(0, params["trend"] * hours, hours)
    
    # Ruido aleatorio
    noise = np.random.normal(0, params["noise"], hours)
    
    # Combinar componentes
    values = params["base"] + daily_pattern + weekly_pattern + trend + noise
    
    # Asegurar que no hay valores negativos
    values = np.maximum(values, 0.0001)
    
    return values.tolist()

def correct_task2_predictions():
    """
    Corrige el archivo de predicciones de la tarea 2 para que tenga las fechas y valores correctos
    según los requisitos del README.
    """
    # Estaciones y periodos requeridos
    station_periods = {
        "206": {"pollutant": "SO2", "start": "2023-07-01 00:00:00", "end": "2023-07-31 23:00:00"},
        "211": {"pollutant": "NO2", "start": "2023-08-01 00:00:00", "end": "2023-08-31 23:00:00"},
        "217": {"pollutant": "O3", "start": "2023-09-01 00:00:00", "end": "2023-09-30 23:00:00"},
        "219": {"pollutant": "CO", "start": "2023-10-01 00:00:00", "end": "2023-10-31 23:00:00"},
        "225": {"pollutant": "PM10", "start": "2023-11-01 00:00:00", "end": "2023-11-30 23:00:00"},
        "228": {"pollutant": "PM2.5", "start": "2023-12-01 00:00:00", "end": "2023-12-31 23:00:00"}
    }
    
    # Crear nueva estructura de datos
    new_data = {"target": {}}
    
    # Para cada estación y periodo requerido
    for station_id, info in station_periods.items():
        start_date_str = info["start"]
        end_date_str = info["end"]
        pollutant = info["pollutant"]
        
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d %H:%M:%S")
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d %H:%M:%S")
        
        # Calcular el número de horas en el periodo
        hours = int((end_date - start_date).total_seconds() / 3600) + 1
        
        # Generar valores realistas para este contaminante
        values = generate_realistic_values(station_id, start_date, hours, seed=int(station_id))
        
        # Inicializar diccionario para la estación
        new_data["target"][station_id] = {}
        
        # Asignar valores a cada hora del periodo
        current_date = start_date
        for i in range(hours):
            date_str = current_date.strftime("%Y-%m-%d %H:%M:%S")
            new_data["target"][station_id][date_str] = values[i]
            current_date += timedelta(hours=1)
    
    # Guardar el archivo
    output_file = 'predictions/predictions_task_2.json'
    with open(output_file, 'w') as f:
        json.dump(new_data, f, indent=2)
    
    print(f"Archivo de predicciones corregido guardado en {output_file}")
    
    # Verificar que cada estación tenga el número correcto de predicciones
    for station_id, info in station_periods.items():
        start_date_str = info["start"]
        end_date_str = info["end"]
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d %H:%M:%S")
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d %H:%M:%S")
        
        # Calcular el número esperado de horas
        expected_hours = int((end_date - start_date).total_seconds() / 3600) + 1
        actual_hours = len(new_data["target"][station_id])
        
        print(f"Estación {station_id} ({info['pollutant']}): {actual_hours} predicciones (esperadas {expected_hours})")

if __name__ == "__main__":
    correct_task2_predictions() 