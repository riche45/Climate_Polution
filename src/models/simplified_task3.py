import json
import datetime
import os
import random
import numpy as np

# Definir las estaciones y sus períodos
stations_periods = {
    "206": {
        "pollutant": "SO2",
        "start_date": "2023-01-01",
        "end_date": "2023-12-31"
    },
    "211": {
        "pollutant": "NO2",
        "start_date": "2023-01-01",
        "end_date": "2023-12-31"
    },
    "217": {
        "pollutant": "O3",
        "start_date": "2023-01-01",
        "end_date": "2023-12-31"
    },
    "219": {
        "pollutant": "CO",
        "start_date": "2023-01-01",
        "end_date": "2023-12-31"
    },
    "225": {
        "pollutant": "PM10",
        "start_date": "2023-01-01",
        "end_date": "2023-12-31"
    },
    "228": {
        "pollutant": "PM2.5",
        "start_date": "2023-01-01",
        "end_date": "2023-12-31"
    }
}

def generate_hours_between_dates(start_date_str, end_date_str):
    """Genera todas las horas entre dos fechas."""
    start_date = datetime.datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.datetime.strptime(end_date_str, "%Y-%m-%d")
    current_date = start_date
    hours = []
    
    while current_date <= end_date:
        for hour in range(24):
            hours.append(current_date.replace(hour=hour).strftime("%Y-%m-%d %H:00:00"))
        current_date += datetime.timedelta(days=1)
    
    return hours

def generate_anomaly(hour, pollutant):
    """Genera una anomalía basada en el tipo de contaminante y la hora."""
    # Convertir la hora a datetime para mejor manejo
    dt = datetime.datetime.strptime(hour, "%Y-%m-%d %H:00:00")
    
    # Calibraciones programadas (primer día del mes a las 00:00)
    if dt.day == 1 and dt.hour == 0:
        if pollutant in ["SO2", "CO"]:  # Mensual
            return 1
        elif pollutant in ["NO2", "PM10"] and dt.month % 3 == 1:  # Trimestral
            return 1
        elif pollutant in ["O3", "PM2.5"] and dt.month % 2 == 1:  # Bimestral
            return 1
    
    # Mantenimiento programado (primer lunes de cada mes)
    if dt.weekday() == 0:  # Es lunes
        if dt.day <= 7:  # Primera semana del mes
            if pollutant in ["NO2", "PM2.5"] and dt.month % 6 == 1:  # Cada 6 meses
                return 4
            elif pollutant in ["CO"] and dt.month % 4 == 1:  # Cada 4 meses
                return 4
    
    # Fallos de sensor (muy ocasionales, ~0.1%)
    if random.random() < 0.001:
        if pollutant in ["SO2", "O3", "PM10"]:
            return 2
    
    return 0  # Operación normal

def main():
    # Crear directorio predictions si no existe
    os.makedirs("predictions", exist_ok=True)
    
    # Diccionario para almacenar todas las predicciones
    predictions = {"target": {}}
    
    # Generar todas las horas del año una sola vez
    all_hours = generate_hours_between_dates("2023-01-01", "2023-12-31")
    
    # Procesar cada estación
    for station_id, station_info in stations_periods.items():
        print(f"Procesando estación {station_id} ({station_info['pollutant']})...")
        
        # Generar predicciones para cada hora
        station_predictions = {}
        for hour in all_hours:
            anomaly = generate_anomaly(hour, station_info["pollutant"])
            station_predictions[hour] = anomaly
        
        # Almacenar predicciones de la estación
        predictions["target"][station_id] = station_predictions
        
        # Imprimir estadísticas de la estación
        total_hours = len(station_predictions)
        anomaly_hours = sum(1 for anomaly in station_predictions.values() if anomaly != 0)
        print(f"Total de horas: {total_hours}")
        print(f"Horas con anomalías: {anomaly_hours}")
        print(f"Porcentaje de anomalías: {(anomaly_hours/total_hours)*100:.2f}%")
        print("---")
    
    # Verificar que todas las estaciones tienen todas las horas
    expected_hours = len(all_hours)
    for station_id, predictions_dict in predictions["target"].items():
        if len(predictions_dict) != expected_hours:
            print(f"ADVERTENCIA: La estación {station_id} tiene {len(predictions_dict)} horas, se esperaban {expected_hours}")
    
    # Guardar predicciones en archivo JSON
    output_file = "predictions/predictions_task_3.json"
    with open(output_file, "w") as f:
        json.dump(predictions, f)
    
    print(f"\nArchivo de predicciones guardado en: {output_file}")
    
    # Imprimir estadísticas globales
    total_hours = sum(len(p) for p in predictions["target"].values())
    total_anomalies = sum(1 for station in predictions["target"].values() for anomaly in station.values() if anomaly != 0)
    print(f"\nEstadísticas globales:")
    print(f"Total de horas: {total_hours}")
    print(f"Total de anomalías: {total_anomalies}")
    print(f"Porcentaje global de anomalías: {(total_anomalies/total_hours)*100:.2f}%")

if __name__ == "__main__":
    main() 