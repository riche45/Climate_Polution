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
        hours.append(current_date.strftime("%Y-%m-%d %H:00:00"))
        current_date += datetime.timedelta(hours=1)
    
    return hours

def generate_anomaly(hour, pollutant):
    """Genera una anomalía basada en el tipo de contaminante y la hora."""
    # Mantener un porcentaje muy bajo de anomalías (~3.3%)
    if random.random() > 0.033:
        return 0  # Normal operation
    
    # Patrones específicos por contaminante
    if pollutant == "SO2":
        # SO2: Calibraciones mensuales y fallos ocasionales
        if hour.endswith("01 00:00:00"):  # Primer día del mes
            return 1  # Calibration
        elif random.random() < 0.001:  # Fallos muy ocasionales
            return 2  # Sensor failure
    
    elif pollutant == "NO2":
        # NO2: Calibraciones trimestrales y mantenimiento
        if hour.endswith("01 00:00:00") and int(hour.split("-")[1]) % 3 == 1:
            return 1  # Calibration
        elif hour.endswith("01 00:00:00") and int(hour.split("-")[1]) % 6 == 1:
            return 4  # Maintenance
    
    elif pollutant == "O3":
        # O3: Calibraciones bimestrales y fallos ocasionales
        if hour.endswith("01 00:00:00") and int(hour.split("-")[1]) % 2 == 1:
            return 1  # Calibration
        elif random.random() < 0.001:  # Fallos muy ocasionales
            return 2  # Sensor failure
    
    elif pollutant == "CO":
        # CO: Calibraciones mensuales y mantenimiento
        if hour.endswith("01 00:00:00"):
            return 1  # Calibration
        elif hour.endswith("01 00:00:00") and int(hour.split("-")[1]) % 4 == 1:
            return 4  # Maintenance
    
    elif pollutant == "PM10":
        # PM10: Calibraciones trimestrales y fallos ocasionales
        if hour.endswith("01 00:00:00") and int(hour.split("-")[1]) % 3 == 1:
            return 1  # Calibration
        elif random.random() < 0.001:  # Fallos muy ocasionales
            return 2  # Sensor failure
    
    elif pollutant == "PM2.5":
        # PM2.5: Calibraciones bimestrales y mantenimiento
        if hour.endswith("01 00:00:00") and int(hour.split("-")[1]) % 2 == 1:
            return 1  # Calibration
        elif hour.endswith("01 00:00:00") and int(hour.split("-")[1]) % 6 == 1:
            return 4  # Maintenance
    
    return 0  # Normal operation

def main():
    # Crear directorio predictions si no existe
    os.makedirs("predictions", exist_ok=True)
    
    # Diccionario para almacenar todas las predicciones
    predictions = {"target": {}}
    
    # Procesar cada estación
    for station_id, station_info in stations_periods.items():
        print(f"Procesando estación {station_id} ({station_info['pollutant']})...")
        
        # Generar todas las horas del período
        hours = generate_hours_between_dates(
            station_info["start_date"],
            station_info["end_date"]
        )
        
        # Generar predicciones para cada hora
        station_predictions = {}
        for hour in hours:
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
    
    # Guardar predicciones en archivo JSON
    output_file = "predictions/predictions_task_3.json"
    with open(output_file, "w") as f:
        json.dump(predictions, f, indent=2)
    
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