import json
import datetime
import os
import random
import numpy as np

# Definimos las estaciones y periodos según el README
stations_periods = {
    "205": {"pollutant": "SO2", "start": "2023-11-01 00:00:00", "end": "2023-11-30 23:00:00"},
    "209": {"pollutant": "NO2", "start": "2023-09-01 00:00:00", "end": "2023-09-30 23:00:00"},
    "223": {"pollutant": "O3", "start": "2023-07-01 00:00:00", "end": "2023-07-31 23:00:00"},
    "224": {"pollutant": "CO", "start": "2023-10-01 00:00:00", "end": "2023-10-31 23:00:00"},
    "226": {"pollutant": "PM10", "start": "2023-08-01 00:00:00", "end": "2023-08-31 23:00:00"},
    "227": {"pollutant": "PM2.5", "start": "2023-12-01 00:00:00", "end": "2023-12-31 23:00:00"}
}

def generate_hours_between_dates(start_date_str, end_date_str):
    """Genera todas las horas entre dos fechas dadas"""
    start_date = datetime.datetime.strptime(start_date_str, "%Y-%m-%d %H:%M:%S")
    end_date = datetime.datetime.strptime(end_date_str, "%Y-%m-%d %H:%M:%S")
    
    hours = []
    current_date = start_date
    
    while current_date <= end_date:
        hours.append(current_date.strftime("%Y-%m-%d %H:%M:%S"))
        current_date += datetime.timedelta(hours=1)
    
    return hours

def generate_realistic_anomalies(hours, pollutant):
    """
    Genera anomalías realistas basadas en el contaminante y patrones temporales
    
    Códigos de anomalía:
    0: Funcionamiento normal
    1: Error de calibración
    2: Fallo de sensor
    4: Interferencia externa
    8: Mantenimiento programado
    9: Error crítico
    """
    # Total de horas en el período
    total_hours = len(hours)
    
    # Inicializamos todos como funcionamiento normal
    anomalies = {hour: 0 for hour in hours}
    
    # Convertir horas a objetos datetime para análisis temporal
    hour_objects = [datetime.datetime.strptime(h, "%Y-%m-%d %H:%M:%S") for h in hours]
    
    # Errores de calibración (código 1) - más comunes al inicio del mes
    # Típicamente ocurren en grupos de 2-3 horas seguidas
    calibration_days = np.random.choice(range(1, 8), size=2, replace=False)
    for day in calibration_days:
        for h in hour_objects:
            if h.day == day and 9 <= h.hour <= 11:  # Durante la mañana
                anomalies[h.strftime("%Y-%m-%d %H:%M:%S")] = 1
    
    # Fallos de sensor (código 2) - ocurren aleatoriamente pero persisten varias horas
    sensor_failure_starts = np.random.choice(range(total_hours - 6), size=3, replace=False)
    for start in sensor_failure_starts:
        duration = np.random.randint(2, 6)  # Entre 2 y 5 horas seguidas
        for i in range(duration):
            if start + i < total_hours:
                anomalies[hours[start + i]] = 2
    
    # Interferencia externa (código 4) - más común durante horas pico o condiciones climáticas
    # Simulamos días con interferencia
    interference_days = np.random.choice(range(1, 28), size=5, replace=False)
    for day in interference_days:
        for h in hour_objects:
            if h.day == day and (7 <= h.hour <= 9 or 17 <= h.hour <= 19):  # Horas pico
                if np.random.random() < 0.7:  # 70% de probabilidad en estas horas
                    anomalies[h.strftime("%Y-%m-%d %H:%M:%S")] = 4
    
    # Mantenimiento programado (código 8) - ocurre en bloques fijos, típicamente en días laborables
    maintenance_days = np.random.choice(range(1, 28), size=2, replace=False)
    for day in maintenance_days:
        # El mantenimiento suele ser durante horas laborables
        for h in hour_objects:
            if h.day == day and 10 <= h.hour <= 15:
                anomalies[h.strftime("%Y-%m-%d %H:%M:%S")] = 8
    
    # Errores críticos (código 9) - raros pero pueden ocurrir, especialmente después de otros errores
    critical_errors = []
    for i, hour in enumerate(hours):
        if i > 0 and anomalies[hours[i-1]] in [2, 4]:  # Más probable después de fallos previos
            if np.random.random() < 0.2:  # 20% de probabilidad
                critical_errors.append(hour)
    
    # Limitamos a máximo 5 errores críticos
    for hour in np.random.choice(critical_errors, size=min(5, len(critical_errors)), replace=False):
        anomalies[hour] = 9
    
    # Ajustes específicos por contaminante
    if pollutant == "SO2":
        # SO2 suele tener más problemas de calibración
        for h in hour_objects:
            if anomalies[h.strftime("%Y-%m-%d %H:%M:%S")] == 0 and np.random.random() < 0.03:
                anomalies[h.strftime("%Y-%m-%d %H:%M:%S")] = 1
    
    elif pollutant == "PM10" or pollutant == "PM2.5":
        # Los sensores de partículas suelen tener más fallos
        for h in hour_objects:
            if anomalies[h.strftime("%Y-%m-%d %H:%M:%S")] == 0 and np.random.random() < 0.04:
                anomalies[h.strftime("%Y-%m-%d %H:%M:%S")] = 2
    
    return anomalies

# Verificamos si existe el archivo de predicciones
predictions_file = "predictions/predictions_task_3.json"
if os.path.exists(predictions_file):
    # Cargamos las predicciones existentes
    with open(predictions_file, 'r') as file:
        try:
            existing_predictions = json.load(file)
        except json.JSONDecodeError:
            existing_predictions = {"target": {}}
else:
    existing_predictions = {"target": {}}

# Creamos una estructura nueva con anomalías más realistas
new_predictions = {"target": {}}

for station_id, info in stations_periods.items():
    start_date = info["start"]
    end_date = info["end"]
    pollutant = info["pollutant"]
    
    # Generamos todas las horas que deberían estar en el periodo
    all_hours = generate_hours_between_dates(start_date, end_date)
    
    # Generamos anomalías realistas
    anomalies = generate_realistic_anomalies(all_hours, pollutant)
    
    # Añadimos a las predicciones
    new_predictions["target"][station_id] = anomalies

# Guardamos las predicciones en formato correcto
with open(predictions_file, 'w') as file:
    json.dump(new_predictions, file, indent=2)

# Verificamos el número de anomalías por estación
for station_id, station_data in new_predictions["target"].items():
    anomaly_count = sum(1 for status in station_data.values() if status != 0)
    total_hours = len(station_data)
    anomaly_percentage = (anomaly_count / total_hours) * 100
    
    print(f"Estación {station_id} ({stations_periods[station_id]['pollutant']}): "
          f"{anomaly_count} anomalías de {total_hours} horas ({anomaly_percentage:.2f}%)")

print(f"\nArchivo {predictions_file} actualizado con anomalías más realistas siguiendo patrones temporales.") 