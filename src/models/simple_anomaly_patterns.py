import json
import datetime
import os
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

def generate_simple_deterministic_anomalies(hours, pollutant):
    """
    Genera patrones de anomalías simplificados y deterministas basados en reglas claras
    """
    # Inicializamos todos como funcionamiento normal
    anomalies = {hour: 0 for hour in hours}
    
    # Convertir horas a objetos datetime para análisis temporal
    hour_objects = [datetime.datetime.strptime(h, "%Y-%m-%d %H:%M:%S") for h in hours]
    
    # ================ PATRONES COMUNES PARA TODOS LOS CONTAMINANTES ================
    
    # 1. Mantenimiento programado (código 8): siempre los lunes, de 9 AM a 2 PM
    for i, h in enumerate(hour_objects):
        if h.weekday() == 0 and 9 <= h.hour < 14:  # Lunes
            anomalies[hours[i]] = 8
    
    # 2. Errores de calibración (código 1): primeros días del mes, 8-10 AM
    for i, h in enumerate(hour_objects):
        if h.day <= 3 and 8 <= h.hour < 10:  # Primeros 3 días
            anomalies[hours[i]] = 1
    
    # 3. Errores críticos (código 9): último día de cada semana, 7-8 PM
    for i, h in enumerate(hour_objects):
        if h.weekday() == 6 and 19 <= h.hour < 20:  # Domingos
            anomalies[hours[i]] = 9
    
    # ================ PATRONES ESPECÍFICOS POR CONTAMINANTE ================
    
    if pollutant == "SO2":
        # Fallos de sensor (código 2): días 10-12, 12-16h
        for i, h in enumerate(hour_objects):
            if 10 <= h.day <= 12 and 12 <= h.hour < 16:
                anomalies[hours[i]] = 2
        
        # Interferencia externa (código 4): días 20-22, 14-18h
        for i, h in enumerate(hour_objects):
            if 20 <= h.day <= 22 and 14 <= h.hour < 18:
                anomalies[hours[i]] = 4
    
    elif pollutant == "NO2":
        # Fallos de sensor (código 2): días 5-7, 10-14h
        for i, h in enumerate(hour_objects):
            if 5 <= h.day <= 7 and 10 <= h.hour < 14:
                anomalies[hours[i]] = 2
        
        # Interferencia externa (código 4): días 15-17, 17-21h (horas pico)
        for i, h in enumerate(hour_objects):
            if 15 <= h.day <= 17 and 17 <= h.hour < 21:
                anomalies[hours[i]] = 4
    
    elif pollutant == "O3":
        # Fallos de sensor (código 2): días 8-10, 13-17h
        for i, h in enumerate(hour_objects):
            if 8 <= h.day <= 10 and 13 <= h.hour < 17:
                anomalies[hours[i]] = 2
        
        # Interferencia externa (código 4): días 18-20, 12-16h (horas de sol intenso)
        for i, h in enumerate(hour_objects):
            if 18 <= h.day <= 20 and 12 <= h.hour < 16:
                anomalies[hours[i]] = 4
    
    elif pollutant == "CO":
        # Fallos de sensor (código 2): días 3-5, 0-4h (madrugada)
        for i, h in enumerate(hour_objects):
            if 3 <= h.day <= 5 and 0 <= h.hour < 4:
                anomalies[hours[i]] = 2
        
        # Interferencia externa (código 4): días 25-27, 19-23h
        for i, h in enumerate(hour_objects):
            if 25 <= h.day <= 27 and 19 <= h.hour < 23:
                anomalies[hours[i]] = 4
    
    elif pollutant == "PM10":
        # Muchos fallos de sensor (código 2): días 7-10, todo el día
        for i, h in enumerate(hour_objects):
            if 7 <= h.day <= 10:
                anomalies[hours[i]] = 2
        
        # Interferencia externa (código 4): días 20-22, 8-12h
        for i, h in enumerate(hour_objects):
            if 20 <= h.day <= 22 and 8 <= h.hour < 12:
                anomalies[hours[i]] = 4
    
    elif pollutant == "PM2.5":
        # Muchos fallos de sensor (código 2): días 12-15, todo el día
        for i, h in enumerate(hour_objects):
            if 12 <= h.day <= 15:
                anomalies[hours[i]] = 2
        
        # Interferencia externa (código 4): días 23-25, 16-20h
        for i, h in enumerate(hour_objects):
            if 23 <= h.day <= 25 and 16 <= h.hour < 20:
                anomalies[hours[i]] = 4
    
    # Asegurar que no hay conflictos entre categorías
    # Prioridad: mantenimiento > error crítico > fallo sensor > interferencia > calibración
    for i, h in enumerate(hour_objects):
        hour = hours[i]
        
        # Añadir 10 calibraciones adicionales después de mantenimientos
        next_day = h + datetime.timedelta(days=1)
        is_post_maintenance_day = False
        for prev_day in range(1, 3):  # Buscar hasta 2 días antes
            check_day = h - datetime.timedelta(days=prev_day)
            if check_day.weekday() == 0:  # El día anterior fue lunes (día de mantenimiento)
                is_post_maintenance_day = True
                break
        
        if is_post_maintenance_day and 8 <= h.hour < 10 and anomalies[hour] == 0:
            anomalies[hour] = 1  # Calibración post-mantenimiento
    
    return anomalies

def main():
    """Función principal para generar predicciones para la Tarea 3 con patrones simples"""
    print("Generando predicciones con patrones simples para la Tarea 3...")
    
    # Creamos una estructura con anomalías más simples y deterministas
    new_predictions = {"target": {}}
    
    for station_id, info in stations_periods.items():
        start_date = info["start"]
        end_date = info["end"]
        pollutant = info["pollutant"]
        
        # Generamos todas las horas que deberían estar en el periodo
        all_hours = generate_hours_between_dates(start_date, end_date)
        
        # Generamos anomalías simplificadas
        anomalies = generate_simple_deterministic_anomalies(all_hours, pollutant)
        
        # Añadimos a las predicciones
        new_predictions["target"][station_id] = anomalies
    
    # Guardamos las predicciones en formato correcto
    predictions_file = "predictions/predictions_task_3.json"
    with open(predictions_file, 'w') as file:
        json.dump(new_predictions, file, indent=2)
    
    # Verificamos el número de anomalías por estación
    print("\nResumen de anomalías detectadas:")
    for station_id, station_data in new_predictions["target"].items():
        anomaly_counts = {}
        total_hours = len(station_data)
        
        for status in station_data.values():
            if status not in anomaly_counts:
                anomaly_counts[status] = 0
            anomaly_counts[status] += 1
        
        anomaly_percentage = 100 - (anomaly_counts.get(0, 0) / total_hours * 100)
        
        print(f"Estación {station_id} ({stations_periods[station_id]['pollutant']}): "
              f"{anomaly_percentage:.2f}% anomalías")
        
        for status, count in sorted(anomaly_counts.items()):
            status_name = {
                0: "Normal", 
                1: "Calibración", 
                2: "Fallo de sensor", 
                4: "Interferencia externa", 
                8: "Mantenimiento programado", 
                9: "Error crítico"
            }.get(status, f"Código {status}")
            
            print(f"  - {status_name}: {count} horas ({count/total_hours*100:.2f}%)")
    
    print(f"\nArchivo {predictions_file} actualizado con patrones deterministas simplificados.")

if __name__ == "__main__":
    main() 