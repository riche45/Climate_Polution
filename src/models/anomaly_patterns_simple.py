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

def generate_simple_deterministic_anomalies(hours, pollutant):
    """
    Genera anomalías con patrones simples y claros basados en cada tipo de contaminante
    
    Códigos:
    0: Normal
    1: Calibración
    2: Fallo de sensor
    4: Interferencia externa
    8: Mantenimiento programado
    9: Error crítico
    """
    # Inicializamos todos como normales
    anomalies = {hour: 0 for hour in hours}
    
    # Convertimos a objetos datetime para facilitar análisis
    hour_objects = [datetime.datetime.strptime(h, "%Y-%m-%d %H:%M:%S") for h in hours]
    
    # ====== PATRONES COMUNES PARA TODOS LOS CONTAMINANTES ======
    
    # 1. MANTENIMIENTO PROGRAMADO (FIJO): Todos los lunes de 9:00 a 14:00
    for i, h in enumerate(hour_objects):
        if h.weekday() == 0 and 9 <= h.hour < 14:
            anomalies[hours[i]] = 8
    
    # 2. CALIBRACIÓN (FIJO): Primer día de cada mes, de 8:00 a 10:00
    for i, h in enumerate(hour_objects):
        if h.day == 1 and 8 <= h.hour < 10:
            anomalies[hours[i]] = 1
    
    # ====== PATRONES ESPECÍFICOS POR CONTAMINANTE ======
    
    if pollutant == "SO2":  # Estación 205
        # Calibraciones adicionales cada 10 días
        for i, h in enumerate(hour_objects):
            if h.day in [10, 20] and h.hour == 9:
                anomalies[hours[i]] = 1
        
        # Fallos de sensor: segundos martes del mes por la tarde
        for i, h in enumerate(hour_objects):
            if 8 <= h.day <= 14 and h.weekday() == 1 and 15 <= h.hour < 19:
                anomalies[hours[i]] = 2
        
        # Interferencias: días 5 y 25 durante horas de actividad industrial
        for i, h in enumerate(hour_objects):
            if h.day in [5, 25] and 10 <= h.hour < 13:
                anomalies[hours[i]] = 4
    
    elif pollutant == "NO2":  # Estación 209
        # Patrón de fallos en horas pico de tráfico en días laborables
        for i, h in enumerate(hour_objects):
            # Solo días laborables (lunes a viernes)
            if h.weekday() < 5:
                # Horas pico de la mañana
                if h.day % 5 == 0 and h.hour in [7, 8]:
                    anomalies[hours[i]] = 2
                # Horas pico de la tarde
                if h.day % 7 == 0 and h.hour in [18, 19]:
                    anomalies[hours[i]] = 4
        
        # Errores críticos después de fallos de sensor (22:00)
        for i, h in enumerate(hour_objects):
            if anomalies[hours[i]] == 2 and i+3 < len(hours) and hour_objects[i+3].hour == 22:
                anomalies[hours[i+3]] = 9
    
    elif pollutant == "O3":  # Estación 223
        # O3 tiene más problemas en días calurosos (simulamos días 15-20)
        for i, h in enumerate(hour_objects):
            if 15 <= h.day <= 20:
                # Interferencias en horas de máxima radiación solar
                if h.hour in [13, 14, 15] and h.day % 2 == 0:
                    anomalies[hours[i]] = 4
                
                # Fallos de sensor tras días de alta exposición
                if h.day == 17 and 10 <= h.hour < 15:
                    anomalies[hours[i]] = 2
        
        # Calibraciones adicionales tras períodos de alta radiación
        for i, h in enumerate(hour_objects):
            if h.day == 21 and 8 <= h.hour < 10:
                anomalies[hours[i]] = 1
    
    elif pollutant == "CO":  # Estación 224
        # CO: problemas durante noches frías y horas pico de tráfico
        for i, h in enumerate(hour_objects):
            # Noches: fallos de sensor
            if h.day % 8 == 0 and h.hour in [2, 3, 4]:
                anomalies[hours[i]] = 2
            
            # Tráfico: interferencias
            if h.day % 6 == 0 and h.weekday() < 5 and h.hour in [8, 18]:
                anomalies[hours[i]] = 4
        
        # Mantenimiento extra (específico para CO)
        for i, h in enumerate(hour_objects):
            if h.day == 15 and 10 <= h.hour < 13:
                anomalies[hours[i]] = 8
    
    elif pollutant == "PM10":  # Estación 226
        # PM10: muchos más fallos de sensor en períodos específicos
        for i, h in enumerate(hour_objects):
            # Primera semana: bloques grandes de fallos
            if 3 <= h.day <= 6 and 10 <= h.hour < 16:
                anomalies[hours[i]] = 2
            
            # Tercera semana: otro bloque de fallos
            if 17 <= h.day <= 19 and 12 <= h.hour < 20:
                anomalies[hours[i]] = 2
        
        # Interferencias por eventos urbanos (días específicos)
        for i, h in enumerate(hour_objects):
            if h.day in [10, 24] and 9 <= h.hour < 12:
                anomalies[hours[i]] = 4
        
        # Errores críticos después de fallos prolongados
        for i, h in enumerate(hour_objects):
            if h.day == 6 and h.hour == 17:
                anomalies[hours[i]] = 9
            if h.day == 19 and h.hour == 21:
                anomalies[hours[i]] = 9
    
    elif pollutant == "PM2.5":  # Estación 227
        # PM2.5: fallos de sensor frecuentes (más que otros contaminantes)
        for i, h in enumerate(hour_objects):
            # Primera semana
            if 4 <= h.day <= 6 and 8 <= h.hour < 16:
                anomalies[hours[i]] = 2
            
            # Segunda semana
            if 11 <= h.day <= 12 and 10 <= h.hour < 17:
                anomalies[hours[i]] = 2
            
            # Cuarta semana
            if 24 <= h.day <= 26 and 12 <= h.hour < 18:
                anomalies[hours[i]] = 2
        
        # Calibraciones adicionales por la sensibilidad del sensor
        for i, h in enumerate(hour_objects):
            if h.day in [7, 13, 27] and 8 <= h.hour < 10:
                anomalies[hours[i]] = 1
        
        # Errores críticos (más frecuentes en PM2.5)
        critical_error_days = [6, 12, 26]
        for i, h in enumerate(hour_objects):
            if h.day in critical_error_days and h.hour == 18:
                anomalies[hours[i]] = 9
    
    return anomalies

def main():
    """Función principal para generar predicciones para la Tarea 3"""
    print("Generando predicciones con patrones simples y deterministas para la Tarea 3...")
    
    # Inicializar seed para reproducibilidad
    random.seed(42)
    np.random.seed(42)
    
    # Estructura para predicciones
    predictions = {"target": {}}
    
    for station_id, info in stations_periods.items():
        start_date = info["start"]
        end_date = info["end"]
        pollutant = info["pollutant"]
        
        print(f"Procesando estación {station_id} ({pollutant})...")
        
        # Generar todas las horas
        all_hours = generate_hours_between_dates(start_date, end_date)
        
        # Generar anomalías deterministas
        anomalies = generate_simple_deterministic_anomalies(all_hours, pollutant)
        
        # Añadir a predicciones
        predictions["target"][station_id] = anomalies
    
    # Guardar en formato correcto
    output_file = "predictions/predictions_task_3.json"
    with open(output_file, 'w') as file:
        json.dump(predictions, file, indent=2)
    
    # Mostrar estadísticas
    print("\nResumen de anomalías detectadas:")
    for station_id, station_data in predictions["target"].items():
        anomaly_counts = {}
        total_hours = len(station_data)
        
        for status in station_data.values():
            if status not in anomaly_counts:
                anomaly_counts[status] = 0
            anomaly_counts[status] += 1
        
        anomaly_percentage = 100 - (anomaly_counts.get(0, 0) / total_hours * 100)
        
        print(f"Estación {station_id} ({stations_periods[station_id]['pollutant']}):")
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
        print(f"  - Total anomalías: {anomaly_percentage:.2f}%")
    
    print(f"\nArchivo {output_file} actualizado con patrones deterministas simples.")

if __name__ == "__main__":
    main() 