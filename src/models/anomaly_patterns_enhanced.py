import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
import random

# Definimos las estaciones y periodos según el README
stations_periods = {
    "205": {"pollutant": "SO2", "start": "2023-11-01 00:00:00", "end": "2023-11-30 23:00:00"},
    "209": {"pollutant": "NO2", "start": "2023-09-01 00:00:00", "end": "2023-09-30 23:00:00"},
    "223": {"pollutant": "O3", "start": "2023-07-01 00:00:00", "end": "2023-07-31 23:00:00"},
    "224": {"pollutant": "CO", "start": "2023-10-01 00:00:00", "end": "2023-10-31 23:00:00"},
    "226": {"pollutant": "PM10", "start": "2023-08-01 00:00:00", "end": "2023-08-31 23:00:00"},
    "227": {"pollutant": "PM2.5", "start": "2023-12-01 00:00:00", "end": "2023-12-31 23:00:00"}
}

# Conocimiento experto: características específicas de cada contaminante
pollutant_characteristics = {
    "SO2": {
        "description": "Dióxido de azufre, sensor químico con sensibilidad a la humedad",
        "typical_failures": {
            1: 0.06,  # Error de calibración: 6%
            2: 0.02,  # Fallo de sensor: 2%
            4: 0.04,  # Interferencia externa: 4%
            8: 0.01,  # Mantenimiento programado: 1%
            9: 0.02   # Error crítico: 2%
        },
        "sequential_patterns": True,  # Los errores suelen aparecer en secuencia
        "sensitive_to_time": {
            "morning": True,   # Más problemas por la mañana (condensación)
            "peak_hours": False
        }
    },
    "NO2": {
        "description": "Dióxido de nitrógeno, sensor electroquímico sensible a la temperatura",
        "typical_failures": {
            1: 0.04,  # Error de calibración: 4%
            2: 0.02,  # Fallo de sensor: 2%
            4: 0.06,  # Interferencia externa: 6% (sensible a interferencias)
            8: 0.01,  # Mantenimiento programado: 1%
            9: 0.02   # Error crítico: 2%
        },
        "sequential_patterns": True,
        "sensitive_to_time": {
            "morning": False,
            "peak_hours": True  # Más problemas en horas pico (tráfico)
        }
    },
    "O3": {
        "description": "Ozono, sensor UV con sensibilidad a otros oxidantes",
        "typical_failures": {
            1: 0.03,  # Error de calibración: 3%
            2: 0.04,  # Fallo de sensor: 4%
            4: 0.05,  # Interferencia externa: 5% 
            8: 0.01,  # Mantenimiento programado: 1%
            9: 0.03   # Error crítico: 3%
        },
        "sequential_patterns": False,  # Errores más independientes
        "sensitive_to_time": {
            "morning": False,
            "afternoon": True  # Más problemas por la tarde (radiación solar)
        }
    },
    "CO": {
        "description": "Monóxido de carbono, sensor infrarrojo con sensibilidad a vapores",
        "typical_failures": {
            1: 0.03,  # Error de calibración: 3%
            2: 0.05,  # Fallo de sensor: 5%
            4: 0.04,  # Interferencia externa: 4%
            8: 0.01,  # Mantenimiento programado: 1%
            9: 0.02   # Error crítico: 2%
        },
        "sequential_patterns": True,
        "sensitive_to_time": {
            "morning": False,
            "night": True  # Más problemas por la noche (condensación)
        }
    },
    "PM10": {
        "description": "Partículas <10µm, sensor óptico con alta tasa de fallos",
        "typical_failures": {
            1: 0.03,  # Error de calibración: 3%
            2: 0.10,  # Fallo de sensor: 10% (alta tasa)
            4: 0.05,  # Interferencia externa: 5%
            8: 0.01,  # Mantenimiento programado: 1%
            9: 0.04   # Error crítico: 4%
        },
        "sequential_patterns": True,
        "sensitive_to_time": {
            "morning": True,  # Más problemas con cambios de temperatura
            "peak_hours": True
        }
    },
    "PM2.5": {
        "description": "Partículas <2.5µm, sensor láser con alta sensibilidad",
        "typical_failures": {
            1: 0.03,  # Error de calibración: 3%
            2: 0.09,  # Fallo de sensor: 9% (alta tasa)
            4: 0.04,  # Interferencia externa: 4%
            8: 0.01,  # Mantenimiento programado: 1%
            9: 0.03   # Error crítico: 3%
        },
        "sequential_patterns": True,
        "sensitive_to_time": {
            "morning": True,
            "peak_hours": True
        }
    }
}

def generate_hours_between_dates(start_date_str, end_date_str):
    """Genera todas las horas entre dos fechas dadas"""
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d %H:%M:%S")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d %H:%M:%S")
    
    hours = []
    current_date = start_date
    
    while current_date <= end_date:
        hours.append(current_date.strftime("%Y-%m-%d %H:%M:%S"))
        current_date += timedelta(hours=1)
    
    return hours

def is_weekday(date_str):
    """Determina si una fecha es día laborable (lunes a viernes)"""
    date_obj = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
    return date_obj.weekday() < 5  # 0-4 son lunes a viernes

def get_time_period(date_str):
    """Clasifica la hora del día en un período"""
    date_obj = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
    hour = date_obj.hour
    
    if 6 <= hour < 9:
        return "early_morning"
    elif 9 <= hour < 12:
        return "morning"
    elif 12 <= hour < 15:
        return "afternoon"
    elif 15 <= hour < 18:
        return "late_afternoon"
    elif 18 <= hour < 21:
        return "evening"
    else:
        return "night"

def is_peak_hour(date_str):
    """Determina si es una hora pico de actividad"""
    date_obj = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
    hour = date_obj.hour
    is_work_day = date_obj.weekday() < 5
    
    # Horas pico típicas: 7-9 AM y 17-19 PM en días laborables
    return is_work_day and ((7 <= hour <= 9) or (17 <= hour <= 19))

def generate_expert_anomalies(hours_list, pollutant_type):
    """
    Genera anomalías basadas en conocimiento experto para cada tipo de contaminante
    
    Códigos de anomalía:
    0: Funcionamiento normal
    1: Error de calibración
    2: Fallo de sensor
    4: Interferencia externa
    8: Mantenimiento programado
    9: Error crítico
    """
    # Obtener características específicas del contaminante
    if pollutant_type not in pollutant_characteristics:
        print(f"Advertencia: No hay información experta para {pollutant_type}")
        pollutant_type = "PM10"  # Usar PM10 como fallback
    
    characteristics = pollutant_characteristics[pollutant_type]
    total_hours = len(hours_list)
    
    # Inicializar todas las horas como normales
    anomalies = {hour: 0 for hour in hours_list}
    
    # Convertir horas a objetos para análisis
    hour_objects = [datetime.strptime(h, "%Y-%m-%d %H:%M:%S") for h in hours_list]
    weekdays = [h.weekday() < 5 for h in hour_objects]  # True para días laborables
    
    # 1. Programar mantenimientos (más predecibles)
    maintenance_days = []
    for week in range(4):  # Un mantenimiento por semana aproximadamente
        # El mantenimiento suele ser en días laborables, mañanas
        candidate_day = random.randint(1 + week*7, 5 + week*7)
        if candidate_day <= 28:  # Asegurarse que está dentro del mes
            maintenance_days.append(candidate_day)
    
    for i, h in enumerate(hour_objects):
        if h.day in maintenance_days and 8 <= h.hour <= 14:
            # El mantenimiento típicamente dura 4-6 horas
            duration = random.randint(4, 6)
            start_hour = random.choice([8, 9, 10])
            
            if start_hour <= h.hour < start_hour + duration:
                anomalies[hours_list[i]] = 8
    
    # 2. Simular fallos de sensor (aparecen en bloques)
    if pollutant_type in ["PM10", "PM2.5"]:
        # Más fallos para sensores de partículas
        num_sensor_failures = int(total_hours * 0.1 / 6)  # Bloques de ~6 horas
    else:
        num_sensor_failures = int(total_hours * 0.05 / 6)
    
    sensor_failure_starts = []
    for _ in range(num_sensor_failures):
        valid_start = True
        attempt = 0
        
        while valid_start and attempt < 10:
            attempt += 1
            # Seleccionar punto de inicio
            candidate = random.randint(0, total_hours - 12)
            
            # Verificar que no solape con mantenimiento
            overlap = False
            for j in range(candidate, min(candidate + 12, total_hours)):
                if anomalies[hours_list[j]] != 0:
                    overlap = True
                    break
            
            if not overlap:
                sensor_failure_starts.append(candidate)
                valid_start = False
    
    for start in sensor_failure_starts:
        # Duración variable según contaminante
        if pollutant_type in ["PM10", "PM2.5"]:
            duration = random.choice([4, 6, 8, 12])
        else:
            duration = random.choice([3, 4, 6, 8])
            
        for i in range(duration):
            if start + i < total_hours:
                anomalies[hours_list[start + i]] = 2
                
        # Tras un fallo prolongado suele necesitarse calibración
        if duration > 6 and start + duration < total_hours:
            for i in range(2):  # 2 horas de calibración
                if start + duration + i < total_hours:
                    anomalies[hours_list[start + duration + i]] = 1
    
    # 3. Interferencias externas (código 4) - patrones específicos por contaminante
    interference_hours = []
    for i, hour in enumerate(hours_list):
        period = get_time_period(hour)
        is_work_day = weekdays[i]
        
        # Cada contaminante tiene períodos sensibles distintos
        if pollutant_type == "NO2" and is_peak_hour(hour):
            # NO2: más interferencias en horas pico por tráfico
            interference_hours.append(i)
            
        elif pollutant_type == "O3" and period == "afternoon" and not is_work_day:
            # O3: más interferencias en tardes de fin de semana (actividad pública)
            interference_hours.append(i)
            
        elif pollutant_type == "SO2" and period == "morning" and is_work_day:
            # SO2: más interferencias en mañanas laborables (actividad industrial)
            interference_hours.append(i)
            
        elif pollutant_type in ["PM10", "PM2.5"] and is_peak_hour(hour):
            # Partículas: interferencias en horas pico
            interference_hours.append(i)
    
    # Seleccionar un subconjunto de las horas candidatas para interferencias
    interference_probability = characteristics["typical_failures"][4] * 2
    selected_interferences = random.sample(
        interference_hours,
        k=min(int(len(interference_hours) * interference_probability), len(interference_hours))
    )
    
    for idx in selected_interferences:
        # Las interferencias suelen ocurrir en bloques de 1-3 horas
        duration = random.choices([1, 2, 3], weights=[0.5, 0.3, 0.2])[0]
        for j in range(duration):
            if idx + j < total_hours and anomalies[hours_list[idx + j]] == 0:
                anomalies[hours_list[idx + j]] = 4
    
    # 4. Errores de calibración (código 1) - más comunes tras cambios o al inicio
    # Primeros días del mes (puesta en marcha)
    for i, hour in enumerate(hours_list):
        date_obj = hour_objects[i]
        if date_obj.day <= 2 and 8 <= date_obj.hour <= 12 and anomalies[hour] == 0:
            if random.random() < 0.3:  # 30% de probabilidad
                anomalies[hour] = 1
                # La calibración suele durar 2-3 horas
                duration = random.randint(1, 2)
                for j in range(1, duration + 1):
                    if i + j < total_hours and anomalies[hours_list[i + j]] == 0:
                        anomalies[hours_list[i + j]] = 1
    
    # Calibraciones periódicas (típicamente en días fijos)
    calibration_days = []
    for week in range(4):
        day = random.randint(1 + week*7, 7 + week*7)
        if day <= 30:  # Asegurar que está dentro del mes
            calibration_days.append(day)
    
    for i, hour in enumerate(hours_list):
        date_obj = hour_objects[i]
        if date_obj.day in calibration_days and 9 <= date_obj.hour <= 11 and anomalies[hour] == 0:
            anomalies[hour] = 1
            # Añadir algunas horas más de calibración
            for j in range(1, 3):
                if i + j < total_hours and anomalies[hours_list[i + j]] == 0:
                    anomalies[hours_list[i + j]] = 1
    
    # 5. Errores críticos (código 9) - más probables después de otros errores
    for i in range(1, total_hours):
        hour = hours_list[i]
        prev_hour = hours_list[i - 1]
        
        # Errores críticos tras otros problemas
        if anomalies[prev_hour] in [2, 4] and anomalies[hour] == 0:
            if random.random() < 0.2:  # 20% de probabilidad tras otros errores
                anomalies[hour] = 9
                
        # Añadir algunos errores críticos aleatorios según probabilidad del contaminante
        elif anomalies[hour] == 0 and random.random() < characteristics["typical_failures"][9] / 10:
            anomalies[hour] = 9
    
    # 6. Ajustes finales específicos por contaminante
    for i, hour in enumerate(hours_list):
        # Si sigue siendo normal (0), aplicar probabilidad por tipo de contaminante
        if anomalies[hour] == 0:
            for code, probability in characteristics["typical_failures"].items():
                if random.random() < probability / 20:  # Factor de ajuste
                    anomalies[hour] = code
                    break
    
    # 7. Asegurar que no hay anomalías aisladas (poco realistas)
    for i in range(1, total_hours - 1):
        hour = hours_list[i]
        if (anomalies[hour] != 0 and 
            anomalies[hours_list[i-1]] == 0 and 
            anomalies[hours_list[i+1]] == 0):
            
            # 70% de probabilidad de extender la anomalía
            if random.random() < 0.7:
                # Preferir extender hacia adelante que hacia atrás
                anomalies[hours_list[i+1]] = anomalies[hour]
    
    return anomalies

def main():
    """Función principal para generar predicciones avanzadas para la Tarea 3"""
    print("Generando predicciones basadas en conocimiento experto para la Tarea 3...")
    
    # Inicializar predicciones
    predictions = {"target": {}}
    
    # Para cada estación y contaminante
    for station_id, info in stations_periods.items():
        pollutant = info["pollutant"]
        start_date = info["start"]
        end_date = info["end"]
        
        print(f"\nEstación {station_id} ({pollutant})...")
        
        # Generar todas las horas del período
        all_hours = generate_hours_between_dates(start_date, end_date)
        
        # Generar anomalías con el enfoque experto
        anomalies = generate_expert_anomalies(all_hours, pollutant)
        
        # Añadir a las predicciones
        predictions["target"][station_id] = anomalies
    
    # Guardar predicciones
    output_file = "predictions/predictions_task_3.json"
    with open(output_file, "w") as f:
        json.dump(predictions, f, indent=2)
    
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
                2: "Sensor anormal", 
                4: "Interferencia externa", 
                8: "Mantenimiento programado", 
                9: "Datos anómalos"
            }.get(status, f"Código {status}")
            
            print(f"  - {status_name}: {count} horas ({count/total_hours*100:.2f}%)")
        
        print(f"  - Total anomalías: {anomaly_percentage:.2f}%")
    
    print(f"\nArchivo {output_file} actualizado con anomalías basadas en conocimiento experto.")

if __name__ == "__main__":
    main() 