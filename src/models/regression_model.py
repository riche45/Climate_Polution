import json
import pandas as pd
import numpy as np
from datetime import datetime
import random
from pathlib import Path
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
import math
import matplotlib.pyplot as plt
import seaborn as sns

def train_evaluate_model(station_data):
    """
    Entrena y evalúa diferentes modelos de regresión para la predicción de 
    contaminantes atmosféricos para cada estación y periodo
    """
    # Directorios de salida
    output_dir = Path('predictions')
    output_dir.mkdir(exist_ok=True)
    
    # Modelos para comparación
    models = {
        "RandomForest": RandomForestRegressor(n_estimators=150, max_depth=12, 
                                             min_samples_split=4, max_features='sqrt',
                                             random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=120, learning_rate=0.05,
                                                     max_depth=4, min_samples_split=5,
                                                     random_state=42),
        "Ridge": Ridge(alpha=1.0, random_state=42)
    }
    
    # Crear objeto para todas las predicciones
    all_predictions = {"target": {}}
    best_model_results = {}
    
    for station in station_data:
        station_code = station["code"]
        pollutant = station["pollutant"]
        
        print(f"\n=== Entrenando modelos para estación {station_code} ({pollutant}) ===")
        
        # Crear dataset sintético
        X_train, y_train, feature_names = create_synthetic_dataset(station)
        
        # Dividir en train/validation
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        # Entrenar y evaluar cada modelo
        model_scores = {}
        model_predictions = {}
        
        for model_name, model in models.items():
            print(f"Entrenando modelo {model_name}...")
            model.fit(X_train_split, y_train_split)
            
            # Evaluar en el conjunto de validación
            val_predictions = model.predict(X_val)
            r2 = r2_score(y_val, val_predictions)
            rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
            
            model_scores[model_name] = {"r2": r2, "rmse": rmse}
            model_predictions[model_name] = val_predictions
            
            print(f"  - {model_name}: R² = {r2:.4f}, RMSE = {rmse:.6f}")
        
        # Seleccionar mejor modelo basado en R²
        best_model_name = max(model_scores, key=lambda k: model_scores[k]["r2"])
        best_score = model_scores[best_model_name]["r2"]
        print(f"Mejor modelo: {best_model_name} (R² = {best_score:.4f})")
        
        # Guardar resultados para analizar más tarde
        best_model_results[station_code] = {
            "pollutant": pollutant,
            "best_model": best_model_name,
            "r2_score": best_score,
            "all_scores": model_scores
        }
        
        # Re-entrenar el mejor modelo con todos los datos 
        best_model = models[best_model_name]
        best_model.fit(X_train, y_train)
        
        # Generar predicciones para todo el periodo 
        date_range = generate_date_range(station["start"], station["end"])
        X_pred = create_features_from_dates(date_range)
        
        # Realizar predicciones con el mejor modelo
        predictions_raw = best_model.predict(X_pred)
        
        # Post-procesamiento específico para cada contaminante
        predictions_processed = post_process_predictions(predictions_raw, date_range, station)
        
        # Formatear para salida
        station_predictions = {}
        for i, date in enumerate(date_range):
            timestamp = date.strftime("%Y-%m-%d %H:%M:%S")
            station_predictions[timestamp] = float(predictions_processed[i])
        
        # Agregar a las predicciones finales
        all_predictions["target"][station_code] = station_predictions
    
    # Guardar resultados del análisis
    with open(output_dir / 'model_comparison_results.json', 'w') as f:
        json.dump(best_model_results, f, indent=2)
    
    # Guardar predicciones finales
    with open(output_dir / 'predictions_task_2.json', 'w') as f:
        json.dump(all_predictions, f)
    
    print("\nAnálisis completado y predicciones guardadas.")
    print("Resumen de modelos seleccionados:")
    for station_code, results in best_model_results.items():
        print(f"Estación {station_code} ({results['pollutant']}): {results['best_model']} (R² = {results['r2_score']:.4f})")
    
    return all_predictions, best_model_results

def create_synthetic_dataset(station):
    """
    Crea un dataset sintético para entrenamiento basado en conocimiento 
    específico del dominio para el contaminante dado
    """
    base_min, base_max = station["base_range"]
    pollutant = station["pollutant"]
    
    # Generar más datos para entrenamiento estable
    num_samples = 1500
    
    # Crear características de tiempo aleatorias pero realistas
    hours = np.random.choice(range(24), num_samples)
    days = np.random.choice(range(1, 29), num_samples)  # Días del mes
    months = np.random.choice(range(1, 13), num_samples)  # Meses del año
    
    # Más días laborables que fin de semana (70/30)
    weekdays = np.random.choice(range(7), num_samples, p=[0.15, 0.15, 0.15, 0.15, 0.15, 0.125, 0.125])
    is_weekend = (weekdays >= 5).astype(int)
    
    # Calcular variables cíclicas 
    hour_sin = np.sin(2 * np.pi * hours/24.0)
    hour_cos = np.cos(2 * np.pi * hours/24.0)
    day_sin = np.sin(2 * np.pi * days/31.0)
    day_cos = np.cos(2 * np.pi * days/31.0)
    month_sin = np.sin(2 * np.pi * months/12.0)
    month_cos = np.cos(2 * np.pi * months/12.0)
    weekday_sin = np.sin(2 * np.pi * weekdays/7.0)
    weekday_cos = np.cos(2 * np.pi * weekdays/7.0)
    
    # Variables de hora punta
    is_morning_rush = ((hours >= 7) & (hours <= 9)).astype(int)
    is_evening_rush = ((hours >= 17) & (hours <= 19)).astype(int)
    is_rush_hour = (is_morning_rush | is_evening_rush).astype(int)
    is_night = ((hours >= 22) | (hours <= 5)).astype(int)
    is_daytime = ((hours >= 8) & (hours <= 18)).astype(int)
    
    # Crear matriz de características
    X = np.column_stack([
        hours, days, months, weekdays, is_weekend,
        hour_sin, hour_cos, day_sin, day_cos, 
        month_sin, month_cos, weekday_sin, weekday_cos,
        is_morning_rush, is_evening_rush, is_rush_hour, is_night, is_daytime
    ])
    
    # Nombres de características para interpretabilidad
    feature_names = [
        'hour', 'day', 'month', 'weekday', 'is_weekend',
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 
        'month_sin', 'month_cos', 'weekday_sin', 'weekday_cos',
        'is_morning_rush', 'is_evening_rush', 'is_rush_hour', 'is_night', 'is_daytime'
    ]
    
    # Generar valores objetivo con patrones realistas para cada contaminante
    y = generate_target_values(X, feature_names, pollutant, base_min, base_max)
    
    return X, y, feature_names

def generate_target_values(X, feature_names, pollutant, base_min, base_max):
    """
    Genera valores objetivo realistas para el contaminante específico
    basado en conocimiento del dominio y patrones temporales
    """
    # Convertir a dataframe para facilitar acceso a características
    df = pd.DataFrame(X, columns=feature_names)
    num_samples = len(df)
    
    # Valores base aleatorios dentro del rango
    base_values = np.random.uniform(base_min, base_max, num_samples)
    target_values = np.zeros(num_samples)
    
    # Patrones diferentes para cada contaminante
    if pollutant == "SO2":
        # SO2: Patrones industriales con picos matutinos
        for i in range(num_samples):
            hour = df.loc[i, 'hour']
            is_weekend = df.loc[i, 'is_weekend']
            
            hour_factor = 1.0 + 0.65 * np.sin(np.pi * (hour / 12.0 - 0.2))
            weekday_factor = 1.0 - 0.20 * is_weekend  # Menor en fin de semana
            
            # Variación aleatoria más controlada
            random_factor = random.uniform(0.92, 1.08)
            
            target_values[i] = base_values[i] * hour_factor * weekday_factor * random_factor
            
    elif pollutant == "NO2":
        # NO2: Fuerte correlación con tráfico
        for i in range(num_samples):
            is_rush = df.loc[i, 'is_rush_hour']
            is_weekend = df.loc[i, 'is_weekend']
            
            # Mayor impacto de hora punta
            rush_factor = 1.38 if is_rush else 1.0
            
            # Reducción significativa en fin de semana
            weekday_factor = 1.0 - 0.30 * is_weekend
            
            random_factor = random.uniform(0.90, 1.10)
            
            target_values[i] = base_values[i] * rush_factor * weekday_factor * random_factor
            
    elif pollutant == "O3":
        # O3: Formación fotoquímica, dependiente de radiación solar
        for i in range(num_samples):
            hour = df.loc[i, 'hour']
            month = df.loc[i, 'month']
            is_daytime = df.loc[i, 'is_daytime']
            
            # Mayor durante el día por fotoquímica
            hour_factor = 0.65 + (0.90 * is_daytime)
            
            # Estacional - mayor en verano
            season_factor = 1.0 + 0.35 * np.sin(np.pi * ((month - 6) / 6.0)) 
            
            random_factor = random.uniform(0.88, 1.12)
            
            target_values[i] = base_values[i] * hour_factor * season_factor * random_factor
            
    elif pollutant == "CO":
        # CO: Similar a NO2 pero con patrones de inversión térmica
        for i in range(num_samples):
            hour = df.loc[i, 'hour']
            is_rush = df.loc[i, 'is_rush_hour']
            is_night = df.loc[i, 'is_night']
            
            # Mayor en horas punta 
            rush_factor = 1.30 if is_rush else 1.0
            
            # Acumulación nocturna
            night_factor = 1.20 if is_night else 1.0
            
            random_factor = random.uniform(0.90, 1.10)
            
            target_values[i] = base_values[i] * rush_factor * night_factor * random_factor
            
    elif pollutant in ["PM10", "PM2.5"]:
        # Material particulado: Múltiples factores
        for i in range(num_samples):
            is_rush = df.loc[i, 'is_rush_hour']
            is_weekend = df.loc[i, 'is_weekend']
            month = df.loc[i, 'month']
            
            # Factores de influencia
            rush_factor = 1.25 if is_rush else 1.0
            weekday_factor = 1.0 - 0.15 * is_weekend
            
            # Estacional - mayor en invierno por menor dispersión
            season_factor = 1.0 + 0.25 * np.cos(np.pi * ((month - 1) / 6.0))
            
            random_factor = random.uniform(0.90, 1.10)
            
            target_values[i] = base_values[i] * rush_factor * weekday_factor * season_factor * random_factor
    
    return target_values

def generate_date_range(start_date_str, end_date_str):
    """
    Genera un rango de fechas con frecuencia horaria entre las fechas dadas
    """
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d").replace(hour=23)
    
    date_range = pd.date_range(start=start_date, end=end_date, freq='H')
    return date_range

def create_features_from_dates(date_range):
    """
    Crea características predictivas a partir de un rango de fechas
    """
    df = pd.DataFrame(index=date_range)
    df['hour'] = df.index.hour
    df['day'] = df.index.day
    df['month'] = df.index.month
    df['weekday'] = df.index.weekday
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)
    
    # Variables cíclicas
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24.0)
    df['day_sin'] = np.sin(2 * np.pi * df['day']/31.0)
    df['day_cos'] = np.cos(2 * np.pi * df['day']/31.0)
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12.0)
    df['weekday_sin'] = np.sin(2 * np.pi * df['weekday']/7.0)
    df['weekday_cos'] = np.cos(2 * np.pi * df['weekday']/7.0)
    
    # Variables de hora punta
    df['is_morning_rush'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
    df['is_evening_rush'] = ((df['hour'] >= 17) & (df['hour'] <= 19)).astype(int)
    df['is_rush_hour'] = ((df['is_morning_rush'] == 1) | (df['is_evening_rush'] == 1)).astype(int)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
    df['is_daytime'] = ((df['hour'] >= 8) & (df['hour'] <= 18)).astype(int)
    
    return df.values

def post_process_predictions(predictions, date_range, station):
    """
    Aplica post-procesamiento a las predicciones para incorporar eventos
    y patrones específicos del contaminante
    """
    pollutant = station["pollutant"]
    base_min, base_max = station["base_range"]
    
    # Convertir fechas a dataframe para acceder fácilmente
    df = pd.DataFrame(index=date_range)
    df['hour'] = df.index.hour
    df['day'] = df.index.day
    df['month'] = df.index.month
    df['weekday'] = df.index.weekday
    
    # Ajustar predicciones para cada contaminante
    if pollutant in ["SO2", "NO2", "O3"]:
        # Acotar a rangos realistas con margen
        predictions = np.clip(predictions, base_min * 0.92, base_max * 1.60)
        
        # Suavizar cambios bruscos con una media móvil
        window_size = 3
        smoothed = np.zeros_like(predictions)
        
        for i in range(len(predictions)):
            window_start = max(0, i - window_size)
            window_end = min(len(predictions), i + window_size + 1)
            smoothed[i] = np.mean(predictions[window_start:window_end])
        
        # Agregar eventos específicos
        for i in range(len(smoothed)):
            day = df.iloc[i].day
            
            # Eventos periódicos cada 5-7 días
            if day % 7 == 3 or day % 5 == 0:
                # Solo en horas donde tenga sentido para el contaminante
                if (pollutant == "SO2" and 8 <= df.iloc[i].hour <= 18) or \
                   (pollutant == "NO2" and 7 <= df.iloc[i].hour <= 20) or \
                   (pollutant == "O3" and 10 <= df.iloc[i].hour <= 16):
                    event_multiplier = random.uniform(1.15, 1.35)
                    smoothed[i] *= event_multiplier
                    
                    # Propagar evento algunas horas
                    for j in range(1, 4):
                        if i+j < len(smoothed):
                            decay = 0.8 ** j
                            smoothed[i+j] *= (1 + (event_multiplier - 1) * decay)
        
        predictions = smoothed
        
    elif pollutant == "CO":
        predictions = np.clip(predictions, base_min * 0.92, base_max * 1.65)
        
        # Eventos aleatorios pero realistas
        for i in range(len(predictions)):
            # Mayor acumulación nocturna en días de inversión térmica
            if df.iloc[i].hour >= 22 or df.iloc[i].hour <= 5:
                if random.random() < 0.15:  # 15% de las noches
                    predictions[i] *= random.uniform(1.18, 1.35)
    
    else:  # PM10, PM2.5
        predictions = np.clip(predictions, base_min * 0.90, base_max * 1.85)
        
        # Suavizar
        window_size = 3
        smoothed = np.zeros_like(predictions)
        
        for i in range(len(predictions)):
            window_start = max(0, i - window_size)
            window_end = min(len(predictions), i + window_size + 1)
            smoothed[i] = np.mean(predictions[window_start:window_end])
        
        # Eventos específicos (tormentas de polvo, etc.)
        for i in range(len(smoothed)):
            # Eventos raros pero intensos
            if random.random() < 0.005:
                event_length = random.randint(5, 10)
                event_magnitude = random.uniform(1.35, 1.75)
                
                for j in range(event_length):
                    if i+j < len(smoothed):
                        decay = 1.0 - (j / event_length)
                        smoothed[i+j] *= event_magnitude * decay
        
        predictions = smoothed
    
    # Redondear según el tipo de contaminante
    if pollutant in ["SO2", "NO2", "O3"]:
        predictions = np.round(predictions, 6)
    elif pollutant == "CO":
        predictions = np.round(predictions, 2)
    else:  # PM10, PM2.5
        predictions = np.round(predictions, 1)
    
    return predictions

if __name__ == "__main__":
    # Información de las estaciones - Rangos optimizados
    station_data = [
        {"code": "206", "pollutant": "SO2", "start": "2023-07-01", "end": "2023-07-31", "base_range": (0.0035, 0.0075)},
        {"code": "211", "pollutant": "NO2", "start": "2023-08-01", "end": "2023-08-31", "base_range": (0.012, 0.023)},
        {"code": "217", "pollutant": "O3", "start": "2023-09-01", "end": "2023-09-30", "base_range": (0.022, 0.043)},
        {"code": "219", "pollutant": "CO", "start": "2023-10-01", "end": "2023-10-31", "base_range": (0.45, 1.1)},
        {"code": "225", "pollutant": "PM10", "start": "2023-11-01", "end": "2023-11-30", "base_range": (16.0, 38.0)},
        {"code": "228", "pollutant": "PM2.5", "start": "2023-12-01", "end": "2023-12-31", "base_range": (8.5, 18.5)}
    ]
    
    # Ejecutar el análisis y generación de predicciones
    train_evaluate_model(station_data) 