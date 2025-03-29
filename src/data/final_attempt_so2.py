import pandas as pd
import numpy as np
import json
from pathlib import Path
from scipy import stats
from datetime import datetime

def load_data():
    """Carga los datos de mediciones"""
    processed_dir = Path("data/processed")
    
    # Cargar datos procesados
    measurement_df = pd.read_csv(processed_dir / "measurement_data_processed.csv")
    
    # En el caso de que ya estén filtrados solo para SO2, no necesitamos filtrar
    # Si existe la columna 'Parameter' o 'Air pollutant', usar esa para filtrar
    if 'Parameter' in measurement_df.columns:
        so2_data = measurement_df[measurement_df['Parameter'] == 'SO2'].copy()
    elif 'Air pollutant' in measurement_df.columns:
        so2_data = measurement_df[measurement_df['Air pollutant'] == 'SO2'].copy()
    else:
        # Si no existe una columna para filtrar por contaminante, asumir que todos son SO2
        so2_data = measurement_df.copy()
    
    # Convertir fecha a datetime si existe la columna
    if 'Measurement date' in so2_data.columns:
        so2_data['Measurement date'] = pd.to_datetime(so2_data['Measurement date'])
        
        # Crear características de tiempo
        so2_data['hour'] = so2_data['Measurement date'].dt.hour
        so2_data['day_of_week'] = so2_data['Measurement date'].dt.dayofweek
        so2_data['month'] = so2_data['Measurement date'].dt.month
        so2_data['day'] = so2_data['Measurement date'].dt.day
    
    print("Datos cargados:", so2_data.shape)
    
    # Verificar que existe la columna SO2
    if 'SO2' not in so2_data.columns:
        # Buscar columnas que podrían contener valores de SO2
        if 'Concentration' in so2_data.columns:
            so2_data.rename(columns={'Concentration': 'SO2'}, inplace=True)
        elif 'Value' in so2_data.columns:
            so2_data.rename(columns={'Value': 'SO2'}, inplace=True)
    
    return so2_data

def calculate_baseline_statistics(so2_data):
    """Calcula estadísticas básicas globales"""
    # Estadísticas de SO2
    print("\nEstadísticas globales de SO2:")
    stats = so2_data['SO2'].describe().round(6)
    print(stats)
    
    # Rango intercuartil
    q25 = so2_data['SO2'].quantile(0.25)
    q75 = so2_data['SO2'].quantile(0.75)
    
    return {
        'mean': so2_data['SO2'].mean(),
        'median': so2_data['SO2'].median(),
        'q25': q25,
        'q75': q75,
        'iqr': q75 - q25
    }

def approach_1_midpoint(baseline_stats):
    """Enfoque 1: Usar el punto medio exacto del rango intercuartil"""
    midpoint = (baseline_stats['q25'] + baseline_stats['q75']) / 2
    print(f"\nEnfoque 1: Punto medio del rango intercuartil: {midpoint}")
    return midpoint

def approach_2_best_historical(historical_values, scores):
    """Enfoque 2: Regresar al valor histórico con mejor rendimiento"""
    best_idx = np.argmax(scores)
    best_value = historical_values[best_idx]
    print(f"\nEnfoque 2: Mejor valor histórico (índice {best_idx}): {best_value}")
    return best_value

def identify_stable_days(so2_data):
    """Identifica los días con menor variabilidad en las mediciones de SO2"""
    if 'day' not in so2_data.columns or 'month' not in so2_data.columns:
        return None
    
    # Agrupar por día y mes
    daily_stats = so2_data.groupby(['month', 'day']).agg({
        'SO2': ['mean', 'std', 'count']
    })
    daily_stats.columns = ['mean', 'std', 'count']
    
    # Filtrar días con suficientes mediciones
    daily_stats = daily_stats[daily_stats['count'] > 100]
    
    # Identificar los días más estables (menor desviación estándar)
    stable_days = daily_stats.sort_values('std').head(10)
    
    print("\nDías más estables:")
    print(stable_days)
    
    return stable_days

def approach_3_stable_days(so2_data, stable_days):
    """Enfoque 3: Valor de SO2 en los días más estables"""
    if stable_days is None or len(stable_days) == 0:
        return None
    
    # Obtener el top 3 de días más estables
    top_days = stable_days.head(3)
    
    # Filtrar datos para esos días
    stable_data = pd.DataFrame()
    for idx in top_days.index:
        month, day = idx
        day_data = so2_data[(so2_data['month'] == month) & (so2_data['day'] == day)]
        stable_data = pd.concat([stable_data, day_data])
    
    # Calcular valor de SO2 en días estables
    stable_value = stable_data['SO2'].mean()
    
    print(f"\nEnfoque 3: Valor de SO2 en días estables: {stable_value}")
    return stable_value

def select_final_value(values, baseline_stats, historical_mean):
    """Selecciona el valor final basado en múltiples criterios"""
    # Calcular distancia al centro del rango intercuartil
    iqr_midpoint = (baseline_stats['q25'] + baseline_stats['q75']) / 2
    distances_to_midpoint = [abs(v - iqr_midpoint) for v in values if v is not None]
    
    # Calcular distancia a la media histórica
    distances_to_historical = [abs(v - historical_mean) for v in values if v is not None]
    
    # Combinar criterios (menor distancia es mejor)
    scores = []
    for i, v in enumerate([v for v in values if v is not None]):
        score = 0.5 * (1 - distances_to_midpoint[i] / max(distances_to_midpoint)) + \
                0.5 * (1 - distances_to_historical[i] / max(distances_to_historical))
        scores.append((v, score))
    
    # Seleccionar valor con mejor score
    best_value, best_score = max(scores, key=lambda x: x[1])
    
    print(f"\nValor final seleccionado: {best_value} (score: {best_score})")
    return best_value

def update_json_value(value):
    """Actualiza el valor en el archivo JSON"""
    # Cargar el archivo JSON
    json_path = Path("predictions/questions.json")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Actualizar el valor
    data["target"]["SO2"] = float(value)
    
    # Guardar los cambios
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\nValor actualizado en questions.json: {value}")

def main():
    # Cargar datos
    print("Cargando datos...")
    so2_data = load_data()
    
    # Calcular estadísticas base
    baseline_stats = calculate_baseline_statistics(so2_data)
    
    # Definir valores históricos y sus "scores" (relativos al desempeño)
    historical_values = [
        0.004102,  # Versión inicial
        0.004368,  # Versión con moda
        0.004524,  # Versión con K-means
        0.004329,  # Versión con DBSCAN
        0.004528,  # Versión con clustering geográfico
        0.00345    # Versión con ensemble
    ]
    
    # Scores asociados (valores hipotéticos basados en el desempeño relativo)
    # Mayor valor = mejor desempeño
    historical_scores = [0.7, 0.85, 0.9, 0.8, 0.9, 0.75]
    
    # Calcular media histórica
    historical_mean = np.mean(historical_values)
    print(f"\nMedia histórica de SO2: {historical_mean}")
    
    # Enfoque 1: Punto medio del rango intercuartil
    midpoint_value = approach_1_midpoint(baseline_stats)
    
    # Enfoque 2: Mejor valor histórico
    best_historical = approach_2_best_historical(historical_values, historical_scores)
    
    # Enfoque 3: Valor en días estables
    stable_days = identify_stable_days(so2_data)
    stable_days_value = approach_3_stable_days(so2_data, stable_days)
    
    # Seleccionar el mejor valor final
    final_values = [midpoint_value, best_historical, stable_days_value]
    final_value = select_final_value(final_values, baseline_stats, historical_mean)
    
    # Actualizar valor en JSON
    update_json_value(final_value)

if __name__ == "__main__":
    main() 