import pandas as pd
import numpy as np
import json
from pathlib import Path
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde
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
    
    # Cálculos básicos
    mean = so2_data['SO2'].mean()
    median = so2_data['SO2'].median()
    mode = so2_data['SO2'].mode().iloc[0]
    
    # Calcular moda con KDE para mayor precisión
    kde = gaussian_kde(so2_data['SO2'])
    x_range = np.linspace(so2_data['SO2'].min(), so2_data['SO2'].max(), 1000)
    density = kde(x_range)
    kde_mode = x_range[np.argmax(density)]
    
    # Rango intercuartil
    q25 = so2_data['SO2'].quantile(0.25)
    q75 = so2_data['SO2'].quantile(0.75)
    
    results = {
        'mean': mean,
        'median': median,
        'mode': mode,
        'kde_mode': kde_mode,
        'q25': q25,
        'q75': q75
    }
    
    return results

def try_different_weightings(stats, historical_values):
    """Prueba diferentes combinaciones de pesos para los estadísticos"""
    combinations = []
    
    # Probar diferentes pesos para mean, median, mode
    weight_options = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    for w_mean in [0.3, 0.4, 0.5]:
        for w_median in [0.2, 0.3, 0.4]:
            for w_mode in [0.2, 0.3, 0.4]:
                if abs(w_mean + w_median + w_mode - 1.0) < 0.001:  # Asegurar que sumen 1
                    weighted_value = (
                        w_mean * stats['mean'] +
                        w_median * stats['median'] +
                        w_mode * stats['kde_mode']
                    )
                    combinations.append({
                        'weights': [w_mean, w_median, w_mode],
                        'value': weighted_value
                    })
    
    # Calcular la media histórica de nuestros valores anteriores
    historical_mean = np.mean(historical_values)
    
    # Encontrar la combinación más cercana a los valores históricos
    best_combo = min(combinations, key=lambda x: abs(x['value'] - historical_mean))
    
    # También devolver un ensemble de todos los valores históricos
    ensemble_value = np.mean(historical_values)
    
    return best_combo, ensemble_value

def create_ensemble_model(so2_data):
    """Crea un modelo ensemble combinando diferentes métodos para calcular SO2"""
    # Método 1: Media global
    mean_value = so2_data['SO2'].mean()
    
    # Método 2: Mediana global
    median_value = so2_data['SO2'].median()
    
    # Método 3: Moda KDE global
    kde = gaussian_kde(so2_data['SO2'])
    x_range = np.linspace(so2_data['SO2'].min(), so2_data['SO2'].max(), 1000)
    density = kde(x_range)
    kde_mode = x_range[np.argmax(density)]
    
    # Método 4: Media por estación (ponderada por número de mediciones)
    station_stats = so2_data.groupby('Station code').agg({
        'SO2': ['mean', 'count']
    })
    station_stats.columns = ['mean', 'count']
    weighted_station_mean = np.average(
        station_stats['mean'],
        weights=station_stats['count']
    )
    
    # Método 5: Cálculo basado en estaciones confiables
    # Detectar y excluir anomalías
    q75 = so2_data['SO2'].quantile(0.75)
    q25 = so2_data['SO2'].quantile(0.25)
    iqr = q75 - q25
    upper_bound = q75 + 1.5 * iqr
    filtered_data = so2_data[so2_data['SO2'] <= upper_bound]
    
    # Calcular estadísticas por estación después de filtrar
    station_reliability = filtered_data.groupby('Station code').agg({
        'SO2': ['mean', 'std', 'count']
    })
    station_reliability.columns = ['mean', 'std', 'count']
    
    # Identificar estaciones confiables (muchas mediciones, baja desviación)
    station_reliability['reliability'] = (
        station_reliability['count'] / station_reliability['count'].max() * 0.6 +
        (1 - station_reliability['std'] / station_reliability['std'].max()) * 0.4
    )
    
    # Seleccionar top 5 estaciones más confiables
    top_stations = station_reliability.sort_values('reliability', ascending=False).head(5).index
    top_station_data = filtered_data[filtered_data['Station code'].isin(top_stations)]
    reliable_value = top_station_data['SO2'].mean()
    
    # Combinar todos los métodos (probar diferentes pesos)
    ensemble_weights = [0.15, 0.20, 0.25, 0.15, 0.25]  # Pesos para cada método
    ensemble_value = (
        ensemble_weights[0] * mean_value +
        ensemble_weights[1] * median_value +
        ensemble_weights[2] * kde_mode +
        ensemble_weights[3] * weighted_station_mean +
        ensemble_weights[4] * reliable_value
    )
    
    return ensemble_value

def manual_tuning(baseline_stats, historical_values, ensemble_value):
    """Realizar ajuste manual basado en la convergencia de resultados"""
    # Calcular media de valores históricos
    historical_mean = np.mean(historical_values)
    
    # Calcular la desviación estándar
    historical_std = np.std(historical_values)
    
    # Calcular un intervalo de confianza
    confidence_interval = (
        historical_mean - historical_std,
        historical_mean + historical_std
    )
    
    # Comprobar si el ensemble está dentro del intervalo
    if confidence_interval[0] <= ensemble_value <= confidence_interval[1]:
        # Si está dentro, usar el ensemble
        final_value = ensemble_value
    else:
        # Si está fuera, ajustar hacia la media histórica
        final_value = (ensemble_value + historical_mean) / 2
    
    # Asegurar que está dentro del rango intercuartil global
    if final_value < baseline_stats['q25'] or final_value > baseline_stats['q75']:
        # Ajustar hacia el centro del rango intercuartil
        final_value = (final_value + (baseline_stats['q25'] + baseline_stats['q75'])/2) / 2
    
    return final_value

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
    print("\nCalculando estadísticas base...")
    baseline_stats = calculate_baseline_statistics(so2_data)
    
    # Definir valores históricos
    historical_values = [
        0.004102,  # Versión inicial
        0.004368,  # Versión con moda
        0.004524,  # Versión con K-means
        0.004329,  # Versión con DBSCAN
        0.004528   # Versión actual
    ]
    print("\nValores históricos de SO2:")
    for i, val in enumerate(historical_values):
        print(f"Versión {i+1}: {val}")
    
    # Estrategia 1: Probar diferentes ponderaciones
    print("\nEstrategia 1: Probando diferentes ponderaciones...")
    best_combo, ensemble_historical = try_different_weightings(baseline_stats, historical_values)
    print(f"Mejor combinación: {best_combo['weights']} -> {best_combo['value']}")
    print(f"Ensemble histórico: {ensemble_historical}")
    
    # Estrategia 2: Crear modelo ensemble
    print("\nEstrategia 2: Creando modelo ensemble...")
    ensemble_value = create_ensemble_model(so2_data)
    print(f"Valor del ensemble: {ensemble_value}")
    
    # Estrategia 3: Ajuste manual
    print("\nEstrategia 3: Realizando ajuste manual...")
    final_value = manual_tuning(baseline_stats, historical_values, ensemble_value)
    print(f"Valor final ajustado: {final_value}")
    
    # Actualizar valor en JSON
    update_json_value(final_value)

if __name__ == "__main__":
    main() 