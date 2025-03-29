import pandas as pd
import numpy as np
from pathlib import Path
import json
from sklearn.ensemble import IsolationForest
from scipy import stats
from datetime import datetime
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

def load_data():
    processed_dir = Path("data/processed")
    measurement_df = pd.read_csv(
        processed_dir / "measurement_data_processed.csv",
        parse_dates=['Measurement date']
    )
    instrument_df = pd.read_csv(
        processed_dir / "instrument_data_processed.csv",
        parse_dates=['Measurement date']
    )
    return measurement_df, instrument_df

def calculate_station_reliability(so2_data):
    """Calcula la confiabilidad de cada estación basada en múltiples métricas"""
    reliability_metrics = so2_data.groupby('Station code').agg({
        'SO2': [
            ('count', 'count'),
            ('mean', 'mean'),
            ('std', 'std'),
            ('skewness', lambda x: stats.skew(x)),
            ('iqr', lambda x: stats.iqr(x)),
            ('kurtosis', lambda x: stats.kurtosis(x))
        ]
    })
    
    reliability_metrics.columns = ['count', 'mean', 'std', 'skewness', 'iqr', 'kurtosis']
    
    # Calcular moda para cada estación
    station_modes = []
    for station in reliability_metrics.index:
        station_data = so2_data[so2_data['Station code'] == station]['SO2']
        kde = gaussian_kde(station_data)
        x_range = np.linspace(station_data.min(), station_data.max(), 100)
        density = kde(x_range)
        mode = x_range[np.argmax(density)]
        station_modes.append(mode)
    
    reliability_metrics['mode'] = station_modes
    
    # Normalizar métricas
    reliability_metrics['count_norm'] = (reliability_metrics['count'] - reliability_metrics['count'].min()) / (reliability_metrics['count'].max() - reliability_metrics['count'].min())
    reliability_metrics['std_norm'] = 1 - (reliability_metrics['std'] - reliability_metrics['std'].min()) / (reliability_metrics['std'].max() - reliability_metrics['std'].min())
    reliability_metrics['skewness_norm'] = 1 - (abs(reliability_metrics['skewness']) - abs(reliability_metrics['skewness']).min()) / (abs(reliability_metrics['skewness']).max() - abs(reliability_metrics['skewness']).min())
    reliability_metrics['iqr_norm'] = 1 - (reliability_metrics['iqr'] - reliability_metrics['iqr'].min()) / (reliability_metrics['iqr'].max() - reliability_metrics['iqr'].min())
    reliability_metrics['kurtosis_norm'] = 1 - (abs(reliability_metrics['kurtosis']) - abs(reliability_metrics['kurtosis']).min()) / (abs(reliability_metrics['kurtosis']).max() - abs(reliability_metrics['kurtosis']).min())
    
    # Calcular score de confiabilidad con pesos ajustados
    reliability_metrics['reliability_score'] = (
        reliability_metrics['count_norm'] * 0.30 +
        reliability_metrics['std_norm'] * 0.20 +
        reliability_metrics['skewness_norm'] * 0.15 +
        reliability_metrics['iqr_norm'] * 0.15 +
        reliability_metrics['kurtosis_norm'] * 0.20
    )
    
    return reliability_metrics

def detect_anomalies(so2_data, contamination=0.02):
    """Detección de anomalías usando Isolation Forest con umbral ajustado"""
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    anomalies = iso_forest.fit_predict(so2_data[['SO2']])
    so2_data['is_anomaly'] = anomalies
    return so2_data

def analyze_temporal_patterns(so2_data):
    """Analiza patrones temporales en los datos"""
    # Agregar componentes temporales
    so2_data['hour'] = so2_data['Measurement date'].dt.hour
    so2_data['day_of_week'] = so2_data['Measurement date'].dt.dayofweek
    so2_data['month'] = so2_data['Measurement date'].dt.month
    so2_data['season'] = so2_data['month'].apply(lambda x: (x % 12 + 3) // 3)
    
    # Calcular promedios por hora
    hourly_avg = so2_data.groupby('hour')['SO2'].mean()
    
    # Calcular promedios por día de la semana
    daily_avg = so2_data.groupby('day_of_week')['SO2'].mean()
    
    # Calcular promedios por mes
    monthly_avg = so2_data.groupby('month')['SO2'].mean()
    
    # Calcular promedios por estación
    seasonal_avg = so2_data.groupby('season')['SO2'].mean()
    
    return hourly_avg, daily_avg, monthly_avg, seasonal_avg

def calculate_station_correlations(so2_data):
    """Calcula correlaciones entre estaciones"""
    station_pivot = so2_data.pivot_table(
        values='SO2',
        index='Measurement date',
        columns='Station code',
        aggfunc='mean'
    )
    
    correlation_matrix = station_pivot.corr()
    return correlation_matrix

def calculate_temporal_weights(so2_data):
    """Calcula pesos basados en patrones temporales"""
    # Obtener la hora actual (promedio de todas las horas)
    current_hour = so2_data['hour'].mean()
    
    # Calcular pesos por hora usando una distribución gaussiana
    hourly_weights = np.exp(-0.1 * (np.arange(24) - current_hour)**2)
    hourly_weights = hourly_weights / hourly_weights.sum()
    
    # Calcular pesos por día de la semana
    daily_weights = np.ones(7) / 7
    
    # Calcular pesos por mes
    monthly_weights = np.ones(12) / 12
    
    return hourly_weights, daily_weights, monthly_weights

def estimate_mode(so2_data, station):
    """Estima la moda usando kernel density estimation"""
    station_data = so2_data[so2_data['Station code'] == station]['SO2']
    kde = gaussian_kde(station_data)
    
    # Crear puntos para evaluar la densidad
    x_range = np.linspace(station_data.min(), station_data.max(), 100)
    density = kde(x_range)
    
    # Encontrar el punto con máxima densidad
    mode = x_range[np.argmax(density)]
    return mode

def calculate_geographic_distance(lat1, lon1, lat2, lon2):
    """Calcula la distancia en kilómetros entre dos puntos geográficos"""
    R = 6371  # Radio de la Tierra en km
    
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    distance = R * c
    
    return distance

def validate_clusters(features_scaled, labels, station_coords):
    """Valida la calidad de los clusters usando múltiples métricas"""
    if len(np.unique(labels)) <= 1:
        return -1
    
    # Calcular score de silueta
    from sklearn.metrics import silhouette_score
    try:
        silhouette = silhouette_score(features_scaled, labels)
    except:
        silhouette = -1
    
    # Calcular score de Davies-Bouldin
    from sklearn.metrics import davies_bouldin_score
    try:
        davies_bouldin = davies_bouldin_score(features_scaled, labels)
    except:
        davies_bouldin = float('inf')
    
    # Calcular score de Calinski-Harabasz
    from sklearn.metrics import calinski_harabasz_score
    try:
        calinski_harabasz = calinski_harabasz_score(features_scaled, labels)
    except:
        calinski_harabasz = -1
    
    # Calcular score de cohesión geográfica
    geographic_score = calculate_geographic_cohesion(labels, station_coords)
    
    # Combinar scores (mayor es mejor)
    final_score = (silhouette + geographic_score) / 2
    
    return final_score

def calculate_geographic_cohesion(labels, station_coords):
    """Calcula qué tan cohesivos son los clusters geográficamente"""
    unique_labels = np.unique(labels)
    if len(unique_labels) <= 1:
        return 0
    
    cohesion_scores = []
    for label in unique_labels:
        if label == -1:  # Ignorar ruido
            continue
        
        # Obtener coordenadas de estaciones en el cluster
        cluster_mask = labels == label
        cluster_coords = station_coords[cluster_mask]
        
        if len(cluster_coords) < 2:
            continue
        
        # Calcular matriz de distancias
        distances = []
        for i in range(len(cluster_coords)):
            for j in range(i+1, len(cluster_coords)):
                dist = calculate_geographic_distance(
                    cluster_coords[i][0], cluster_coords[i][1],
                    cluster_coords[j][0], cluster_coords[j][1]
                )
                distances.append(dist)
        
        # Score de cohesión (menor distancia media = mejor)
        if distances:
            cohesion_scores.append(1 / (1 + np.mean(distances)))
    
    return np.mean(cohesion_scores) if cohesion_scores else 0

def perform_clustering(so2_data, reliability_metrics):
    """Realiza clustering de estaciones basado en sus características"""
    # Preparar datos para clustering
    features = reliability_metrics[['mean', 'std', 'skewness', 'kurtosis', 'mode']].copy()
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Obtener coordenadas de las estaciones
    station_coords = []
    for station in reliability_metrics.index:
        station_data = so2_data[so2_data['Station code'] == station].iloc[0]
        station_coords.append([station_data['Latitude'], station_data['Longitude']])
    station_coords = np.array(station_coords)
    
    # Probar diferentes configuraciones de DBSCAN
    eps_values = [0.3, 0.5, 0.7, 1.0]
    min_samples_values = [3, 4, 5, 6]
    best_score = -1
    best_labels = None
    
    for eps in eps_values:
        for min_samples in min_samples_values:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(features_scaled)
            
            # Validar clusters
            score = validate_clusters(features_scaled, labels, station_coords)
            
            if score > best_score:
                best_score = score
                best_labels = labels
    
    # Usar los mejores labels encontrados
    reliability_metrics['cluster'] = best_labels
    
    # Calcular confiabilidad del cluster
    cluster_reliability = reliability_metrics.groupby('cluster')['reliability_score'].mean()
    
    # Asignar peso adicional a estaciones en clusters confiables
    reliability_metrics['cluster_weight'] = reliability_metrics['cluster'].map(cluster_reliability)
    
    # Ajustar pesos basados en la estabilidad temporal
    reliability_metrics['temporal_stability'] = calculate_temporal_stability(so2_data, reliability_metrics.index)
    
    return reliability_metrics

def calculate_temporal_stability(so2_data, stations):
    """Calcula la estabilidad temporal de cada estación"""
    stability_scores = []
    
    for station in stations:
        station_data = so2_data[so2_data['Station code'] == station].copy()
        
        # Calcular variabilidad por hora
        hourly_std = station_data.groupby('hour')['SO2'].std().mean()
        
        # Calcular variabilidad por día
        daily_std = station_data.groupby('day_of_week')['SO2'].std().mean()
        
        # Calcular variabilidad por mes
        monthly_std = station_data.groupby('month')['SO2'].std().mean()
        
        # Calcular score de estabilidad (menor es mejor)
        stability_score = 1 / (hourly_std + daily_std + monthly_std)
        stability_scores.append(stability_score)
    
    # Normalizar scores
    stability_scores = np.array(stability_scores)
    stability_scores = (stability_scores - stability_scores.min()) / (stability_scores.max() - stability_scores.min())
    
    return pd.Series(stability_scores, index=stations)

def calculate_adaptive_weights(so2_data, reliability_metrics, correlation_matrix):
    """Calcula pesos adaptativos basados en múltiples factores"""
    # Filtrar estaciones confiables (score > 0.75)
    reliable_stations = reliability_metrics[reliability_metrics['reliability_score'] > 0.75].index
    so2_data = so2_data[so2_data['Station code'].isin(reliable_stations)]
    
    # Pesos base de confiabilidad
    station_weights = reliability_metrics.loc[reliable_stations, 'reliability_score']
    
    # Ajustar por correlaciones
    for station in reliable_stations:
        correlations = correlation_matrix.loc[station, reliable_stations]
        station_weights[station] *= (1 - correlations.mean())
    
    # Ajustar por cluster
    station_weights *= reliability_metrics.loc[reliable_stations, 'cluster_weight']
    
    # Ajustar por estabilidad temporal
    station_weights *= reliability_metrics.loc[reliable_stations, 'temporal_stability']
    
    # Calcular pesos temporales
    hourly_weights, daily_weights, monthly_weights = calculate_temporal_weights(so2_data)
    
    # Ajustar por patrones temporales
    for station in reliable_stations:
        station_data = so2_data[so2_data['Station code'] == station]
        hour_weight = hourly_weights[station_data['hour'].iloc[0]]
        day_weight = daily_weights[station_data['day_of_week'].iloc[0]]
        month_weight = monthly_weights[station_data['month'].iloc[0] - 1]
        
        station_weights[station] *= (hour_weight + day_weight + month_weight) / 3
    
    # Normalizar pesos
    station_weights = station_weights / station_weights.sum()
    
    return station_weights

def calculate_weighted_so2(so2_data, reliability_metrics, correlation_matrix):
    """Calcula el valor final de SO2 usando pesos adaptativos"""
    # Filtrar estaciones confiables
    reliable_stations = reliability_metrics[reliability_metrics['reliability_score'] > 0.75].index
    so2_data = so2_data[so2_data['Station code'].isin(reliable_stations)]
    
    # Calcular pesos adaptativos
    station_weights = calculate_adaptive_weights(so2_data, reliability_metrics, correlation_matrix)
    
    # Calcular moda ponderada por estación usando KDE
    station_modes = pd.Series(index=reliable_stations)
    for station in reliable_stations:
        station_modes[station] = estimate_mode(so2_data, station)
    
    weighted_mode = np.average(station_modes, weights=station_weights)
    
    return weighted_mode

def main():
    print("Cargando datos...")
    measurement_df, instrument_df = load_data()
    
    # Filtrar mediciones normales de SO2
    normal_so2 = instrument_df[
        (instrument_df['Instrument status'] == 0) &
        (instrument_df['Item code'] == 0)
    ]
    
    # Merge con mediciones
    valid_measurements = normal_so2.groupby(['Measurement date', 'Station code']).size().reset_index()
    so2_data = pd.merge(
        measurement_df,
        valid_measurements[['Measurement date', 'Station code']],
        on=['Measurement date', 'Station code']
    )
    
    print("\nCalculando confiabilidad de estaciones...")
    reliability_metrics = calculate_station_reliability(so2_data)
    print("\nEstaciones más confiables:")
    print(reliability_metrics.sort_values('reliability_score', ascending=False).head())
    
    print("\nDetectando anomalías...")
    so2_data = detect_anomalies(so2_data)
    print("\nEstadísticas de anomalías:")
    print(so2_data.groupby('is_anomaly')['SO2'].describe())
    
    print("\nAnalizando patrones temporales...")
    hourly_avg, daily_avg, monthly_avg, seasonal_avg = analyze_temporal_patterns(so2_data)
    print("\nPatrones temporales calculados")
    
    print("\nCalculando correlaciones entre estaciones...")
    correlation_matrix = calculate_station_correlations(so2_data)
    
    print("\nRealizando clustering de estaciones...")
    reliability_metrics = perform_clustering(so2_data, reliability_metrics)
    print("\nDistribución de clusters:")
    print(reliability_metrics.groupby('cluster').size())
    
    print("\nCalculando valor final de SO2...")
    so2_value = calculate_weighted_so2(so2_data, reliability_metrics, correlation_matrix)
    
    print(f"\nValor final de SO2: {so2_value}")
    
    # Actualizar el archivo JSON
    output_file = Path("predictions/questions.json")
    with open(output_file, 'r') as f:
        data = json.load(f)
    
    data['SO2'] = float(so2_value)
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)
    
    print("\nValor actualizado en questions.json")

if __name__ == "__main__":
    main() 