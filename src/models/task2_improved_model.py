import pandas as pd
import numpy as np
from pathlib import Path
import json
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgbm
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
import warnings
import os
import time
from sklearn.ensemble import IsolationForest
warnings.filterwarnings('ignore')

# Definir las tareas de predicción según las especificaciones del README
PREDICTION_TASKS = {
    206: {'pollutant': 'SO2', 'start_date': '2023-07-01 00:00:00', 'end_date': '2023-07-31 23:00:00'},
    211: {'pollutant': 'NO2', 'start_date': '2023-08-01 00:00:00', 'end_date': '2023-08-31 23:00:00'},
    217: {'pollutant': 'O3', 'start_date': '2023-09-01 00:00:00', 'end_date': '2023-09-30 23:00:00'},
    219: {'pollutant': 'CO', 'start_date': '2023-10-01 00:00:00', 'end_date': '2023-10-31 23:00:00'},
    225: {'pollutant': 'PM10', 'start_date': '2023-11-01 00:00:00', 'end_date': '2023-11-30 23:00:00'},
    228: {'pollutant': 'PM2.5', 'start_date': '2023-12-01 00:00:00', 'end_date': '2023-12-31 23:00:00'}
}

# Hiperparámetros optimizados específicamente por contaminante
HYPERPARAMETERS = {
    'SO2': {
        'learning_rate': 0.03,
        'n_estimators': 300,
        'num_leaves': 31,
        'reg_alpha': 0.2,
        'reg_lambda': 0.2,
        'min_child_samples': 20,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5
    },
    'NO2': {
        'learning_rate': 0.04,
        'n_estimators': 250,
        'num_leaves': 31,
        'reg_alpha': 0.3,
        'reg_lambda': 0.3,
        'min_child_samples': 20,
        'feature_fraction': 0.75,
        'bagging_fraction': 0.85,
        'bagging_freq': 5
    },
    'O3': {
        'learning_rate': 0.05,
        'n_estimators': 200,
        'num_leaves': 31,
        'reg_alpha': 0.2,
        'reg_lambda': 0.2,
        'min_child_samples': 25,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.8,
        'bagging_freq': 5
    },
    'CO': {
        'learning_rate': 0.05,
        'n_estimators': 200,
        'num_leaves': 31,
        'reg_alpha': 0.5,
        'reg_lambda': 0.5,
        'min_child_samples': 30,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.75,
        'bagging_freq': 5
    },
    'PM10': {
        'learning_rate': 0.03,
        'n_estimators': 250,
        'num_leaves': 31,
        'reg_alpha': 0.2,
        'reg_lambda': 0.2,
        'min_child_samples': 25,
        'feature_fraction': 0.75,
        'bagging_fraction': 0.8,
        'bagging_freq': 5
    },
    'PM2.5': {
        'learning_rate': 0.03,
        'n_estimators': 250,
        'num_leaves': 31,
        'reg_alpha': 0.2,
        'reg_lambda': 0.2,
        'min_child_samples': 25,
        'feature_fraction': 0.75,
        'bagging_fraction': 0.8,
        'bagging_freq': 5
    }
}

def load_processed_data():
    """Cargar y preprocesar los datos de medición."""
    processed_dir = Path("data/processed")
    
    try:
        measurement_df = pd.read_csv(
            processed_dir / "measurement_data_processed.csv",
            parse_dates=['Measurement date']
        )
        print(f"Datos cargados: {measurement_df.shape} filas")
        
        # Aplicar detección de outliers estadística
        pollutants = ['SO2', 'NO2', 'O3', 'CO', 'PM10', 'PM2.5']
        for pollutant in pollutants:
            if pollutant in measurement_df.columns:
                q1 = measurement_df[pollutant].quantile(0.01)
                q3 = measurement_df[pollutant].quantile(0.99)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                # Contar outliers
                outliers = measurement_df[(measurement_df[pollutant] < lower_bound) | 
                                         (measurement_df[pollutant] > upper_bound)]
                if len(outliers) > 0:
                    print(f"Encontrados {len(outliers)} outliers en {pollutant}")
                    
                    # Winsorizar outliers (menos agresivo que el recorte)
                    measurement_df[pollutant] = np.where(
                        measurement_df[pollutant] > upper_bound,
                        upper_bound,
                        np.where(
                            measurement_df[pollutant] < lower_bound,
                            lower_bound,
                            measurement_df[pollutant]
                        )
                    )
        
        # Imputación de valores perdidos
        measurement_df = measurement_df.fillna(method='ffill').fillna(method='bfill')
        
        return measurement_df
    except Exception as e:
        print(f"Error al cargar datos: {e}")
        raise

def add_time_features(df):
    """Añadir características temporales avanzadas."""
    # Componentes temporales básicos
    df['hour'] = df['Measurement date'].dt.hour
    df['day'] = df['Measurement date'].dt.day
    df['month'] = df['Measurement date'].dt.month
    df['day_of_week'] = df['Measurement date'].dt.dayofweek
    df['day_of_year'] = df['Measurement date'].dt.dayofyear
    
    # Codificación cíclica para características temporales
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    df['day_sin'] = np.sin(2 * np.pi * df['day']/31)
    df['day_cos'] = np.cos(2 * np.pi * df['day']/31)
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    df['weekday_sin'] = np.sin(2 * np.pi * df['day_of_week']/7)
    df['weekday_cos'] = np.cos(2 * np.pi * df['day_of_week']/7)
    
    # Características categóricas
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    df['is_business_hour'] = df['hour'].apply(lambda x: 1 if 8 <= x <= 18 else 0)
    df['is_night'] = df['hour'].apply(lambda x: 1 if x < 6 or x >= 22 else 0)
    
    # Estaciones
    df['season'] = pd.cut(
        df['month'],
        bins=[0, 3, 6, 9, 12],
        labels=['winter', 'spring', 'summer', 'fall']
    )
    df = pd.get_dummies(df, columns=['season'])
    
    return df

def prepare_features(df, target_station, target_pollutant):
    """Preparar características avanzadas para el modelo."""
    print(f"Preparando características para Estación {target_station}, Contaminante {target_pollutant}")
    
    # Filtrar datos para la estación objetivo
    station_data = df[df['Station code'] == target_station].copy()
    
    # Asegurar datos ordenados temporalmente
    station_data = station_data.sort_values('Measurement date')
    
    # Añadir características temporales
    station_data = add_time_features(station_data)
    
    # Lags específicos para cada contaminante
    lag_sets = {
        'SO2': [1, 2, 3, 4, 6, 12, 24, 48, 72, 24*7],
        'NO2': [1, 2, 3, 4, 6, 12, 24, 48, 72, 24*7],
        'O3': [1, 2, 3, 4, 6, 12, 24, 48, 72, 24*7],
        'CO': [1, 2, 3, 4, 6, 12, 24, 48, 72, 24*7],
        'PM10': [1, 2, 3, 4, 6, 12, 24, 48, 72, 24*7],
        'PM2.5': [1, 2, 3, 4, 6, 12, 24, 48, 72, 24*7]
    }
    
    # Usar lags específicos o predeterminados
    lag_values = lag_sets.get(target_pollutant, [1, 2, 3, 4, 6, 12, 24, 48, 72, 24*7])
    
    # Añadir características de lag
    for lag in lag_values:
        station_data[f'{target_pollutant}_lag_{lag}'] = station_data[target_pollutant].shift(lag)
    
    # Ventanas específicas para cada contaminante
    window_sets = {
        'SO2': [6, 12, 24, 48, 72, 24*7],
        'NO2': [6, 12, 24, 48, 72, 24*7],
        'O3': [6, 12, 24, 48, 72, 24*7],
        'CO': [6, 12, 24, 48, 72, 24*7],
        'PM10': [6, 12, 24, 48, 72, 24*7],
        'PM2.5': [6, 12, 24, 48, 72, 24*7]
    }
    
    # Usar ventanas específicas o predeterminadas
    window_values = window_sets.get(target_pollutant, [6, 12, 24, 48, 72, 24*7])
    
    # Añadir estadísticas de ventana deslizante
    for window in window_values:
        station_data[f'{target_pollutant}_rolling_mean_{window}'] = station_data[target_pollutant].rolling(window=window).mean()
        station_data[f'{target_pollutant}_rolling_std_{window}'] = station_data[target_pollutant].rolling(window=window).std()
        station_data[f'{target_pollutant}_rolling_max_{window}'] = station_data[target_pollutant].rolling(window=window).max()
        station_data[f'{target_pollutant}_rolling_min_{window}'] = station_data[target_pollutant].rolling(window=window).min()
    
    # Características por día de la semana y hora del día
    dow_mean = station_data.groupby('day_of_week')[target_pollutant].transform('mean')
    hour_mean = station_data.groupby('hour')[target_pollutant].transform('mean')
    month_mean = station_data.groupby('month')[target_pollutant].transform('mean')
    station_data[f'{target_pollutant}_dow_mean'] = dow_mean
    station_data[f'{target_pollutant}_hour_mean'] = hour_mean
    station_data[f'{target_pollutant}_month_mean'] = month_mean
    
    # Añadir características de EWMA con diferentes alfas
    alphas = [0.1, 0.3, 0.5, 0.7, 0.9]
    for alpha in alphas:
        station_data[f'{target_pollutant}_ewma_{int(alpha*10)}'] = station_data[target_pollutant].ewm(alpha=alpha).mean()
    
    # Relaciones con otros contaminantes
    pollutants = ['SO2', 'NO2', 'O3', 'CO', 'PM10', 'PM2.5']
    for other_pollutant in pollutants:
        if other_pollutant != target_pollutant and other_pollutant in station_data.columns:
            # Relación simple de proporción entre contaminantes
            station_data[f'ratio_{target_pollutant}_{other_pollutant}'] = (
                station_data[target_pollutant] / 
                station_data[other_pollutant].replace(0, np.nan).fillna(station_data[other_pollutant].mean())
            )
            
            # Correlación móvil
            window = 24  # 24 horas es un buen equilibrio
            station_data[f'corr_{target_pollutant}_{other_pollutant}_{window}'] = (
                station_data[target_pollutant].rolling(window=window)
                .corr(station_data[other_pollutant])
            )
    
    # Diferencias temporales
    for diff_period in [1, 24]:
        station_data[f'{target_pollutant}_diff_{diff_period}'] = station_data[target_pollutant].diff(diff_period)
    
    # Manejo de valores perdidos e infinitos
    station_data = station_data.replace([np.inf, -np.inf], np.nan)
    station_data = station_data.fillna(method='ffill').fillna(method='bfill')
    
    # Asegurar que no hay valores perdidos en la columna objetivo
    if station_data[target_pollutant].isnull().any():
        station_data[target_pollutant] = station_data[target_pollutant].fillna(station_data[target_pollutant].median())
    
    print(f"Preparación de características completada: {station_data.shape} filas, {station_data.shape[1]} columnas")
    return station_data

def train_model(station_data, target_pollutant):
    """Entrenar modelo LightGBM optimizado con validación temporal."""
    # Preparar características y objetivo
    exclude_columns = ['Measurement date', 'Station code', target_pollutant]
    exclude_columns.extend([col for col in station_data.columns if not pd.api.types.is_numeric_dtype(station_data[col])])
    
    numeric_columns = [col for col in station_data.columns if col not in exclude_columns]
    print(f"Usando {len(numeric_columns)} características")
    
    X = station_data[numeric_columns]
    y = station_data[target_pollutant]
    
    # Escalar características
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Validación temporal con purga
    tscv = TimeSeriesSplit(n_splits=5)  # 5 divisiones para una mejor evaluación
    
    # Obtener hiperparámetros para este contaminante
    params = HYPERPARAMETERS.get(target_pollutant, {
        'learning_rate': 0.05,
        'n_estimators': 200,
        'num_leaves': 31,
        'reg_alpha': 0.3,
        'reg_lambda': 0.3,
        'min_child_samples': 20,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5
    })
    
    # Entrenar modelo final
    model = lgbm.LGBMRegressor(
        n_estimators=params['n_estimators'],
        learning_rate=params['learning_rate'],
        num_leaves=params['num_leaves'],
        reg_alpha=params['reg_alpha'],
        reg_lambda=params['reg_lambda'],
        min_child_samples=params['min_child_samples'],
        feature_fraction=params['feature_fraction'],
        bagging_fraction=params['bagging_fraction'],
        bagging_freq=params['bagging_freq'],
        random_state=42,
        n_jobs=-1
    )
    
    # Entrenar el modelo final en todos los datos
    model.fit(X_scaled, y)
    
    # Evaluar modelo con validación cruzada temporal y purga
    cv_scores = {
        'mae': [],
        'rmse': [],
        'r2': []
    }
    
    # Período de purga (24 horas)
    purge_period = 24
    
    print("Realizando validación cruzada...")
    for train_idx, test_idx in tscv.split(X_scaled):
        # Implementar purga temporal
        max_train_idx = max(train_idx)
        min_test_idx = min(test_idx)
        
        # Eliminar datos del período de purga
        if min_test_idx - max_train_idx < purge_period:
            purge_start = max(0, min_test_idx - purge_period)
            train_idx = [idx for idx in train_idx if idx < purge_start]
        
        if len(train_idx) == 0:
            continue  # Saltar esta división si no quedan datos de entrenamiento
        
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Entrenar modelo en este fold
        fold_model = lgbm.LGBMRegressor(
            n_estimators=params['n_estimators'],
            learning_rate=params['learning_rate'],
            num_leaves=params['num_leaves'],
            reg_alpha=params['reg_alpha'],
            reg_lambda=params['reg_lambda'],
            min_child_samples=params['min_child_samples'],
            feature_fraction=params['feature_fraction'],
            bagging_fraction=params['bagging_fraction'],
            bagging_freq=params['bagging_freq'],
            random_state=42,
            n_jobs=-1
        )
        
        fold_model.fit(X_train, y_train)
        
        # Hacer predicciones
        y_pred = fold_model.predict(X_test)
        
        # Calcular métricas
        cv_scores['mae'].append(mean_absolute_error(y_test, y_pred))
        cv_scores['rmse'].append(np.sqrt(mean_squared_error(y_test, y_pred)))
        cv_scores['r2'].append(r2_score(y_test, y_pred))
    
    # Imprimir puntuaciones de validación cruzada
    print("Puntuaciones de validación cruzada:")
    print(f"MAE: {np.mean(cv_scores['mae']):.4f} (±{np.std(cv_scores['mae']):.4f})")
    print(f"RMSE: {np.mean(cv_scores['rmse']):.4f} (±{np.std(cv_scores['rmse']):.4f})")
    print(f"R²: {np.mean(cv_scores['r2']):.4f} (±{np.std(cv_scores['r2']):.4f})")
    
    return model, numeric_columns, scaler, cv_scores

def prepare_prediction_data(future_dates, historical_data, target_pollutant, feature_names):
    """Preparar datos para predicción con características avanzadas."""
    print(f"Preparando datos de predicción para {len(future_dates)} timestamps futuros")
    
    # Inicializar DataFrame con fechas
    pred_data = pd.DataFrame(index=future_dates)
    
    # Obtener últimos valores conocidos para inicializar características
    last_known_values = historical_data.tail(max(72, 24*7)).copy()  # Tomar suficientes datos históricos
    
    # Añadir características temporales básicas
    pred_data['hour'] = pred_data.index.hour
    pred_data['day'] = pred_data.index.day
    pred_data['month'] = pred_data.index.month
    pred_data['day_of_week'] = pred_data.index.dayofweek
    pred_data['day_of_year'] = pred_data.index.dayofyear
    
    # Añadir características cíclicas
    pred_data['hour_sin'] = np.sin(2 * np.pi * pred_data['hour']/24)
    pred_data['hour_cos'] = np.cos(2 * np.pi * pred_data['hour']/24)
    pred_data['day_sin'] = np.sin(2 * np.pi * pred_data['day']/31)
    pred_data['day_cos'] = np.cos(2 * np.pi * pred_data['day']/31)
    pred_data['month_sin'] = np.sin(2 * np.pi * pred_data['month']/12)
    pred_data['month_cos'] = np.cos(2 * np.pi * pred_data['month']/12)
    pred_data['weekday_sin'] = np.sin(2 * np.pi * pred_data['day_of_week']/7)
    pred_data['weekday_cos'] = np.cos(2 * np.pi * pred_data['day_of_week']/7)
    
    # Añadir características categóricas
    pred_data['is_weekend'] = pred_data['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    pred_data['is_business_hour'] = pred_data['hour'].apply(lambda x: 1 if 8 <= x <= 18 else 0)
    pred_data['is_night'] = pred_data['hour'].apply(lambda x: 1 if x < 6 or x >= 22 else 0)
    
    # Añadir estaciones
    season_mapping = {1: 'winter', 2: 'winter', 3: 'spring', 4: 'spring', 5: 'spring', 
                      6: 'summer', 7: 'summer', 8: 'summer', 9: 'fall', 10: 'fall', 
                      11: 'fall', 12: 'winter'}
    
    for season in ['winter', 'spring', 'summer', 'fall']:
        pred_data[f'season_{season}'] = pred_data['month'].map(
            lambda m: 1 if season_mapping[m] == season else 0
        )
    
    # Inicializar características de lag con últimos valores conocidos
    lags = [1, 2, 3, 4, 6, 12, 24, 48, 72, 24*7]
    for lag in lags:
        if lag < len(last_known_values):
            pred_data[f'{target_pollutant}_lag_{lag}'] = last_known_values[target_pollutant].iloc[-lag]
        else:
            # Si no tenemos suficientes datos históricos, usar el último valor
            pred_data[f'{target_pollutant}_lag_{lag}'] = last_known_values[target_pollutant].iloc[-1]
    
    # Inicializar estadísticas de ventana deslizante
    windows = [6, 12, 24, 48, 72, 24*7]
    for window in windows:
        if window < len(last_known_values):
            window_data = last_known_values[target_pollutant].iloc[-window:]
            pred_data[f'{target_pollutant}_rolling_mean_{window}'] = window_data.mean()
            pred_data[f'{target_pollutant}_rolling_std_{window}'] = window_data.std()
            pred_data[f'{target_pollutant}_rolling_max_{window}'] = window_data.max()
            pred_data[f'{target_pollutant}_rolling_min_{window}'] = window_data.min()
        else:
            # Usar todos los datos disponibles
            pred_data[f'{target_pollutant}_rolling_mean_{window}'] = last_known_values[target_pollutant].mean()
            pred_data[f'{target_pollutant}_rolling_std_{window}'] = last_known_values[target_pollutant].std()
            pred_data[f'{target_pollutant}_rolling_max_{window}'] = last_known_values[target_pollutant].max()
            pred_data[f'{target_pollutant}_rolling_min_{window}'] = last_known_values[target_pollutant].min()
    
    # Inicializar características EWMA
    alphas = [0.1, 0.3, 0.5, 0.7, 0.9]
    for alpha in alphas:
        pred_data[f'{target_pollutant}_ewma_{int(alpha*10)}'] = last_known_values[target_pollutant].ewm(alpha=alpha).mean().iloc[-1]
    
    # Medias por día de la semana y hora
    dow_means = historical_data.groupby('day_of_week')[target_pollutant].mean().to_dict()
    hour_means = historical_data.groupby('hour')[target_pollutant].mean().to_dict()
    month_means = historical_data.groupby('month')[target_pollutant].mean().to_dict()
    
    pred_data[f'{target_pollutant}_dow_mean'] = pred_data['day_of_week'].map(
        lambda dow: dow_means.get(dow, last_known_values[target_pollutant].mean())
    )
    
    pred_data[f'{target_pollutant}_hour_mean'] = pred_data['hour'].map(
        lambda h: hour_means.get(h, last_known_values[target_pollutant].mean())
    )
    
    pred_data[f'{target_pollutant}_month_mean'] = pred_data['month'].map(
        lambda m: month_means.get(m, last_known_values[target_pollutant].mean())
    )
    
    # Inicializar diferencias temporales
    for diff_period in [1, 24]:
        pred_data[f'{target_pollutant}_diff_{diff_period}'] = 0  # Inicializar a cero
    
    # Asegurar que solo usamos las características presentes en el modelo
    required_columns = set(feature_names)
    available_columns = set(pred_data.columns)
    
    # Añadir columnas faltantes
    missing_columns = required_columns - available_columns
    for col in missing_columns:
        pred_data[col] = 0  # Inicializar con valor por defecto
    
    # Eliminar columnas extras
    extra_columns = available_columns - required_columns
    if extra_columns:
        pred_data = pred_data.drop(columns=extra_columns)
    
    # Asegurar orden correcto de las columnas
    pred_data = pred_data[feature_names]
    
    print(f"Datos de predicción preparados con {pred_data.shape[1]} características")
    return pred_data

def analyze_feature_importance(model, feature_names, target_pollutant):
    """Analizar y visualizar la importancia de características."""
    try:
        # Extraer importancia de características
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        })
        
        # Ordenar por importancia
        importance_df = importance_df.sort_values('Importance', ascending=False).head(20)
        
        # Crear visualización
        plt.figure(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title(f'Top 20 Características para Predicción de {target_pollutant}')
        plt.tight_layout()
        
        # Guardar el gráfico
        output_dir = Path("reports/figures")
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / f"{target_pollutant}_feature_importance_task2.png")
        plt.close()
        
        print(f"Gráfico de importancia guardado en reports/figures/{target_pollutant}_feature_importance_task2.png")
        
        return importance_df
    except Exception as e:
        print(f"Error al analizar importancia de características: {e}")
        return None 

def predict_pollutant_improved(measurement_df, station_code, pollutant, start_date, end_date):
    """Generar predicciones optimizadas para un contaminante en una estación específica."""
    print(f"Iniciando predicciones para Estación {station_code} - {pollutant}")
    start_time = time.time()
    
    # Preparar datos para la estación
    station_data = prepare_features(measurement_df, station_code, pollutant)
    
    # Entrenar modelo
    model, feature_names, scaler, cv_scores = train_model(station_data, pollutant)
    
    # Analizar importancia de características
    analyze_feature_importance(model, feature_names, pollutant)
    
    # Crear rango de fechas para predicción
    prediction_dates = pd.date_range(start=start_date, end=end_date, freq='h')
    
    # Preparar datos de predicción
    prediction_data = prepare_prediction_data(prediction_dates, station_data, pollutant, feature_names)
    
    # Aplicar el mismo escalado usado en entrenamiento
    prediction_data_scaled = scaler.transform(prediction_data)
    
    print(f"Generando predicciones para {len(prediction_dates)} timestamps")
    
    # Generar predicciones iniciales
    predictions = model.predict(prediction_data_scaled)
    
    # Crear DataFrame de predicciones
    results = pd.DataFrame({
        'date': prediction_dates,
        'value': predictions
    })
    
    # Actualizar predicciones de manera iterativa
    for i in range(1, len(prediction_dates)):
        # Actualizar características de lag para cada paso
        date = prediction_dates[i]
        
        # Actualizar predicciones para características dependientes del tiempo
        for j in range(min(i, 24)):  # Solo actualizar hasta 24 horas atrás
            lag_idx = i - j - 1
            if lag_idx >= 0 and f'{pollutant}_lag_{j+1}' in feature_names:
                prediction_data.loc[date, f'{pollutant}_lag_{j+1}'] = results.loc[lag_idx, 'value']
        
        # Actualizar estadísticas de ventana deslizante
        for window in [6, 12, 24]:
            if i >= window:
                window_values = results.iloc[i-window:i]['value'].values
                
                if f'{pollutant}_rolling_mean_{window}' in feature_names:
                    prediction_data.loc[date, f'{pollutant}_rolling_mean_{window}'] = np.mean(window_values)
                
                if f'{pollutant}_rolling_std_{window}' in feature_names:
                    prediction_data.loc[date, f'{pollutant}_rolling_std_{window}'] = np.std(window_values)
                
                if f'{pollutant}_rolling_max_{window}' in feature_names:
                    prediction_data.loc[date, f'{pollutant}_rolling_max_{window}'] = np.max(window_values)
                
                if f'{pollutant}_rolling_min_{window}' in feature_names:
                    prediction_data.loc[date, f'{pollutant}_rolling_min_{window}'] = np.min(window_values)
        
        # Actualizar diferencias temporales
        if f'{pollutant}_diff_1' in feature_names:
            prediction_data.loc[date, f'{pollutant}_diff_1'] = results.iloc[i-1]['value'] - results.iloc[i-2]['value'] if i >= 2 else 0
        
        if f'{pollutant}_diff_24' in feature_names and i >= 24:
            prediction_data.loc[date, f'{pollutant}_diff_24'] = results.iloc[i-1]['value'] - results.iloc[i-25]['value'] if i >= 25 else 0
        
        # Actualizar EWMA
        alphas = [0.1, 0.3, 0.5, 0.7, 0.9]
        for alpha in alphas:
            col = f'{pollutant}_ewma_{int(alpha*10)}'
            if col in feature_names:
                if i == 1:
                    prediction_data.loc[date, col] = results.iloc[0]['value']
                else:
                    prev_ewma = prediction_data.iloc[i-1][col]
                    prev_value = results.iloc[i-1]['value']
                    prediction_data.loc[date, col] = alpha * prev_value + (1 - alpha) * prev_ewma
        
        # Re-escalar y predecir
        X_pred_scaled = scaler.transform(prediction_data.iloc[i:i+1])
        predictions[i] = model.predict(X_pred_scaled)[0]
    
    # Asegurar todas las predicciones son valores válidos
    predictions = np.clip(predictions, 0, None)  # No valores negativos
    results['value'] = predictions
    
    # Guardar modelo
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, f'models/{station_code}_{pollutant}_model_task2.pkl')
    joblib.dump(scaler, f'models/{station_code}_{pollutant}_scaler_task2.pkl')
    with open(f'models/{station_code}_{pollutant}_features_task2.json', 'w') as f:
        json.dump(feature_names, f)
    
    end_time = time.time()
    print(f"Predicciones completadas para Estación {station_code} - {pollutant} en {end_time - start_time:.2f} segundos")
    
    return {date.strftime('%Y-%m-%d %H:%M:%S'): float(value) for date, value in zip(results['date'], results['value'])}

def main():
    """Ejecutar el pipeline completo de predicción."""
    print("Iniciando pipeline de predicción para Task 2...")
    start_time = datetime.now()
    print(f"Tiempo de inicio: {start_time}")
    
    # Cargar datos
    measurement_df = load_processed_data()
    
    # Generar predicciones
    predictions = {"target": {}}
    
    # Procesar cada tarea de predicción
    for station_code, task_info in PREDICTION_TASKS.items():
        pollutant = task_info['pollutant']
        start_date = task_info['start_date']
        end_date = task_info['end_date']
        
        station_predictions = predict_pollutant_improved(
            measurement_df,
            station_code,
            pollutant,
            start_date,
            end_date
        )
        
        predictions["target"][str(station_code)] = station_predictions
    
    # Guardar resultados
    os.makedirs('predictions', exist_ok=True)
    with open('predictions/predictions_task_2.json', 'w') as f:
        json.dump(predictions, f, indent=2)
    
    end_time = datetime.now()
    execution_time = end_time - start_time
    
    print("="*80)
    print(f"Pipeline completado en {execution_time}")
    print("Predicciones guardadas en predictions/predictions_task_2.json")
    print("="*80)

if __name__ == "__main__":
    main()