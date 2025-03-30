import pandas as pd
import numpy as np
from pathlib import Path
import json
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgbm
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
import warnings
from sklearn.ensemble import IsolationForest
import os
import time
warnings.filterwarnings('ignore')

def load_processed_data():
    """Load processed datasets with efficient error handling."""
    processed_dir = Path("data/processed")
    
    try:
        measurement_df = pd.read_csv(
            processed_dir / "measurement_data_processed.csv",
            parse_dates=['Measurement date']
        )
        print(f"Data loaded successfully: {measurement_df.shape} rows")
        
        # Simple outlier detection - faster than IsolationForest
        pollutants = ['SO2', 'NO2', 'O3', 'CO', 'PM10', 'PM2.5']
        for pollutant in pollutants:
            if pollutant in measurement_df.columns:
                # Apply simple statistical outlier detection
                q1 = measurement_df[pollutant].quantile(0.25)
                q3 = measurement_df[pollutant].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                # Count outliers
                outliers = measurement_df[(measurement_df[pollutant] < lower_bound) | 
                                         (measurement_df[pollutant] > upper_bound)]
                if len(outliers) > 0:
                    print(f"Found {len(outliers)} outliers in {pollutant}")
                    
                    # Cap outliers
                    measurement_df[pollutant] = np.where(
                        measurement_df[pollutant] > upper_bound,
                        upper_bound,
                        measurement_df[pollutant]
                    )
                    measurement_df[pollutant] = np.where(
                        measurement_df[pollutant] < lower_bound,
                        lower_bound,
                        measurement_df[pollutant]
                    )
        
        # Simple imputation for missing values
        measurement_df = measurement_df.fillna(method='ffill').fillna(method='bfill')
        
        return measurement_df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def add_time_features(df):
    """Add essential time-based features."""
    # Extract basic time components
    df['hour'] = df['Measurement date'].dt.hour
    df['day'] = df['Measurement date'].dt.day
    df['month'] = df['Measurement date'].dt.month
    df['day_of_week'] = df['Measurement date'].dt.dayofweek
    df['day_of_year'] = df['Measurement date'].dt.dayofyear
    
    # Simple cyclical encoding for time features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    df['day_sin'] = np.sin(2 * np.pi * df['day']/31)
    df['day_cos'] = np.cos(2 * np.pi * df['day']/31)
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    
    # Basic categorical features
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Season encoding
    df['season'] = pd.cut(
        df['month'],
        bins=[0, 3, 6, 9, 12],
        labels=['winter', 'spring', 'summer', 'fall']
    )
    df = pd.get_dummies(df, columns=['season'])
    
    return df

def detect_outliers(data, contamination=0.015):
    """Detect outliers for each pollutant using IsolationForest"""
    pollutants = ['SO2', 'NO2', 'O3', 'CO', 'PM10', 'PM2.5']
    outliers_mask = pd.DataFrame(index=data.index)
    
    for pollutant in pollutants:
        if pollutant in data.columns:
            # Filtrar valores no nulos
            valid_idx = data[pollutant].notna()
            X = data.loc[valid_idx, pollutant].values.reshape(-1, 1)
            
            # Aplicar IsolationForest
            iso = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
            outliers = iso.fit_predict(X)
            
            # Crear máscara de outliers (-1 son outliers, 1 son inliers)
            mask = pd.Series(False, index=data.index)
            mask.loc[valid_idx] = (outliers == -1)
            outliers_mask[pollutant] = mask
            
            # Reportar número de outliers
            n_outliers = mask.sum()
            print(f"Found {n_outliers} outliers in {pollutant}")
    
    # Agregar columnas de outliers al dataframe original
    for pollutant in pollutants:
        if pollutant in outliers_mask.columns:
            outlier_col = f"{pollutant}_outlier"
            data[outlier_col] = outliers_mask[pollutant]
    
    return data

def prepare_features(df, target_station, target_pollutant):
    """Prepare optimized feature set for the model."""
    print(f"Preparing features for Station {target_station}, Pollutant {target_pollutant}")
    
    # Filter data for target station
    station_data = df[df['Station code'] == target_station].copy()
    
    # Ensure time-ordered data
    station_data = station_data.sort_values('Measurement date')
    
    # Add time features (efficient version)
    station_data = add_time_features(station_data)
    
    # Add selective lag features (most important only)
    key_lags = [1, 24, 48, 24*7]  # Hour ago, day ago, 2 days ago, week ago
    for lag in key_lags:
        station_data[f'{target_pollutant}_lag_{lag}'] = station_data[target_pollutant].shift(lag)
    
    # Add selective rolling statistics (only the most useful windows)
    key_windows = [24, 24*7]  # Daily and weekly
    for window in key_windows:
        station_data[f'{target_pollutant}_rolling_mean_{window}'] = station_data[target_pollutant].rolling(window=window).mean()
        station_data[f'{target_pollutant}_rolling_std_{window}'] = station_data[target_pollutant].rolling(window=window).std()
    
    # Add day-of-week and hour-of-day averages (very informative features)
    dow_mean = station_data.groupby('day_of_week')[target_pollutant].transform('mean')
    hour_mean = station_data.groupby('hour')[target_pollutant].transform('mean')
    station_data[f'{target_pollutant}_dow_mean'] = dow_mean
    station_data[f'{target_pollutant}_hour_mean'] = hour_mean
    
    # Add other pollutants basic relationships (if available)
    pollutants = ['SO2', 'NO2', 'O3', 'CO', 'PM10', 'PM2.5']
    for other_pollutant in pollutants:
        if other_pollutant != target_pollutant and other_pollutant in station_data.columns:
            # Simple correlation feature (24h window only)
            window = 24
            corr_name = f'corr_{target_pollutant}_{other_pollutant}_{window}'
            
            # Simple ratio between pollutants
            station_data[f'ratio_{target_pollutant}_{other_pollutant}'] = (
                station_data[target_pollutant] / 
                station_data[other_pollutant].replace(0, np.nan).fillna(station_data[other_pollutant].mean())
            )
    
    # Handle missing values efficiently
    station_data = station_data.replace([np.inf, -np.inf], np.nan)
    station_data = station_data.fillna(method='ffill').fillna(method='bfill')
    
    # Make sure we don't have missing values in target column
    if station_data[target_pollutant].isnull().any():
        station_data[target_pollutant] = station_data[target_pollutant].fillna(station_data[target_pollutant].median())
    
    print(f"Feature preparation complete: {station_data.shape} rows, {station_data.shape[1]} columns")
    return station_data

def train_optimized_model(station_data, target_pollutant):
    """Train an efficient LightGBM model with regularization, early stopping and purga temporal."""
    # Prepare features and target
    exclude_columns = ['Measurement date', 'Station code', target_pollutant]
    exclude_columns.extend([col for col in station_data.columns if not pd.api.types.is_numeric_dtype(station_data[col])])
    
    numeric_columns = [col for col in station_data.columns if col not in exclude_columns]
    print(f"Using {len(numeric_columns)} features")
    
    X = station_data[numeric_columns]
    y = station_data[target_pollutant]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Time series cross-validation (smaller split for speed)
    tscv = TimeSeriesSplit(n_splits=3)
    
    # Optimized hyperparameters by contaminante
    hyperparams = {
        'SO2': {
            'learning_rate': 0.05,
            'n_estimators': 150,
            'num_leaves': 31,
            'reg_alpha': 0.3,
            'reg_lambda': 0.3,
            'min_child_samples': 20,
            'feature_fraction': 0.8
        },
        'NO2': {
            'learning_rate': 0.05,
            'n_estimators': 150,
            'num_leaves': 31,
            'reg_alpha': 0.3,
            'reg_lambda': 0.3,
            'min_child_samples': 20,
            'feature_fraction': 0.8
        },
        'O3': {
            'learning_rate': 0.05,
            'n_estimators': 150,
            'num_leaves': 31,
            'reg_alpha': 0.3,
            'reg_lambda': 0.3,
            'min_child_samples': 20,
            'feature_fraction': 0.8
        },
        'CO': {
            'learning_rate': 0.05,
            'n_estimators': 150,
            'num_leaves': 31,
            'reg_alpha': 0.5,
            'reg_lambda': 0.5,
            'min_child_samples': 30,
            'feature_fraction': 0.7
        },
        'PM10': {
            'learning_rate': 0.05,
            'n_estimators': 150,
            'num_leaves': 31,
            'reg_alpha': 0.3,
            'reg_lambda': 0.3,
            'min_child_samples': 20,
            'feature_fraction': 0.8
        },
        'PM2.5': {
            'learning_rate': 0.05,
            'n_estimators': 150,
            'num_leaves': 31,
            'reg_alpha': 0.3,
            'reg_lambda': 0.3,
            'min_child_samples': 20,
            'feature_fraction': 0.8
        }
    }
    
    # Usar parámetros específicos o default
    params = hyperparams.get(target_pollutant, {
        'learning_rate': 0.05,
        'n_estimators': 150,
        'num_leaves': 31,
        'reg_alpha': 0.3,
        'reg_lambda': 0.3,
        'min_child_samples': 20,
        'feature_fraction': 0.8
    })
    
    # Train model with added regularization
    model = lgbm.LGBMRegressor(
        n_estimators=params['n_estimators'],
        learning_rate=params['learning_rate'],
        num_leaves=params['num_leaves'],
        reg_alpha=params['reg_alpha'],
        reg_lambda=params['reg_lambda'],
        min_child_samples=params['min_child_samples'],
        feature_fraction=params['feature_fraction'],
        random_state=42,
        n_jobs=-1
    )
    
    # Entrenar el modelo final en todos los datos
    model.fit(X_scaled, y)
    
    # Validate model using cross-validation with purga temporal
    cv_scores = {
        'mae': [],
        'rmse': [],
        'r2': []
    }
    
    # Período de purga (24 horas para evitar fugas de información)
    purge_period = 24
    
    print("Performing cross-validation...")
    for train_idx, test_idx in tscv.split(X_scaled):
        # Implementar purga temporal
        max_train_idx = max(train_idx)
        min_test_idx = min(test_idx)
        
        # Eliminar datos del período de purga
        if min_test_idx - max_train_idx < purge_period:
            purge_start = min_test_idx - purge_period
            train_idx = [idx for idx in train_idx if idx < purge_start]
        
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Train model on this fold - simplificado para evitar problemas
        fold_model = lgbm.LGBMRegressor(
            n_estimators=params['n_estimators'],
            learning_rate=params['learning_rate'],
            num_leaves=params['num_leaves'],
            reg_alpha=params['reg_alpha'],
            reg_lambda=params['reg_lambda'],
            min_child_samples=params['min_child_samples'],
            feature_fraction=params['feature_fraction'],
            random_state=42,
            n_jobs=-1
        )
        
        # Entrenamiento simple sin early stopping
        fold_model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = fold_model.predict(X_test)
        
        # Calculate metrics
        cv_scores['mae'].append(mean_absolute_error(y_test, y_pred))
        cv_scores['rmse'].append(np.sqrt(mean_squared_error(y_test, y_pred)))
        cv_scores['r2'].append(r2_score(y_test, y_pred))
    
    # Print CV scores
    print("Cross-validation scores:")
    print(f"MAE: {np.mean(cv_scores['mae']):.4f} (±{np.std(cv_scores['mae']):.4f})")
    print(f"RMSE: {np.mean(cv_scores['rmse']):.4f} (±{np.std(cv_scores['rmse']):.4f})")
    print(f"R²: {np.mean(cv_scores['r2']):.4f} (±{np.std(cv_scores['r2']):.4f})")
    
    return model, numeric_columns, scaler

def analyze_feature_importance(model, feature_names, target_pollutant):
    """Analyze and visualize feature importance."""
    try:
        # Extract feature importance
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False).head(20)
        
        # Create visualization
        plt.figure(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title(f'Top 20 Features for {target_pollutant} Prediction')
        plt.tight_layout()
        
        # Save the plot
        output_dir = Path("reports/figures")
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / f"{target_pollutant}_feature_importance_optimized.png")
        plt.close()
        
        print(f"Feature importance plot saved to reports/figures/{target_pollutant}_feature_importance_optimized.png")
        
        return importance_df
    except Exception as e:
        print(f"Error analyzing feature importance: {e}")
        return None

def prepare_prediction_data(dates, historical_data, target_pollutant, feature_names):
    """Prepare data for prediction with minimal features."""
    print(f"Preparing prediction data for {len(dates)} future timestamps")
    
    # Initialize DataFrame with dates
    pred_data = pd.DataFrame(index=range(len(dates)))
    
    # Asegurarnos de usar exactamente las mismas columnas que en entrenamiento
    for feature in feature_names:
        pred_data[feature] = 0.0  # Inicializar con valores por defecto
    
    # Add time features (asegurándonos que usan los mismos nombres)
    for i, date in enumerate(dates):
        # Características temporales
        hour = date.hour
        day = date.day
        month = date.month
        day_of_week = date.dayofweek
        day_of_year = date.dayofyear
        
        # Establecer características temporales si están en feature_names
        if 'hour' in feature_names:
            pred_data.loc[i, 'hour'] = hour
        if 'day' in feature_names:
            pred_data.loc[i, 'day'] = day
        if 'month' in feature_names:
            pred_data.loc[i, 'month'] = month
        if 'day_of_week' in feature_names:
            pred_data.loc[i, 'day_of_week'] = day_of_week
        if 'day_of_year' in feature_names:
            pred_data.loc[i, 'day_of_year'] = day_of_year
            
        # Características cíclicas
        if 'hour_sin' in feature_names:
            pred_data.loc[i, 'hour_sin'] = np.sin(2 * np.pi * hour/24)
        if 'hour_cos' in feature_names:
            pred_data.loc[i, 'hour_cos'] = np.cos(2 * np.pi * hour/24)
        if 'day_sin' in feature_names:
            pred_data.loc[i, 'day_sin'] = np.sin(2 * np.pi * day/31)
        if 'day_cos' in feature_names:
            pred_data.loc[i, 'day_cos'] = np.cos(2 * np.pi * day/31)
        if 'month_sin' in feature_names:
            pred_data.loc[i, 'month_sin'] = np.sin(2 * np.pi * month/12)
        if 'month_cos' in feature_names:
            pred_data.loc[i, 'month_cos'] = np.cos(2 * np.pi * month/12)
            
        # Fin de semana
        if 'is_weekend' in feature_names:
            pred_data.loc[i, 'is_weekend'] = 1 if day_of_week >= 5 else 0
        
        # Variables ficticias (dummies) para temporada
        season = None
        if month in [12, 1, 2]:
            season = 'winter'
        elif month in [3, 4, 5]:
            season = 'spring'
        elif month in [6, 7, 8]:
            season = 'summer'
        else:
            season = 'fall'
            
        # Asignar las variables dummy si existen en features
        for s in ['winter', 'spring', 'summer', 'fall']:
            col = f'season_{s}'
            if col in feature_names:
                pred_data.loc[i, col] = 1 if season == s else 0
    
    # Get most recent data points for lags
    last_known_values = historical_data.sort_values('Measurement date').tail(24*7)
    
    # Initialize lag features with last known values
    key_lags = [1, 24, 48, 24*7]
    for lag in key_lags:
        lag_feature = f'{target_pollutant}_lag_{lag}'
        if lag_feature in feature_names:
            if lag < len(last_known_values):
                last_val = last_known_values[target_pollutant].iloc[-lag]
                pred_data[lag_feature] = last_val
            else:
                mean_val = last_known_values[target_pollutant].mean()
                pred_data[lag_feature] = mean_val
    
    # Initialize rolling statistics with historical values
    key_windows = [24, 24*7]
    for window in key_windows:
        mean_feature = f'{target_pollutant}_rolling_mean_{window}'
        std_feature = f'{target_pollutant}_rolling_std_{window}'
        
        if mean_feature in feature_names:
            if window < len(last_known_values):
                mean_val = last_known_values[target_pollutant].iloc[-window:].mean()
                pred_data[mean_feature] = mean_val
            else:
                mean_val = last_known_values[target_pollutant].mean()
                pred_data[mean_feature] = mean_val
        
        if std_feature in feature_names:
            if window < len(last_known_values):
                std_val = last_known_values[target_pollutant].iloc[-window:].std()
                pred_data[std_feature] = std_val
            else:
                std_val = last_known_values[target_pollutant].std()
                pred_data[std_feature] = std_val
    
    # Day-of-week and hour-of-day averages from historical data
    dow_feature = f'{target_pollutant}_dow_mean'
    hour_feature = f'{target_pollutant}_hour_mean'
    
    if dow_feature in feature_names:
        dow_means = historical_data.groupby('day_of_week')[target_pollutant].mean().to_dict()
        for i, date in enumerate(dates):
            dow = date.dayofweek
            pred_data.loc[i, dow_feature] = dow_means.get(dow, last_known_values[target_pollutant].mean())
    
    if hour_feature in feature_names:
        hour_means = historical_data.groupby('hour')[target_pollutant].mean().to_dict()
        for i, date in enumerate(dates):
            hour = date.hour
            pred_data.loc[i, hour_feature] = hour_means.get(hour, last_known_values[target_pollutant].mean())
    
    # Add other pollutant relationships if in feature_names
    pollutants = ['SO2', 'NO2', 'O3', 'CO', 'PM10', 'PM2.5']
    for other_pollutant in pollutants:
        if other_pollutant != target_pollutant and other_pollutant in historical_data.columns:
            ratio_feature = f'ratio_{target_pollutant}_{other_pollutant}'
            if ratio_feature in feature_names:
                # Use the mean ratio from historical data
                other_mean = historical_data[other_pollutant].replace(0, np.nan).fillna(historical_data[other_pollutant].mean()).mean()
                if other_mean != 0:
                    mean_ratio = (historical_data[target_pollutant].mean() / other_mean)
                else:
                    mean_ratio = 1.0
                pred_data[ratio_feature] = mean_ratio
    
    # Verificar que todas las características estén presentes
    missing_features = set(feature_names) - set(pred_data.columns)
    if missing_features:
        print(f"Warning: Still missing {len(missing_features)} features. Setting to 0.")
        for feature in missing_features:
            pred_data[feature] = 0.0
    
    # Verificar que no hay columnas adicionales
    extra_columns = set(pred_data.columns) - set(feature_names)
    if extra_columns:
        print(f"Warning: Found {len(extra_columns)} extra columns. Removing them.")
        pred_data = pred_data[feature_names]
    
    print(f"Prediction data prepared with {pred_data.shape[1]} features")
    return pred_data

def predict_pollutant_optimized(measurement_df, station_code, pollutant, start_date, end_date):
    """Generate optimized predictions for a pollutant at a specific station."""
    print(f"Starting predictions for Station {station_code} - {pollutant}")
    
    # Prepare data for the station
    station_data = prepare_features(measurement_df, station_code, pollutant)
    
    # Train model
    model, feature_names, scaler = train_optimized_model(station_data, pollutant)
    
    # Analyze feature importance
    analyze_feature_importance(model, feature_names, pollutant)
    
    # Create date range for prediction
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    prediction_dates = pd.date_range(start=start_date, end=end_date, freq='H')
    
    print(f"Preparing prediction data for {len(prediction_dates)} future timestamps")
    
    # Prepare prediction data
    prediction_data = prepare_prediction_data(prediction_dates, station_data, pollutant, feature_names)
    
    # Aplicar el mismo escalado usado en entrenamiento
    prediction_data_scaled = scaler.transform(prediction_data)
    
    print(f"Generating predictions for {len(prediction_dates)} timestamps")
    
    # Generate predictions
    predictions = model.predict(prediction_data_scaled)
    
    # Ensure all predictions are valid numbers
    predictions = np.clip(predictions, 0, None)  # No negative values
    
    # Create output format
    result = pd.DataFrame({
        'date': prediction_dates,
        'value': predictions,
        'station': station_code,
        'pollutant': pollutant
    })
    
    # Save model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, f'models/{station_code}_{pollutant}_model_optimized.pkl')
    print(f"Model saved to models/{station_code}_{pollutant}_model_optimized.pkl")
    
    # Visualize predictions
    visualize_predictions(station_code, pollutant, station_data, result)
    
    print(f"Predictions complete for Station {station_code} - {pollutant}")
    
    return result

def visualize_predictions(station_code, pollutant, historical_data, predictions):
    """Visualize historical data and predictions with simplified plot."""
    try:
        # Ahora predictions es un DataFrame, usamos directamente
        pred_df = predictions
        
        # Get historical data
        hist_df = historical_data[['Measurement date', pollutant]].copy()
        hist_df = hist_df.rename(columns={'Measurement date': 'date'})
        
        # Get last 30 days of historical data
        last_date = hist_df['date'].max()
        start_date = last_date - pd.Timedelta(days=30)
        hist_df = hist_df[hist_df['date'] >= start_date]
        
        # Create visualization
        plt.figure(figsize=(15, 7))
        plt.plot(hist_df['date'], hist_df[pollutant], label='Historical Data', color='blue')
        plt.plot(pred_df['date'], pred_df['value'], label='Predictions', color='red', linestyle='--')
        
        plt.title(f'Station {station_code} - {pollutant} Forecast (Optimized)')
        plt.xlabel('Date')
        plt.ylabel(f'{pollutant} Concentration')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add vertical line separating historical from predictions
        plt.axvline(x=last_date, color='green', linestyle='-', label='Forecast Start')
        
        plt.tight_layout()
        
        # Save the plot
        output_dir = Path("reports/figures")
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / f"{station_code}_{pollutant}_forecast_optimized.png")
        plt.close()
        
        print(f"Forecast visualization saved to reports/figures/{station_code}_{pollutant}_forecast_optimized.png")
    except Exception as e:
        print(f"Error visualizing predictions: {e}")

def main():
    """Run the optimized forecasting pipeline."""
    print("Starting optimized forecasting pipeline...")
    start_time = datetime.now()
    print(f"Start time: {start_time}")
    
    # Load data
    measurement_df = load_processed_data()
    
    # Detect outliers using IsolationForest (más preciso)
    measurement_df = detect_outliers(measurement_df)
    
    # Add time features
    measurement_df = add_time_features(measurement_df)
    
    # Prediction tasks
    prediction_tasks = [
        {'station': 206, 'pollutant': 'SO2', 'start_date': '2023-04-01', 'end_date': '2023-04-30'},
        {'station': 211, 'pollutant': 'NO2', 'start_date': '2023-04-01', 'end_date': '2023-04-30'},
        {'station': 217, 'pollutant': 'O3', 'start_date': '2023-04-01', 'end_date': '2023-04-30'},
        {'station': 219, 'pollutant': 'CO', 'start_date': '2023-04-01', 'end_date': '2023-04-30'},
        {'station': 225, 'pollutant': 'PM10', 'start_date': '2023-04-01', 'end_date': '2023-04-30'},
        {'station': 228, 'pollutant': 'PM2.5', 'start_date': '2023-04-01', 'end_date': '2023-04-30'}
    ]
    
    # Store all predictions
    all_predictions = []
    
    # Process each task
    for task in prediction_tasks:
        predictions = predict_pollutant_optimized(
            measurement_df, 
            task['station'], 
            task['pollutant'], 
            task['start_date'], 
            task['end_date']
        )
        all_predictions.append(predictions)
    
    # Format predictions for JSON output
    predictions_dict = {}
    for pred_df in all_predictions:
        station = str(pred_df['station'].iloc[0])
        pollutant = pred_df['pollutant'].iloc[0]
        
        if station not in predictions_dict:
            predictions_dict[station] = {}
        
        predictions_dict[station][pollutant] = []
        for _, row in pred_df.iterrows():
            predictions_dict[station][pollutant].append({
                'date': row['date'].strftime('%Y-%m-%d %H:%M:%S'),
                'value': float(row['value'])
            })
    
    # Save results
    os.makedirs('predictions', exist_ok=True)
    with open('predictions/predictions_task_2_optimized.json', 'w') as f:
        json.dump(predictions_dict, f, indent=2)
    print("Predictions saved to predictions/predictions_task_2_optimized.json")
    
    # Calculate execution time
    end_time = datetime.now()
    execution_time = end_time - start_time
    
    print("="*80)
    print(f"Optimized forecasting pipeline completed in {execution_time}")
    print("Predictions saved to predictions/predictions_task_2_optimized.json")
    print("="*80)

if __name__ == "__main__":
    main() 