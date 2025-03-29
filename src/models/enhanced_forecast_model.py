import pandas as pd
import numpy as np
from pathlib import Path
import json
import joblib
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import VotingRegressor, StackingRegressor, RandomForestRegressor, GradientBoostingRegressor, IsolationForest
from sklearn.neighbors import KNeighborsRegressor
from sklearn.impute import KNNImputer
from sklearn.feature_selection import RFECV, SelectFromModel
import xgboost as xgb
from lightgbm import LGBMRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

def load_processed_data():
    """Load processed datasets with advanced error handling."""
    processed_dir = Path("data/processed")
    
    try:
        measurement_df = pd.read_csv(
            processed_dir / "measurement_data_processed.csv",
            parse_dates=['Measurement date']
        )
        print(f"Data loaded successfully: {measurement_df.shape} rows")
        
        # Check data quality
        missing_values = measurement_df.isnull().sum()
        if missing_values.sum() > 0:
            print(f"Warning: Found {missing_values.sum()} missing values")
            
        # Advanced anomaly detection with Isolation Forest for each pollutant
        pollutants = ['SO2', 'NO2', 'O3', 'CO', 'PM10', 'PM2.5']
        for pollutant in pollutants:
            if pollutant in measurement_df.columns:
                # Get data for this pollutant (removing NaN values)
                pollutant_data = measurement_df[[pollutant]].dropna()
                
                if len(pollutant_data) > 0:
                    # Fit Isolation Forest
                    iso_forest = IsolationForest(
                        contamination=0.05,  # Assuming 5% outliers
                        random_state=42,
                        n_estimators=100
                    )
                    
                    # Predict anomalies
                    outlier_pred = iso_forest.fit_predict(pollutant_data)
                    
                    # Get indices of outliers (-1 indicates outlier)
                    outlier_indices = np.where(outlier_pred == -1)[0]
                    
                    # Count outliers
                    print(f"Found {len(outlier_indices)} outliers in {pollutant}")
                    
                    # Create anomaly flag column
                    anomaly_col = f"{pollutant}_is_anomaly"
                    measurement_df[anomaly_col] = 0
                    
                    # Mark detected anomalies
                    for idx in outlier_indices:
                        measurement_df.loc[pollutant_data.index[idx], anomaly_col] = 1
                    
                    # Calculate statistics for anomalies and normal data
                    anomalies = measurement_df[measurement_df[anomaly_col] == 1][pollutant]
                    normal_data = measurement_df[measurement_df[anomaly_col] == 0][pollutant]
                    
                    if len(anomalies) > 0:
                        # Cap extreme outliers (retain some information but reduce extreme influence)
                        # Use 99th percentile of normal data as cap
                        upper_cap = normal_data.quantile(0.99)
                        measurement_df[pollutant] = np.where(
                            measurement_df[anomaly_col] == 1,
                            np.minimum(measurement_df[pollutant], upper_cap),
                            measurement_df[pollutant]
                        )
        
        # Use KNN imputation for missing values
        numeric_cols = measurement_df.select_dtypes(include=[np.number]).columns.tolist()
        non_anomaly_cols = [col for col in numeric_cols if not col.endswith('_is_anomaly')]
        
        # If there are missing values, use KNN imputation
        if missing_values.sum() > 0:
            imputer = KNNImputer(n_neighbors=5, weights='distance')
            measurement_df[non_anomaly_cols] = imputer.fit_transform(measurement_df[non_anomaly_cols])
            
            # Add flags for imputed values
            for col in non_anomaly_cols:
                if missing_values[col] > 0:
                    measurement_df[f"{col}_imputed"] = 0
                    # Mark previously NaN values as imputed
                    measurement_df.loc[measurement_df[col].isnull(), f"{col}_imputed"] = 1
        
        return measurement_df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def add_advanced_time_features(df):
    """Add advanced time-based features."""
    # Extract basic time components
    df['hour'] = df['Measurement date'].dt.hour
    df['day'] = df['Measurement date'].dt.day
    df['month'] = df['Measurement date'].dt.month
    df['year'] = df['Measurement date'].dt.year
    df['day_of_week'] = df['Measurement date'].dt.dayofweek
    df['quarter'] = df['Measurement date'].dt.quarter
    df['week_of_year'] = df['Measurement date'].dt.isocalendar().week
    
    # Cyclical encoding for time features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    df['day_sin'] = np.sin(2 * np.pi * df['day']/31)
    df['day_cos'] = np.cos(2 * np.pi * df['day']/31)
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week']/7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week']/7)
    
    # Time categories
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    df['is_holiday'] = 0  # Placeholder for holiday detection
    
    # Time of day categories
    df['time_of_day'] = pd.cut(
        df['hour'], 
        bins=[0, 6, 12, 18, 24], 
        labels=['night', 'morning', 'afternoon', 'evening']
    )
    df = pd.get_dummies(df, columns=['time_of_day'])
    
    # Season encoding with more precise calculations
    # Compute meteorological seasons (Northern Hemisphere)
    month_day = 100 * df['month'] + df['day']
    season_bins = [0, 228, 531, 831, 1130, 1231]  # Dec31=1231, Mar20=320, Jun21=621, Sep22=922, Dec21=1221
    season_labels = ['winter', 'spring', 'summer', 'fall', 'winter']
    df['season'] = pd.cut(month_day, bins=season_bins, labels=season_labels, ordered=False)
    df = pd.get_dummies(df, columns=['season'])
    
    # Add more granular time features
    df['is_morning_rush'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
    df['is_evening_rush'] = ((df['hour'] >= 16) & (df['hour'] <= 19)).astype(int)
    df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17) & (df['day_of_week'] < 5)).astype(int)
    
    # Monthly seasonality
    for month in range(1, 13):
        df[f'is_month_{month}'] = (df['month'] == month).astype(int)
    
    return df

def add_weather_proxy_features(station_data, target_pollutant):
    """Add proxy features that simulate weather impacts (without actual weather data)."""
    # Season-based temperature proxy (seasonal cycles)
    station_data['temp_proxy'] = np.sin(2 * np.pi * station_data['month'] / 12 - np.pi/2)
    
    # Humidity proxy based on time of year (higher in winter, lower in summer in many regions)
    station_data['humidity_proxy'] = -station_data['temp_proxy'] * 0.5 + 0.5
    
    # Wind speed proxy (tend to be higher in certain months)
    station_data['wind_proxy'] = np.cos(2 * np.pi * station_data['month'] / 12) * 0.3 + np.random.normal(0, 0.1, len(station_data))
    
    # Daily temperature cycle proxy (highest in afternoon, lowest before dawn)
    station_data['daily_temp_cycle'] = np.sin(2 * np.pi * (station_data['hour'] - 3) / 24)
    
    # Interaction features between pollutant and weather proxies
    station_data[f'{target_pollutant}_temp_interaction'] = station_data[target_pollutant] * station_data['temp_proxy']
    station_data[f'{target_pollutant}_humidity_interaction'] = station_data[target_pollutant] * station_data['humidity_proxy']
    
    return station_data

def add_trend_features(station_data, target_pollutant):
    """Add trend-based features to capture long-term patterns."""
    # Daily trends
    station_data['day_trend'] = station_data[target_pollutant].rolling(window=24).mean()
    
    # Weekly trends
    station_data['week_trend'] = station_data[target_pollutant].rolling(window=24*7).mean()
    
    # Monthly trends
    station_data['month_trend'] = station_data[target_pollutant].rolling(window=24*30).mean()
    
    # Trend directionality
    station_data['trend_direction'] = station_data[target_pollutant].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    
    # Exponentially weighted moving averages (EWMA)
    alphas = [0.05, 0.1, 0.3, 0.5]
    for alpha in alphas:
        station_data[f'{target_pollutant}_ewma_{int(alpha*100)}'] = station_data[target_pollutant].ewm(alpha=alpha).mean()
    
    return station_data

def add_statistical_features(station_data, target_pollutant):
    """Add statistical features to capture data distribution characteristics."""
    # Calculate rolling statistics
    windows = [6, 12, 24, 48]
    for window in windows:
        # Standard deviation to capture volatility
        station_data[f'{target_pollutant}_std_{window}'] = station_data[target_pollutant].rolling(window=window).std()
        
        # Min and max to capture range
        station_data[f'{target_pollutant}_min_{window}'] = station_data[target_pollutant].rolling(window=window).min()
        station_data[f'{target_pollutant}_max_{window}'] = station_data[target_pollutant].rolling(window=window).max()
        
        # Quantiles to capture distribution
        station_data[f'{target_pollutant}_q25_{window}'] = station_data[target_pollutant].rolling(window=window).quantile(0.25)
        station_data[f'{target_pollutant}_q75_{window}'] = station_data[target_pollutant].rolling(window=window).quantile(0.75)
        
        # Skewness and kurtosis to capture shape
        station_data[f'{target_pollutant}_skew_{window}'] = station_data[target_pollutant].rolling(window=window).apply(lambda x: skew(x) if len(x) > 2 else 0)
        station_data[f'{target_pollutant}_kurt_{window}'] = station_data[target_pollutant].rolling(window=window).apply(lambda x: kurtosis(x) if len(x) > 2 else 0)
        
        # Rate of change features
        station_data[f'{target_pollutant}_roc_{window}'] = station_data[target_pollutant].pct_change(periods=window)
    
    return station_data

def prepare_features(df, target_station, target_pollutant):
    """Prepare comprehensive feature set for the model."""
    print(f"Preparing features for Station {target_station}, Pollutant {target_pollutant}")
    
    # Filter data for target station
    station_data = df[df['Station code'] == target_station].copy()
    
    # Ensure time-ordered data
    station_data = station_data.sort_values('Measurement date')
    
    # Handle anomalies if flagged in preprocessing
    anomaly_col = f"{target_pollutant}_is_anomaly"
    if anomaly_col in station_data.columns:
        # Create distance-to-anomaly feature - useful for modeling
        station_data['days_since_anomaly'] = 0
        station_data['days_until_anomaly'] = 0
        
        # Calculate days since/until an anomaly
        anomaly_dates = station_data[station_data[anomaly_col] == 1]['Measurement date'].values
        for idx, row in station_data.iterrows():
            date = row['Measurement date']
            # Days since last anomaly
            past_anomalies = anomaly_dates[anomaly_dates < date]
            if len(past_anomalies) > 0:
                last_anomaly = max(past_anomalies)
                station_data.loc[idx, 'days_since_anomaly'] = (date - last_anomaly).total_seconds() / (60 * 60 * 24)
            
            # Days until next anomaly
            future_anomalies = anomaly_dates[anomaly_dates > date]
            if len(future_anomalies) > 0:
                next_anomaly = min(future_anomalies)
                station_data.loc[idx, 'days_until_anomaly'] = (next_anomaly - date).total_seconds() / (60 * 60 * 24)
    
    # Add advanced time features
    station_data = add_advanced_time_features(station_data)
    
    # Add weather proxy features
    station_data = add_weather_proxy_features(station_data, target_pollutant)
    
    # Add lag features (previous hours) with more lags
    for lag in [1, 2, 3, 6, 12, 24, 36, 48, 72, 96, 120, 24*7, 24*14]:
        station_data[f'{target_pollutant}_lag_{lag}'] = station_data[target_pollutant].shift(lag)
    
    # Add rolling window statistics with different window sizes
    for window in [3, 6, 12, 24, 36, 48, 72, 96, 24*7, 24*14]:
        station_data[f'{target_pollutant}_rolling_mean_{window}'] = station_data[target_pollutant].rolling(window=window).mean()
        station_data[f'{target_pollutant}_rolling_median_{window}'] = station_data[target_pollutant].rolling(window=window).median()
        station_data[f'{target_pollutant}_rolling_std_{window}'] = station_data[target_pollutant].rolling(window=window).std()
        station_data[f'{target_pollutant}_rolling_min_{window}'] = station_data[target_pollutant].rolling(window=window).min()
        station_data[f'{target_pollutant}_rolling_max_{window}'] = station_data[target_pollutant].rolling(window=window).max()
    
    # Add trend-based features
    station_data = add_trend_features(station_data, target_pollutant)
    
    # Add statistical features
    station_data = add_statistical_features(station_data, target_pollutant)
    
    # Add cross-pollutant relationships if multiple pollutants are available
    pollutants = ['SO2', 'NO2', 'O3', 'CO', 'PM10', 'PM2.5']
    for other_pollutant in pollutants:
        if other_pollutant != target_pollutant and other_pollutant in station_data.columns:
            # Correlation between target and other pollutant
            window_sizes = [24, 48, 72, 24*7]
            for window in window_sizes:
                # Calculate rolling correlation
                target_values = station_data[target_pollutant].rolling(window=window)
                other_values = station_data[other_pollutant].rolling(window=window)
                corr_name = f'corr_{target_pollutant}_{other_pollutant}_{window}'
                
                # Calculate correlation (handle edge cases)
                def safe_corr(x, y):
                    if len(x) <= 1 or len(y) <= 1:
                        return 0
                    if x.std() == 0 or y.std() == 0:
                        return 0
                    return np.corrcoef(x, y)[0, 1]
                
                # Apply to each window
                correlations = []
                for i in range(len(station_data)):
                    if i < window:
                        correlations.append(0)  # Not enough data
                    else:
                        x = station_data[target_pollutant].iloc[i-window:i].values
                        y = station_data[other_pollutant].iloc[i-window:i].values
                        correlations.append(safe_corr(x, y))
                
                station_data[corr_name] = correlations
            
            # Other simple relationships
            station_data[f'ratio_{target_pollutant}_{other_pollutant}'] = station_data[target_pollutant] / station_data[other_pollutant].replace(0, np.nan).fillna(station_data[other_pollutant].mean())
    
    # Add day-of-week and hour-of-day averages
    dow_mean = station_data.groupby('day_of_week')[target_pollutant].transform('mean')
    hour_mean = station_data.groupby('hour')[target_pollutant].transform('mean')
    station_data[f'{target_pollutant}_dow_mean'] = dow_mean
    station_data[f'{target_pollutant}_hour_mean'] = hour_mean
    
    # Add deviations from typical patterns
    station_data[f'{target_pollutant}_dow_deviation'] = station_data[target_pollutant] - dow_mean
    station_data[f'{target_pollutant}_hour_deviation'] = station_data[target_pollutant] - hour_mean
    
    # Handle missing values in features
    station_data = station_data.replace([np.inf, -np.inf], np.nan)
    
    # Forward fill missing values first, then backfill any remaining
    station_data = station_data.fillna(method='ffill')
    station_data = station_data.fillna(method='bfill')
    
    # Make sure we don't have missing values in target column
    if station_data[target_pollutant].isnull().any():
        print(f"Warning: Target column {target_pollutant} still has {station_data[target_pollutant].isnull().sum()} missing values after imputation")
        # Fill any remaining missing values with the median
        station_data[target_pollutant] = station_data[target_pollutant].fillna(station_data[target_pollutant].median())
    
    print(f"Feature preparation complete: {station_data.shape} rows, {station_data.shape[1]} columns")
    return station_data

def train_model_with_validation(station_data, target_pollutant, optimize_hyperparams=False):
    """Train an ensemble model with time-series cross-validation."""
    # Prepare data for modeling
    features = station_data.drop(['Measurement date', 'Station code', target_pollutant], axis=1, errors='ignore')
    
    # Filter only numeric columns
    numeric_features = features.select_dtypes(include=[np.number])
    print(f"Using {numeric_features.shape[1]} numeric features")
    
    # Apply feature selection to reduce dimensionality if we have many features
    if numeric_features.shape[1] > 50:
        print(f"Reduced features from {numeric_features.shape[1]} to ", end="")
        
        # Use Lasso for feature selection
        selector = SelectFromModel(
            Lasso(alpha=0.005, random_state=42),
            max_features=max(42, int(numeric_features.shape[1] * 0.4))  # Keep at most 40% or 42 features
        )
        
        X_selected = selector.fit_transform(numeric_features, station_data[target_pollutant])
        selected_indices = selector.get_support()
        selected_features = numeric_features.columns[selected_indices].tolist()
        
        print(f"{len(selected_features)}")
        
        # Keep only selected features
        X = numeric_features[selected_features]
    else:
        X = numeric_features
        selected_features = numeric_features.columns.tolist()
    
    # Extract target variable
    y = station_data[target_pollutant]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create time-series cross-validation splits
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Initialize models dictionary
    models = {}
    
    # If optimize_hyperparams is True, use Bayesian optimization to find best parameters
    if optimize_hyperparams:
        print("Optimizing hyperparameters with Bayesian search...")
        
        # Define hyperparameter search spaces
        xgb_space = {
            'n_estimators': Integer(50, 500),
            'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
            'max_depth': Integer(3, 10),
            'subsample': Real(0.6, 1.0),
            'colsample_bytree': Real(0.6, 1.0),
            'min_child_weight': Integer(1, 10),
            'gamma': Real(0, 5)
        }
        
        lgbm_space = {
            'n_estimators': Integer(50, 500),
            'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
            'max_depth': Integer(3, 10),
            'subsample': Real(0.6, 1.0),
            'colsample_bytree': Real(0.6, 1.0),
            'reg_alpha': Real(0, 10),
            'reg_lambda': Real(0, 10),
            'min_child_samples': Integer(5, 50)
        }
        
        rf_space = {
            'n_estimators': Integer(50, 300),
            'max_depth': Integer(5, 20),
            'min_samples_split': Integer(2, 10),
            'min_samples_leaf': Integer(1, 5),
            'max_features': Categorical(['sqrt', 'log2', None])
        }
        
        gbr_space = {
            'n_estimators': Integer(50, 300),
            'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
            'max_depth': Integer(3, 10),
            'min_samples_split': Integer(2, 10),
            'min_samples_leaf': Integer(1, 5),
            'subsample': Real(0.6, 1.0)
        }
        
        # Setup Bayesian search objects
        xgb_search = BayesSearchCV(
            xgb.XGBRegressor(random_state=42),
            xgb_space,
            n_iter=20,
            cv=tscv,
            n_jobs=-1,
            scoring='neg_mean_squared_error',
            random_state=42
        )
        
        lgbm_search = BayesSearchCV(
            LGBMRegressor(random_state=42),
            lgbm_space,
            n_iter=20,
            cv=tscv,
            n_jobs=-1,
            scoring='neg_mean_squared_error',
            random_state=42
        )
        
        rf_search = BayesSearchCV(
            RandomForestRegressor(random_state=42),
            rf_space,
            n_iter=15,
            cv=tscv,
            n_jobs=-1,
            scoring='neg_mean_squared_error',
            random_state=42
        )
        
        gbr_search = BayesSearchCV(
            GradientBoostingRegressor(random_state=42),
            gbr_space,
            n_iter=15,
            cv=tscv,
            n_jobs=-1,
            scoring='neg_mean_squared_error',
            random_state=42
        )
        
        # Execute searches
        print("Optimizing XGBoost...")
        xgb_search.fit(X_scaled, y)
        models['xgb'] = xgb_search.best_estimator_
        print(f"Best XGBoost parameters: {xgb_search.best_params_}")
        
        print("Optimizing LightGBM...")
        lgbm_search.fit(X_scaled, y)
        models['lgbm'] = lgbm_search.best_estimator_
        print(f"Best LightGBM parameters: {lgbm_search.best_params_}")
        
        print("Optimizing Random Forest...")
        rf_search.fit(X_scaled, y)
        models['rf'] = rf_search.best_estimator_
        print(f"Best Random Forest parameters: {rf_search.best_params_}")
        
        print("Optimizing Gradient Boosting...")
        gbr_search.fit(X_scaled, y)
        models['gbr'] = gbr_search.best_estimator_
        print(f"Best Gradient Boosting parameters: {gbr_search.best_params_}")
        
    else:
        # Use default hyperparameters for each model
        models['xgb'] = xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        models['lgbm'] = LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        models['rf'] = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        models['gbr'] = GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.8,
            random_state=42
        )
        
        # Train individual models
        for name, model in models.items():
            model.fit(X_scaled, y)
    
    # Create stacking ensemble
    final_estimator = Ridge(alpha=1.0)
    stack_levels = [
        ('xgb', models['xgb']),
        ('lgbm', models['lgbm']),
        ('rf', models['rf']),
        ('gbr', models['gbr'])
    ]
    
    ensemble = StackingRegressor(
        estimators=stack_levels,
        final_estimator=final_estimator,
        cv=3,
        n_jobs=-1
    )
    
    # Train ensemble
    ensemble.fit(X_scaled, y)
    
    # Validate model using cross-validation
    cv_scores = {
        'mae': [],
        'rmse': [],
        'r2': []
    }
    
    print("Performing cross-validation...")
    for train_idx, test_idx in tscv.split(X_scaled):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Train ensemble on this fold
        fold_ensemble = StackingRegressor(
            estimators=stack_levels,
            final_estimator=final_estimator,
            cv=3,
            n_jobs=-1
        )
        fold_ensemble.fit(X_train, y_train)
        
        # Make predictions
        y_pred = fold_ensemble.predict(X_test)
        
        # Calculate metrics
        cv_scores['mae'].append(mean_absolute_error(y_test, y_pred))
        cv_scores['rmse'].append(np.sqrt(mean_squared_error(y_test, y_pred)))
        cv_scores['r2'].append(r2_score(y_test, y_pred))
    
    # Print average cross-validation scores
    print("Cross-validation scores:")
    print(f"MAE: {np.mean(cv_scores['mae']):.4f} (±{np.std(cv_scores['mae']):.4f})")
    print(f"RMSE: {np.mean(cv_scores['rmse']):.4f} (±{np.std(cv_scores['rmse']):.4f})")
    print(f"R²: {np.mean(cv_scores['r2']):.4f} (±{np.std(cv_scores['r2']):.4f})")
    
    return ensemble, scaler, selected_features, cv_scores, models

def analyze_feature_importance(model, feature_names, target_pollutant):
    """Analyze and visualize feature importance for interpretability."""
    try:
        # Extract feature importance (only works for tree-based models)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'estimators_') and hasattr(model.estimators_[0], 'feature_importances_'):
            importances = np.mean([est.feature_importances_ for est in model.estimators_], axis=0)
        else:
            # For VotingRegressor, try to extract from the first tree-based model
            for name, est in model.estimators_:
                if hasattr(est, 'feature_importances_'):
                    importances = est.feature_importances_
                    break
            else:
                print("Could not extract feature importances")
                return
        
        # Create a DataFrame for better visualization
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
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
        plt.savefig(output_dir / f"{target_pollutant}_feature_importance.png")
        plt.close()
        
        print(f"Feature importance plot saved to reports/figures/{target_pollutant}_feature_importance.png")
        
        return importance_df
    except Exception as e:
        print(f"Error analyzing feature importance: {e}")
        return None

def generate_future_dates(start_date, end_date):
    """Generate hourly dates between start and end date."""
    dates = pd.date_range(start=start_date, end=end_date, freq='H')
    return dates

def prepare_advanced_prediction_data(dates, historical_data, target_pollutant):
    """Prepare advanced data for prediction with all engineered features."""
    print(f"Preparing prediction data for {len(dates)} future timestamps")
    
    # Initialize DataFrame with dates
    pred_data = pd.DataFrame(index=dates)
    pred_data['Measurement date'] = dates
    
    # Add time features
    pred_data = add_advanced_time_features(pred_data)
    
    # Get most recent data points for lags
    last_known_values = historical_data.sort_values('Measurement date').tail(24*7)  # Last week
    
    # Initialize lag features with last known values
    for lag in [1, 2, 3, 6, 12, 24, 48, 72, 24*7]:
        if lag < len(last_known_values):
            pred_data[f'{target_pollutant}_lag_{lag}'] = last_known_values[target_pollutant].iloc[-lag]
        else:
            pred_data[f'{target_pollutant}_lag_{lag}'] = last_known_values[target_pollutant].mean()
    
    # Initialize rolling statistics with historical values
    for window in [3, 6, 12, 24, 48, 72, 24*7]:
        if window < len(last_known_values):
            pred_data[f'{target_pollutant}_rolling_mean_{window}'] = last_known_values[target_pollutant].iloc[-window:].mean()
            pred_data[f'{target_pollutant}_rolling_median_{window}'] = last_known_values[target_pollutant].iloc[-window:].median()
            pred_data[f'{target_pollutant}_rolling_std_{window}'] = last_known_values[target_pollutant].iloc[-window:].std()
        else:
            pred_data[f'{target_pollutant}_rolling_mean_{window}'] = last_known_values[target_pollutant].mean()
            pred_data[f'{target_pollutant}_rolling_median_{window}'] = last_known_values[target_pollutant].median()
            pred_data[f'{target_pollutant}_rolling_std_{window}'] = last_known_values[target_pollutant].std()
    
    # Initialize trend features
    pred_data['day_trend'] = last_known_values[target_pollutant].mean()
    pred_data['week_trend'] = last_known_values[target_pollutant].mean()
    pred_data['month_trend'] = last_known_values[target_pollutant].mean()
    pred_data['trend_direction'] = 0
    
    # Initialize EWMA features
    alphas = [0.05, 0.1, 0.3, 0.5]
    for alpha in alphas:
        pred_data[f'{target_pollutant}_ewma_{int(alpha*100)}'] = last_known_values[target_pollutant].ewm(alpha=alpha).mean().iloc[-1]
    
    # Initialize statistical features
    windows = [6, 12, 24, 48]
    for window in windows:
        if window < len(last_known_values):
            window_data = last_known_values[target_pollutant].iloc[-window:]
            pred_data[f'{target_pollutant}_std_{window}'] = window_data.std()
            pred_data[f'{target_pollutant}_min_{window}'] = window_data.min()
            pred_data[f'{target_pollutant}_max_{window}'] = window_data.max()
            pred_data[f'{target_pollutant}_q25_{window}'] = window_data.quantile(0.25)
            pred_data[f'{target_pollutant}_q75_{window}'] = window_data.quantile(0.75)
            try:
                pred_data[f'{target_pollutant}_skew_{window}'] = skew(window_data) if len(window_data) > 2 else 0
                pred_data[f'{target_pollutant}_kurt_{window}'] = kurtosis(window_data) if len(window_data) > 2 else 0
            except:
                pred_data[f'{target_pollutant}_skew_{window}'] = 0
                pred_data[f'{target_pollutant}_kurt_{window}'] = 0
            pred_data[f'{target_pollutant}_roc_{window}'] = 0
        else:
            # Fallback to global statistics
            pred_data[f'{target_pollutant}_std_{window}'] = last_known_values[target_pollutant].std()
            pred_data[f'{target_pollutant}_min_{window}'] = last_known_values[target_pollutant].min()
            pred_data[f'{target_pollutant}_max_{window}'] = last_known_values[target_pollutant].max()
            pred_data[f'{target_pollutant}_q25_{window}'] = last_known_values[target_pollutant].quantile(0.25)
            pred_data[f'{target_pollutant}_q75_{window}'] = last_known_values[target_pollutant].quantile(0.75)
            pred_data[f'{target_pollutant}_skew_{window}'] = 0
            pred_data[f'{target_pollutant}_kurt_{window}'] = 0
            pred_data[f'{target_pollutant}_roc_{window}'] = 0
    
    # Day-of-week and hour-of-day averages from historical data
    dow_means = historical_data.groupby('day_of_week')[target_pollutant].mean()
    hour_means = historical_data.groupby('hour')[target_pollutant].mean()
    
    for i, row in pred_data.iterrows():
        dow = row['day_of_week']
        hour = row['hour']
        pred_data.at[i, f'{target_pollutant}_dow_mean'] = dow_means.get(dow, last_known_values[target_pollutant].mean())
        pred_data.at[i, f'{target_pollutant}_hour_mean'] = hour_means.get(hour, last_known_values[target_pollutant].mean())
    
    print(f"Prediction data prepared with {pred_data.shape[1]} features")
    return pred_data 

def predict_pollutant_advanced(measurement_df, station_code, pollutant, start_date, end_date, optimize_hyperparams=False):
    """Generate sophisticated predictions for a specific station and pollutant."""
    print(f"\n{'='*80}")
    print(f"Starting predictions for Station {station_code} - {pollutant}")
    print(f"{'='*80}")
    
    # Prepare training data with advanced features
    station_data = prepare_features(measurement_df, station_code, pollutant)
    
    # Train ensemble model with validation
    model, scaler, feature_names, cv_scores, base_models = train_model_with_validation(
        station_data, pollutant, optimize_hyperparams
    )
    
    # Analyze feature importance
    importance_df = analyze_feature_importance(model, feature_names, pollutant)
    
    # Generate future dates
    future_dates = generate_future_dates(start_date, end_date)
    
    # Prepare prediction data with all engineered features
    pred_data = prepare_advanced_prediction_data(future_dates, station_data, pollutant)
    
    # Ensure we only use features that exist in the trained model
    missing_features = [f for f in feature_names if f not in pred_data.columns]
    if missing_features:
        print(f"Warning: {len(missing_features)} features missing in prediction data")
        for feature in missing_features:
            pred_data[feature] = 0  # Default value for missing features
    
    # Make predictions in a rolling manner
    print(f"Generating rolling predictions for {len(future_dates)} timestamps")
    predictions = {}
    prediction_intervals = {}
    
    # Save model
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, models_dir / f"{station_code}_{pollutant}_model.pkl")
    joblib.dump(scaler, models_dir / f"{station_code}_{pollutant}_scaler.pkl")
    with open(models_dir / f"{station_code}_{pollutant}_features.json", 'w') as f:
        json.dump(feature_names, f)
    
    print(f"Model saved to models/{station_code}_{pollutant}_model.pkl")
    
    # Make predictions with uncertainty estimation
    for i, date in enumerate(future_dates):
        try:
            # Get the features for this timestamp
            X_pred = pred_data.loc[date:date, feature_names]
            
            # Scale features
            X_pred_scaled = scaler.transform(X_pred)
            
            # Use the base models to estimate uncertainty
            base_predictions = []
            for name, base_model in base_models.items():
                base_pred = base_model.predict(X_pred_scaled)[0]
                base_predictions.append(base_pred)
            
            # Ensemble prediction
            pred = model.predict(X_pred_scaled)[0]
            
            # Calculate confidence interval
            if len(base_predictions) >= 2:
                std_dev = np.std(base_predictions)
                lower_bound = max(0, pred - 1.96 * std_dev)
                upper_bound = pred + 1.96 * std_dev
            else:
                # Use CV error as a proxy for uncertainty
                mean_rmse = np.mean(cv_scores['rmse'])
                lower_bound = max(0, pred - 1.96 * mean_rmse)
                upper_bound = pred + 1.96 * mean_rmse
            
            # Round and ensure non-negative values
            pred = max(0, round(float(pred), 3))
            lower_bound = max(0, round(float(lower_bound), 3))
            upper_bound = max(0, round(float(upper_bound), 3))
            
            # Store prediction and intervals
            predictions[date.strftime('%Y-%m-%d %H:%M:%S')] = pred
            prediction_intervals[date.strftime('%Y-%m-%d %H:%M:%S')] = {
                'lower': lower_bound,
                'upper': upper_bound,
                'std': round(float(std_dev) if len(base_predictions) >= 2 else mean_rmse, 3)
            }
            
            # Update features for next prediction (only if not the last date)
            if i < len(future_dates) - 1:
                next_date = future_dates[i + 1]
                
                # Update lag features
                for lag in [1, 2, 3, 6, 12, 24, 48, 72]:
                    lag_feature = f'{pollutant}_lag_{lag}'
                    if lag_feature in feature_names:
                        if lag == 1:
                            pred_data.loc[next_date:, lag_feature] = pred
                        else:
                            previous_lag = f'{pollutant}_lag_{lag-1}'
                            if previous_lag in pred_data.columns:
                                pred_data.loc[next_date:, lag_feature] = pred_data.loc[date, previous_lag]
                
                # Update rolling statistics for the next prediction
                for window in [3, 6, 12, 24]:
                    # Only update if these features are used by the model
                    mean_feature = f'{pollutant}_rolling_mean_{window}'
                    median_feature = f'{pollutant}_rolling_median_{window}'
                    std_feature = f'{pollutant}_rolling_std_{window}'
                    
                    # Get the last 'window' predictions including the current one
                    if i + 1 >= window:
                        recent_dates = future_dates[i-window+1:i+1]
                        recent_preds = [predictions[d.strftime('%Y-%m-%d %H:%M:%S')] for d in recent_dates]
                    else:
                        # Mix of historical and predicted values
                        historical_count = window - (i + 1)
                        historical_values = station_data[pollutant].iloc[-historical_count:].tolist()
                        pred_dates = future_dates[:i+1]
                        pred_values = [predictions[d.strftime('%Y-%m-%d %H:%M:%S')] for d in pred_dates]
                        recent_preds = historical_values + pred_values
                    
                    # Update rolling statistics if they're used in the model
                    if mean_feature in feature_names:
                        pred_data.loc[next_date:, mean_feature] = np.mean(recent_preds)
                    
                    if median_feature in feature_names and len(recent_preds) > 0:
                        pred_data.loc[next_date:, median_feature] = np.median(recent_preds)
                    
                    if std_feature in feature_names and len(recent_preds) > 1:
                        pred_data.loc[next_date:, std_feature] = np.std(recent_preds)
                
                # Update trend features if used
                if 'day_trend' in feature_names and i > 24:
                    day_preds = [predictions[d.strftime('%Y-%m-%d %H:%M:%S')] for d in future_dates[i-24:i+1]]
                    pred_data.loc[next_date:, 'day_trend'] = np.mean(day_preds)
                
                if 'trend_direction' in feature_names and i > 0:
                    current_pred = predictions[date.strftime('%Y-%m-%d %H:%M:%S')]
                    prev_pred = predictions[future_dates[i-1].strftime('%Y-%m-%d %H:%M:%S')]
                    trend_dir = 1 if current_pred > prev_pred else (-1 if current_pred < prev_pred else 0)
                    pred_data.loc[next_date:, 'trend_direction'] = trend_dir
        
        except Exception as e:
            print(f"Error predicting for {date}: {e}")
            predictions[date.strftime('%Y-%m-%d %H:%M:%S')] = station_data[pollutant].median()
            prediction_intervals[date.strftime('%Y-%m-%d %H:%M:%S')] = {
                'lower': station_data[pollutant].quantile(0.025),
                'upper': station_data[pollutant].quantile(0.975),
                'std': station_data[pollutant].std()
            }
    
    print(f"Predictions complete for Station {station_code} - {pollutant}")
    
    # Visualize predictions with confidence intervals
    visualize_predictions(station_code, pollutant, station_data, predictions, prediction_intervals)
    
    return predictions, prediction_intervals

def visualize_predictions(station_code, pollutant, historical_data, predictions, prediction_intervals=None):
    """Visualize historical data and predictions with confidence intervals."""
    try:
        # Convert predictions to DataFrame
        pred_df = pd.DataFrame(list(predictions.items()), columns=['date', pollutant])
        pred_df['date'] = pd.to_datetime(pred_df['date'])
        
        # Add confidence intervals if available
        if prediction_intervals:
            lower_bound = []
            upper_bound = []
            for date in pred_df['date']:
                date_str = date.strftime('%Y-%m-%d %H:%M:%S')
                if date_str in prediction_intervals:
                    lower_bound.append(prediction_intervals[date_str]['lower'])
                    upper_bound.append(prediction_intervals[date_str]['upper'])
                else:
                    lower_bound.append(pred_df.loc[pred_df['date'] == date, pollutant].values[0])
                    upper_bound.append(pred_df.loc[pred_df['date'] == date, pollutant].values[0])
            
            pred_df['lower_bound'] = lower_bound
            pred_df['upper_bound'] = upper_bound
        
        # Get historical data
        hist_df = historical_data[historical_data['Station code'] == station_code].copy()
        hist_df = hist_df[['Measurement date', pollutant]]
        hist_df = hist_df.rename(columns={'Measurement date': 'date'})
        
        # Get last 30 days of historical data
        last_date = hist_df['date'].max()
        start_date = last_date - pd.Timedelta(days=30)
        hist_df = hist_df[hist_df['date'] >= start_date]
        
        # Create visualization
        plt.figure(figsize=(15, 7))
        
        # Plot historical data
        plt.plot(hist_df['date'], hist_df[pollutant], label='Historical Data', color='blue')
        
        # Plot predictions
        plt.plot(pred_df['date'], pred_df[pollutant], label='Predictions', color='red', linestyle='--')
        
        # Add confidence intervals if available
        if prediction_intervals:
            plt.fill_between(
                pred_df['date'], 
                pred_df['lower_bound'], 
                pred_df['upper_bound'], 
                color='red', 
                alpha=0.2, 
                label='95% Confidence Interval'
            )
        
        plt.title(f'Station {station_code} - {pollutant} Forecast with Confidence Intervals')
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
        plt.savefig(output_dir / f"{station_code}_{pollutant}_forecast.png")
        plt.close()
        
        print(f"Forecast visualization saved to reports/figures/{station_code}_{pollutant}_forecast.png")
    except Exception as e:
        print(f"Error visualizing predictions: {e}")

def compare_station_performance(tasks, cv_scores_dict, prediction_intervals_dict):
    """Analyze and compare model performance across different stations and pollutants."""
    try:
        # Create output directory
        output_dir = Path("reports/analysis")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance comparison dataframe
        performance_data = []
        
        for station_code, (pollutant, _, _) in tasks.items():
            if station_code in cv_scores_dict:
                cv_scores = cv_scores_dict[station_code]
                
                # Calculate average uncertainty from prediction intervals
                avg_uncertainty = 0
                if station_code in prediction_intervals_dict:
                    intervals = prediction_intervals_dict[station_code]
                    uncertainties = [interval_data['std'] for interval_data in intervals.values()]
                    avg_uncertainty = np.mean(uncertainties)
                
                # Add to performance data
                performance_data.append({
                    'Station': station_code,
                    'Pollutant': pollutant,
                    'MAE': np.mean(cv_scores['mae']),
                    'RMSE': np.mean(cv_scores['rmse']),
                    'R²': np.mean(cv_scores['r2']),
                    'Avg Uncertainty': avg_uncertainty
                })
        
        # Create DataFrame
        performance_df = pd.DataFrame(performance_data)
        
        # Sort by R² (best performance first)
        performance_df = performance_df.sort_values('R²', ascending=False)
        
        # Create performance comparison plots
        plt.figure(figsize=(12, 8))
        
        # R² comparison
        plt.subplot(2, 1, 1)
        sns.barplot(x='Station', y='R²', hue='Pollutant', data=performance_df)
        plt.title('Model Performance (R²) by Station and Pollutant')
        plt.ylim(0, 1)
        
        # Error metrics comparison
        plt.subplot(2, 1, 2)
        
        # Melt the dataframe for easier plotting
        metrics_df = performance_df.melt(
            id_vars=['Station', 'Pollutant'],
            value_vars=['MAE', 'RMSE'],
            var_name='Metric',
            value_name='Value'
        )
        
        sns.barplot(x='Station', y='Value', hue='Metric', data=metrics_df)
        plt.title('Error Metrics by Station')
        plt.yscale('log')  # Log scale for better visibility
        
        plt.tight_layout()
        plt.savefig(output_dir / "station_performance_comparison.png")
        plt.close()
        
        # Save detailed performance data
        performance_df.to_csv(output_dir / "model_performance.csv", index=False)
        
        print("\nStation Performance Comparison:")
        print(performance_df.to_string(index=False))
        print(f"\nPerformance comparison saved to {output_dir}")
        
        return performance_df
    
    except Exception as e:
        print(f"Error comparing station performance: {e}")
        return None

def main(optimize_hyperparams=False):
    """Run the enhanced forecasting pipeline."""
    print("Starting enhanced forecasting pipeline...")
    
    start_time = datetime.now()
    print(f"Start time: {start_time}")
    
    # Load data
    measurement_df = load_processed_data()
    
    # Prediction tasks
    tasks = {
        206: ('SO2', '2023-07-01 00:00:00', '2023-07-31 23:00:00'),
        211: ('NO2', '2023-08-01 00:00:00', '2023-08-31 23:00:00'),
        217: ('O3', '2023-09-01 00:00:00', '2023-09-30 23:00:00'),
        219: ('CO', '2023-10-01 00:00:00', '2023-10-31 23:00:00'),
        225: ('PM10', '2023-11-01 00:00:00', '2023-11-30 23:00:00'),
        228: ('PM2.5', '2023-12-01 00:00:00', '2023-12-31 23:00:00')
    }
    
    # Create results directory
    predictions_dir = Path("predictions")
    predictions_dir.mkdir(exist_ok=True)
    
    # Generate predictions
    predictions = {"target": {}}
    
    # Store cross-validation scores and prediction intervals for performance comparison
    cv_scores_dict = {}
    prediction_intervals_dict = {}
    
    for station_code, (pollutant, start_date, end_date) in tasks.items():
        # Generate predictions for this station-pollutant combination
        station_predictions, intervals = predict_pollutant_advanced(
            measurement_df,
            station_code,
            pollutant,
            start_date,
            end_date,
            optimize_hyperparams
        )
        
        # Store performance metrics for comparison
        _, _, _, cv_scores, _ = train_model_with_validation(
            prepare_features(measurement_df, station_code, pollutant),
            pollutant,
            optimize_hyperparams=False  # Just for evaluation, don't reoptimize
        )
        
        cv_scores_dict[station_code] = cv_scores
        prediction_intervals_dict[station_code] = intervals
        
        # Add predictions to results
        predictions["target"][str(station_code)] = station_predictions
    
    # Compare station performance
    performance_df = compare_station_performance(tasks, cv_scores_dict, prediction_intervals_dict)
    
    # Save predictions to JSON
    with open(predictions_dir / "predictions_task_2.json", 'w') as f:
        json.dump(predictions, f, indent=2)
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\n{'='*80}")
    print(f"Enhanced forecasting pipeline completed in {duration}")
    print(f"Predictions saved to predictions/predictions_task_2.json")
    print(f"{'='*80}")
    
    return predictions

if __name__ == "__main__":
    # Run the pipeline with hyperparameter optimization
    main(optimize_hyperparams=True)