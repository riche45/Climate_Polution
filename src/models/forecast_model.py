import pandas as pd
import numpy as np
from pathlib import Path
import json
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
from datetime import datetime, timedelta

def load_processed_data():
    """Load processed datasets."""
    processed_dir = Path("data/processed")
    
    measurement_df = pd.read_csv(
        processed_dir / "measurement_data_processed.csv",
        parse_dates=['Measurement date']
    )
    return measurement_df

def prepare_features(df, target_station, target_pollutant):
    """Prepare features for the model."""
    station_data = df[df['Station code'] == target_station].copy()
    
    # Time-based features
    station_data['hour'] = station_data['Measurement date'].dt.hour
    station_data['day'] = station_data['Measurement date'].dt.day
    station_data['month'] = station_data['Measurement date'].dt.month
    station_data['day_of_week'] = station_data['Measurement date'].dt.dayofweek
    
    # Lag features (previous hours)
    for lag in [1, 2, 3, 6, 12, 24]:
        station_data[f'{target_pollutant}_lag_{lag}'] = station_data[target_pollutant].shift(lag)
    
    # Rolling mean features
    for window in [3, 6, 12, 24]:
        station_data[f'{target_pollutant}_rolling_mean_{window}'] = station_data[target_pollutant].rolling(window=window).mean()
    
    # Drop rows with NaN values
    station_data = station_data.dropna()
    
    return station_data

def train_model(station_data, target_pollutant):
    """Train XGBoost model."""
    # Prepare features and target
    feature_columns = [col for col in station_data.columns if col.endswith(('hour', 'day', 'month', 'day_of_week')) 
                      or col.startswith(f'{target_pollutant}_')]
    
    X = station_data[feature_columns]
    y = station_data[target_pollutant]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,  # Lower learning rate for better performance
        max_depth=7,  # Slightly deeper trees
        subsample=0.8,  # Randomly sample 80% of the training data
        colsample_bytree=0.8,  # Use 80% of features at each split
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_scaled, y)
    
    return model, scaler, feature_columns

def generate_future_dates(start_date, end_date):
    """Generate hourly dates between start and end date."""
    dates = pd.date_range(start=start_date, end=end_date, freq='h')
    return dates

def prepare_prediction_data(dates, last_known_values, target_pollutant):
    """Prepare data for prediction."""
    pred_data = pd.DataFrame(index=dates)
    pred_data['hour'] = pred_data.index.hour
    pred_data['day'] = pred_data.index.day
    pred_data['month'] = pred_data.index.month
    pred_data['day_of_week'] = pred_data.index.dayofweek
    
    # Initialize lag features with last known values
    for lag in [1, 2, 3, 6, 12, 24]:
        pred_data[f'{target_pollutant}_lag_{lag}'] = last_known_values[target_pollutant].iloc[-lag]
    
    # Initialize rolling means with the mean of last known values
    for window in [3, 6, 12, 24]:
        pred_data[f'{target_pollutant}_rolling_mean_{window}'] = last_known_values[target_pollutant].iloc[-window:].mean()
    
    return pred_data

def predict_pollutant(measurement_df, station_code, pollutant, start_date, end_date):
    """Generate predictions for a specific station and pollutant."""
    # Prepare training data
    station_data = prepare_features(measurement_df, station_code, pollutant)
    
    # Train model
    model, scaler, feature_columns = train_model(station_data, pollutant)
    
    # Generate future dates
    future_dates = generate_future_dates(start_date, end_date)
    
    # Prepare prediction data
    pred_data = prepare_prediction_data(future_dates, station_data.tail(24), pollutant)
    
    # Make predictions
    predictions = {}
    for date in future_dates:
        # Scale features
        X_pred = pred_data.loc[date:date, feature_columns]
        X_pred_scaled = scaler.transform(X_pred)
        
        # Predict
        pred = model.predict(X_pred_scaled)[0]
        predictions[date.strftime('%Y-%m-%d %H:%M:%S')] = round(float(pred), 2)
        
        # Update lag features for next prediction
        if date != future_dates[-1]:
            next_date = date + pd.Timedelta(hours=1)
            for lag in [1, 2, 3, 6, 12, 24]:
                pred_data.loc[next_date:, f'{pollutant}_lag_{lag}'] = pred
    
    return predictions

def main():
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
    
    # Generate predictions
    predictions = {"target": {}}
    for station_code, (pollutant, start_date, end_date) in tasks.items():
        print(f"Generating predictions for Station {station_code} - {pollutant}")
        station_predictions = predict_pollutant(
            measurement_df,
            station_code,
            pollutant,
            start_date,
            end_date
        )
        predictions["target"][str(station_code)] = station_predictions
    
    # Save predictions
    predictions_dir = Path("predictions")
    predictions_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    with open(predictions_dir / "predictions_task_2.json", "w") as f:
        json.dump(predictions, f, indent=2)
    
    print("Predictions saved to predictions_task_2.json!")

if __name__ == "__main__":
    main()
