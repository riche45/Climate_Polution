import pandas as pd
import numpy as np
import json
from pathlib import Path

def load_processed_data():
    """Load processed datasets."""
    processed_dir = Path("data/processed")
    
    measurement_df = pd.read_csv(
        processed_dir / "measurement_data_processed.csv",
        parse_dates=['Measurement date']
    )
    instrument_df = pd.read_csv(
        processed_dir / "instrument_data_processed.csv",
        parse_dates=['Measurement date']
    )
    pollutant_df = pd.read_csv(processed_dir / "pollutant_data_processed.csv")
    
    return measurement_df, instrument_df, pollutant_df

def get_pollutant_code(pollutant_name, pollutant_df):
    """Get Item code for a pollutant name."""
    return pollutant_df[pollutant_df['Item name'] == pollutant_name]['Item code'].iloc[0]

def get_normal_measurements(measurement_df, instrument_df, pollutant_name=None, pollutant_df=None):
    """Get measurements with normal status (code 0)."""
    if pollutant_name:
        # Get Item code for the pollutant
        pollutant_code = get_pollutant_code(pollutant_name, pollutant_df)
        # Get normal status for specific pollutant
        normal_instruments = instrument_df[
            (instrument_df['Instrument status'] == 0) &
            (instrument_df['Item code'] == pollutant_code)
        ]
    else:
        # Get normal status for all measurements
        normal_instruments = instrument_df[instrument_df['Instrument status'] == 0]
    
    # Get only measurements that have normal status
    valid_measurements = normal_instruments.groupby(['Measurement date', 'Station code']).size()
    if pollutant_name:
        valid_measurements = valid_measurements[valid_measurements >= 1]
    else:
        valid_measurements = valid_measurements[valid_measurements >= 6]  # All pollutants should be normal
    valid_measurements = valid_measurements.reset_index()
    
    # Merge with measurement data
    normal_measurements = pd.merge(
        measurement_df,
        valid_measurements[['Measurement date', 'Station code']],
        on=['Measurement date', 'Station code']
    )
    
    return normal_measurements

def get_season(month):
    """Get season from month number."""
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:  # month in [9, 10, 11]
        return 'Fall'

def answer_questions(measurement_df, instrument_df, pollutant_df):
    """Answer all questions for Task 1."""
    # Q1: Average daily SO2 concentration across all districts
    so2_measurements = get_normal_measurements(measurement_df, instrument_df, 'SO2', pollutant_df)
    # Calculate daily average for each station
    so2_measurements['date'] = so2_measurements['Measurement date'].dt.date
    # Calculate daily average in one step
    daily_so2 = so2_measurements.groupby(['Station code', 'date'])['SO2'].mean().reset_index()
    # Calculate station average
    station_avg = daily_so2.groupby('Station code')['SO2'].mean()
    # Calculate final average
    q1 = station_avg.mean()
    
    # Q2: Average CO levels per season at station 209
    co_measurements = get_normal_measurements(measurement_df, instrument_df, 'CO', pollutant_df)
    station_209 = co_measurements[co_measurements['Station code'] == 209].copy()
    
    # Apply season to all months
    station_209['Season'] = station_209['Measurement date'].dt.month.map(get_season)
    
    # Calculate average CO by season
    q2 = station_209.groupby('Season')['CO'].mean().round(5).to_dict()
    # Ensure all seasons are present
    all_seasons = {'Winter': 0.0, 'Spring': 0.0, 'Summer': 0.0, 'Fall': 0.0}
    all_seasons.update(q2)
    
    # Q3: Hour with highest O3 variability
    o3_measurements = get_normal_measurements(measurement_df, instrument_df, 'O3', pollutant_df)
    # Calculate standard deviation for each hour across all stations
    q3 = o3_measurements.groupby('Hour')['O3'].std().idxmax()
    
    # Q4: Station with most "Abnormal data" measurements
    abnormal_data = instrument_df[instrument_df['Instrument status'] == 9]
    station_abnormal_counts = abnormal_data.groupby('Station code').size()
    q4 = station_abnormal_counts.idxmax()
    
    # Q5: Station with most "not normal" measurements
    not_normal = instrument_df[instrument_df['Instrument status'] != 0]
    station_not_normal_counts = not_normal.groupby('Station code').size()
    q5 = station_not_normal_counts.idxmax()
    
    # Q6: Count of records by category for PM2.5
    pm25_measurements = get_normal_measurements(measurement_df, instrument_df, 'PM2.5', pollutant_df)
    pm25_thresholds = pollutant_df[pollutant_df['Item name'] == 'PM2.5'].iloc[0]
    
    def get_category(value):
        if pd.isna(value):
            return None
        # Use inclusive ranges for classification
        if value <= pm25_thresholds['Good']:
            return 'Good'
        elif value <= pm25_thresholds['Normal']:
            return 'Normal'
        elif value <= pm25_thresholds['Bad']:
            return 'Bad'
        else:
            return 'Very bad'
    
    # Drop NaN values before categorizing
    pm25_data = pm25_measurements['PM2.5'].dropna()
    pm25_categories = pm25_data.apply(get_category)
    q6 = pm25_categories.value_counts().to_dict()
    
    # Format answers
    answers = {
        "target": {
            "Q1": round(q1, 5),
            "Q2": {k: round(v, 5) for k, v in all_seasons.items()},
            "Q3": int(q3),
            "Q4": int(q4),
            "Q5": int(q5),
            "Q6": {k: int(v) for k, v in q6.items() if k is not None}
        }
    }
    
    return answers

def calculate_so2():
    measurement_df, instrument_df, _ = load_processed_data()
    
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
    
    # Filtrar estaciones con más de 24,000 mediciones
    station_counts = so2_data.groupby('Station code').size()
    reliable_stations = station_counts[station_counts > 24000].index
    so2_data = so2_data[so2_data['Station code'].isin(reliable_stations)]
    
    # Calcular pesos basados en el número de mediciones
    station_weights = station_counts[reliable_stations]
    station_weights = station_weights / station_weights.sum()
    
    # Calcular mediana ponderada por estación
    station_medians = so2_data.groupby('Station code')['SO2'].median()
    weighted_median = np.average(station_medians, weights=station_weights)
    
    return weighted_median

def main():
    # Calcular SO2
    so2_value = calculate_so2()
    
    # Cargar el archivo JSON existente
    output_file = Path("predictions/questions.json")
    with open(output_file, 'r') as f:
        data = json.load(f)
    
    # Actualizar el valor de SO2
    data['SO2'] = float(so2_value)
    
    # Guardar el archivo JSON actualizado
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"SO2 value calculated: {so2_value}")

if __name__ == "__main__":
    main()
