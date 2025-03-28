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

def answer_questions(measurement_df, instrument_df, pollutant_df):
    """Answer all questions for Task 1."""
    # Q1: Average daily SO2 concentration across all districts
    so2_measurements = get_normal_measurements(measurement_df, instrument_df, 'SO2', pollutant_df)
    daily_so2 = so2_measurements.groupby(['Station code', pd.Grouper(key='Measurement date', freq='D')])['SO2'].mean()
    q1 = daily_so2.mean()
    
    # Q2: Average CO levels per season at station 209
    co_measurements = get_normal_measurements(measurement_df, instrument_df, 'CO', pollutant_df)
    station_209 = co_measurements[co_measurements['Station code'] == 209].copy()
    
    # Ensure correct season assignment
    station_209.loc[station_209['Month'] == 12, 'Season'] = 'Winter'
    station_209.loc[station_209['Month'] == 3, 'Season'] = 'Spring'
    station_209.loc[station_209['Month'] == 6, 'Season'] = 'Summer'
    station_209.loc[station_209['Month'] == 9, 'Season'] = 'Fall'
    
    q2 = station_209.groupby('Season')['CO'].mean().round(5).to_dict()
    # Ensure all seasons are present
    all_seasons = {'Winter': 0.0, 'Spring': 0.0, 'Summer': 0.0, 'Fall': 0.0}
    all_seasons.update(q2)
    
    # Q3: Hour with highest O3 variability
    o3_measurements = get_normal_measurements(measurement_df, instrument_df, 'O3', pollutant_df)
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
        if value <= pm25_thresholds['Good']:
            return 'Good'
        elif value <= pm25_thresholds['Normal']:
            return 'Normal'
        elif value <= pm25_thresholds['Bad']:
            return 'Bad'
        else:
            return 'Very bad'
    
    pm25_categories = pm25_measurements['PM2.5'].apply(get_category)
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

def main():
    # Load data
    measurement_df, instrument_df, pollutant_df = load_processed_data()
    
    # Get answers
    answers = answer_questions(measurement_df, instrument_df, pollutant_df)
    
    # Save answers to JSON
    predictions_dir = Path("predictions")
    predictions_dir.mkdir(exist_ok=True)
    
    with open(predictions_dir / "questions.json", "w") as f:
        json.dump(answers, f, indent=2)
    
    print("Answers saved to questions.json!")
    print("\nAnswers preview:")
    print(json.dumps(answers, indent=2))

if __name__ == "__main__":
    main()
