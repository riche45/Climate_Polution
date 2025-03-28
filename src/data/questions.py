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

def get_normal_measurements(measurement_df, instrument_df):
    """Get measurements with normal status (code 0)."""
    normal_instruments = instrument_df[instrument_df['Instrument status'] == 0]
    # Merge to get only normal measurements
    normal_measurements = pd.merge(
        measurement_df,
        normal_instruments[['Measurement date', 'Station code']].drop_duplicates(),
        on=['Measurement date', 'Station code']
    )
    return normal_measurements

def answer_questions(measurement_df, instrument_df, pollutant_df):
    """Answer all questions for Task 1."""
    # Get normal measurements
    normal_measurements = get_normal_measurements(measurement_df, instrument_df)
    
    # Q1: Average daily SO2 concentration across all districts
    q1 = normal_measurements.groupby('Station code')['SO2'].mean().mean()
    
    # Q2: Average CO levels per season at station 209
    station_209 = normal_measurements[normal_measurements['Station code'] == 209]
    q2 = station_209.groupby('Season')['CO'].mean().to_dict()
    # Ensure all seasons are present even if no data
    all_seasons = {'Winter': 0.0, 'Spring': 0.0, 'Summer': 0.0, 'Fall': 0.0}
    all_seasons.update(q2)
    q2 = {k: round(v, 5) for k, v in all_seasons.items()}
    
    # Q3: Hour with highest O3 variability
    q3 = normal_measurements.groupby('Hour')['O3'].std().idxmax()
    
    # Q4: Station with most "Abnormal data" measurements
    abnormal_data = instrument_df[instrument_df['Instrument status'] == 9]
    q4 = abnormal_data['Station code'].value_counts().idxmax()
    
    # Q5: Station with most "not normal" measurements
    not_normal = instrument_df[instrument_df['Instrument status'] != 0]
    q5 = not_normal['Station code'].value_counts().idxmax()
    
    # Q6: Count of records by category for PM2.5
    # First get the thresholds for PM2.5
    pm25_thresholds = pollutant_df[pollutant_df['Item name'] == 'PM2.5'].iloc[0]
    
    def get_category(value):
        if value <= pm25_thresholds['Good']:
            return 'Good'
        elif value <= pm25_thresholds['Normal']:
            return 'Normal'
        elif value <= pm25_thresholds['Bad']:
            return 'Bad'
        else:
            return 'Very bad'
    
    pm25_data = normal_measurements['PM2.5'].dropna()
    q6 = pm25_data.apply(get_category).value_counts().to_dict()
    
    # Format answers
    answers = {
        "target": {
            "Q1": round(q1, 5),
            "Q2": q2,
            "Q3": int(q3),
            "Q4": int(q4),
            "Q5": int(q5),
            "Q6": {k: int(v) for k, v in q6.items()}
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
