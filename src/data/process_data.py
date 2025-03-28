import pandas as pd
import numpy as np
from pathlib import Path

def load_data():
    """Load all datasets."""
    data_dir = Path("data/raw")
    
    # Load measurement data
    print("Loading measurement data...")
    measurement_df = pd.read_csv(
        data_dir / "measurement_data.csv",
        parse_dates=['Measurement date']
    )
    
    # Load instrument data
    print("Loading instrument data...")
    instrument_df = pd.read_csv(
        data_dir / "instrument_data.csv",
        parse_dates=['Measurement date']
    )
    
    # Load pollutant data
    print("Loading pollutant data...")
    pollutant_df = pd.read_csv(data_dir / "pollutant_data.csv")
    
    return measurement_df, instrument_df, pollutant_df

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

def process_data(measurement_df, instrument_df, pollutant_df):
    """Process and clean the data."""
    # Add season information using the custom function
    measurement_df['Month'] = measurement_df['Measurement date'].dt.month
    measurement_df['Season'] = measurement_df['Month'].apply(get_season)
    
    # Add hour information
    measurement_df['Hour'] = measurement_df['Measurement date'].dt.hour
    
    return measurement_df, instrument_df, pollutant_df

def main():
    # Load data
    measurement_df, instrument_df, pollutant_df = load_data()
    
    # Process data
    measurement_df, instrument_df, pollutant_df = process_data(
        measurement_df, instrument_df, pollutant_df
    )
    
    # Save processed data
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    print("Saving processed data...")
    measurement_df.to_csv(processed_dir / "measurement_data_processed.csv", index=False)
    instrument_df.to_csv(processed_dir / "instrument_data_processed.csv", index=False)
    pollutant_df.to_csv(processed_dir / "pollutant_data_processed.csv", index=False)
    
    print("Data processing completed!")

if __name__ == "__main__":
    main() 