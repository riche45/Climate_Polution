import os
import requests
from pathlib import Path

def download_file(url, filename):
    """Download a file from a URL to a specific filename."""
    print(f"Downloading {filename}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    file_path = Path("data/raw") / filename
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print(f"Downloaded {filename}")

def main():
    # URLs for the datasets
    urls = {
        "measurement_data.zip": "https://cdn.nuwe.io/challenges-ds-datasets/hackathon-schneider-pollutant/measurement_data.zip",
        "instrument_data.zip": "https://cdn.nuwe.io/challenges-ds-datasets/hackathon-schneider-pollutant/instrument_data.zip",
        "pollutant_data.csv": "https://cdn.nuwe.io/challenges-ds-datasets/hackathon-schneider-pollutant/pollutant_data.csv"
    }
    
    # Create data directories if they don't exist
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    # Download each dataset
    for filename, url in urls.items():
        download_file(url, filename)

if __name__ == "__main__":
    main() 