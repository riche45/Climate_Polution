import os
import zipfile
from pathlib import Path

def extract_zip(zip_path, extract_to):
    """Extract a ZIP file to a specific directory."""
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted {zip_path}")

def main():
    # Create raw data directory if it doesn't exist
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # List of ZIP files to extract
    zip_files = [
        "measurement_data.zip",
        "instrument_data.zip"
    ]
    
    # Extract each ZIP file
    for zip_file in zip_files:
        zip_path = raw_dir / zip_file
        if zip_path.exists():
            extract_zip(zip_path, raw_dir)
        else:
            print(f"Warning: {zip_file} not found")

if __name__ == "__main__":
    main() 