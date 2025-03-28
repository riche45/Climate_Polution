import os
import shutil
from pathlib import Path

def cleanup():
    """Remove temporary files and directories."""
    raw_dir = Path("data/raw")
    
    # Remove ZIP files
    for zip_file in raw_dir.glob("*.zip"):
        print(f"Removing {zip_file}")
        zip_file.unlink()
    
    # Remove __MACOSX directory if it exists
    macosx_dir = raw_dir / "__MACOSX"
    if macosx_dir.exists():
        print("Removing __MACOSX directory")
        shutil.rmtree(macosx_dir)

if __name__ == "__main__":
    cleanup() 