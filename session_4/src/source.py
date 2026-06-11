import pandas as pd
import sys
import os

# Add parent directory to path to import metadata
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metadata import DATA_PATH

def load_data(file_path=None):
    """
    Load CSV data from the specified file path
    
    Args:
        file_path (str, optional): Path to CSV file. Defaults to DATA_PATH from metadata
        
    Returns:
        pd.DataFrame: Loaded dataframe
    """
    if file_path is None:
        file_path = DATA_PATH
    
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}")
        print(f"Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        raise
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

if __name__ == "__main__":
    # Test the function
    df = load_data()
    print(df.head())