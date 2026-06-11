import os
from datetime import datetime

# Model parameters
MODEL_PARAMS = {
    'max_depth': 5,
    'min_samples_split': 10,
    'random_state': 42
}

# Column categories
NUMERICAL_COLUMNS = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
CATEGORICAL_COLUMNS = ['Geography', 'Gender']
TARGET_COLUMN = 'Exited'

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', 'datasets', 'Churn_Modelling_train_test.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
SRC_DIR = os.path.join(BASE_DIR, 'src')

# Create models directory if it doesn't exist
os.makedirs(MODELS_DIR, exist_ok=True)

# Your name for model filename
YOUR_NAME = "your_name"  # Replace with your actual name

def get_timestamp():
    """Get current timestamp for model filename"""
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")