import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import sys
import os

# Add parent directory to path to import metadata
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metadata import NUMERICAL_COLUMNS, CATEGORICAL_COLUMNS, TARGET_COLUMN

def handle_missing_values(df):
    """Handle missing values in the dataframe"""
    # Check for missing values
    if df.isnull().sum().sum() > 0:
        # Fill numerical missing values with median
        for col in NUMERICAL_COLUMNS:
            if col in df.columns and df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        # Fill categorical missing values with mode
        for col in CATEGORICAL_COLUMNS:
            if col in df.columns and df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
    
    return df

def encode_categorical_features(df):
    """Encode categorical features using Label Encoding"""
    df_encoded = df.copy()
    encoders = {}
    
    for col in CATEGORICAL_COLUMNS:
        if col in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            encoders[col] = le
    
    return df_encoded, encoders

def scale_features(df, scaler=None):
    """Scale numerical features"""
    from sklearn.preprocessing import StandardScaler
    
    df_scaled = df.copy()
    available_numerical = [col for col in NUMERICAL_COLUMNS if col in df_scaled.columns]
    
    if scaler is None:
        scaler = StandardScaler()
        df_scaled[available_numerical] = scaler.fit_transform(df_scaled[available_numerical])
    else:
        df_scaled[available_numerical] = scaler.transform(df_scaled[available_numerical])
    
    return df_scaled, scaler

def prepare_features(df):
    """
    Prepare features for training/inference
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        X (pd.DataFrame): Features
        y (pd.Series): Target variable (if available)
        encoders (dict): Label encoders for categorical variables
        scaler: Fitted scaler for numerical features
    """
    # Drop unnecessary columns if they exist
    df_clean = df.copy()
    columns_to_drop = ['RowNumber', 'CustomerId', 'Surname']
    for col in columns_to_drop:
        if col in df_clean.columns:
            df_clean.drop(columns=[col], inplace=True)
    
    # Handle missing values
    df_clean = handle_missing_values(df_clean)
    
    # Separate features and target
    if TARGET_COLUMN in df_clean.columns:
        y = df_clean[TARGET_COLUMN].copy()
        X = df_clean.drop(columns=[TARGET_COLUMN])
    else:
        y = None
        X = df_clean.copy()
    
    # Encode categorical features
    X, encoders = encode_categorical_features(X)
    
    # Scale numerical features
    X, scaler = scale_features(X)
    
    return X, y, encoders, scaler

def transform_data(df, encoders=None, scaler=None, fit=True):
    """
    Main transformation function for the pipeline
    
    Args:
        df (pd.DataFrame): Input dataframe
        encoders (dict, optional): Existing encoders for inference
        scaler (optional): Existing scaler for inference
        fit (bool): Whether to fit new encoders/scaler or use existing ones
        
    Returns:
        tuple: (X, y, encoders, scaler)
    """
    if fit:
        return prepare_features(df)
    else:
        # Transform using existing encoders and scaler
        df_clean = df.copy()
        columns_to_drop = ['RowNumber', 'CustomerId', 'Surname']
        for col in columns_to_drop:
            if col in df_clean.columns:
                df_clean.drop(columns=[col], inplace=True)
        
        df_clean = handle_missing_values(df_clean)
        
        if TARGET_COLUMN in df_clean.columns:
            y = df_clean[TARGET_COLUMN].copy()
            X = df_clean.drop(columns=[TARGET_COLUMN])
        else:
            y = None
            X = df_clean.copy()
        
        # Transform categorical features
        for col in CATEGORICAL_COLUMNS:
            if col in X.columns and col in encoders:
                X[col] = encoders[col].transform(X[col].astype(str))
        
        # Transform numerical features
        available_numerical = [col for col in NUMERICAL_COLUMNS if col in X.columns]
        X[available_numerical] = scaler.transform(X[available_numerical])
        
        return X, y, encoders, scaler

if __name__ == "__main__":
    # Test the transformation functions
    from source import load_data
    
    df = load_data()
    X, y, encoders, scaler = prepare_features(df)
    print(f"Transformed features shape: {X.shape}")
    print(f"Target shape: {y.shape if y is not None else 'No target'}")