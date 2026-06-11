import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.transform import (
    handle_missing_values,
    encode_categorical_features,
    scale_features,
    prepare_features
)
from src.source import load_data

class TestDataTransformer:
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        return pd.DataFrame({
            'CreditScore': [600, 700, 800, np.nan, 650],
            'Age': [25, 35, 45, 55, np.nan],
            'Tenure': [1, 2, 3, 4, 5],
            'Balance': [0, 50000, 100000, 150000, 200000],
            'NumOfProducts': [1, 2, 1, 3, 2],
            'EstimatedSalary': [30000, 50000, 70000, 90000, 110000],
            'Geography': ['France', 'Spain', 'Germany', 'France', 'Spain'],
            'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
            'Exited': [0, 1, 0, 1, 0]
        })
    
    @pytest.fixture
    def real_data(self):
        """Load real data for testing"""
        try:
            df = load_data()
            return df.head(100)  # Use first 100 rows for faster testing
        except:
            pytest.skip("Could not load real data")
    
    def test_handle_missing_values(self, sample_data):
        """Test missing value handling"""
        df_clean = handle_missing_values(sample_data.copy())
        
        # Check that there are no missing values
        assert df_clean.isnull().sum().sum() == 0
        
        # Check that shape remains the same
        assert df_clean.shape == sample_data.shape
    
    def test_encode_categorical_features(self, sample_data):
        """Test categorical feature encoding"""
        df_encoded, encoders = encode_categorical_features(sample_data.copy())
        
        # Check that categorical columns are now numeric
        assert df_encoded['Geography'].dtype in ['int64', 'float64']
        assert df_encoded['Gender'].dtype in ['int64', 'float64']
        
        # Check that values are within expected range
        assert df_encoded['Geography'].between(0, 2).all()  # 3 categories
        assert df_encoded['Gender'].between(0, 1).all()  # 2 categories
        
        # Check that encoders were created
        assert len(encoders) == 2
        assert 'Geography' in encoders
        assert 'Gender' in encoders
    
    def test_scale_features(self, sample_data):
        """Test feature scaling"""
        # First encode categorical features
        df_encoded, _ = encode_categorical_features(sample_data.copy())
        
        # Scale numerical features
        df_scaled, scaler = scale_features(df_encoded.copy())
        
        # Check that numerical columns are scaled (mean ~0, std ~1)
        numerical_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
        for col in numerical_cols:
            if col in df_scaled.columns:
                # Allow small tolerance for mean and std
                assert abs(df_scaled[col].mean()) < 1e-6
                assert abs(df_scaled[col].std() - 1) < 1e-6
        
        # Test inverse transformation with the same scaler
        df_restored, _ = scale_features(df_encoded.copy(), scaler=scaler)
        pd.testing.assert_frame_equal(df_restored, df_scaled)
    
    def test_prepare_features_returns_correct_components(self, sample_data):
        """Test that prepare_features returns all required components"""
        X, y, encoders, scaler = prepare_features(sample_data.copy())
        
        # Check return types
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert isinstance(encoders, dict)
        assert scaler is not None
        
        # Check that target is correctly extracted
        assert y.name == 'Exited'
        assert len(y) == len(sample_data)
        
        # Check that features don't contain target column
        assert 'Exited' not in X.columns
        
        # Check that shape is correct
        expected_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 
                            'NumOfProducts', 'EstimatedSalary', 'Geography', 'Gender']
        actual_features = [col for col in expected_features if col in X.columns]
        assert len(actual_features) == len(expected_features)
    
    def test_feature_dimensions_after_transform(self, real_data):
        """Test that transformation maintains correct dimensions with real data"""
        if real_data is None:
            pytest.skip("No real data available")
        
        X, y, encoders, scaler = prepare_features(real_data.copy())
        
        # Check that number of samples is preserved
        assert len(X) == len(real_data)
        assert len(y) == len(real_data)
        
        # Check that no missing values remain
        assert X.isnull().sum().sum() == 0
        
        # Check that all features are numeric
        assert all(X[col].dtype in ['int64', 'float64'] for col in X.columns)
    
    def test_no_data_leakage(self, sample_data):
        """Test that there's no data leakage between train and test"""
        # Split data
        train_data = sample_data.iloc[:3].copy()
        test_data = sample_data.iloc[3:].copy()
        
        # Fit on training data
        X_train, y_train, encoders, scaler = prepare_features(train_data)
        
        # Transform test data using training artifacts
        from src.transform import transform_data
        X_test, y_test, _, _ = transform_data(test_data, encoders=encoders, scaler=scaler, fit=False)
        
        # Check that test transformation worked
        assert len(X_test) == len(test_data)
        assert X_test.isnull().sum().sum() == 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])