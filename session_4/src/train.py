from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import sys
import os

# Add parent directory to path to import metadata
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metadata import MODEL_PARAMS

def train_model(X_train, y_train, params=None):
    """
    Train a decision tree model
    
    Args:
        X_train (pd.DataFrame or np.array): Training features
        y_train (pd.Series or np.array): Training target
        params (dict, optional): Model parameters. Defaults to MODEL_PARAMS from metadata
        
    Returns:
        model: Trained decision tree model
    """
    if params is None:
        params = MODEL_PARAMS
    
    model = DecisionTreeClassifier(**params)
    model.fit(X_train, y_train)
    
    print("Model training completed!")
    print(f"Model parameters: {params}")
    
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        
    Returns:
        dict: Evaluation metrics
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return {
        'accuracy': accuracy,
        'predictions': y_pred
    }

if __name__ == "__main__":
    # Test the training function
    from source import load_data
    from transform import prepare_features
    
    df = load_data()
    X, y, _, _ = prepare_features(df)
    
    # Split for testing
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)