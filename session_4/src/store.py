import joblib
import os
import sys
from datetime import datetime

# Add parent directory to path to import metadata
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metadata import MODELS_DIR, YOUR_NAME

def save_model(model, name=None):
    """
    Save the trained model to the models directory
    
    Args:
        model: Trained model to save
        name (str, optional): Custom name for the model. 
                             If None, uses format: class_model-{your_name}-{timestamp}.joblib
        
    Returns:
        str: Path where the model was saved
    """
    if name is None:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        filename = f"class_model-{YOUR_NAME}-{timestamp}.joblib"
    else:
        filename = f"{name}.joblib"
    
    model_path = os.path.join(MODELS_DIR, filename)
    
    # Ensure the models directory exists
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Save the model
    joblib.dump(model, model_path)
    print(f"Model saved successfully to: {model_path}")
    
    return model_path

def load_model(model_path):
    """
    Load a saved model
    
    Args:
        model_path (str): Path to the saved model file
        
    Returns:
        model: Loaded model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")
    
    model = joblib.load(model_path)
    print(f"Model loaded successfully from: {model_path}")
    
    return model

def save_artifacts(encoders, scaler, artifact_dir=None):
    """
    Save preprocessing artifacts (encoders and scaler)
    
    Args:
        encoders (dict): Dictionary of label encoders
        scaler: Fitted scaler
        artifact_dir (str, optional): Directory to save artifacts
        
    Returns:
        dict: Paths to saved artifacts
    """
    if artifact_dir is None:
        artifact_dir = MODELS_DIR
    
    artifact_paths = {}
    
    # Save encoders
    for col, encoder in encoders.items():
        encoder_path = os.path.join(artifact_dir, f"encoder_{col}.joblib")
        joblib.dump(encoder, encoder_path)
        artifact_paths[f"encoder_{col}"] = encoder_path
    
    # Save scaler
    if scaler is not None:
        scaler_path = os.path.join(artifact_dir, "scaler.joblib")
        joblib.dump(scaler, scaler_path)
        artifact_paths["scaler"] = scaler_path
    
    print(f"Artifacts saved to: {artifact_dir}")
    return artifact_paths

if __name__ == "__main__":
    # Test the save/load functions
    from sklearn.tree import DecisionTreeClassifier
    
    # Create a dummy model
    dummy_model = DecisionTreeClassifier()
    
    # Save the model
    saved_path = save_model(dummy_model)
    
    # Load the model
    loaded_model = load_model(saved_path)
    print(f"Model loaded successfully: {type(loaded_model)}")